# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
NAVTEX Decoder — Benchmark Harness (test harness, part 2)
===========================================================

Feeds synthetic transmissions (navtex_synth.py) through the real decoding
pipeline (navtex_session.decode_characters, exactly as the application
runs it) and scores the decoded text against the known message.

Subcommands
------------
    python navtex_bench.py one   --snr -8             # one trial, prints the text
    python navtex_bench.py sweep --snr -14:0:2 --trials 10 --jobs 4
    python navtex_bench.py noise --snr -6 --minutes 5 # false text on noise alone

Common options (all subcommands)
---------------------------------
    --profile-file F --profile-name N   decoder settings from a TOML profile
                                        (default: built-in Profile defaults)
    --set key=value                     override a Profile field (repeatable),
                                        e.g. --set loop_gain=0.19
    --message K | random                standard message 0-2, or a random one
    --preamble S                        phasing preamble length, seconds
    --ppm P                             transmitter clock error, ppm
    --offset HZ                         tone mistuning relative to the decoder
    --fade HZ                           Rayleigh fading, Doppler bandwidth
    --impulses RATE --impulse-db DB     impulsive noise, per second / level
    --cw HZ --sir DB                    CW interferer frequency / SIR

Scoring
--------
The decoded text is aligned to the reference with a semi-global alignment:
the whole reference must be matched, but junk before and after it in the
decoded text is free. Every reference character is then classed as correct,
or as one of

    lost     deleted: no decoded character at all
    flagged  replaced by (or an extra) '~' or '!', which the decoder marks
             itself as unreliable
    silent   replaced by (or an extra) wrong character with no flag

CER (character error rate) = (lost + flagged + silent) / reference length.
Silent errors are the damaging ones, so silent CER is reported separately.

Lock time is measured from the start of the message's first character to
the moment "ZCZC" has been decoded. Even on a perfect signal it is about
0.85 s (four characters plus the five-slot repeat delay). The time stamp is
the amount of audio consumed when the character came out, accurate to one
20 ms chunk.
"""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from navtex_config import Profile, load_profile
from navtex_session import SignalStrengthTracker, build_config, decode_characters
from navtex_soft_fec_combine import SoftBitSync
from navtex_step1_sampling_windowing import AudioSource, Windower
from navtex_step2_tone_detection import ToneDetector
from navtex_synth import (
    STANDARD_MESSAGES,
    Channel,
    apply_channel,
    build_transmission,
    ebn0_db,
    random_message,
    theory_raw_ber,
)

FLAG_CHARS = "~!"


# ---------------------------------------------------------------------------
# Running the decoder on an array
# ---------------------------------------------------------------------------

class ArraySource(AudioSource):
    """An AudioSource that plays a NumPy array in 20 ms chunks. `consumed`
    is the number of samples handed to the decoder so far."""

    def __init__(self, audio: np.ndarray, block: int = 960):
        self._audio = np.asarray(audio, dtype=np.float32)
        self._block = block
        self.consumed = 0

    def chunks(self) -> Iterator[np.ndarray]:
        for start in range(0, len(self._audio), self._block):
            chunk = self._audio[start:start + self._block]
            self.consumed = start + len(chunk)
            yield chunk


@dataclass
class DecodeResult:
    text: str                  # everything the decoder printed
    times_s: List[float]       # audio time at which each character came out
    runtime_s: float           # wall-clock decode time
    signal_level: int          # SignalStrengthTracker reading at the end (00-99)


def run_decoder(audio: np.ndarray, profile: Profile) -> DecodeResult:
    """Decodes `audio` with the full production pipeline."""
    source = ArraySource(audio)
    tracker = SignalStrengthTracker(window=profile.signal_strength_window)
    chars: List[str] = []
    times: List[float] = []
    started = time.perf_counter()
    for ch in decode_characters(source, profile, tracker):
        chars.append(ch)
        times.append(source.consumed / profile.sample_rate)
    return DecodeResult("".join(chars), times, time.perf_counter() - started, tracker.level)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class Score:
    ref_len: int
    correct: int
    lost: int
    flagged: int
    silent: int

    @property
    def errors(self) -> int:
        return self.lost + self.flagged + self.silent

    @property
    def cer(self) -> float:
        return self.errors / self.ref_len if self.ref_len else 0.0

    @property
    def silent_cer(self) -> float:
        return self.silent / self.ref_len if self.ref_len else 0.0


def align_score(ref: str, hyp: str) -> Score:
    """Semi-global alignment of `ref` against `hyp` (see module docstring)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return Score(0, 0, 0, 0, 0)
    hyp_arr = np.frombuffer(hyp.encode("utf-32-le"), dtype=np.uint32) if m else np.zeros(0, np.uint32)
    index = np.arange(m + 1)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    for i in range(1, n + 1):
        mismatch = (hyp_arr != ord(ref[i - 1])).astype(np.int32)
        tmp = np.empty(m + 1, dtype=np.int32)
        tmp[0] = i
        if m:
            tmp[1:] = np.minimum(D[i - 1, :-1] + mismatch, D[i - 1, 1:] + 1)
        D[i] = np.minimum.accumulate(tmp - index) + index

    i, j = n, int(np.argmin(D[n]))
    correct = lost = flagged = silent = 0
    while i > 0:
        if j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] == hyp[j - 1]:
                correct += 1
            elif hyp[j - 1] in FLAG_CHARS:
                flagged += 1
            else:
                silent += 1
            i -= 1
            j -= 1
        elif D[i, j] == D[i - 1, j] + 1:
            lost += 1
            i -= 1
        else:                                   # extra decoded character
            if hyp[j - 1] in FLAG_CHARS:
                flagged += 1
            else:
                silent += 1
            j -= 1
    return Score(n, correct, lost, flagged, silent)


def measure_raw_ber(tx, audio: np.ndarray, profile: Profile,
                    skip_bits: int = 100) -> Tuple[int, int]:
    """Raw bit errors of the front end alone (windowing, tone detection and
    bit-clock recovery), compared with the bits actually sent. Returns
    (errors, bits compared).

    The first `skip_bits` transmitted bits are ignored so the bit-clock loop
    can settle. Decisions are matched to sent bits by trying a few offsets
    either side of the silent lead-in and keeping the best, which also
    absorbs the loop's start-up delay."""
    config = build_config(profile)
    source = ArraySource(audio)
    decisions = [d.bit for d in SoftBitSync(config, loop_gain=profile.loop_gain).process_stream(
        ToneDetector(config).process_stream(Windower(config).frames(source)))]
    got = np.array(decisions, dtype=bool)
    sent = tx.bits
    lead_bits = int(round(tx.bits_start_s * config.baud))
    best: Optional[Tuple[int, int]] = None
    for delta in range(-6, 7):
        first = lead_bits + delta + skip_bits               # decision index for sent bit `skip_bits`
        count = min(len(sent) - skip_bits, len(got) - first)
        if first < 0 or count <= 0:
            continue
        errors = int(np.count_nonzero(got[first:first + count] != sent[skip_bits:skip_bits + count]))
        if best is None or errors < best[0]:
            best = (errors, count)
    if best is None:
        raise ValueError("audio too short to measure bit errors")
    return best


def lock_time_s(result: DecodeResult, message_start_s: float) -> Optional[float]:
    """Seconds from the message start until ZCZC has been decoded, or None."""
    k = result.text.find("ZCZC")
    if k < 0:
        return None
    return result.times_s[k + 3] - message_start_s


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrialSpec:
    profile: Profile
    message: str
    snr_db: Optional[float]
    seed: int
    channel: Channel = Channel()          # impairments other than SNR and seed
    preamble_s: float = 5.0
    clock_ppm: float = 0.0
    freq_offset_hz: float = 0.0


@dataclass
class TrialResult:
    snr_db: Optional[float]
    seed: int
    cer: float
    silent_cer: float
    lost: int
    flagged: int
    silent: int
    ref_len: int
    exact: bool
    header_ok: bool
    lock_s: Optional[float]
    signal_level: int
    runtime_s: float
    audio_s: float


def run_trial(spec: TrialSpec) -> TrialResult:
    p = spec.profile
    tx = build_transmission(
        spec.message, sample_rate=p.sample_rate, mark_freq=p.mark_freq,
        space_freq=p.space_freq, preamble_s=spec.preamble_s,
        clock_ppm=spec.clock_ppm, freq_offset_hz=spec.freq_offset_hz)
    channel = dataclasses.replace(spec.channel, snr_db=spec.snr_db, seed=spec.seed)
    audio = apply_channel(tx.audio, tx.sample_rate, tx.signal_power, channel)
    result = run_decoder(audio, p)
    score = align_score(tx.expected_text, result.text)
    header = tx.expected_text.splitlines()[0]
    return TrialResult(
        snr_db=spec.snr_db, seed=spec.seed, cer=score.cer, silent_cer=score.silent_cer,
        lost=score.lost, flagged=score.flagged, silent=score.silent, ref_len=score.ref_len,
        exact=score.errors == 0, header_ok=header in result.text,
        lock_s=lock_time_s(result, tx.message_start_s),
        signal_level=result.signal_level, runtime_s=result.runtime_s,
        audio_s=tx.duration_s)


def run_ber_trial(spec: TrialSpec) -> Tuple[Optional[float], int, int]:
    """Raw bit errors of the front end for one transmission. Returns
    (snr_db, errors, bits compared)."""
    p = spec.profile
    tx = build_transmission(
        spec.message, sample_rate=p.sample_rate, mark_freq=p.mark_freq,
        space_freq=p.space_freq, preamble_s=spec.preamble_s,
        clock_ppm=spec.clock_ppm, freq_offset_hz=spec.freq_offset_hz)
    channel = dataclasses.replace(spec.channel, snr_db=spec.snr_db, seed=spec.seed)
    audio = apply_channel(tx.audio, tx.sample_rate, tx.signal_power, channel)
    errors, count = measure_raw_ber(tx, audio, p)
    return spec.snr_db, errors, count


@dataclass(frozen=True)
class NoiseSpec:
    profile: Profile
    snr_db: float            # sets the noise level, relative to a standard signal
    duration_s: float
    seed: int
    channel: Channel = Channel()


def run_noise_trial(spec: NoiseSpec) -> Tuple[int, int, float]:
    """Decodes noise with no signal. Returns (unflagged characters,
    flagged characters, minutes of audio) -- unflagged characters are false
    text the decoder presented as real."""
    p = spec.profile
    n = int(spec.duration_s * p.sample_rate)
    channel = dataclasses.replace(spec.channel, snr_db=spec.snr_db, seed=spec.seed)
    audio = apply_channel(np.zeros(n), p.sample_rate, 0.1 ** 2 / 2.0, channel)
    text = run_decoder(audio, p).text
    flagged = sum(1 for c in text if c in FLAG_CHARS)
    unflagged = sum(1 for c in text if c not in FLAG_CHARS and not c.isspace())
    return unflagged, flagged, spec.duration_s / 60.0


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def run_sweep(specs: Sequence[TrialSpec], jobs: int = 1) -> List[TrialResult]:
    if jobs <= 1:
        return [run_trial(s) for s in specs]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(run_trial, specs, chunksize=1))


def summarise(results: Sequence[TrialResult]) -> List[Dict[str, float]]:
    """One row per SNR: averages over that SNR's trials."""
    rows = []
    for snr in sorted({r.snr_db for r in results}, key=lambda s: (s is None, s)):
        group = [r for r in results if r.snr_db == snr]
        total = sum(r.ref_len for r in group)
        locked = [r.lock_s for r in group if r.lock_s is not None]
        rows.append(dict(
            snr_db=snr, trials=len(group),
            cer=sum(r.lost + r.flagged + r.silent for r in group) / total,
            silent_cer=sum(r.silent for r in group) / total,
            flagged=sum(r.flagged for r in group) / total,
            lost=sum(r.lost for r in group) / total,
            exact=sum(r.exact for r in group) / len(group),
            header_ok=sum(r.header_ok for r in group) / len(group),
            locked=len(locked) / len(group),
            lock_s=float(np.median(locked)) if locked else float("nan"),
        ))
    return rows


def print_summary(rows: Sequence[Dict[str, float]], ref_bw_hz: float = 2500.0) -> None:
    print(f"{'SNR dB':>7} {'Eb/N0':>6} {'theory':>7} {'trials':>6} {'CER %':>7} {'silent %':>8} "
          f"{'flagged %':>9} {'lost %':>7} {'exact %':>7} {'header %':>8} {'lock s':>7}")
    for r in rows:
        snr = r["snr_db"]
        if snr is None:
            head = f"{'none':>7} {'':>6} {'':>7}"
        else:
            head = f"{snr:7.1f} {ebn0_db(snr, ref_bw_hz):6.1f} {100 * theory_raw_ber(snr, ref_bw_hz):6.2f}%"
        print(f"{head} {r['trials']:6d} {100 * r['cer']:7.2f} {100 * r['silent_cer']:8.2f} "
              f"{100 * r['flagged']:9.2f} {100 * r['lost']:7.2f} {100 * r['exact']:7.0f} "
              f"{100 * r['header_ok']:8.0f} {r['lock_s']:7.2f}")
    print("\ntheory = ideal non-coherent FSK raw bit error rate at that SNR (the decoder cannot beat it)")


def write_csv(path: str, results: Sequence[TrialResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[fld.name for fld in dataclasses.fields(TrialResult)])
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_snrs(text: str) -> List[float]:
    """'-14:0:2' (start:stop:step, stop inclusive) or '-10,-8,-6'."""
    if ":" in text:
        start, stop, step = (float(x) for x in text.split(":"))
        count = int(math.floor((stop - start) / step + 1e-9)) + 1
        return [round(start + i * step, 6) for i in range(count)]
    return [float(x) for x in text.split(",")]


def build_profile(args: argparse.Namespace) -> Profile:
    if args.profile_file:
        if not args.profile_name:
            raise SystemExit("--profile-file needs --profile-name")
        profile = load_profile(args.profile_file, args.profile_name)
    else:
        profile = Profile()
    overrides = {}
    for item in args.set or []:
        key, _, raw = item.partition("=")
        if key not in {f.name for f in dataclasses.fields(Profile)}:
            raise SystemExit(f"--set: {key!r} is not a Profile field")
        try:
            overrides[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            overrides[key] = raw
    profile = dataclasses.replace(profile, **overrides)
    profile.validate()
    return profile


def build_channel(args: argparse.Namespace) -> Channel:
    return Channel(ref_bw_hz=args.ref_bw, fade_doppler_hz=args.fade,
                   impulse_rate_hz=args.impulses, impulse_level_db=args.impulse_db,
                   interferer_hz=args.cw, interferer_sir_db=args.sir)


def pick_message(args: argparse.Namespace, seed: int) -> str:
    if args.message == "random":
        return random_message(np.random.default_rng(seed))
    return STANDARD_MESSAGES[int(args.message) % len(STANDARD_MESSAGES)]


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile-file")
    p.add_argument("--profile-name")
    p.add_argument("--set", action="append", metavar="KEY=VALUE")
    p.add_argument("--message", default="0", help="0-2 for a standard message, or 'random'")
    p.add_argument("--preamble", type=float, default=5.0, help="phasing preamble, seconds")
    p.add_argument("--ppm", type=float, default=0.0, help="transmitter clock error, ppm")
    p.add_argument("--offset", type=float, default=0.0, help="tone mistuning, Hz")
    p.add_argument("--fade", type=float, default=0.0, help="Rayleigh fading Doppler bandwidth, Hz")
    p.add_argument("--impulses", type=float, default=0.0, help="impulse rate, per second")
    p.add_argument("--impulse-db", type=float, default=10.0)
    p.add_argument("--cw", type=float, default=None, help="CW interferer frequency, Hz")
    p.add_argument("--sir", type=float, default=0.0, help="signal-to-interferer ratio, dB")
    p.add_argument("--ref-bw", type=float, default=2500.0, help="SNR reference bandwidth, Hz")
    p.add_argument("--seed", type=int, default=1)


def make_spec(args, profile: Profile, snr: Optional[float], seed: int) -> TrialSpec:
    return TrialSpec(profile=profile, message=pick_message(args, seed), snr_db=snr, seed=seed,
                     channel=build_channel(args), preamble_s=args.preamble,
                     clock_ppm=args.ppm, freq_offset_hz=args.offset)


def cmd_one(args) -> None:
    profile = build_profile(args)
    spec = make_spec(args, profile, args.snr, args.seed)
    tx = build_transmission(spec.message, sample_rate=profile.sample_rate,
                            mark_freq=profile.mark_freq, space_freq=profile.space_freq,
                            preamble_s=spec.preamble_s, clock_ppm=spec.clock_ppm,
                            freq_offset_hz=spec.freq_offset_hz)
    channel = dataclasses.replace(spec.channel, snr_db=spec.snr_db, seed=spec.seed)
    result = run_decoder(apply_channel(tx.audio, tx.sample_rate, tx.signal_power, channel), profile)
    score = align_score(tx.expected_text, result.text)
    print("--- sent ---")
    print(tx.expected_text)
    print("--- decoded ---")
    print(result.text)
    print("--- score ---")
    lock = lock_time_s(result, tx.message_start_s)
    print(f"CER {100 * score.cer:.2f}%  (lost {score.lost}, flagged {score.flagged}, "
          f"silent {score.silent} of {score.ref_len})  lock "
          f"{'never' if lock is None else f'{lock:.2f} s'}  decode time {result.runtime_s:.1f} s "
          f"for {tx.duration_s:.1f} s of audio")


def cmd_sweep(args) -> None:
    profile = build_profile(args)
    snrs = parse_snrs(args.snr)
    specs = [make_spec(args, profile, snr, args.seed + 1000 * si + t)
             for si, snr in enumerate(snrs) for t in range(args.trials)]
    print(f"{len(specs)} trials ({len(snrs)} SNRs x {args.trials}), {args.jobs} job(s)...",
          file=sys.stderr)
    started = time.perf_counter()
    results = run_sweep(specs, args.jobs)
    print(f"done in {time.perf_counter() - started:.0f} s\n", file=sys.stderr)
    print_summary(summarise(results), args.ref_bw)
    if args.csv:
        write_csv(args.csv, results)
        print(f"per-trial results written to {args.csv}")


def cmd_ber(args) -> None:
    profile = build_profile(args)
    snrs = parse_snrs(args.snr)
    specs = [make_spec(args, profile, snr, args.seed + 1000 * si + t)
             for si, snr in enumerate(snrs) for t in range(args.trials)]
    if args.jobs <= 1:
        results = [run_ber_trial(s) for s in specs]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(run_ber_trial, specs, chunksize=1))
    print(f"{'SNR dB':>7} {'Eb/N0':>6} {'bits':>9} {'measured':>9} {'theory':>8} {'loss dB':>8}")
    for snr in snrs:
        errors = sum(e for s, e, _ in results if s == snr)
        bits = sum(n for s, _, n in results if s == snr)
        measured = errors / bits
        theory = theory_raw_ber(snr, args.ref_bw)
        if 0.0 < measured < 0.5:
            effective = -2.0 * math.log(2.0 * measured)            # Eb/N0 that would give this BER
            loss = f"{ebn0_db(snr, args.ref_bw) - 10.0 * math.log10(effective):8.2f}"
        else:
            loss = f"{'n/a':>8}"
        print(f"{snr:7.1f} {ebn0_db(snr, args.ref_bw):6.1f} {bits:9d} {100 * measured:8.3f}% "
              f"{100 * theory:7.3f}% {loss}")
    print("\nloss = how many dB worse than an ideal non-coherent FSK detector the front end is")


def cmd_noise(args) -> None:
    profile = build_profile(args)
    channel = build_channel(args)
    seconds = args.minutes * 60.0
    unflagged, flagged, minutes = run_noise_trial(
        NoiseSpec(profile, args.snr, seconds, args.seed, channel))
    print(f"{minutes:.1f} min of noise at SNR {args.snr} dB: "
          f"{unflagged} unflagged characters ({unflagged / minutes:.1f}/min), "
          f"{flagged} flagged ({flagged / minutes:.1f}/min)")


def _join_negative_values(argv: List[str]) -> List[str]:
    """argparse reads a value like '-14:0:2' or '-10,-8' as an option, so
    '--snr -14:0:2' is rewritten as '--snr=-14:0:2'. Plain negative numbers
    (--snr -8) already work and are left alone."""
    out: List[str] = []
    for i, arg in enumerate(argv):
        if i > 0 and argv[i - 1] in ("--snr", "--offset", "--sir", "--impulse-db") and arg.startswith("-"):
            out[-1] = f"{out[-1]}={arg}"
        else:
            out.append(arg)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("one", help="one trial; prints sent and decoded text")
    add_common(one)
    one.add_argument("--snr", type=float, default=None, help="SNR in dB (omit for no noise)")
    one.set_defaults(func=cmd_one)

    sweep = sub.add_parser("sweep", help="CER against SNR")
    add_common(sweep)
    sweep.add_argument("--snr", default="-14:0:2", help="'start:stop:step' or a comma list, dB")
    sweep.add_argument("--trials", type=int, default=10, help="trials per SNR")
    sweep.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    sweep.add_argument("--csv", help="write per-trial results to this CSV file")
    sweep.set_defaults(func=cmd_sweep)

    ber = sub.add_parser("ber", help="raw bit error rate of the front end against SNR")
    add_common(ber)
    ber.add_argument("--snr", default="-12:0:2", help="'start:stop:step' or a comma list, dB")
    ber.add_argument("--trials", type=int, default=10, help="trials per SNR")
    ber.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ber.set_defaults(func=cmd_ber)

    noise = sub.add_parser("noise", help="false text on noise alone")
    add_common(noise)
    noise.add_argument("--snr", type=float, required=True,
                       help="sets the noise level, relative to a standard signal")
    noise.add_argument("--minutes", type=float, default=5.0)
    noise.set_defaults(func=cmd_noise)

    args = parser.parse_args(_join_negative_values(sys.argv[1:] if argv is None else list(argv)))
    args.func(args)


if __name__ == "__main__":
    main()
