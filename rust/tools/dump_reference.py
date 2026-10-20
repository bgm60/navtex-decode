#!/usr/bin/env python3
# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Reference-vector capture for the Rust port
===========================================

Runs a signal through the real Python decoder and saves what every stage
produced, so each Rust stage can be tested against it.

Usage (run from anywhere; the decoder modules are found in the repo root):

    python rust/tools/dump_reference.py synth --name synth_snr10_seed1 --seed 1 --snr 10
    python rust/tools/dump_reference.py wav my_recording.wav --name real_clip1
    python rust/tools/dump_reference.py scenario slip-drop --name scen_slip_drop
    python rust/tools/dump_reference.py combine-cases

Output goes to rust/test-vectors/<name>/ (change the root with --out).

Files written for every signal (synth, wav, scenario)
-----------------------------------------------------
input.wav            The exact mono signal the decoder received, at the
                     decoder's sample rate, as 16-bit PCM. (A WAV in another
                     format or rate is converted first, exactly as the
                     decoder's own FileSource does it, and then written
                     here; the reference run reads THIS file back, so what
                     is stored is what the reference results came from.)
                     16-bit PCM values are read as value / 32768.
window.csv           The analysis window coefficients (float32 values).
tone.csv             One row per analysis frame:   frame, diff, bit
bits.csv             One row per recovered bit:    bit_index, bit, confidence, soft_value
lock_<v>_chars.csv   Characters from the weak-signal lock, for variant <v>:
                                                   seq, bit_index, codepoints
lock_<v>_trace.csv   One row per comparison (14 bit decisions) of that lock:
                                                   bin, s0..s13, locked, h,
                                                   releases, switches, pending, emitted
decoded_<v>.txt      The characters as readable text.
manifest.json        Settings used, counts, library versions, and checks.

The three lock variants build on each other, so the Rust port can be tested
one layer at a time:

    core     the lock, replay, output delay and release only
             (follow = False, max_letter_run = 0, ghost_hold = 0)
    follow   core plus moving the lock to a better alignment
             (follow = True,  max_letter_run = 0, ghost_hold = 0)
    full     everything, with the decoder's default settings

Trace columns: s0..s13 are the 14 hypothesis scores of the comparison;
locked is 1 while a lock is held, h the locked hypothesis (-1 if none);
releases and switches are running totals; pending is the number of
characters held back; emitted the number released so far (by the end of
the comparison, not counting the final flush at the end of the stream).

Other file
----------
combine_cases.csv    (from the combine-cases command; one file in the root
                     of the output folder.) Sequences of character pairs
                     through the soft combining rule, with the characters
                     it produced and the shift state afterwards.

CSV conventions
---------------
* Plain comma separated, one header row, "\\n" line endings, no quoting.
* bit is 1 for mark and 0 for space.
* Floats are written with Python's repr(), which round-trips exactly.
* codepoints is the Unicode code point of the character(s) as decimal
  numbers (space separated if there is more than one), so no text needs
  escaping.

This script only reads the decoder; it changes nothing in it. The lock
trace relies on some of the lock's private attributes (see TracingSoftLock),
so it may need updating if navtex_soft_lock.py is restructured.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np          # noqa: E402
import scipy                # noqa: E402
import soundfile as sf      # noqa: E402

from navtex_config import Profile                                          # noqa: E402
from navtex_session import build_config                                    # noqa: E402
from navtex_soft_fec_combine import SoftBitSync, SoftFecCombiner           # noqa: E402
from navtex_soft_lock import SoftLockDecoder, SoftLockParams               # noqa: E402
from navtex_step1_sampling_windowing import FileSource, Windower           # noqa: E402
from navtex_step2_tone_detection import ToneDetector                       # noqa: E402
from navtex_step4_character_decode import CONTROL, LETTERS, PHASING_1, PHASING_2   # noqa: E402

DEFAULT_OUT = REPO_ROOT / "rust" / "test-vectors"

LOCK_VARIANTS = {
    "core": SoftLockParams(follow=False, max_letter_run=0, ghost_hold=0),
    "follow": SoftLockParams(follow=True, max_letter_run=0, ghost_hold=0),
    "full": SoftLockParams(),
}

SCENARIOS = {
    "slip-drop": "one bit lost from the audio in the middle of a message (the lock must follow)",
    "slip-dup": "one bit repeated in the audio in the middle of a message (the lock must follow)",
    "two-stations": "a second, equally strong station takes over from the first",
    "two-messages": "two messages with noise between them (lock, release, lock again)",
    "idle-gap": "idle signals between two messages (the junk filters)",
    "weak": "a single message at -10 dB (many damaged characters)",
}


# ---------------------------------------------------------------------------
# Running the Python pipeline
# ---------------------------------------------------------------------------

def run_front_end(chunks, config, loop_gain):
    """Windowing, tone detection and bit sync, exactly as the production
    pipeline chains them. Returns (tone samples, bit decisions)."""
    windower = Windower(config)
    detector = ToneDetector(config)
    bitsync = SoftBitSync(config, loop_gain=loop_gain)
    tones = []
    decisions = []
    for chunk in chunks:
        for frame in windower.push(chunk):
            sample = detector.process(frame)
            tones.append(sample)
            decisions.extend(bitsync.push(sample))
    return tones, decisions


def array_chunks(audio, size):
    for start in range(0, len(audio), size):
        yield audio[start:start + size]


class TracingSoftLock(SoftLockDecoder):
    """The soft lock, recording for every comparison its 14 scores and its
    state afterwards. It overrides two internal methods and reads private
    attributes, but changes none of the decoder's behaviour."""

    def __init__(self, params=None):
        super().__init__(params)
        self.trace = []
        self._emitted = 0
        self._last_scores = None

    def _bin_scores(self, b):
        scores = super()._bin_scores(b)
        self._last_scores = scores
        return scores

    def _step(self, b):
        for item in super()._step(b):
            self._emitted += 1
            yield item
        self.trace.append((
            b, [float(s) for s in self._last_scores], self._locked,
            self._h if self._locked else -1, self.releases, self.switches,
            len(self._pending), self._emitted,
        ))


def run_lock(decisions, params):
    decoder = TracingSoftLock(params)
    events = []
    for d in decisions:
        events.extend(decoder.push(d.soft_value))
    events.extend(decoder.finish())
    return decoder, events


# ---------------------------------------------------------------------------
# Building the input signal
# ---------------------------------------------------------------------------

def make_tx(profile, seed, n_lines, **kwargs):
    from navtex_synth import build_transmission, random_message

    text = random_message(np.random.default_rng(seed), n_lines=n_lines)
    return build_transmission(text, sample_rate=profile.sample_rate,
                              mark_freq=profile.mark_freq, space_freq=profile.space_freq,
                              **kwargs)


def synthetic_audio(args, profile):
    """A synthetic transmission through a noisy channel (all seeded)."""
    from navtex_synth import Channel, apply_channel

    tx = make_tx(profile, args.seed, args.lines)
    channel = Channel(snr_db=args.snr, fade_doppler_hz=args.fade_doppler, seed=args.seed)
    audio = apply_channel(tx.audio, tx.sample_rate, tx.signal_power, channel)
    info = {
        "kind": "synthetic",
        "seed": args.seed,
        "lines": args.lines,
        "channel": dataclasses.asdict(channel),
        "expected_text": tx.expected_text,
    }
    return audio, info


def scenario_audio(args, profile):
    """The situations the lock's rarer behaviours need. They follow the
    ones built in test_soft_lock.py, with shorter gaps to keep the files
    small. All seeds are fixed."""
    from navtex_synth import Channel, apply_channel

    fs = profile.sample_rate
    kind = args.kind
    info = {"kind": "scenario", "scenario": kind, "about": SCENARIOS[kind]}

    if kind in ("slip-drop", "slip-dup"):
        tx = make_tx(profile, 12, 4)
        spb = fs // 100                                     # samples per bit
        cut = int(round(tx.bits_start_s * fs)) + (len(tx.bits) // 2) * spb
        clean = tx.audio
        if kind == "slip-drop":
            clean = np.concatenate([clean[:cut], clean[cut + spb:]])
        else:
            clean = np.concatenate([clean[:cut + spb], clean[cut:]])
        audio = apply_channel(clean, fs, tx.signal_power, Channel(snr_db=0.0, seed=7))
        info.update(snr_db=0.0, slip_at_sample=cut)
        return audio, info

    if kind == "two-stations":
        a, b = make_tx(profile, 31, 4), make_tx(profile, 32, 4)
        start_b = int(len(a.audio) - 10 * fs) + 503         # an odd bit phase
        n = max(len(a.audio), start_b + len(b.audio)) + 5 * fs
        x = np.zeros(n)
        x[:len(a.audio)] += a.audio
        x[start_b:start_b + len(b.audio)] += b.audio
        audio = apply_channel(x, fs, a.signal_power, Channel(snr_db=-4.0, seed=1))
        info.update(snr_db=-4.0, second_station_starts_at_sample=start_b)
        return audio, info

    if kind == "two-messages":
        rng = np.random.default_rng(4)
        txs = [make_tx(profile, 21, 2), make_tx(profile, 22, 2)]
        power = txs[0].signal_power
        snr, seed, parts = -8.0, 100, []
        for tx in txs:
            gap = int(rng.uniform(12, 15) * fs) + int(rng.integers(0, 997))
            parts.append(apply_channel(np.zeros(gap), fs, power, Channel(snr_db=snr, seed=seed)))
            parts.append(apply_channel(tx.audio, fs, tx.signal_power,
                                       Channel(snr_db=snr, seed=seed + 1)))
            seed += 2
        parts.append(apply_channel(np.zeros(20 * fs), fs, power, Channel(snr_db=snr, seed=seed)))
        info.update(snr_db=snr)
        return np.concatenate(parts), info

    if kind == "idle-gap":
        first = make_tx(profile, 1, 2, preamble_s=5.0, tail_slots=40)
        second = make_tx(profile, 2, 2, preamble_s=1.0, tail_slots=12)
        audio = apply_channel(np.concatenate([first.audio, second.audio]), fs,
                              first.signal_power, Channel(snr_db=6.0, seed=2))
        info.update(snr_db=6.0)
        return audio, info

    if kind == "weak":
        tx = make_tx(profile, 5, 3)
        audio = apply_channel(tx.audio, fs, tx.signal_power, Channel(snr_db=-10.0, seed=2))
        info.update(snr_db=-10.0, expected_text=tx.expected_text)
        return audio, info

    raise SystemExit(f"unknown scenario {kind}")


def wav_audio(args, config):
    """A recording, converted to mono at the decoder's rate exactly as the
    decoder's own FileSource does it."""
    chunks = list(FileSource(config, str(args.path)).chunks())
    if not chunks:
        raise SystemExit(f"{args.path} contains no audio")
    audio = np.concatenate(chunks).astype(np.float32)
    return audio, {"kind": "wav", "source_file": Path(args.path).name}


# ---------------------------------------------------------------------------
# Writing files
# ---------------------------------------------------------------------------

def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def num(x):
    """Full-precision text for a float."""
    return repr(float(x))


def codepoints(text):
    return " ".join(str(ord(c)) for c in text)


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def write_lock_files(out, variant, decoder, events):
    write_csv(out / f"lock_{variant}_chars.csv", ["seq", "bit_index", "codepoints"],
              ((i, pos, codepoints(ch)) for i, (ch, pos) in enumerate(events)))
    header = (["bin"] + [f"s{i}" for i in range(14)]
              + ["locked", "h", "releases", "switches", "pending", "emitted"])
    write_csv(out / f"lock_{variant}_trace.csv", header,
              ([b, *map(num, scores), int(locked), h, releases, switches, pending, emitted]
               for b, scores, locked, h, releases, switches, pending, emitted in decoder.trace))
    write_text(out / f"decoded_{variant}.txt", "".join(ch for ch, _ in events))


# ---------------------------------------------------------------------------
# The capture of one signal
# ---------------------------------------------------------------------------

def capture(audio, source_info, name, out_root, profile):
    config = build_config(profile)
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)

    # 1. The input signal, stored as 16-bit PCM and read back so that the
    #    stored file is exactly what the reference run sees.
    wav_path = out / "input.wav"
    sf.write(str(wav_path), np.clip(audio, -1.0, 1.0), profile.sample_rate, subtype="PCM_16")
    stored, rate = sf.read(str(wav_path), dtype="float32")
    assert rate == profile.sample_rate and stored.ndim == 1

    # 2. Reference run: read the stored file the way the decoder reads files.
    chunks = FileSource(config, str(wav_path)).chunks()
    tones, decisions = run_front_end(chunks, config, profile.loop_gain)

    # 3. Check that chunk size does not change the front-end results. The
    #    Rust port will be fed chunks of whatever size the caller likes, so
    #    this property is one it has to keep.
    chunking_ok = True
    for size in (7, 777, 48000):
        t2, d2 = run_front_end(array_chunks(stored, size), config, profile.loop_gain)
        same = ([(t.diff, t.bit) for t in tones] == [(t.diff, t.bit) for t in t2]
                and [(d.bit, d.confidence, d.soft_value) for d in decisions]
                == [(d.bit, d.confidence, d.soft_value) for d in d2])
        if not same:
            chunking_ok = False
            print(f"WARNING: chunk size {size} gives different front-end results", file=sys.stderr)

    # 4. The weak-signal lock, in its three layers.
    lock_summary = {}
    results = {}
    for variant, params in LOCK_VARIANTS.items():
        decoder, events = run_lock(decisions, params)
        results[variant] = (decoder, events)
        write_lock_files(out, variant, decoder, events)
        lock_summary[variant] = {
            "params": dataclasses.asdict(params),
            "comparisons": len(decoder.trace),
            "chars": len(events),
            "releases": decoder.releases,
            "switches": decoder.switches,
        }

    # 5. The front-end files and the manifest.
    window = Windower(config)._window_fn
    write_csv(out / "window.csv", ["index", "value"],
              ((i, num(v)) for i, v in enumerate(window)))
    write_csv(out / "tone.csv", ["frame", "diff", "bit"],
              ((i, num(t.diff), int(t.bit)) for i, t in enumerate(tones)))
    write_csv(out / "bits.csv", ["bit_index", "bit", "confidence", "soft_value"],
              ((i, int(d.bit), num(d.confidence), num(d.soft_value))
               for i, d in enumerate(decisions)))

    manifest = {
        "name": name,
        "source": source_info,
        "samples": int(len(stored)),
        "sample_rate": profile.sample_rate,
        "counts": {"frames": len(tones), "bits": len(decisions)},
        "front_end": {
            "window_type": config.window_type,
            "window_size": config.window_size,
            "hop_size": config.hop_size,
            "samples_per_symbol": config.samples_per_symbol,
            "oversample": config.oversample,
            "mark_freq": config.mark_freq,
            "space_freq": config.space_freq,
            "loop_gain": profile.loop_gain,
        },
        "lock": lock_summary,
        "checks": {"chunk_size_independent": chunking_ok},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "python": sys.version.split()[0]},
        "notes": [
            "The window is scipy's periodic ('fftbins=True') window, stored as float32; "
            "windowing multiplies float32 samples by float32 coefficients, and the "
            "tone detector then works in float64.",
            "Frame count = floor((samples - window_size) / hop_size) + 1.",
            "soft_value and confidence are sums of float64 diffs, added the way "
            "Python 3.12 and later sum() adds them (compensated).",
        ],
    }
    write_text(out / "manifest.json", json.dumps(manifest, indent=2) + "\n")

    print(f"{name}: {len(stored)} samples, {len(tones)} frames, {len(decisions)} bits")
    for variant, s in lock_summary.items():
        print(f"  lock {variant:6s} {s['comparisons']:4d} comparisons, {s['chars']:4d} chars, "
              f"{s['releases']} releases, {s['switches']} switches")
    removed = lock_summary["follow"]["chars"] - lock_summary["full"]["chars"]
    print(f"  the junk filters removed {removed} characters")
    print(f"  chunk-size independent: {chunking_ok}")
    print(f"  written to {out}")
    return 0 if chunking_ok else 1


# ---------------------------------------------------------------------------
# Cases for the soft combining rule
# ---------------------------------------------------------------------------

def make_combine_cases(args, out_root):
    """Sequences of character pairs through SoftFecCombiner._combine: the
    rule that decides one character from its two copies and tracks the
    LTRS/FIGS shift state. Each sequence uses a fresh combiner, so the
    shift state carries along it."""
    rng = np.random.default_rng(args.seed)
    traffic = list(LETTERS) + list(CONTROL)
    shifts = [code for code, name in CONTROL.items() if name in ("LTRS", "FIGS")]
    phasing = [PHASING_1, PHASING_2]

    def random_code():
        return "".join(rng.choice(["0", "1"], size=7))

    def soft_copy(code, sigma):
        out = []
        for ch in code:
            magnitude = rng.uniform(0.3, 6.0)
            value = (magnitude if ch == "1" else -magnitude) + rng.normal(0.0, sigma)
            out.append(float(value))
        return out

    rows = []
    stats = Counter()
    for seq in range(args.sequences):
        combiner = SoftFecCombiner()
        sigma = float(rng.choice([0.3, 1.0, 2.0, 3.5]))
        for step in range(args.length):
            r = rng.random()
            if r < 0.10:
                dx_true = str(rng.choice(shifts))
            elif r < 0.18:
                dx_true = str(rng.choice(phasing))
            elif r < 0.26:
                dx_true = random_code()                     # any weight at all
            else:
                dx_true = str(rng.choice(traffic))
            rx_true = dx_true if rng.random() < 0.85 else str(rng.choice(traffic))
            dx_soft = soft_copy(dx_true, sigma)
            rx_soft = soft_copy(rx_true, sigma)
            dx = "".join("1" if v >= 0 else "0" for v in dx_soft)
            rx = "".join("1" if v >= 0 else "0" for v in rx_soft)
            out = "".join(combiner._combine(dx, dx_soft, rx, rx_soft))
            weight = sum(1 for a, b in zip(dx_soft, rx_soft) if a + b >= 0)
            stats[f"combined weight {weight if weight in (3, 4, 5) else 'other'}"] += 1
            kind = ("nothing" if out == "" else "'~'" if out == "~" else "'!'" if out == "!"
                    else "a character")
            stats[f"output: {kind}"] += 1
            rows.append([seq, step, dx, rx, *map(num, dx_soft), *map(num, rx_soft),
                         codepoints(out), int(combiner._letters_mode)])

    header = (["seq", "step", "dx", "rx"] + [f"dx{i}" for i in range(7)]
              + [f"rx{i}" for i in range(7)] + ["out", "letters_mode"])
    out_root.mkdir(parents=True, exist_ok=True)
    write_csv(out_root / "combine_cases.csv", header, rows)
    print(f"combine_cases.csv: {len(rows)} pairs in {args.sequences} sequences")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"written to {out_root / 'combine_cases.csv'}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="folder that will hold the per-signal folders "
                        f"(default: {DEFAULT_OUT})")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="generate a seeded synthetic signal")
    s.add_argument("--name", required=True, help="name of the output folder")
    s.add_argument("--seed", type=int, default=1, help="seed for message and noise (default 1)")
    s.add_argument("--snr", type=float, default=None,
                   help="SNR in dB in a 2500 Hz bandwidth; omit for no noise")
    s.add_argument("--lines", type=int, default=4, help="lines in the message (default 4)")
    s.add_argument("--fade-doppler", type=float, default=0.0,
                   help="Rayleigh fading Doppler bandwidth in Hz; 0 for none")

    w = sub.add_parser("wav", help="use a recording")
    w.add_argument("path", type=Path, help="WAV file")
    w.add_argument("--name", required=True, help="name of the output folder")

    c = sub.add_parser("scenario", help="a situation that exercises the lock's rarer behaviours")
    c.add_argument("kind", choices=sorted(SCENARIOS),
                   help="; ".join(f"{k}: {v}" for k, v in sorted(SCENARIOS.items())))
    c.add_argument("--name", required=True, help="name of the output folder")

    k = sub.add_parser("combine-cases", help="cases for the soft combining rule")
    k.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    k.add_argument("--sequences", type=int, default=30, help="number of sequences (default 30)")
    k.add_argument("--length", type=int, default=200, help="pairs per sequence (default 200)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "combine-cases":
        return make_combine_cases(args, args.out)

    profile = Profile()                      # the decoder's current defaults
    config = build_config(profile)
    if args.command == "synth":
        audio, source_info = synthetic_audio(args, profile)
    elif args.command == "scenario":
        audio, source_info = scenario_audio(args, profile)
    else:
        audio, source_info = wav_audio(args, config)
    return capture(audio, source_info, args.name, args.out, profile)


if __name__ == "__main__":
    sys.exit(main())
