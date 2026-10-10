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
    python navtex_bench.py sweep --snr -14:0:2 --trials 20 --jobs 4
    python navtex_bench.py sweep --snr -16:-4:1 --oracle --target 10,50
    python navtex_bench.py ber   --snr -12:2:2        # raw bit errors of the front end
    python navtex_bench.py noise --snr -6 --minutes 5 # false text on noise alone

Common options (all subcommands)
---------------------------------
    --profile-file F --profile-name N   decoder settings from a TOML profile
                                        (default: built-in Profile defaults)
    --set key=value                     override a Profile field (repeatable),
                                        e.g. --set loop_gain=0.19
    --message K | random                standard message 0-2, or a random one
                                        (default: 0 for "one", random for "sweep")
    --preamble S                        phasing preamble length, seconds
    --ppm P                             transmitter clock error, ppm
    --offset HZ                         tone mistuning relative to the decoder
    --fade HZ                           Rayleigh fading, Doppler bandwidth
    --impulses RATE --impulse-db DB     impulsive noise, per second / level
    --cw HZ --sir DB                    CW interferer frequency / SIR

Sweep options
--------------
    --target 10,50      print the SNR at which character recovery first
                        reaches each percentage (linear interpolation)
    --oracle            also decode the same audio with the character
                        alignment (A) and then alignment and DX/RX parity (B)
                        forced to the truth. B is a ceiling for what better
                        synchronisation could achieve; the gap between
                        production and B shows how much is on the table.

Scoring
--------
Two scores are reported, for different regimes.

Character recovery (position-matched) is the headline figure for weak
signals. A character counts as recovered only if the decoder emits it for
the same transmitted slot that carried it, so garbage between the real
characters is tolerated and does not inflate the score. The denominator is
the letters, digits and punctuation sent (not space, CR or LF).

    recov %    recovered characters / characters sent
    chance %   the same comparison with the output shifted 20-60 slots, i.e.
               how often garbage matches by coincidence (typically 1-3%)
    hdr8 %     recovery of the first eight visible characters (ZCZC and the
               station code), which is usually what identifies the sender

Slots are identified by matching the decoder's own bit decisions to the sent
bits, so the scorer needs no help from the decoder. A tagged copy of the
production grouping and combining logic (see run_tagged_pipeline) does the
attribution; its text is checked against decode_characters in the tests, so
if the decoder changes and the copy is not updated the tests fail.

CER (character error rate) is the older, alignment-based score: the decoded
text is aligned to the reference with a semi-global alignment (the whole
reference must be matched, junk before and after it is free) and every
reference character is classed as correct, or as one of

    lost     deleted: no decoded character at all
    flagged  replaced by (or an extra) '~' or '!', which the decoder marks
             itself as unreliable
    silent   replaced by (or an extra) wrong character with no flag

CER = (lost + flagged + silent) / reference length. It is the better measure
when the text is mostly right, and is unreliable once most of it is garbage,
because a best-fit alignment gives chance matches credit.

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
from navtex_session import SignalStrengthTracker, build_config, build_grouper, decode_characters
from navtex_soft_fec_combine import SoftBitSync, SoftCharacterGrouper, SoftFecCombiner
from navtex_step1_sampling_windowing import AudioSource, Windower
from navtex_step2_tone_detection import ToneDetector
from navtex_step4_character_decode import CONTROL, FIGURES, LETTERS, PHASING_1, PHASING_2
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
PHASING = frozenset({PHASING_1, PHASING_2})
CHANCE_OFFSETS = [k for k in range(-60, 61, 2) if abs(k) >= 20]    # even keeps RX-slot parity
HEADER_CHARS = 8                                                   # "ZCZC" + station code


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
# Position-matched scoring
# ---------------------------------------------------------------------------
#
# The decoder prints text with no record of which transmitted slot each
# character came from, and a long stretch of garbage defeats any
# text-alignment score. So the front end is run once more here and the
# production grouping and combining logic is replayed over its bit decisions
# with each group of soft values tagged with the slot it came from. Every
# character the combiner emits is then known to belong to a slot, and can be
# compared with the character that slot carried.

class _Tagged(list):
    """A list of soft values that also carries the transmitted slot index."""
    slot: Optional[int] = None


class _Recording:
    """Mixin for a combiner: records the character emitted for each tagged
    RX slot."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.out: Dict[int, str] = {}
        self._slot: Optional[int] = None

    def _combine(self, dx, dxs, rx, rxs):
        self._slot = getattr(rxs, "slot", None)
        yield from super()._combine(dx, dxs, rx, rxs)

    def _decode(self, code):
        for ch in super()._decode(code):
            self.out[self._slot] = ch
            yield ch


class _RecordingFec(_Recording, SoftFecCombiner):
    pass


def _recording_fec(profile: Profile) -> _RecordingFec:
    """A SoftFecCombiner built from the profile exactly as
    navtex_session.build_fec builds it, with recording added."""
    return _RecordingFec(
        lock_window=profile.lock_window, acquire_threshold=profile.fec_acquire_threshold,
        switch_margin=profile.fec_switch_margin,
        min_samples_for_rate=profile.min_samples_for_rate, rate_window=profile.rate_window)


class _OracleGrouper(SoftCharacterGrouper):
    """Character alignment fixed to the truth (re-pointed block by block if
    the bit clock slips). Never acquires or reconsiders."""

    def __init__(self, phase_at, **kw):
        super().__init__(**kw)
        self._phase_at = phase_at
        self._active_phase = phase_at(0)

    def _reconsider(self) -> None:
        pass

    def push_bit(self, bit, soft_value=0.0):
        phase = self._phase_at(self.sync._total_bits)
        if phase != self._active_phase:
            self._active_phase = phase
            self._group = []
            self._group_soft = []
        yield from super().push_bit(bit, soft_value)


def front_end_decisions(audio: np.ndarray, profile: Profile) -> list:
    """The front end's soft bit decisions (windowing, tone detection and
    bit-clock recovery), exactly as the production pipeline produces them."""
    config = build_config(profile)
    return list(SoftBitSync(config, loop_gain=profile.loop_gain).process_stream(
        ToneDetector(config).process_stream(Windower(config).frames(ArraySource(audio)))))


def sent_slots(tx) -> List[str]:
    """The transmitted 7-bit codewords, in slot order (even = DX, odd = RX)."""
    bits = "".join("1" if b else "0" for b in tx.bits)
    return [bits[i:i + 7] for i in range(0, len(bits), 7)]


def visible_truth(slots: Sequence[str]) -> List[Tuple[int, str]]:
    """(RX slot, character) for every visible character sent: letters, digits
    and punctuation, not space, CR, LF or shifts. The RX slot is the one
    that repeats the character, five slots after its DX slot, because that
    is the slot at which the decoder emits it."""
    letters = True
    visible = []
    for j in range(0, len(slots), 2):
        code = slots[j]
        if code in CONTROL:
            if CONTROL[code] == "LTRS":
                letters = True
            elif CONTROL[code] == "FIGS":
                letters = False
        elif code in LETTERS:
            ch = (LETTERS if letters else FIGURES)[code]
            if ch is not None and ch.strip():
                visible.append((j + 5, ch))
    return visible


def estimate_slot_starts(decisions: Sequence, tx, lead_bits: int, block_slots: int = 100,
                         search: int = 6) -> np.ndarray:
    """Decision index at which each transmitted slot starts, found by
    matching the decoder's hard bit decisions to the sent bits block by block
    (so a bit-clock slip is followed). Needs the signal to be better than
    about 40% raw bit error; below that the estimate is unreliable."""
    got = np.fromiter((d.bit for d in decisions), dtype=bool, count=len(decisions))
    sent = tx.bits
    n_slots = len(sent) // 7
    deltas = np.zeros(n_slots, dtype=int)
    prev: Optional[int] = None
    for b0 in range(0, n_slots, block_slots):
        b1 = min(n_slots, b0 + block_slots)
        z = 7 * (b1 - b0)
        candidates = range(-search, search + 1) if prev is None else range(prev - 3, prev + 4)
        best_d, best_err = None, 2.0
        for d in candidates:
            a = lead_bits + d + 7 * b0
            if a < 0 or a + z > len(got):
                continue
            err = np.count_nonzero(got[a:a + z] != sent[7 * b0:7 * b1]) / z
            if err < best_err:
                best_d, best_err = d, err
        d = best_d if (best_d is not None and best_err < 0.42) else (prev or 0)
        deltas[b0:b1] = d
        prev = d
    return lead_bits + deltas + 7 * np.arange(n_slots)


def run_tagged_pipeline(decisions: Sequence, profile: Profile, grouper, fec: _RecordingFec,
                        slot_of) -> str:
    """A copy of navtex_soft_fec_combine.decode_bit_stream_soft that tags each
    group's soft values with the transmitted slot it ends in. Returns the
    decoded text; the per-slot characters are left in fec.out.

    KEEP IN SYNC BY HAND with decode_bit_stream_soft: test_synth_roundtrip
    checks that this copy and the production decoder give identical text."""
    prev_phase: Optional[int] = None
    consecutive = 0
    text: List[str] = []
    for bd in decisions:
        for code, soft in grouper.push_bit(bd.bit, bd.soft_value):
            tagged = _Tagged(soft)
            tagged.slot = slot_of(grouper.sync._total_bits - 1)
            if grouper._active_phase != prev_phase:
                fec.reset()
                prev_phase = grouper._active_phase
                consecutive = 0
            if code in PHASING:
                consecutive += 1
            else:
                if consecutive >= profile.phasing_burst_threshold:
                    fec.reset()
                consecutive = 0
            text.extend(fec.push(code, tagged))
    return "".join(text)


def oracle_parity_decode(decisions: Sequence, starts: np.ndarray, profile: Profile) -> Dict[int, str]:
    """Decodes with alignment AND DX/RX parity taken from the truth: every RX
    slot is combined with the DX slot five earlier by the production combining
    rule. No lock logic is involved, so this is what perfect synchronisation
    would give with the production combiner."""
    groups = []
    for k, a in enumerate(starts):
        if a + 7 > len(decisions):
            break
        window = decisions[a:a + 7]
        soft = _Tagged([d.soft_value for d in window])
        soft.slot = k
        groups.append(("".join("1" if d.bit else "0" for d in window), soft))
    fec = _recording_fec(profile)
    for s in range(5, len(groups), 2):
        (dx, dxs), (rx, rxs) = groups[s - 5], groups[s]
        for _ in fec._combine(dx, dxs, rx, rxs):
            pass
    return fec.out


@dataclass
class Positional:
    """Position-matched counts for one decoder variant on one trial."""
    hits: int = 0           # visible characters emitted for the right slot
    n: int = 0              # visible characters sent
    chance: int = 0         # hits when the output is shifted (coincidences)
    chance_n: int = 0
    hdr: int = 0            # hits among the first HEADER_CHARS visible characters
    hdr_n: int = 0

    @property
    def recovery(self) -> float:
        return self.hits / self.n if self.n else 0.0


def positional_score(out: Dict[int, str], visible: Sequence[Tuple[int, str]]) -> Positional:
    hits = sum(1 for s, c in visible if out.get(s) == c)
    chance = sum(1 for k in CHANCE_OFFSETS for s, c in visible if out.get(s + k) == c)
    header = visible[:HEADER_CHARS]
    return Positional(hits=hits, n=len(visible), chance=chance,
                      chance_n=len(visible) * len(CHANCE_OFFSETS),
                      hdr=sum(1 for s, c in header if out.get(s) == c), hdr_n=len(header))


def positional_scores(tx, audio: np.ndarray, profile: Profile, oracle: bool = False,
                      verify: bool = False) -> Dict[str, Positional]:
    """Position-matched scores for the production decoder ("P") and, if
    `oracle`, for the same bit decisions decoded with the character alignment
    forced to the truth ("A") and with alignment and parity forced ("B").
    `verify` raises AssertionError if the tagged copy of the pipeline does not
    reproduce decode_characters' text exactly."""
    decisions = front_end_decisions(audio, profile)
    lead_bits = int(round(tx.bits_start_s * build_config(profile).baud))
    starts = estimate_slot_starts(decisions, tx, lead_bits)
    n_slots = len(starts)

    def slot_of(g: int) -> int:
        return int(min(max(np.searchsorted(starts, g, side="right") - 1, 0), n_slots - 1))

    visible = visible_truth(sent_slots(tx))
    fec = _recording_fec(profile)
    text = run_tagged_pipeline(decisions, profile, build_grouper(profile), fec, slot_of)
    if verify:
        tracker = SignalStrengthTracker(window=profile.signal_strength_window)
        reference = "".join(decode_characters(ArraySource(audio), profile, tracker))
        assert reference == text, "tagged pipeline differs from the production decoder"
    scores = {"P": positional_score(fec.out, visible)}
    if oracle:
        fec_a = _recording_fec(profile)
        grouper = _OracleGrouper(lambda pos: int(starts[slot_of(pos)] % 7),
                                 sync_window=profile.sync_window)
        run_tagged_pipeline(decisions, profile, grouper, fec_a, slot_of)
        scores["A"] = positional_score(fec_a.out, visible)
        scores["B"] = positional_score(oracle_parity_decode(decisions, starts, profile), visible)
    return scores


def crossing(snrs: Sequence[float], values: Sequence[float], target: float) -> Optional[float]:
    """The SNR at which `values` (recovery %, rising with SNR) first reaches
    `target`, by linear interpolation. Returns -inf if the first point is
    already at or above the target (the crossing lies below the range
    tested) and None if the target is never reached."""
    if not values:
        return None
    if values[0] >= target:
        return -math.inf
    for (s0, v0), (s1, v1) in zip(zip(snrs, values), zip(snrs[1:], values[1:])):
        if v0 < target <= v1:
            return s0 + (target - v0) / (v1 - v0) * (s1 - s0)
    return None


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
    oracle: bool = False                  # also score the forced-synchronisation decodes


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
    # position-matched scoring (see "Position-matched scoring" above)
    recov_hits: int = 0
    recov_n: int = 0
    chance_hits: int = 0
    chance_n: int = 0
    hdr_hits: int = 0
    hdr_n: int = 0
    oracle: bool = False
    a_hits: int = 0                       # character alignment forced to the truth
    b_hits: int = 0                       # alignment and DX/RX parity forced to the truth


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
    pos = positional_scores(tx, audio, p, oracle=spec.oracle)
    return TrialResult(
        snr_db=spec.snr_db, seed=spec.seed, cer=score.cer, silent_cer=score.silent_cer,
        lost=score.lost, flagged=score.flagged, silent=score.silent, ref_len=score.ref_len,
        exact=score.errors == 0, header_ok=header in result.text,
        lock_s=lock_time_s(result, tx.message_start_s),
        signal_level=result.signal_level, runtime_s=result.runtime_s,
        audio_s=tx.duration_s,
        recov_hits=pos["P"].hits, recov_n=pos["P"].n, chance_hits=pos["P"].chance,
        chance_n=pos["P"].chance_n, hdr_hits=pos["P"].hdr, hdr_n=pos["P"].hdr_n,
        oracle=spec.oracle,
        a_hits=pos["A"].hits if spec.oracle else 0, b_hits=pos["B"].hits if spec.oracle else 0)


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
            recov=100 * sum(r.recov_hits for r in group) / max(1, sum(r.recov_n for r in group)),
            chance=100 * sum(r.chance_hits for r in group) / max(1, sum(r.chance_n for r in group)),
            hdr8=100 * sum(r.hdr_hits for r in group) / max(1, sum(r.hdr_n for r in group)),
            oracle=all(r.oracle for r in group),
            recov_a=100 * sum(r.a_hits for r in group) / max(1, sum(r.recov_n for r in group)),
            recov_b=100 * sum(r.b_hits for r in group) / max(1, sum(r.recov_n for r in group)),
        ))
    return rows


def print_summary(rows: Sequence[Dict[str, float]], ref_bw_hz: float = 2500.0,
                  targets: Sequence[float] = ()) -> None:
    oracle = bool(rows) and all(r["oracle"] for r in rows)
    head = (f"{'SNR dB':>7} {'Eb/N0':>6} {'theory':>7} {'trials':>6} | {'recov %':>7} {'chance %':>8} "
            f"{'hdr8 %':>6}")
    if oracle:
        head += f" {'A %':>6} {'B %':>6}"
    head += f" | {'CER %':>6} {'silent %':>8} {'exact %':>7} {'lock s':>7}"
    print(head)
    for r in rows:
        snr = r["snr_db"]
        if snr is None:
            lead = f"{'none':>7} {'':>6} {'':>7}"
        else:
            lead = f"{snr:7.1f} {ebn0_db(snr, ref_bw_hz):6.1f} {100 * theory_raw_ber(snr, ref_bw_hz):6.2f}%"
        line = f"{lead} {r['trials']:6d} | {r['recov']:7.1f} {r['chance']:8.2f} {r['hdr8']:6.0f}"
        if oracle:
            line += f" {r['recov_a']:6.1f} {r['recov_b']:6.1f}"
        line += (f" | {100 * r['cer']:6.2f} {100 * r['silent_cer']:8.2f} {100 * r['exact']:7.0f} "
                 f"{r['lock_s']:7.2f}")
        print(line)
    print("\ntheory = ideal non-coherent FSK raw bit error rate at that SNR (the decoder cannot beat it)")
    print("recov = characters emitted for the right slot; chance = coincidence rate; hdr8 = first 8 characters")
    if oracle:
        print("A = character alignment forced to the truth; B = alignment and DX/RX parity forced "
              "(a ceiling, not an achievable decoder)")
    snrs = [r["snr_db"] for r in rows if r["snr_db"] is not None]
    if targets and len(snrs) == len(rows) and len(snrs) > 1:
        columns = [("production", "recov")] + ([("A", "recov_a"), ("B", "recov_b")] if oracle else [])
        for target in targets:
            parts = []
            for name, key in columns:
                c = crossing(snrs, [r[key] for r in rows], target)
                parts.append(f"{name}: " + ("not reached" if c is None
                                            else f"below {snrs[0]:.1f} dB" if c == -math.inf
                                            else f"{c:.2f} dB"))
            print(f"SNR for {target:g}% character recovery: " + "   ".join(parts))


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
                     clock_ppm=args.ppm, freq_offset_hz=args.offset,
                     oracle=getattr(args, "oracle", False))


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
    audio = apply_channel(tx.audio, tx.sample_rate, tx.signal_power, channel)
    pos = positional_scores(tx, audio, profile, oracle=spec.oracle)
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
    p = pos["P"]
    line = (f"recovered {p.hits} of {p.n} characters ({100 * p.recovery:.1f}%; chance "
            f"{100 * p.chance / max(1, p.chance_n):.1f}%); {p.hdr} of the first {p.hdr_n} "
            f"characters")
    if spec.oracle:
        line += (f"; forced alignment {100 * pos['A'].recovery:.1f}%, forced alignment and parity "
                 f"{100 * pos['B'].recovery:.1f}%")
    print(line)


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
    targets = [float(x) for x in args.target.split(",")] if args.target else []
    print_summary(summarise(results), args.ref_bw, targets)
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
    one.add_argument("--oracle", action="store_true", help="also decode with forced synchronisation")
    one.set_defaults(func=cmd_one)

    sweep = sub.add_parser("sweep", help="character recovery and CER against SNR")
    add_common(sweep)
    sweep.set_defaults(message="random")
    sweep.add_argument("--snr", default="-14:0:2", help="'start:stop:step' or a comma list, dB")
    sweep.add_argument("--trials", type=int, default=10, help="trials per SNR")
    sweep.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    sweep.add_argument("--csv", help="write per-trial results to this CSV file")
    sweep.add_argument("--oracle", action="store_true",
                       help="also decode with forced alignment (A) and forced alignment and parity (B)")
    sweep.add_argument("--target", default="10,50",
                       help="comma list of recovery percentages to find the SNR for (blank to skip)")
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
