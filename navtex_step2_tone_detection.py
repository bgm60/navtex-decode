"""
NAVTEX 100-baud FSK Decoder — Step 2: Mark/Space Tone Detection
==================================================================

Pipeline this module is the second stage of:

    [1] Sampling & windowing        (navtex_step1_sampling_windowing.py)
    [2] Mark/space tone detection   <-- this file
    [3] Bit-clock recovery / symbol timing
    [4] CCIR 476 (SITOR-B) character decode + FEC
    [5] Message assembly / B1B2B3B4 header parsing

Scope of this module
---------------------
Consumes the `Frame` stream produced by Step 1's `Windower` and, for each
frame, measures the signal energy at the mark and space frequencies. It
produces one `ToneSample` per frame carrying both raw energies and a
gain-independent differential value. It does NOT decide where symbol
boundaries fall — because frames are 8x oversampled per bit (Step 1's
default), most consecutive ToneSamples describe the *same* bit measured at
slightly different offsets. Picking out one decision per bit is Step 3's
job (bit-clock recovery); this module's output is exactly what that stage
needs to work with.

Design summary
---------------
Each frame is already windowed (Hamming, by default) which matters here:
mark and space are only 170 Hz apart, so without windowing, energy from
one tone leaks significantly into the other's detector (spectral leakage
from the frame's rectangular-window sidelobes). The window applied in
Step 1 suppresses that leakage before it ever reaches this stage.

Energy at each target frequency is measured with a single-frequency DFT
correlation — i.e. what the classic recursive Goertzel algorithm computes,
just evaluated via a vectorized numpy dot product against precomputed
cos/sin basis vectors rather than the sequential recursion. Both give
numerically identical results; the vectorized form is simply faster under
numpy than emulating a per-sample recursive loop in Python. Because the
target frequency is not rounded to the nearest FFT bin, this works cleanly
even though window_size (480 samples @ 48 kHz -> 100 Hz bin spacing)
doesn't land mark/space exactly on integer bins.

Decision rule: whichever of mark/space has more energy wins (standard
non-coherent FSK detection — no fixed absolute threshold, so it doesn't
care about overall signal level). The normalized `diff` value gives a
continuous, gain-independent signal in [-1, +1] that Step 3 can use for
timing recovery (e.g. locating its zero-crossings / transitions), and
`total_power` is a rough per-frame "is there signal here at all" measure
useful later for detecting silence between transmissions or squelch.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from navtex_step1_sampling_windowing import (
    AudioSource,
    Frame,
    FileSource,
    NavtexConfig,
    SyntheticNavtexSource,
    Windower,
)


# ---------------------------------------------------------------------------
# Output of the tone detector
# ---------------------------------------------------------------------------

@dataclass
class ToneSample:
    """Per-frame mark/space energy measurement, handed to Step 3."""

    timestamp: float       # seconds, same as the source Frame
    start_sample: int      # sample index of the frame start
    mark_power: float       # energy at mark_freq
    space_power: float      # energy at space_freq
    diff: float             # (mark - space) / (mark + space), in [-1, +1]
    total_power: float      # mark + space -- rough signal-present measure
    bit: bool                # hard decision: True = mark (1), False = space (0)


# ---------------------------------------------------------------------------
# Tone detector
# ---------------------------------------------------------------------------

class ToneDetector:
    """Per-frame Goertzel-equivalent mark/space energy detector.

    The cos/sin correlation vectors depend only on window_size, the target
    frequencies, and sample_rate — all fixed for the lifetime of a
    NavtexConfig — so they're precomputed once here rather than per frame.
    """

    def __init__(self, config: NavtexConfig):
        self.config = config
        n = config.window_size
        t = np.arange(n)
        self._mark_cos = np.cos(2 * np.pi * config.mark_freq * t / config.sample_rate)
        self._mark_sin = np.sin(2 * np.pi * config.mark_freq * t / config.sample_rate)
        self._space_cos = np.cos(2 * np.pi * config.space_freq * t / config.sample_rate)
        self._space_sin = np.sin(2 * np.pi * config.space_freq * t / config.sample_rate)

    @staticmethod
    def _tone_power(samples: np.ndarray, cos_vec: np.ndarray, sin_vec: np.ndarray) -> float:
        re = float(np.dot(samples, cos_vec))
        im = float(np.dot(samples, sin_vec))
        return re * re + im * im

    def process(self, frame: Frame) -> ToneSample:
        x = frame.windowed.astype(np.float64)
        mark_power = self._tone_power(x, self._mark_cos, self._mark_sin)
        space_power = self._tone_power(x, self._space_cos, self._space_sin)
        total = mark_power + space_power
        diff = (mark_power - space_power) / total if total > 1e-12 else 0.0
        return ToneSample(
            timestamp=frame.timestamp,
            start_sample=frame.start_sample,
            mark_power=mark_power,
            space_power=space_power,
            diff=diff,
            total_power=total,
            bit=mark_power >= space_power,
        )

    def process_stream(self, frames: Iterator[Frame]) -> Iterator[ToneSample]:
        for frame in frames:
            yield self.process(frame)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def _demo():
    """Feeds a synthetic (or WAV-file) signal through Step 1 + Step 2.

    Against the synthetic source, this also scores hard-decision accuracy
    against the known transmitted bits, checked only at frames that land
    exactly on a symbol boundary (every `oversample`-th frame) — i.e. this
    isolates tone-detection correctness from symbol-timing, since Step 3
    hasn't been built yet.
    """
    config = NavtexConfig()
    print("Config:", config.describe())

    if len(sys.argv) > 1:
        source: AudioSource = FileSource(config, sys.argv[1])
        print(f"Source: WAV file {sys.argv[1]!r}")
    else:
        source = SyntheticNavtexSource(config, duration_s=2.0)
        print("Source: synthetic mark/space test tone (no file given)")

    windower = Windower(config)
    detector = ToneDetector(config)
    samples_per_symbol = config.samples_per_symbol

    correct = 0
    checked = 0
    total_ones = 0
    total_frames = 0
    print_every_n_symbols = 20  # spread printout across the whole stream,
                                # not just the first symbol (which is only
                                # `oversample` frames long and therefore a
                                # single bit repeated)

    for sample in detector.process_stream(windower.frames(source)):
        total_frames += 1
        total_ones += int(sample.bit)

        is_symbol_start = sample.start_sample % samples_per_symbol == 0
        symbol_index = sample.start_sample // samples_per_symbol
        if is_symbol_start and symbol_index % print_every_n_symbols == 0:
            print(f"t={sample.timestamp:7.3f}s  mark={sample.mark_power:10.2f}  "
                  f"space={sample.space_power:10.2f}  diff={sample.diff:+.3f}  "
                  f"bit={int(sample.bit)}")

        if isinstance(source, SyntheticNavtexSource) and hasattr(source, "bits"):
            if sample.start_sample % samples_per_symbol == 0:
                symbol_index = sample.start_sample // samples_per_symbol
                if symbol_index < len(source.bits):
                    checked += 1
                    if bool(sample.bit) == bool(source.bits[symbol_index]):
                        correct += 1

    if total_frames:
        print(f"\nOverall (all oversampled frames): {total_ones}/{total_frames} "
              f"decided mark/1 ({100 * total_ones / total_frames:.1f}%)")

    if checked:
        print(f"Symbol-aligned decode accuracy: {correct}/{checked} "
              f"({100 * correct / checked:.1f}%)")


if __name__ == "__main__":
    _demo()
