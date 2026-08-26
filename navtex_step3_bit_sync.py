"""
NAVTEX 100-baud FSK Decoder — Step 3: Bit-Clock Recovery
==================================================================

Pipeline this module is the third stage of:

    [1] Sampling & windowing        (navtex_step1_sampling_windowing.py)
    [2] Mark/space tone detection   (navtex_step2_tone_detection.py)
    [3] Bit-clock recovery          <-- this file
    [4] CCIR 476 (SITOR-B) character decode + FEC
    [5] Message assembly / B1B2B3B4 header parsing

Scope of this module
---------------------
Consumes the `ToneSample` stream from Step 2 (~`oversample` looks per
symbol, default 8) and produces exactly one `BitDecision` per transmitted
bit. Two problems have to be solved to do that, and they're coupled:

  1. We don't know the transmitter's bit-boundary phase. Step 1/2 frame the
     audio starting from sample 0 of whatever we happened to start
     recording; nothing guarantees that lines up with an actual bit edge.
  2. The transmitter's and receiver's sample clocks are two independent
     oscillators, so the effective symbol period as measured in our
     samples may drift slowly relative to our nominal
     `samples_per_symbol`, even if only by a few hundred ppm.

Design summary
---------------
This uses a proportional (Type-I) digital PLL, the same basic technique
used in software FSK demodulators such as minimodem and multimon-ng:

  * A phase accumulator tracks fractional progress through the current
    symbol, advancing by `hop_size / samples_per_symbol` per incoming
    ToneSample. When it wraps past 1.0, a symbol boundary has been
    reached.
  * Every frame's `diff` value (Step 2's gain-independent mark/space
    energy difference) is accumulated across the current symbol.  At the
    boundary, the buffered values are summed (integrate-and-dump): the
    sign gives the bit, the mean |diff| gives a confidence score in
    [0, 1] — a useful signal-quality metric for later logging.
  * Whenever the hard-decision bit flips between two consecutive frames —
    a genuine data transition, since CCIR 476's 4-of-7 character code
    guarantees these occur often — the *exact* crossing point is found by
    linearly interpolating where `diff` crosses zero between those two
    frames. Real transitions only ever happen at true bit boundaries, so
    any gap between that interpolated crossing and our assumed boundary
    is a direct timing-error measurement, which nudges the phase
    accumulator (`loop_gain` controls how strongly). Default of 0.05
    reflects further hands-on tuning against real weak-signal/heavy-noise
    recordings, done after (and superseding) the 0.19 value described in
    an earlier pass, which was itself already lower than the original
    0.25. At 0.05 the loop is deliberately quite sluggish -- it barely
    reacts to any single transition's timing-error measurement -- trading
    fast reacquisition for strong resistance to noise-induced spurious
    transitions, which measured as the best tradeoff for catching a weak,
    fading signal in a heavy-noise environment even though it sits well
    outside what would normally be considered a "textbook" PLL gain
    range. This value has NOT been established as related to the
    separate phasing/lock issue seen on specific stations (that issue
    predates and is unaffected by this tuning pass) -- see project notes.

Known limitation
------------------
Because this is proportional-only, a *constant* clock-rate mismatch
between transmitter and receiver leaves a small steady-state phase error
rather than converging to exactly zero — a full Type-II loop would add an
integral term to track frequency, not just phase, at the cost of another
tunable gain and slower/more complex convergence behaviour. The demo below
deliberately stress-tests this with a simulated clock offset so the
question of "does it matter in practice" is answered with data rather than
assumption, rather than building that complexity in speculatively.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterator, List, Optional

from navtex_step1_sampling_windowing import (
    AudioSource,
    FileSource,
    NavtexConfig,
    SyntheticNavtexSource,
    Windower,
)
from navtex_step2_tone_detection import ToneDetector, ToneSample


# ---------------------------------------------------------------------------
# Output of the bit-clock recovery stage
# ---------------------------------------------------------------------------

@dataclass
class BitDecision:
    """One recovered bit, handed to Step 4."""

    bit: bool          # True = mark (1), False = space (0)
    confidence: float   # mean |diff| over the integration window, in [0, 1]
    mean_total_power: float  # mean (mark_power + space_power) over the window --
                              # NOT gain-normalized, so only meaningful as a
                              # relative measure within one recording (or across
                              # recordings made at the same receiver gain/AGC
                              # setting), unlike `confidence` which is.
    timestamp: float     # seconds, of the first frame integrated for this bit
    start_sample: int    # sample index of the first frame integrated
    n_frames: int          # number of oversampled frames integrated (diagnostic)


# ---------------------------------------------------------------------------
# Bit-clock recovery (digital PLL)
# ---------------------------------------------------------------------------

class BitSync:
    """Converts an oversampled ToneSample stream into one BitDecision per
    symbol, self-aligning to the transmitter's bit clock. See module
    docstring for the algorithm.
    """

    def __init__(self, config: NavtexConfig, loop_gain: float = 0.05):
        # loop_gain default: see module docstring -- 0.05 is a deliberate,
        # hands-on-tuned value for weak-signal/heavy-noise performance, not
        # a leftover from an earlier pass. Do not "fix" this back toward
        # 0.19/0.25 without re-validating against real weak-signal
        # recordings first.
        self.config = config
        self.loop_gain = loop_gain
        self._phase = 0.0
        self._phase_step = config.hop_size / config.samples_per_symbol
        self._buffer: List[ToneSample] = []
        self._prev_bit: Optional[bool] = None
        self._prev_diff: Optional[float] = None

        # Diagnostics, handy while tuning loop_gain against real recordings.
        self.transitions_seen = 0
        self.symbols_emitted = 0

    def push(self, sample: ToneSample) -> Iterator[BitDecision]:
        transition = self._prev_bit is not None and sample.bit != self._prev_bit
        self._buffer.append(sample)

        phase_before_step = self._phase
        self._phase += self._phase_step

        if transition:
            self.transitions_seen += 1
            if self._prev_diff is not None and self._prev_diff != sample.diff:
                # Fraction of the way from the previous frame to this one
                # where the (roughly linear, see Step 2's transition trace)
                # diff curve crosses zero.
                t = self._prev_diff / (self._prev_diff - sample.diff)
                t = min(1.0, max(0.0, t))
            else:
                t = 0.5
            crossing_phase = phase_before_step + t * self._phase_step
            nearest_boundary = round(crossing_phase)
            error = crossing_phase - nearest_boundary
            self._phase -= self.loop_gain * error

        if self._phase >= 1.0:
            total = sum(s.diff for s in self._buffer)
            decided_bit = total >= 0
            confidence = sum(abs(s.diff) for s in self._buffer) / len(self._buffer)
            mean_power = sum(s.total_power for s in self._buffer) / len(self._buffer)
            first = self._buffer[0]
            yield BitDecision(
                bit=decided_bit,
                confidence=confidence,
                mean_total_power=mean_power,
                timestamp=first.timestamp,
                start_sample=first.start_sample,
                n_frames=len(self._buffer),
            )
            self.symbols_emitted += 1
            self._buffer = []
            self._phase -= 1.0

        self._prev_bit = sample.bit
        self._prev_diff = sample.diff

    def process_stream(self, samples: Iterator[ToneSample]) -> Iterator[BitDecision]:
        for s in samples:
            yield from self.push(s)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def _run_synthetic_case(label: str, *, duration_s: float, start_offset_samples: int,
                         symbol_period_override: Optional[float], snr_db: float = 20.0,
                         lock_acquisition_symbols: int = 15) -> None:
    """Runs Steps 1-3 against a synthetic case with a deliberately
    misaligned and/or drifting bit clock, and scores BitSync's output
    against the known transmitted bits — skipping the first
    `lock_acquisition_symbols` decisions to give the loop time to converge.
    """
    config = NavtexConfig()
    source = SyntheticNavtexSource(
        config, duration_s=duration_s, snr_db=snr_db, seed=7,
        start_offset_samples=start_offset_samples,
        symbol_period_override=symbol_period_override,
    )
    windower = Windower(config)
    detector = ToneDetector(config)
    bitsync = BitSync(config)

    decisions = list(bitsync.process_stream(
        detector.process_stream(windower.frames(source))
    ))

    true_bits = source.bits
    scored = decisions[lock_acquisition_symbols:]
    n = min(len(scored), len(true_bits) - lock_acquisition_symbols)
    correct = sum(
        1 for i in range(n)
        if bool(scored[i].bit) == bool(true_bits[lock_acquisition_symbols + i])
    )
    mean_conf = sum(d.confidence for d in scored[:n]) / n if n else 0.0

    print(f"{label}: emitted={len(decisions)} true_bits={len(true_bits)} "
          f"scored={n} accuracy={100 * correct / n if n else 0:.1f}% "
          f"mean_confidence={mean_conf:.3f} transitions={bitsync.transitions_seen}")


def _demo():
    config = NavtexConfig()
    print("Config:", config.describe())

    if len(sys.argv) > 1:
        # Real file: no ground truth available, just report what came out.
        source: AudioSource = FileSource(config, sys.argv[1])
        print(f"Source: WAV file {sys.argv[1]!r}\n")
        windower = Windower(config)
        detector = ToneDetector(config)
        bitsync = BitSync(config)

        decisions = list(bitsync.process_stream(
            detector.process_stream(windower.frames(source))
        ))
        bits_str = "".join(str(int(d.bit)) for d in decisions)
        print(f"Recovered {len(decisions)} bits, {bitsync.transitions_seen} transitions seen.")
        print("First 200 bits:", bits_str[:200])
        mean_conf = sum(d.confidence for d in decisions) / len(decisions) if decisions else 0.0
        print(f"Mean confidence: {mean_conf:.3f}")
        return

    print("\nSynthetic self-tests (each stresses a different aspect of "
          "timing recovery):\n")

    # Baseline: perfectly aligned, no drift -- sanity check only.
    _run_synthetic_case(
        "Aligned, no drift          ",
        duration_s=3.0, start_offset_samples=0, symbol_period_override=None,
    )

    # Arbitrary start offset -- the loop must self-align from an unknown
    # phase, exactly as it will have to on any real recording.
    _run_synthetic_case(
        "Unaligned start (+217 smp) ",
        duration_s=3.0, start_offset_samples=217, symbol_period_override=None,
    )

    # Modest clock drift: +500 ppm (480 -> 480.24 samples/symbol). Far
    # larger than a real sound card's crystal tolerance, used to see how
    # the proportional-only loop copes under stress.
    _run_synthetic_case(
        "500 ppm clock drift        ",
        duration_s=5.0, start_offset_samples=137,
        symbol_period_override=config.samples_per_symbol * 1.0005,
    )

    # More aggressive drift: +2000 ppm.
    _run_synthetic_case(
        "2000 ppm clock drift       ",
        duration_s=5.0, start_offset_samples=137,
        symbol_period_override=config.samples_per_symbol * 1.002,
    )

    # Noisy AND unaligned AND drifting, combined.
    _run_synthetic_case(
        "Combined: noisy+offset+drift",
        duration_s=5.0, start_offset_samples=311,
        symbol_period_override=config.samples_per_symbol * 1.0008,
        snr_db=5.0,
    )


if __name__ == "__main__":
    _demo()
