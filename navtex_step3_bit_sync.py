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

from dataclasses import dataclass
from typing import List, Optional

from navtex_step1_sampling_windowing import NavtexConfig
from navtex_step2_tone_detection import ToneSample


# ---------------------------------------------------------------------------
# Output of the bit-clock recovery stage
# ---------------------------------------------------------------------------

@dataclass
class BitDecision:
    """One recovered bit, handed to Step 4."""

    bit: bool          # True = mark (1), False = space (0)
    confidence: float   # mean |diff| over the integration window, in [0, 1]


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
