"""
NAVTEX 100-baud FSK Decoder — Soft (Energy-Level) Diversity Combining
=======================================================================

Add-on, opt-in alternative to FecCombiner's hard-decision-then-compare
combining, built per the project discussion on soft diversity combining:
FecCombiner currently compares two ALREADY hard-decided 7-bit codewords
(DX and RX), which discards the graded per-bit evidence Step 2/3 actually
measured before either branch commits to a bit. This module instead adds
the two branches' signed per-bit evidence together BEFORE hard-deciding
-- classic soft/energy-level diversity combining -- which can recover
characters that neither branch could resolve alone, whenever their
individual errors land at different bit positions.

Deliberately kept as a separate module rather than editing
navtex_step3_bit_sync.py / navtex_step4_character_decode.py in place, so
the existing, already-validated decode path is completely unaffected.

What "soft value" means here
------------------------------
BitSync already computes, internally, the exact statistic it uses to
decide each bit: `total = sum(s.diff for s in self._buffer)` (see
BitSync.push). The sign of `total` gives the hard bit (`total >= 0`);
its magnitude is a genuine, gain-independent measure of how strongly the
evidence favored that decision. This is NOT the same thing as
BitDecision.confidence (`mean(|diff|)` over the same buffer): confidence
already discards sign before averaging, and per the confounding noted in
navtex_step4_character_decode.py's ENABLE_SINGLE_BIT_CORRECTION
docstring, that makes it a poor proxy for actual bit reliability. `total`
is exactly the pre-hard-decision statistic itself -- the correct thing
to carry forward and combine, not a derived summary of it.

SoftBitSync below exposes `total` as `soft_value` on a new
SoftBitDecision. It's a full copy of BitSync.push() with that one
addition (no other behaviour changed) -- kept as a copy rather than a
partial override because `total` is a local variable computed deep
inside the method, not something the original exposes a hook for.
KEEP IN SYNC BY HAND: if BitSync's PLL/integration logic in
navtex_step3_bit_sync.py ever changes, mirror the change here too, or
this module's soft values will stop corresponding to what the real
decoder is actually doing.

Design of the combiner itself
-------------------------------
SoftFecCombiner overrides only FecCombiner._combine. Everything else --
parity acquisition, the passive/active switch logic, history replay,
reset -- is inherited completely unchanged, because none of it inspects
the per-bit values it's passed; it only ever compares whole codewords
for equality. That means the carefully-tuned parity-lock behaviour
(LAG=5, ACQUIRE_THRESHOLD, SWITCH_MARGIN, MIN_SAMPLES_FOR_RATE, the
phasing-burst reset dance) is reused verbatim and cannot regress.

Inside _combine, soft combining is tried FIRST; if it doesn't resolve to
a valid codeword, this falls back to calling the ORIGINAL FecCombiner
._combine with the original hard-decided dx/rx codewords, unchanged --
i.e. this can only ever do as well as, or better than, plain hard-
decision combining, never worse. That fallback matters specifically for
frequency-selective fading or narrowband interference (see project
discussion): those produce evidence that's confidently WRONG rather than
weak/ambiguous, which naive amplitude-weighted combining could in
principle be misled by in a way plain hard-decision-then-compare isn't.
Falling back rather than trusting the soft result unconditionally is a
deliberate defensive choice, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

from navtex_step2_tone_detection import ToneSample
from navtex_step3_bit_sync import BitDecision, BitSync
from navtex_step4_character_decode import (
    CharacterGrouper,
    FecCombiner,
    _ALL_VALID_CODES,
    _PHASING_CODES,
    _try_single_bit_correction,
    _weight,
)


# ---------------------------------------------------------------------------
# Step 3 extension: expose the raw signed integration total per bit
# ---------------------------------------------------------------------------

@dataclass
class SoftBitDecision(BitDecision):
    soft_value: float = 0.0  # signed sum of diff over the integration window --
                              # the exact pre-hard-decision statistic (sign =
                              # bit, magnitude = evidence strength). See module
                              # docstring for why this differs from `confidence`.


class SoftBitSync(BitSync):
    """BitSync, but also exposes the raw signed soft value behind each bit.

    Full copy of BitSync.push() -- see module docstring's "KEEP IN SYNC BY
    HAND" note. The only change from the original is computing `total`
    (already present in the original) into `soft_value` on a
    SoftBitDecision instead of a plain BitDecision; the PLL/timing logic
    itself is untouched, byte-for-byte the same as BitSync.push().
    """

    def push(self, sample: ToneSample) -> Iterator[SoftBitDecision]:
        transition = self._prev_bit is not None and sample.bit != self._prev_bit
        self._buffer.append(sample)

        phase_before_step = self._phase
        self._phase += self._phase_step

        if transition:
            if self._prev_diff is not None and self._prev_diff != sample.diff:
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
            yield SoftBitDecision(
                bit=decided_bit,
                confidence=confidence,
                soft_value=total,
            )
            self._buffer = []
            self._phase -= 1.0

        self._prev_bit = sample.bit
        self._prev_diff = sample.diff

    def process_stream(self, samples: Iterator[ToneSample]) -> Iterator[SoftBitDecision]:
        for s in samples:
            yield from self.push(s)


# ---------------------------------------------------------------------------
# Step 4, layer 1 extension: thread soft values through character grouping
# ---------------------------------------------------------------------------

class SoftCharacterGrouper(CharacterGrouper):
    """CharacterGrouper, but groups soft_value per bit and yields it
    alongside each codeword as (code, soft_values).

    Full copy of CharacterGrouper.push_bit() -- see module docstring's
    "KEEP IN SYNC BY HAND" note. Character-sync/phase-acquisition logic
    (self.sync, _try_acquire, _reconsider) is entirely inherited and
    untouched; only the per-bit bookkeeping in push_bit is extended.
    """

    def __init__(self, sync_window: int = 250,
                 acquire_threshold: Optional[float] = None,
                 drop_threshold: Optional[float] = None,
                 switch_margin: Optional[float] = None,
                 min_groups_for_acquire: Optional[int] = None):
        super().__init__(sync_window, acquire_threshold, drop_threshold,
                          switch_margin, min_groups_for_acquire)
        self._group_soft: List[float] = []

    def push_bit(self, bit: bool, soft_value: float = 0.0) -> Iterator[tuple]:
        self.sync.push_bit(bit)
        current_global_pos = self.sync._total_bits - 1

        if self._active_phase is None:
            self._try_acquire()
            return

        if not self._group and current_global_pos % 7 != self._active_phase:
            return

        self._group.append(bit)
        self._group_soft.append(soft_value)
        if len(self._group) == 7:
            code = ''.join('1' if b else '0' for b in self._group)
            yield code, list(self._group_soft)
            self._group = []
            self._group_soft = []
            self._reconsider()


# ---------------------------------------------------------------------------
# Step 4, layer 2: soft-combining FecCombiner
# ---------------------------------------------------------------------------

class SoftFecCombiner(FecCombiner):
    """FecCombiner with soft (energy-level) diversity combining in
    _combine. Parity acquisition/switching/reset are all inherited from
    FecCombiner unchanged -- see module docstring for why that's safe.
    """

    def _combine(self, dx: str, dx_soft: List[float], rx: str, rx_soft: List[float]) -> Iterator[str]:
        if dx in _PHASING_CODES and rx in _PHASING_CODES:
            return

        # --- Primary path: soft combining ---
        combined = [d + r for d, r in zip(dx_soft, rx_soft)]
        combined_code = ''.join('1' if v >= 0 else '0' for v in combined)
        combined_weight = _weight(combined_code)

        if combined_weight == 4:
            yield from self._decode(combined_code)
            return

        if combined_weight in (3, 5):
            corrected = _try_single_bit_correction(
                combined_code, [abs(v) for v in combined], _ALL_VALID_CODES
            )
            if corrected is not None:
                yield from self._decode(corrected)
                return

        # --- Fallback: original hard-decision-then-compare logic,
        # unchanged. Ensures this combiner never does worse than plain
        # FecCombiner, even on inputs (e.g. confidently-wrong evidence
        # from selective fading/interference) where soft combining isn't
        # well-suited. dx/rx are already the correct hard decisions
        # (sign of dx_soft/rx_soft always matches them, since that's how
        # BitSync decided them in the first place), so they're passed
        # through as-is.
        yield from super()._combine(dx, dx_soft, rx, rx_soft)


# ---------------------------------------------------------------------------
# Top-level pipeline (mirrors navtex_step4_character_decode.decode_bit_stream)
# ---------------------------------------------------------------------------

def decode_bit_stream_soft(bit_decisions: Iterator[SoftBitDecision],
                            grouper: Optional["SoftCharacterGrouper"] = None,
                            fec: Optional["SoftFecCombiner"] = None,
                            phasing_burst_threshold: int = 4) -> Iterator[str]:
    """Same structure/reset logic as decode_bit_stream, using the soft
    grouper/combiner. KEEP IN SYNC BY HAND with decode_bit_stream if its
    phasing-burst reset logic ever changes.
    """
    if grouper is None:
        grouper = SoftCharacterGrouper()
    if fec is None:
        fec = SoftFecCombiner()
    prev_phase: Optional[int] = None
    consecutive_phasing = 0
    for bd in bit_decisions:
        for code, soft_values in grouper.push_bit(bd.bit, bd.soft_value):
            if grouper._active_phase != prev_phase:
                fec.reset()
                prev_phase = grouper._active_phase
                consecutive_phasing = 0

            if code in _PHASING_CODES:
                consecutive_phasing += 1
            else:
                if consecutive_phasing >= phasing_burst_threshold:
                    fec.reset()
                consecutive_phasing = 0

            yield from fec.push(code, soft_values)
