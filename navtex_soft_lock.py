# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
NAVTEX 100-baud FSK Decoder — Soft-Evidence Lock (weak-signal lock)
=====================================================================

An alternative to the hard-decision character-alignment and DX/RX parity
locks (CharacterGrouper and the parity half of FecCombiner), selected by
the profile setting `weak_signal_lock`. It finds the alignment and parity
from the *graded* evidence in the bit decisions instead of from whole
codewords matching exactly, so it locks on signals whose raw bit error
rate is far too high for the hard-decision lock. On synthetic signals it
reaches 10% character recovery about 5 dB lower in SNR than the old lock.
Everything upstream (windowing, tone detection, SoftBitSync) and the
combining rule (SoftFecCombiner._combine) are reused unchanged.

How it works
------------
SITOR-B sends every character twice: first (DX), then again five
character slots later (RX). Comparing each 7-bit group with the group
five slots earlier therefore gives a score that is high when the two are
copies of one character and about zero otherwise. Two things are unknown:
the bit phase (which of 7 positions starts a character) and the DX/RX
parity (which slots are the repeats). That is 7 x 2 = 14 hypotheses.

1. Score. Each completed 7-bit group is compared with the one five
   groups earlier, for every bit phase, using the soft values. The score
   is the sum of two standardised terms: the cosine similarity of the two
   soft vectors, and the correlation of their sum with the best of the 32
   valid traffic codewords. Both terms use only the signs and relative
   sizes of the soft values, so the score does not depend on audio gain.
   The standardising constants are the noise-only mean and standard
   deviation of each term, measured on synthetic noise.
2. Window. Each hypothesis keeps the sum of its last `window` scores,
   turned into a z-score (sum / (score_sd x sqrt(window))). On noise a
   z-score is about N(0, 1); on a genuine signal the true hypothesis
   climbs well above it.
3. Lock. When the best z reaches `z_star` and leads the runner-up by at
   least `gap`, lock to that phase and parity. `z_star` is the one
   sensitivity setting: it was calibrated on 24 hours of noise alone.
4. Replay. The soft history is kept, so on lock the message is decoded
   retroactively from the start of the detection window (minus a margin),
   then continues live. Lock latency therefore costs no text.
5. Release. The lock is dropped when the locked hypothesis, or either of
   its two one-bit-slip neighbours, scores below `release_z` over a
   `release_window` for `release_hold` consecutive comparisons. After a
   release the detector starts again from empty, so the tail of the old
   message cannot immediately re-lock.
6. Output delay. Characters are held back by `output_delay` comparisons
   (about 2.5 s from the end of the character) so the stray characters
   that decode after a message ends can be discarded when the release
   decision comes.

Units: one *comparison* is two character slots, 14 bit decisions, 0.14 s.

Not included, deliberately: tracking of bit-clock slips inside a lock
(it made false slip decisions on synthetic signals, which rarely slip) and
any guard against a steady carrier (judged unlikely in real use).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from navtex_soft_fec_combine import SoftFecCombiner
from navtex_step4_character_decode import CONTROL, LETTERS

# The 32 codewords that carry traffic (letters and control), as +-1 vectors.
_CODES32 = list(LETTERS) + list(CONTROL)
_SIGNS32 = np.array([[1.0 if c == "1" else -1.0 for c in code] for code in _CODES32])
_EPS = 1e-12

BINS_PER_STEP = 14          # decisions per comparison (two 7-bit slots)
SECONDS_PER_COMPARISON = BINS_PER_STEP / 100.0

# weak_signal_lock setting -> z_star. Measured false locks per hour of noise
# alone (24 h): 3.5 -> 1.8, 4.0 -> 0.3, 5.0 -> none.
LEVELS = {"conservative": 5.0, "normal": 4.0, "sensitive": 3.5}
CHOICES = ("off",) + tuple(LEVELS)


@dataclass(frozen=True)
class SoftLockParams:
    """Internal constants. Changing `window`, `gap` or the null statistics
    means z_star must be recalibrated, which is why none of these are
    profile settings."""
    z_star: float = LEVELS["normal"]
    window: int = 45                  # comparisons in the detection window
    gap: float = 1.0                  # z lead over the runner-up
    score_sd: float = 1.4168          # sd of the score on noise alone
    cosine_null: Tuple[float, float] = (0.0026, 0.3793)       # (mean, sd) on noise
    correlation_null: Tuple[float, float] = (0.715, 0.1284)   # (mean, sd) on noise
    release_window: int = 30          # comparisons
    release_z: float = 1.5
    release_hold: int = 5             # consecutive low comparisons
    output_delay: int = 15            # comparisons (see module docstring)
    replay_margin: int = 10           # comparisons before the window start

    @classmethod
    def for_level(cls, level: str) -> "SoftLockParams":
        if level not in LEVELS:
            raise ValueError(f"unknown weak_signal_lock level {level!r}; "
                             f"expected one of {', '.join(CHOICES)}")
        return cls(z_star=LEVELS[level])


def comparison_scores(x: np.ndarray, y: np.ndarray, params: SoftLockParams) -> np.ndarray:
    """Score for each row pair: x and y are (n, 7) soft values of a group
    and the group five slots later."""
    both = x + y
    cosine = (x * y).sum(1) / (np.sqrt((x * x).sum(1) * (y * y).sum(1)) + _EPS)
    corr = (both @ _SIGNS32.T).max(1) / (math.sqrt(7.0) * np.sqrt((both * both).sum(1)) + _EPS)
    return ((cosine - params.cosine_null[0]) / params.cosine_null[1]
            + (corr - params.correlation_null[0]) / params.correlation_null[1])


def _slip_neighbours(h: int) -> Tuple[int, int]:
    """Hypotheses reached from h = phase*2 + parity by a +-1 bit slip."""
    phi, par = divmod(h, 2)
    out = []
    for d in (1, -1):
        p, q = phi + d, par
        if p == 7:
            p, q = 0, 1 - par
        elif p == -1:
            p, q = 6, 1 - par
        out.append(p * 2 + q)
    return out[0], out[1]


_PHI = np.repeat(np.arange(7), 2)            # bit phase of each hypothesis
_PAR = np.tile(np.arange(2), 7)              # parity of the repeat (RX) group
_LAST = 55                                   # decisions after 14*bin by which all 14 entries exist
_BUFFER_MAX = 6000                           # decisions of soft history kept (trimmed to half)


class SoftLockDecoder:
    """Streaming soft-evidence lock and decoder. Feed it bit decisions
    (anything with a soft_value attribute, as from SoftBitSync) and it
    yields decoded characters.

        decoder = SoftLockDecoder(SoftLockParams.for_level("normal"))
        for ch in decoder.decode(bit_decisions):
            ...

    `state` is "searching" until a lock is held and "data" while locked.
    """

    def __init__(self, params: Optional[SoftLockParams] = None):
        self.params = params or SoftLockParams()
        p = self.params
        self._buf: List[float] = []
        self._base = 0               # global index of _buf[0]
        self._n = 0                  # decisions seen
        self._next_bin = 0           # next comparison bin to score
        self._hist: Deque[np.ndarray] = deque(maxlen=p.window)
        self._locked = False
        self._h = 0
        self._neighbours: Tuple[int, int] = (0, 0)
        self._low_run = 0
        self._prev_end = 0           # decisions before this belong to a finished lock
        self._fec: Optional[SoftFecCombiner] = None
        self._p0 = 0
        self._i0 = 0
        self._t = 0                  # groups completed since lock
        self._recent: Deque[Tuple[str, List[float]]] = deque(maxlen=6)
        self._pending: Deque[Tuple[int, str, int]] = deque()   # (end, char, rx group start)

    # --- public ------------------------------------------------------------

    @property
    def state(self) -> str:
        return "data" if self._locked else "searching"

    def decode(self, bit_decisions: Iterable) -> Iterator[str]:
        for ch, _pos in self.events(bit_decisions):
            yield ch

    def events(self, bit_decisions: Iterable) -> Iterator[Tuple[str, int]]:
        """Like decode, but yields (character, index of the first bit
        decision of the repeat group that produced it) so a test can tell
        which transmitted slot each character came from."""
        for bd in bit_decisions:
            yield from self.push(bd.soft_value)
        yield from self.finish()

    def push(self, soft_value: float) -> Iterator[Tuple[str, int]]:
        self._buf.append(float(soft_value))
        self._n += 1
        while self._n >= BINS_PER_STEP * self._next_bin + _LAST:
            yield from self._step(self._next_bin)
            self._next_bin += 1
        if len(self._buf) > _BUFFER_MAX:
            drop = len(self._buf) - _BUFFER_MAX // 2
            del self._buf[:drop]
            self._base += drop

    def finish(self) -> Iterator[Tuple[str, int]]:
        """Call at the end of the stream: emits what is still held back."""
        if self._locked:
            self._advance_groups()
        while self._pending:
            _, ch, pos = self._pending.popleft()
            yield ch, pos
        self._locked = False

    # --- internals ---------------------------------------------------------

    def _soft(self, a: int, b: int) -> np.ndarray:
        return np.asarray(self._buf[a - self._base:b - self._base])

    def _bin_scores(self, b: int) -> np.ndarray:
        """Scores of the 14 hypotheses for comparison bin b."""
        i = 2 * b + (_PAR == 0)                       # earlier group of the pair
        xs = _PHI + 7 * i
        ys = _PHI + 7 * (i + 5)
        off = np.arange(7)
        buf = np.asarray(self._buf)
        x = buf[(xs[:, None] + off) - self._base]
        y = buf[(ys[:, None] + off) - self._base]
        return comparison_scores(x, y, self.params)

    def _step(self, b: int) -> Iterator[Tuple[str, int]]:
        p = self.params
        self._hist.append(self._bin_scores(b))
        if self._locked:
            self._advance_groups()
            yield from self._maybe_release(b)
            return
        if len(self._hist) < p.window:
            return
        z = np.sum(self._hist, axis=0) / (p.score_sd * math.sqrt(p.window))
        order = np.argsort(z)
        top, second = z[order[-1]], z[order[-2]]
        if top >= p.z_star and top - second >= p.gap:
            self._lock(int(order[-1]), b)

    def _lock(self, h: int, b: int) -> None:
        p = self.params
        self._locked = True
        self._h = h
        self._neighbours = _slip_neighbours(h)
        self._low_run = 0
        self._fec = SoftFecCombiner()
        phi = int(_PHI[h])
        m0 = max((b - p.window + 1 - p.replay_margin) * BINS_PER_STEP,
                 self._prev_end, self._base)
        self._i0 = max(0, -(-(m0 - phi) // 7))
        self._p0 = phi + 7 * self._i0
        while self._p0 < self._base:                  # never start before the kept history
            self._i0 += 1
            self._p0 += 7
        self._t = 0
        self._recent.clear()
        self._pending.clear()
        self._advance_groups()

    def _advance_groups(self) -> None:
        """Processes every 7-bit group that has completed since the last
        call, queueing the characters from each repeat group."""
        par = int(_PAR[self._h])
        while self._p0 + 7 * self._t + 7 <= self._n:
            start = self._p0 + 7 * self._t
            soft = self._soft(start, start + 7)
            code = "".join("1" if v >= 0 else "0" for v in soft)
            self._recent.append((code, [float(v) for v in soft]))
            if self._t >= 5 and (self._i0 + self._t) % 2 == par:
                dx_code, dx_soft = self._recent[-6]
                rx_code, rx_soft = code, self._recent[-1][1]
                for ch in self._fec._combine(dx_code, dx_soft, rx_code, rx_soft):
                    self._pending.append((start + 7, ch, start))
            self._t += 1

    def _maybe_release(self, b: int) -> Iterator[Tuple[str, int]]:
        p = self.params
        limit = BINS_PER_STEP * (b - p.output_delay) + BINS_PER_STEP
        recent = np.sum(list(self._hist)[-p.release_window:], axis=0)
        z = recent / (p.score_sd * math.sqrt(p.release_window))
        best = max(z[self._h], z[self._neighbours[0]], z[self._neighbours[1]])
        self._low_run = self._low_run + 1 if best < p.release_z else 0
        while self._pending and self._pending[0][0] <= limit:
            _, ch, pos = self._pending.popleft()
            yield ch, pos
        if self._low_run >= p.release_hold:
            self._pending.clear()                     # what is left is the stray tail
            self._locked = False
            self._prev_end = max(self._prev_end, limit)
            self._hist.clear()                        # start again from empty
