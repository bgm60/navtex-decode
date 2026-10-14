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

7. Following. A lock can be lost without the signal going away: a bit
   slipped in the bit-clock recovery after a burst of noise or a deep fade,
   or a second station took over. While locked, every hypothesis keeps
   being scored over a short `switch_window`. If another alignment is
   clearly ahead of the locked one (above a threshold, by a margin, for
   `switch_hold` comparisons in a row) the lock moves to it. The point at
   which the signal changed is found by choosing the split that best
   explains the stored scores; characters decoded under the old alignment
   after that point are dropped and decoding continues under the new one.
   A one-bit-slip neighbour needs less evidence than an unrelated
   alignment, and two stations of similar strength do not make the lock
   flip between them, because the margin is not met while both score well.
   As a last resort the lock is released if its own alignment has scored
   low for a long time.

8. Phasing runs. The phasing signals sent before a message are a 7-bit
   pattern repeated, which the lock can read one bit out of step, as a
   long run of one letter (typically K). Real text never has a run of
   `max_letter_run` identical letters, so such a run is dropped.

9. Idle look-alikes. The same effect usually shows as a mixture of a few
   letters and unreadable characters, not a single letter. A group pair
   is an idle look-alike when its two groups are each within one bit of
   the same one-bit-or-more rotation of the two different phasing codes.
   Two copies of a real character agree, so text cannot look like that.
   `ghost_hold` of them in a row are dropped, with the ones just before.

Units: one *comparison* is two character slots, 14 bit decisions, 0.14 s.

Not included, deliberately: any guard against a steady carrier (judged
unlikely in real use).
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
    follow: bool = True               # move the lock to a clearly better alignment
    switch_window: int = 12           # comparisons used to compare alignments
    slip_z: float = 4.0               # challenger z needed: one-bit-slip neighbour
    slip_margin: float = 2.0          # ... and its lead over the locked alignment
    switch_z: float = 5.0             # challenger z needed: unrelated alignment
    switch_margin: float = 3.0        # ... and its lead
    switch_hold: int = 3              # consecutive comparisons the rule must hold
    own_release_hold: int = 25        # release if the locked alignment alone stays low
    max_letter_run: int = 8           # drop runs of this many identical letters (0 = keep them)
    ghost_hold: int = 3               # idle-phasing look-alike pairs in a row before they are dropped (0 = off)

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


_PHASING_A, _PHASING_B = "1111000", "0110011"          # the two phasing signals


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _rotate(code: str, k: int) -> str:
    return code[k:] + code[:k]


# (earlier group, repeat group) targets of an idle signal read k bits out of step
_GHOST_TARGETS = [(_rotate(x, k), _rotate(y, k)) for k in range(1, 7)
                  for x, y in ((_PHASING_A, _PHASING_B), (_PHASING_B, _PHASING_A))]


def _is_idle_lookalike(dx: str, rx: str) -> bool:
    """True when the two groups are each within one bit of an idle pair
    read out of step (see item 9 of the module docstring)."""
    return any(_hamming(dx, a) <= 1 and _hamming(rx, b) <= 1 for a, b in _GHOST_TARGETS)


_PHI = np.repeat(np.arange(7), 2)            # bit phase of each hypothesis
_PAR = np.tile(np.arange(2), 7)              # parity of the repeat (RX) group
_LAST = 55                                   # decisions after 14*bin by which all 14 entries exist
SPLIT_OFFSET = 35                            # decisions: a comparison bin looks 5 slots ahead
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
        self._own_low_run = 0
        self._challenge_run = 0
        self._dwell = 0              # comparisons since the lock last started or moved
        self.switches = 0            # times the lock has moved to another alignment
        self.releases = 0            # times a lock has been released
        self._run_char = ""          # the letter in the current run of identical letters
        self._run_len = 0
        self._dropping = False
        self._ghost_run = 0          # idle look-alike pairs in a row
        self._ghost_keys: List[int] = []
        self._ghost_dropping = False
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
        self._own_low_run = 0
        self._challenge_run = 0
        self._dwell = 0
        self._reset_run()
        self._reset_ghost()
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
                if self._skip_idle_lookalike(dx_code, rx_code, start):
                    self._t += 1
                    continue
                for ch in self._fec._combine(dx_code, dx_soft, rx_code, rx_soft):
                    self._queue(start + 7, ch, start)
            self._t += 1

    def _reset_run(self) -> None:
        self._run_char, self._run_len, self._dropping = "", 0, False

    def _reset_ghost(self) -> None:
        self._ghost_run, self._ghost_keys, self._ghost_dropping = 0, [], False

    def _skip_idle_lookalike(self, dx: str, rx: str, key: int) -> bool:
        """Tracks idle look-alike pairs (item 9 above). Returns True for a pair
        that is to be dropped without decoding. The first `ghost_hold - 1` of a
        run are decoded as usual, because they cannot yet be told from text;
        when the run reaches `ghost_hold` the characters they produced, which
        are still held back, are removed too."""
        hold = self.params.ghost_hold
        if hold <= 0:
            return False
        if dx in (_PHASING_A, _PHASING_B) and rx in (_PHASING_A, _PHASING_B):
            return False                                  # genuine idle: handled by the combiner
        if not _is_idle_lookalike(dx, rx):
            self._reset_ghost()
            return False
        self._ghost_run += 1
        if self._ghost_run < hold:
            self._ghost_keys.append(key)
            return False
        if not self._ghost_dropping:
            self._ghost_dropping = True
            earlier = set(self._ghost_keys)
            self._pending = deque(item for item in self._pending if item[2] not in earlier)
        return True

    def _queue(self, end: int, ch: str, pos: int) -> None:
        """Adds a decoded character to the held-back queue, dropping runs
        of `max_letter_run` or more identical letters (see item 8 above).
        Characters are held for 15 comparisons, longer than the run limit,
        so the start of a run is still in the queue when it is recognised."""
        limit = self.params.max_letter_run
        if limit <= 0 or not ch.isalpha():
            self._reset_run()
            self._pending.append((end, ch, pos))
            return
        if ch == self._run_char:
            self._run_len += 1
        else:
            self._run_char, self._run_len, self._dropping = ch, 1, False
        if self._dropping:
            return
        if self._run_len >= limit:
            self._dropping = True
            for _ in range(self._run_len - 1):
                if self._pending and self._pending[-1][1] == ch:
                    self._pending.pop()
            return
        self._pending.append((end, ch, pos))

    def _maybe_release(self, b: int) -> Iterator[Tuple[str, int]]:
        p = self.params
        if p.follow:
            self._maybe_switch(b)
        limit = BINS_PER_STEP * (b - p.output_delay) + BINS_PER_STEP
        recent = np.sum(list(self._hist)[-p.release_window:], axis=0)
        z = recent / (p.score_sd * math.sqrt(p.release_window))
        best = max(z[self._h], z[self._neighbours[0]], z[self._neighbours[1]])
        self._low_run = self._low_run + 1 if best < p.release_z else 0
        self._own_low_run = self._own_low_run + 1 if z[self._h] < p.release_z else 0
        while self._pending and self._pending[0][0] <= limit:
            _, ch, pos = self._pending.popleft()
            yield ch, pos
        if self._low_run >= p.release_hold or (p.follow and self._own_low_run >= p.own_release_hold):
            self._pending.clear()                     # what is left is the stray tail
            self._locked = False
            self.releases += 1
            self._prev_end = max(self._prev_end, limit)
            self._hist.clear()                        # start again from empty

    def _maybe_switch(self, b: int) -> None:
        """Moves the lock to another alignment that is clearly better."""
        p = self.params
        s = p.switch_window
        self._dwell += 1
        if len(self._hist) < s or self._dwell < s:
            self._challenge_run = 0
            return
        z = np.sum(list(self._hist)[-s:], axis=0) / (p.score_sd * math.sqrt(s))
        c = int(np.argmax(z))
        if c == self._h:
            self._challenge_run = 0
            return
        near = c in self._neighbours
        need_z, need_margin = (p.slip_z, p.slip_margin) if near else (p.switch_z, p.switch_margin)
        if z[c] >= need_z and z[c] - z[self._h] >= need_margin:
            self._challenge_run += 1
        else:
            self._challenge_run = 0
        if self._challenge_run >= p.switch_hold:
            self._switch(c, b, near)

    def _switch(self, c: int, b: int, near: bool) -> None:
        """Re-aligns to hypothesis c from the point where the stored scores
        say the signal changed from the old alignment to the new one."""
        p = self.params
        hist = np.array(list(self._hist))                     # (m, 14), oldest first
        m = len(hist)
        old = np.concatenate(([0.0], np.cumsum(hist[:, self._h])))
        new = np.concatenate(([0.0], np.cumsum(hist[:, c])))
        k = int(np.argmax(old + (new[-1] - new)))             # comparisons that still belong to the old alignment
        cut = BINS_PER_STEP * (b - m + 1 + k) + SPLIT_OFFSET
        cut = max(cut, self._base)
        kept = deque(item for item in self._pending if item[0] <= cut)
        self._pending = kept
        self._h = c
        self._neighbours = _slip_neighbours(c)
        if not near:
            self._fec = SoftFecCombiner()
        phi = int(_PHI[c])
        self._i0 = max(0, -(-(cut - phi) // 7))
        self._p0 = phi + 7 * self._i0
        while self._p0 < self._base:
            self._i0 += 1
            self._p0 += 7
        self._t = 0
        self._recent.clear()
        self._reset_run()
        self._reset_ghost()
        self._low_run = 0
        self._own_low_run = 0
        self._challenge_run = 0
        self._dwell = 0
        self.switches += 1
        self._advance_groups()
