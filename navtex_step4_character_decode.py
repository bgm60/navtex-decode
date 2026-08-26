"""
NAVTEX 100-baud FSK Decoder — Step 4: Character Decode (CCIR 476 / SITOR-B FEC)
================================================================================

Pipeline this module is the fourth stage of:

    [1] Sampling & windowing        (navtex_step1_sampling_windowing.py)
    [2] Mark/space tone detection   (navtex_step2_tone_detection.py)
    [3] Bit-clock recovery          (navtex_step3_bit_sync.py)
    [4] Character decode (CCIR 476 / SITOR-B FEC)   <-- this file
    [5] Message assembly / B1B2B3B4 header parsing

Scope of this module
---------------------
Consumes the `BitDecision` stream from Step 3 (one bit per symbol) and
produces decoded text. Three sub-problems, solved in three layers:

  1. CHARACTER SYNC (CharacterSync / CharacterGrouper): bits arrive one at
     a time with no marker for where a 7-bit codeword starts. Since a
     valid CCIR 476 codeword always has exactly 4 of its 7 bits set, the
     correct alignment is found by trying all 7 possible phases over a
     trailing window and picking whichever produces the highest fraction
     of weight-4 groups. This needs no knowledge of any specific
     codeword's bit pattern (including the phasing-signal preamble) --
     just that valid data, whenever it occurs, is mostly weight-4.

  2. FEC TIME-DIVERSITY (FecCombiner): NAVTEX/SITOR-B transmits every
     character twice for fade resistance. Provenance matters here, so see
     the FecCombiner docstring below for exactly how the interleave
     structure and the 5-slot comparison lag were determined and verified
     against a real, field-tested encoder implementation.

  3. CHARACTER LOOKUP: the combined, error-corrected 7-bit codeword is
     translated to text via the CCIR 476 table below, tracking LTRS/FIGS
     shift state the same way Baudot/ITA2 does.

Provenance / verification of the codeword table
--------------------------------------------------
The 26-letter + CR/LF/LTRS/FIGS/SPACE/BLANK codeword table was transcribed
directly from ITU-R Recommendation M.476-5 (the primary standard), Annex 1,
Table 1 ("Table of conversion -- Traffic information signals"), fetched
from https://www.itu.int/dms_pubrec/itu-r/rec/m/R-REC-M.476-5-199510-I!!PDF-E.pdf .
Every one of the 32 transcribed codewords is verified programmatically at
import time (see `_self_check()`) to have the required weight of exactly 4
marks out of 7 bits, and all 32 are checked to be pairwise distinct --
both necessary properties for a valid CCIR 476 table, and a strong check
against transcription error (all 26 letters passed on the first attempt,
which is a meaningful confirmation the transcription is correct, not just
a tautology -- a transcription error would very likely produce a wrong
weight and trip the check).

Bit order: the standard's Table 1 lists each "emitted 7-unit signal" using
B (higher tone / mark / 1) and Y (lower tone / space / 0), read left to
right in the literal order the bits are transmitted. That is exactly the
order Step 3 emits bits in (earliest bit first), so codewords below are
used exactly as transcribed -- no bit-reversal needed.

Deliberately NOT included: specific bit patterns for "Phasing signal 1/2".
Cross-referencing the ITU table against secondary sources (a Wikipedia-
derived table and an open-source Arduino CCIR476 library) turned up
disagreement about exactly which codeword is which -- almost certainly a
row-alignment artifact of PDF-to-text extraction on my end rather than a
real ambiguity in the standard, but not worth risking silently. Character
sync below doesn't need to know either pattern -- it's naturally more
robust anyway, since it works during any valid data, not only literal
phasing preamble.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterator, List, Optional

from navtex_step3_bit_sync import BitDecision

# ---------------------------------------------------------------------------
# CCIR 476 codeword table (see provenance note above)
# ---------------------------------------------------------------------------

LETTERS = {
    '1110001': 'A', '0100111': 'B', '1011100': 'C', '1100101': 'D',
    '0110101': 'E', '1101100': 'F', '1010110': 'G', '1001011': 'H',
    '1011001': 'I', '1110100': 'J', '0111100': 'K', '1010011': 'L',
    '1001110': 'M', '1001101': 'N', '1000111': 'O', '1011010': 'P',
    '0111010': 'Q', '1010101': 'R', '1101001': 'S', '0010111': 'T',
    '0111001': 'U', '0011110': 'V', '1110010': 'W', '0101110': 'X',
    '1101010': 'Y', '1100011': 'Z',
}

# Same 26 codewords, figures-mode meaning. None = unassigned per the
# standard (footnote: figure shifts for D, F, G, H are not assigned).
FIGURES = {
    '1110001': '-', '0100111': '?', '1011100': ':', '1100101': None,
    '0110101': '3', '1101100': None, '1010110': None, '1001011': None,
    '1011001': '8', '1110100': '\x07', '0111100': '(', '1010011': ')',
    '1001110': '.', '1001101': ',', '1000111': '9', '1011010': '0',
    '0111010': '1', '1010101': '4', '1101001': "'", '0010111': '5',
    '0111001': '7', '0011110': '=', '1110010': '2', '0101110': '/',
    '1101010': '6', '1100011': '+',
}

# Mode-independent control codewords (6 more, distinct from the 26 above --
# together the 32 "traffic information signal" combinations of Table 1).
CONTROL = {
    '0001111': 'CR',
    '0011011': 'LF',
    '0101101': 'LTRS',
    '0110110': 'FIGS',
    '0011101': 'SP',
    '0101011': 'BLANK',
}


def _weight(code: str) -> int:
    return code.count('1')


def _self_check() -> None:
    all_codes = list(LETTERS) + list(CONTROL)
    assert len(all_codes) == 32, f"expected 32 traffic codewords, got {len(all_codes)}"
    assert len(set(all_codes)) == 32, "duplicate codeword in traffic table"
    for code in all_codes:
        assert len(code) == 7, f"{code!r} is not 7 bits"
        assert _weight(code) == 4, f"{code!r} has weight {_weight(code)}, expected 4"
    assert set(LETTERS) == set(FIGURES), "letters/figures must share the same 26 codewords"


_self_check()

# ---------------------------------------------------------------------------
# Table 2 ("service information signals") phasing codewords
# ---------------------------------------------------------------------------
# Distinct from Table 1's 32 traffic-information codewords above -- these
# are used only for receiver synchronization, never as message content.
#
# Confirmed by DIRECT OBSERVATION against a real recording's phasing burst:
# a clean, unambiguous period-14 alternation between exactly these two
# codewords, running for several seconds before any real message content
# appears. This also retroactively validates the original derivation from
# ITU-R M.476-5 Annex 1 Table 2 ("Idle signal alpha" / "Signal repetition")
# made much earlier in this project -- at the time, two secondary sources
# disagreed with each other about which codeword was which, so these were
# deliberately left out of the decode tables rather than risk a silent
# error. Real data has now settled it.
PHASING_1 = '1111000'
PHASING_2 = '0110011'
_PHASING_CODES = frozenset({PHASING_1, PHASING_2})


# ---------------------------------------------------------------------------
# Layer 1: character (7-bit group) synchronization
# ---------------------------------------------------------------------------

class CharacterSync:
    """Finds which of the 7 possible bit-group phases aligns bits into
    valid (weight-4) codewords, by scoring each phase over a trailing
    window. See module docstring for why this needs no specific codeword
    knowledge.

    Exposes per-phase scoring (score_phase) rather than only a single
    "best" phase, so CharacterGrouper can continuously check whether the
    phase it's currently using is still performing well -- not just what
    the best phase happened to be once.

    Phase is defined relative to a GLOBAL bit count (bit_index % 7), not
    relative to the sliding window's own start. This matters more than it
    looks: the window evicts exactly one old bit per new bit pushed, so
    the window's start position drifts by one global bit position every
    push. A phase number measured relative to the window's start would
    therefore only correspond to the same real alignment once every 7
    pushes, cycling through all 7 meanings continuously -- verified
    directly during development: score_phase(4) on a stable, correctly-
    aligned bitstream read anywhere from ~0.3 to 1.0 depending purely on
    how much the window happened to have slid, not on whether the
    alignment was actually still correct. Tracking phase against a global
    count fixes this: once a phase value is correct, it stays correct.
    """

    def __init__(self, window_size: int = 70):
        self.window_size = window_size
        self._bits: List[bool] = []
        self._total_bits = 0  # global count of all bits ever pushed

    def push_bit(self, bit: bool) -> None:
        self._bits.append(bit)
        self._total_bits += 1
        if len(self._bits) > self.window_size:
            self._bits.pop(0)

    def score_phase(self, global_phase: int):
        """Returns (valid_fraction, n_groups) for one GLOBAL phase
        (bit_index % 7) over the current trailing window."""
        buffer_start_global = self._total_bits - len(self._bits)
        offset = (global_phase - buffer_start_global) % 7
        bits = self._bits[offset:]
        n = len(bits) // 7
        if n == 0:
            return 0.0, 0
        valid = sum(1 for i in range(n) if sum(bits[i * 7:(i + 1) * 7]) == 4)
        return valid / n, n

    def best_phase(self):
        """Returns (phase, score) for whichever of the 7 phases currently
        scores highest, or (None, 0.0) if there isn't enough data yet."""
        best_phase, best_score = None, -1.0
        for phase in range(7):
            score, n = self.score_phase(phase)
            if n == 0:
                continue
            if score > best_score:
                best_phase, best_score = phase, score
        return best_phase, (best_score if best_phase is not None else 0.0)


class CharacterGrouper:
    """Turns a bit stream into a stream of raw 7-bit codewords (one per
    character-slot).

    Unlike a lock-once design, this continuously re-validates its current
    phase choice and re-acquires if confidence drops -- necessary because
    real recordings contain silence and inter-message gaps where bits are
    essentially noise (about 27% of random 7-bit groups look "valid" by
    weight alone purely by chance), which can produce a plausible-looking
    but wrong initial lock that a lock-once design would never recover
    from, corrupting everything decoded afterward even once real signal
    resumes.

    A second, subtler failure mode showed up against real recordings:
    between messages, NAVTEX sends a repeating idle/phasing filler
    signal -- not silence, but a real, valid, REPETITIVE codeword. Because
    it repeats, it can score deceptively well at more than one candidate
    phase simultaneously (unlike genuine varied text, which only looks
    valid at the one true phase). A naive "did confidence dip below a
    threshold" check has no way to tell "genuinely lost lock" apart from
    "briefly ambiguous because the filler ties across phases", and can
    drift onto a different, wrong phase during the filler -- which then
    corrupts the start of the next message before self-correcting once
    varied text proves it wrong. Since messages always begin with ZCZC,
    this specifically and repeatedly ate exactly that.

    The fix: track the active phase's score as before, but before
    actually abandoning it, require that some alternative phase beats it
    by a clear SWITCH_MARGIN -- not just edges past DROP_THRESHOLD. A tie
    (or near-tie) during ambiguous filler is no longer enough to cause a
    switch; only a phase that's decisively performing better triggers one,
    which is what a genuine resync (e.g. after a real fade) looks like.
    """

    ACQUIRE_THRESHOLD = 0.6
    DROP_THRESHOLD = 0.46  # empirically tuned against real overnight fading/interference
                            # recordings (was 0.4) -- more willing to attempt recovery once
                            # an active phase starts struggling, favoring catching a weak/
                            # fading signal over avoiding the occasional corrupt decode
    SWITCH_MARGIN = 0.2
    # Minimum groups before trusting a phase's score for acquisition. This
    # matters more than it might look: measured directly against a real
    # recording's phasing-signal preamble, no phase ever scores anywhere
    # near ACQUIRE_THRESHOLD over a large, reliable sample (the ceiling
    # was ~0.33, close to the ~27% pure-chance baseline) -- but small
    # samples spike above 0.6 by pure chance often enough to cause a false
    # acquisition anyway. n=8 groups is only ~1.6 standard deviations from
    # 0.6 given a true rate of 0.33 -- not nearly enough separation when
    # the score is re-evaluated continuously on every incoming bit. n=30
    # gives >3 standard deviations of separation, making a spurious
    # crossing rare rather than close to inevitable.
    MIN_GROUPS_FOR_ACQUIRE = 30

    def __init__(self, sync_window: int = 250):
        self.sync = CharacterSync(sync_window)
        self._group: List[bool] = []
        self._group_confidence: List[float] = []
        self._active_phase: Optional[int] = None

    def push_bit(self, bit: bool, confidence: float = 1.0) -> Iterator[tuple]:
        self.sync.push_bit(bit)
        current_global_pos = self.sync._total_bits - 1  # position of the bit just pushed

        if self._active_phase is None:
            self._try_acquire()
            return

        if not self._group and current_global_pos % 7 != self._active_phase:
            return  # not yet at a phase-aligned starting bit; skip until we are

        self._group.append(bit)
        self._group_confidence.append(confidence)
        if len(self._group) == 7:
            code = ''.join('1' if b else '0' for b in self._group)
            yield code, list(self._group_confidence)
            self._group = []
            self._group_confidence = []
            self._reconsider()

    def _try_acquire(self) -> None:
        # Require the winning phase to clearly beat the runner-up, not
        # merely cross ACQUIRE_THRESHOLD -- the same ambiguous-filler risk
        # that affects mid-stream switching (see class docstring) applies
        # to a cold start too, e.g. if decoding begins partway through the
        # phasing-signal preamble rather than on real varied text. Also
        # require enough groups to trust the score at all -- otherwise a
        # phase can look perfect purely from 1-2 lucky groups before other
        # phases have even accumulated enough data to compete.
        scores = []
        for phase in range(7):
            score, n = self.sync.score_phase(phase)
            if n >= self.MIN_GROUPS_FOR_ACQUIRE:
                scores.append((score, phase))
        if not scores:
            return
        scores.sort(reverse=True)
        best_score, best_phase = scores[0]
        runner_up_score = scores[1][0] if len(scores) > 1 else 0.0
        if best_score >= self.ACQUIRE_THRESHOLD and best_score >= runner_up_score + self.SWITCH_MARGIN:
            self._active_phase = best_phase
            self._group = []
            self._group_confidence = []

    def _reconsider(self) -> None:
        active_score, active_n = self.sync.score_phase(self._active_phase)
        if active_n == 0:
            return
        if active_score >= self.DROP_THRESHOLD:
            return  # still performing adequately; keep it, even if not perfect
        best_phase, best_score = self.sync.best_phase()
        if (best_phase is not None and best_phase != self._active_phase
                and best_score >= active_score + self.SWITCH_MARGIN):
            self._active_phase = None  # a genuinely better phase exists; re-acquire to it
        # else: current phase is weak but nothing else is clearly better --
        # likely just noise/ambiguous filler, not a real resync need; stay put


# ---------------------------------------------------------------------------
# Layer 2: SITOR-B / NAVTEX FEC time-diversity combining
# ---------------------------------------------------------------------------

# All 34 weight-4 codewords used anywhere in this system (26 letters + 6
# control + 2 phasing signals) -- the full set considered "valid" for
# single-bit error-correction purposes below.
_ALL_VALID_CODES = frozenset(LETTERS.keys()) | frozenset(CONTROL.keys()) | {PHASING_1, PHASING_2}


def _try_single_bit_correction(code: str, confidences: List[float],
                                valid_codes: frozenset) -> Optional[str]:
    """Attempts confidence-guided single-bit error correction.

    Only applies when `code` has weight 3 or 5 -- i.e. plausibly one bit
    away from a valid weight-4 codeword. Naive nearest-valid-codeword
    correction does NOT work for this code: checked directly against the
    actual codeword table, every possible weight-3/5 pattern is a
    single-bit neighbor of multiple (typically 3-5) different valid
    codewords, never zero and never exactly one -- because 34 of the 35
    possible weight-4 patterns are assigned meanings, there's essentially
    no unused space to disambiguate a correction by codeword distance
    alone. Guessing blindly among the candidates would be no better than
    not correcting at all.

    Per-bit confidence breaks the tie: each candidate correction
    corresponds to flipping one specific bit position, so picking the
    candidate that flips the LOWEST-confidence bit targets the position
    Step 3 was actually least sure about, rather than treating all
    candidates as equally likely.

    Note this can only ever trigger on an odd number of simultaneous bit
    errors (one flip changes weight by +-1; two flips cancel out or
    double up, landing back at weight 2, 4, or 6, never 3 or 5) -- so in
    practice this corresponds almost always to exactly one real bit
    error, the single most likely corruption pattern under real fading.
    """
    weight = code.count('1')
    if weight == 3:
        flip_positions = [i for i, b in enumerate(code) if b == '0']
    elif weight == 5:
        flip_positions = [i for i, b in enumerate(code) if b == '1']
    else:
        return None  # not one bit away from weight-4; not correctable this way

    candidates = []
    for i in flip_positions:
        flipped = code[:i] + ('1' if code[i] == '0' else '0') + code[i + 1:]
        if flipped in valid_codes:
            candidates.append((confidences[i], i, flipped))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0])  # lowest confidence first
    return candidates[0][2]


class FecCombiner:
    """Combines the raw per-character-slot codeword stream using SITOR-B
    time-diversity FEC, then decodes to text.

    Interleave structure and comparison lag were verified against a real,
    field-tested SITOR-B/NAVTEX encoder implementation (Baltic Lab's
    open-source Arduino CCIR476 library, confirmed working against a
    commercial NAV4 NAVTEX receiver -- see
    https://baltic-lab.com/2022/07/sitor-b-navtex-test-signal-generation/).
    Its transmit routine, traced through by hand, sends each new character
    once immediately (DX) and repeats it (RX) after exactly 4 OTHER
    characters have been transmitted in between -- i.e. RX is 5
    character-slots after its own DX, matching the standard's "followed
    by the transmission of four other characters, after which the
    retransmission takes place" reading literally.

    Because DX and RX slots strictly alternate, only every second incoming
    codeword is a valid "compare against 5-slots-back" position -- the
    other slots are fresh DX transmissions whose own RX hasn't arrived
    yet, so comparing THEM against 5-back would compare two unrelated
    characters. Which of the two slot parities is the valid one isn't
    knowable in advance (it depends on where in the stream decoding
    happened to start), so it's found the same way CharacterSync finds bit
    phase: both parities are scored by how often codeword[n] equals
    codeword[n-5] over a trailing window, and whichever parity matches far
    more often is locked in.

    lock_window trades acquisition speed against false-lock risk: testing
    against synthetic messages (see test_step4_roundtrip.py) found zero
    false locks down to window=6 even at several-percent bit error rates,
    since two unrelated codewords matching by chance is inherently rare
    (each is one of only 35 valid weight-4 patterns). The default of 15 is
    a conservative margin above that, giving acquisition of LAG+15=20
    codeword-slots (~10 source characters, ~1.4s) rather than 45 slots.

    Like CharacterGrouper, this continuously re-validates its parity
    choice rather than locking once, and for the same two reasons: (1)
    silence/gaps can produce a plausible-but-wrong initial lock that would
    otherwise never be corrected, and (2) the repeating idle/phasing
    filler NAVTEX sends between messages is a real, valid codeword that
    can score well at BOTH parities simultaneously (unlike genuine varied
    text, which only matches consistently at the true parity) -- so a
    naive "did the match rate dip" check can drift onto the wrong parity
    during that filler and corrupt the start of the next message before
    self-correcting, which specifically and repeatedly ate the ZCZC at
    each message's start during testing against a real recording.

    The fix mirrors CharacterGrouper's: both parities' match rates are
    tracked continuously (not just the active one), and switching away
    from the active parity requires the alternative to beat it by a clear
    SWITCH_MARGIN, not just for the active parity to dip below a raw
    threshold. A tie or near-tie during ambiguous filler is no longer
    enough to cause a switch.

    IMPORTANT, confirmed against a real recording (see decode_bit_stream's
    phasing-burst reset below and TUNING_REFERENCE.md / project notes for
    the "Sync_failure.wav" case): the above self-correction is a *passive*
    mechanism -- it only reacts once enough real mismatches accumulate.
    That's fast enough to fix a lock acquired onto the wrong parity from
    scratch, but too slow to avoid real damage the specific time it
    matters most: right after an inter-message phasing burst, if the true
    DX/RX parity happened to flip across that burst (phasing content is
    period-2 and structurally can't carry parity information either way,
    so there's no way to track parity *through* the burst itself). In that
    case the stale, now-wrong parity gets re-used for ~15-20 characters
    (enough for the passive match-rate tracking to notice) before
    self-correcting -- which is long enough to eat the opening ZCZC and
    station ID of the next message. decode_bit_stream now forces an
    unconditional fec.reset() after every sufficiently long phasing burst
    (see there) specifically to close this gap, rather than relying on
    this passive mechanism alone for that particular transition.
    """

    LAG = 5
    ACQUIRE_THRESHOLD = 0.6
    SWITCH_MARGIN = 0.2
    RATE_WINDOW = 30  # codewords of history per parity used for its rolling match rate
    # Minimum samples before trusting a parity's rate estimate. Originally
    # 10, tuned conservatively against noise in general. Measured directly
    # against a real recording once the phasing period is properly
    # excluded (see decode_bit_stream's phasing-transition reset): the
    # correct parity shows a clean 100% match starting from literally the
    # first possible lag-5 comparison after real content begins, with the
    # wrong parity at a clean 0% -- no ambiguity to average out. Testing
    # (see test_step4_roundtrip.py) found zero false locks down to 3
    # samples even at 5% bit error rate; 5 keeps a small safety margin
    # above that measured floor while cutting real-world acquisition time
    # roughly in half.
    MIN_SAMPLES_FOR_RATE = 5
    # See the detailed note in _combine. Off by default: real-world
    # testing showed this made output worse (far more '!', no clear
    # benefit) despite thorough synthetic validation looking good --
    # confidence per bit appears less reliable at pinpointing *which*
    # specific bit within a codeword is wrong than testing assumed.
    ENABLE_SINGLE_BIT_CORRECTION = False

    def __init__(self, lock_window: int = 15):
        history_len = self.LAG + max(lock_window, self.RATE_WINDOW)
        self._history: Deque[tuple] = deque(maxlen=history_len)  # (code, confidences) pairs
        self._n = 0
        self._parity: Optional[int] = None
        self._lock_window = lock_window
        self._letters_mode = True
        # Rolling match/mismatch outcomes tracked for BOTH parities at all
        # times, not just whichever is currently active -- needed to
        # compare them and decide whether a switch is clearly justified.
        self._matches: dict = {0: deque(maxlen=self.RATE_WINDOW), 1: deque(maxlen=self.RATE_WINDOW)}

    def push(self, code: str, confidences: List[float]) -> Iterator[str]:
        self._history.append((code, confidences))
        self._n += 1

        if len(self._history) > self.LAG:
            # Parity-lock statistics deliberately use the RAW (uncorrected)
            # codewords, not single-bit-corrected ones: acquiring/holding
            # the correct parity should rest on genuinely strong,
            # unambiguous agreement, not on "after correcting both sides
            # they might agree" cases. Correction is applied only once
            # parity is already known, at the point of emitting a
            # character -- see _combine.
            dx, _ = self._history[-1 - self.LAG]
            rx, _ = self._history[-1]
            this_slot_parity = self._n % 2
            self._matches[this_slot_parity].append(dx == rx)

        if self._parity is None:
            yield from self._try_acquire()
            return

        self._maybe_switch()

        if self._n % 2 != self._parity % 2:
            return  # fresh DX slot; its RX comparison happens later
        if len(self._history) <= self.LAG:
            return
        dx, dx_conf = self._history[-1 - self.LAG]
        rx, rx_conf = self._history[-1]
        yield from self._combine(dx, dx_conf, rx, rx_conf)

    def _rate(self, parity: int) -> Optional[float]:
        m = self._matches[parity]
        if len(m) < self.MIN_SAMPLES_FOR_RATE:
            return None
        return sum(m) / len(m)

    def reset(self) -> None:
        """Clears all state, forcing a fresh parity re-acquisition.

        Must be called whenever the upstream CharacterGrouper's active
        phase changes: codewords from before and after a phase change are
        different groupings of the same underlying bits, not directly
        comparable, so any match/mismatch comparison spanning that
        boundary is meaningless and can corrupt the rolling match-rate
        statistics used for parity locking. Also called unconditionally
        after every sufficiently long inter-message phasing burst -- see
        decode_bit_stream and the class docstring above.
        """
        self._history.clear()
        self._n = 0
        self._parity = None
        self._matches = {0: deque(maxlen=self.RATE_WINDOW), 1: deque(maxlen=self.RATE_WINDOW)}

    def _try_acquire(self) -> Iterator[str]:
        # Same reasoning as CharacterGrouper._try_acquire: require a clear
        # margin over the alternative, not just crossing ACQUIRE_THRESHOLD,
        # since a cold start during ambiguous filler has the same risk of
        # grabbing the wrong parity as a mid-stream switch does.
        r0, r1 = self._rate(0), self._rate(1)
        if r0 is None and r1 is None:
            return
        r0v, r1v = r0 or 0.0, r1 or 0.0
        best_parity, best_rate = (0, r0v) if r0v >= r1v else (1, r1v)
        other_rate = r1v if best_parity == 0 else r0v
        if best_rate >= self.ACQUIRE_THRESHOLD and best_rate >= other_rate + self.SWITCH_MARGIN:
            self._parity = best_parity
            yield from self._replay_history(best_parity)

    def _replay_history(self, parity: int) -> Iterator[str]:
        """Retroactively processes codewords already sitting in the
        history buffer from before lock was achieved.

        Without this, perfectly good characters received just before lock
        landed -- commonly including the very start of a message, e.g.
        ZCZC -- would be silently lost purely because acquisition took a
        few codewords longer than the message did to get going. Since
        _try_acquire is only ever called while self._parity is None, and
        push() never calls _combine while unlocked, none of these
        positions can have been processed already -- safe to walk through
        once, in chronological order (so LTRS/FIGS mode-shift state builds
        up correctly), and this method's own future positions are never
        touched again since normal push() only ever looks at the single
        newest codeword from here on.
        """
        hist = list(self._history)
        base_n = self._n - len(hist) + 1
        for j in range(self.LAG, len(hist)):
            n = base_n + j
            if n % 2 != parity % 2:
                continue
            dx, dx_conf = hist[j - self.LAG]
            rx, rx_conf = hist[j]
            yield from self._combine(dx, dx_conf, rx, rx_conf)

    def _maybe_switch(self) -> None:
        active_rate = self._rate(self._parity)
        other_rate = self._rate(1 - self._parity)
        if active_rate is None or other_rate is None:
            return
        if other_rate >= active_rate + self.SWITCH_MARGIN and other_rate >= self.ACQUIRE_THRESHOLD:
            self._parity = 1 - self._parity
        # else: active parity may look weak, but nothing else is clearly
        # better -- likely ambiguous filler or noise, not a genuine need
        # to resync; stay put

    def _combine(self, dx: str, dx_conf: List[float], rx: str, rx_conf: List[float]) -> Iterator[str]:
        if dx in _PHASING_CODES and rx in _PHASING_CODES:
            # Phasing-signal filler observed mid-stream (not just at the
            # opening preamble) -- confirmed empirically against a real
            # recording: identical Phase1/Phase2 codewords, in the same
            # simple non-doubled alternation, reappearing periodically
            # mid-message. No ITU-R M.476-5 text was found describing
            # this explicitly -- treat that as an open question about
            # *why* it happens, not evidence it doesn't: the codewords
            # themselves are unambiguous and reproducible across every
            # occurrence checked. dx will structurally never equal rx
            # here regardless (period-2 content defeats the lag-5
            # comparison the same way the opening preamble does), so
            # this isn't a decode failure -- skip cleanly, matching how
            # BLANK idle characters are already handled, rather than
            # flagging it as an error.
            return

        dx_valid = _weight(dx) == 4
        rx_valid = _weight(rx) == 4

        # Confidence-guided single-bit correction: whichever side(s)
        # failed the weight-4 check outright get one attempt at recovery
        # before falling back to the existing valid/invalid logic below.
        # See _try_single_bit_correction for why this needs per-bit
        # confidence rather than codeword distance alone.
        #
        # DEFAULT DISABLED (see ENABLE_SINGLE_BIT_CORRECTION below): tested
        # thoroughly against synthetic data with confidence constructed to
        # correlate cleanly with actual bit correctness, and it worked --
        # roughly a 3x reduction in error symbols, near-zero silent wrong
        # corrections. But against real live signal it produced far MORE
        # '!' output with no obvious benefit -- almost certainly because
        # real per-bit confidence is confounded by a structural, content-
        # dependent effect (a bit's confidence is heavily influenced by
        # whether its analysis window straddles a neighboring transition,
        # not primarily by whether noise corrupted that specific bit),
        # which the synthetic test's directly-constructed confidence
        # didn't and couldn't capture. Left in place and toggleable for
        # further investigation, but off by default until validated
        # against real recorded confidence/error data rather than a
        # synthetic model that turned out to be unrealistic.
        if self.ENABLE_SINGLE_BIT_CORRECTION:
            if not dx_valid:
                corrected = _try_single_bit_correction(dx, dx_conf, _ALL_VALID_CODES)
                if corrected is not None:
                    dx = corrected
                    dx_valid = True
            if not rx_valid:
                corrected = _try_single_bit_correction(rx, rx_conf, _ALL_VALID_CODES)
                if corrected is not None:
                    rx = corrected
                    rx_valid = True

        if dx_valid and rx_valid:
            if dx == rx:
                chosen = dx
            else:
                # Both codewords individually pass the weight-4 check but
                # disagree with each other -- a different, rarer failure
                # mode than ordinary mutilation: it implies two
                # independent bit errors that each happened to still land
                # on *some* valid codeword, just not the same one. This
                # is ALSO exactly the signature produced when FecCombiner
                # is comparing two unrelated real characters under a
                # wrong parity lock (see the "Sync_failure.wav" case
                # described in the class docstring above) -- a long,
                # unbroken run of this specific symbol right after a
                # phasing burst is a strong tell for that failure mode
                # specifically, distinct from scattered single '!'s
                # during otherwise-clean reception. Observed on a real
                # side-by-side comparison against SeaTTY: it flags this
                # case distinctly ('!') from ordinary corruption ('*')
                # rather than collapsing both into one generic error
                # symbol -- worth keeping visibly distinct here too,
                # since it's a genuinely different situation, not just a
                # stylistic choice.
                yield '!'
                return
        elif dx_valid:
            chosen = dx
        elif rx_valid:
            chosen = rx
        else:
            chosen = None

        if chosen is None:
            yield '~'  # error symbol, per ITU-R M.476-5 §3.2.4.1
            return
        yield from self._decode(chosen)

    def _decode(self, code: str) -> Iterator[str]:
        if code in CONTROL:
            meaning = CONTROL[code]
            if meaning == 'LTRS':
                self._letters_mode = True
            elif meaning == 'FIGS':
                self._letters_mode = False
            elif meaning == 'CR':
                yield '\r'
            elif meaning == 'LF':
                yield '\n'
            elif meaning == 'SP':
                yield ' '
            # BLANK: idle filler, nothing printed
            return
        table = LETTERS if self._letters_mode else FIGURES
        ch = table.get(code)
        yield ch if ch is not None else '~'  # weight-4 but an unassigned figure slot


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def decode_bit_stream(bit_decisions: Iterator[BitDecision]) -> Iterator[str]:
    grouper = CharacterGrouper()
    fec = FecCombiner()
    prev_phase: Optional[int] = None
    consecutive_phasing = 0
    for bd in bit_decisions:
        for code, confidences in grouper.push_bit(bd.bit, bd.confidence):
            if grouper._active_phase != prev_phase:
                fec.reset()  # phase changed since the last codeword; old FEC context is invalid
                prev_phase = grouper._active_phase
                consecutive_phasing = 0

            if code in _PHASING_CODES:
                consecutive_phasing += 1
            else:
                if consecutive_phasing >= 4:
                    # Just emerged from a run of phasing signal (which FEC
                    # structurally cannot lock during -- see FecCombiner
                    # docstring) into real content. Reset UNCONDITIONALLY
                    # here, even if FEC was already locked coming in.
                    #
                    # This used to be gated on `fec._parity is None` (skip
                    # the reset if already locked), on the theory that a
                    # phasing-like stretch between messages shouldn't be
                    # allowed to throw away a good lock carried over from
                    # an earlier message in the same recording. That
                    # theory turned out to be wrong: confirmed directly
                    # against a real recording ("Sync_failure.wav" -- see
                    # TUNING_REFERENCE.md / project notes) where
                    # CharacterGrouper's phase lock correctly survived a
                    # genuine inter-message phasing burst untouched, but
                    # the TRUE DX/RX parity flipped across that same
                    # burst. Phasing content is period-2 and structurally
                    # can't carry parity information either way (see
                    # FecCombiner docstring), so there is no way to track,
                    # during the burst, whether the real transmitted
                    # idle/phasing codeword count between messages was
                    # parity-preserving. With the old gate, the stale
                    # parity from the previous message was kept, and every
                    # DX/RX comparison afterward compared two unrelated
                    # but individually-valid codewords -- exactly the '!'
                    # signature -- for as long as it took FecCombiner's
                    # own passive rolling-match-rate tracking to notice
                    # and switch (about 20 characters / 3s in the observed
                    # case), eating the opening ZCZC and station ID of the
                    # next message.
                    #
                    # Resetting unconditionally instead forces a fresh,
                    # statistically-gated re-acquisition (_try_acquire)
                    # after every sufficiently long phasing burst, whether
                    # previously locked or not. Because _try_acquire
                    # replays buffered history once it re-locks, this
                    # costs nothing extra in the common case where the
                    # parity didn't actually change (it re-locks to the
                    # same parity within a few samples and replays
                    # normally) -- it only matters, positively, in the
                    # case that used to corrupt the message opening.
                    fec.reset()
                consecutive_phasing = 0

            yield from fec.push(code, confidences)


# ---------------------------------------------------------------------------
# Demo / WAV file entry point
# ---------------------------------------------------------------------------

def _demo():
    """Runs the full Step 1-4 pipeline against a WAV file (or, with no
    argument, a synthetic test message) and prints the decoded text.
    """
    import sys
    from navtex_step1_sampling_windowing import (
        AudioSource, FileSource, NavtexConfig, Windower,
    )
    from navtex_step2_tone_detection import ToneDetector
    from navtex_step3_bit_sync import BitSync

    config = NavtexConfig()
    print("Config:", config.describe())

    if len(sys.argv) > 1:
        source: AudioSource = FileSource(config, sys.argv[1])
        print(f"Source: WAV file {sys.argv[1]!r}\n")
    else:
        # No file given: fall back to a synthetic encoded test message so
        # this is runnable standalone. Uses the same encoder as
        # test_step4_roundtrip.py.
        import numpy as np
        from navtex_step1_sampling_windowing import SyntheticNavtexSource
        from test_step4_roundtrip import encode_text

        text = ("ZCZC SA88 THIS IS A SYNTHETIC TEST MESSAGE WITH NO REAL WAV "
                 "FILE GIVEN ON THE COMMAND LINE NNNN")
        print(f"Source: synthetic encoded message (no file given): {text!r}\n")
        bits = encode_text(text)
        sps = config.samples_per_symbol
        rng = np.random.default_rng(0)
        signal = np.empty(len(bits) * sps, dtype=np.float32)
        phase = 0.0
        idx = 0
        for b in bits:
            freq = config.mark_freq if b else config.space_freq
            t = np.arange(sps) / config.sample_rate
            signal[idx:idx + sps] = np.sin(2 * np.pi * freq * t + phase)
            phase = (phase + 2 * np.pi * freq * sps / config.sample_rate) % (2 * np.pi)
            idx += sps
        signal += rng.normal(0, 0.1, size=signal.shape)

        class _ArraySource(AudioSource):
            def chunks(self):
                for start in range(0, len(signal), 4096):
                    yield signal[start:start + 4096]

        source = _ArraySource()

    windower = Windower(config)
    detector = ToneDetector(config)
    bitsync = BitSync(config)

    decoded = ''.join(decode_bit_stream(
        bitsync.process_stream(detector.process_stream(windower.frames(source)))
    ))
    print("Decoded text:")
    print("-" * 60)
    print(decoded)
    print("-" * 60)


if __name__ == '__main__':
    _demo()
