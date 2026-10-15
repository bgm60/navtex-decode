# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Tests for the weak-signal lock (navtex_soft_lock.py) and the
`weak_signal_lock` profile setting that selects it.

* Configuration: the setting, its validation, and that the settings the
  new lock does not use are not checked while it is on.
* The lock itself, on synthetic signals from navtex_synth: it decodes a
  clean transmission completely, reaches signals the standard decoder
  cannot, ignores noise, follows two messages with a gap between them, and
  reports when it holds a lock.
* That switching the setting off leaves the standard decoder exactly as it
  was (the rest of the test suite covers its behaviour).

Run from the project folder:    python -m pytest test_soft_lock.py
"""

from __future__ import annotations

import dataclasses
import difflib

import numpy as np
import pytest

from navtex_bench import (ArraySource, estimate_slot_starts, front_end_decisions, positional_score,
                          positional_scores, sent_slots, visible_truth)
from navtex_config import ConfigError, Profile, load_profile
from navtex_session import (DecodeSession, SignalStrengthTracker, build_soft_lock, decode_characters,
                            weak_signal_lock_enabled)
from navtex_soft_lock import (BINS_PER_STEP, DEFAULT_Z_STAR, SoftLockDecoder, SoftLockParams,
                              _is_idle_lookalike, _slip_neighbours, comparison_scores)
from navtex_synth import Channel, apply_channel, build_transmission, random_message

PROFILE = Profile()
FS = PROFILE.sample_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def transmission(seed=3, n_lines=4):
    return build_transmission(random_message(np.random.default_rng(seed), n_lines=n_lines),
                              sample_rate=FS, mark_freq=PROFILE.mark_freq, space_freq=PROFILE.space_freq)


def recovered(events, tx, decisions, base=0):
    """Position-matched score of (character, position) events against the
    transmitted message that starts `base` decisions into the stream."""
    lead = int(round(tx.bits_start_s * 100))
    sub = decisions[base:base + len(tx.bits) // 7 * 7 + 400]
    starts = estimate_slot_starts(sub, tx, lead) + base
    out = {}
    for ch, pos in events:
        g = pos + 3
        if starts[0] <= g < starts[-1] + 7 and ch not in "~!":
            out[int(np.searchsorted(starts, g, side="right") - 1)] = ch
    return positional_score(out, visible_truth(sent_slots(tx)))


def soft_profile(**kw):
    return dataclasses.replace(PROFILE, weak_signal_lock=True, **kw)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_setting_defaults_to_off_and_the_threshold_is_the_conservative_value():
    assert PROFILE.weak_signal_lock is False
    assert DEFAULT_Z_STAR == 5.0 and SoftLockParams().z_star == DEFAULT_Z_STAR


@pytest.mark.parametrize("old", ["off", "conservative", "normal", "sensitive", 1, "true"])
def test_old_or_non_boolean_values_are_configuration_errors(old):
    with pytest.raises(ConfigError, match="weak_signal_lock"):
        dataclasses.replace(PROFILE, weak_signal_lock=old).validate()


def test_character_sync_and_fec_settings_are_not_checked_while_the_lock_is_on():
    bad = dict(sync_window=10, min_groups_for_acquire=25, rate_window=2, min_samples_for_rate=9)
    with pytest.raises(ConfigError):
        dataclasses.replace(PROFILE, **bad).validate()
    dataclasses.replace(PROFILE, weak_signal_lock=True, **bad).validate()   # no error


def test_the_setting_loads_from_a_toml_profile(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text('[a]\nmode = "live"\nweak_signal_lock = true\n[b]\nmode = "live"\n')
    assert load_profile(str(path), "a").weak_signal_lock is True
    assert load_profile(str(path), "b").weak_signal_lock is False


@pytest.mark.parametrize("old", ["off", "conservative", "normal", "sensitive"])
def test_old_toml_level_strings_are_rejected(tmp_path, old):
    path = tmp_path / "p.toml"
    path.write_text(f'[a]\nmode = "live"\nweak_signal_lock = "{old}"\n')
    with pytest.raises(ConfigError, match="weak_signal_lock.*true or false"):
        load_profile(str(path), "a")


def test_session_uses_the_lock_only_when_the_setting_is_on():
    off = DecodeSession(ArraySource(np.zeros(10)), PROFILE, sinks=[])
    on = DecodeSession(ArraySource(np.zeros(10)), soft_profile(), sinks=[])
    assert not weak_signal_lock_enabled(PROFILE) and off._soft_lock is None
    assert weak_signal_lock_enabled(soft_profile()) and on._soft_lock is not None
    assert off.sync_state == on.sync_state == "searching"


# ---------------------------------------------------------------------------
# Scoring and bookkeeping
# ---------------------------------------------------------------------------

def test_a_repeated_character_scores_high_and_unrelated_groups_score_near_zero():
    rng = np.random.default_rng(0)
    params = SoftLockParams()
    code = np.array([1, -1, 1, 1, -1, -1, 1], dtype=float)        # a weight-4 codeword
    good = comparison_scores(np.tile(code * 3, (200, 1)) + rng.normal(0, 1, (200, 7)),
                             np.tile(code * 3, (200, 1)) + rng.normal(0, 1, (200, 7)), params)
    noise = comparison_scores(rng.normal(0, 3, (4000, 7)), rng.normal(0, 3, (4000, 7)), params)
    assert good.mean() > 3 and abs(noise.mean()) < 0.1
    assert noise.std() == pytest.approx(params.score_sd, rel=0.05)


def test_slip_neighbours_move_the_phase_by_one_and_flip_parity_at_the_wrap():
    assert _slip_neighbours(2 * 3 + 0) == (2 * 4 + 0, 2 * 2 + 0)        # phase 3, parity 0
    assert _slip_neighbours(2 * 6 + 0)[0] == 2 * 0 + 1                   # +1 from 6 wraps, flips parity
    assert _slip_neighbours(2 * 0 + 1)[1] == 2 * 6 + 0                   # -1 from 0 wraps, flips parity


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def test_a_clean_transmission_is_decoded_completely_and_in_order():
    tx = transmission()
    audio = apply_channel(tx.audio, tx.sample_rate, tx.signal_power, Channel(snr_db=None, seed=1))
    decisions = front_end_decisions(audio, PROFILE)
    decoder = SoftLockDecoder()
    events = list(decoder.events(decisions))
    score = recovered(events, tx, decisions)
    assert score.hits == score.n
    positions = [pos for _, pos in events]
    assert positions == sorted(positions)
    assert decoder.state == "searching"                      # finish() releases the lock
    assert "".join(ch for ch, _ in events).replace("\r", "").count("ZCZC") >= 1


def test_it_decodes_a_signal_the_standard_decoder_cannot():
    tx = transmission(seed=5)
    audio = apply_channel(tx.audio, tx.sample_rate, tx.signal_power, Channel(snr_db=-10.0, seed=2))
    soft = positional_scores(tx, audio, soft_profile())["P"]
    standard = positional_scores(tx, audio, PROFILE)["P"]
    assert soft.hits / soft.n > 0.30
    assert standard.hits / standard.n < 0.05
    assert soft.chance / soft.chance_n < 0.05                 # not coincidence


def test_the_bench_scores_the_real_module_exactly_as_the_session_decodes():
    tx = transmission(seed=6)
    audio = apply_channel(tx.audio, tx.sample_rate, tx.signal_power, Channel(snr_db=-9.0, seed=3))
    positional_scores(tx, audio, soft_profile(), verify=True)      # raises if the text differs


def test_noise_alone_produces_no_text_and_no_lock():
    audio = apply_channel(np.zeros(8 * 60 * FS), FS, 0.005, Channel(snr_db=0.0, seed=11))
    decisions = front_end_decisions(audio, PROFILE)
    decoder = SoftLockDecoder(SoftLockParams())
    states = set()
    text = []
    for bd in decisions:
        text.extend(ch for ch, _ in decoder.push(bd.soft_value))
        states.add(decoder.state)
    text.extend(ch for ch, _ in decoder.finish())
    assert text == [] and states == {"searching"}


def test_two_messages_with_noise_between_are_both_decoded_and_the_lock_follows_them():
    rng = np.random.default_rng(4)
    snr = -8.0
    txs = [transmission(seed=21, n_lines=3), transmission(seed=22, n_lines=3)]
    power = txs[0].signal_power
    parts, bases, t, seed = [], [], 0, 100
    for tx in txs:
        gap = int(rng.uniform(25, 40) * FS) + int(rng.integers(0, 997))   # not a whole number of bits
        parts.append(apply_channel(np.zeros(gap), FS, power, Channel(snr_db=snr, seed=seed)))
        seed += 1
        t += gap
        bases.append(int(round(t / (FS / 100.0))))
        parts.append(apply_channel(tx.audio, FS, tx.signal_power, Channel(snr_db=snr, seed=seed)))
        seed += 1
        t += len(tx.audio)
    parts.append(apply_channel(np.zeros(30 * FS), FS, power, Channel(snr_db=snr, seed=seed)))
    audio = np.concatenate(parts)
    decisions = front_end_decisions(audio, PROFILE)

    decoder = SoftLockDecoder()
    events, states = [], []
    for bd in decisions:
        events.extend(decoder.push(bd.soft_value))
        states.append(decoder.state)
    events.extend(decoder.finish())
    for tx, base in zip(txs, bases):
        score = recovered(events, tx, decisions, base)
        assert score.hits / score.n > 0.6
    # locked during each message, searching again before the end of the final gap
    for tx, base in zip(txs, bases):
        mid = base + int(len(tx.bits) * 0.6)
        assert states[mid] == "data"
    assert states[-1] == "searching"
    assert states[0] == "searching"


def test_streaming_in_pieces_gives_the_same_text_as_all_at_once():
    tx = transmission(seed=8, n_lines=2)
    audio = apply_channel(tx.audio, FS, tx.signal_power, Channel(snr_db=-9.0, seed=4))
    decisions = front_end_decisions(audio, PROFILE)
    whole = list(SoftLockDecoder().events(decisions))
    decoder, pieces = SoftLockDecoder(), []
    for i in range(0, len(decisions), 37):
        for bd in decisions[i:i + 37]:
            pieces.extend(decoder.push(bd.soft_value))
    pieces.extend(decoder.finish())
    assert pieces == whole and len(whole) > 0


def test_long_streams_do_not_grow_the_history_without_bound():
    decoder = SoftLockDecoder()
    rng = np.random.default_rng(1)
    for v in rng.normal(0, 3, 40000):
        for _ in decoder.push(v):
            pass
    assert len(decoder._buf) <= 6000
    assert decoder._base + len(decoder._buf) == 40000


def test_default_profile_still_uses_the_standard_decoder():
    tx = transmission(seed=9, n_lines=2)
    audio = apply_channel(tx.audio, FS, tx.signal_power, Channel(snr_db=-5.0, seed=5))
    text = "".join(decode_characters(ArraySource(audio), PROFILE, SignalStrengthTracker()))
    assert "ZCZC" in text
    assert BINS_PER_STEP == 14
    assert build_soft_lock(soft_profile()).params.z_star == DEFAULT_Z_STAR


# ---------------------------------------------------------------------------
# Following: bit slips and a second station
# ---------------------------------------------------------------------------

def _decode(decisions, params=None):
    decoder = SoftLockDecoder(params or SoftLockParams())
    events = []
    for bd in decisions:
        events.extend(decoder.push(bd.soft_value))
    events.extend(decoder.finish())
    return events, decoder


def _text(events):
    return "".join(ch for ch, _ in events)


@pytest.mark.parametrize("snr", [0.0, -6.0])
@pytest.mark.parametrize("slip", ["drop", "duplicate"])
def test_a_one_bit_slip_inside_a_lock_is_followed(snr, slip):
    tx = transmission(seed=12, n_lines=6)
    audio = apply_channel(tx.audio, FS, tx.signal_power, Channel(snr_db=snr, seed=7))
    decisions = front_end_decisions(audio, PROFILE)
    reference = _text(_decode(decisions)[0])
    middle = int(round(tx.bits_start_s * 100)) + len(tx.bits) // 2
    slipped = list(decisions)
    if slip == "drop":
        del slipped[middle]
    else:
        slipped.insert(middle, slipped[middle])
    events, decoder = _decode(slipped)
    kept = sum(b.size for b in difflib.SequenceMatcher(None, reference, _text(events),
                                                       autojunk=False).get_matching_blocks())
    assert decoder.switches == 1
    assert kept > 0.95 * len(reference)
    off_events, off_decoder = _decode(slipped, SoftLockParams(follow=False))
    assert off_decoder.switches == 0
    off_kept = sum(b.size for b in difflib.SequenceMatcher(None, reference, _text(off_events),
                                                           autojunk=False).get_matching_blocks())
    assert off_kept < 0.7 * len(reference)             # without following the rest is lost


def _two_stations(gain_b_db, overlap_s, snr, seeds=(31, 32)):
    a = transmission(seed=seeds[0], n_lines=4)
    b = transmission(seed=seeds[1], n_lines=4)
    start_b = int(len(a.audio) - overlap_s * FS) + 503              # an odd bit phase
    n = max(len(a.audio), start_b + len(b.audio)) + 5 * FS
    x = np.zeros(n)
    x[:len(a.audio)] += a.audio
    x[start_b:start_b + len(b.audio)] += 10 ** (gain_b_db / 20) * b.audio
    audio = apply_channel(x, FS, a.signal_power, Channel(snr_db=snr, seed=1))
    return a, b, front_end_decisions(audio, PROFILE), int(round(start_b / (FS / 100.0)))


@pytest.mark.parametrize("gain_db", [0, 5, 10])
def test_a_second_station_that_is_as_strong_takes_over_when_the_first_ends(gain_db):
    a, b, decisions, base_b = _two_stations(gain_db, overlap_s=10, snr=-4.0)
    events, decoder = _decode(decisions)
    score_b = recovered(events, b, decisions, base_b)
    assert score_b.hits / score_b.n > 0.9
    assert decoder.switches >= 1 or gain_db == 10                    # a stronger B wins without needing a switch
    off_events, _ = _decode(decisions, SoftLockParams(follow=False))
    score_a = recovered(events, a, decisions, 0)
    score_a_off = recovered(off_events, a, decisions, 0)
    assert score_a.hits >= score_a_off.hits - 3                      # the first station is not sacrificed


def test_a_much_weaker_second_station_does_not_disturb_the_first():
    a, b, decisions, _ = _two_stations(-10, overlap_s=10, snr=-4.0)
    events, decoder = _decode(decisions)
    score_a = recovered(events, a, decisions, 0)
    assert score_a.hits / score_a.n > 0.95
    assert decoder.switches == 0


def test_following_makes_no_difference_to_a_single_signal_without_slips():
    tx = transmission(seed=41, n_lines=4)
    audio = apply_channel(tx.audio, FS, tx.signal_power, Channel(snr_db=-9.0, seed=8))
    decisions = front_end_decisions(audio, PROFILE)
    on, decoder = _decode(decisions)
    off, _ = _decode(decisions, SoftLockParams(follow=False))
    assert on == off and decoder.switches == 0


# ---------------------------------------------------------------------------
# Closing the open line, and runs of one letter
# ---------------------------------------------------------------------------

def test_a_run_of_identical_letters_is_dropped_but_the_text_around_it_is_kept():
    decoder = SoftLockDecoder()
    for i, ch in enumerate("AB" + "K" * 12 + "CD"):
        decoder._queue(i, ch, i)
    assert "".join(ch for _, ch, _ in decoder._pending) == "ABCD"
    keep = SoftLockDecoder(SoftLockParams(max_letter_run=0))
    for i, ch in enumerate("AB" + "K" * 12 + "CD"):
        keep._queue(i, ch, i)
    assert "".join(ch for _, ch, _ in keep._pending) == "AB" + "K" * 12 + "CD"


def test_short_runs_digits_and_spaces_are_never_dropped():
    decoder = SoftLockDecoder()
    text = "LL" + "K" * 7 + " " * 12 + "0" * 10 + "\r\n" * 6
    for i, ch in enumerate(text):
        decoder._queue(i, ch, i)
    assert "".join(ch for _, ch, _ in decoder._pending) == text


def test_the_letter_run_from_idle_signals_between_two_messages_is_dropped():
    def build(seed, pre, tail):
        msg = random_message(np.random.default_rng(seed), n_lines=2)
        return build_transmission(msg, sample_rate=FS, mark_freq=PROFILE.mark_freq,
                                  space_freq=PROFILE.space_freq, preamble_s=pre, tail_slots=tail)
    first, second = build(1, 5.0, 40), build(2, 1.0, 12)
    audio = apply_channel(np.concatenate([first.audio, second.audio]), FS, first.signal_power,
                          Channel(snr_db=6.0, seed=2))
    decisions = front_end_decisions(audio, PROFILE)
    kept, _ = _decode(decisions, SoftLockParams(max_letter_run=0, ghost_hold=0))
    dropped, _ = _decode(decisions, SoftLockParams(ghost_hold=0))
    import re
    assert re.search(r"([A-Z])\1{7,}", _text(kept))                  # the synthetic idle gap does produce one
    assert not re.search(r"([A-Z])\1{7,}", _text(dropped))
    assert "ZCZC FD67" in _text(dropped)                              # the second message still arrives


def test_end_line_starts_a_new_timestamped_line_for_the_next_text():
    import datetime
    from navtex_outputs import LineAssembler, OutputSink

    class Recorder(OutputSink):
        def __init__(self):
            self.events = []
        def line_start(self, timestamp, strength):
            self.events.append(("start", timestamp.second, strength))
        def write(self, text):
            self.events.append(("write", text))
        def line_end(self):
            self.events.append(("end",))

    ticks = iter(range(100))
    rec = Recorder()
    assembler = LineAssembler([rec], strength=lambda: 7,
                              clock=lambda: datetime.datetime(2026, 1, 1, 0, 0, next(ticks)))
    assembler.end_line()                                   # nothing in progress: no event
    assert rec.events == []
    assembler.feed("AB")
    assembler.end_line()
    assembler.feed("CD")
    assert rec.events == [("start", 0, 7), ("write", "A"), ("write", "B"),
                          ("write", "\r"), ("end",), ("write", "\n"),
                          ("start", 1, 7), ("write", "C"), ("write", "D")]


def test_the_session_ends_the_line_when_the_lock_is_released():
    class StubLock:
        releases = 0
        state = "searching"
        def decode(self, _bit_decisions):
            yield "A"
            yield "B"
            self.releases += 1
            yield "C"

    from navtex_outputs import OutputSink
    lines, current = [], []

    class Collect(OutputSink):
        def line_start(self, timestamp, strength):
            current.clear()
        def write(self, text):
            current.append(text)
        def line_end(self):
            lines.append("".join(current).replace("\r", "").replace("\n", ""))

    session = DecodeSession(ArraySource(np.zeros(FS)), soft_profile(), sinks=[Collect()])
    session._soft_lock = StubLock()
    session.run()
    assert lines == ["AB", "C"]


# ---------------------------------------------------------------------------
# Idle look-alikes
# ---------------------------------------------------------------------------

def test_idle_signals_read_out_of_step_are_recognised_and_real_pairs_are_not():
    from navtex_step4_character_decode import CONTROL, LETTERS
    assert _is_idle_lookalike("1110000", "1100111")           # seen in a synthetic idle gap
    assert _is_idle_lookalike("0111100", "1011001")           # the 'K' of the overnight logs
    assert not _is_idle_lookalike("1111000", "0110011")       # genuine idle: handled elsewhere
    codes = list(LETTERS) + list(CONTROL)
    for code in codes:                                        # two copies of a real character agree
        assert not _is_idle_lookalike(code, code)
        for bit in range(7):                                  # ... even with a bit error in one copy
            flipped = code[:bit] + ("1" if code[bit] == "0" else "0") + code[bit + 1:]
            assert not _is_idle_lookalike(code, flipped)
            assert not _is_idle_lookalike(flipped, code)


def test_look_alike_pairs_are_dropped_only_once_three_come_in_a_row():
    decoder = SoftLockDecoder()
    decoder._pending.append((0, "Q", 100))                    # earlier real text must survive
    for i in range(2):
        decoder._queue(10 + i, "Z", i)                         # what the first two pairs would have produced
        assert not decoder._skip_idle_lookalike("1110000", "1100111", i)
    assert [c for _, c, _ in decoder._pending] == ["Q", "Z", "Z"]
    assert decoder._skip_idle_lookalike("1110000", "1100111", 2)       # the third: dropped, and the two before go
    assert [c for _, c, _ in decoder._pending] == ["Q"]
    assert decoder._skip_idle_lookalike("1110000", "1100111", 3)       # and so are the rest of the run
    assert not decoder._skip_idle_lookalike("1001101", "1001101", 4)   # a real character ends the run
    assert not decoder._skip_idle_lookalike("1110000", "1100111", 5)   # a new run starts counting again


def test_the_idle_gap_between_two_messages_no_longer_prints_the_look_alike_letters():
    def build(seed, pre, tail):
        msg = random_message(np.random.default_rng(seed), n_lines=2)
        return build_transmission(msg, sample_rate=FS, mark_freq=PROFILE.mark_freq,
                                  space_freq=PROFILE.space_freq, preamble_s=pre, tail_slots=tail)
    first, second = build(1, 5.0, 40), build(2, 1.0, 12)
    audio = apply_channel(np.concatenate([first.audio, second.audio]), FS, first.signal_power,
                          Channel(snr_db=6.0, seed=2))
    decisions = front_end_decisions(audio, PROFILE)
    off = _text(_decode(decisions, SoftLockParams(max_letter_run=0, ghost_hold=0))[0])
    on = _text(_decode(decisions, SoftLockParams(max_letter_run=0))[0])
    assert "ZZZZZZ" in off and "ZZZZ" not in on
    assert "ZCZC FD67" in on
