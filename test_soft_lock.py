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

import numpy as np
import pytest

from navtex_bench import (ArraySource, estimate_slot_starts, front_end_decisions, positional_score,
                          positional_scores, sent_slots, visible_truth)
from navtex_config import (WEAK_SIGNAL_LOCK_CHOICES, ConfigError, Profile, load_profile)
from navtex_session import (DecodeSession, SignalStrengthTracker, build_soft_lock, decode_characters,
                            weak_signal_lock_enabled)
from navtex_soft_lock import (BINS_PER_STEP, CHOICES, LEVELS, SoftLockDecoder, SoftLockParams,
                              _slip_neighbours, comparison_scores)
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


def soft_profile(level="normal", **kw):
    return dataclasses.replace(PROFILE, weak_signal_lock=level, **kw)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_setting_defaults_to_off_and_the_choices_match_the_module():
    assert PROFILE.weak_signal_lock == "off"
    assert tuple(WEAK_SIGNAL_LOCK_CHOICES) == tuple(CHOICES)
    assert set(LEVELS) == set(CHOICES) - {"off"}
    assert LEVELS["conservative"] > LEVELS["normal"] > LEVELS["sensitive"]


def test_an_unknown_choice_is_a_configuration_error():
    with pytest.raises(ConfigError, match="weak_signal_lock"):
        dataclasses.replace(PROFILE, weak_signal_lock="strong").validate()


def test_character_sync_and_fec_settings_are_not_checked_while_the_lock_is_on():
    bad = dict(sync_window=10, min_groups_for_acquire=25, rate_window=2, min_samples_for_rate=9)
    with pytest.raises(ConfigError):
        dataclasses.replace(PROFILE, **bad).validate()
    dataclasses.replace(PROFILE, weak_signal_lock="normal", **bad).validate()   # no error


def test_the_setting_loads_from_a_toml_profile(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text('[a]\nmode = "live"\nweak_signal_lock = "sensitive"\n[b]\nmode = "live"\n')
    assert load_profile(str(path), "a").weak_signal_lock == "sensitive"
    assert load_profile(str(path), "b").weak_signal_lock == "off"


def test_levels_map_to_thresholds_and_unknown_levels_are_rejected():
    assert SoftLockParams.for_level("normal").z_star == LEVELS["normal"]
    with pytest.raises(ValueError):
        SoftLockParams.for_level("off")


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
    decoder = SoftLockDecoder(SoftLockParams.for_level("conservative"))
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
    assert build_soft_lock(soft_profile("sensitive")).params.z_star == LEVELS["sensitive"]
