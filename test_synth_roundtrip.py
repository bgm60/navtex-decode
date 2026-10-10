# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Tests for the synthetic-signal harness (navtex_synth.py, navtex_bench.py).

Two kinds of test live here:

* Checks on the harness itself: the encoder and interleave, the scorer,
  the noise calibration, the fading envelope, and agreement between the
  decoder's raw bit error rate and the theoretical curve. If one of these
  fails, the benchmark numbers cannot be trusted.
* Checks on the position-matched scorer (visible_truth, the tagged copy of
  the decoder pipeline, the forced-synchronisation oracle, crossing).
* Decoder regression checks: clean and moderately noisy transmissions must
  decode perfectly. These are deliberately generous, so they catch breakage
  rather than small changes in sensitivity (the sweep reports those).

Run from the project folder:    python -m pytest test_synth_roundtrip.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.signal import welch

from navtex_bench import (CHANCE_OFFSETS, TrialSpec, align_score, crossing, estimate_slot_starts,
                          front_end_decisions, measure_raw_ber, positional_score, positional_scores,
                          run_trial, sent_slots, visible_truth)
from navtex_config import Profile
from navtex_synth import (
    FIGS,
    IDLE_DX,
    IDLE_RX,
    LTRS,
    SENDABLE,
    STANDARD_MESSAGES,
    Channel,
    _rayleigh_envelope,
    apply_channel,
    build_transmission,
    ebn0_db,
    encode_text,
    interleave,
    random_message,
    theory_raw_ber,
)

PROFILE = Profile()


# ---------------------------------------------------------------------------
# Encoder and interleave
# ---------------------------------------------------------------------------

def test_encode_inserts_shifts_only_when_the_character_set_changes():
    letters = encode_text("AB")
    figures = encode_text("A1 2B")
    assert letters[0] == LTRS and len(letters) == 3
    assert FIGS in figures and figures.count(FIGS) == 1 and figures.count(LTRS) == 2


def test_encode_rejects_characters_with_no_code():
    with pytest.raises(ValueError):
        encode_text("a")          # lower case has no CCIR 476 code


def test_standard_and_random_messages_are_sendable():
    messages = list(STANDARD_MESSAGES) + [random_message(np.random.default_rng(s)) for s in range(5)]
    for message in messages:
        assert set(message) <= SENDABLE


def test_interleave_repeats_each_dx_slot_five_slots_later():
    slots = interleave(encode_text(STANDARD_MESSAGES[0]), preamble_dx_slots=4, tail_dx_slots=12)
    assert all(len(s) == 7 and s.count("1") == 4 for s in slots)
    for t in range(5, len(slots), 2):             # every RX slot
        assert slots[t] == slots[t - 5] or (slots[t - 5] == IDLE_DX and slots[t] == IDLE_RX)


def test_transmission_timing_is_consistent():
    tx = build_transmission(STANDARD_MESSAGES[0])
    assert 0 < tx.message_start_s < tx.message_end_s < tx.duration_s
    assert len(tx.bits) % 7 == 0
    assert math.isclose(tx.bits_start_s, 2.0)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def test_score_ignores_junk_before_and_after():
    score = align_score("ZCZC GA86", "~~Q!ZCZC GA86~~~~")
    assert score.errors == 0 and score.correct == 9


def test_score_classifies_each_kind_of_error():
    assert align_score("ABCDE", "ABXDE").silent == 1
    assert align_score("ABCDE", "AB~DE").flagged == 1
    assert align_score("ABCDE", "ABDE").lost == 1
    extra = align_score("ABCDE", "ABCXDE")
    assert extra.silent == 1 and extra.lost == 0
    assert align_score("ABCDE", "").lost == 5 and align_score("ABCDE", "").cer == 1.0


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("snr_db", [-10.0, 0.0])
def test_noise_level_matches_requested_snr(snr_db):
    power = 0.005
    noise = apply_channel(np.zeros(48000 * 30), 48000, power, Channel(snr_db=snr_db, seed=3))
    f, psd = welch(noise.astype(np.float64), fs=48000, nperseg=8192)
    band = (f >= 300) & (f <= 2800)                       # a 2500 Hz slice of the passband
    measured_db = 10 * math.log10(power / np.trapezoid(psd[band], f[band]))
    assert abs(measured_db - snr_db) < 0.5


def test_fading_envelope_has_unit_mean_power():
    envelope = _rayleigh_envelope(300 * 8000, 8000, 1.0, np.random.default_rng(5))
    assert abs(float(np.mean(envelope ** 2)) - 1.0) < 0.15


def test_theory_matches_known_values():
    assert math.isclose(ebn0_db(0.0), 13.979, abs_tol=0.01)
    assert math.isclose(theory_raw_ber(-8.0), 0.0690, abs_tol=0.001)


# ---------------------------------------------------------------------------
# Raw bit errors against theory
# ---------------------------------------------------------------------------

def test_front_end_bit_errors_track_theory_within_a_dB():
    """The detector cannot beat an ideal one, and should be within about a
    dB of it. This validates the SNR calibration and the bit alignment."""
    errors = bits = 0
    for seed in (11, 12, 13):
        tx = build_transmission(STANDARD_MESSAGES[1])
        audio = apply_channel(tx.audio, tx.sample_rate, tx.signal_power,
                              Channel(snr_db=-6.0, seed=seed))
        e, n = measure_raw_ber(tx, audio, PROFILE)
        errors, bits = errors + e, bits + n
    measured = errors / bits
    effective_ebn0 = -2.0 * math.log(2.0 * measured)
    loss_db = ebn0_db(-6.0) - 10 * math.log10(effective_ebn0)
    assert -0.3 < loss_db < 1.5, f"front end is {loss_db:.2f} dB from ideal"


# ---------------------------------------------------------------------------
# Decoder regression
# ---------------------------------------------------------------------------

def _spec(message: str, snr_db, seed: int = 1, **kwargs) -> TrialSpec:
    return TrialSpec(profile=PROFILE, message=message, snr_db=snr_db, seed=seed, **kwargs)


@pytest.mark.parametrize("index", range(len(STANDARD_MESSAGES)))
def test_noiseless_round_trip_is_exact(index):
    result = run_trial(_spec(STANDARD_MESSAGES[index], None))
    assert result.exact and result.header_ok
    assert 0.7 < result.lock_s < 1.3        # about 0.85 s: four characters plus the repeat delay


@pytest.mark.parametrize("seed", range(3))
def test_round_trip_with_random_text_is_exact(seed):
    message = random_message(np.random.default_rng(100 + seed), n_lines=6)
    assert run_trial(_spec(message, None, seed)).exact


@pytest.mark.parametrize("ppm", [-500.0, 500.0])
def test_tolerates_transmitter_clock_error(ppm):
    assert run_trial(_spec(STANDARD_MESSAGES[0], 0.0, clock_ppm=ppm)).exact


@pytest.mark.parametrize("seed", range(3))
def test_moderate_noise_decodes_exactly(seed):
    assert run_trial(_spec(STANDARD_MESSAGES[2], -2.0, seed)).exact


def test_nothing_decodes_from_noise_alone_with_a_valid_message_header():
    """Pure noise must not produce a message header (false lock check)."""
    n = 48000 * 60
    noise = apply_channel(np.zeros(n), 48000, 0.005, Channel(snr_db=0.0, seed=9))
    from navtex_bench import run_decoder
    assert "ZCZC" not in run_decoder(noise, PROFILE).text


# ---------------------------------------------------------------------------
# Position-matched scoring
# ---------------------------------------------------------------------------

def _tx_and_audio(message: str, snr_db, seed: int = 1):
    tx = build_transmission(message)
    audio = apply_channel(tx.audio, tx.sample_rate, tx.signal_power, Channel(snr_db=snr_db, seed=seed))
    return tx, audio


@pytest.mark.parametrize("seed", range(3))
def test_visible_truth_is_every_non_whitespace_character_in_order(seed):
    message = (STANDARD_MESSAGES[seed] if seed < len(STANDARD_MESSAGES)
               else random_message(np.random.default_rng(seed)))
    visible = visible_truth(sent_slots(build_transmission(message)))
    assert "".join(c for _, c in visible) == "".join(message.split())
    assert all(s % 2 == 1 for s, _ in visible)                    # always an RX slot
    assert [s for s, _ in visible] == sorted({s for s, _ in visible})


def test_slot_starts_are_regular_on_a_clean_signal():
    tx, audio = _tx_and_audio(STANDARD_MESSAGES[0], None)
    lead_bits = int(round(tx.bits_start_s * 100))
    starts = estimate_slot_starts(front_end_decisions(audio, PROFILE), tx, lead_bits)
    assert len(starts) == len(tx.bits) // 7
    assert set(np.diff(starts).tolist()) == {7}
    assert abs(int(starts[0]) - lead_bits) <= 6


def test_positional_score_counts_hits_header_and_chance():
    visible = [(5 + 2 * i, ch) for i, ch in enumerate("ZCZCGA86AB")]
    out = {5: "Z", 7: "C", 9: "Z", 11: "X",                       # three right, one wrong
           25: "Z"}                                               # a coincidence, 20 slots after slot 5
    score = positional_score(out, visible)
    assert (score.hits, score.n) == (3, 10)
    assert (score.hdr, score.hdr_n) == (3, 8)
    assert score.chance == 1 and score.chance_n == 10 * len(CHANCE_OFFSETS)
    assert math.isclose(score.recovery, 0.3)


def test_crossing_interpolates_and_reports_the_edges():
    snrs = [-10.0, -8.0, -6.0]
    assert crossing(snrs, [0.0, 20.0, 100.0], 10.0) == pytest.approx(-9.0)
    assert crossing(snrs, [0.0, 20.0, 100.0], 60.0) == pytest.approx(-7.0)
    assert crossing(snrs, [30.0, 40.0, 50.0], 10.0) == -math.inf     # crossing is below the range
    assert crossing(snrs, [0.0, 1.0, 2.0], 50.0) is None            # never reached


@pytest.mark.parametrize("snr_db, seed", [(None, 1), (-6.0, 2), (-7.5, 3), (-8.5, 4)])
def test_tagged_pipeline_reproduces_the_production_decoder(snr_db, seed):
    """The scorer replays a copy of decode_bit_stream_soft; if the decoder
    changes and the copy does not, this fails (verify=True raises)."""
    tx, audio = _tx_and_audio(STANDARD_MESSAGES[1], snr_db, seed)
    positional_scores(tx, audio, PROFILE, verify=True)


def test_noiseless_recovery_is_complete_for_every_variant():
    tx, audio = _tx_and_audio(STANDARD_MESSAGES[0], None)
    scores = positional_scores(tx, audio, PROFILE, oracle=True)
    for variant in ("P", "A", "B"):
        assert scores[variant].recovery == 1.0
        assert scores[variant].hdr == scores[variant].hdr_n == 8
    assert scores["P"].chance / scores["P"].chance_n < 0.1


def test_forced_synchronisation_reaches_where_the_production_lock_cannot():
    """At -10 dB the production DX/RX parity lock fails but the soft
    combiner can still read the text when told the parity (the harness's
    headroom finding). This pins that the oracle really is a ceiling."""
    tx, audio = _tx_and_audio(STANDARD_MESSAGES[1], -10.0, seed=5)
    scores = positional_scores(tx, audio, PROFILE, oracle=True)
    assert scores["P"].recovery < 0.05
    assert scores["B"].recovery > 0.25
    assert scores["B"].recovery >= scores["A"].recovery >= scores["P"].recovery - 0.02


def test_header_recovery_is_reported_on_noise_only_audio_as_zero():
    n = 48000 * 40
    noise = apply_channel(np.zeros(n), 48000, 0.005, Channel(snr_db=0.0, seed=3))
    tx = build_transmission(STANDARD_MESSAGES[0])
    audio = np.zeros(len(tx.audio), dtype=np.float32)
    audio[:] = noise[:len(audio)]
    scores = positional_scores(tx, audio, PROFILE)
    assert scores["P"].hits <= 3 and scores["P"].hdr == 0
