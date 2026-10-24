# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests of the Rust decoding engine (profile setting engine = "rust"):
that it gives the same text, the same line layout and the same signal
strength as the Python engine, and that its settings are checked.

The comparison tests need the navtex_rs module (maturin develop --release
in rust/navtex-py); without it they are skipped."""

import dataclasses
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

import navtex_decode
from navtex_config import ConfigError, Profile, load_profile
from navtex_outputs import OutputSink
from navtex_session import DecodeSession, build_config, build_rust_lock, decode_characters
from navtex_session import SignalStrengthTracker
from navtex_step1_sampling_windowing import FileSource
from navtex_synth import Channel, apply_channel, build_transmission, random_message

sf = pytest.importorskip("soundfile")

FS = Profile().sample_rate
REAL_RECORDINGS = sorted(Path(__file__).parent.glob("test-wavs/*.wav"))


# ---------------------------------------------------------------------------
# Settings (these do not need the Rust module)
# ---------------------------------------------------------------------------

def test_the_default_engine_is_python():
    assert Profile().engine == "python"
    Profile().validate()


def test_engine_must_be_python_or_rust():
    with pytest.raises(ConfigError, match="engine must be"):
        Profile(engine="fortran").validate()


@pytest.mark.parametrize("change", [
    {"weak_signal_lock": False},
    {"window_type": "hann"},
    {"oversample": 4},
])
def test_the_rust_engine_needs_the_settings_it_implements(change):
    with pytest.raises(ConfigError, match='engine = "rust" needs'):
        Profile(engine="rust", **change).validate()
    Profile(engine="rust").validate()


def test_engine_can_be_set_in_a_toml_profile(tmp_path):
    path = tmp_path / "navtex.toml"
    path.write_text('[p]\nmode = "live"\nengine = "rust"\n')
    assert load_profile(str(path), "p").engine == "rust"
    path.write_text('[p]\nmode = "live"\nengine = "go"\n')
    with pytest.raises(ConfigError):
        load_profile(str(path), "p")


def test_a_missing_rust_module_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "navtex_rs", None)    # makes the import fail
    with pytest.raises(ConfigError, match="navtex_rs"):
        build_rust_lock(Profile(engine="rust"))


# ---------------------------------------------------------------------------
# Python engine against Rust engine
# ---------------------------------------------------------------------------

class RecordingSink(OutputSink):
    """Records the line events a session delivers, without their times."""

    def __init__(self):
        self.lines = []          # [strength, text] per line
        self.closed = False

    def line_start(self, timestamp, strength):
        self.lines.append([strength, ""])

    def write(self, text):
        self.lines[-1][1] += text

    def close(self):
        self.closed = True


def write_wav(path, audio):
    sf.write(str(path), audio, FS)
    return path


@pytest.fixture(scope="module")
def signals(tmp_path_factory):
    """A few synthetic recordings: clean, noisy, and two messages in noise
    with a gap between them (so the lock is released between them)."""
    folder = tmp_path_factory.mktemp("rust_engine")
    p = Profile()

    def tx(seed, lines):
        return build_transmission(random_message(np.random.default_rng(seed), n_lines=lines),
                                  sample_rate=FS, mark_freq=p.mark_freq, space_freq=p.space_freq)

    clean = tx(3, 3)
    noisy = tx(4, 3)
    first, second = tx(21, 2), tx(22, 2)
    snr, power = -4.0, first.signal_power
    gap = apply_channel(np.zeros(14 * FS), FS, power, Channel(snr_db=snr, seed=100))
    tail = apply_channel(np.zeros(20 * FS), FS, power, Channel(snr_db=snr, seed=101))
    two = np.concatenate([
        apply_channel(first.audio, FS, first.signal_power, Channel(snr_db=snr, seed=102)),
        gap,
        apply_channel(second.audio, FS, second.signal_power, Channel(snr_db=snr, seed=103)),
        tail,
    ])
    return {
        "clean": write_wav(folder / "clean.wav", clean.audio),
        "noisy": write_wav(folder / "noisy.wav",
                           apply_channel(noisy.audio, FS, noisy.signal_power,
                                         Channel(snr_db=2.0, seed=7))),
        "two_messages": write_wav(folder / "two.wav", two),
    }


def run_session(wav, engine):
    profile = Profile(mode="file", wav_file=str(wav), engine=engine)
    sink = RecordingSink()
    session = DecodeSession(FileSource(build_config(profile), str(wav)), profile, [sink])
    session.run()
    return sink, session


def compare_engines(wav):
    pytest.importorskip("navtex_rs")
    python_sink, python_session = run_session(wav, "python")
    rust_sink, rust_session = run_session(wav, "rust")
    assert python_sink.lines, "the recording produced no text, so the test proves nothing"
    # The same text on the same lines...
    assert [text for _, text in rust_sink.lines] == [text for _, text in python_sink.lines]
    # ...with a signal strength that agrees. The Rust engine updates the meter a
    # chunk at a time, so it has read slightly further ahead when a line starts.
    # That is invisible while the level is steady, but the last lines start as
    # the signal dies away and the level is falling fast, so they get more room.
    pairs = list(zip(rust_sink.lines, python_sink.lines))
    for index, ((rust_strength, _), (python_strength, _)) in enumerate(pairs):
        allowed = 6 if index < len(pairs) - 2 else 25
        assert abs(rust_strength - python_strength) <= allowed, f"line {index}"
    assert abs(rust_session.tracker.level - python_session.tracker.level) <= 1
    assert rust_sink.closed and python_sink.closed
    return python_sink


@pytest.mark.parametrize("name", ["clean", "noisy", "two_messages"])
def test_same_lines_on_synthetic_signals(signals, name):
    sink = compare_engines(signals[name])
    if name == "two_messages":
        assert len(sink.lines) > 4, "two messages should make several lines"


@pytest.mark.skipif(not REAL_RECORDINGS, reason="no recordings in test-wavs")
@pytest.mark.parametrize("wav", REAL_RECORDINGS, ids=lambda p: p.stem[:20])
def test_same_lines_on_real_recordings(wav):
    compare_engines(wav)


def test_the_character_stream_matches_and_chunk_size_does_not_matter(signals):
    pytest.importorskip("navtex_rs")
    wav = signals["noisy"]
    audio, _ = sf.read(str(wav), dtype="float32")
    python_profile = Profile()
    rust_profile = dataclasses.replace(python_profile, engine="rust")

    class ArraySource:
        def __init__(self, chunk):
            self.chunk = chunk

        def chunks(self):
            for start in range(0, len(audio), self.chunk):
                yield audio[start:start + self.chunk]

    def text(profile, chunk):
        tracker = SignalStrengthTracker()
        return "".join(decode_characters(ArraySource(chunk), profile, tracker))

    reference = text(python_profile, 960)
    assert reference
    for chunk in (7, 960, 48000):
        assert text(rust_profile, chunk) == reference, f"chunk size {chunk}"


def test_the_command_line_flag_gives_the_same_database_rows(tmp_path, signals):
    pytest.importorskip("navtex_rs")
    config = tmp_path / "navtex.toml"
    config.write_text(f'[p]\nmode = "file"\nwav_file = "{signals["clean"].as_posix()}"\n'
                      f'db_file = "{(tmp_path / "log.db").as_posix()}"\nquiet = true\n')

    def rows(engine):
        db = tmp_path / f"{engine}.db"
        text = config.read_text().replace("log.db", db.name)
        path = tmp_path / f"{engine}.toml"
        path.write_text(text.replace(f'"{(tmp_path / db.name).as_posix()}"', f'"{db.as_posix()}"')
                        .replace(f'db_file = "{db.name}"', f'db_file = "{db.as_posix()}"'))
        navtex_decode.main(["p", "--config", str(path), "--engine", engine])
        with sqlite3.connect(db) as conn:
            return conn.execute("SELECT text_output FROM navtex_log").fetchall()

    python_rows, rust_rows = rows("python"), rows("rust")
    assert python_rows and rust_rows == python_rows
