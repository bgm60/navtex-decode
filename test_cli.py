# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""End-to-end tests of the command-line front end (navtex_decode.main),
decoding a short synthetic WAV file. Covers quiet mode."""

import sqlite3

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

import navtex_decode
from navtex_config import ConfigError, Profile
from navtex_synth import build_transmission, random_message

FS = Profile().sample_rate


@pytest.fixture(scope="module")
def wav(tmp_path_factory):
    p = Profile()
    tx = build_transmission(random_message(np.random.default_rng(3), n_lines=2),
                            sample_rate=FS, mark_freq=p.mark_freq, space_freq=p.space_freq)
    path = tmp_path_factory.mktemp("wav") / "tx.wav"
    sf.write(str(path), tx.audio, FS)
    return path


def config(tmp_path, wav, extra=""):
    path = tmp_path / "navtex.toml"
    path.write_text(f'[p]\nmode = "file"\nwav_file = "{wav.as_posix()}"\n{extra}')
    return str(path)


def rows(db):
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT text_output FROM navtex_log").fetchall()


def test_normal_run_prints_status_and_text(tmp_path, wav, capsys):
    db = tmp_path / "log.db"
    navtex_decode.main(["p", "--config", config(tmp_path, wav, f'db_file = "{db.as_posix()}"')])
    out = capsys.readouterr().out
    assert "Decoding file" in out and "Logging decoded lines" in out
    assert "ZCZC" in out and rows(db)


def test_quiet_flag_prints_nothing_but_still_logs(tmp_path, wav, capsys):
    db = tmp_path / "log.db"
    navtex_decode.main(["p", "-q", "--config", config(tmp_path, wav, f'db_file = "{db.as_posix()}"')])
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert any("ZCZC" in r[0] for r in rows(db))


def test_quiet_profile_setting_and_no_quiet_override(tmp_path, wav, capsys):
    db = tmp_path / "log.db"
    cfg = config(tmp_path, wav, f'db_file = "{db.as_posix()}"\nquiet = true\n')
    navtex_decode.main(["p", "--config", cfg])
    assert capsys.readouterr().out == ""
    navtex_decode.main(["p", "--no-quiet", "--config", cfg])
    assert "ZCZC" in capsys.readouterr().out


def test_quiet_without_a_log_or_database_is_an_error(tmp_path, wav, capsys):
    with pytest.raises(SystemExit) as exc:
        navtex_decode.main(["p", "-q", "--config", config(tmp_path, wav)])
    assert exc.value.code == 2
    assert "log_dir or db_file" in capsys.readouterr().err


def test_quiet_without_output_in_the_profile_is_a_config_error():
    with pytest.raises(ConfigError, match="log_dir or db_file"):
        Profile(quiet=True).validate()
    Profile(quiet=True, db_file="x.db").validate()
