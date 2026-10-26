#!/usr/bin/env python3
# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Reference capture for the Rust program's settings loader
==========================================================

Runs a collection of TOML snippets, and every profile of the example file,
through the Python program's load_profile, and records what came out.

Usage (run from anywhere):

    python rust/tools/dump_config.py

Files written to rust/test-vectors/
-----------------------------------
config_cases.txt
    One block per case:

        ### <name>
        profile: <profile name>
        --- toml
        <the TOML text, or the single line @example for the example file>
        --- result
        ok                       followed by one  key=value  line per setting, or
        error: <message>         with the path of the config file written as <FILE>

    Values are written as Python writes them (strings in quotes, None, True,
    False), so the Rust test can compare line by line. Only the settings the
    Rust program uses are listed. signal_window is the Python program's
    signal_strength_window, which the Rust program calls signal_quality_window.

navtex.example.toml
    A snapshot of the repo's navtex.toml.example, so the Rust tests do not
    depend on the Python part of the repo being present.

Only cases where the two programs are meant to behave alike are recorded.
Settings the Rust program ignores or refuses (for example the character-sync
settings, or weak_signal_lock = false) are tested in the Rust code itself.
"""

import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
VECTORS = HERE.parent / "test-vectors"
sys.path.insert(0, str(ROOT))

from navtex_config import ConfigError, load_profile             # noqa: E402

CASES = [
    ("minimal_file", "p", '[p]\nmode = "file"\nwav_file = "a.wav"\n'),
    ("defaults_only", "p", '[p]\nmode = "live"\n'),
    ("empty_profile", "p", '[p]\n'),
    ("everything_used", "p", """[p]
mode = "file"
wav_file = "rec.wav"
log_dir = "logs"
db_file = "log.db"
quiet = true
sample_rate = 48000
centre_freq = 1650
tones_inverted = true
loop_gain = 0
signal_strength_window = 50
"""),
    ("live_with_device_index", "p", '[p]\nmode = "live"\ndevice = 2\n'),
    ("live_with_device_name", "p", '[p]\nmode = "live"\ndevice = "USB Audio"\n'),
    ("device_numeric_string_stays_a_string", "p", '[p]\nmode = "live"\ndevice = "3"\n'),
    ("int_widened_to_float", "p", '[p]\nmode = "live"\ncentre_freq = 1800\nloop_gain = 1\n'),
    ("float_values", "p", '[p]\nmode = "live"\ncentre_freq = 1712.5\nloop_gain = 0.0125\n'),
    ("two_profiles_choose_second", "second", '[first]\nmode = "live"\n\n[second]\nmode = "file"\nwav_file = "b.wav"\n'),
    ("quiet_with_log_dir", "p", '[p]\nmode = "live"\nquiet = true\nlog_dir = "x"\n'),
    ("quiet_with_db_file", "p", '[p]\nmode = "live"\nquiet = true\ndb_file = "x.db"\n'),
    ("comments_and_blank_lines", "p", '# top\n\n[p]  # the profile\nmode = "file"  # how\n\nwav_file = "c.wav"\n'),
    ("literal_string_path", "p", "[p]\nmode = \"file\"\nwav_file = 'C:\\Users\\me\\rec.wav'\n"),
    # --- errors ---
    ("bad_mode", "p", '[p]\nmode = "tape"\n'),
    ("file_without_wav_file", "p", '[p]\nmode = "file"\n'),
    ("quiet_without_a_log", "p", '[p]\nmode = "live"\nquiet = true\n'),
    ("mode_wrong_type", "p", '[p]\nmode = 5\n'),
    ("loop_gain_string", "p", '[p]\nmode = "live"\nloop_gain = "abc"\n'),
    ("quiet_integer", "p", '[p]\nmode = "live"\nquiet = 1\n'),
    ("tones_inverted_string", "p", '[p]\nmode = "live"\ntones_inverted = "yes"\n'),
    ("device_float", "p", '[p]\nmode = "live"\ndevice = 1.5\n'),
    ("device_bool", "p", '[p]\nmode = "live"\ndevice = true\n'),
    ("log_dir_integer", "p", '[p]\nmode = "live"\nlog_dir = 7\n'),
    ("wav_file_bool", "p", '[p]\nmode = "file"\nwav_file = true\n'),
    ("centre_freq_bool", "p", '[p]\nmode = "live"\ncentre_freq = true\n'),
    ("sample_rate_float", "p", '[p]\nmode = "live"\nsample_rate = 48000.0\n'),
    ("signal_window_string", "p", '[p]\nmode = "live"\nsignal_strength_window = "100"\n'),
    ("array_value", "p", '[p]\nmode = "live"\nlog_dir = [1, 2]\n'),
    ("string_with_apostrophe", "p", '[p]\nmode = "live"\nloop_gain = "it\'s"\n'),
    ("centre_freq_too_low", "p", '[p]\nmode = "live"\ncentre_freq = 50\n'),
    ("centre_freq_too_high", "p", '[p]\nmode = "live"\ncentre_freq = 24000\n'),
    ("mark_freq_replaced", "p", '[p]\nmode = "live"\nmark_freq = 1785.0\n'),
    ("space_freq_replaced", "p", '[p]\nmode = "live"\nspace_freq = 1615.0\n'),
    ("missing_profile", "nope", '[a]\nmode = "live"\n\n[b]\nmode = "live"\n'),
    ("missing_profile_no_sections", "p", '# nothing here\n'),
    ("profile_is_not_a_table", "p", 'p = 3\n'),
    ("first_error_wins_in_file_order", "p", '[p]\nmode = "live"\nloop_gain = "x"\nquiet = 2\n'),
]


def fmt(value) -> str:
    if value is None or isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return repr(value)
    return repr(value)


def describe(profile) -> list:
    return [
        f"mode={fmt(profile.mode)}",
        f"wav_file={fmt(profile.wav_file)}",
        f"device={fmt(profile.device)}",
        f"log_dir={fmt(profile.log_dir)}",
        f"db_file={fmt(profile.db_file)}",
        f"quiet={fmt(profile.quiet)}",
        f"sample_rate={fmt(profile.sample_rate)}",
        f"centre_freq={fmt(profile.centre_freq)}",
        f"tones_inverted={fmt(profile.tones_inverted)}",
        f"loop_gain={fmt(profile.loop_gain)}",
        f"signal_window={fmt(profile.signal_strength_window)}",
    ]


def run_case(path: Path, profile_name: str) -> str:
    try:
        profile = load_profile(str(path), profile_name)
    except ConfigError as e:
        message = str(e).replace(str(path), "<FILE>")
        assert "\n" not in message, message
        return "error: " + message
    return "\n".join(["ok"] + describe(profile))


def main():
    shutil.copyfile(ROOT / "navtex.toml.example", VECTORS / "navtex.example.toml")
    blocks = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.toml"
        for name, profile_name, toml_text in CASES:
            path.write_text(toml_text, encoding="utf-8", newline="")
            blocks.append(f"### {name}\nprofile: {profile_name}\n--- toml\n{toml_text.rstrip(chr(10))}\n"
                          f"--- result\n{run_case(path, profile_name)}\n")
        example = VECTORS / "navtex.example.toml"
        with open(example, "rb") as f:
            profiles = [k for k, v in tomllib.load(f).items() if isinstance(v, dict)]
        for profile_name in profiles:
            blocks.append(f"### example:{profile_name}\nprofile: {profile_name}\n--- toml\n@example\n"
                          f"--- result\n{run_case(example, profile_name)}\n")
    (VECTORS / "config_cases.txt").write_text("\n".join(blocks), encoding="utf-8", newline="")
    print(f"{len(CASES)} snippets and {len(profiles)} example profiles written to config_cases.txt")


if __name__ == "__main__":
    main()
