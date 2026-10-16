# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
NAVTEX 100-baud FSK Decoder — Command-Line Interface
=====================================================

Runs the full Step 1-4 pipeline (sampling -> tone detection -> bit-clock
recovery -> character decode) against either a live audio input device or
a WAV file, streaming decoded text to the console as it arrives, and
optionally logging it to a UTC-timestamped text file.

This module is only the command-line front end: argument parsing,
console messages and Ctrl+C. The pipeline and decode loop live in
navtex_session.py, and the console and log-file outputs in
navtex_outputs.py.

Usage
------
All input-source settings and tunable parameters live in a TOML config
file, under one or more named `[profile]` sections -- see navtex_config.py
for the full schema and navtex.toml.example for a starting point.

    # List available audio input devices (standalone diagnostic; exits
    # immediately, no profile/config needed)
    python navtex_decode.py --list-devices

    # Decode using the [my_profile] section of the default config file
    # (navtex.toml in the current directory)
    python navtex_decode.py my_profile

    # Unattended: no console output, only the log file / database
    python navtex_decode.py my_profile --quiet

    # Decode using a specific config file
    python navtex_decode.py my_profile --config /path/to/myconfig.toml

Each profile's `mode` key ("file" or "live") determines whether it reads
a WAV file (`wav_file`) or a live input device (`device`, optional);
`log_dir` (optional, any mode) enables logging to a text file in that
directory, and `db_file` (optional, any mode) enables logging to an
SQLite database. Either, both or neither can be used.

Log files are named navtex_<UTC timestamp>.txt, e.g.
navtex_20260813_154210Z.txt -- timestamped by when decoding started, not
per-message. Ctrl+C stops live decoding cleanly and closes the log file
and audio device.

Each logged line is prefixed "[YYYYMMDD HH:MM:SS SS]" where SS is a
two-digit (00-99) relative signal strength reading -- a rolling average of
Step 3's per-bit confidence at the moment that line started (see
navtex_session.SignalStrengthTracker).
"""

from __future__ import annotations

import argparse
import dataclasses
import sqlite3
from typing import Optional, Sequence

from navtex_config import ConfigError, load_profile
from navtex_outputs import ConsoleSink, OutputSink, SqliteLogSink, TextLogSink
from navtex_session import DecodeSession, build_config, open_source


def list_devices() -> None:
    """Prints every audio device sounddevice can see (inputs and outputs)."""
    try:
        import sounddevice as sd
    except (ImportError, OSError):
        print("sounddevice is not installed or its native library is missing.")
        print("Install with: pip install sounddevice")
        return
    print(sd.query_devices())


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profile", nargs="?",
                         help="Name of the [profile] section to load from the config file")
    parser.add_argument("--config", default="navtex.toml",
                         help="Path to the TOML config file (default: navtex.toml)")
    parser.add_argument("--list-devices", action="store_true",
                         help="List available audio input devices and exit "
                              "(standalone diagnostic action -- ignores --config/profile)")
    parser.add_argument("-q", "--quiet", action="store_true", default=None,
                         help="No status messages and no decoded text on the console; "
                              "only errors are shown. Needs log_dir or db_file in the "
                              "profile. Overrides the profile's quiet setting.")
    parser.add_argument("--no-quiet", dest="quiet", action="store_false",
                         help="Show status messages and decoded text even if the "
                              "profile sets quiet = true")
    args = parser.parse_args(argv)

    if args.list_devices:
        list_devices()
        return

    if not args.profile:
        parser.error("Provide a profile name (or use --list-devices)")

    try:
        profile = load_profile(args.config, args.profile)
        if args.quiet is not None:
            profile = dataclasses.replace(profile, quiet=args.quiet)
            profile.validate()
    except ConfigError as e:
        parser.error(str(e))

    def say(message: str) -> None:
        """Status message: suppressed in quiet mode."""
        if not profile.quiet:
            print(message)

    source, description = open_source(profile, build_config(profile))
    if profile.mode == "live":
        say(f"Listening on {description}... (Ctrl+C to stop)")
    else:
        say(f"Decoding file: {profile.wav_file}")

    sinks: list[OutputSink] = [] if profile.quiet else [ConsoleSink()]
    try:
        if profile.log_dir:
            log = TextLogSink(profile.log_dir, description)
            say(f"Logging decoded text to: {log.path}")
            sinks.append(log)
        if profile.db_file:
            db = SqliteLogSink(profile.db_file)
            say(f"Logging decoded lines to database: {db.path}")
            sinks.append(db)
    except (OSError, sqlite3.Error) as e:
        for sink in sinks:
            sink.close()
        source.close()
        parser.exit(1, f"{parser.prog}: error: could not open log output: {e}\n")

    session = DecodeSession(source, profile, sinks)
    try:
        session.run()   # returns at end of file; live decoding runs until Ctrl+C
    except KeyboardInterrupt:
        say("\n\nStopped by user.")


if __name__ == "__main__":
    main()
