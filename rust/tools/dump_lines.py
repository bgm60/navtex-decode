#!/usr/bin/env python3
# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Reference capture for the Rust command-line program: signal quality and lines
==============================================================================

Adds two files to each reference set in rust/test-vectors/ that has an
input.wav, so the Rust signal-quality meter and line assembler can be tested
against the real Python decoder.

Usage (run from anywhere; the decoder modules are found in the repo root):

    python rust/tools/dump_lines.py              every set that has an input.wav
    python rust/tools/dump_lines.py real_clip1   named sets only

Files written
-------------
quality.csv   One row per recovered bit:   bit_index, level
              `level` is what the Python SignalStrengthTracker reads (00-99)
              after it has been given that bit's confidence. The confidences
              themselves are the ones already in bits.csv.
lines.csv     The events the Python decoder (engine = "python") sends to its
              output sinks when it decodes input.wav, one per row:
                  S,<level>        a line starts; <level> is the 00-99 reading
                  W,<codepoint>    one character is written
                  E                the line ends
              Timestamps are left out: they come from the clock.

The Python name for the 00-99 reading is "signal strength". The Rust program
calls it "signal quality"; the capture files just say "level".
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                   # the repo root, which holds the decoder
VECTORS = HERE.parent / "test-vectors"
sys.path.insert(0, str(ROOT))

from navtex_config import Profile                                   # noqa: E402
from navtex_outputs import OutputSink                               # noqa: E402
from navtex_session import DecodeSession, SignalStrengthTracker, build_config  # noqa: E402
from navtex_step1_sampling_windowing import FileSource              # noqa: E402


class EventRecorder(OutputSink):
    """Records line events as rows of text."""

    def __init__(self):
        self.rows = []

    def line_start(self, timestamp, strength):
        self.rows.append(f"S,{strength}")

    def write(self, text):
        self.rows.extend(f"W,{ord(c)}" for c in text)

    def line_end(self):
        self.rows.append("E")


def write_lines(wav: Path, out: Path) -> int:
    profile = Profile(mode="file", wav_file=str(wav), engine="python")
    recorder = EventRecorder()
    session = DecodeSession(FileSource(build_config(profile), str(wav)), profile, [recorder])
    session.run()
    out.write_text("\n".join(recorder.rows) + "\n", encoding="utf-8", newline="")
    return len(recorder.rows)


def write_quality(bits_csv: Path, out: Path) -> int:
    profile = Profile()
    tracker = SignalStrengthTracker(window=profile.signal_strength_window)
    rows = ["bit_index,level"]
    for line in bits_csv.read_text(encoding="utf-8").splitlines()[1:]:
        index, _bit, confidence, _soft = line.split(",")
        tracker.update(float(confidence))
        rows.append(f"{index},{tracker.level}")
    out.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="")
    return len(rows) - 1


def main(argv):
    names = argv[1:] or sorted(p.parent.name for p in VECTORS.glob("*/input.wav"))
    for name in names:
        folder = VECTORS / name
        if not (folder / "input.wav").is_file():
            sys.exit(f"{name}: no input.wav in {folder}")
        n_events = write_lines(folder / "input.wav", folder / "lines.csv")
        n_bits = write_quality(folder / "bits.csv", folder / "quality.csv")
        print(f"{name}: {n_events} line events, {n_bits} quality readings")


if __name__ == "__main__":
    main(sys.argv)
