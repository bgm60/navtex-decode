#!/usr/bin/env python3
# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Checks the Rust decoder through its Python interface (navtex_rs).

Build and install the extension first (from rust/navtex-py, in the virtual
environment you use for the project):

    maturin develop --release

Then, from anywhere:

    python rust/tools/check_rust_decoder.py

For every reference set that has audio, this feeds input.wav to
navtex_rs.Decoder in several chunk sizes and compares the result with what
the Python decoder produced (rust/test-vectors/<set>/):

* every character and the bit index it came from  (lock_full_chars.csv)
* the decoded text                                (decoded_full.txt)
* the release and switch counts                   (manifest.json)
* the per-bit confidences                         (bits.csv)

It also checks the Decoder's argument checking. The exit code is 0 if
everything passed.
"""

from __future__ import annotations

import array
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import navtex_rs
except ImportError:
    sys.exit("navtex_rs is not installed: run 'maturin develop --release' in rust/navtex-py first")

VECTORS = Path(__file__).resolve().parents[1] / "test-vectors"
CHUNK_SIZES = (4096, 777, 48000, None)       # None = the whole file in one call
failures = []


def check(ok, message):
    if not ok:
        failures.append(message)
        print(f"  FAIL: {message}")


def read_expected_chars(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(int(row["bit_index"]), "".join(chr(int(n)) for n in row["codepoints"].split()))
                for row in csv.DictReader(f)]


def run(audio, chunk, collect_confidences=False):
    """Decodes the whole audio; returns (events, confidences, states seen, decoder)."""
    decoder = navtex_rs.Decoder()
    events, confidences, states = [], [], set()
    step = len(audio) if chunk is None else chunk
    for start in range(0, len(audio), step):
        events.extend(decoder.push(audio[start:start + step]))
        states.add(decoder.state)
        if collect_confidences:
            confidences.extend(decoder.take_confidences())
    events.extend(decoder.finish())
    return events, confidences, states, decoder


def check_set(set_dir):
    name = set_dir.name
    audio, rate = sf.read(str(set_dir / "input.wav"), dtype="float32")
    assert rate == 48000 and audio.ndim == 1
    want_chars = read_expected_chars(set_dir / "lock_full_chars.csv")
    # newline="" keeps carriage returns as they are in the file
    with open(set_dir / "decoded_full.txt", newline="", encoding="utf-8") as f:
        want_text = f.read().replace("\r", "")
    manifest = json.loads((set_dir / "manifest.json").read_text())["lock"]["full"]
    with open(set_dir / "bits.csv", newline="") as f:
        want_conf = [float(row["confidence"]) for row in csv.DictReader(f)]

    reference = None
    for chunk in CHUNK_SIZES:
        label = f"{name}, chunk {chunk or 'whole file'}"
        events, confidences, states, decoder = run(audio, chunk, collect_confidences=(reference is None))
        got = [(pos, ch) for ch, pos, _ in events]
        check(got == want_chars, f"{label}: characters differ from the Python decoder "
                                 f"({len(got)} against {len(want_chars)})")
        text = "".join(ch for ch, _, _ in events).replace("\r", "")
        check(text == want_text, f"{label}: decoded text differs")
        check(decoder.releases == manifest["releases"],
              f"{label}: releases {decoder.releases} against {manifest['releases']}")
        check(decoder.switches == manifest["switches"],
              f"{label}: switches {decoder.switches} against {manifest['switches']}")
        numbers = [n for _, _, n in events]
        check(numbers == sorted(numbers) and (not numbers or numbers[-1] <= decoder.releases),
              f"{label}: release numbers should never go down and end at or below {decoder.releases}")
        check(decoder.state == "searching", f"{label}: state after finish is {decoder.state}")
        if want_chars and chunk is not None:     # a single call only shows the state at its end
            check("data" in states, f"{label}: never reported the state 'data'")
        if reference is None:
            reference = got
            check(len(confidences) == len(want_conf),
                  f"{label}: {len(confidences)} confidences against {len(want_conf)}")
            worst = max((abs(a - b) for a, b in zip(confidences, want_conf)), default=0.0)
            check(worst <= 1e-9, f"{label}: confidence error {worst:e}")
        else:
            check(got == reference, f"{label}: differs from the first chunk size")
    print(f"{name}: {len(want_chars)} characters, {manifest['releases']} releases, "
          f"{manifest['switches']} switches")


def check_arguments():
    print("argument checks")
    decoder = navtex_rs.Decoder()
    check(decoder.push(np.zeros(0, dtype=np.float32)) == [], "an empty chunk should give no text")
    check(decoder.push(array.array("f", [0.0] * 1000)) == [], "array.array('f') should be accepted")
    check(decoder.state == "searching" and decoder.releases == 0 and decoder.switches == 0,
          "a fresh decoder should be searching with no releases or switches")
    for bad in (np.zeros(10, dtype=np.float64), [0.0, 1.0], "audio"):
        try:
            decoder.push(bad)
        except (TypeError, BufferError):         # the wrong kind of audio data
            continue
        failures.append(f"push({type(bad).__name__}) should raise TypeError or BufferError")
        print(f"  FAIL: push({type(bad).__name__}) should raise TypeError or BufferError")
    for kwargs in ({"sample_rate": 10}, {"mark_freq": 0.0}, {"space_freq": 30000.0},
                   {"oversample": 0}, {"oversample": 100}, {"loop_gain": -1.0},
                   {"loop_gain": float("nan")}):
        try:
            navtex_rs.Decoder(**kwargs)
        except ValueError:
            continue
        failures.append(f"Decoder({kwargs}) should raise ValueError")
        print(f"  FAIL: Decoder({kwargs}) should raise ValueError")


def main():
    sets = sorted(d for d in VECTORS.iterdir()
                  if (d / "input.wav").is_file() and (d / "lock_full_chars.csv").is_file())
    if not sets:
        sys.exit(f"no reference sets with audio found in {VECTORS}")
    for set_dir in sets:
        check_set(set_dir)
    check_arguments()
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED")
        return 1
    print(f"all checks passed on {len(sets)} sets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
