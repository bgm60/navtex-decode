#!/usr/bin/env python3
# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Reference-vector capture for the Rust port
===========================================

Runs a signal through the real Python decoder and saves what every stage
produced, so each Rust stage can be tested against it.

Usage (run from anywhere; the decoder modules are found in the repo root):

    python rust/tools/dump_reference.py synth --name synth_snr10_seed1 --seed 1 --snr 10
    python rust/tools/dump_reference.py wav my_recording.wav --name real_clip1

Output goes to rust/test-vectors/<name>/ (change the root with --out).

Files written
-------------
input.wav            The exact mono signal the decoder received, at the
                     decoder's sample rate, as 16-bit PCM. (A WAV in another
                     format or rate is converted first, exactly as the
                     decoder's own FileSource does it, and then written
                     here; the reference run reads THIS file back, so what
                     is stored is what the reference results came from.)
                     16-bit PCM values are read as value / 32768.
window.csv           The analysis window coefficients (float32 values).
tone.csv             One row per analysis frame:   frame, diff, bit
bits.csv             One row per recovered bit:    bit_index, bit, confidence, soft_value
soft_lock_chars.csv  Characters from the weak-signal lock (the default
                     decoder):                     seq, bit_index, codepoints
classic_chars.csv    Characters from the classic hard-decision lock:
                                                   seq, codepoints
decoded_soft_lock.txt, decoded_classic.txt
                     The same characters as readable text.
manifest.json        Settings used, counts, library versions, and checks.

CSV conventions
---------------
* Plain comma separated, one header row, "\\n" line endings, no quoting.
* bit is 1 for mark and 0 for space.
* Floats are written with Python's repr(), which round-trips exactly.
* codepoints is the Unicode code point of the character(s) as decimal
  numbers (space separated if there is more than one), so no text needs
  escaping.

This script only reads the decoder; it changes nothing in it.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np          # noqa: E402
import scipy                # noqa: E402
import soundfile as sf      # noqa: E402

from navtex_config import Profile                                              # noqa: E402
from navtex_session import build_config, build_fec, build_grouper, build_soft_lock   # noqa: E402
from navtex_soft_fec_combine import SoftBitSync, decode_bit_stream_soft        # noqa: E402
from navtex_soft_lock import SoftLockParams                                    # noqa: E402
from navtex_step1_sampling_windowing import FileSource, Windower               # noqa: E402
from navtex_step2_tone_detection import ToneDetector                           # noqa: E402

DEFAULT_OUT = REPO_ROOT / "rust" / "test-vectors"


# ---------------------------------------------------------------------------
# Running the Python pipeline
# ---------------------------------------------------------------------------

def run_front_end(chunks, config, loop_gain):
    """Windowing, tone detection and bit sync, exactly as the production
    pipeline chains them. Returns (tone samples, bit decisions)."""
    windower = Windower(config)
    detector = ToneDetector(config)
    bitsync = SoftBitSync(config, loop_gain=loop_gain)
    tones = []
    decisions = []
    for chunk in chunks:
        for frame in windower.push(chunk):
            sample = detector.process(frame)
            tones.append(sample)
            decisions.extend(bitsync.push(sample))
    return tones, decisions


def array_chunks(audio, size):
    for start in range(0, len(audio), size):
        yield audio[start:start + size]


# ---------------------------------------------------------------------------
# Building the input signal
# ---------------------------------------------------------------------------

def synthetic_audio(args, profile):
    """A synthetic transmission through a noisy channel (all seeded)."""
    from navtex_synth import Channel, apply_channel, build_transmission, random_message

    text = random_message(np.random.default_rng(args.seed), n_lines=args.lines)
    tx = build_transmission(text, sample_rate=profile.sample_rate,
                            mark_freq=profile.mark_freq, space_freq=profile.space_freq)
    channel = Channel(snr_db=args.snr, fade_doppler_hz=args.fade_doppler, seed=args.seed)
    audio = apply_channel(tx.audio, tx.sample_rate, tx.signal_power, channel)
    info = {
        "kind": "synthetic",
        "seed": args.seed,
        "lines": args.lines,
        "channel": dataclasses.asdict(channel),
        "expected_text": tx.expected_text,
    }
    return audio, info


def wav_audio(args, config):
    """A recording, converted to mono at the decoder's rate exactly as the
    decoder's own FileSource does it."""
    chunks = list(FileSource(config, str(args.path)).chunks())
    if not chunks:
        raise SystemExit(f"{args.path} contains no audio")
    audio = np.concatenate(chunks).astype(np.float32)
    return audio, {"kind": "wav", "source_file": Path(args.path).name}


# ---------------------------------------------------------------------------
# Writing files
# ---------------------------------------------------------------------------

def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def num(x):
    """Full-precision text for a float."""
    return repr(float(x))


def codepoints(text):
    return " ".join(str(ord(c)) for c in text)


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="folder that will hold the per-signal folders "
                        f"(default: {DEFAULT_OUT})")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="generate a seeded synthetic signal")
    s.add_argument("--name", required=True, help="name of the output folder")
    s.add_argument("--seed", type=int, default=1, help="seed for message and noise (default 1)")
    s.add_argument("--snr", type=float, default=None,
                   help="SNR in dB in a 2500 Hz bandwidth; omit for no noise")
    s.add_argument("--lines", type=int, default=4, help="lines in the message (default 4)")
    s.add_argument("--fade-doppler", type=float, default=0.0,
                   help="Rayleigh fading Doppler bandwidth in Hz; 0 for none")

    w = sub.add_parser("wav", help="use a recording")
    w.add_argument("path", type=Path, help="WAV file")
    w.add_argument("--name", required=True, help="name of the output folder")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    profile = Profile()                      # the decoder's current defaults
    config = build_config(profile)
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)

    # 1. The input signal, stored as 16-bit PCM and read back so that the
    #    stored file is exactly what the reference run sees.
    if args.command == "synth":
        audio, source_info = synthetic_audio(args, profile)
    else:
        audio, source_info = wav_audio(args, config)
    wav_path = out / "input.wav"
    sf.write(str(wav_path), np.clip(audio, -1.0, 1.0), profile.sample_rate, subtype="PCM_16")

    stored, rate = sf.read(str(wav_path), dtype="float32")
    assert rate == profile.sample_rate and stored.ndim == 1

    # 2. Reference run: read the stored file the way the decoder reads files.
    chunks = FileSource(config, str(wav_path)).chunks()
    tones, decisions = run_front_end(chunks, config, profile.loop_gain)

    # 3. Check that chunk size does not change the front-end results. The
    #    Rust port will be fed chunks of whatever size the caller likes, so
    #    this property is one it has to keep.
    chunking_ok = True
    for size in (7, 777, 48000):
        t2, d2 = run_front_end(array_chunks(stored, size), config, profile.loop_gain)
        same = ([(t.diff, t.bit) for t in tones] == [(t.diff, t.bit) for t in t2]
                and [(d.bit, d.confidence, d.soft_value) for d in decisions]
                == [(d.bit, d.confidence, d.soft_value) for d in d2])
        if not same:
            chunking_ok = False
            print(f"WARNING: chunk size {size} gives different front-end results", file=sys.stderr)

    # 4. Character stages.
    soft_lock = build_soft_lock(profile)
    soft_events = list(soft_lock.events(iter(decisions)))

    classic_profile = dataclasses.replace(profile, weak_signal_lock=False)
    classic = list(decode_bit_stream_soft(
        iter(decisions),
        grouper=build_grouper(classic_profile),
        fec=build_fec(classic_profile),
        phasing_burst_threshold=classic_profile.phasing_burst_threshold,
    ))

    # 5. Write everything.
    window = Windower(config)._window_fn
    write_csv(out / "window.csv", ["index", "value"],
              ((i, num(v)) for i, v in enumerate(window)))
    write_csv(out / "tone.csv", ["frame", "diff", "bit"],
              ((i, num(t.diff), int(t.bit)) for i, t in enumerate(tones)))
    write_csv(out / "bits.csv", ["bit_index", "bit", "confidence", "soft_value"],
              ((i, int(d.bit), num(d.confidence), num(d.soft_value))
               for i, d in enumerate(decisions)))
    write_csv(out / "soft_lock_chars.csv", ["seq", "bit_index", "codepoints"],
              ((i, pos, codepoints(ch)) for i, (ch, pos) in enumerate(soft_events)))
    write_csv(out / "classic_chars.csv", ["seq", "codepoints"],
              ((i, codepoints(ch)) for i, ch in enumerate(classic)))
    write_text(out / "decoded_soft_lock.txt", "".join(ch for ch, _ in soft_events))
    write_text(out / "decoded_classic.txt", "".join(classic))

    manifest = {
        "name": args.name,
        "source": source_info,
        "samples": int(len(stored)),
        "sample_rate": profile.sample_rate,
        "counts": {
            "frames": len(tones),
            "bits": len(decisions),
            "soft_lock_chars": len(soft_events),
            "classic_chars": len(classic),
        },
        "front_end": {
            "window_type": config.window_type,
            "window_size": config.window_size,
            "hop_size": config.hop_size,
            "samples_per_symbol": config.samples_per_symbol,
            "oversample": config.oversample,
            "mark_freq": config.mark_freq,
            "space_freq": config.space_freq,
            "loop_gain": profile.loop_gain,
        },
        "profile": dataclasses.asdict(profile),
        "soft_lock_params": dataclasses.asdict(SoftLockParams()),
        "checks": {"chunk_size_independent": chunking_ok},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "python": sys.version.split()[0]},
        "notes": [
            "The window is scipy's periodic ('fftbins=True') window, stored as float32; "
            "windowing multiplies float32 samples by float32 coefficients, and the "
            "tone detector then works in float64.",
            "Frame count = floor((samples - window_size) / hop_size) + 1.",
            "soft_value and confidence are sums of float64 diffs taken in order.",
        ],
    }
    write_text(out / "manifest.json", json.dumps(manifest, indent=2) + "\n")

    print(f"{args.name}: {len(stored)} samples, {len(tones)} frames, {len(decisions)} bits, "
          f"{len(soft_events)} soft-lock chars, {len(classic)} classic chars")
    print(f"chunk-size independent: {chunking_ok}")
    print(f"written to {out}")
    return 0 if chunking_ok else 1


if __name__ == "__main__":
    sys.exit(main())
