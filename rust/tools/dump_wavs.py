#!/usr/bin/env python3
# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Reference capture for the Rust WAV reader
==========================================

Writes a handful of short WAV files in the formats the Rust program accepts,
and for each one the exact samples the Python decoder's FileSource would
deliver from it.

Usage (run from anywhere):

    python rust/tools/dump_wavs.py

Files written to rust/test-vectors/wav_formats/
-----------------------------------------------
<name>.wav   A short test signal at 48 kHz.
<name>.f32   The samples Python gets from it: read with soundfile as float32
             in blocks of 4096 frames, the channels averaged to mono with
             numpy's mean (exactly as FileSource does), stored as raw
             little-endian 32-bit floats.

The signals are seeded noise with full-scale and zero values added, so the
conversion of every sample format is tested at its extremes.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

OUT = Path(__file__).resolve().parent.parent / "test-vectors" / "wav_formats"
RATE = 48000
FRAMES = 4500            # a little more than one 4096-frame block

# name: (channels, soundfile subtype)
FORMATS = {
    "mono_u8": (1, "PCM_U8"),
    "mono_16": (1, "PCM_16"),
    "stereo_16": (2, "PCM_16"),
    "three_16": (3, "PCM_16"),
    "mono_24": (1, "PCM_24"),
    "stereo_24": (2, "PCM_24"),
    "mono_32": (1, "PCM_32"),
    "mono_float": (1, "FLOAT"),
    "stereo_float": (2, "FLOAT"),
}


def signal(channels: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.uniform(-1.0, 1.0, size=(FRAMES, channels))
    data[0, :] = 0.0
    data[1, :] = -1.0           # the most negative value
    data[2, :] = 0.999          # near the most positive
    data[3, :] = 1.0 / 3.0
    data[4, :] = -1.0 / 7.0
    return data


def python_samples(path: Path) -> np.ndarray:
    """What FileSource.chunks() yields, joined up."""
    out = []
    with sf.SoundFile(str(path)) as f:
        assert f.samplerate == RATE
        for block in f.blocks(blocksize=4096, dtype="float32", always_2d=True):
            out.append(block.mean(axis=1).astype(np.float32))
    return np.concatenate(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for seed, (name, (channels, subtype)) in enumerate(FORMATS.items(), start=1):
        wav = OUT / f"{name}.wav"
        sf.write(str(wav), signal(channels, seed), RATE, subtype=subtype, format="WAV")
        samples = python_samples(wav)
        (OUT / f"{name}.f32").write_bytes(samples.astype("<f4").tobytes())
        print(f"{name}: {channels} channel(s), {subtype}, {len(samples)} samples")


if __name__ == "__main__":
    sys.exit(main())
