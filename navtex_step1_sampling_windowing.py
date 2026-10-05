# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
NAVTEX 100-baud FSK Decoder — Step 1: Audio Sampling & Windowing
==================================================================

Pipeline this module is the first stage of:

    [1] Sampling & windowing   <-- this file
    [2] Mark/space tone detection (e.g. Goertzel algorithm)
    [3] Bit-clock recovery / symbol timing
    [4] CCIR 476 (SITOR-B) character decode + FEC
    [5] Message assembly / B1B2B3B4 header parsing

Scope of this module
---------------------
This file ONLY captures audio (from a live input device or a WAV file) and
turns the continuous sample stream into overlapping, windowed analysis
frames. It does not attempt any tone/frequency detection — that is Step 2.

Design summary
---------------
NAVTEX / SITOR-B is 100 baud -> one bit = 10 ms.
At 48 kHz that is 480 samples/bit (`samples_per_symbol`).

Two competing needs shape the frame size:
  * Frequency resolution: to separate the mark/space tones (170 Hz apart)
    you want a long analysis window (long window = narrow spectral lobe).
    One full symbol period (480 samples) is the natural upper bound if you
    still want one frequency "look" per bit.
  * Timing resolution: the receiver's sample clock is not synchronized to
    the transmitter's bit clock, so Step 2/3 will need several looks per
    bit to recover bit timing (e.g. an early-late gate). That means hopping
    the window forward by a fraction of a symbol rather than by a whole
    symbol.

This module resolves that by keeping the window length at one full symbol
period, but advancing it by `samples_per_symbol / oversample` samples each
step (default oversample=8 -> ~1.25 ms hop, 60 samples @ 48 kHz). That
gives Step 2 eight overlapping spectral estimates per bit to work with.

Both the window length, hop size, oversample factor and window function are
config-driven so they can be re-tuned once Step 2's detector performance is
known, without touching this module's logic.

Dependencies
-------------
    pip install numpy scipy sounddevice soundfile

`sounddevice` (live capture) and `soundfile` (WAV I/O) are optional at
import time — the module degrades gracefully if either is missing, so you
can still use whichever source you have installed.
"""

from __future__ import annotations

import queue
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Iterator, List, NamedTuple, Optional

import numpy as np
from scipy.signal import get_window as _sp_get_window
from scipy.signal import resample_poly

try:
    import sounddevice as sd
    _HAVE_SOUNDDEVICE = True
except (ImportError, OSError):
    # OSError covers e.g. the PortAudio native library being missing even
    # though the `sounddevice` Python package is installed.
    _HAVE_SOUNDDEVICE = False

try:
    import soundfile as sf
    _HAVE_SOUNDFILE = True
except (ImportError, OSError):
    _HAVE_SOUNDFILE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NavtexConfig:
    """All tunable parameters for the sampling/windowing stage."""

    sample_rate: int = 48000      # Hz — matches typical sound-card capture rate
    baud: float = 100.0           # NAVTEX / SITOR-B symbol rate
    oversample: int = 8           # analysis frames produced per symbol period
    window_type: str = "hamming"  # any name accepted by scipy.signal.get_window
    mark_freq: float = 1785.0     # Hz — audio tone for binary '1' (mark)
    space_freq: float = 1615.0    # Hz — audio tone for binary '0' (space)
    # These values are specific to this project's receiver setup (1700 Hz
    # audio center, 170 Hz shift, mark = the HIGHER tone) -- determined
    # empirically via calibrate_tone_frequencies.py against a real
    # recording, not assumed from the ITU spec. The ITU recommendation
    # (M.476-5 §1.4) suggests 1700 Hz center for the *transmit*-side SSB
    # audio convention, but explicitly acknowledges real equipment varies
    # ("some existing equipment uses 1500 Hz... may require special
    # measures to achieve compatibility") -- a receiver's demodulated
    # audio center is a function of its own BFO/tuning offset, not fixed
    # by the standard. If you change receiver, SDR software, or tuning
    # offset, re-run calibrate_tone_frequencies.py against a fresh
    # recording rather than assuming these values still apply. Also note
    # mark/space orientation (which tone is "higher") is likewise a
    # receiver convention, not a fixed rule -- this setup has mark as the
    # higher tone, opposite of the "maritime tones" convention documented
    # in some references.

    @property
    def samples_per_symbol(self) -> int:
        """Samples spanning one 100-baud bit period (480 @ 48 kHz)."""
        return round(self.sample_rate / self.baud)

    @property
    def window_size(self) -> int:
        """Analysis frame length. One full symbol period by default —
        see module docstring for the frequency- vs time-resolution trade-off.
        """
        return self.samples_per_symbol

    @property
    def hop_size(self) -> int:
        """Samples advanced between successive frames."""
        return max(1, self.samples_per_symbol // self.oversample)


# ---------------------------------------------------------------------------
# Frame produced by the windower
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """One analysis frame handed to Step 2 (tone detection)."""

    windowed: np.ndarray   # raw samples * window function, shape (window_size,)


# ---------------------------------------------------------------------------
# Audio sources — live device or file, both yield float32 mono chunks
# ---------------------------------------------------------------------------

class AudioSource:
    """Common interface: iterate over chunks() to get raw audio chunks.

    stop() asks chunks() to finish at the next opportunity, so a decoding
    session can be ended cleanly (for example from a GUI's Stop button).
    It may be called from any thread, before or during iteration.
    """

    def chunks(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class LiveMicSource(AudioSource):
    """Captures from a Windows input device (e.g. a USB sound card fed by
    an SSB receiver's audio-out) using sounddevice's callback API.
    """

    def __init__(self, config: NavtexConfig, device: Optional[int | str] = None,
                 block_ms: float = 20.0,
                 warn: Optional[Callable[[str], None]] = None):
        if not _HAVE_SOUNDDEVICE:
            raise RuntimeError("sounddevice is not installed: pip install sounddevice")
        self.config = config
        self.device = device
        self._warn = warn
        self.block_size = round(config.sample_rate * block_ms / 1000)
        # Audio chunks from the callback thread; None is the stop marker.
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._stream = sd.InputStream(
            samplerate=config.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            device=self.device,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        if status:
            # Overflow/underflow etc. — surface it but keep running. This
            # runs on the audio driver's thread; `warn` must be safe to
            # call from there (a Qt signal emit is).
            message = f"[LiveMicSource] audio status: {status}"
            if self._warn is None:
                print(message, file=sys.stderr)
            else:
                self._warn(message)
        self._queue.put(indata[:, 0].copy())

    def chunks(self) -> Iterator[np.ndarray]:
        self._stream.start()
        try:
            while True:
                chunk = self._queue.get()
                if chunk is None:
                    return
                yield chunk
        finally:
            self._stream.stop()

    def stop(self) -> None:
        # Wakes chunks() even while it is waiting for audio.
        self._queue.put(None)

    def close(self) -> None:
        self._stream.close()


class InputDevice(NamedTuple):
    """One audio input device, as offered in a device picker."""
    index: int          # sounddevice index (can change when devices are added/removed)
    name: str           # device name as reported by the driver
    hostapi: str        # host API, e.g. "MME", "Windows WASAPI", "ALSA"
    query: str          # "name, hostapi": a stable value to store in a profile
    usable: bool        # True if it accepted the requested sample rate, mono
    is_default: bool    # the system's default input device


def list_input_devices(sample_rate: int) -> List[InputDevice]:
    """Every audio device with at least one input channel, checked for
    whether it can capture at `sample_rate` in mono.

    `query` ("device name, host API") is what a profile should store:
    unlike the index it survives devices being plugged in or removed,
    and sounddevice treats an exact match on it as unambiguous even when
    the same device appears under several host APIs (as on Windows).
    """
    if not _HAVE_SOUNDDEVICE:
        raise RuntimeError("sounddevice is not installed: pip install sounddevice")
    try:
        default_input = sd.default.device[0]
    except Exception:  # noqa: BLE001 -- no default device is not an error here
        default_input = -1
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] < 1:
            continue
        hostapi = sd.query_hostapis(info["hostapi"])["name"]
        try:
            sd.check_input_settings(device=index, channels=1, dtype="float32",
                                    samplerate=sample_rate)
            usable = True
        except Exception:  # noqa: BLE001 -- PortAudio raises various types
            usable = False
        devices.append(InputDevice(index, info["name"], hostapi,
                                   f"{info['name']}, {hostapi}", usable,
                                   index == default_input))
    return devices


class FileSource(AudioSource):
    """Reads a WAV file for offline development/testing. Resamples to the
    configured sample_rate if the file doesn't already match, and mixes
    down to mono if needed.
    """

    def __init__(self, config: NavtexConfig, path: str,
                 block_size: int = 4096):
        if not _HAVE_SOUNDFILE:
            raise RuntimeError("soundfile is not installed: pip install soundfile")
        self.config = config
        self.path = path
        self.block_size = block_size
        self._stop_requested = threading.Event()

    def chunks(self) -> Iterator[np.ndarray]:
        with sf.SoundFile(self.path) as f:
            native_rate = f.samplerate
            for block in f.blocks(blocksize=self.block_size, dtype="float32",
                                   always_2d=True):
                if self._stop_requested.is_set():
                    return
                mono = block.mean(axis=1)
                if native_rate != self.config.sample_rate:
                    mono = resample_poly(mono, self.config.sample_rate, native_rate)
                yield mono.astype(np.float32)

    def stop(self) -> None:
        self._stop_requested.set()


# ---------------------------------------------------------------------------
# Windower — turns a chunk stream into overlapping analysis frames
# ---------------------------------------------------------------------------

class Windower:
    """Buffers incoming audio chunks and emits fixed-length, overlapping
    windowed frames at a constant hop, per `NavtexConfig`.
    """

    def __init__(self, config: NavtexConfig):
        self.config = config
        self._window_fn = _sp_get_window(
            config.window_type, config.window_size, fftbins=True
        ).astype(np.float32)
        self._buffer = np.empty(0, dtype=np.float32)

    def push(self, chunk: np.ndarray) -> Iterator[Frame]:
        """Feed in one chunk of raw samples; yields zero or more Frames."""
        self._buffer = np.concatenate([self._buffer, chunk.astype(np.float32)])

        window_size = self.config.window_size
        hop = self.config.hop_size

        while len(self._buffer) >= window_size:
            yield Frame(windowed=self._buffer[:window_size] * self._window_fn)
            self._buffer = self._buffer[hop:]

    def frames(self, source: AudioSource) -> Iterator[Frame]:
        """Convenience: drive an AudioSource end-to-end and yield Frames."""
        for chunk in source.chunks():
            yield from self.push(chunk)
