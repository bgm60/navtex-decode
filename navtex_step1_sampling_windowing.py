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
import time
from dataclasses import dataclass
from typing import Iterator, Optional
from bgm_normalize import bgm_normalize   # BGM: Added audio normalization

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
    # These values are specific to this project's receiver setup (1300 Hz
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

    def describe(self) -> str:
        return (
            f"sample_rate={self.sample_rate} Hz, baud={self.baud}, "
            f"window={self.window_size} samples "
            f"({1000 * self.window_size / self.sample_rate:.2f} ms), "
            f"hop={self.hop_size} samples "
            f"({1000 * self.hop_size / self.sample_rate:.2f} ms, "
            f"{self.oversample}x oversampled), window_type={self.window_type!r}"
        )


# ---------------------------------------------------------------------------
# Frame produced by the windower
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """One analysis frame handed to Step 2 (tone detection)."""

    raw: np.ndarray        # unwindowed samples, float32, shape (window_size,)
    windowed: np.ndarray   # raw * window function
    start_sample: int      # index of raw[0] within the overall stream
    timestamp: float       # seconds, start_sample / sample_rate


# ---------------------------------------------------------------------------
# Audio sources — live device or file, both yield float32 mono chunks
# ---------------------------------------------------------------------------

class AudioSource:
    """Common interface: iterate over it to get raw audio chunks."""

    def chunks(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class LiveMicSource(AudioSource):
    """Captures from a Windows input device (e.g. a USB sound card fed by
    an SSB receiver's audio-out) using sounddevice's callback API.
    """

    def __init__(self, config: NavtexConfig, device: Optional[int | str] = None,
                 block_ms: float = 20.0):
        if not _HAVE_SOUNDDEVICE:
            raise RuntimeError("sounddevice is not installed: pip install sounddevice")
        self.config = config
        self.device = device
        self.block_size = round(config.sample_rate * block_ms / 1000)
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
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
            # Overflow/underflow etc. — surface it but keep running.
            print(f"[LiveMicSource] audio status: {status}", file=sys.stderr)
        self._queue.put(indata[:, 0].copy())

    def chunks(self) -> Iterator[np.ndarray]:
        self._stream.start()
        try:
            while True:
                yield self._queue.get()
        finally:
            self._stream.stop()

    def close(self) -> None:
        self._stream.close()


class FileSource(AudioSource):
    """Reads a WAV file for offline development/testing. Resamples to the
    configured sample_rate if the file doesn't already match, and mixes
    down to mono if needed.
    """

    def __init__(self, config: NavtexConfig, path: str,
                 block_size: int = 4096, simulate_realtime: bool = False):
        if not _HAVE_SOUNDFILE:
            raise RuntimeError("soundfile is not installed: pip install soundfile")
        self.config = config
        self.path = path
        self.block_size = block_size
        self.simulate_realtime = simulate_realtime

    def chunks(self) -> Iterator[np.ndarray]:
        with sf.SoundFile(self.path) as f:
            native_rate = f.samplerate
            for block in f.blocks(blocksize=self.block_size, dtype="float32",
                                   always_2d=True):
                mono = block.mean(axis=1)
                if native_rate != self.config.sample_rate:
                    mono = resample_poly(mono, self.config.sample_rate, native_rate)
                if self.simulate_realtime:
                    time.sleep(len(mono) / self.config.sample_rate)
                yield mono.astype(np.float32)


class SyntheticNavtexSource(AudioSource):
    """Generates a synthetic mark/space FSK test tone (no real NAVTEX
    framing) purely so the pipeline can be exercised without hardware.
    Alternates mark/space every `bits_per_tone` bits from a fixed pseudo-
    random bit sequence.
    """

    def __init__(self, config: NavtexConfig, duration_s: float = 5.0,
                 snr_db: float = 20.0, seed: int = 0, block_size: int = 4096,
                 start_offset_samples: int = 0,
                 symbol_period_override: Optional[float] = None):
        """
        start_offset_samples: silence/noise prepended before the tone
            begins, so the true bit grid does NOT line up with sample 0 —
            exactly like a real recording, where nothing guarantees the
            capture started exactly on a bit boundary. Used to test that
            bit-clock recovery (Step 3) can self-align.
        symbol_period_override: if given, use this many samples per symbol
            instead of config.samples_per_symbol (may be non-integer). Lets
            you simulate a transmitter/receiver clock-rate mismatch, e.g.
            480.2 instead of 480, to test how well timing recovery tracks
            drift. Segment lengths are chosen with fractional-carry
            rounding so the long-run average period matches this value
            exactly even though each individual segment is an integer
            number of samples.
        """
        self.config = config
        self.duration_s = duration_s
        self.snr_db = snr_db
        self.rng = np.random.default_rng(seed)
        self.block_size = block_size
        self.start_offset_samples = start_offset_samples
        self.symbol_period_override = symbol_period_override

    def chunks(self) -> Iterator[np.ndarray]:
        cfg = self.config
        n_bits = int(self.duration_s * cfg.baud)
        bits = self.rng.integers(0, 2, size=n_bits)
        self.bits = bits  # exposed for ground-truth validation in tests/demos
        sps_nominal = self.symbol_period_override or cfg.samples_per_symbol

        # Fractional-carry rounding: each segment is an integer number of
        # samples, but the running average length converges to sps_nominal
        # even when it isn't itself an integer (same idea as Bresenham's
        # line algorithm applied to symbol timing).
        seg_lens = []
        carry = 0.0
        for _ in bits:
            carry += sps_nominal
            seg_len = int(round(carry))
            carry -= seg_len
            seg_lens.append(seg_len)

        tone_samples = sum(seg_lens)
        total_samples = self.start_offset_samples + tone_samples
        signal = np.empty(total_samples, dtype=np.float32)

        if self.start_offset_samples:
            signal[: self.start_offset_samples] = self.rng.normal(
                0, 0.05, size=self.start_offset_samples
            )

        phase = 0.0
        idx = self.start_offset_samples
        for bit, seg_len in zip(bits, seg_lens):
            freq = cfg.mark_freq if bit else cfg.space_freq
            t = np.arange(seg_len) / cfg.sample_rate
            seg = np.sin(2 * np.pi * freq * t + phase)
            signal[idx: idx + seg_len] = seg
            phase = (phase + 2 * np.pi * freq * seg_len / cfg.sample_rate) % (2 * np.pi)
            idx += seg_len

        noise_power = 10 ** (-self.snr_db / 10)
        signal += self.rng.normal(0, np.sqrt(noise_power), size=signal.shape)

        for start in range(0, total_samples, self.block_size):
            yield signal[start:start + self.block_size]


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
        self._next_start_sample = 0

    def push(self, chunk: np.ndarray) -> Iterator[Frame]:
        """Feed in one chunk of raw samples; yields zero or more Frames."""
        self._buffer = np.concatenate([self._buffer, chunk.astype(np.float32)])

        window_size = self.config.window_size
        hop = self.config.hop_size

        while len(self._buffer) >= window_size:
            raw = self._buffer[:window_size].copy()
            new_raw = bgm_normalize(raw)
            windowed = new_raw * self._window_fn
            yield Frame(
                raw=raw,
                windowed=windowed,
                start_sample=self._next_start_sample,
                timestamp=self._next_start_sample / self.config.sample_rate,
            )
            self._buffer = self._buffer[hop:]
            self._next_start_sample += hop

    def frames(self, source: AudioSource) -> Iterator[Frame]:
        """Convenience: drive an AudioSource end-to-end and yield Frames."""
        for chunk in source.chunks():
            yield from self.push(chunk)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def _demo():
    """Runs the pipeline against a synthetic signal (or a WAV file, if a
    path is passed on the command line) and prints basic frame stats so you
    can confirm the plumbing works before wiring up Step 2.
    """
    config = NavtexConfig()
    print("Config:", config.describe())

    if len(sys.argv) > 1:
        source: AudioSource = FileSource(config, sys.argv[1])
        print(f"Source: WAV file {sys.argv[1]!r}")
    else:
        source = SyntheticNavtexSource(config, duration_s=2.0)
        print("Source: synthetic mark/space test tone (no file given)")

    windower = Windower(config)
    count = 0
    t0 = time.time()
    for frame in windower.frames(source):
        count += 1
        if count % 200 == 0:
            rms = float(np.sqrt(np.mean(frame.windowed ** 2)))
            print(f"frame #{count:5d}  t={frame.timestamp:7.3f}s  "
                  f"start_sample={frame.start_sample:8d}  rms={rms:.4f}")
    elapsed = time.time() - t0
    print(f"\nProduced {count} frames in {elapsed:.2f}s "
          f"({config.describe()})")


if __name__ == "__main__":
    _demo()
