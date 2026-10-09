# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
NAVTEX Decoder — Decoding Session
===================================

Builds the Step 1-4 decoding pipeline from a profile and runs it,
delivering decoded text to a set of output sinks (see navtex_outputs.py).

This is the part of the application shared by every front end: the
command-line tool (navtex_decode.py) and the GUI. It has no
knowledge of the console, argument parsing or Ctrl+C.

Typical use:

    config = build_config(profile)
    source, description = open_source(profile, config)
    session = DecodeSession(source, profile, sinks=[ConsoleSink()])
    session.run()          # returns when the source ends or stop() is called

`run()` blocks, so a GUI runs it on a worker thread and calls `stop()`
from the GUI thread; stop() is safe to call from any thread.

While it runs, three read-only values can be polled from any thread, for
example to drive meters:

    session.tracker.level         00-99 signal-strength reading
    session.audio_level.peak_dbfs  audio input peak level, dB full scale
    session.sync_state             "searching", "sync" or "data"
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Callable, Deque, Iterable, Iterator, Optional, Tuple

import numpy as np

from navtex_config import Profile
from navtex_outputs import LineAssembler, OutputSink, WarnFn, warn_to_stderr
from navtex_soft_fec_combine import (
    SoftBitSync,
    SoftCharacterGrouper,
    SoftFecCombiner,
    decode_bit_stream_soft,
)
from navtex_step1_sampling_windowing import (
    AudioSource,
    FileSource,
    LiveMicSource,
    NavtexConfig,
    Windower,
)
from navtex_step2_tone_detection import ToneDetector


# ---------------------------------------------------------------------------
# Signal strength
# ---------------------------------------------------------------------------

class SignalStrengthTracker:
    """Rolling average of Step 3's per-bit confidence, reported as a
    two-digit 00-99 reading.

    Confidence is gain-independent and already in [0, 1]. A single bit's
    value is noisy (it dips whenever the analysis window straddles a
    transition), so it is averaged over a trailing window: the default of
    100 bits is about one second at 100 baud, long enough to smooth that
    jitter and short enough to follow a fade within a line or two.

    The average is then mapped linearly from [FLOOR, CEILING] onto
    [0, 99]. FLOOR sits just above the highest one-second average seen on
    noise alone, so noise reads 00 while weak but readable text does
    not. CEILING is just above the average for a clean, strong signal.
    Both were calibrated for the Hamming window and oversample = 8, and
    shift slightly with other settings.
    """

    FLOOR = 0.50    # confidence at which decoding fails -> 00 (previously 0.55)
    CEILING = 0.92  # confidence of a clean, strong signal -> 99

    def __init__(self, window: int = 100):
        self._values: Deque[float] = deque(maxlen=window)

    def update(self, confidence: float) -> None:
        self._values.append(confidence)

    @property
    def level(self) -> int:
        values = list(self._values)   # snapshot: may be read from another thread
        if not values:
            return 0
        avg = sum(values) / len(values)
        scaled = (avg - self.FLOOR) / (self.CEILING - self.FLOOR)
        return max(0, min(99, round(scaled * 99)))


def tap_bit_decisions(bit_decisions, tracker: SignalStrengthTracker,
                      hook: Optional[Callable[[], None]] = None, every: int = 50):
    """Passes the bit-decision stream through unchanged, updating
    `tracker` with each bit's confidence on the way, and calling `hook`
    (if given) once every `every` bits.

    Generators are pull-based, so by the time a decoded character comes
    out of Step 4, `tracker` already includes every bit used to produce
    it, and reading it at that point gives a current value.
    """
    count = 0
    for bd in bit_decisions:
        tracker.update(bd.confidence)
        if hook is not None:
            count += 1
            if count >= every:
                count = 0
                hook()
        yield bd


# ---------------------------------------------------------------------------
# Audio input level
# ---------------------------------------------------------------------------

class AudioLevelMeter:
    """Peak level of the incoming audio, for setting sound-card gain.

    This is separate from the signal-strength reading: it says nothing
    about decoding, only whether the audio reaching the decoder is too
    quiet, sensible, or clipping. The decoder itself is gain-independent,
    so anything comfortably below clipping works.

    Values are updated on the decoding thread and may be read from any
    thread.
    """

    PEAK_HOLD = 0.3        # seconds: peak_dbfs is the highest level over this period
    CLIP_HOLD = 2.0        # seconds: clipping stays flagged this long after it happens
    CLIP_LEVEL = 0.999     # |sample| at or above this counts as clipping
    SILENCE_DBFS = -100.0  # reported when there is no audio at all

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._peaks: Deque[Tuple[float, float]] = deque(maxlen=256)  # (time, peak)
        self._last_clip = -math.inf

    def update(self, chunk: np.ndarray) -> None:
        if len(chunk) == 0:
            return
        peak = float(np.max(np.abs(chunk)))
        now = self._clock()
        self._peaks.append((now, peak))
        if peak >= self.CLIP_LEVEL:
            self._last_clip = now

    @property
    def peak_dbfs(self) -> float:
        """Highest peak over the last PEAK_HOLD seconds, in dBFS (0 dB is
        full scale). SILENCE_DBFS if nothing recent has arrived."""
        cutoff = self._clock() - self.PEAK_HOLD
        recent = [peak for t, peak in list(self._peaks) if t >= cutoff]
        peak = max(recent, default=0.0)
        if peak <= 0.0:
            return self.SILENCE_DBFS
        return max(self.SILENCE_DBFS, 20.0 * math.log10(peak))

    @property
    def clipping(self) -> bool:
        """True if the audio clipped within the last CLIP_HOLD seconds."""
        return self._clock() - self._last_clip <= self.CLIP_HOLD


def tap_chunks(source: AudioSource, meter: AudioLevelMeter) -> Iterator[np.ndarray]:
    """Passes the source's audio chunks through unchanged, updating
    `meter` on the way."""
    for chunk in source.chunks():
        meter.update(chunk)
        yield chunk


# ---------------------------------------------------------------------------
# Building the pipeline
# ---------------------------------------------------------------------------

def build_config(profile: Profile) -> NavtexConfig:
    """The audio/tone settings from a profile, as used by Steps 1-3. The
    mark and space frequencies come from the profile's centre_freq and
    tones_inverted (see Profile.mark_freq and Profile.space_freq)."""
    return NavtexConfig(
        sample_rate=profile.sample_rate,
        oversample=profile.oversample,
        window_type=profile.window_type,
        mark_freq=profile.mark_freq,
        space_freq=profile.space_freq,
    )


def open_live_source(config: NavtexConfig, device=None,
                     warn: Optional[WarnFn] = None) -> Tuple[AudioSource, str]:
    """Opens a live audio input. `device` is an index, a name substring
    (or "name, host API"), or None for the system default. A numeric
    string is treated as an index. Audio driver warnings go to `warn`,
    which is called on the audio driver's thread, or to stderr if None.
    Returns the source and a human-readable description."""
    if device is not None:
        try:
            device = int(device)
        except ValueError:
            pass  # a name substring; sounddevice accepts that too
    source = LiveMicSource(config, device=device, warn=warn)
    description = (f"live audio device {device!r}" if device is not None
                   else "live audio (default device)")
    return source, description


def open_source(profile: Profile, config: NavtexConfig) -> Tuple[AudioSource, str]:
    """Opens the audio source a profile asks for (live device or WAV
    file). Returns the source and a human-readable description."""
    if profile.mode == "live":
        return open_live_source(config, profile.device)
    return FileSource(config, profile.wav_file), f"WAV file {profile.wav_file!r}"


def build_grouper(profile: Profile) -> SoftCharacterGrouper:
    return SoftCharacterGrouper(
        sync_window=profile.sync_window,
        acquire_threshold=profile.char_acquire_threshold,
        drop_threshold=profile.char_drop_threshold,
        switch_margin=profile.char_switch_margin,
        min_groups_for_acquire=profile.min_groups_for_acquire,
    )


def build_fec(profile: Profile) -> SoftFecCombiner:
    return SoftFecCombiner(
        lock_window=profile.lock_window,
        acquire_threshold=profile.fec_acquire_threshold,
        switch_margin=profile.fec_switch_margin,
        min_samples_for_rate=profile.min_samples_for_rate,
        rate_window=profile.rate_window,
    )


def decode_characters(source: AudioSource, profile: Profile,
                      tracker: SignalStrengthTracker,
                      grouper: Optional[SoftCharacterGrouper] = None,
                      fec: Optional[SoftFecCombiner] = None,
                      audio_level: Optional[AudioLevelMeter] = None,
                      bit_hook: Optional[Callable[[], None]] = None,
                      hook_every: int = 50) -> Iterator[str]:
    """The full Step 1-4 pipeline: audio in, decoded characters out.
    Updates `tracker` with every bit decision along the way, and
    `audio_level` (if given) with every audio chunk, and calls `bit_hook`
    (if given) every `hook_every` bits. `grouper` and `fec` are built from
    the profile unless supplied (DecodeSession supplies its own so it can
    report their status)."""
    config = build_config(profile)
    windower = Windower(config)
    detector = ToneDetector(config)
    bitsync = SoftBitSync(config, loop_gain=profile.loop_gain)
    grouper = grouper if grouper is not None else build_grouper(profile)
    fec = fec if fec is not None else build_fec(profile)
    if audio_level is None:
        frames = windower.frames(source)
    else:
        frames = (frame for chunk in tap_chunks(source, audio_level)
                  for frame in windower.push(chunk))
    bit_decisions = tap_bit_decisions(
        bitsync.process_stream(detector.process_stream(frames)),
        tracker, hook=bit_hook, every=hook_every,
    )
    return decode_bit_stream_soft(bit_decisions, grouper=grouper, fec=fec,
                                  phasing_burst_threshold=profile.phasing_burst_threshold)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class DecodeSession:
    """Runs one decoding session: one audio source, one profile, and a set
    of output sinks.

    run() decodes until the source ends or stop() is called, then closes
    every sink and the source, including when it exits with an exception
    (such as KeyboardInterrupt in the CLI).

    `sync_state` is one of:
        "searching"  no character alignment: no signal, or only noise
        "sync"       bits are grouping into valid characters, but no
                     message text is getting through (phasing between
                     messages, or the start of a message)
        "data"       message text is being decoded. On a weak signal
                     some characters may be wrong; the signal-strength
                     reading says how good reception is.
    It is refreshed about twice a second of audio.

    "data" means the character alignment is held and the locked DX/RX
    parity's recent match rate is at least DATA_MIN_MATCH_RATE. Measured
    with synthetic signals, correct text still came through at match
    rates down to about 0.17, while noise with the alignment still held
    gave 0.00. Requiring the alignment matters: once it is lost, no new
    codewords reach the combiner, so its match rate freezes at its last
    value instead of falling.
    """

    SYNC_CHECK_BITS = 50          # how often (in bits) sync_state is refreshed
    DATA_MIN_MATCH_RATE = 0.15

    def __init__(self, source: AudioSource, profile: Profile,
                 sinks: Iterable[OutputSink], warn: WarnFn = warn_to_stderr):
        self.source = source
        self.profile = profile
        self.tracker = SignalStrengthTracker(window=profile.signal_strength_window)
        self.audio_level = AudioLevelMeter()
        self.sync_state = "searching"
        self._grouper = build_grouper(profile)
        self._fec = build_fec(profile)
        self._sinks = list(sinks)
        self._warn = warn

    def run(self) -> None:
        assembler = LineAssembler(self._sinks, strength=self._strength,
                                  warn=self._warn)
        try:
            for ch in decode_characters(self.source, self.profile, self.tracker,
                                        grouper=self._grouper, fec=self._fec,
                                        audio_level=self.audio_level,
                                        bit_hook=self._update_sync_state,
                                        hook_every=self.SYNC_CHECK_BITS):
                assembler.feed(ch)
        finally:
            assembler.close()
            self.source.close()

    def _strength(self) -> int:
        return self.tracker.level

    def _update_sync_state(self) -> None:
        # Runs on the decoding thread (the only thread that touches the
        # grouper and combiner), called from the bit-decision tap.
        rate = self._fec.match_rate()
        if (self._grouper.is_aligned() and rate is not None
                and rate >= self.DATA_MIN_MATCH_RATE):
            self.sync_state = "data"
        elif self._grouper.in_sync():
            self.sync_state = "sync"
        else:
            self.sync_state = "searching"

    def stop(self) -> None:
        """Asks the session to finish. Returns immediately; run() returns
        shortly afterwards. Safe to call from any thread."""
        self.source.stop()
