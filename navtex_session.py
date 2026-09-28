"""
NAVTEX Decoder — Decoding Session
===================================

Builds the Step 1-4 decoding pipeline from a profile and runs it,
delivering decoded text to a set of output sinks (see navtex_outputs.py).

This is the part of the application shared by every front end: the
command-line tool (navtex_decode.py) and, in future, the GUI. It has no
knowledge of the console, argument parsing or Ctrl+C.

Typical use:

    config = build_config(profile)
    source, description = open_source(profile, config)
    session = DecodeSession(source, profile, sinks=[ConsoleSink()])
    session.run()          # returns when the source ends or stop() is called

`run()` blocks, so a GUI runs it on a worker thread and calls `stop()`
from the GUI thread; stop() is safe to call from any thread.
`session.tracker.level` gives the current 00-99 signal-strength reading
at any time, for example to drive a meter.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Iterator, Tuple

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

    FLOOR = 0.55    # confidence at which decoding fails -> 00
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


def tap_bit_decisions(bit_decisions, tracker: SignalStrengthTracker):
    """Passes the bit-decision stream through unchanged, updating
    `tracker` with each bit's confidence on the way.

    Generators are pull-based, so by the time a decoded character comes
    out of Step 4, `tracker` already includes every bit used to produce
    it, and reading it at that point gives a current value.
    """
    for bd in bit_decisions:
        tracker.update(bd.confidence)
        yield bd


# ---------------------------------------------------------------------------
# Building the pipeline
# ---------------------------------------------------------------------------

def build_config(profile: Profile) -> NavtexConfig:
    """The audio/tone settings from a profile, as used by Steps 1-3."""
    return NavtexConfig(
        sample_rate=profile.sample_rate,
        oversample=profile.oversample,
        window_type=profile.window_type,
        mark_freq=profile.mark_freq,
        space_freq=profile.space_freq,
    )


def open_live_source(config: NavtexConfig, device=None) -> Tuple[AudioSource, str]:
    """Opens a live audio input. `device` is an index, a name substring,
    or None for the system default. A numeric string is treated as an
    index. Returns the source and a human-readable description."""
    if device is not None:
        try:
            device = int(device)
        except ValueError:
            pass  # a name substring; sounddevice accepts that too
    source = LiveMicSource(config, device=device)
    description = (f"live audio device {device!r}" if device is not None
                   else "live audio (default device)")
    return source, description


def open_source(profile: Profile, config: NavtexConfig) -> Tuple[AudioSource, str]:
    """Opens the audio source a profile asks for (live device or WAV
    file). Returns the source and a human-readable description."""
    if profile.mode == "live":
        return open_live_source(config, profile.device)
    return FileSource(config, profile.wav_file), f"WAV file {profile.wav_file!r}"


def decode_characters(source: AudioSource, profile: Profile,
                      tracker: SignalStrengthTracker) -> Iterator[str]:
    """The full Step 1-4 pipeline: audio in, decoded characters out.
    Updates `tracker` with every bit decision along the way."""
    config = build_config(profile)
    windower = Windower(config)
    detector = ToneDetector(config)
    bitsync = SoftBitSync(config, loop_gain=profile.loop_gain)
    grouper = SoftCharacterGrouper(
        sync_window=profile.sync_window,
        acquire_threshold=profile.char_acquire_threshold,
        drop_threshold=profile.char_drop_threshold,
        switch_margin=profile.char_switch_margin,
        min_groups_for_acquire=profile.min_groups_for_acquire,
    )
    fec = SoftFecCombiner(
        lock_window=profile.lock_window,
        acquire_threshold=profile.fec_acquire_threshold,
        switch_margin=profile.fec_switch_margin,
        min_samples_for_rate=profile.min_samples_for_rate,
        rate_window=profile.rate_window,
    )
    bit_decisions = tap_bit_decisions(
        bitsync.process_stream(detector.process_stream(windower.frames(source))),
        tracker,
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
    """

    def __init__(self, source: AudioSource, profile: Profile,
                 sinks: Iterable[OutputSink], warn: WarnFn = warn_to_stderr):
        self.source = source
        self.profile = profile
        self.tracker = SignalStrengthTracker(window=profile.signal_strength_window)
        self._sinks = list(sinks)
        self._warn = warn

    def run(self) -> None:
        assembler = LineAssembler(self._sinks, strength=lambda: self.tracker.level,
                                  warn=self._warn)
        try:
            for ch in decode_characters(self.source, self.profile, self.tracker):
                assembler.feed(ch)
        finally:
            assembler.close()
            self.source.close()

    def stop(self) -> None:
        """Asks the session to finish. Returns immediately; run() returns
        shortly afterwards. Safe to call from any thread."""
        self.source.stop()
