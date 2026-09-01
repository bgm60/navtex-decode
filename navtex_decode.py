"""
NAVTEX 100-baud FSK Decoder — Live Audio / File Decode with Optional Logging
=============================================================================

Runs the full Step 1-4 pipeline (sampling -> tone detection -> bit-clock
recovery -> character decode) against either a live audio input device or
a WAV file, streaming decoded text to the console as it arrives, and
optionally logging it to a UTC-timestamped text file.

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

    # Decode using a specific config file
    python navtex_decode.py my_profile --config /path/to/myconfig.toml

Each profile's `mode` key ("file" or "live") determines whether it reads
a WAV file (`wav_file`) or a live input device (`device`, optional);
`log_dir` (optional, any mode) enables logging to a directory.

Log files are named navtex_<UTC timestamp>.txt, e.g.
navtex_20260813_154210Z.txt -- timestamped by when decoding started, not
per-message. Ctrl+C stops live decoding cleanly and closes the log file
and audio device.

Each logged line is prefixed "[YYYYMMDD HH:MM:SS] SS " where SS is a
two-digit (00-99) relative signal strength reading -- a rolling average of
Step 3's per-bit confidence at the moment that line started, not a raw
per-character value (see SignalStrengthTracker below).

On Windows, both the log file and console output disable Python's default
text-mode newline translation (see make_log_file's newline="" and run()'s
sys.stdout.reconfigure(newline="")). Decoded text already contains its own
literal '\\r'/'\\n' -- CR and LF are independent CCIR 476 codewords -- so
without this, Windows would translate every '\\n' to '\\r\\n' and turn an
already-correct '\\r\\n' pair into '\\r\\r\\n', showing up as a spurious
blank line before every real line.
"""

from __future__ import annotations

# import external modules and packages.

import argparse
import datetime
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Optional, TextIO

from navtex_config import ConfigError, Profile, load_profile
from navtex_step1_sampling_windowing import (
    AudioSource,
    FileSource,
    LiveMicSource,
    NavtexConfig,
    Windower,
)
from navtex_step2_tone_detection import ToneDetector
from navtex_soft_fec_combine import SoftBitSync as BitSync
from navtex_soft_fec_combine import SoftCharacterGrouper, SoftFecCombiner
from navtex_soft_fec_combine import decode_bit_stream_soft as decode_bit_stream

# List all available audio devices (both input and output).

def list_devices() -> None:
    try:
        import sounddevice as sd
    except (ImportError, OSError):
        print("sounddevice is not installed or its native library is missing.")
        print("Install with: pip install sounddevice")
        return
    print(sd.query_devices())

# Used as a  "signal strength" metric for logging.
# It is actually a bit of a misnomer: it is not a raw power reading, but rather a rolling calculation of bit confidence.
# This means it sufferd from the 'cliff edge' effect of the Step 3 integration window.

class SignalStrengthTracker:
    """Maintains a rolling average of Step 3's per-bit confidence
    (BitDecision.confidence -- gain-independent, already in [0, 1], and
    explicitly documented there as "a useful signal-quality metric for
    later logging") and exposes it as a two-digit 00-99 reading.

    Deliberately NOT built on BitDecision.mean_total_power: that field is
    raw receiver-gain-dependent power, explicitly documented as only
    meaningful as a relative measure within one recording made at a fixed
    gain/AGC setting -- a poor fit for a number meant to be read directly
    off a log line. confidence, by contrast, is already gain-normalized.

    Averaged over a trailing window rather than reported per-bit: a
    single bit's confidence is noisy (it dips near any transition purely
    from where the integration window happens to sit relative to it --
    see Step 3's docstring), so a raw per-bit value would flicker
    distractingly rather than read as a stable "signal strength". Default
    window of 100 bits is ~1s at 100 baud -- long enough to smooth that
    per-bit jitter, short enough to still track a real fade within a
    couple of lines.
    """

    def __init__(self, window: int = 100):
        self._values: Deque[float] = deque(maxlen=window)

    def update(self, confidence: float) -> None:
        self._values.append(confidence)

    @property
    def level(self) -> int:
        """Current reading, 0-99. 0 (not None/blank) until any bits have
        arrived, so the very first log line -- before a full window has
        accumulated -- still gets a sensible number rather than a gap."""
        if not self._values:
            return 0
        avg = sum(self._values) / len(self._values)
        return max(0, min(99, round(avg * 99)))


def tap_bit_decisions(bit_decisions, tracker: SignalStrengthTracker):
    """Passes a BitDecision stream through unchanged, updating `tracker`
    with each bit's confidence along the way.

    This is the only hook into the pipeline for signal-strength logging;
    Steps 3/4 themselves are untouched. Because generators are pull-based,
    by the time decode_bit_stream yields a character downstream, `tracker`
    already reflects every bit consumed to produce it -- so reading
    tracker.level at that point (or shortly after, at the next line
    boundary) is a legitimate "current" reading, not a stale one.
    """
    for bd in bit_decisions:
        tracker.update(bd.confidence)
        yield bd


class TimestampedLineWriter:
    """Wraps a text file so each line gets a UTC timestamp prefix, added
    right as that line starts.

    Text arrives one character at a time from the decoder, so this can't
    prepend a timestamp after the fact -- it watches for the character
    immediately following a line-ending run and writes the prefix right
    there. The header block (see make_log_file) is written directly to
    the underlying file before this wrapper is created, so it's
    unaffected.

    Line boundary = a '\\r\\n' pair, a lone '\\r', or a lone '\\n' -- CR and
    LF are decoded as two fully independent CCIR 476 codewords (see
    navtex_step4_character_decode.py), so on a clean signal they always
    arrive paired, but nothing guarantees that on a noisy one; a lone
    '\\r' with no matching '\\n' (or vice versa) is a real weak-signal
    failure mode, and every standard text viewer treats a bare '\\r' as
    its own line break regardless.

    A '\\r' immediately followed by '\\n' is treated as ONE boundary (so an
    ordinary line ending doesn't get double-prefixed), but that's the
    ONLY thing that gets collapsed -- two separate, consecutive boundaries
    (e.g. '\\r\\n\\r\\n', a genuinely blank line between two lines of real
    content) each still get their own prefix. Collapsing every run
    unconditionally is tempting but wrong: it would also erase the
    timestamp on any real blank line, which is exactly a line "containing
    only a CR, LF, or CRLF" -- the previous version of this class did
    exactly that, silently, and lost its timestamp for having no other
    content to attach it to.

    Recognising a '\\r\\n' pair needs one character of lookahead, which a
    character-at-a-time writer doesn't have yet when the '\\r' arrives --
    handled by deferring: seeing '\\r' doesn't immediately decide whether
    a boundary has completed; that's only resolved once the following
    character is known to be '\\n' (pair) or something else (the '\\r' was
    already a complete boundary on its own).

    Each prefix is "[YYYYMMDD HH:MM:SS] SS ", where SS is a two-digit
    00-99 relative signal strength reading pulled live from `tracker`
    (see SignalStrengthTracker) at the moment that line starts -- not
    baked in when the writer was constructed, so it tracks fades/recovery
    across a long session.

    Flushes only at line boundaries, not after every character. Only
    flushing per-line matters more than it looks: a live session writing
    for hours generates an enormous number of tiny flush operations at
    per-character granularity, and this is exactly what preceded a real
    crash during an 8-hour overnight session logging to a Google Drive
    path (`G:\\My Drive\\...`) -- cloud-sync virtual filesystems can
    invalidate an open file handle mid-session, and hammering one with a
    flush per character makes hitting that far more likely than
    necessary.

    Also resilient to the failure itself: logging is a convenience, not
    something that should be allowed to take down live reception running
    unattended overnight. On a write/flush error, this closes and
    reopens the same file (append mode) once and retries; if that also
    fails, file logging is disabled for the rest of the session (with a
    one-time warning to stderr) while decoding and console output
    continue completely normally.
    """

    def __init__(self, path: Path, f: TextIO, tracker: SignalStrengthTracker):
        self._path = path
        self._f = f
        self._tracker = tracker
        self._need_prefix = True   # next char written -- content or a new boundary -- needs a fresh prefix first
        self._pending_cr = False   # just wrote '\r'; still waiting to see if '\n' follows to decide if it's a pair
        self._disabled = False

    def write(self, s: str) -> None:
        if self._disabled:
            return
        try:
            for ch in s:
                if self._pending_cr:
                    self._pending_cr = False
                    if ch == "\n":
                        # Completes the '\r\n' pair as a single boundary.
                        self._f.write(ch)
                        self._need_prefix = True
                        self._f.flush()
                        continue
                    # The '\r' was already a complete boundary on its own
                    # (not followed by '\n') -- fall through and let `ch`
                    # be handled normally as whatever comes next.
                    self._need_prefix = True

                if self._need_prefix:
                    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d %H:%M:%S")
                    self._f.write(f"[{ts} {self._tracker.level:02d}] ")
                    self._need_prefix = False

                self._f.write(ch)

                if ch == "\r":
                    self._pending_cr = True
                    self._f.flush()
                elif ch == "\n":
                    self._need_prefix = True
                    self._f.flush()
        except OSError as e:
            self._recover_from_error("write", e)

    def flush(self) -> None:
        if self._disabled:
            return
        try:
            self._f.flush()
        except OSError as e:
            self._recover_from_error("flush", e)

    def close(self) -> None:
        if self._disabled:
            return
        try:
            self._f.close()
        except OSError as e:
            print(f"[log warning] error closing log file ({e}) -- ignoring, shutting down anyway.",
                  file=sys.stderr)

    def _recover_from_error(self, action: str, error: Exception) -> None:
        print(f"\n[log warning] log file {action} failed ({error}); attempting to reopen...",
              file=sys.stderr)
        try:
            self._f.close()
        except OSError:
            pass
        try:
            self._f = open(self._path, "a", encoding="utf-8", newline="")
            # Discard any in-flight '\r' lookahead across the error --
            # worst case a genuine '\r\n' pair interrupted mid-write gets
            # logged as two boundaries instead of one, a harmless
            # cosmetic edge case next to actually losing the session.
            self._need_prefix = True
            self._pending_cr = False
            print("[log warning] log file reopened successfully, continuing.", file=sys.stderr)
        except OSError as reopen_error:
            print(f"[log warning] could not reopen log file ({reopen_error}); "
                  "file logging disabled for the rest of this session "
                  "(console output continues normally).", file=sys.stderr)
            self._disabled = True


def make_log_file(log_dir: str, source_description: str,
                   tracker: SignalStrengthTracker) -> TimestampedLineWriter:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    start = datetime.datetime.now(datetime.timezone.utc)
    ts = start.strftime("%Y%m%d_%H%M%SZ")
    path = Path(log_dir) / f"navtex_{ts}.txt"
    # newline="" disables Python's automatic newline translation. Without
    # it, on Windows, text-mode writing translates every '\n' to '\r\n' --
    # but the decoded NAVTEX text already contains '\r\n' line endings of
    # its own, so that translation turns them into '\r\r\n', which shows
    # up as a spurious extra blank line before every real line. This bug
    # is Windows-specific (Linux's line separator is already '\n', so the
    # translation is a no-op there) and doesn't show up in development
    # done on a non-Windows machine.
    f = open(path, "w", encoding="utf-8", newline="")
    f.write(f"# NAVTEX decode log\n")
    f.write(f"# Started: {start.isoformat()}\n")
    f.write(f"# Source: {source_description}\n")
    f.write("#\n")
    f.flush()
    print(f"Logging decoded text to: {path}")
    return TimestampedLineWriter(path, f, tracker)

# This is the main entry point for the decoding pipeline.
# It sets up the processing chain and handles logging and output.

def run(source: AudioSource, profile: Profile, source_description: str) -> None:
    # Same reasoning as make_log_file's newline="" -- decoded text already
    # contains its own literal '\r'/'\n' (CR and LF are independent CCIR
    # 476 codewords), so Python's default text-mode translation on
    # Windows (every '\n' -> '\r\n') would double up an already-correct
    # '\r\n' pair into '\r\r\n' on-screen, same bug as the log file had,
    # just invisible in most terminals rather than fixed. reconfigure()
    # requires Python 3.7+; sys.stdout can in rare cases (e.g. output
    # redirected through something that replaces it) not support
    # reconfigure, so this degrades gracefully rather than crashing --
    # console output would just fall back to Windows' default translation

    # in that case, exactly as before this change.
    try:
        sys.stdout.reconfigure(newline="")
    except (AttributeError, ValueError):
        pass
    
    # Costruct a NavtexConfig object from the profile settings.

    config = NavtexConfig(
        sample_rate=profile.sample_rate,
        oversample=profile.oversample,
        window_type=profile.window_type,
        mark_freq=profile.mark_freq,
        space_freq=profile.space_freq,
    )

    # Initislize the selected windowing profile to protect against spectral leakage and improve tone detection.

    windower = Windower(config)

    detector = ToneDetector(config)
    bitsync = BitSync(config, loop_gain=profile.loop_gain)

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

    tracker = SignalStrengthTracker(window=profile.signal_strength_window)
    log_file = make_log_file(profile.log_dir, source_description, tracker) if profile.log_dir else None

    # Tap the bit stream here (before Step 4 consumes it) so `tracker`
    # stays current for every log line, however the pipeline downstream
    # groups bits into characters.
    bit_decisions = tap_bit_decisions(
        bitsync.process_stream(detector.process_stream(windower.frames(source))),
        tracker,
    )

    try:
        for ch in decode_bit_stream(bit_decisions, grouper=grouper, fec=fec,
                                     phasing_burst_threshold=profile.phasing_burst_threshold):
            sys.stdout.write(ch)
            sys.stdout.flush()
            if log_file is not None:
                log_file.write(ch)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        if log_file is not None:
            log_file.close()
        close = getattr(source, "close", None)
        if close is not None:
            close()


def main() -> None:

    # Set-up argparse to parse command-line arguments.

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profile", nargs="?",
                         help="Name of the [profile] section to load from the config file")
    parser.add_argument("--config", default="navtex.toml",
                         help="Path to the TOML config file (default: navtex.toml)")
    parser.add_argument("--list-devices", action="store_true",
                         help="List available audio input devices and exit "
                              "(standalone diagnostic action -- ignores --config/profile)")
    args = parser.parse_args()

    # Print the list of available audio input devices and exit if requested.

    if args.list_devices:
        list_devices()
        return
    
    # Load the specified profile from the config file, or error if not provided.

    if not args.profile:
        parser.error("Provide a profile name (or use --list-devices)")
        return

    try:
        profile = load_profile(args.config, args.profile)
    except ConfigError as e:
        parser.error(str(e))
        return

    # Build a configuratuion objectfrom the profile settings and command line arguments.

    config = NavtexConfig(
        sample_rate=profile.sample_rate,
        oversample=profile.oversample,
        window_type=profile.window_type,
        mark_freq=profile.mark_freq,
        space_freq=profile.space_freq,
    )

    # Select the audio source based on the profile's mode (live or file).

    if profile.mode == "live":
        device = profile.device
        if device is not None:
            try:
                device = int(device)
            except ValueError:
                pass  # leave as a name-substring string; sounddevice accepts that too
        source: AudioSource = LiveMicSource(config, device=device)
        description = f"live audio device {device!r}" if device is not None else "live audio (default device)"
        print(f"Listening on {description}... (Ctrl+C to stop)")
    else:
        source = FileSource(config, profile.wav_file)
        description = f"WAV file {profile.wav_file!r}"
        print(f"Decoding file: {profile.wav_file}")

    # Run the decoding pipeline with the selected source and profile.
    # Returns when the source is exhausted (file) or interrupted by Ctrl+C (live).

    run(source, profile, description)

if __name__ == "__main__":
    main()
