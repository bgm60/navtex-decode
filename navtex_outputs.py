"""
NAVTEX Decoder — Line Assembly and Output Sinks
=================================================

Decoded text leaves the pipeline one character at a time. This module
turns that character stream into line events and delivers them to any
number of output "sinks" (console, text log file, and in future a
database or GUI window), so that line-boundary handling, timestamps and
signal-strength readings are worked out once, in one place, rather than
separately by every output.

    decoded characters
            |
            v
      LineAssembler  ----->  ConsoleSink
      (CR/LF rules,  ----->  TextLogSink
       timestamps,   ----->  ... any other OutputSink
       strength)

Line events
------------
Every sink receives the same three kinds of event, in this order for
each line:

    line_start(timestamp, strength)   the first character of a new line
                                      has arrived
    write(text)                       characters, passed through verbatim,
                                      including any CR/LF
    line_end()                        the line is complete

`timestamp` is the UTC time at which the line started, and `strength` is
the 00-99 signal-strength reading at that moment (see
navtex_session.SignalStrengthTracker).

Line boundaries
----------------
CR and LF are two independent CCIR 476 codewords. On a clean signal they
arrive as a CR LF pair, but on a noisy one either can be lost, so a lone
CR or a lone LF also ends a line (every standard text viewer treats both
that way). The rules are:

  - CR ends the line immediately.
  - An LF that directly follows a CR belongs to that same line ending.
    It is still passed to write() (so the text log reproduces the
    original characters exactly), but it does not start or end a line.
  - Any other LF ends the line (starting an empty line first if needed,
    so a genuinely blank line still gets its own timestamp).

Sinks that want the line's text without its terminator (a database, a
GUI) simply ignore CR and LF characters in write(); sinks that reproduce
the raw text (console, log file) write everything they are given.

Error isolation
----------------
Output is a convenience, and must never stop live reception running
unattended. If a sink raises an exception, LineAssembler reports it
through the `warn` callback, stops sending events to that sink for the
rest of the session, and carries on delivering to the others. Sinks with
their own recovery logic (TextLogSink) handle expected errors internally
and normally never reach this fallback.
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional, TextIO

WarnFn = Callable[[str], None]


def warn_to_stderr(message: str) -> None:
    """Default warning handler: print to stderr (keeps the CLI's existing
    behaviour). A GUI would pass its own handler instead."""
    print(message, file=sys.stderr)


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


# ---------------------------------------------------------------------------
# Sink interface
# ---------------------------------------------------------------------------

class OutputSink:
    """Base class for anything that receives decoded text. Override only
    the events you need; the defaults do nothing."""

    def line_start(self, timestamp: datetime.datetime, strength: int) -> None:
        pass

    def write(self, text: str) -> None:
        pass

    def line_end(self) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Line assembler
# ---------------------------------------------------------------------------

class LineAssembler:
    """Converts the decoded character stream into line events for a set
    of sinks. See the module docstring for the line-boundary rules.

    `strength` is a zero-argument callable returning the current 00-99
    reading, read at the start of each line. `clock` returns the current
    UTC time; it is a parameter only so tests can supply a fixed clock.
    """

    def __init__(self, sinks: Iterable[OutputSink], strength: Callable[[], int],
                 clock: Callable[[], datetime.datetime] = utc_now,
                 warn: WarnFn = warn_to_stderr):
        self._sinks: List[OutputSink] = list(sinks)
        self._strength = strength
        self._clock = clock
        self._warn = warn
        self._in_line = False      # a line has started and not yet ended
        self._after_cr = False     # previous character was a CR

    def feed(self, text: str) -> None:
        for ch in text:
            if ch == "\n" and self._after_cr:
                # Second half of a CR LF pair: part of the line ending
                # already reported at the CR.
                self._after_cr = False
                self._emit("write", ch)
                continue
            self._after_cr = False

            if not self._in_line:
                self._emit("line_start", self._clock(), self._strength())
                self._in_line = True

            self._emit("write", ch)

            if ch == "\r" or ch == "\n":
                self._emit("line_end")
                self._in_line = False
                self._after_cr = ch == "\r"

    def close(self) -> None:
        """Ends any unfinished line, then closes every sink."""
        if self._in_line:
            self._emit("line_end")
            self._in_line = False
        for sink in list(self._sinks):
            try:
                sink.close()
            except Exception as e:  # noqa: BLE001 -- see "Error isolation"
                self._warn(f"[output warning] error closing {type(sink).__name__} ({e}) -- ignoring.")
        self._sinks = []

    def _emit(self, event: str, *args) -> None:
        for sink in list(self._sinks):
            try:
                getattr(sink, event)(*args)
            except Exception as e:  # noqa: BLE001 -- see "Error isolation"
                self._warn(f"\n[output warning] {type(sink).__name__} failed ({e}); "
                           "output to it is disabled for the rest of this session.")
                self._sinks.remove(sink)


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

class ConsoleSink(OutputSink):
    """Writes decoded text to stdout exactly as received.

    Python's automatic newline translation is disabled on stdout: the
    decoded text already contains its own CR and LF characters, and on
    Windows the translation would turn every CR LF into CR CR LF,
    producing a spurious blank line before every real one. reconfigure()
    is not supported by every replacement stdout, in which case the
    default behaviour is kept rather than failing.
    """

    def __init__(self, stream: Optional[TextIO] = None):
        self._stream = stream if stream is not None else sys.stdout
        try:
            self._stream.reconfigure(newline="")
        except (AttributeError, ValueError):
            pass

    def write(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()


# ---------------------------------------------------------------------------
# Text log file
# ---------------------------------------------------------------------------

class TextLogSink(OutputSink):
    """Writes decoded text to a UTC-timestamped log file.

    The file is created in `log_dir` (created if needed) as
    navtex_<YYYYMMDD_HHMMSS>Z.txt, named after the time decoding started,
    and begins with a short '#' header. Each line is prefixed with
    "[YYYYMMDD HH:MM:SS SS] ", the UTC time the line started and its
    00-99 signal-strength reading. Text, including CR and LF, is
    otherwise written exactly as decoded.

    Newline translation is disabled (newline=""), for the same reason as
    in ConsoleSink.

    The file is flushed at each CR or LF rather than per character, and
    forced to disk with fsync at most once per `fsync_interval` seconds.
    Per-character flushing over a long session makes a failure far more
    likely on cloud-synced or network drives, which can invalidate an
    open file handle mid-session.

    On a write, flush or fsync error, the file is closed and reopened in
    append mode once, and any continuation of the current line is given a
    fresh prefix. If that also fails, logging to this file is
    disabled for the rest of the session with a warning, while decoding
    and every other output carry on normally.
    """

    def __init__(self, log_dir: str, source_description: str,
                 fsync_interval: float = 3600.0,
                 warn: WarnFn = warn_to_stderr,
                 clock: Callable[[], datetime.datetime] = utc_now):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        start = clock()
        self.path = Path(log_dir) / f"navtex_{start.strftime('%Y%m%d_%H%M%SZ')}.txt"
        self._warn = warn
        self._fsync_interval = fsync_interval
        self._clock = clock
        self._disabled = False
        self._reprefix = False     # file was reopened mid-line; prefix the continuation
        self._last_strength = 0
        self._f = open(self.path, "w", encoding="utf-8", newline="")
        self._f.write("# NAVTEX decode log\n")
        self._f.write(f"# Started: {start.isoformat()}\n")
        self._f.write(f"# Source: {source_description}\n")
        self._f.write("#\n")
        self._f.flush()
        self._last_fsync = time.monotonic()

    def line_start(self, timestamp: datetime.datetime, strength: int) -> None:
        if self._disabled:
            return
        self._last_strength = strength
        self._reprefix = False
        try:
            self._write_prefix(timestamp, strength)
        except OSError as e:
            self._recover_from_error("write", e)

    def write(self, text: str) -> None:
        if self._disabled:
            return
        try:
            if self._reprefix:
                self._reprefix = False
                if text not in ("\r", "\n"):
                    self._write_prefix(self._clock(), self._last_strength)
            self._f.write(text)
            if text.endswith(("\r", "\n")):
                self._f.flush()
                self._maybe_fsync()
        except OSError as e:
            self._recover_from_error("write", e)

    def line_end(self) -> None:
        self._reprefix = False

    def close(self) -> None:
        if self._disabled:
            return
        try:
            self._f.close()
        except OSError as e:
            self._warn(f"[log warning] error closing log file ({e}) -- ignoring, shutting down anyway.")

    def _write_prefix(self, timestamp: datetime.datetime, strength: int) -> None:
        self._f.write(f"[{timestamp.strftime('%Y%m%d %H:%M:%S')} {strength:02d}] ")

    def _maybe_fsync(self) -> None:
        now = time.monotonic()
        if now - self._last_fsync >= self._fsync_interval:
            try:
                os.fsync(self._f.fileno())
                self._last_fsync = now
            except OSError as e:
                self._recover_from_error("fsync", e)

    def _recover_from_error(self, action: str, error: Exception) -> None:
        self._warn(f"\n[log warning] log file {action} failed ({error}); attempting to reopen...")
        try:
            self._f.close()
        except OSError:
            pass
        try:
            self._f = open(self.path, "a", encoding="utf-8", newline="")
            # Text written before the failure may not have reached the
            # file, so anything that continues the current line starts
            # with a fresh prefix rather than running on from whatever
            # was last saved. A fresh handle has had nothing synced yet,
            # so the fsync timer restarts too.
            self._reprefix = True
            self._last_fsync = time.monotonic()
            self._warn("[log warning] log file reopened successfully, continuing.")
        except OSError as reopen_error:
            self._warn(f"[log warning] could not reopen log file ({reopen_error}); "
                       "file logging disabled for the rest of this session "
                       "(other output continues normally).")
            self._disabled = True
