# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

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
       timestamps,   ----->  SqliteLogSink
       strength)     ----->  ... any other OutputSink

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
import sqlite3
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


# ---------------------------------------------------------------------------
# SQLite database
# ---------------------------------------------------------------------------

class SqliteLogSink(OutputSink):
    """Writes each decoded line as a row in an SQLite database.

    Table `navtex_log`, created if it does not already exist:

        id               INTEGER PRIMARY KEY   row number, in arrival order
        utc_timestamp    TEXT                  'YYYY-MM-DD HH:MM:SS', UTC time
                                               the line started (SQLite's own
                                               date/time format)
        signal_strength  INTEGER               00-99 reading at line start
        text_output      TEXT                  the line exactly as decoded,
                                               without its CR/LF

    An index on utc_timestamp keeps time-range queries fast as the table
    grows.

    Rows are appended, so one database accumulates lines across any
    number of sessions. Lines that are empty or contain only spaces are
    not stored. Each row is committed as soon as its line ends, so
    nothing already written is lost if the program or computer stops
    unexpectedly; an unfinished last line is saved when the session
    closes.

    The database uses WAL journal mode, which lets other programs (or a
    future GUI) read it while the decoder is writing. WAL needs the
    database to be on a local disk: on a network or cloud-synced drive
    SQLite may refuse WAL (a warning is printed and the default mode is
    used) or, worse, behave unreliably, so a local path is strongly
    recommended.

    On a database error, the connection is closed and reopened once and
    the failed row retried. If that also fails, database logging is
    disabled for the rest of the session with a warning, while decoding
    and every other output carry on normally.

    The connection may be used from a thread other than the one that
    created the sink (a GUI creates sinks on its own thread and decodes
    on a worker thread); it is only ever used by one thread at a time.
    """

    TABLE = "navtex_log"

    def __init__(self, db_file: str, warn: WarnFn = warn_to_stderr,
                 busy_timeout: float = 10.0):
        self.path = Path(db_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._warn = warn
        self._busy_timeout = busy_timeout
        self._disabled = False
        self._timestamp = ""
        self._strength = 0
        self._text: List[str] = []
        self._conn = self._connect()

    def line_start(self, timestamp: datetime.datetime, strength: int) -> None:
        self._timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        self._strength = strength
        self._text = []

    def write(self, text: str) -> None:
        self._text.append(text.replace("\r", "").replace("\n", ""))

    def line_end(self) -> None:
        text = "".join(self._text)
        self._text = []
        if self._disabled or not text.strip():
            return
        row = (self._timestamp, self._strength, text)
        try:
            self._insert(row)
        except sqlite3.Error as e:
            self._recover_from_error(e, row)

    def close(self) -> None:
        if self._disabled:
            return
        try:
            self._conn.close()
        except sqlite3.Error as e:
            self._warn(f"[database warning] error closing database ({e}) -- ignoring, shutting down anyway.")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=self._busy_timeout,
                               check_same_thread=False)
        try:
            mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if mode.lower() != "wal":
                self._warn(f"[database warning] {self.path} could not use WAL mode "
                           f"(using {mode!r}); other programs may be blocked from "
                           "reading it while decoding. Is it on a network drive?")
            conn.execute(f"""CREATE TABLE IF NOT EXISTS {self.TABLE} (
                                 id              INTEGER PRIMARY KEY,
                                 utc_timestamp   TEXT    NOT NULL,
                                 signal_strength INTEGER NOT NULL,
                                 text_output     TEXT    NOT NULL)""")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {self.TABLE}_utc_timestamp "
                         f"ON {self.TABLE} (utc_timestamp)")
            conn.commit()
        except sqlite3.Error:
            conn.close()
            raise
        return conn

    def _insert(self, row: tuple) -> None:
        with self._conn:   # commits on success, rolls back on error
            self._conn.execute(
                f"INSERT INTO {self.TABLE} (utc_timestamp, signal_strength, text_output) "
                "VALUES (?, ?, ?)", row)

    def _recover_from_error(self, error: Exception, row: tuple) -> None:
        self._warn(f"\n[database warning] database write failed ({error}); attempting to reconnect...")
        try:
            self._conn.close()
        except sqlite3.Error:
            pass
        try:
            self._conn = self._connect()
            self._insert(row)
            self._warn("[database warning] database reconnected successfully, continuing.")
        except sqlite3.Error as retry_error:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._warn(f"[database warning] could not reconnect to database ({retry_error}); "
                       "database logging disabled for the rest of this session "
                       "(other output continues normally).")
            self._disabled = True
