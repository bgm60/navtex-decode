#!/usr/bin/env python3
# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Reference capture for the Rust command-line program: time formats, text logs
and the SQLite database
================================================================================

Usage (run from anywhere; the decoder modules are found in the repo root,
and rust/test-vectors/<set>/lines.csv must already exist: see dump_lines.py):

    python rust/tools/dump_logs.py

Files written
-------------
rust/test-vectors/time_cases.csv
    Timestamps with the text Python makes from each:
        micros, file_stamp, line_stamp, iso, db_stamp
    micros is microseconds since 1970-01-01 00:00:00 UTC. The others are
    strftime('%Y%m%d_%H%M%S'), strftime('%Y%m%d %H:%M:%S'), isoformat() and
    strftime('%Y-%m-%d %H:%M:%S') (the database's time format).

rust/test-vectors/<set>/textlog.escaped.txt
    The text log the Python TextLogSink writes when it is sent the line events
    in lines.csv, using fixed times (below) instead of the clock. CR, LF and
    backslash are escaped so that git cannot alter line endings:
        backslash -> \\\\     CR -> \\r     LF -> \\n followed by a real newline
    The Rust test makes the same escapes of its own file and compares.

rust/test-vectors/<set>/dblog.tsv
    The rows the Python SqliteLogSink stores when it is sent the same line
    events with the same fixed times, one per line:
        id <TAB> utc_timestamp <TAB> signal_strength <TAB> text_output
    with backslash and tab escaped in the text as \\\\ and \\t.

rust/test-vectors/db_schema.txt
    What SQLite reports about the database the Python sink creates:
    PRAGMA table_info and index_list/index_info, as plain text, plus the
    journal mode.

The fixed times: decoding "started" at 2026-10-18 17:36:36.25 UTC, and line
number k (counting from 0) starts 1 s plus k * 7.5 s later.
"""

import calendar
import datetime
import random
import sqlite3
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
VECTORS = HERE.parent / "test-vectors"
sys.path.insert(0, str(ROOT))

from navtex_outputs import SqliteLogSink, TextLogSink                 # noqa: E402

START_MICROS = calendar.timegm((2026, 10, 18, 17, 36, 36)) * 1_000_000 + 250_000
SOURCE = "test source"
EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def at(micros: int) -> datetime.datetime:
    return EPOCH + datetime.timedelta(microseconds=micros)


def line_time(k: int) -> datetime.datetime:
    return at(START_MICROS + 1_000_000 + k * 7_500_000)


def escape(text: str) -> str:
    return (text.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n\n"))


def write_time_cases(out: Path) -> int:
    rng = random.Random(12345)
    edges = [
        (1970, 1, 1, 0, 0, 0, 0), (1999, 12, 31, 23, 59, 59, 999999),
        (2000, 2, 29, 12, 0, 0, 0), (2000, 3, 1, 0, 0, 0, 1),
        (2024, 2, 29, 23, 59, 59, 0), (2026, 12, 31, 23, 59, 59, 500000),
        (2038, 1, 19, 3, 14, 8, 0), (2100, 2, 28, 23, 59, 59, 999999),
        (2100, 3, 1, 0, 0, 0, 0), (2099, 12, 31, 23, 59, 59, 999999),
        (2026, 10, 18, 17, 36, 36, 250000), (2026, 10, 25, 1, 30, 0, 0),
    ]
    values = [calendar.timegm(e[:6]) * 1_000_000 + e[6] for e in edges]
    top = calendar.timegm((2100, 1, 1, 0, 0, 0)) * 1_000_000
    values += [rng.randrange(top) for _ in range(3000)]
    values += [rng.randrange(top // 1_000_000) * 1_000_000 for _ in range(500)]   # whole seconds
    lines = ["micros,file_stamp,line_stamp,iso,db_stamp"]
    for v in values:
        t = at(v)
        lines.append(f"{v},{t.strftime('%Y%m%d_%H%M%S')},{t.strftime('%Y%m%d %H:%M:%S')},{t.isoformat()},{t.strftime('%Y-%m-%d %H:%M:%S')}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    return len(values)


def write_text_log(set_dir: Path) -> int:
    events = (set_dir / "lines.csv").read_text(encoding="utf-8").splitlines()
    with tempfile.TemporaryDirectory() as tmp:
        sink = TextLogSink(tmp, SOURCE, clock=lambda: at(START_MICROS))
        line = -1
        for row in events:
            if row.startswith("S,"):
                line += 1
                sink.line_start(line_time(line), int(row[2:]))
            elif row.startswith("W,"):
                sink.write(chr(int(row[2:])))
            elif row == "E":
                sink.line_end()
            else:
                raise ValueError(f"{set_dir.name}: unexpected row {row!r}")
        sink.close()
        path = sink.path
        assert path.name == "navtex_20261018_173636Z.txt", path.name
        text = path.read_bytes().decode("utf-8")
    (set_dir / "textlog.escaped.txt").write_text(escape(text), encoding="utf-8", newline="")
    return line + 1


def escape_tsv(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\t", "\\t")


def replay(events, sink) -> int:
    """Sends line events (rows of lines.csv) to an output, using the fixed times."""
    line = -1
    for row in events:
        if row.startswith("S,"):
            line += 1
            sink.line_start(line_time(line), int(row[2:]))
        elif row.startswith("W,"):
            sink.write(chr(int(row[2:])))
        elif row == "E":
            sink.line_end()
        else:
            raise ValueError(f"unexpected row {row!r}")
    return line + 1


def describe_schema(conn: sqlite3.Connection) -> list:
    """The table and index layout as text, in a form that does not depend on
    how the CREATE statements happened to be spaced."""
    out = []
    for row in conn.execute("PRAGMA table_info(navtex_log)"):
        out.append("column " + " ".join(str(v) for v in row))
    for _seq, name, unique, origin, partial in conn.execute("PRAGMA index_list(navtex_log)"):
        columns = [r[2] for r in conn.execute(f"PRAGMA index_info({name})")]
        out.append(f"index {name} unique={unique} origin={origin} partial={partial} columns={','.join(columns)}")
    out.append("journal_mode " + conn.execute("PRAGMA journal_mode").fetchone()[0])
    return out


def write_database_reference(set_dir: Path):
    events = (set_dir / "lines.csv").read_text(encoding="utf-8").splitlines()
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "log.db"
        sink = SqliteLogSink(str(db_path))
        replay(events, sink)
        sink.close()
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT id, utc_timestamp, signal_strength, text_output "
                                "FROM navtex_log ORDER BY id").fetchall()
            schema = describe_schema(conn)
        finally:
            conn.close()
    text = "".join(f"{i}\t{ts}\t{strength}\t{escape_tsv(t)}\n" for i, ts, strength, t in rows)
    (set_dir / "dblog.tsv").write_text(text, encoding="utf-8", newline="")
    return len(rows), schema


def main():
    n = write_time_cases(VECTORS / "time_cases.csv")
    print(f"time_cases.csv: {n} timestamps")
    schemas = set()
    for lines_csv in sorted(VECTORS.glob("*/lines.csv")):
        count = write_text_log(lines_csv.parent)
        rows, schema = write_database_reference(lines_csv.parent)
        schemas.add("\n".join(schema))
        print(f"{lines_csv.parent.name}: text log with {count} lines, database with {rows} rows")
    assert len(schemas) == 1, "the sets disagree about the database layout"
    (VECTORS / "db_schema.txt").write_text(schemas.pop() + "\n", encoding="utf-8", newline="")
    print("db_schema.txt written")


if __name__ == "__main__":
    main()
