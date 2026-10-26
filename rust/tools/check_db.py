#!/usr/bin/env python3
# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Checks that a database written by the Rust program can be used by the
Python program, and that it has the layout the Python program creates.

Usage:   python rust/tools/check_db.py path/to/log.db

It (1) compares the table and index layout with rust/test-vectors/db_schema.txt,
(2) lets the Python SqliteLogSink append one line, and (3) reads everything
back, checking that the earlier rows are untouched and the new one is last.
A line is added to the database, so use a copy if you want to keep it as it is.
Exits with an error message if anything differs.
"""

import datetime
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from dump_logs import describe_schema                       # noqa: E402
from navtex_outputs import SqliteLogSink                    # noqa: E402


def main(argv):
    if len(argv) != 2:
        sys.exit(__doc__)
    path = Path(argv[1])
    if not path.is_file():
        sys.exit(f"{path}: no such file")
    expected = (HERE.parent / "test-vectors" / "db_schema.txt").read_text(encoding="utf-8").splitlines()

    with sqlite3.connect(path) as conn:
        before = conn.execute("SELECT id, utc_timestamp, signal_strength, text_output "
                              "FROM navtex_log ORDER BY id").fetchall()
        layout = describe_schema(conn)
    if layout != expected:
        sys.exit("The layout differs from the Python program's:\n  got      "
                 + "\n           ".join(layout) + "\n  expected " + "\n           ".join(expected))

    sink = SqliteLogSink(str(path))
    stamp = datetime.datetime(2030, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    sink.line_start(stamp, 42)
    sink.write("PYTHON WAS HERE")
    sink.write("\r\n")
    sink.line_end()
    sink.close()

    with sqlite3.connect(path) as conn:
        after = conn.execute("SELECT id, utc_timestamp, signal_strength, text_output "
                             "FROM navtex_log ORDER BY id").fetchall()
    if after[:len(before)] != before:
        sys.exit("Earlier rows changed")
    if len(after) != len(before) + 1 or after[-1][1:] != ("2030-01-02 03:04:05", 42, "PYTHON WAS HERE"):
        sys.exit(f"The appended row is wrong: {after[-1] if after else None}")
    if before and after[-1][0] != before[-1][0] + 1:
        sys.exit("The new row's id does not continue the numbering")
    print(f"OK: {len(before)} existing rows kept, layout matches, Python appended row {after[-1][0]}")


if __name__ == "__main__":
    main(sys.argv)
