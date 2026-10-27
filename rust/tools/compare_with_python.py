#!/usr/bin/env python3
# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
End-to-end comparison of the Rust program with the Python program
====================================================================

Decodes the same recordings with both programs, each writing a text log and
a SQLite database, and checks that everything they produce is the same:

  * the text printed on the console,
  * the text log (apart from the clock times in it, and its header),
  * the database rows (signal strength and text of every line),

and that the Python program can use a database written by the Rust program
(rust/tools/check_db.py). It also shows how long each program took.

Usage (from anywhere):

    python rust/tools/compare_with_python.py                 # the usual set
    python rust/tools/compare_with_python.py a.wav b.wav     # your own files

    --rust-bin PATH    the navtex-cli executable
                       (default: rust/target/release/navtex-cli[.exe])
    --python-dir DIR   the folder holding navtex_decode.py (default: the
                       folder above rust/)
    --keep             keep the working folder and print where it is

With no files given it uses every recording in test-wavs/ and the input.wav of
every reference set in rust/test-vectors/. Build the Rust program first:

    cd rust && cargo build --release

Exit status: 0 if every recording agrees, 1 if any differs.
"""

import argparse
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUST_DIR = HERE.parent
DEFAULT_PYTHON_DIR = RUST_DIR.parent

# [YYYYMMDD HH:MM:SS QQ] at the start of a text-log line: keep only the quality.
PREFIX = re.compile(r"\[\d{8} \d\d:\d\d:\d\d (\d\d)\] ")
# The three status lines both programs print before the decoded text.
STATUS = ("Decoding file: ", "Logging decoded text to: ", "Logging decoded lines to database: ")
DB_TIME = re.compile(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d")


def toml_literal(path: Path) -> str:
    """A TOML literal string (single quotes), so Windows backslashes are left alone."""
    return "'" + str(path) + "'"


def write_config(path: Path, wav: Path, log_dir: Path, db: Path) -> None:
    path.write_text(
        "[p]\n"
        'mode = "file"\n'
        f"wav_file = {toml_literal(wav)}\n"
        f"log_dir = {toml_literal(log_dir)}\n"
        f"db_file = {toml_literal(db)}\n",
        encoding="utf-8",
    )


def timed(command, cwd):
    started = time.perf_counter()
    done = subprocess.run(command, cwd=cwd, capture_output=True)
    return done, time.perf_counter() - started


def console_text(done) -> str:
    """What the program printed, without its three status lines."""
    text = done.stdout.decode("utf-8", errors="replace")
    lines = text.split("\n")
    skipped = 0
    while skipped < len(lines) and lines[skipped].startswith(STATUS):
        skipped += 1
    return "\n".join(lines[skipped:])


def text_log(log_dir: Path):
    files = sorted(log_dir.glob("navtex_*.txt"))
    if len(files) != 1:
        return None, f"{len(files)} text logs in {log_dir.name} (expected 1)"
    raw = files[0].read_bytes().decode("utf-8")
    parts = raw.split("\n", 4)
    if len(parts) < 5:
        return None, "the text log is shorter than its header"
    return PREFIX.sub(r"[\1] ", parts[4]), None


def db_rows(db: Path):
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT signal_strength, text_output FROM navtex_log ORDER BY id").fetchall(), \
               [r[0] for r in conn.execute("SELECT utc_timestamp FROM navtex_log ORDER BY id")]


def first_difference(a, b) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"first difference at item {i}: python {x!r}, rust {y!r}"
    return f"one is longer: python has {len(a)}, rust has {len(b)}"


def compare_one(wav: Path, work: Path, python_dir: Path, rust_bin: Path):
    """Returns (problems, python_seconds, rust_seconds)."""
    problems = []
    runs = {}
    for name in ("py", "rs"):
        folder = work / name
        (folder / "logs").mkdir(parents=True)
        write_config(folder / "config.toml", wav, folder / "logs", folder / "log.db")
    done_py, t_py = timed([sys.executable, str(python_dir / "navtex_decode.py"), "p",
                           "--config", str(work / "py" / "config.toml")], python_dir)
    done_rs, t_rs = timed([str(rust_bin), "p", "--config", str(work / "rs" / "config.toml")], work)
    for name, done in (("python", done_py), ("rust", done_rs)):
        if done.returncode != 0:
            problems.append(f"{name} exited with status {done.returncode}: "
                            f"{done.stderr.decode('utf-8', errors='replace').strip()[:300]}")
        runs[name] = done
    if problems:
        return problems, t_py, t_rs

    if console_text(done_py) != console_text(done_rs):
        problems.append("console text differs: " + first_difference(console_text(done_py), console_text(done_rs)))

    log_py, why_py = text_log(work / "py" / "logs")
    log_rs, why_rs = text_log(work / "rs" / "logs")
    if why_py or why_rs:
        problems.append(f"text log: python: {why_py or 'ok'}; rust: {why_rs or 'ok'}")
    elif log_py != log_rs:
        problems.append("text log differs: " + first_difference(log_py, log_rs))

    rows_py, times_py = db_rows(work / "py" / "log.db")
    rows_rs, times_rs = db_rows(work / "rs" / "log.db")
    if rows_py != rows_rs:
        problems.append("database rows differ: " + first_difference(rows_py, rows_rs))
    if not all(DB_TIME.fullmatch(t) for t in times_rs):
        problems.append("a Rust database time is not in the form YYYY-MM-DD HH:MM:SS")

    copy = work / "rs_copy.db"
    shutil.copyfile(work / "rs" / "log.db", copy)
    check = subprocess.run([sys.executable, str(HERE / "check_db.py"), str(copy)], capture_output=True, cwd=python_dir)
    if check.returncode != 0:
        problems.append("check_db.py: " + (check.stderr or check.stdout).decode("utf-8", errors="replace").strip()[:300])
    return problems, t_py, t_rs


def default_recordings(python_dir: Path):
    found = sorted((python_dir / "test-wavs").glob("*.wav"))
    found += sorted((RUST_DIR / "test-vectors").glob("*/input.wav"))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="recordings to compare (default: the usual set)")
    exe = "navtex-cli.exe" if sys.platform == "win32" else "navtex-cli"
    parser.add_argument("--rust-bin", type=Path, default=RUST_DIR / "target" / "release" / exe)
    parser.add_argument("--python-dir", type=Path, default=DEFAULT_PYTHON_DIR)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    if not args.rust_bin.is_file():
        sys.exit(f"{args.rust_bin}: not found. Build it first:  cd rust && cargo build --release")
    if not (args.python_dir / "navtex_decode.py").is_file():
        sys.exit(f"{args.python_dir}: navtex_decode.py not found (use --python-dir)")
    recordings = [f.resolve() for f in args.files] or default_recordings(args.python_dir)
    if not recordings:
        sys.exit("no recordings found")

    root = Path(tempfile.mkdtemp(prefix="navtex-compare-"))
    failures = 0
    total_py = total_rs = 0.0
    try:
        for number, wav in enumerate(recordings):
            if not wav.is_file():
                print(f"FAIL  {wav}: no such file")
                failures += 1
                continue
            work = root / str(number)
            work.mkdir()
            problems, t_py, t_rs = compare_one(wav, work, args.python_dir, args.rust_bin.resolve())
            total_py += t_py
            total_rs += t_rs
            label = f"{wav.parent.name}/{wav.name}" if wav.name == "input.wav" else wav.name
            print(f"{'FAIL' if problems else 'ok  '}  {label}   python {t_py:5.1f} s   rust {t_rs:5.2f} s")
            for problem in problems:
                print(f"        {problem}")
            failures += bool(problems)
    finally:
        if args.keep:
            print(f"\nworking files kept in {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)

    print(f"\n{len(recordings) - failures} of {len(recordings)} recordings agree"
          f"   (python {total_py:.1f} s, rust {total_rs:.1f} s in total, including start-up)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
