# Where the reference files come from

Everything in `test-vectors/` was captured from the original Python decoder
(the repository `bgm60/navtex-decode`) by scripts in that repository's
`rust/tools/` folder. The Python repository is tagged `rust-oracle-final` at the
commit these files match, so the scripts and the Python decoder that produced
them can always be recovered:

```
git clone https://github.com/bgm60/navtex-decode
cd navtex-decode
git checkout rust-oracle-final
```

| Script | What it captured |
|---|---|
| `dump_reference.py` | Windowing, tone detection, bit decisions, the lock and its characters for each recording (`window.csv`, `tone.csv`, `bits.csv`, `soft_lock_chars.csv`, `decoded_*.txt`, ...), and `combine_cases.csv`. |
| `dump_lines.py` | Line events and signal-quality readings (`lines.csv`, `quality.csv`). |
| `dump_logs.py` | Text-log and database contents, the time formats, and the database layout (`textlog.escaped.txt`, `dblog.tsv`, `time_cases.csv`, `db_schema.txt`). |
| `dump_wavs.py` | WAV files in every supported format with the samples the Python reader gives (`wav_formats/`). |
| `dump_config.py` | The Python settings loader's results and error messages for 38 snippets and the example profiles (`config_cases.txt`, `navtex.example.toml`). |

Two more scripts in that folder check the finished program against Python:
`check_db.py` (a database written by Rust can be used by Python) and
`compare_with_python.py` (decodes the same recordings with both programs and
compares the text, the text log and the database rows).

The reference files are plain data and do not need to be regenerated unless
the Python decoder itself changes.
