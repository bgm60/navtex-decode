# NAVTEX decoder (Rust)

A command-line decoder for NAVTEX, the maritime safety broadcast service. It
takes audio from a receiver, either live from a sound-card input or from a
WAV recording, and turns the 100-baud FSK signal (CCIR 476 / SITOR-B) into
text on the console, with optional timestamped logging to a text file and/or a
SQLite database.

It uses soft-decision combining of each character's two transmissions,
self-aligning bit and character synchronisation, and a weak-signal lock that
keeps producing usable text through fades and noise.

There is no graphical interface. Everything is controlled by a settings file
and a few command-line options.

## Building

You need the Rust toolchain (<https://rustup.rs>). Then, from this folder:

```
cargo build --release
```

The program is `target/release/navtex-cli` (`navtex-cli.exe` on Windows). To
install it somewhere on your path instead:

```
cargo install --path navtex-cli
```

SQLite is built into the program, so nothing else needs installing on Windows.
On Linux the sound library needs the ALSA headers (`libasound2-dev` on Debian
and Ubuntu, `alsa-lib-devel` on Fedora).

## Quick start

```
navtex-cli --list-devices                   # the audio inputs, numbered
navtex-cli --wav recording.wav              # decode a recording to the console
navtex-cli live --config navtex.toml        # decode live, using the profile "live"
navtex-cli live --config navtex.toml --level   # check the input level first
```

`navtex.example.toml` is a ready-made settings file: copy it to `navtex.toml`
and edit it. Ctrl+C stops decoding cleanly: the current line is ended, the
logs are closed and the program exits with status 0.

## Command line

```
navtex-cli [OPTIONS] [PROFILE]
```

| Option | Meaning |
|---|---|
| `PROFILE` | The profile (a `[section]`) to use from the settings file. |
| `--config PATH` | The settings file. Default `navtex.toml`. |
| `--wav FILE` | Decode this recording. It sets `mode = "file"`; with no profile, the defaults are used. |
| `-q`, `--quiet` | No status messages and no decoded text on the console. Needs `log_dir` or `db_file`. Errors and warnings are still shown. |
| `--no-quiet` | Show messages and text even if the profile says `quiet = true`. |
| `--list-devices` | List the audio input devices and exit. |
| `--level` | Show a live input-level meter and the signal quality instead of the text; nothing is logged. Works with live audio or `--wav`. Not allowed with quiet. |
| `-h`, `--help`, `-V`, `--version` | Help and version. |

Options override the profile.

**Exit status:** 0 on success (including a stop with Ctrl+C); 1 if the audio or
an output cannot be read or written, or the audio input fails (a recording that
ends early is decoded as far as it goes, then reported); 2 for a mistake on the
command line or in the settings.

## Settings file

The file is TOML. Each `[section]` is a self-contained profile; a key that is
left out takes its default. Text in single quotes is taken literally, which is
the easiest way to write a Windows path.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `"live"` or `"file"` | `"live"` | Where the audio comes from. |
| `wav_file` | text | none | The recording to decode. Required when `mode = "file"`. |
| `device` | number or text | system default | Live input: its number in `--list-devices`, or part of its name (capitals do not matter). If part of a name matches several devices, give more of it. |
| `log_dir` | text | none | Folder for the text log (created if missing). |
| `db_file` | text | none | SQLite database for the decoded lines. |
| `quiet` | true/false | false | As `--quiet`. |
| `sample_rate` | number | 48000 | Only 48000 is supported. |
| `centre_freq` | number | 1700.0 | Hz, the midpoint of the mark and space tones. They are always 170 Hz apart, so mark = centre + 85 and space = centre - 85. |
| `tones_inverted` | true/false | false | True if the mark tone is the lower one. |
| `loop_gain` | number | 0.05 | Gain of the bit-clock loop. |
| `signal_quality_window` | number | 100 | How many bit decisions are averaged for the 00-99 quality reading. |

### Keys from the Python program's settings files

An existing Python `navtex.toml` can be used as it is. These keys are accepted
but have no effect, and each gives a warning: `weak_signal_lock = true`,
`engine`, `window_type = "hamming"`, `oversample = 8`, `sync_window`,
`char_acquire_threshold`, `char_drop_threshold`, `char_switch_margin`,
`min_groups_for_acquire`, `fec_acquire_threshold`, `fec_switch_margin`,
`min_samples_for_rate`, `rate_window`, `lock_window` and
`phasing_burst_threshold`. The old name `signal_strength_window` still works
for `signal_quality_window`, with a warning.

These ask for something this program does not do, so they are refused:
`weak_signal_lock = false`, any `window_type` other than `"hamming"`, any
`oversample` other than 8, and any `sample_rate` other than 48000.
`mark_freq` and `space_freq` are no longer used; set `centre_freq` and
`tones_inverted` instead.

## Audio input

Only 48000 Hz is supported: a recording or a device at any other rate is an
error (resampling is not implemented). WAV files may hold 8, 16, 24 or 32-bit
integer or 32-bit float samples; several channels are averaged to one.

Live input uses the system's audio library (WASAPI on Windows, which is the
one it is used and tested with; ALSA on Linux and CoreAudio on macOS should work
too), so the device list holds only what the system offers for recording. If a device cannot record at 48000 Hz, `--list-devices`
says so; set the device's format to 48000 Hz in the system's sound settings.
If the device disappears while running, the logs are closed and the program
exits with status 1.

Use `--level` to set the receiver's output level. The bar should sit about
half way, and `CLIP` must not appear. When a signal is present the decoder
shows `Locked` and a quality reading.

## What is written

**Console.** The decoded text, exactly as received, preceded by status lines
(`Decoding file:`, `Listening on ...`, `Logging ...`). The 00-99 signal quality
is not shown on the console; it is in the logs.

**Text log.** One file per run, named `navtex_YYYYMMDD_HHMMSSZ.txt` after the
UTC start time. It begins with a short header of `#` lines. Each line of text is
prefixed `[YYYYMMDD HH:MM:SS QQ] `, the UTC time the line started and its signal
quality 00-99. The file is flushed at the end of every line and forced to disk
once an hour. If a write fails, the file is reopened once; if that fails too,
file logging is switched off with a warning and decoding continues.

**Database.** Table `navtex_log` with the columns `id`, `utc_timestamp`
(`YYYY-MM-DD HH:MM:SS`, UTC), `signal_strength` (the 00-99 quality, under its
old column name) and `text_output`, with an index on the timestamp. Lines with
no text are not stored. The database is opened in WAL mode with a ten-second
busy timeout; an existing database is appended to. If the connection fails it
is reopened once, then database logging is switched off with a warning.

**Signal quality** is the average confidence of the last 100 bit decisions
(`signal_quality_window`), scaled so that an average of 0.50 or less reads 00
and 0.92 or more reads 99.

## Tests

```
cargo test --release
```

The tests compare the Rust code with reference files in `test-vectors/`, which
were captured from the original Python decoder (the front end, the bit
decisions, the lock, the characters, the lines, the quality readings, the
text-log and database contents, the settings loader and its messages, and the
WAV reader), and test the program itself by running it. The reference files are
data, and nothing needs Python to run the tests.

## Layout

| Folder | Contents |
|---|---|
| `navtex-core/` | The decoder: front end, tone detection, bit clock, soft lock, character set. No dependencies. |
| `navtex-cli/` | The program: settings, quality meter, line assembly, console, text log, database, WAV and live audio, level meter. |
| `test-vectors/` | Reference recordings and the expected results. |

## Licence

GNU General Public Licence, version 3 or later. See `LICENSE`.
