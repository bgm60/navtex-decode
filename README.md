# NAVTEX Decoder

A from-scratch software decoder for NAVTEX, the maritime safety broadcast
service. It takes demodulated audio from a receiver, either live from a
sound-card input or from a WAV recording, and turns the 100-baud FSK signal
(CCIR 476 / SITOR-B) into decoded text on the console, with optional
timestamped logging to a file.

The decoder uses soft-decision combining of each character's two FEC
transmissions, self-aligning bit and character synchronisation, and
automatic recovery after fades and inter-message phasing, so it keeps
producing usable text on weak and fading signals.

## How it works

The decoder is a streaming pipeline of four stages. Each stage pulls data
from the one before it, so live audio is decoded continuously with only a
short delay.

| Stage | Module | What it does |
|---|---|---|
| 1. Sampling and windowing | `navtex_step1_sampling_windowing.py` | Reads audio from a live device or WAV file and cuts it into overlapping, windowed frames, one bit period long, eight per bit. |
| 2. Tone detection | `navtex_step2_tone_detection.py` | Measures mark and space tone energy in each frame (Goertzel-equivalent) and produces a gain-independent mark/space difference. |
| 3. Bit-clock recovery | `navtex_step3_bit_sync.py`, `navtex_soft_fec_combine.py` | A digital phase-locked loop aligns to the transmitter's bit timing and emits one decision per bit, with a signed soft value and a confidence score. |
| 4. Character decode | `navtex_step4_character_decode.py`, `navtex_soft_fec_combine.py` | Finds 7-bit character alignment from the CCIR 476 weight-4 property, locks onto the DX/RX interleave, soft-combines the two copies of each character, and translates codewords to text. |

`navtex_step4_character_decode.py` holds the CCIR 476 tables and the core
synchronisation and FEC logic. `navtex_soft_fec_combine.py` extends stages
3 and 4 to carry soft values through the pipeline and holds the top-level
decode loop that the application runs.

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for a full description of each
stage and of every configuration parameter.

## Project files

| File | Purpose |
|---|---|
| `navtex_decode.py` | Application entry point: loads a profile, runs the pipeline, writes console output and log files. |
| `navtex_config.py` | Loads and validates TOML configuration profiles. |
| `navtex_step1_sampling_windowing.py` | Audio sources and the frame windower. |
| `navtex_step2_tone_detection.py` | Mark/space tone detector. |
| `navtex_step3_bit_sync.py` | Bit-decision type and bit-clock loop state. |
| `navtex_step4_character_decode.py` | CCIR 476 tables, character sync, FEC parity locking and character lookup. |
| `navtex_soft_fec_combine.py` | Soft-value bit sync, character grouping and FEC combining, and the decode loop. |
| `navtex.toml.example` | Example configuration file. |
| `DOCUMENTATION.md` | Detailed application and configuration documentation. |
| `requirements.txt` | Python package dependencies. |

## Requirements

- Python 3.11 or later (profiles are read with the standard-library
  `tomllib`).
- `numpy` and `scipy`.
- `sounddevice` for live audio input.
- `soundfile` for WAV file input.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Configuration

All settings, including the audio source, logging directory and decoder
tuning parameters, are held in named profiles in a TOML file. Copy the
example to create your own:

```bash
cp navtex.toml.example navtex.toml
```

Each `[section]` is a self-contained profile. Any key left out uses the
built-in default. For example:

```toml
[live_518]
mode = "live"
device = 4                  # index or part of the device name
log_dir = "logs/518kHz/"

[recording]
mode = "file"
wav_file = "recordings/example.wav"
```

Profiles are validated strictly when loaded: unknown keys, wrong value
types and inconsistent parameter combinations are reported as errors
rather than silently ignored. `navtex.toml` is excluded from version
control, since it usually contains machine-specific device numbers and
paths.

## Usage

```bash
# List the available audio input devices
python navtex_decode.py --list-devices

# Run a profile from navtex.toml in the current directory
python navtex_decode.py live_518

# Run a profile from another config file
python navtex_decode.py recording --config /path/to/other.toml
```

Live decoding runs until stopped with Ctrl+C. File decoding stops at the
end of the recording.

## Output

Decoded text is streamed to the console as it arrives. In the text, `~`
marks a character that could not be decoded, and `!` marks a character
whose two transmitted copies were each valid but disagreed.

If `log_dir` is set, the text is also written to
`navtex_<YYYYMMDD_HHMMSS>Z.txt` in that directory, named after the UTC
start time. Each line is prefixed with its UTC time and a relative signal
strength reading from 00 to 99:

```
[20260924 13:50:49 85] ZCZC SA02
```

## Receiver setup

The audio frequencies of the mark and space tones depend on the receiver's
tuning and demodulator settings, not on the NAVTEX standard, and so does
which tone is the higher one. The defaults (`mark_freq = 1785`,
`space_freq = 1615`) suit a receiver producing tones centred on 1700 Hz
with mark as the higher tone. If the tones are wrong or swapped, the
decoder runs but produces nothing useful, so set `mark_freq` and
`space_freq` in your profile to match your receiver.

## Credits and provenance

- The CCIR 476 codeword tables were transcribed from **ITU-R
  Recommendation M.476-5**, Annex 1. The document is not redistributed
  here (ITU copyright); see
  <https://www.itu.int/rec/R-REC-M.476-5-199510-I>.
- The SITOR-B FEC interleave structure and DX/RX comparison lag were
  verified by hand against **Baltic Lab's** open-source, field-tested
  Arduino CCIR476 library. No code was copied; the transmit behaviour was
  traced and independently reimplemented:
  <https://baltic-lab.com/2022/07/sitor-b-navtex-test-signal-generation/>.
