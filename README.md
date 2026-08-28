# NAVTEX Decoder

A from-scratch NAVTEX (100-baud FSK, CCIR 476 / SITOR-B) decoder, built as a
four-stage pipeline that turns raw audio (live mic input or a WAV file)
into decoded text.

## Pipeline

```
[1] Sampling & windowing        navtex_step1_sampling_windowing.py
[2] Mark/space tone detection   navtex_step2_tone_detection.py
[3] Bit-clock recovery          navtex_step3_bit_sync.py
[4] Character decode + FEC      navtex_step4_character_decode.py
```

1. **Sampling & windowing** — captures audio and turns the continuous
   sample stream into overlapping analysis frames (one window per symbol,
   hopped forward 8x per bit by default).
2. **Tone detection** — measures mark/space tone energy per frame via a
   vectorized Goertzel-equivalent DFT correlation.
3. **Bit-clock recovery** — a proportional (Type-I) digital PLL locks onto
   symbol timing and produces one bit decision per transmitted symbol,
   with a per-bit confidence score.
4. **Character decode** — recovers 7-bit CCIR 476 codeword sync
   (constant-weight-4 property), combines each character's two FEC
   transmissions (time diversity), and maps codewords to text via the
   ITU-R M.476-5 table with LTRS/FIGS shift tracking.

## Tools

| Script | Purpose |
|---|---|
| `navtex_live_decode.py` | Run the full pipeline against a live audio device or a WAV file; stream decoded text to the console, optionally logging to a timestamped file. |
| `calibrate_tone_frequencies.py` | Determine the correct mark/space frequencies for your specific receiver by searching for whichever pair makes the weight-4 codeword property show up reliably in a real recording. |
| `navtex_confidence_plot.py` | Plot per-bit confidence and signal power over time for one or more recordings, for comparing/diagnosing reception quality. |
| `test_step4_roundtrip.py` | Round-trip test: encodes known text with a reference SITOR-B FEC encoder and verifies Step 4 decodes it back correctly. |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`sounddevice` is only needed for `--live` mic input; decoding a WAV file
works without it.

## Usage

```bash
# List available audio input devices
python navtex_live_decode.py --list-devices

# Decode live from the default input device
python navtex_live_decode.py --live

# Decode live from a specific device, with logging
python navtex_live_decode.py --live --device 2 --log-dir logs/

# Decode a WAV file, with logging
python navtex_live_decode.py recording.wav --log-dir logs/

# Don't know your receiver's tone frequencies? Calibrate first:
python calibrate_tone_frequencies.py recording.wav [--start SEC] [--duration SEC]

# Inspect signal quality across one or more recordings:
python navtex_confidence_plot.py recording1.wav [recording2.wav ...] [-o OUTDIR]
```

## Testing

```bash
python test_step4_roundtrip.py
```

## Docs

- [`TUNING_REFERENCE.md`](TUNING_REFERENCE.md) — notes on tuning/calibrating
  the pipeline's parameters against real recordings.
- [`docs/status-dashboard.html`](docs/status-dashboard.html) — a project
  status snapshot.

## Credits / provenance

- The CCIR 476 codeword table and mark/space conventions were transcribed
  from **ITU-R Recommendation M.476-5**, Annex 1 (the primary standard).
  The document itself isn't redistributed in this repo (ITU copyright); see
  the official recommendation at
  <https://www.itu.int/rec/R-REC-M.476-5-199510-I>.
- The SITOR-B FEC interleave structure and DX/RX comparison lag used in
  `navtex_step4_character_decode.py`'s `FecCombiner` were verified by hand
  against **Baltic Lab's** open-source, field-tested Arduino CCIR476
  library (confirmed working against a commercial NAV4 NAVTEX receiver).
  No code was copied from that project — the transmit behavior was traced
  and independently reimplemented — but credit where it's due:
  <https://baltic-lab.com/2022/07/sitor-b-navtex-test-signal-generation/>.

## Notes

- Standard mark/space tone assumptions (e.g. "1615/1785 Hz, mark low")
  don't always hold — real receiver hardware varies, and a wrong guess
  fails silently (the decoder runs, but output is garbage). Use
  `calibrate_tone_frequencies.py` against a real recording before trusting
  decoded output from a new setup.
- Log files from `navtex_live_decode.py` are UTC-timestamped
  (`navtex_<UTC timestamp>.txt`) and excluded from version control via
  `.gitignore`, along with `.wav` recordings and generated `.png` plots.
