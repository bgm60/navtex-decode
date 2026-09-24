# NAVTEX Decoder — Application Documentation

This document describes what the decoder does, how data flows through each
processing stage, how to configure it with TOML profiles, and what every
configuration parameter controls.

---

## 1. Overview

NAVTEX is a maritime safety broadcast service that transmits text messages
(navigational warnings, weather forecasts, search-and-rescue information)
on MF frequencies such as 518 kHz and 490 kHz. The transmissions use
100-baud frequency-shift keying (FSK) with a 170 Hz shift, carrying
characters coded according to CCIR 476 / SITOR-B (ITU-R Recommendation
M.476-5). Each character is sent twice, a few characters apart, so that a
short fade or burst of interference is less likely to destroy both copies.

The decoder takes demodulated audio from a receiver, either live from a
sound-card input or from a WAV recording, and turns it into decoded text.
The text is streamed to the console as it arrives and can optionally be
written to a timestamped log file.

The application has a single entry point, `navtex_decode.py`, and all of
its settings come from a named profile in a TOML configuration file.

### Files

| File | Role |
|---|---|
| `navtex_decode.py` | Entry point. Loads a profile, builds the pipeline, handles console output and logging. |
| `navtex_config.py` | Loads, type-checks and validates TOML profiles. |
| `navtex_step1_sampling_windowing.py` | Audio sources (live device, WAV file) and the windower that produces analysis frames. |
| `navtex_step2_tone_detection.py` | Measures mark and space tone energy in each frame. |
| `navtex_step3_bit_sync.py` | Bit-decision data type and the shared state of the bit-clock recovery loop. |
| `navtex_step4_character_decode.py` | CCIR 476 code tables, character synchronisation, FEC parity locking and character lookup. |
| `navtex_soft_fec_combine.py` | The live implementations of bit-clock recovery, character grouping and soft-decision FEC combining used by the decoder, plus the top-level decode loop. |
| `navtex.toml.example` | Example configuration file to copy and edit. |

### Requirements

Python 3.11 or later is required, because profiles are read with the
standard-library `tomllib` module. The Python packages used are `numpy` and
`scipy` for signal processing, `soundfile` for reading WAV files, and
`sounddevice` for live audio capture. `sounddevice` is only needed for live
mode, and `soundfile` only for file mode.

---

## 2. Running the decoder

```bash
# Decode using the [my_profile] section of navtex.toml in the current directory
python navtex_decode.py my_profile

# Decode using a profile from a different config file
python navtex_decode.py my_profile --config /path/to/other.toml

# List the audio devices sounddevice can see, then exit
python navtex_decode.py --list-devices
```

In live mode the decoder runs until stopped with Ctrl+C, which closes the
audio device and log file cleanly. In file mode it stops when the end of
the recording is reached.

---

## 3. Data flow

The decoder is a chain of Python generators. Each stage pulls items from
the stage before it, processes them, and yields its own output to the next
stage. Nothing is buffered beyond what each stage needs internally, so
live audio is decoded continuously with only a short delay.

```
 Audio source (live device or WAV file)
        │  float32 mono sample chunks
        ▼
 [1] Windower ──────────────────────────  Frame (windowed samples)
        │  one frame every 1/8 of a bit
        ▼
 [2] ToneDetector ──────────────────────  ToneSample (diff, bit)
        │
        ▼
 [3] SoftBitSync (digital PLL) ─────────  SoftBitDecision (bit, confidence, soft_value)
        │                     │
        │                     └──► SignalStrengthTracker (for log line prefixes)
        ▼
 [4a] SoftCharacterGrouper ─────────────  (7-bit codeword, 7 soft values)
        │
        ▼
      Phasing-burst / phase-change monitor (decode_bit_stream_soft)
        │
        ▼
 [4b] SoftFecCombiner ──────────────────  combined codeword
        │
        ▼
 [4c] Character lookup (LTRS/FIGS) ─────  text characters
        │
        ▼
 Console output  +  optional log file
```

### 3.1 Audio input

Both audio sources produce a stream of mono, 32-bit floating-point sample
chunks at the configured `sample_rate`.

The live source (`LiveMicSource`) opens the selected input device through
`sounddevice` at `sample_rate`, in 20 ms blocks. Audio arrives on a
callback thread and is handed to the decoder through a queue. Any
overflow or underflow status reported by the audio driver is printed to
stderr, and capture carries on.

The file source (`FileSource`) reads the WAV file in blocks of 4096
samples. Multi-channel recordings are mixed down to mono by averaging the
channels, and if the file's native sample rate differs from `sample_rate`
it is resampled with a polyphase filter so that the rest of the pipeline
always sees the same rate.

### 3.2 Stage 1 — Windowing

NAVTEX runs at 100 baud, so each bit lasts 10 ms, which is 480 samples at
48 kHz. The windower cuts the continuous sample stream into overlapping
analysis frames, each exactly one bit period long, and multiplies each
frame by a window function (`window_type`).

The frame length is a trade-off. A longer frame gives finer frequency
resolution, which helps separate the mark and space tones that are only
170 Hz apart, but one bit period is the natural upper limit if each frame
is to describe a single bit. A frame starting at an arbitrary point will
usually straddle two bits, so rather than taking one frame per bit the
windower advances by only a fraction of a bit each time: the hop is
`samples_per_symbol / oversample`, which is 60 samples (1.25 ms) with the
default of 8. Later stages therefore get eight overlapping looks at every
bit, which is what the bit-clock recovery stage needs to find where the
bit boundaries actually are.

The window function tapers the edges of each frame. Without it, energy
from one tone leaks into the detector for the other tone through spectral
sidelobes, which matters because the two tones are so close together.

### 3.3 Stage 2 — Tone detection

For each frame, the tone detector measures the energy at the mark
frequency and at the space frequency. It does this with a single-frequency
discrete Fourier transform at each target frequency, which gives the same
result as the Goertzel algorithm. The sine and cosine reference vectors are
computed once at start-up and each measurement is then a pair of dot
products. Because the target frequencies are used exactly rather than
rounded to the nearest FFT bin, the tones do not need to fall on bin
centres.

From the two energies, the detector produces a `ToneSample` with two
values. The first is a hard decision, `bit`, which is mark (1) if the mark
energy is at least the space energy and space (0) otherwise. The second is
a normalised difference:

```
diff = (mark_energy − space_energy) / (mark_energy + space_energy)
```

`diff` ranges from −1 (pure space) to +1 (pure mark) and does not depend on
the overall signal level, so the decoder needs no absolute threshold and is
unaffected by receiver gain. It is set to zero if both energies are
effectively zero.

### 3.4 Stage 3 — Bit-clock recovery

The receiver's sample clock is not synchronised to the transmitter's bit
clock. At the start of a recording the decoder does not know where bit
boundaries fall, and the two clocks can drift slightly relative to each
other over time. Bit-clock recovery solves both problems with a
first-order (proportional) digital phase-locked loop.

A phase accumulator tracks progress through the current bit. It advances
by `1 / oversample` for every incoming `ToneSample`, and each `ToneSample`
is added to a buffer for the current bit. When the phase reaches 1.0, a bit
boundary has been reached and one bit decision is emitted.

Timing correction comes from transitions. Whenever the hard decision
changes between two consecutive frames, the signal has crossed a genuine
bit boundary somewhere between them. The loop estimates exactly where by
linear interpolation of the point at which `diff` crosses zero, compares
that with the nearest boundary it expected, and moves its phase towards the
observed position by `loop_gain` times the error. CCIR 476 codewords
always contain both marks and spaces, so transitions occur frequently
enough to keep the loop aligned.

At each bit boundary, the buffered `diff` values are integrated (summed)
to produce the bit decision:

- `soft_value` is the signed sum of `diff` over the bit. Its sign gives the
  bit (zero or positive is mark), and its magnitude shows how strongly the
  evidence favoured that decision. This is the value used for soft
  combining in stage 4.
- `bit` is the hard decision taken from the sign of `soft_value`.
- `confidence` is the mean of `|diff|` over the bit, a value from 0 to 1.
  It is used only for the signal-strength reading described in section
  3.8, not for decoding.

Because the loop is proportional only, a constant difference between the
transmitter and receiver clock rates leaves a small, steady timing offset
rather than being corrected completely.

### 3.5 Stage 4a — Character synchronisation

Bits arrive one at a time with nothing to mark where each 7-bit character
begins. The decoder finds the correct alignment from a property of the
CCIR 476 code: every valid codeword has exactly four marks and three
spaces (a weight of 4).

`CharacterSync` keeps the most recent `sync_window` bits. For each of the
seven possible alignments, it splits those bits into 7-bit groups and
calculates the fraction of groups with a weight of 4. At the correct
alignment, real traffic scores close to 100 %. At a wrong alignment, or on
random noise, only about 27 % of groups have a weight of 4 by chance.
Alignments are counted against the total number of bits received since
start-up, so a given alignment keeps the same number as the window slides
forward.

The grouper acquires a lock when three conditions are met:

1. At least `min_groups_for_acquire` complete groups are available for
   scoring.
2. The best alignment's score is at least `char_acquire_threshold`.
3. The best score beats the runner-up by at least `char_switch_margin`.

Once locked, bits are collected into 7-bit codewords starting at the
locked alignment. Each codeword is passed on together with the seven
`soft_value`s of its bits.

After every codeword, the lock is re-checked. If the locked alignment's
score has fallen below `char_drop_threshold`, and another alignment now
beats it by at least `char_switch_margin`, the lock is released and
acquisition starts again. The margin requirement matters because the
repetitive phasing signal sent between messages can score well at more
than one alignment at once; without it the decoder could hop to a wrong
alignment during phasing and corrupt the start of the next message.

### 3.6 Phasing and reset handling

Between messages, and sometimes within them, NAVTEX stations send a
phasing signal: an alternation of the two special codewords `1111000` and
`0110011`. The top-level decode loop (`decode_bit_stream_soft`) watches the
codeword stream for two situations that invalidate the FEC combiner's state:

- **Alignment change.** If the character grouper changes alignment,
  codewords before and after the change are different groupings of the
  bits and cannot be compared, so the FEC combiner is reset.
- **End of a phasing burst.** When at least `phasing_burst_threshold`
  consecutive phasing codewords are followed by a non-phasing codeword, the
  FEC combiner is reset. Phasing content cannot show which slots carry
  first transmissions and which carry repeats, and that relationship can
  change across a burst, so the combiner is made to re-establish it from
  the new message's content.

### 3.7 Stage 4b — FEC combining

SITOR-B sends every character twice. The first transmission (DX) is
followed by four other character slots, and the repeat (RX) comes in the
fifth slot after it. DX and RX slots alternate, so in any stream of
codewords only every second slot is an RX slot whose content matches the
codeword five slots earlier. The five-slot spacing is fixed by the
protocol.

**Parity locking.** Which of the two slot parities (even or odd) holds the
RX transmissions depends on where decoding started, so the combiner has to
work it out. For every codeword, it checks whether it equals the codeword
five slots earlier and records the result against that slot's parity. Each
parity keeps a rolling record of its last `rate_window` results, and its
match rate is only trusted once it has at least `min_samples_for_rate`
results. The correct parity matches almost every time, while the wrong
parity compares unrelated characters and rarely matches.

The combiner locks onto a parity when that parity's match rate is at least
`fec_acquire_threshold` and beats the other parity by at least
`fec_switch_margin`. On locking, it replays the codewords already held in
its history buffer, so characters received while the lock was being
established, often the start of a message, are not lost. While locked, it
keeps tracking both parities, and switches if the other parity becomes
better by at least `fec_switch_margin` while also meeting
`fec_acquire_threshold`.

**Combining.** For each RX slot, the combiner has two copies of the same
character: the DX codeword from five slots earlier and the RX codeword
just received, each with seven soft values. If both are phasing codewords,
nothing is output. Otherwise, the soft values of the two copies are added
bit by bit and a new codeword is formed from the signs of the sums. This is
soft-decision diversity combining. A bit that was weak or wrong in one copy
can be outvoted by a strong, correct bit in the other, so the combination
can recover a character that neither copy could deliver on its own.

The combined codeword is then handled in one of three ways:

1. **Weight 4:** it is accepted and passed to character lookup.
2. **Weight 3 or 5:** it is probably one bit away from a valid codeword.
   The decoder tries flipping the least certain bits (smallest combined
   magnitude first) and accepts the first flip that produces a valid
   codeword.
3. **Otherwise:** the decoder falls back to comparing the two original
   copies directly, which is the classic hard-decision method:
   - both copies valid and identical: that character is used;
   - both copies valid but different: `!` is output;
   - only one copy valid: that copy is used;
   - neither copy valid: `~` is output.

The fallback means soft combining can only improve on hard-decision
combining, never make it worse. This matters most when interference is
strong and wrong rather than weak, where adding soft values could be
misled.

### 3.8 Stage 4c — Character lookup

Accepted codewords are translated to text using the CCIR 476 tables in
ITU-R M.476-5. As in Baudot/ITA2, the same 26 codewords mean letters or
figures depending on the current shift state. The `LTRS` and `FIGS`
control codewords switch between the two tables and produce no output. The
decoder starts in letters mode.

The other control codewords are handled as follows: `CR` produces a
carriage return, `LF` a line feed, `SP` a space, and `BLANK` (idle filler)
produces nothing. A figures-mode codeword with no assigned figure produces
`~`.

In the output, `~` marks a character that could not be decoded, and `!`
marks a character where both copies were individually valid but
disagreed. A long, unbroken run of `!` usually means the FEC combiner is
temporarily locked to the wrong parity.

### 3.9 Output and logging

Decoded characters are written to the console as soon as they are
produced. CR and LF are separate codewords in CCIR 476, so the decoded text
already contains its own line endings. The console and the log file both
disable Python's automatic newline translation, which would otherwise
double up line endings on Windows.

If `log_dir` is set, a log file named `navtex_<YYYYMMDD_HHMMSS>Z.txt` is
created in that directory (the directory is created if needed), named
after the UTC time at which decoding started. The file begins with a short
header recording the start time and audio source. Each decoded line is
then prefixed with its UTC start time and a two-digit signal-strength
reading:

```
[20260924 13:50:49 85] ZCZC SA02
```

The signal-strength reading (00–99) is the average of the per-bit
`confidence` over the last `signal_strength_window` bits, scaled to 0–99.
It is a relative measure of how cleanly the tones are being separated, not
a calibrated signal level. Any of a CR, an LF, or a CR LF pair counts as
the end of a line, so a line ending damaged by noise still starts a new
prefixed line.

The log file is flushed at the end of every line and forced to disk once an
hour. If writing fails, for example because a network or cloud-synced
drive drops the file handle, the decoder reopens the file once and carries
on. If that also fails, file logging is switched off for the rest of the
session and a warning is printed, but decoding and console output
continue.

---

## 4. Profile configuration

### 4.1 The configuration file

All settings live in a TOML file. By default the decoder reads
`navtex.toml` in the current directory; `--config` selects another file.
`navtex.toml` is excluded from version control because it normally
contains machine-specific device numbers and paths. `navtex.toml.example`
is the template to copy.

A file contains one or more profiles, each a TOML table (a `[section]`).
The profile name is given on the command line. Profiles typically
represent different real-world setups, for example live reception from a
particular receiver, decoding recordings, or different signal conditions.

```toml
[weak_signal_live]
mode = "live"
device = 4
log_dir = "D:/Radio/Logs/518kHz/"
loop_gain = 0.05
char_drop_threshold = 0.46
min_groups_for_acquire = 25

[recording]
mode = "file"
wav_file = "recordings/hamburg.wav"
log_dir = "logs/"
```

### 4.2 How profiles are read

Each profile is self-contained. There is no shared defaults section and no
inheritance between profiles; any key a profile leaves out takes the
built-in default listed in section 5.

Profiles are checked strictly when loaded, and any problem stops the
decoder with a clear error message rather than a traceback:

- A missing file, a TOML syntax error, or an unknown profile name is
  reported, and for an unknown profile the available profile names are
  listed.
- A key that is not a recognised parameter is an error, so a misspelt key
  cannot silently fall back to its default. The error lists the valid
  keys.
- Values must be of the right type. Whole numbers are accepted for decimal
  parameters (`loop_gain = 0` is fine), but `true`/`false` are never
  accepted as numbers or the other way round.
- `mode` must be `"file"` or `"live"`, and file mode requires `wav_file`.
- `sync_window` must be greater than `min_groups_for_acquire × 7`,
  otherwise character synchronisation could never collect enough groups
  to acquire.
- `rate_window` must be at least `min_samples_for_rate`, otherwise the FEC
  combiner could never collect enough results to lock.

---

## 5. Configuration parameters

### 5.1 Input source and logging

| Parameter | Default | Description |
|---|---|---|
| `mode` | `"live"` | `"live"` to capture from an audio device, `"file"` to decode a WAV file. |
| `wav_file` | none | Path to the WAV file. Required in file mode, ignored in live mode. |
| `device` | system default | Live-mode input device, given either as an index number or as part of the device name. Run with `--list-devices` to see what is available. |
| `log_dir` | none | Directory for log files. Leave it out to disable logging. |
| `signal_strength_window` | `100` | Number of bits averaged for the signal-strength reading in log line prefixes. 100 bits is one second; a larger value gives a steadier reading that responds more slowly to fades. It has no effect on decoding. |

### 5.2 Audio and tone detection

These describe the audio coming from the receiver. They are not tuning
knobs, and they must match the receiver setup for decoding to work.

| Parameter | Default | Description |
|---|---|---|
| `sample_rate` | `48000` | Sample rate in Hz. In live mode, the device is opened at this rate. In file mode, recordings at other rates are resampled to it. It also sets the frame length (`sample_rate / 100` samples per bit). |
| `mark_freq` | `1785.0` | Audio frequency in Hz of the mark (binary 1) tone. |
| `space_freq` | `1615.0` | Audio frequency in Hz of the space (binary 0) tone. |
| `oversample` | `8` | Number of analysis frames per bit (see section 3.2). |
| `window_type` | `"hamming"` | Window function applied to each frame. Any name accepted by `scipy.signal.get_window` can be used, such as `"hamming"`, `"hann"` or `"blackman"`. |

**Mark and space frequencies.** The audio frequencies of the NAVTEX tones
depend on the receiver's tuning and demodulator settings, not on the
NAVTEX standard, and which tone is the higher one also depends on the
receiver. The defaults suit a receiver producing tones centred on 1700 Hz
with mark as the higher tone. If the tones are wrong or swapped, the
decoder runs but produces no text or only garbage, so these values should
be checked whenever the receiver, its software or its tuning offset
changes.

**Oversample.** More frames per bit give the bit-clock loop finer timing
information, at a proportional cost in processing. Fewer frames reduce
processing but make timing coarser. This value affects every later stage,
so it is best left at 8 unless there is a specific reason to change it.

**Window type.** Windows with lower sidelobes, such as Blackman, reduce the
leakage of each tone into the other tone's detector, at the cost of
slightly broader frequency selectivity. Because the two NAVTEX tones are
only 170 Hz apart, crosstalk between them is often the more important
factor.

### 5.3 Bit-clock recovery

| Parameter | Default | Description |
|---|---|---|
| `loop_gain` | `0.05` | Fraction of each measured timing error that the loop corrects. |

A higher gain makes the loop react faster, so it pulls into alignment
sooner and follows timing changes more closely, but noise that creates
false transitions also moves it more. A lower gain makes the loop steadier
and more resistant to noise, but slower to align at the start and to
recover after a fade. The default of 0.05 is deliberately low and favours
stability on weak, noisy signals. For a strong, clean signal, a higher
value (the conservative profile in `navtex.toml.example` uses 0.19) trades
some of that noise resistance for faster alignment.

### 5.4 Character synchronisation

| Parameter | Default | Description |
|---|---|---|
| `sync_window` | `250` | Number of recent bits used to score the seven possible alignments. |
| `min_groups_for_acquire` | `25` | Minimum number of complete 7-bit groups that must be scored before an alignment can be acquired. |
| `char_acquire_threshold` | `0.6` | Minimum fraction of weight-4 groups required to acquire an alignment. |
| `char_drop_threshold` | `0.46` | If the locked alignment's score falls below this, the decoder starts checking whether a clearly better alignment exists. |
| `char_switch_margin` | `0.2` | How much better one alignment must score than another to acquire it, or to abandon the current lock for it. |

**`sync_window`.** A larger window averages over more bits, so scores are
steadier but react more slowly to a genuine change in alignment. A smaller
window reacts faster but is noisier. It must stay above
`min_groups_for_acquire × 7` bits, and some margin above that is advisable.

**`min_groups_for_acquire`.** This is the main control over how quickly the
decoder acquires character sync, both at start-up and after losing lock.
Small samples can reach a high score purely by chance, particularly during
phasing or noise, so lowering this value speeds acquisition but raises the
risk of locking onto noise. Values much below about 20 noticeably erode
that protection. Raising it makes acquisition slower and more certain.

**`char_acquire_threshold`.** Random data produces weight-4 groups about
27 % of the time. Lowering the threshold makes acquisition easier on a weak
signal but reduces the separation from that chance level; raising it makes
acquisition slower and more conservative.

**`char_drop_threshold`.** A higher value makes the decoder more willing to
consider re-acquiring once a lock starts to struggle. A lower value makes
it hold on to an existing lock through poorer conditions. Either way, a
switch only happens if another alignment is better by `char_switch_margin`,
so this setting decides when to look for an alternative, not whether to
switch.

**`char_switch_margin`.** This protects against switching between
alignments that merely tie, which happens during repetitive phasing
content. A smaller margin allows faster moves to a genuinely better
alignment but brings back that risk; a larger one makes the lock stickier.

### 5.5 FEC parity locking and phasing

| Parameter | Default | Description |
|---|---|---|
| `fec_acquire_threshold` | `0.6` | Minimum DX/RX match rate for a parity to be locked, or switched to. |
| `fec_switch_margin` | `0.2` | How much better one parity's match rate must be than the other's to lock or switch to it. |
| `min_samples_for_rate` | `5` | Minimum number of DX/RX comparisons before a parity's match rate is trusted. |
| `rate_window` | `30` | Number of recent comparisons per parity used to calculate its match rate. |
| `lock_window` | `15` | Together with `rate_window`, sets the length of the codeword history buffer. |
| `phasing_burst_threshold` | `4` | Number of consecutive phasing codewords treated as a phasing burst, after which the FEC combiner is reset. |

**`fec_acquire_threshold` and `fec_switch_margin`.** These work like their
character-synchronisation counterparts, applied to the choice of slot
parity. Lowering either makes the combiner lock sooner, with more risk of
locking to the wrong parity; raising either makes it more cautious.

**`min_samples_for_rate`.** This is the main control over how quickly FEC
locks at the start of a message. On a clean signal the correct parity
matches almost immediately, so a low value works well. On a weak signal,
bit errors make early match rates less reliable, so lowering it further
increases the risk of a wrong lock.

**`rate_window`.** A longer window gives steadier match rates but responds
more slowly to a genuine change of parity; a shorter one responds faster
but is noisier. It must be at least `min_samples_for_rate`.

**`lock_window`.** The history buffer holds `5 + max(lock_window,
rate_window)` codewords. It determines how many earlier codewords can be
replayed when a lock is achieved. With the default values, `rate_window` is
the larger, so `lock_window` has no effect unless it is set above
`rate_window`.

**`phasing_burst_threshold`.** A lower value recognises shorter phasing
bursts, which can help when a fading signal only lets a few phasing
codewords through cleanly, but it also makes FEC re-acquire after shorter
stretches of filler. A higher value requires a longer clean phasing run
before the reset happens, which may not occur on a poor signal.

### 5.6 Parameters that should be changed together

**`min_groups_for_acquire` and `sync_window`.** The window must hold more
than `min_groups_for_acquire × 7` bits. With the defaults (25 groups, 175
bits, in a 250-bit window) there is a reasonable margin; reducing
`sync_window` without also reducing `min_groups_for_acquire` can stop
acquisition altogether.

**`rate_window`, `lock_window` and `min_samples_for_rate`.** `rate_window`
must be at least `min_samples_for_rate`, and the larger of `rate_window` and
`lock_window` sets the history buffer length. Changing one can change the
effective behaviour of the others.

**Acquire thresholds and switch margins.** In both character
synchronisation and FEC locking, a candidate must reach the acquire
threshold and also beat the alternative by the switch margin. Relaxing
only one of the pair may make little difference if the other is the
limiting condition.

### 5.7 Fixed values

Some values are set by the NAVTEX / SITOR-B protocol and are deliberately
not configurable: the 100-baud symbol rate, the 7-bit codeword length, the
five-slot DX/RX spacing, and the codeword tables themselves.
