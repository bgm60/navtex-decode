# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
NAVTEX Decoder — Synthetic Signal Generator (test harness, part 1)
====================================================================

Generates a known NAVTEX / SITOR-B transmission as audio, and degrades it
with a configurable radio channel, so the decoder can be measured against
ground truth. Part 2 (navtex_bench.py) feeds the audio through the real
decoding pipeline and scores the result.

    text --> encode_text --> interleave --> modulate --> apply_channel
              (CCIR 476)     (SITOR-B)       (FSK)        (noise, fades...)

What is generated
------------------
* CCIR 476 codewords for the text, using the decoder's own code tables
  (inverted), with LTRS/FIGS shifts inserted wherever the character set
  changes.
* The SITOR-B / NAVTEX time-diversity interleave: DX and RX slots strictly
  alternate; a DX slot carries a new character and the RX slot five slots
  later repeats it, so four other slots sit in between.
* A phasing preamble before the message, and idle slots after it, so the
  last characters' repeats are sent. Idle slots alternate PHASING_2 (DX
  position) and PHASING_1 (RX position), a period-14-bit pattern. The
  decoder ignores which phasing code is which, so this assignment is an
  assumption that does not affect decoding.
* Continuous-phase 100-baud FSK at the requested mark/space frequencies,
  with optional transmitter clock error (ppm) and tone mistuning (Hz).

Channel model (all optional, all seeded for repeatability)
-----------------------------------------------------------
* White Gaussian noise, band-limited to a receiver-like audio passband
  (300-3000 Hz by default), at a stated SNR.
* Flat Rayleigh fading with a given Doppler bandwidth.
* Impulsive noise (atmospheric crashes): Poisson bursts of decaying noise.
* A continuous-wave interferer at a chosen audio frequency.

SNR definition
---------------
SNR is the average power of the unfaded FSK signal divided by the noise
power in a reference bandwidth (2500 Hz by default). For orthogonal
non-coherent FSK this gives

    Eb/N0 = SNR * ref_bw / baud        (+13.98 dB at 2500 Hz and 100 baud)

and an ideal-detector raw bit error rate of 0.5 * exp(-Eb/N0 / 2); see
theory_raw_ber(). The decoder's Hamming-windowed detector is expected to
sit a little worse than that ideal, never better, which makes the formula
a useful sanity check on the harness itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, sosfilt

from navtex_step4_character_decode import CONTROL, FIGURES, LETTERS, PHASING_1, PHASING_2

# ---------------------------------------------------------------------------
# CCIR 476 encoding
# ---------------------------------------------------------------------------

_LETTER_CODE = {ch: code for code, ch in LETTERS.items()}
_FIGURE_CODE = {ch: code for code, ch in FIGURES.items() if ch is not None}
_CONTROL_CODE = {name: code for code, name in CONTROL.items()}

LTRS = _CONTROL_CODE["LTRS"]
FIGS = _CONTROL_CODE["FIGS"]
IDLE_DX = PHASING_2   # idle signal sent in a DX slot
IDLE_RX = PHASING_1   # idle signal sent in an RX slot

# Characters this encoder can send (what the decoder can print)
SENDABLE = frozenset(_LETTER_CODE) | frozenset(_FIGURE_CODE) | {"\r", "\n", " "}


def encode_text(text: str) -> List[str]:
    """Text to a list of 7-bit codeword strings, starting with LTRS (the
    decoder assumes letters mode at start-up) and adding LTRS/FIGS shifts
    whenever the character set changes. CR, LF and space are sent as their
    own mode-independent codewords. Raises ValueError for a character with
    no CCIR 476 code."""
    codes = [LTRS]
    letters = True
    for ch in text:
        if ch == "\r":
            codes.append(_CONTROL_CODE["CR"])
        elif ch == "\n":
            codes.append(_CONTROL_CODE["LF"])
        elif ch == " ":
            codes.append(_CONTROL_CODE["SP"])
        elif ch in _LETTER_CODE:
            if not letters:
                codes.append(LTRS)
                letters = True
            codes.append(_LETTER_CODE[ch])
        elif ch in _FIGURE_CODE:
            if letters:
                codes.append(FIGS)
                letters = False
            codes.append(_FIGURE_CODE[ch])
        else:
            raise ValueError(f"character {ch!r} has no CCIR 476 code")
    return codes


def interleave(codes: Sequence[str], preamble_dx_slots: int, tail_dx_slots: int) -> List[str]:
    """Builds the transmitted slot sequence.

    Even slots are DX slots (a new character, or an idle signal); odd slots
    are RX slots, each repeating the DX slot five positions earlier. RX
    slots whose DX was idle carry the other idle signal, so the preamble
    and tail alternate IDLE_DX / IDLE_RX every slot."""
    dx_sequence = [IDLE_DX] * preamble_dx_slots + list(codes) + [IDLE_DX] * tail_dx_slots
    slots: List[str] = []
    for i, dx in enumerate(dx_sequence):
        slots.append(dx)                      # slot 2i: DX
        source = (2 * i + 1) - 5              # the DX slot this RX slot repeats
        if source < 0:
            slots.append(IDLE_RX)
        else:
            repeated = slots[source]
            slots.append(IDLE_RX if repeated == IDLE_DX else repeated)
    return slots


# ---------------------------------------------------------------------------
# Sample messages
# ---------------------------------------------------------------------------

def make_message(station_id: str, lines: Sequence[str]) -> str:
    """Wraps lines in a NAVTEX message frame: ZCZC header, the lines, NNNN,
    each terminated by CR LF, with a final blank line."""
    body = "".join(f"{line}\r\n" for line in lines)
    return f"ZCZC {station_id}\r\n{body}NNNN\r\n\r\n"


STANDARD_MESSAGES = [
    make_message("GA86", [
        "WZ 616/26",
        "HUMBER.",
        "INNER BANK.",
        "PLATFORM SOUTHWARK 53-11.0N 002-05.8E UNLIT AND ALL NAVAIDS INOPERATIVE.",
    ]),
    make_message("GB12", [
        "GALE WARNING 04.",
        "AT 091200 UTC.",
        "LOW 987 EXPECTED 54N 003W 995 BY 101200 UTC.",
        "WINDS SW 8 TO 9 VEERING NW 7 TO GALE 8 LATER.",
    ]),
    make_message("GC07", [
        "NAVAREA I WARNING 1234/26.",
        "NORTH SEA.",
        "WRECK 55-04.8N 001-31.5W.",
        "MARKED BY LIGHTED BUOY. KEEP CLEAR.",
    ]),
]

_VOCABULARY = (
    "WARNING BUOY LIGHT UNLIT VESSEL CABLE PIPELINE DREDGING EXERCISES WRECK "
    "SHOAL DEPTH CHANNEL HARBOUR APPROACHES NORTH SOUTH EAST WEST OFFSHORE "
    "PLATFORM MARKED CAUTION ADVISED REPORTED POSITION KEEP CLEAR FIRING "
    "FORECAST STORM GALE FOG VISIBILITY PRESSURE FALLING RISING AREA TRAFFIC"
).split()


def random_message(rng: np.random.Generator, n_lines: int = 5) -> str:
    """A random NAVTEX-like message mixing words, numbers and positions, so
    letters/figures shifts occur often."""
    lines = []
    for _ in range(n_lines):
        words = [str(rng.choice(_VOCABULARY)) for _ in range(int(rng.integers(3, 7)))]
        kind = int(rng.integers(0, 3))
        if kind == 0:
            words.append(f"{int(rng.integers(0, 90)):02d}-{int(rng.integers(0, 60)):02d}."
                         f"{int(rng.integers(0, 10))}N {int(rng.integers(0, 180)):03d}-"
                         f"{int(rng.integers(0, 60)):02d}.{int(rng.integers(0, 10))}W")
        elif kind == 1:
            words.append(f"{int(rng.integers(1, 9999))}/{int(rng.integers(10, 30))}")
        lines.append(" ".join(words) + ".")
    station = "".join(rng.choice(list("ABCDEFGH"), 2)) + f"{int(rng.integers(0, 100)):02d}"
    return make_message(station, lines)


# ---------------------------------------------------------------------------
# FSK modulation
# ---------------------------------------------------------------------------

@dataclass
class Transmission:
    """A clean (noise-free) transmission and everything needed to score it."""

    audio: np.ndarray        # float64 samples, clean signal with silent lead/tail
    sample_rate: int
    signal_power: float      # average power of the active signal (amplitude^2 / 2)
    expected_text: str       # what a perfect decoder prints
    message_start_s: float   # when the first printable character starts (DX slot)
    message_end_s: float     # when the last printable character's repeat ends
    duration_s: float
    bits: np.ndarray         # every transmitted bit, in order (True = mark)
    bits_start_s: float      # when the first bit starts (the silent lead-in length)


def build_transmission(text: str, *, sample_rate: int = 48000, baud: float = 100.0,
                       mark_freq: float = 1785.0, space_freq: float = 1615.0,
                       preamble_s: float = 5.0, tail_slots: int = 12,
                       lead_s: float = 2.0, tail_s: float = 2.0,
                       clock_ppm: float = 0.0, freq_offset_hz: float = 0.0,
                       amplitude: float = 0.1) -> Transmission:
    """Encodes `text` and modulates it as continuous-phase FSK.

    clock_ppm       transmitter bit period is (1 + ppm*1e-6) times nominal, as
                    measured by the receiver's sample clock. Positive = slower.
    freq_offset_hz  added to both tones, simulating receiver mistuning relative
                    to the mark/space frequencies the decoder expects.
    preamble_s      length of the phasing preamble before the message.
    tail_slots      idle DX slots after the message (at least 3 are needed to
                    send the last repeats).
    lead_s, tail_s  silence (noise only, once the channel is applied) before
                    and after the transmission.
    """
    if tail_slots < 3:
        raise ValueError("tail_slots must be at least 3 so the last repeats are sent")
    codes = encode_text(text)
    pair_s = 14.0 / baud                                 # one DX + one RX slot
    n_pre = max(0, round(preamble_s / pair_s))
    slots = interleave(codes, n_pre, tail_slots)
    bits = np.array([c == "1" for slot in slots for c in slot], dtype=bool)

    spb = sample_rate / baud * (1.0 + clock_ppm * 1e-6)  # samples per bit
    n_active = int(math.ceil(len(bits) * spb))
    bit_index = np.minimum((np.arange(n_active) / spb).astype(np.int64), len(bits) - 1)
    freq = np.where(bits[bit_index], mark_freq, space_freq) + freq_offset_hz
    phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
    active = amplitude * np.sin(phase)

    lead = int(round(lead_s * sample_rate))
    tail = int(round(tail_s * sample_rate))
    audio = np.concatenate([np.zeros(lead), active, np.zeros(tail)])

    slot_s = 7.0 * spb / sample_rate
    first_print = next(i for i, c in enumerate(codes) if c not in (LTRS, FIGS))
    last_print = max(i for i, c in enumerate(codes) if c not in (LTRS, FIGS))
    start_slot = 2 * (n_pre + first_print)
    end_slot = 2 * (n_pre + last_print) + 5
    return Transmission(
        audio=audio, sample_rate=sample_rate,
        signal_power=amplitude ** 2 / 2.0,
        expected_text=text,
        message_start_s=lead_s + start_slot * slot_s,
        message_end_s=lead_s + (end_slot + 1) * slot_s,
        duration_s=len(audio) / sample_rate,
        bits=bits, bits_start_s=lead_s,
    )


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Channel:
    """Radio channel impairments. Every field is optional; the defaults give
    a perfect channel."""

    snr_db: Optional[float] = None      # None = no noise
    ref_bw_hz: float = 2500.0           # bandwidth the SNR is quoted in
    noise_band: Tuple[float, float] = (300.0, 3000.0)  # receiver audio passband
    fade_doppler_hz: float = 0.0        # 0 = no fading; else Rayleigh, this Doppler bandwidth
    impulse_rate_hz: float = 0.0        # mean impulses per second (needs snr_db)
    impulse_level_db: float = 10.0      # impulse peak std, dB above in-band noise rms
    impulse_decay_ms: float = 2.0       # impulse decay time constant
    interferer_hz: Optional[float] = None   # CW interferer audio frequency
    interferer_sir_db: float = 0.0      # signal-to-interferer power ratio
    seed: int = 0


def ebn0_db(snr_db: float, ref_bw_hz: float = 2500.0, baud: float = 100.0) -> float:
    return snr_db + 10.0 * math.log10(ref_bw_hz / baud)


def theory_raw_ber(snr_db: float, ref_bw_hz: float = 2500.0, baud: float = 100.0) -> float:
    """Raw bit error rate of an ideal non-coherent binary FSK detector at
    this SNR (Eb/N0 from ebn0_db). The real decoder should do no better."""
    ebn0 = 10.0 ** (ebn0_db(snr_db, ref_bw_hz, baud) / 10.0)
    return 0.5 * math.exp(-ebn0 / 2.0)


def _rayleigh_envelope(n: int, fs: int, doppler_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Rayleigh fading envelope with unit mean power, n samples at fs."""
    fs_env = max(20.0, 20.0 * doppler_hz)
    pad = 5.0 / doppler_hz                                # discard the filter's start-up
    m = int(math.ceil((n / fs + pad) * fs_env)) + 2
    sos = butter(2, doppler_hz, btype="low", fs=fs_env, output="sos")
    impulse = np.zeros(int(fs_env * 20.0 / doppler_hz) + 1)
    impulse[0] = 1.0
    variance = float(np.sum(sosfilt(sos, impulse) ** 2))  # output variance for unit white input
    c = sosfilt(sos, rng.standard_normal(m)) + 1j * sosfilt(sos, rng.standard_normal(m))
    envelope = np.abs(c[int(pad * fs_env):]) / math.sqrt(2.0 * variance)
    return np.interp(np.arange(n) / fs, np.arange(len(envelope)) / fs_env, envelope)


def apply_channel(audio: np.ndarray, sample_rate: int, signal_power: float,
                  channel: Channel) -> np.ndarray:
    """Returns the audio after fading, interference and noise, as float32.

    `signal_power` is the reference power the SNR is measured against (use
    Transmission.signal_power; for a noise-only test, pass any positive
    value and an all-zero `audio`)."""
    rng = np.random.default_rng(channel.seed)
    fs = sample_rate
    n = len(audio)
    x = np.asarray(audio, dtype=np.float64).copy()

    if channel.fade_doppler_hz > 0.0:
        x *= _rayleigh_envelope(n, fs, channel.fade_doppler_hz, rng)

    if channel.interferer_hz is not None:
        amp = math.sqrt(2.0 * signal_power * 10.0 ** (-channel.interferer_sir_db / 10.0))
        x += amp * np.sin(2.0 * np.pi * channel.interferer_hz * np.arange(n) / fs
                          + rng.uniform(0.0, 2.0 * np.pi))

    noise_rms: Optional[float] = None
    if channel.snr_db is not None:
        lo, hi = channel.noise_band
        sigma_w = math.sqrt(signal_power * (fs / 2.0)
                            / (channel.ref_bw_hz * 10.0 ** (channel.snr_db / 10.0)))
        sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
        x += sosfilt(sos, rng.standard_normal(n) * sigma_w)
        noise_rms = sigma_w * math.sqrt((hi - lo) / (fs / 2.0))

    if channel.impulse_rate_hz > 0.0:
        if noise_rms is None:
            raise ValueError("impulse noise needs snr_db, which sets the level it is relative to")
        tau = channel.impulse_decay_ms * 1e-3 * fs
        length = int(5 * tau) + 1
        decay = np.exp(-np.arange(length) / tau)
        peak = noise_rms * 10.0 ** (channel.impulse_level_db / 20.0)
        for _ in range(int(rng.poisson(channel.impulse_rate_hz * n / fs))):
            pos = int(rng.integers(0, n))
            burst = rng.standard_normal(length) * decay * peak
            x[pos:pos + length] += burst[:max(0, n - pos)]

    return x.astype(np.float32)
