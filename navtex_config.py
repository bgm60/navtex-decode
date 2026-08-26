"""
NAVTEX Decoder — TOML Profile Configuration
=============================================

Loads a named profile (a `[section]` in a TOML file) containing every
input-source setting and tunable parameter the decoder needs, replacing
the old collection of individual CLI flags.

Each profile is fully self-contained -- there is deliberately no
`[defaults]` section that others inherit from. Profiles are meant to
represent distinct real-world setups (different receivers, different
signal conditions, different logging destinations), and implicit
inheritance across them is exactly the kind of hidden state this project
has already been burned by once (see TUNING_REFERENCE.md / project
history) -- better for each profile to say plainly what it uses, even at
the cost of some repetition between profiles.

Example TOML file
-------------------
    [strong_signal_file]
    mode = "file"
    wav_file = "recording.wav"
    log_dir = "logs/"
    loop_gain = 0.05
    min_groups_for_acquire = 25

    [weak_dx_live]
    mode = "live"
    device = 2
    log_dir = "logs/weak_dx/"
    loop_gain = 0.05
    char_drop_threshold = 0.46
    min_groups_for_acquire = 25
    sync_window = 250

Run with:  python navtex_decode.py weak_dx_live --config myconfig.toml

Only keys you actually want to override from the code's built-in
defaults need to be present -- see Profile's field defaults below, which
mirror the values documented in TUNING_REFERENCE.md at the time this
module was written.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional, get_type_hints

if sys.version_info >= (3, 11):
    import tomllib
else:
    raise RuntimeError(
        "navtex_config requires Python 3.11+ for the standard-library "
        "tomllib module. Install `tomli` and adjust the import above if "
        "you need to support an older Python."
    )


class ConfigError(Exception):
    """Raised for anything wrong with a TOML config file or profile --
    missing file, missing/duplicate section, unknown key, wrong type, or
    a coupled-parameter constraint violation. Callers (navtex_decode.py's
    main()) catch this specifically and print a clean message rather than
    a raw traceback, since this is a user-facing configuration mistake,
    not a bug.
    """


@dataclass
class Profile:
    """One fully-resolved, validated profile. Field defaults below match
    the current in-code defaults documented in TUNING_REFERENCE.md --
    NOT necessarily "recommended" values, just what the pipeline already
    does if a key is omitted from the TOML profile.

    Deliberately excluded from this schema, per TUNING_REFERENCE.md's "Do
    not tune these": `baud` and `FecCombiner.LAG`. Both are fixed by the
    SITOR-B/NAVTEX protocol itself, not tuning choices, so there's no
    field for them here and no way to set them from a profile.
    """

    # --- Input source (replaces the old --live/--device/--log-dir/wav_file CLI flags) ---
    mode: str = "file"              # "file" or "live"
    wav_file: Optional[str] = None  # required if mode == "file"
    device: Optional[str] = None    # index or name-substring; only used if mode == "live"
    log_dir: Optional[str] = None   # omit to disable logging

    # --- Step 1: sampling/windowing (NavtexConfig) ---
    # Calibration values -- specific to a given receiver/SDR setup, not
    # casual tuning knobs. Re-derive with calibrate_tone_frequencies.py
    # against a fresh recording if you change receiver hardware, rather
    # than hand-editing these.
    sample_rate: int = 48000
    oversample: int = 8
    window_type: str = "hamming"
    mark_freq: float = 1785.0
    space_freq: float = 1615.0

    # --- Step 3: bit-clock recovery (BitSync) ---
    loop_gain: float = 0.05

    # --- Step 4: character sync (CharacterGrouper) ---
    sync_window: int = 250
    char_acquire_threshold: float = 0.6
    char_drop_threshold: float = 0.46
    char_switch_margin: float = 0.2
    min_groups_for_acquire: int = 25

    # --- Step 4: FEC parity locking (FecCombiner) ---
    fec_acquire_threshold: float = 0.6
    fec_switch_margin: float = 0.2
    min_samples_for_rate: int = 5
    rate_window: int = 30
    lock_window: int = 15
    phasing_burst_threshold: int = 4

    # --- Console/log signal-strength reading (SignalStrengthTracker) ---
    signal_strength_window: int = 100

    def validate(self) -> None:
        """Checks the coupled-parameter relationships documented in
        TUNING_REFERENCE.md's "Coupled parameters" section. Raises
        ConfigError with a specific, actionable message on violation --
        this is exactly the kind of mistake a TOML file makes easy to
        introduce silently (e.g. copy-pasting one profile's sync_window
        into another that has a different min_groups_for_acquire), so
        it's checked explicitly rather than left to fail confusingly at
        runtime (acquisition simply never happening, with no obvious
        cause).
        """
        if self.mode not in ("file", "live"):
            raise ConfigError(f"mode must be \"file\" or \"live\", got {self.mode!r}")
        if self.mode == "file" and not self.wav_file:
            raise ConfigError("mode = \"file\" requires wav_file to be set")

        # sync_window must comfortably exceed min_groups_for_acquire * 7
        # bits -- "comfortably" interpreted here as "at all", i.e. the
        # hard structural floor. TUNING_REFERENCE.md's own current
        # values (30 groups x 7 = 210 bits vs 250-bit window) leave only
        # 40 bits of margin, so this only raises on an outright violation,
        # not on thin-but-valid margins -- it's not this validator's job
        # to second-guess how much margin is "enough", only to catch a
        # configuration that cannot work at all.
        min_bits = self.min_groups_for_acquire * 7
        if self.sync_window <= min_bits:
            raise ConfigError(
                f"sync_window ({self.sync_window}) must be greater than "
                f"min_groups_for_acquire * 7 ({min_bits}) -- otherwise "
                f"acquisition can stall entirely (never enough groups in "
                f"the window to satisfy the minimum). See "
                f"TUNING_REFERENCE.md's \"Coupled parameters\" section."
            )

        # RATE_WINDOW and lock_window, together with the fixed LAG=5,
        # size FecCombiner's history buffer. Nothing here can actually
        # break structurally the way sync_window/min_groups_for_acquire
        # can (history_len = LAG + max(lock_window, rate_window) is
        # always well-defined) -- but flag an unusually small rate_window
        # anyway, since TUNING_REFERENCE.md's own measured false-lock
        # floor was 3 samples, and this is the one place a user-supplied
        # profile could quietly drop below a value that's actually been
        # validated against real data.
        if self.rate_window < self.min_samples_for_rate:
            raise ConfigError(
                f"rate_window ({self.rate_window}) must be >= "
                f"min_samples_for_rate ({self.min_samples_for_rate}) -- "
                f"otherwise a parity's rate estimate can never reach the "
                f"minimum sample count within its own rolling window."
            )


def _coerce(field_name: str, field_type: type, raw: Any) -> Any:
    """Light type-checking for one TOML value against its Profile field's
    declared type. TOML itself already distinguishes strings/ints/floats/
    bools, so this mostly catches the case of a profile using an int
    where a float field is documented (e.g. `loop_gain = 0` -- valid TOML,
    wrong intent) and int/float mismatches; Optional[...] fields accept
    None seamlessly since a missing key never reaches this function at
    all (see load_profile).
    """
    if field_type is float and isinstance(raw, int) and not isinstance(raw, bool):
        return float(raw)
    if field_type is Optional[str] and isinstance(raw, str):
        return raw
    if field_type is int and isinstance(raw, bool):
        raise ConfigError(f"{field_name}: expected int, got bool")
    if field_type in (int, float, str, bool) and not isinstance(raw, field_type):
        raise ConfigError(
            f"{field_name}: expected {field_type.__name__}, got {type(raw).__name__} ({raw!r})"
        )
    return raw


def load_profile(config_path: str, profile_name: str) -> Profile:
    """Loads and validates one named profile from a TOML file.

    Raises ConfigError (never a raw exception from tomllib or a KeyError)
    for every user-facing failure mode: missing file, malformed TOML,
    missing profile, or an unrecognized key -- an unrecognized key is
    treated as an error rather than silently ignored, since a typo'd key
    name (e.g. `sync_windo`) would otherwise fail silently by just using
    the built-in default, which is exactly the kind of quiet drift this
    project has already been bitten by once.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"could not parse {config_path}: {e}") from e

    if profile_name not in data:
        available = ", ".join(sorted(data.keys())) or "(none defined)"
        raise ConfigError(
            f"no profile named {profile_name!r} in {config_path}. "
            f"Available profiles: {available}"
        )

    section = data[profile_name]
    if not isinstance(section, dict):
        raise ConfigError(f"[{profile_name}] must be a table of key = value pairs")

    # get_type_hints (not f.type from dataclasses.fields) is required
    # here because this module uses `from __future__ import annotations`
    # -- under postponed evaluation, f.type is just the string "float" /
    # "Optional[str]" / etc., not an actual type object, which would
    # silently break every isinstance()/identity check below.
    resolved_types = get_type_hints(Profile)
    known_fields = {f.name: resolved_types[f.name] for f in fields(Profile)}
    kwargs: dict = {}
    for key, raw_value in section.items():
        if key not in known_fields:
            raise ConfigError(
                f"[{profile_name}] has unrecognized key {key!r}. "
                f"Valid keys: {', '.join(sorted(known_fields))}"
            )
        field_type = known_fields[key]
        # Optional[T] fields: unwrap to T for the coercion/type check,
        # `None` is never present here since a key is either given (with
        # a real value) or simply absent from the TOML table.
        underlying = getattr(field_type, "__args__", (field_type,))[0]
        kwargs[key] = _coerce(key, underlying, raw_value)

    profile = Profile(**kwargs)
    profile.validate()
    return profile
