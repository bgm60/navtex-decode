"""
NAVTEX Decoder — TOML Profile Configuration
=============================================

Loads a named profile (a `[section]` in a TOML file) containing every
input-source setting and tunable parameter the decoder needs, replacing
the old collection of individual CLI flags.

Each profile is fully self-contained -- there is deliberately no
`[defaults]` section that others inherit from. Profiles are meant to
represent distinct real-world setups (different receivers, different
signal conditions, different logging destinations).

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
mirror the values documented in DOCUMENTATION.md
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional, Union, get_type_hints

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
    mode: str = "live"              # "file" or "live"
    wav_file: Optional[str] = None  # required if mode == "file"
    device: Optional[Union[int, str]] = None  # index or name-substring; only used if mode == "live"
    log_dir: Optional[str] = None   # omit to disable logging
    db_file: Optional[str] = None   # SQLite database path; omit to disable database logging

    # --- Step 1: sampling/windowing (NavtexConfig) ---
    # Calibration values -- specific to a given receiver/SDR setup, not
    # casual tuning knobs.
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
    """Type-checks/coerces one TOML value against its Profile field's
    declared type, which may itself be a Union of several accepted types
    (e.g. `device: Optional[Union[int, str]]`, since a device is validly
    either an index or a name substring).

    Accepts `raw` if it matches ANY member of the field's type (aside
    from NoneType, which never reaches here -- see load_profile). A bare
    int is also widened to float for a float-typed field member, since
    TOML's own int/float distinction is stricter than this schema needs
    (e.g. `loop_gain = 0` is valid TOML and a reasonable thing to write,
    even though 0 is an int literal).

    bool is explicitly excluded from matching a plain `int` member --
    Python's bool is a subclass of int, so `isinstance(True, int)` is
    True, which would otherwise silently accept `enable_x = true` for an
    int-typed field or a stray `1`/`0` for a bool-typed field as if it
    were the other type.
    """
    members = getattr(field_type, "__args__", (field_type,))
    members = tuple(m for m in members if m is not type(None))

    for member in members:
        if member is bool:
            if isinstance(raw, bool):
                return raw
            continue
        if member is int:
            if isinstance(raw, int) and not isinstance(raw, bool):
                return raw
            continue
        if member is float:
            if isinstance(raw, bool):
                continue
            if isinstance(raw, (int, float)):
                return float(raw)
            continue
        if member is str:
            if isinstance(raw, str):
                return raw
            continue

    expected = " or ".join(m.__name__ for m in members)
    raise ConfigError(f"{field_name}: expected {expected}, got {type(raw).__name__} ({raw!r})")


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
        kwargs[key] = _coerce(key, field_type, raw_value)

    profile = Profile(**kwargs)
    profile.validate()
    return profile


# ---------------------------------------------------------------------------
# Listing and saving profiles (used by the GUI)
# ---------------------------------------------------------------------------
#
# Saving uses the third-party `tomlkit` package rather than the standard
# library, because tomllib can only read TOML. tomlkit edits the file in
# place, keeping its comments, ordering and layout, so a profile saved
# from the GUI changes only the keys that changed. It is imported only
# when needed, so the command-line tool does not require it.

STARTER_CONFIG = """\
# NAVTEX Decoder profiles.
#
# Each [section] is one self-contained profile. Any key left out uses the
# built-in default. Profiles can be edited here or from the GUI's
# Settings dialog; see DOCUMENTATION.md for what each key does.

[default]
mode = "live"
"""


def read_profile_tables(config_path: str) -> dict:
    """Every profile table in the file, as plain dicts keyed by profile
    name, without validating them. Raises ConfigError if the file is
    missing or is not valid TOML."""
    path = Path(config_path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"could not parse {config_path}: {e}") from e
    return {name: table for name, table in data.items() if isinstance(table, dict)}


def load_profile_for_editing(config_path: str, profile_name: str):
    """Like load_profile, but for an editor that must be able to open a
    profile even when it is invalid, so the problem can be fixed.

    Returns (profile, problem). If the profile is valid, problem is None.
    Otherwise problem is load_profile's error message, and the profile
    holds every value that could be read, with built-in defaults for any
    unknown or wrongly-typed key. The result is NOT validated; saving it
    with save_profile validates it.
    """
    try:
        return load_profile(config_path, profile_name), None
    except ConfigError as e:
        problem = str(e)
    tables = read_profile_tables(config_path)
    if profile_name not in tables:
        raise ConfigError(problem)
    resolved_types = get_type_hints(Profile)
    known_fields = {f.name: resolved_types[f.name] for f in fields(Profile)}
    kwargs: dict = {}
    for key, raw_value in tables[profile_name].items():
        if key in known_fields:
            try:
                kwargs[key] = _coerce(key, known_fields[key], raw_value)
            except ConfigError:
                pass
    return Profile(**kwargs), problem


def _load_document(path: Path):
    import tomlkit
    try:
        return tomlkit.parse(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomlkit.exceptions.ParseError as e:
        raise ConfigError(f"could not parse {path}: {e}") from e


def _write_document(path: Path, doc) -> None:
    # Write to a temporary file and swap it in, so a failure part-way
    # through never leaves a half-written config file behind.
    import tomlkit
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(tomlkit.dumps(doc), encoding="utf-8", newline="")
    os.replace(tmp, path)


def save_profile(config_path: str, name: str, profile: Profile) -> None:
    """Validates `profile` and writes it to the file as [name], creating
    the section if it does not exist yet.

    Within an existing section, keys already present are updated in
    place (keeping any comment on the same line), keys whose value now
    differs from the built-in default are added (after the section's
    existing keys), and optional keys that are now unset (log_dir,
    db_file, device...) are removed. `mode` is always written. A key that
    is present but equal to its default is left in place, so the file
    never loses something the user wrote deliberately.
    """
    import tomlkit
    profile.validate()
    path = Path(config_path)
    doc = _load_document(path)
    defaults = Profile()
    if name in doc:
        table = doc[name]
        if not isinstance(table, dict):
            raise ConfigError(f"[{name}] in {config_path} is not a table")
    else:
        doc[name] = tomlkit.table()
        table = doc[name]
    for f in fields(Profile):
        value = getattr(profile, f.name)
        if value is None:
            if f.name in table:
                del table[f.name]
        elif f.name in table:
            if table[f.name] != value:
                old = table.item(f.name)
                new = tomlkit.item(value)
                new.trivia.comment = old.trivia.comment
                new.trivia.comment_ws = old.trivia.comment_ws
                table[f.name] = new
        elif value != getattr(defaults, f.name) or f.name == "mode":
            _add_key(table, f.name, value)
    _write_document(path, doc)


def _add_key(table, key: str, value) -> None:
    """Adds a key directly after the table's last existing key, rather
    than at the very end of the table. Comments that sit after a table's
    last key usually introduce the NEXT section, so appending after them
    would make the new key look as if it belonged to that section."""
    import tomlkit
    body = table.value.body
    last_index = max((i for i, (k, _) in enumerate(body) if k is not None), default=None)
    if last_index is None or last_index == len(body) - 1:
        table[key] = value
    else:
        # Container._insert_at is not part of tomlkit's documented API,
        # so fall back to a plain append if it ever changes.
        try:
            table.value._insert_at(last_index + 1, key, tomlkit.item(value))
        except (AttributeError, TypeError):
            table[key] = value


def delete_profile(config_path: str, name: str) -> None:
    """Removes [name] from the file. Raises ConfigError if it is not there."""
    path = Path(config_path)
    doc = _load_document(path)
    if name not in doc:
        raise ConfigError(f"no profile named {name!r} in {config_path}")
    del doc[name]
    _write_document(path, doc)
