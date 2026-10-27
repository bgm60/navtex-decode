//! Settings files: named profiles in a TOML file.
//!
//! The file format is the one the Python program uses (`navtex.toml`), so
//! one file can serve both programs while they live side by side. A profile
//! is one `[section]`; keys left out take the defaults in [`Profile::default`].
//!
//! What is different from Python:
//!
//! * Settings that only matter to the old Python locks (character sync, FEC
//!   parity lock, ...) are still *accepted*, so existing files keep working,
//!   but they are ignored and each one produces a warning.
//! * `signal_strength_window` has been renamed `signal_quality_window`. The old
//!   name still works, with a warning.
//! * Four settings are refused if they ask for something this program does
//!   not implement: `weak_signal_lock = false`, a `window_type` other than
//!   "hamming", an `oversample` other than 8, and a `sample_rate` other than
//!   48000.
//!
//! Error messages for everything the two programs share are word-for-word
//! the Python messages; `tests/` compares them against `config_cases.txt`,
//! which `tools/dump_config.py` captured from the Python loader.

use std::fmt::Write as _;
use std::path::Path;

use toml::Value;

/// The mark and space tones are always this far apart (ITU-R M.476).
pub const TONE_SHIFT: f64 = 170.0;

/// The only sample rate this program implements.
pub const SUPPORTED_RATE: i64 = 48000;

/// Where the audio comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    File,
    Live,
}

/// A live audio device: its number in the device list, or part of its name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
    Index(i64),
    Name(String),
}

/// One validated profile.
#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    pub mode: Mode,
    pub wav_file: Option<String>,
    pub device: Option<Device>,
    pub log_dir: Option<String>,
    pub db_file: Option<String>,
    pub quiet: bool,
    pub sample_rate: i64,
    pub centre_freq: f64,
    pub tones_inverted: bool,
    pub loop_gain: f64,
    pub signal_quality_window: i64,
}

impl Default for Profile {
    fn default() -> Self {
        Profile {
            mode: Mode::Live,
            wav_file: None,
            device: None,
            log_dir: None,
            db_file: None,
            quiet: false,
            sample_rate: SUPPORTED_RATE,
            centre_freq: 1700.0,
            tones_inverted: false,
            loop_gain: 0.05,
            signal_quality_window: 100,
        }
    }
}

impl Profile {
    /// Audio frequency of the mark (binary 1) tone, in Hz.
    pub fn mark_freq(&self) -> f64 {
        let offset = TONE_SHIFT / 2.0;
        if self.tones_inverted { self.centre_freq - offset } else { self.centre_freq + offset }
    }

    /// Audio frequency of the space (binary 0) tone, in Hz.
    pub fn space_freq(&self) -> f64 {
        let offset = TONE_SHIFT / 2.0;
        if self.tones_inverted { self.centre_freq + offset } else { self.centre_freq - offset }
    }

    /// Checks that the settings make sense together. The command line calls
    /// this again after it changes a setting.
    pub fn validate(&self) -> Result<(), String> {
        if self.mode == Mode::File && self.wav_file.as_deref().is_none_or(str::is_empty) {
            return Err("mode = \"file\" requires wav_file to be set".to_string());
        }
        let has_log = self.log_dir.as_deref().is_some_and(|s| !s.is_empty())
            || self.db_file.as_deref().is_some_and(|s| !s.is_empty());
        if self.quiet && !has_log {
            return Err("quiet = true needs log_dir or db_file to be set, \
                        otherwise the decoder would produce no output at all"
                .to_string());
        }
        if self.sample_rate != SUPPORTED_RATE {
            return Err(format!(
                "sample_rate = {} is not supported: only {SUPPORTED_RATE} Hz is implemented \
                 (resampling is not available)",
                self.sample_rate
            ));
        }
        let low = self.centre_freq - TONE_SHIFT / 2.0;
        let high = self.centre_freq + TONE_SHIFT / 2.0;
        let nyquist = self.sample_rate as f64 / 2.0;
        if low <= 0.0 || high >= nyquist {
            return Err(format!(
                "centre_freq ({} Hz) puts the tones at {} and {} Hz; both must be between \
                 0 Hz and half the sample rate ({} Hz).",
                fmt_g(self.centre_freq),
                fmt_g(low),
                fmt_g(high),
                fmt_g(nyquist)
            ));
        }
        Ok(())
    }
}

/// A profile that has been read, plus anything the user should be told.
#[derive(Debug, Clone, PartialEq)]
pub struct Loaded {
    pub profile: Profile,
    pub warnings: Vec<String>,
}

/// What type a setting must have.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Int,
    Float,
    Str,
    Bool,
    IntOrStr,
}

impl Kind {
    fn expected(self) -> &'static str {
        match self {
            Kind::Int => "int",
            Kind::Float => "float",
            Kind::Str => "str",
            Kind::Bool => "bool",
            Kind::IntOrStr => "int or str",
        }
    }
}

/// Settings that are used.
const USED: &[(&str, Kind)] = &[
    ("mode", Kind::Str),
    ("wav_file", Kind::Str),
    ("device", Kind::IntOrStr),
    ("log_dir", Kind::Str),
    ("db_file", Kind::Str),
    ("quiet", Kind::Bool),
    ("sample_rate", Kind::Int),
    ("centre_freq", Kind::Float),
    ("tones_inverted", Kind::Bool),
    ("loop_gain", Kind::Float),
    ("signal_quality_window", Kind::Int),
];

/// The old name of `signal_quality_window`.
const OLD_WINDOW_KEY: &str = "signal_strength_window";

const LOCK_REASON: &str = "the character-sync and FEC-lock settings are not used by the weak-signal lock";

/// Settings that older files contain. They are type-checked like Python does,
/// then ignored with a warning.
const IGNORED: &[(&str, Kind, &str)] = &[
    ("weak_signal_lock", Kind::Bool, "navtex-cli always uses the weak-signal lock"),
    ("engine", Kind::Str, "navtex-cli always uses the Rust decoder"),
    ("window_type", Kind::Str, "navtex-cli always uses the Hamming window"),
    ("oversample", Kind::Int, "navtex-cli always uses 8x oversampling"),
    ("sync_window", Kind::Int, LOCK_REASON),
    ("char_acquire_threshold", Kind::Float, LOCK_REASON),
    ("char_drop_threshold", Kind::Float, LOCK_REASON),
    ("char_switch_margin", Kind::Float, LOCK_REASON),
    ("min_groups_for_acquire", Kind::Int, LOCK_REASON),
    ("fec_acquire_threshold", Kind::Float, LOCK_REASON),
    ("fec_switch_margin", Kind::Float, LOCK_REASON),
    ("min_samples_for_rate", Kind::Int, LOCK_REASON),
    ("rate_window", Kind::Int, LOCK_REASON),
    ("lock_window", Kind::Int, LOCK_REASON),
    ("phasing_burst_threshold", Kind::Int, LOCK_REASON),
];

const TONES_REPLACED: &str = "which is no longer used: the tones are now set with centre_freq \
    (the midpoint between mark and space, which are always 170 Hz apart) \
    and tones_inverted (true if mark is the lower tone). \
    Please edit the profile to use those instead.";

/// A value that passed its type check.
enum Checked {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

/// Reads one named profile from a TOML file and validates it.
pub fn load_profile(config_path: &str, profile_name: &str) -> Result<Loaded, String> {
    let path = Path::new(config_path);
    if !path.is_file() {
        return Err(format!("config file not found: {config_path}"));
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("could not read {config_path}: {e}"))?;
    parse_profile(&text, config_path, profile_name)
}

/// The part of [`load_profile`] that works on text, so tests need no files.
pub fn parse_profile(text: &str, config_path: &str, profile_name: &str) -> Result<Loaded, String> {
    let data: toml::Table = text
        .parse()
        .map_err(|e: toml::de::Error| format!("could not parse {config_path}: {}", e.to_string().trim_end()))?;

    let Some(section) = data.get(profile_name) else {
        let mut names: Vec<&str> = data.keys().map(String::as_str).collect();
        names.sort_unstable();
        let available = if names.is_empty() { "(none defined)".to_string() } else { names.join(", ") };
        return Err(format!(
            "no profile named {} in {config_path}. Available profiles: {available}",
            py_str_repr(profile_name)
        ));
    };
    let Value::Table(section) = section else {
        return Err(format!("[{profile_name}] must be a table of key = value pairs"));
    };

    let mut profile = Profile::default();
    let mut warnings = Vec::new();
    // The ignored settings whose values are refused when they ask for
    // something this program cannot do. Checked after the shared validation,
    // so that the shared mistakes are reported first.
    let mut weak_signal_lock = true;
    let mut window_type = "hamming".to_string();
    let mut oversample = 8;
    let mut new_window_key = false;
    let mut old_window_key = false;

    // The table keeps the order of the file (the `preserve_order` feature),
    // so the first mistake in the file is the one reported, as in Python.
    for (key, raw) in section {
        if key == "mark_freq" || key == "space_freq" {
            return Err(format!("[{profile_name}] uses {}, {TONES_REPLACED}", py_str_repr(key)));
        }
        let known = USED.iter().find(|(k, _)| k == key).map(|&(_, kind)| kind);
        let ignored = IGNORED.iter().find(|(k, _, _)| k == key);
        let kind = if key == OLD_WINDOW_KEY {
            Kind::Int
        } else if let Some(kind) = known {
            kind
        } else if let Some(&(_, kind, _)) = ignored {
            kind
        } else {
            return Err(format!(
                "[{profile_name}] has unrecognized key {}. Valid keys: {}",
                py_str_repr(key),
                valid_keys().join(", ")
            ));
        };
        let value = check_type(key, kind, raw)?;

        match (key.as_str(), value) {
            ("mode", Checked::Str(s)) => {
                profile.mode = match s.as_str() {
                    "file" => Mode::File,
                    "live" => Mode::Live,
                    _ => return Err(format!("mode must be \"file\" or \"live\", got {}", py_str_repr(&s))),
                }
            }
            ("wav_file", Checked::Str(s)) => profile.wav_file = Some(s),
            ("device", Checked::Int(i)) => profile.device = Some(Device::Index(i)),
            ("device", Checked::Str(s)) => profile.device = Some(Device::Name(s)),
            ("log_dir", Checked::Str(s)) => profile.log_dir = Some(s),
            ("db_file", Checked::Str(s)) => profile.db_file = Some(s),
            ("quiet", Checked::Bool(b)) => profile.quiet = b,
            ("sample_rate", Checked::Int(i)) => profile.sample_rate = i,
            ("centre_freq", Checked::Float(f)) => profile.centre_freq = f,
            ("tones_inverted", Checked::Bool(b)) => profile.tones_inverted = b,
            ("loop_gain", Checked::Float(f)) => profile.loop_gain = f,
            ("signal_quality_window", Checked::Int(i)) => {
                new_window_key = true;
                profile.signal_quality_window = i;
            }
            (OLD_WINDOW_KEY, Checked::Int(i)) => {
                old_window_key = true;
                profile.signal_quality_window = i;
                warnings.push(format!(
                    "[{profile_name}] '{OLD_WINDOW_KEY}' is deprecated: \
                     use 'signal_quality_window' instead"
                ));
            }
            ("weak_signal_lock", Checked::Bool(b)) => weak_signal_lock = b,
            ("window_type", Checked::Str(s)) => window_type = s,
            ("oversample", Checked::Int(i)) => oversample = i,
            _ => {}
        }
        if let Some(&(_, _, reason)) = ignored {
            warnings.push(format!("[{profile_name}] '{key}' is ignored: {reason}"));
        }
    }

    // The same order as Python's checks, so the same mistake is reported first.
    profile.validate()?;
    // Python's own check of the engine name is not repeated here: any value of
    // the right type is ignored.
    if !weak_signal_lock {
        return Err("weak_signal_lock = false is not supported: navtex-cli has only the \
                    weak-signal lock (remove the key or set it to true)"
            .to_string());
    }
    if window_type != "hamming" {
        return Err(format!(
            "window_type = {} is not supported: navtex-cli implements only \"hamming\"",
            py_str_repr(&window_type)
        ));
    }
    if oversample != 8 {
        return Err(format!("oversample = {oversample} is not supported: navtex-cli implements only 8"));
    }
    if new_window_key && old_window_key {
        return Err(format!(
            "[{profile_name}] sets both 'signal_quality_window' and '{OLD_WINDOW_KEY}': \
             use only 'signal_quality_window'"
        ));
    }
    if profile.signal_quality_window < 1 {
        let key = if old_window_key { OLD_WINDOW_KEY } else { "signal_quality_window" };
        return Err(format!("{key} must be at least 1, got {}", profile.signal_quality_window));
    }
    Ok(Loaded { profile, warnings })
}

/// Every key a profile may contain, sorted, as Python lists them.
fn valid_keys() -> Vec<&'static str> {
    let mut keys: Vec<&str> = USED.iter().map(|&(k, _)| k).collect();
    keys.extend(IGNORED.iter().map(|&(k, _, _)| k));
    keys.push(OLD_WINDOW_KEY);
    keys.sort_unstable();
    keys
}

/// Python's rules for accepting a value: an int is accepted where a float is
/// wanted, but a bool is never accepted as a number.
fn check_type(key: &str, kind: Kind, raw: &Value) -> Result<Checked, String> {
    let accepted = match (kind, raw) {
        (Kind::Bool, Value::Boolean(b)) => Some(Checked::Bool(*b)),
        (Kind::Int | Kind::IntOrStr, Value::Integer(i)) => Some(Checked::Int(*i)),
        (Kind::Float, Value::Integer(i)) => Some(Checked::Float(*i as f64)),
        (Kind::Float, Value::Float(f)) => Some(Checked::Float(*f)),
        (Kind::Str | Kind::IntOrStr, Value::String(s)) => Some(Checked::Str(s.clone())),
        _ => None,
    };
    accepted.ok_or_else(|| {
        let hint = if key == "weak_signal_lock" {
            " (the levels were replaced by a switch: use true or false)"
        } else {
            ""
        };
        format!(
            "{key}: expected {}, got {} ({}){hint}",
            kind.expected(),
            py_type_name(raw),
            py_repr(raw)
        )
    })
}

// ---------------------------------------------------------------------------
// Python-style text, so that error messages match the Python program's
// ---------------------------------------------------------------------------

fn py_type_name(v: &Value) -> &'static str {
    match v {
        Value::String(_) => "str",
        Value::Integer(_) => "int",
        Value::Float(_) => "float",
        Value::Boolean(_) => "bool",
        Value::Datetime(d) => match (d.date.is_some(), d.time.is_some()) {
            (true, true) => "datetime",
            (true, false) => "date",
            _ => "time",
        },
        Value::Array(_) => "list",
        Value::Table(_) => "dict",
    }
}

/// `repr()` of a value, as Python writes it.
fn py_repr(v: &Value) -> String {
    match v {
        Value::String(s) => py_str_repr(s),
        Value::Integer(i) => i.to_string(),
        Value::Float(f) => py_float_repr(*f),
        Value::Boolean(true) => "True".to_string(),
        Value::Boolean(false) => "False".to_string(),
        Value::Datetime(d) => d.to_string(),
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Table(t) => format!(
            "{{{}}}",
            t.iter()
                .map(|(k, v)| format!("{}: {}", py_str_repr(k), py_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// `repr()` of a string: single quotes unless the text has a single quote and
/// no double quote.
pub fn py_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `repr()` of a float: `1700.0`, `0.0125`, `1e-05`, `inf`, `nan`.
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    // Rust's shortest round-trip digits are the same digits Python prints;
    // only the exponent notation differs (`1e-5` against `1e-05`).
    let text = format!("{f:?}");
    match text.split_once('e') {
        Some((mantissa, exp)) => {
            let (sign, digits) = match exp.strip_prefix('-') {
                Some(d) => ('-', d),
                None => ('+', exp),
            };
            format!("{mantissa}e{sign}{digits:0>2}")
        }
        None => text,
    }
}

/// Python's `format(x, "g")`: six significant digits, trailing zeros removed.
pub fn fmt_g(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if !x.is_finite() {
        return py_float_repr(x);
    }
    // Rounding to six digits first can change the exponent (999999.5 -> 1e+06).
    let sci = format!("{x:.5e}");
    let (mantissa, exp) = sci.split_once('e').expect("exponent form always has an e");
    let exp: i32 = exp.parse().expect("exponent is a number");
    let trim = |s: &str| -> String {
        if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s.to_string() }
    };
    if !(-4..6).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mantissa), exp.abs())
    } else {
        trim(&format!("{:.*}", (5 - exp) as usize, x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    fn load(text: &str) -> Result<Loaded, String> {
        parse_profile(text, "cfg.toml", "p")
    }

    fn err(text: &str) -> String {
        load(text).expect_err("the profile should be refused")
    }

    // --- Python oracle ----------------------------------------------------

    /// One case from `config_cases.txt`.
    struct Case {
        name: String,
        profile: String,
        toml: String,
        result: String,
    }

    fn read_cases() -> Vec<Case> {
        let path = format!("{}/../test-vectors/config_cases.txt", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(path).expect("config_cases.txt is missing");
        let text = text.replace("\r\n", "\n");
        let mut cases = Vec::new();
        for block in text.split("### ").skip(1) {
            let (name, rest) = block.split_once('\n').unwrap();
            let rest = rest.strip_prefix("profile: ").unwrap();
            let (profile, rest) = rest.split_once('\n').unwrap();
            let rest = rest.strip_prefix("--- toml\n").unwrap();
            let (toml, result) = rest.split_once("\n--- result\n").unwrap();
            cases.push(Case {
                name: name.to_string(),
                profile: profile.to_string(),
                toml: toml.to_string(),
                result: result.trim_end().to_string(),
            });
        }
        cases
    }

    fn opt_str(v: &Option<String>) -> String {
        v.as_deref().map_or("None".to_string(), py_str_repr)
    }

    /// The profile in the same words `dump_config.py` uses.
    fn describe(p: &Profile) -> String {
        let device = match &p.device {
            None => "None".to_string(),
            Some(Device::Index(i)) => i.to_string(),
            Some(Device::Name(s)) => py_str_repr(s),
        };
        let b = |v: bool| if v { "True" } else { "False" };
        [
            "ok".to_string(),
            format!("mode={}", py_str_repr(if p.mode == Mode::File { "file" } else { "live" })),
            format!("wav_file={}", opt_str(&p.wav_file)),
            format!("device={device}"),
            format!("log_dir={}", opt_str(&p.log_dir)),
            format!("db_file={}", opt_str(&p.db_file)),
            format!("quiet={}", b(p.quiet)),
            format!("sample_rate={}", p.sample_rate),
            format!("centre_freq={}", py_float_repr(p.centre_freq)),
            format!("tones_inverted={}", b(p.tones_inverted)),
            format!("loop_gain={}", py_float_repr(p.loop_gain)),
            format!("signal_window={}", p.signal_quality_window),
        ]
        .join("\n")
    }

    #[test]
    fn every_case_agrees_with_the_python_loader() {
        let cases = read_cases();
        assert!(cases.len() >= 40, "only {} cases found", cases.len());
        let example = format!("{}/../test-vectors/navtex.example.toml", env!("CARGO_MANIFEST_DIR"));
        let mut checked = (0, 0);
        for case in &cases {
            let toml = if case.toml == "@example" {
                std::fs::read_to_string(&example).unwrap().replace("\r\n", "\n")
            } else {
                case.toml.clone()
            };
            let got = match parse_profile(&toml, "<FILE>", &case.profile) {
                Ok(loaded) => {
                    checked.0 += 1;
                    describe(&loaded.profile)
                }
                Err(e) => {
                    checked.1 += 1;
                    format!("error: {e}")
                }
            };
            assert_eq!(got, case.result, "case {}", case.name);
        }
        // Both kinds of outcome were really exercised.
        assert!(checked.0 >= 15 && checked.1 >= 20, "{checked:?}");
    }

    // --- The text helpers -------------------------------------------------

    #[test]
    fn python_style_g_format() {
        let cases = [
            (50.0, "50"), (-35.0, "-35"), (135.0, "135"), (24000.0, "24000"), (23915.0, "23915"),
            (1700.5, "1700.5"), (0.0, "0"), (0.0001, "0.0001"), (0.00001234, "1.234e-05"),
            (123456.0, "123456"), (1234567.0, "1.23457e+06"), (999999.5, "1e+06"),
            (0.1 + 0.2, "0.3"), (-85.0, "-85"),
        ];
        for (x, want) in cases {
            assert_eq!(fmt_g(x), want, "{x}");
        }
    }

    #[test]
    fn python_style_float_and_string_repr() {
        assert_eq!(py_float_repr(1700.0), "1700.0");
        assert_eq!(py_float_repr(0.0125), "0.0125");
        assert_eq!(py_float_repr(1e-5), "1e-05");
        assert_eq!(py_float_repr(1e22), "1e+22");
        assert_eq!(py_float_repr(f64::INFINITY), "inf");
        assert_eq!(py_str_repr("abc"), "'abc'");
        assert_eq!(py_str_repr("it's"), "\"it's\"");
        assert_eq!(py_str_repr("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(py_str_repr("C:\\x\n"), "'C:\\\\x\\n'");
    }

    // --- Things the Python oracle does not cover --------------------------

    #[test]
    fn defaults_match_the_python_defaults() {
        let p = load("[p]\n").unwrap();
        assert_eq!(p.profile, Profile::default());
        assert!(p.warnings.is_empty());
        assert_eq!(p.profile.mark_freq(), 1785.0);
        assert_eq!(p.profile.space_freq(), 1615.0);
    }

    #[test]
    fn inverted_tones_swap_mark_and_space() {
        let p = load("[p]\ntones_inverted = true\ncentre_freq = 1500\n").unwrap().profile;
        assert_eq!((p.mark_freq(), p.space_freq()), (1415.0, 1585.0));
    }

    #[test]
    fn old_lock_settings_are_accepted_with_one_warning_each() {
        let text = "[p]\nweak_signal_lock = true\nengine = \"python\"\nwindow_type = \"hamming\"\n\
                    oversample = 8\nsync_window = 250\nchar_acquire_threshold = 0.6\n\
                    char_drop_threshold = 0.46\nchar_switch_margin = 0.2\nmin_groups_for_acquire = 25\n\
                    fec_acquire_threshold = 0.6\nfec_switch_margin = 0.2\nmin_samples_for_rate = 5\n\
                    rate_window = 30\nlock_window = 15\nphasing_burst_threshold = 4\n";
        let loaded = load(text).unwrap();
        assert_eq!(loaded.profile, Profile::default());
        assert_eq!(loaded.warnings.len(), 15);
        assert!(loaded.warnings[0].contains("[p] 'weak_signal_lock' is ignored"));
        assert!(loaded.warnings.iter().all(|w| w.starts_with("[p] '") && w.contains("is ignored")));
    }

    #[test]
    fn ignored_settings_are_still_type_checked() {
        assert_eq!(err("[p]\nsync_window = \"x\"\n"), "sync_window: expected int, got str ('x')");
        assert_eq!(
            err("[p]\nweak_signal_lock = \"normal\"\n"),
            "weak_signal_lock: expected bool, got str ('normal') \
             (the levels were replaced by a switch: use true or false)"
        );
        assert_eq!(err("[p]\nchar_drop_threshold = true\n"), "char_drop_threshold: expected float, got bool (True)");
    }

    #[test]
    fn the_four_unsupported_settings_are_refused() {
        assert!(err("[p]\nweak_signal_lock = false\n").starts_with("weak_signal_lock = false is not supported"));
        assert_eq!(
            err("[p]\nwindow_type = \"hann\"\n"),
            "window_type = 'hann' is not supported: navtex-cli implements only \"hamming\""
        );
        assert_eq!(
            err("[p]\noversample = 4\n"),
            "oversample = 4 is not supported: navtex-cli implements only 8"
        );
        assert_eq!(
            err("[p]\nsample_rate = 44100\n"),
            "sample_rate = 44100 is not supported: only 48000 Hz is implemented (resampling is not available)"
        );
    }

    #[test]
    fn the_shared_mistakes_are_reported_before_the_unsupported_ones() {
        assert!(err("[p]\nmode = \"tape\"\nweak_signal_lock = false\n").starts_with("mode must be"));
    }

    #[test]
    fn signal_strength_window_is_renamed() {
        let new = load("[p]\nsignal_quality_window = 40\n").unwrap();
        assert_eq!(new.profile.signal_quality_window, 40);
        assert!(new.warnings.is_empty());

        let old = load("[p]\nsignal_strength_window = 60\n").unwrap();
        assert_eq!(old.profile.signal_quality_window, 60);
        assert_eq!(
            old.warnings,
            ["[p] 'signal_strength_window' is deprecated: use 'signal_quality_window' instead"]
        );

        assert_eq!(
            err("[p]\nsignal_quality_window = 40\nsignal_strength_window = 60\n"),
            "[p] sets both 'signal_quality_window' and 'signal_strength_window': use only 'signal_quality_window'"
        );
        assert_eq!(err("[p]\nsignal_quality_window = 0\n"), "signal_quality_window must be at least 1, got 0");
        assert_eq!(err("[p]\nsignal_strength_window = -3\n"), "signal_strength_window must be at least 1, got -3");
    }

    #[test]
    fn an_unknown_key_lists_the_valid_ones() {
        let message = err("[p]\nsync_windo = 5\n");
        assert!(message.starts_with("[p] has unrecognized key 'sync_windo'. Valid keys: centre_freq, char_acquire_threshold, "));
        assert!(message.contains("signal_quality_window"));
        assert!(message.contains("signal_strength_window"));
    }

    #[test]
    fn bad_toml_and_missing_files_are_reported() {
        let message = err("[p\nmode = ");
        assert!(message.starts_with("could not parse cfg.toml: "), "{message}");

        let dir = TempDir::new();
        let missing = dir.path().join("nothing.toml");
        let missing = missing.to_str().unwrap();
        assert_eq!(load_profile(missing, "p").unwrap_err(), format!("config file not found: {missing}"));

        let file = dir.path().join("c.toml");
        std::fs::write(&file, "[p]\nmode = \"file\"\nwav_file = \"a.wav\"\n").unwrap();
        let loaded = load_profile(file.to_str().unwrap(), "p").unwrap();
        assert_eq!(loaded.profile.wav_file.as_deref(), Some("a.wav"));
    }

    #[test]
    fn validate_can_be_run_again_after_the_command_line_changes_things() {
        let mut p = Profile { mode: Mode::File, wav_file: Some("a.wav".into()), quiet: true, ..Profile::default() };
        assert!(p.validate().unwrap_err().starts_with("quiet = true needs log_dir or db_file"));
        p.db_file = Some("x.db".into());
        assert!(p.validate().is_ok());
        p.wav_file = Some(String::new());
        assert_eq!(p.validate().unwrap_err(), "mode = \"file\" requires wav_file to be set");
    }

    #[test]
    fn every_example_profile_loads_with_warnings_only() {
        let path = format!("{}/../test-vectors/navtex.example.toml", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(path).unwrap();
        let loaded = parse_profile(&text, "example", "conservative").unwrap();
        // char sync and FEC settings in the file are reported as ignored
        assert!(loaded.warnings.iter().any(|w| w.contains("'sync_window' is ignored")));
        assert_eq!(loaded.profile.loop_gain, 0.19);
    }

    #[test]
    fn the_rust_example_file_loads_cleanly() {
        let path = format!("{}/../navtex.example.toml", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(path).unwrap();
        let table: toml::Table = text.parse().unwrap();
        let names: Vec<&String> = table.keys().collect();
        assert_eq!(names, ["live", "live_inverted", "recording"]);
        for name in names {
            let loaded = parse_profile(&text, "navtex.example.toml", name).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(loaded.warnings.is_empty(), "{name}: {:?}", loaded.warnings);
        }
        let inverted = parse_profile(&text, "x", "live_inverted").unwrap().profile;
        assert!(inverted.tones_inverted && inverted.device == Some(Device::Name("USB Audio".into())));
        let recording = parse_profile(&text, "x", "recording").unwrap().profile;
        assert!(recording.quiet && recording.mode == Mode::File);
    }
}
