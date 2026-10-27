//! Tests of the finished program: they run the real `navtex-cli` executable.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const CLIP: &str = "scen_two_messages";

fn run_cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_navtex-cli")).args(args).output().expect("cannot run navtex-cli")
}

fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../test-vectors")
}

/// The text the Python program printed for a set: the characters of the
/// `W,<codepoint>` rows of `lines.csv`.
fn python_text(set: &str) -> String {
    std::fs::read_to_string(vectors().join(set).join("lines.csv"))
        .expect("cannot read lines.csv")
        .lines()
        .filter_map(|row| row.strip_prefix("W,"))
        .map(|code| char::from_u32(code.parse().unwrap()).unwrap())
        .collect()
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("the output is not UTF-8")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// A folder in the temporary area, removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("navtex-cli-it-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn decoding_a_recording_prints_what_the_python_program_prints() {
    for set in ["real_clip1", "real_clip2", "scen_two_messages", "synth_snr10_seed1"] {
        let path = vectors().join(set).join("input.wav");
        let path = path.to_str().unwrap();
        let output = run_cli(&["--wav", path]);
        assert!(output.status.success(), "{set}: {}", stderr(&output));
        assert_eq!(stderr(&output), "", "{set}: nothing should go to stderr");
        let want = format!("Decoding file: {path}\n{}", python_text(set));
        assert_eq!(stdout(&output), want, "{set}");
    }
}

fn wav(set: &str) -> String {
    vectors().join(set).join("input.wav").to_str().unwrap().to_string()
}

/// Writes a settings file into `dir`. `body` is the text of the profile `[p]`.
/// (Paths in the body must be TOML literal strings, in single quotes, so
/// that the backslashes of Windows paths are left alone.)
fn write_config(dir: &TempDir, body: &str) -> String {
    let path = dir.0.join("navtex.toml");
    std::fs::write(&path, format!("[p]\n{body}")).unwrap();
    path.to_str().unwrap().to_string()
}

fn literal(path: &Path) -> String {
    format!("'{}'", path.to_str().unwrap())
}

#[test]
fn help_succeeds_and_describes_the_options() {
    for flag in ["-h", "--help"] {
        let output = run_cli(&[flag]);
        assert!(output.status.success());
        let text = stdout(&output);
        for want in [
            "Usage: navtex-cli", "--wav <FILE>", "--config <PATH>", "--quiet", "--no-quiet", "--list-devices", "--level",
            "Exit status", "Ctrl+C",
        ] {
            assert!(text.contains(want), "{flag}: missing {want:?} in\n{text}");
        }
    }
}

#[test]
fn a_mistake_on_the_command_line_exits_with_2() {
    for args in [vec![], vec!["a", "b"], vec!["--bogus"], vec!["--wav"], vec!["-q", "--no-quiet", "--wav", "x.wav"]] {
        let output = run_cli(&args);
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert_eq!(stdout(&output), "", "{args:?}");
        assert!(stderr(&output).contains("error:"), "{args:?}: {}", stderr(&output));
    }
    let output = run_cli(&[]);
    assert!(stderr(&output).contains("give a profile name, or --wav FILE"), "{}", stderr(&output));
}

#[test]
fn a_profile_can_decode_a_file_and_write_both_logs() {
    let dir = TempDir::new("logs");
    let logs = dir.0.join("logs");
    let db = dir.0.join("log.db");
    let config = write_config(
        &dir,
        &format!("mode = 'file'\nwav_file = '{}'\nlog_dir = {}\ndb_file = {}\n", wav(CLIP), literal(&logs), literal(&db)),
    );
    let output = run_cli(&["p", "--config", &config]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert_eq!(stderr(&output), "");

    // The status lines, then the text.
    let out = stdout(&output);
    let mut lines = out.splitn(4, '\n');
    assert_eq!(lines.next().unwrap(), format!("Decoding file: {}", wav(CLIP)));
    let log_line = lines.next().unwrap();
    assert!(log_line.starts_with(&format!("Logging decoded text to: {}", logs.display())), "{log_line}");
    assert!(log_line.ends_with("Z.txt"), "{log_line}");
    assert_eq!(lines.next().unwrap(), format!("Logging decoded lines to database: {}", db.display()));
    assert_eq!(lines.next().unwrap(), python_text(CLIP));

    // The text log: every line starts with a [date time quality] prefix, and
    // once the prefixes are taken off the text is what Python printed.
    let mut names: Vec<_> = std::fs::read_dir(&logs).unwrap().map(|e| e.unwrap().path()).collect();
    assert_eq!(names.len(), 1);
    let log = std::fs::read_to_string(names.remove(0)).unwrap();
    let body = log.splitn(5, '\n').nth(4).expect("the 4-line header");
    let (text, qualities) = strip_prefixes(body);
    assert_eq!(text, python_text(CLIP));

    // The database: strength and text of each line as Python stored them.
    let conn = rusqlite::Connection::open(&db).unwrap();
    let mut statement = conn.prepare("SELECT signal_strength, text_output FROM navtex_log ORDER BY id").unwrap();
    let rows: Vec<(i64, String)> =
        statement.query_map([], |r| Ok((r.get(0)?, r.get(1)?))).unwrap().map(Result::unwrap).collect();
    let want: Vec<(i64, String)> = std::fs::read_to_string(vectors().join(CLIP).join("dblog.tsv"))
        .unwrap()
        .lines()
        .map(|row| {
            let f: Vec<&str> = row.split('\t').collect();
            (f[2].parse().unwrap(), f[3].replace("\\t", "\t").replace("\\\\", "\\"))
        })
        .collect();
    assert_eq!(rows, want);
    assert_eq!(qualities, rows.iter().map(|r| r.0).collect::<Vec<_>>(), "text-log and database readings agree");
}

/// Splits the body of a text log into the decoded text and the quality number
/// of each non-empty line's `[YYYYMMDD HH:MM:SS QQ] ` prefix.
fn strip_prefixes(body: &str) -> (String, Vec<i64>) {
    let chars: Vec<char> = body.chars().collect();
    let (mut text, mut qualities) = (String::new(), Vec::new());
    let (mut i, mut line_start) = (0, true);
    while i < chars.len() {
        if line_start && chars[i] == '[' && i + 23 <= chars.len() {
            let prefix: String = chars[i..i + 23].iter().collect();
            if prefix.as_bytes()[9] == b' ' && prefix.ends_with("] ") {
                // The database leaves out lines with no text, the text log does not.
                if !matches!(chars.get(i + 23), Some('\r' | '\n') | None) {
                    qualities.push(prefix[19..21].parse().unwrap());
                }
                i += 23;
                line_start = false;
                continue;
            }
        }
        text.push(chars[i]);
        line_start = chars[i] == '\r' || chars[i] == '\n';
        i += 1;
    }
    (text, qualities)
}

#[test]
fn quiet_prints_nothing_and_no_quiet_overrides_the_profile() {
    let dir = TempDir::new("quiet");
    let db = dir.0.join("q.db");
    let config = write_config(&dir, &format!("mode = 'file'\nwav_file = '{}'\ndb_file = {}\nquiet = true\n", wav(CLIP), literal(&db)));
    let output = run_cli(&["p", "--config", &config]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert_eq!((stdout(&output).as_str(), stderr(&output).as_str()), ("", ""));
    let rows: i64 = rusqlite::Connection::open(&db)
        .unwrap()
        .query_row("SELECT COUNT(*) FROM navtex_log", [], |r| r.get(0))
        .unwrap();
    assert!(rows > 5, "the database should still be filled: {rows}");

    let output = run_cli(&["p", "--config", &config, "--no-quiet"]);
    assert!(output.status.success());
    assert!(stdout(&output).starts_with("Decoding file: "));
    assert!(stdout(&output).ends_with(&python_text(CLIP)));
}

#[test]
fn quiet_without_a_log_is_refused() {
    let output = run_cli(&["--wav", &wav(CLIP), "-q"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("quiet = true needs log_dir or db_file"), "{}", stderr(&output));
}

#[test]
fn the_wav_option_replaces_the_profiles_file() {
    let dir = TempDir::new("wavopt");
    let config = write_config(&dir, "mode = 'file'\nwav_file = 'nonexistent.wav'\n");
    let output = run_cli(&["p", "--config", &config, "--wav", &wav(CLIP)]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert_eq!(stdout(&output), format!("Decoding file: {}\n{}", wav(CLIP), python_text(CLIP)));
}

#[test]
fn the_profiles_tone_settings_are_used() {
    // Tones swapped around: nothing sensible can be decoded, but the profile
    // must change the result, which proves the setting reaches the decoder.
    let dir = TempDir::new("tones");
    let config = write_config(&dir, &format!("mode = 'file'\nwav_file = '{}'\ntones_inverted = true\n", wav(CLIP)));
    let output = run_cli(&["p", "--config", &config]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert_ne!(stdout(&output), format!("Decoding file: {}\n{}", wav(CLIP), python_text(CLIP)));
}

#[test]
fn settings_mistakes_exit_with_2_and_say_what_is_wrong() {
    let dir = TempDir::new("bad");
    let cases = [
        ("mode = 'tape'\n", "mode must be \"file\" or \"live\", got 'tape'"),
        ("mode = 'file'\n", "mode = \"file\" requires wav_file to be set"),
        ("loop_gain = 'x'\n", "loop_gain: expected float, got str ('x')"),
        ("mark_freq = 1785.0\n", "uses 'mark_freq', which is no longer used"),
        ("sync_windo = 3\n", "has unrecognized key 'sync_windo'"),
        ("weak_signal_lock = false\n", "weak_signal_lock = false is not supported"),
        ("window_type = 'hann'\n", "window_type = 'hann' is not supported"),
        ("oversample = 4\n", "oversample = 4 is not supported"),
        ("sample_rate = 44100\n", "sample_rate = 44100 is not supported"),
    ];
    for (body, want) in cases {
        let config = write_config(&dir, body);
        let output = run_cli(&["p", "--config", &config]);
        assert_eq!(output.status.code(), Some(2), "{body}");
        assert_eq!(stdout(&output), "", "{body}");
        assert!(stderr(&output).contains(want), "{body}: {}", stderr(&output));
    }
}

#[test]
fn a_missing_profile_or_settings_file_exits_with_2() {
    let dir = TempDir::new("noprofile");
    let config = write_config(&dir, "mode = 'live'\n");
    let output = run_cli(&["other", "--config", &config]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("no profile named 'other'") && stderr(&output).contains("Available profiles: p"));

    let missing = dir.0.join("none.toml");
    let output = run_cli(&["p", "--config", missing.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("config file not found"), "{}", stderr(&output));
}

#[test]
fn old_settings_give_warnings_but_the_decoding_goes_ahead() {
    let dir = TempDir::new("warn");
    let config = write_config(
        &dir,
        &format!("mode = 'file'\nwav_file = '{}'\nsync_window = 250\nsignal_strength_window = 100\n", wav(CLIP)),
    );
    let output = run_cli(&["p", "--config", &config]);
    assert!(output.status.success(), "{}", stderr(&output));
    let message = stderr(&output);
    assert!(message.contains("warning: [p] 'sync_window' is ignored"), "{message}");
    assert!(message.contains("warning: [p] 'signal_strength_window' is deprecated"), "{message}");
    assert_eq!(stdout(&output), format!("Decoding file: {}\n{}", wav(CLIP), python_text(CLIP)));

    // Warnings are shown even when the program is quiet.
    let db = dir.0.join("w.db");
    let config = write_config(
        &dir,
        &format!("mode = 'file'\nwav_file = '{}'\nsync_window = 250\ndb_file = {}\nquiet = true\n", wav(CLIP), literal(&db)),
    );
    let output = run_cli(&["p", "--config", &config]);
    assert!(stderr(&output).contains("'sync_window' is ignored"));
    assert_eq!(stdout(&output), "");
}

#[test]
fn a_log_that_cannot_be_opened_exits_with_1() {
    let dir = TempDir::new("badlog");
    let blocker = dir.0.join("file.txt");
    std::fs::write(&blocker, "x").unwrap();
    // A folder cannot be made inside a file.
    let config = write_config(
        &dir,
        &format!("mode = 'file'\nwav_file = '{}'\nlog_dir = {}\n", wav(CLIP), literal(&blocker.join("logs"))),
    );
    let output = run_cli(&["p", "--config", &config]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("error: could not open log output"), "{}", stderr(&output));
    assert_eq!(stdout(&output), format!("Decoding file: {}\n", wav(CLIP)), "nothing is decoded");
}

#[test]
fn a_missing_file_exits_with_1_and_says_why() {
    let dir = TempDir::new("missing");
    let path = dir.0.join("nothing.wav");
    let output = run_cli(&["--wav", path.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(stdout(&output), "", "no status line for a file that could not be opened");
    assert!(stderr(&output).starts_with("error: cannot open "), "{}", stderr(&output));
}

#[test]
fn a_file_at_the_wrong_sample_rate_is_refused_before_decoding() {
    let dir = TempDir::new("rate");
    let path = dir.0.join("rate.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44_100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).unwrap();
    for i in 0..1000 {
        writer.write_sample((i % 100) as i16).unwrap();
    }
    writer.finalize().unwrap();
    let output = run_cli(&["--wav", path.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(stdout(&output), "");
    let message = stderr(&output);
    assert!(message.contains("44100 Hz") && message.contains("48000 Hz"), "{message}");
}

#[test]
fn a_file_that_is_not_a_wav_exits_with_1() {
    let dir = TempDir::new("notwav");
    let path = dir.0.join("text.wav");
    std::fs::write(&path, "just some text").unwrap();
    let output = run_cli(&["--wav", path.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("not a readable WAV file"), "{}", stderr(&output));
}

#[test]
fn a_cut_short_recording_decodes_what_is_there_then_reports_the_error() {
    let dir = TempDir::new("cut");
    let path = dir.0.join("cut.wav");
    let mut bytes = std::fs::read(vectors().join("real_clip1").join("input.wav")).unwrap();
    bytes.truncate(bytes.len() / 2); // the header still says the whole length
    std::fs::write(&path, &bytes).unwrap();

    let output = run_cli(&["--wav", path.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    let text = stdout(&output);
    let body = text.strip_prefix(&format!("Decoding file: {}\n", path.to_str().unwrap())).expect("status line");
    let full = python_text("real_clip1");
    // Half the recording is about 32 seconds, which is roughly 100 characters of text.
    assert!(body.len() > 60, "the first half should have been decoded: {body:?}");
    assert!(body.starts_with(&full[..60]), "the text should begin as the whole recording's does");
    let message = stderr(&output);
    assert!(message.contains("error:") && message.contains("ended early or is damaged"), "{message}");
}

// --- Step 6: live audio, the level meter, stopping ---------------------------
//
// These tests must pass on a computer with or without a sound card, and must
// never wait for live audio, so they only try devices that cannot exist.

#[test]
fn listing_devices_works_with_or_without_a_sound_card() {
    let output = run_cli(&["--list-devices"]);
    match output.status.code() {
        Some(0) => {
            assert!(stdout(&output).starts_with("Audio input devices ("), "{}", stdout(&output));
            assert!(stdout(&output).contains("  0: "), "{}", stdout(&output));
            assert!(stdout(&output).contains("device = ..."));
        }
        Some(1) => assert!(stderr(&output).contains("error:"), "{}", stderr(&output)),
        other => panic!("unexpected exit status {other:?}: {}", stderr(&output)),
    }
}

#[test]
fn a_live_profile_with_a_device_that_does_not_exist_exits_with_1() {
    let dir = TempDir::new("nodevice");
    for (setting, want) in [
        ("device = 99", "device 99 does not exist"),
        ("device = '99'", "device 99 does not exist"),
        ("device = 'no-such-device-xyz'", "has \"no-such-device-xyz\" in its name"),
    ] {
        let config = write_config(&dir, &format!("mode = 'live'\n{setting}\n"));
        let output = run_cli(&["p", "--config", &config]);
        assert_eq!(output.status.code(), Some(1), "{setting}: {}", stderr(&output));
        assert_eq!(stdout(&output), "", "{setting}");
        assert!(stderr(&output).contains(want), "{setting}: {}", stderr(&output));
    }
}

#[test]
fn the_level_meter_shows_the_level_and_quality_and_logs_nothing() {
    let dir = TempDir::new("level");
    let logs = dir.0.join("logs");
    let db = dir.0.join("l.db");
    let config = write_config(
        &dir,
        &format!("mode = 'file'\nwav_file = '{}'\nlog_dir = {}\ndb_file = {}\n", wav(CLIP), literal(&logs), literal(&db)),
    );
    let output = run_cli(&["p", "--config", &config, "--level"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.starts_with(&format!("Decoding file: {}\nInput level meter: nothing is logged.\n", wav(CLIP))), "{text}");
    let redraws = text.matches('\r').count();
    assert!(redraws > 100, "about ten readings a second: {redraws}");
    assert!(text.contains("Level [") && text.contains(" dBFS") && text.contains("Quality"), "{text}");
    assert!(text.contains("Locked") && text.contains("Searching"), "the recording starts unlocked and then locks");
    assert!(!text.contains("STORM"), "the decoded text is not printed");
    assert!(!logs.exists() && !db.exists(), "nothing is logged in level mode");
}

#[test]
fn the_level_meter_cannot_be_combined_with_quiet() {
    let dir = TempDir::new("levelq");
    let db = dir.0.join("q.db");
    let config = write_config(&dir, &format!("mode = 'file'\nwav_file = '{}'\ndb_file = {}\nquiet = true\n", wav(CLIP), literal(&db)));
    let output = run_cli(&["p", "--config", &config, "--level"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("--level"), "{}", stderr(&output));
}

/// Ctrl+C in the middle of a recording: the text so far is kept, the logs are
/// closed and the program says so and exits normally.
///
/// The recording is fed through a named pipe, so the test controls exactly
/// when the program is in the middle of reading it.
#[cfg(unix)]
#[test]
fn ctrl_c_stops_decoding_cleanly_and_keeps_the_logs() {
    use std::io::Write;
    use std::process::Stdio;

    let dir = TempDir::new("ctrlc");
    let fifo = dir.0.join("live.wav");
    assert!(Command::new("mkfifo").arg(&fifo).status().unwrap().success(), "mkfifo failed");
    let db = dir.0.join("c.db");
    let config = write_config(&dir, &format!("mode = 'file'\nwav_file = {}\ndb_file = {}\n", literal(&fifo), literal(&db)));
    let bytes = std::fs::read(vectors().join("real_clip1").join("input.wav")).unwrap();

    let child = Command::new(env!("CARGO_BIN_EXE_navtex-cli"))
        .args(["p", "--config", &config])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let pid = child.id().to_string();
    let mut pipe = std::fs::OpenOptions::new().write(true).open(&fifo).unwrap();
    // Half of the recording, then a pause with the program waiting for more.
    let first = bytes.len() / 2;
    pipe.write_all(&bytes[..first]).unwrap();
    pipe.flush().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(700));
    assert!(Command::new("kill").args(["-INT", &pid]).status().unwrap().success());
    std::thread::sleep(std::time::Duration::from_millis(200));
    // The rest arrives; the program has been asked to stop, so it must not decode it.
    let _ = pipe.write_all(&bytes[first..]);
    drop(pipe);

    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.ends_with("\n\nStopped by user.\n"), "{text:?}");
    let status = format!("Decoding file: {}\nLogging decoded lines to database: {}\n", fifo.display(), db.display());
    let body = text.strip_prefix(&status).unwrap();
    let body = body.strip_suffix("\n\nStopped by user.\n").unwrap();
    let full = python_text("real_clip1");
    assert!(body.len() > 60 && body.len() < full.len() - 30, "stopped part-way: {} of {}", body.len(), full.len());
    assert!(full.starts_with(&body[..body.len() - 20]), "what was decoded is the start of the full text\n{body:?}\n{full:?}");

    // The database was closed properly with the lines so far in it.
    let rows: i64 = rusqlite::Connection::open(&db)
        .unwrap()
        .query_row("SELECT COUNT(*) FROM navtex_log", [], |r| r.get(0))
        .unwrap();
    assert!(rows >= 5, "{rows}");
}
