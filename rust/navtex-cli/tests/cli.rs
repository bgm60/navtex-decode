//! Tests of the finished program: they run the real `navtex-cli` executable.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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
        let output = run_cli(&[path]);
        assert!(output.status.success(), "{set}: {}", stderr(&output));
        assert_eq!(stderr(&output), "", "{set}: nothing should go to stderr");
        let want = format!("Decoding file: {path}\n{}", python_text(set));
        assert_eq!(stdout(&output), want, "{set}");
    }
}

#[test]
fn help_succeeds_and_describes_the_program() {
    for flag in ["-h", "--help"] {
        let output = run_cli(&[flag]);
        assert!(output.status.success());
        assert!(stdout(&output).contains("usage: navtex-cli FILE.wav"));
        assert!(stdout(&output).contains("48000 Hz"));
    }
}

#[test]
fn a_mistake_on_the_command_line_exits_with_2_and_shows_the_usage() {
    for args in [vec![], vec!["a.wav", "b.wav"], vec!["--bogus"]] {
        let output = run_cli(&args);
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert_eq!(stdout(&output), "");
        assert!(stderr(&output).contains("usage: navtex-cli FILE.wav"), "{args:?}");
    }
}

#[test]
fn a_missing_file_exits_with_1_and_says_why() {
    let dir = TempDir::new("missing");
    let path = dir.0.join("nothing.wav");
    let output = run_cli(&[path.to_str().unwrap()]);
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
    let output = run_cli(&[path.to_str().unwrap()]);
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
    let output = run_cli(&[path.to_str().unwrap()]);
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

    let output = run_cli(&[path.to_str().unwrap()]);
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
