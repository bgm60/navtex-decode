//! Helpers for the tests that read the reference files in `test-vectors/`.
//! Only compiled for `cargo test`.
//!
//! (`navtex-core` has similar helpers, but those are private to its own
//! tests, so the few that are needed here are repeated.)

use crate::lines::{LineSink, SinkResult};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::SystemTime;

/// Path of a file inside one reference set.
pub fn vector_path(set: &str, file: &str) -> String {
    format!("{}/../test-vectors/{}/{}", env!("CARGO_MANIFEST_DIR"), set, file)
}

/// The names of every reference set that has both an `input.wav` and the
/// given file, in alphabetical order.
pub fn sets_with(file: &str) -> Vec<String> {
    let root = format!("{}/../test-vectors", env!("CARGO_MANIFEST_DIR"));
    let entries = std::fs::read_dir(&root).unwrap_or_else(|e| panic!("cannot read {root}: {e}"));
    let mut names: Vec<String> = entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().join("input.wav").is_file() && entry.path().join(file).is_file())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no reference sets with {file} in {root}; see PROVENANCE.md");
    names
}

/// A 16-bit mono WAV read as `f32` (each sample divided by 32768).
pub fn read_wav(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "expected a mono file");
    assert_eq!(spec.bits_per_sample, 16, "expected 16-bit samples");
    reader.samples::<i16>().map(|s| f32::from(s.expect("bad sample")) / 32768.0).collect()
}

fn read_lines_of(path: &str) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    text.lines().map(str::to_string).collect()
}

/// The confidence of every bit decision in a set's `bits.csv`.
pub fn read_confidences(set: &str) -> Vec<f64> {
    read_lines_of(&vector_path(set, "bits.csv"))
        .iter()
        .skip(1)
        .map(|line| line.split(',').nth(2).expect("missing confidence").parse().expect("bad confidence"))
        .collect()
}

/// The reading after every bit decision, from a set's `quality.csv`.
pub fn read_quality(set: &str) -> Vec<u8> {
    read_lines_of(&vector_path(set, "quality.csv"))
        .iter()
        .skip(1)
        .map(|line| line.split(',').nth(1).expect("missing level").parse().expect("bad level"))
        .collect()
}

/// The line events Python produced, from a set's `lines.csv`, in the same
/// text form as [`Recorder`] writes them: `S,<level>`, `W,<codepoint>`, `E`.
pub fn read_line_events(set: &str) -> Vec<String> {
    read_lines_of(&vector_path(set, "lines.csv"))
}

/// An output that records what it is told, as rows of text, into a list the
/// test keeps hold of. (`Rc<RefCell<..>>` is a shared, changeable value:
/// the assembler owns the output, so the test needs a second handle to
/// read the list afterwards.)
pub struct Recorder {
    rows: Rc<RefCell<Vec<String>>>,
}

impl Recorder {
    pub fn new(rows: &Rc<RefCell<Vec<String>>>) -> Self {
        Recorder { rows: Rc::clone(rows) }
    }
}

impl LineSink for Recorder {
    fn name(&self) -> &str {
        "Recorder"
    }

    fn line_start(&mut self, _at: SystemTime, quality: u8) -> SinkResult {
        self.rows.borrow_mut().push(format!("S,{quality}"));
        Ok(())
    }

    fn write(&mut self, ch: char) -> SinkResult {
        self.rows.borrow_mut().push(format!("W,{}", u32::from(ch)));
        Ok(())
    }

    fn line_end(&mut self) -> SinkResult {
        self.rows.borrow_mut().push("E".to_string());
        Ok(())
    }

    fn close(&mut self) -> SinkResult {
        self.rows.borrow_mut().push("close".to_string());
        Ok(())
    }
}

// ----- shared by the output tests -----

/// 2026-10-18 17:36:36.25 UTC in microseconds since 1970: the "start" time
/// of the reference text logs (see `rust/tools/dump_logs.py`).
pub const START_MICROS: u64 = 1_792_344_996_250_000;

/// The time at which line number `k` (from 0) starts in the reference logs:
/// one second after the start, then every 7.5 seconds.
pub fn line_time(k: usize) -> SystemTime {
    std::time::UNIX_EPOCH + std::time::Duration::from_micros(START_MICROS + 1_000_000 + k as u64 * 7_500_000)
}

/// Plays the line events Python recorded for a set into an output, using the
/// reference times, as `dump_logs.py` did with the Python output.
pub fn replay(set: &str, sink: &mut dyn LineSink) {
    let mut line = None;
    for row in read_line_events(set) {
        let result = if let Some(level) = row.strip_prefix("S,") {
            let k = line.map_or(0, |k| k + 1);
            line = Some(k);
            sink.line_start(line_time(k), level.parse().expect("bad level"))
        } else if let Some(code) = row.strip_prefix("W,") {
            sink.write(char::from_u32(code.parse().expect("bad code")).expect("bad character"))
        } else if row == "E" {
            sink.line_end()
        } else {
            panic!("{set}: unexpected row {row:?}");
        };
        result.unwrap_or_else(|e| panic!("{set}: the output failed: {e}"));
    }
}

/// The same escaping `dump_logs.py` uses, so files can be compared without
/// line endings getting in the way: backslash, CR and LF become `\\`, `\r`
/// and `\n` followed by a real newline.
pub fn escape(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\r' => out.push_str("\\r"),
            '\n' => out.push_str("\\n\n"),
            other => out.push(other),
        }
    }
    out
}

/// A memory buffer that can be written to by an output and read by the test.
#[derive(Clone, Default)]
pub struct SharedBuf(Rc<RefCell<Vec<u8>>>);

impl SharedBuf {
    pub fn new() -> Self {
        SharedBuf::default()
    }

    pub fn contents(&self) -> Vec<u8> {
        self.0.borrow().clone()
    }
}

impl std::io::Write for SharedBuf {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A folder in the system temporary area that is removed when dropped.
pub struct TempDir(std::path::PathBuf);

impl TempDir {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("navtex-cli-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&path).expect("cannot create a temporary folder");
        TempDir(path)
    }

    pub fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Escapes backslash and tab, as `dump_logs.py` does for the database rows,
/// so each row can be compared as one line of text.
pub fn escape_tsv(text: &str) -> String {
    text.replace('\\', "\\\\").replace('\t', "\\t")
}
