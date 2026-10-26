//! The text log: decoded text written to a time-stamped file.
//!
//! The file is created in the log folder (made if it does not exist) as
//! `navtex_YYYYMMDD_HHMMSSZ.txt`, named after the time decoding started.
//! It begins with a short header of `#` lines. Every line is prefixed with
//! `[YYYYMMDD HH:MM:SS QQ] `: the UTC time the line started and its 00 to
//! 99 signal quality. The text itself, CR and LF included, is written
//! exactly as decoded.
//!
//! Output is flushed at each CR or LF, and forced to disk at most once an
//! hour. Flushing every character over a long session makes a failure much
//! more likely on cloud-synced or network drives.
//!
//! If a write, flush or sync fails, the file is closed and reopened (in
//! append mode) once, and whatever continues the interrupted line gets a
//! fresh prefix, because the end of the old line may not have reached the
//! disk. If the reopen fails too, file logging is switched off for the rest
//! of the session with a warning; decoding and the other outputs go on.
//! The warnings are worded as in the Python program.

use crate::lines::{LineSink, SinkResult};
use crate::utc::UtcTime;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

/// How often the log is forced to disk by default: once an hour.
pub const DEFAULT_FSYNC_INTERVAL: Duration = Duration::from_secs(3600);

/// The file a log is written to. It is a trait so that tests can supply a
/// file that fails on demand; the real one is [`DiskFile`].
pub trait LogFile {
    fn write_str(&mut self, text: &str) -> io::Result<()>;
    /// Hands everything written so far to the operating system.
    fn flush(&mut self) -> io::Result<()>;
    /// Asks the operating system to put it on the disk itself.
    fn sync(&mut self) -> io::Result<()>;
    fn close(&mut self) -> io::Result<()>;
}

/// A real file on disk. Writes are buffered until `flush`.
struct DiskFile {
    writer: BufWriter<File>,
}

impl DiskFile {
    fn new(file: File) -> Self {
        DiskFile { writer: BufWriter::new(file) }
    }
}

impl LogFile for DiskFile {
    fn write_str(&mut self, text: &str) -> io::Result<()> {
        self.writer.write_all(text.as_bytes())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    fn sync(&mut self) -> io::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()
    }

    fn close(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

/// How the sink gets the file back after a failure.
type Reopen = Box<dyn FnMut() -> io::Result<Box<dyn LogFile>>>;

pub struct TextLogSink {
    path: PathBuf,
    file: Box<dyn LogFile>,
    reopen: Reopen,
    clock: Box<dyn FnMut() -> SystemTime>,
    warn: Box<dyn FnMut(&str)>,
    fsync_interval: Duration,
    last_fsync: Instant,
    /// File logging has been switched off after an unrecoverable error.
    disabled: bool,
    /// The file was reopened mid-line: prefix whatever continues it.
    reprefix: bool,
    last_quality: u8,
}

impl TextLogSink {
    /// Creates the log file in `log_dir`, named from the current time, and
    /// writes its header. `source` says what is being decoded.
    pub fn create(log_dir: &Path, source: &str) -> io::Result<Self> {
        TextLogSink::create_with(
            log_dir,
            source,
            Box::new(SystemTime::now),
            Box::new(|message| eprintln!("{message}")),
        )
    }

    /// As [`create`](Self::create), with the clock and the place warnings go replaced.
    pub fn create_with(
        log_dir: &Path,
        source: &str,
        mut clock: Box<dyn FnMut() -> SystemTime>,
        warn: Box<dyn FnMut(&str)>,
    ) -> io::Result<Self> {
        fs::create_dir_all(log_dir)?;
        let start = clock();
        let name = format!("navtex_{}Z.txt", UtcTime::from_system_time(start).file_stamp());
        let path = log_dir.join(name);
        // A new file for each session; an existing file of the same name is
        // overwritten, as in the Python program.
        let file: Box<dyn LogFile> = Box::new(DiskFile::new(File::create(&path)?));
        let reopen_path = path.clone();
        let reopen: Reopen = Box::new(move || {
            let file = OpenOptions::new().append(true).create(true).open(&reopen_path)?;
            Ok(Box::new(DiskFile::new(file)) as Box<dyn LogFile>)
        });
        TextLogSink::from_parts(path, file, reopen, start, source, clock, warn)
    }

    /// Builds the sink from its parts and writes the header.
    fn from_parts(
        path: PathBuf,
        mut file: Box<dyn LogFile>,
        reopen: Reopen,
        start: SystemTime,
        source: &str,
        clock: Box<dyn FnMut() -> SystemTime>,
        warn: Box<dyn FnMut(&str)>,
    ) -> io::Result<Self> {
        file.write_str("# NAVTEX decode log\n")?;
        file.write_str(&format!("# Started: {}\n", UtcTime::from_system_time(start).iso()))?;
        file.write_str(&format!("# Source: {source}\n"))?;
        file.write_str("#\n")?;
        file.flush()?;
        Ok(TextLogSink {
            path,
            file,
            reopen,
            clock,
            warn,
            fsync_interval: DEFAULT_FSYNC_INTERVAL,
            last_fsync: Instant::now(),
            disabled: false,
            reprefix: false,
            last_quality: 0,
        })
    }

    /// Changes how often the log is forced to disk.
    pub fn with_fsync_interval(mut self, interval: Duration) -> Self {
        self.fsync_interval = interval;
        self
    }

    /// The log file's path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn write_prefix(&mut self, at: SystemTime, quality: u8) -> io::Result<()> {
        let stamp = UtcTime::from_system_time(at).line_stamp();
        self.file.write_str(&format!("[{stamp} {quality:02}] "))
    }

    /// Forces the log to disk if the interval has passed. A failure is
    /// handled here, not passed on.
    fn maybe_fsync(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_fsync) >= self.fsync_interval {
            match self.file.sync() {
                Ok(()) => self.last_fsync = now,
                Err(error) => self.recover_from_error("fsync", &error),
            }
        }
    }

    /// Closes the file and opens it again in append mode, once. If that
    /// fails, switches file logging off.
    fn recover_from_error(&mut self, action: &str, error: &io::Error) {
        (self.warn)(&format!("\n[log warning] log file {action} failed ({error}); attempting to reopen..."));
        let _ = self.file.close();
        match (self.reopen)() {
            Ok(file) => {
                self.file = file;
                // Text written before the failure may not have reached the
                // file, so whatever continues the current line starts with a
                // fresh prefix. A fresh handle has had nothing synced yet, so
                // the sync timer restarts too.
                self.reprefix = true;
                self.last_fsync = Instant::now();
                (self.warn)("[log warning] log file reopened successfully, continuing.");
            }
            Err(reopen_error) => {
                (self.warn)(&format!(
                    "[log warning] could not reopen log file ({reopen_error}); file logging \
                     disabled for the rest of this session (other output continues normally)."
                ));
                self.disabled = true;
            }
        }
    }

    /// The part of `write` that can fail with an I/O error.
    fn write_char(&mut self, ch: char) -> io::Result<()> {
        if self.reprefix {
            self.reprefix = false;
            if ch != '\r' && ch != '\n' {
                let now = (self.clock)();
                self.write_prefix(now, self.last_quality)?;
            }
        }
        let mut buffer = [0u8; 4];
        self.file.write_str(ch.encode_utf8(&mut buffer))?;
        if ch == '\r' || ch == '\n' {
            self.file.flush()?;
            self.maybe_fsync();
        }
        Ok(())
    }
}

impl LineSink for TextLogSink {
    fn name(&self) -> &str {
        "TextLogSink"
    }

    // The log recovers from its own I/O errors, so these never return `Err`:
    // the assembler's last-resort "drop this output" is not needed.

    fn line_start(&mut self, at: SystemTime, quality: u8) -> SinkResult {
        if self.disabled {
            return Ok(());
        }
        self.last_quality = quality;
        self.reprefix = false;
        if let Err(error) = self.write_prefix(at, quality) {
            self.recover_from_error("write", &error);
        }
        Ok(())
    }

    fn write(&mut self, ch: char) -> SinkResult {
        if self.disabled {
            return Ok(());
        }
        if let Err(error) = self.write_char(ch) {
            self.recover_from_error("write", &error);
        }
        Ok(())
    }

    fn line_end(&mut self) -> SinkResult {
        self.reprefix = false;
        Ok(())
    }

    fn close(&mut self) -> SinkResult {
        if self.disabled {
            return Ok(());
        }
        if let Err(error) = self.file.close() {
            (self.warn)(&format!(
                "[log warning] error closing log file ({error}) -- ignoring, shutting down anyway."
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{self, TempDir};
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use std::time::UNIX_EPOCH;

    /// 2026-10-18 17:36:36.25 UTC, the "start" time of the reference logs.
    fn start_time() -> SystemTime {
        UNIX_EPOCH + Duration::from_micros(testutil::START_MICROS)
    }

    fn escape_file(path: &Path) -> String {
        testutil::escape(&String::from_utf8(fs::read(path).expect("cannot read log")).unwrap())
    }

    /// The main test: replay the line events Python received for each
    /// recording, with the fixed times, into the Rust log. The two files must
    /// match exactly, header and all.
    #[test]
    fn matches_the_python_log_for_every_recording() {
        let sets = testutil::sets_with("textlog.escaped.txt");
        assert!(sets.len() >= 10, "too few sets: {}", sets.len());
        for set in sets {
            let dir = TempDir::new();
            let mut sink = TextLogSink::create_with(
                dir.path(),
                "test source",
                Box::new(start_time),
                Box::new(|message| panic!("unexpected warning: {message}")),
            )
            .expect("cannot create the log");
            assert_eq!(sink.path().file_name().unwrap(), "navtex_20261018_173636Z.txt");
            testutil::replay(&set, &mut sink);
            sink.close().unwrap();

            let want = fs::read_to_string(testutil::vector_path(&set, "textlog.escaped.txt")).unwrap();
            // Git may have changed the line endings of the reference file.
            let want = want.replace("\r\n", "\n");
            assert_eq!(escape_file(sink.path()), want, "{set}");
            println!("{set}: log identical to Python");
        }
    }

    #[test]
    fn a_missing_folder_is_created() {
        let dir = TempDir::new();
        let nested = dir.path().join("a").join("b");
        let mut sink = TextLogSink::create(&nested, "somewhere").expect("cannot create the log");
        sink.line_start(start_time(), 7).unwrap();
        sink.write('A').unwrap();
        sink.close().unwrap();
        assert!(nested.is_dir());
        let text = fs::read_to_string(sink.path()).unwrap();
        assert!(text.starts_with("# NAVTEX decode log\n# Started: "), "{text}");
        assert!(text.contains("# Source: somewhere\n#\n"), "{text}");
        assert!(text.ends_with("] A") && text.contains(" 07] "), "{text}");
    }

    #[test]
    fn an_unwritable_folder_is_an_error_not_a_panic() {
        let dir = TempDir::new();
        let blocker = dir.path().join("file");
        fs::write(&blocker, "x").unwrap();
        // A folder cannot be created inside a file.
        assert!(TextLogSink::create(&blocker.join("logs"), "x").is_err());
    }

    // ----- failures, with a file that can be made to fail -----

    /// What a fake file has been asked to do, shared with the test.
    #[derive(Default)]
    struct Fake {
        contents: RefCell<String>,
        fail_writes: Cell<bool>,
        fail_flush: Cell<bool>,
        fail_sync: Cell<bool>,
        fail_close: Cell<bool>,
        fail_reopen: Cell<bool>,
        syncs: Cell<usize>,
        reopens: Cell<usize>,
    }

    struct FakeFile(Rc<Fake>);

    impl LogFile for FakeFile {
        fn write_str(&mut self, text: &str) -> io::Result<()> {
            if self.0.fail_writes.get() {
                return Err(io::Error::other("disk full"));
            }
            self.0.contents.borrow_mut().push_str(text);
            Ok(())
        }
        fn flush(&mut self) -> io::Result<()> {
            if self.0.fail_flush.get() { Err(io::Error::other("flush failed")) } else { Ok(()) }
        }
        fn sync(&mut self) -> io::Result<()> {
            self.0.syncs.set(self.0.syncs.get() + 1);
            if self.0.fail_sync.get() { Err(io::Error::other("sync failed")) } else { Ok(()) }
        }
        fn close(&mut self) -> io::Result<()> {
            if self.0.fail_close.get() { Err(io::Error::other("close failed")) } else { Ok(()) }
        }
    }

    struct Harness {
        sink: TextLogSink,
        fake: Rc<Fake>,
        warnings: Rc<RefCell<Vec<String>>>,
    }

    /// A sink over a fake file. A "reopen" gives another fake handle onto the
    /// same contents, like reopening a file in append mode. The clock ticks
    /// one second each time it is read, starting just after the start time.
    fn harness() -> Harness {
        let fake = Rc::new(Fake::default());
        let warnings = Rc::new(RefCell::new(Vec::new()));
        let collected = Rc::clone(&warnings);
        let reopen_fake = Rc::clone(&fake);
        let reopen: Reopen = Box::new(move || {
            reopen_fake.reopens.set(reopen_fake.reopens.get() + 1);
            if reopen_fake.fail_reopen.get() {
                Err(io::Error::other("cannot reopen"))
            } else {
                reopen_fake.fail_writes.set(false);
                reopen_fake.fail_flush.set(false);
                reopen_fake.fail_sync.set(false);
                Ok(Box::new(FakeFile(Rc::clone(&reopen_fake))) as Box<dyn LogFile>)
            }
        });
        let ticks = Cell::new(0u64);
        let clock = Box::new(move || {
            ticks.set(ticks.get() + 1);
            start_time() + Duration::from_secs(100 + ticks.get())
        });
        let sink = TextLogSink::from_parts(
            PathBuf::from("fake.txt"),
            Box::new(FakeFile(Rc::clone(&fake))),
            reopen,
            start_time(),
            "fake",
            clock,
            Box::new(move |message| collected.borrow_mut().push(message.to_string())),
        )
        .unwrap();
        Harness { sink, fake, warnings }
    }

    /// The time `start_time() + seconds` as a line prefix, quality 50.
    fn prefix(seconds: u64) -> String {
        let at = UtcTime::from_system_time(start_time() + Duration::from_secs(seconds));
        format!("[{} 50] ", at.line_stamp())
    }

    const HEADER: &str = "# NAVTEX decode log\n# Started: 2026-10-18T17:36:36.250000+00:00\n# Source: fake\n#\n";

    #[test]
    fn a_failed_write_reopens_the_file_and_reprefixes_the_line() {
        let mut h = harness();
        h.sink.line_start(start_time(), 50).unwrap();
        h.sink.write('A').unwrap();
        h.fake.fail_writes.set(true);
        h.sink.write('B').unwrap(); // fails, reopens; 'B' is lost
        h.sink.write('C').unwrap(); // continues the line with a fresh prefix
        h.sink.write('\r').unwrap();

        let want = format!("{HEADER}{}A{}C\r", prefix(0), prefix(101));
        assert_eq!(*h.fake.contents.borrow(), want);
        let warnings = h.warnings.borrow();
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert_eq!(warnings[0], "\n[log warning] log file write failed (disk full); attempting to reopen...");
        assert_eq!(warnings[1], "[log warning] log file reopened successfully, continuing.");
    }

    #[test]
    fn a_line_ending_after_a_reopen_gets_no_prefix() {
        let mut h = harness();
        h.sink.line_start(start_time(), 50).unwrap();
        h.fake.fail_writes.set(true);
        h.sink.write('A').unwrap(); // fails and reopens
        h.sink.write('\r').unwrap();
        h.sink.write('\n').unwrap();
        assert_eq!(*h.fake.contents.borrow(), format!("{HEADER}{}\r\n", prefix(0)));
    }

    #[test]
    fn a_new_line_after_a_reopen_uses_its_own_prefix_only() {
        let mut h = harness();
        h.sink.line_start(start_time(), 50).unwrap();
        h.fake.fail_writes.set(true);
        h.sink.write('A').unwrap();
        h.sink.line_end().unwrap();
        h.sink.line_start(start_time() + Duration::from_secs(9), 50).unwrap();
        h.sink.write('B').unwrap();
        assert_eq!(*h.fake.contents.borrow(), format!("{HEADER}{}{}B", prefix(0), prefix(9)));
    }

    #[test]
    fn a_failed_prefix_write_at_line_start_also_recovers() {
        let mut h = harness();
        h.fake.fail_writes.set(true);
        h.sink.line_start(start_time(), 50).unwrap();
        assert_eq!(h.fake.reopens.get(), 1);
        h.sink.write('A').unwrap();
        assert_eq!(*h.fake.contents.borrow(), format!("{HEADER}{}A", prefix(101)));
    }

    #[test]
    fn a_failed_flush_at_a_line_ending_recovers() {
        let mut h = harness();
        h.sink.line_start(start_time(), 50).unwrap();
        h.sink.write('A').unwrap();
        h.fake.fail_flush.set(true);
        h.sink.write('\r').unwrap();
        assert_eq!(h.fake.reopens.get(), 1);
        assert!(h.warnings.borrow()[0].contains("log file write failed (flush failed)"));
    }

    #[test]
    fn a_failed_reopen_switches_logging_off_without_stopping_anything() {
        let mut h = harness();
        h.sink.line_start(start_time(), 50).unwrap();
        h.fake.fail_writes.set(true);
        h.fake.fail_reopen.set(true);
        assert!(h.sink.write('A').is_ok());
        assert!(h.sink.write('B').is_ok());
        assert!(h.sink.line_start(start_time(), 50).is_ok());
        assert!(h.sink.close().is_ok());
        assert_eq!(h.fake.reopens.get(), 1, "it should only try once");
        let warnings = h.warnings.borrow();
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert_eq!(
            warnings[1],
            "[log warning] could not reopen log file (cannot reopen); file logging disabled for \
             the rest of this session (other output continues normally)."
        );
    }

    #[test]
    fn the_log_is_forced_to_disk_only_when_the_interval_has_passed() {
        // An hour has not passed, so no sync at all.
        let mut h = harness();
        h.sink.line_start(start_time(), 50).unwrap();
        h.sink.write('\r').unwrap();
        assert_eq!(h.fake.syncs.get(), 0);

        // An interval of zero has always passed: every line ending syncs.
        let mut h = harness();
        h.sink = h.sink.with_fsync_interval(Duration::ZERO);
        h.sink.line_start(start_time(), 50).unwrap();
        h.sink.write('A').unwrap();
        assert_eq!(h.fake.syncs.get(), 0, "only at a CR or LF");
        h.sink.write('\r').unwrap();
        h.sink.write('\n').unwrap();
        assert_eq!(h.fake.syncs.get(), 2);
    }

    #[test]
    fn a_failed_sync_recovers_like_any_other_failure() {
        let mut h = harness();
        h.sink = h.sink.with_fsync_interval(Duration::ZERO);
        h.sink.line_start(start_time(), 50).unwrap();
        h.fake.fail_sync.set(true);
        h.sink.write('\r').unwrap();
        assert_eq!(h.fake.reopens.get(), 1);
        assert!(h.warnings.borrow()[0].contains("log file fsync failed (sync failed)"));
    }

    #[test]
    fn a_failed_close_is_reported_and_ignored() {
        let mut h = harness();
        h.fake.fail_close.set(true);
        assert!(h.sink.close().is_ok());
        assert_eq!(
            h.warnings.borrow()[0],
            "[log warning] error closing log file (close failed) -- ignoring, shutting down anyway."
        );
    }
}
