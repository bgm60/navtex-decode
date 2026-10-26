//! The database output: each decoded line stored as a row in an SQLite file.
//!
//! The table is `navtex_log`, created if it does not exist:
//!
//! | column            | type    | holds                                              |
//! |-------------------|---------|----------------------------------------------------|
//! | `id`              | INTEGER | row number, in arrival order (the primary key)    |
//! | `utc_timestamp`   | TEXT    | `YYYY-MM-DD HH:MM:SS`, the UTC time the line began |
//! | `signal_strength` | INTEGER | the 00 to 99 reading at the start of the line      |
//! | `text_output`     | TEXT    | the line as decoded, without its CR and LF         |
//!
//! (The column is still called `signal_strength`, though the program now
//! calls the figure "signal quality", so that databases can be shared with
//! the Python program.) An index on `utc_timestamp` keeps time-range
//! queries fast as the table grows.
//!
//! Rows are appended, so one database collects lines from any number of
//! sessions. Lines that are empty or only spaces are not stored. Each row is
//! committed as soon as its line ends, so nothing already written is lost if
//! the program or the computer stops; the unfinished last line is stored
//! when the session closes.
//!
//! The database uses WAL journal mode, which lets other programs read it
//! while the decoder is writing. WAL needs a local disk: on a network or
//! cloud-synced drive SQLite may refuse it (a warning is printed and the
//! default mode is used) or behave unreliably.
//!
//! If an insert fails, the connection is closed and reopened once and the
//! row tried again. If that also fails, database logging is switched off for
//! the rest of the session with a warning, while decoding and the other
//! outputs carry on. The warnings are worded as in the Python program.

use crate::lines::{LineSink, SinkResult};
use crate::utc::UtcTime;
use rusqlite::{Connection, params};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

const TABLE: &str = "navtex_log";

/// How long to wait for another program that has the database locked.
pub const DEFAULT_BUSY_TIMEOUT: Duration = Duration::from_secs(10);

/// What the sink needs from a database connection. It is a trait so that
/// tests can supply one that fails on demand; the real one is [`SqliteDb`].
pub trait Database {
    fn insert(&mut self, utc_timestamp: &str, quality: u8, text: &str) -> Result<(), String>;
    fn close(&mut self) -> Result<(), String>;
}

/// A freshly opened connection, with any warnings that opening it produced.
pub struct Connected {
    pub db: Box<dyn Database>,
    pub warnings: Vec<String>,
}

/// How the sink makes (and remakes) its connection.
type Connect = Box<dyn FnMut() -> Result<Connected, String>>;

/// A real SQLite connection.
pub struct SqliteDb {
    /// `None` once closed.
    conn: Option<Connection>,
}

impl SqliteDb {
    /// Opens the database at `path` (creating it if needed), switches it to
    /// WAL mode, and makes sure the table and index exist.
    pub fn open(path: &Path, busy_timeout: Duration) -> Result<Connected, String> {
        let conn = Connection::open(path).map_err(|e| e.to_string())?;
        let mut warnings = Vec::new();
        let result = (|| -> rusqlite::Result<()> {
            conn.busy_timeout(busy_timeout)?;
            // The pragma answers with the mode that is now in force.
            let mode: String = conn.query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))?;
            if !mode.eq_ignore_ascii_case("wal") {
                warnings.push(format!(
                    "[database warning] {} could not use WAL mode (using '{mode}'); other programs \
                     may be blocked from reading it while decoding. Is it on a network drive?",
                    path.display()
                ));
            }
            conn.execute_batch(&format!(
                "CREATE TABLE IF NOT EXISTS {TABLE} (
                     id              INTEGER PRIMARY KEY,
                     utc_timestamp   TEXT    NOT NULL,
                     signal_strength INTEGER NOT NULL,
                     text_output     TEXT    NOT NULL);
                 CREATE INDEX IF NOT EXISTS {TABLE}_utc_timestamp ON {TABLE} (utc_timestamp);"
            ))
        })();
        result.map_err(|e| e.to_string())?;
        Ok(Connected { db: Box::new(SqliteDb { conn: Some(conn) }), warnings })
    }
}

impl Database for SqliteDb {
    fn insert(&mut self, utc_timestamp: &str, quality: u8, text: &str) -> Result<(), String> {
        let conn = self.conn.as_ref().ok_or("the database is closed")?;
        // With no transaction open, SQLite commits each statement as it finishes.
        conn.execute(
            &format!("INSERT INTO {TABLE} (utc_timestamp, signal_strength, text_output) VALUES (?1, ?2, ?3)"),
            params![utc_timestamp, i64::from(quality), text],
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
    }

    fn close(&mut self) -> Result<(), String> {
        match self.conn.take() {
            Some(conn) => conn.close().map_err(|(_, e)| e.to_string()),
            None => Ok(()),
        }
    }
}

pub struct SqliteLogSink {
    path: PathBuf,
    db: Box<dyn Database>,
    connect: Connect,
    warn: Box<dyn FnMut(&str)>,
    /// Database logging has been switched off after an unrecoverable error.
    disabled: bool,
    timestamp: String,
    quality: u8,
    text: String,
}

impl SqliteLogSink {
    /// Opens (or creates) the database at `path`, making its folder if
    /// needed. Warnings go to stderr.
    pub fn create(path: &Path) -> Result<Self, String> {
        SqliteLogSink::create_with(path, DEFAULT_BUSY_TIMEOUT, Box::new(|message| eprintln!("{message}")))
    }

    /// As [`create`](Self::create), with the busy timeout and the place
    /// warnings go replaced.
    pub fn create_with(
        path: &Path,
        busy_timeout: Duration,
        warn: Box<dyn FnMut(&str)>,
    ) -> Result<Self, String> {
        if let Some(folder) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(folder).map_err(|e| e.to_string())?;
        }
        let connect_path = path.to_path_buf();
        let connect: Connect = Box::new(move || SqliteDb::open(&connect_path, busy_timeout));
        SqliteLogSink::from_parts(path.to_path_buf(), connect, warn)
    }

    /// Builds the sink from a way of connecting.
    fn from_parts(path: PathBuf, mut connect: Connect, mut warn: Box<dyn FnMut(&str)>) -> Result<Self, String> {
        let connected = connect()?;
        for message in &connected.warnings {
            warn(message);
        }
        Ok(SqliteLogSink {
            path,
            db: connected.db,
            connect,
            warn,
            disabled: false,
            timestamp: String::new(),
            quality: 0,
            text: String::new(),
        })
    }

    /// The database file's path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// After a failed insert: reconnect once and try the row again.
    fn recover_from_error(&mut self, error: &str, row: &(String, u8, String)) {
        (self.warn)(&format!("\n[database warning] database write failed ({error}); attempting to reconnect..."));
        let _ = self.db.close();
        let retry = (self.connect)().and_then(|connected| {
            for message in &connected.warnings {
                (self.warn)(message);
            }
            self.db = connected.db;
            self.db.insert(&row.0, row.1, &row.2)
        });
        match retry {
            Ok(()) => (self.warn)("[database warning] database reconnected successfully, continuing."),
            Err(retry_error) => {
                let _ = self.db.close();
                (self.warn)(&format!(
                    "[database warning] could not reconnect to database ({retry_error}); database \
                     logging disabled for the rest of this session (other output continues normally)."
                ));
                self.disabled = true;
            }
        }
    }
}

impl LineSink for SqliteLogSink {
    fn name(&self) -> &str {
        "SqliteLogSink"
    }

    // Like the text log, this recovers from its own errors, so these never
    // return `Err`.

    fn line_start(&mut self, at: SystemTime, quality: u8) -> SinkResult {
        self.timestamp = UtcTime::from_system_time(at).db_stamp();
        self.quality = quality;
        self.text.clear();
        Ok(())
    }

    fn write(&mut self, ch: char) -> SinkResult {
        if ch != '\r' && ch != '\n' {
            self.text.push(ch);
        }
        Ok(())
    }

    fn line_end(&mut self) -> SinkResult {
        let text = std::mem::take(&mut self.text);
        // (Python's `strip()` and Rust's `trim()` agree on what is blank,
        // apart from four control codes that never occur in decoded text.)
        if self.disabled || text.trim().is_empty() {
            return Ok(());
        }
        let row = (self.timestamp.clone(), self.quality, text);
        if let Err(error) = self.db.insert(&row.0, row.1, &row.2) {
            self.recover_from_error(&error, &row);
        }
        Ok(())
    }

    fn close(&mut self) -> SinkResult {
        if self.disabled {
            return Ok(());
        }
        if let Err(error) = self.db.close() {
            (self.warn)(&format!(
                "[database warning] error closing database ({error}) -- ignoring, shutting down anyway."
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lines::LineAssembler;
    use crate::testutil::{self, TempDir};
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use std::time::UNIX_EPOCH;

    fn start_time() -> SystemTime {
        UNIX_EPOCH + Duration::from_micros(testutil::START_MICROS)
    }

    fn open_sink(path: &Path) -> SqliteLogSink {
        SqliteLogSink::create_with(
            path,
            DEFAULT_BUSY_TIMEOUT,
            Box::new(|message| panic!("unexpected warning: {message}")),
        )
        .expect("cannot create the database")
    }

    /// Every row of the table as `id<TAB>timestamp<TAB>strength<TAB>text`,
    /// with backslash and tab escaped in the text as the reference file does.
    fn read_rows(path: &Path) -> Vec<String> {
        let conn = Connection::open(path).unwrap();
        let mut statement = conn
            .prepare("SELECT id, utc_timestamp, signal_strength, text_output FROM navtex_log ORDER BY id")
            .unwrap();
        statement
            .query_map([], |row| {
                Ok(format!(
                    "{}\t{}\t{}\t{}",
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    testutil::escape_tsv(&row.get::<_, String>(3)?)
                ))
            })
            .unwrap()
            .map(Result::unwrap)
            .collect()
    }

    /// The table and index layout as text, in the form `db_schema.txt` uses.
    fn describe_schema(path: &Path) -> Vec<String> {
        let conn = Connection::open(path).unwrap();
        let mut out = Vec::new();
        let mut columns = conn.prepare("PRAGMA table_info(navtex_log)").unwrap();
        let rows = columns
            .query_map([], |row| {
                let default: Option<String> = row.get(4)?;
                Ok(format!(
                    "column {} {} {} {} {} {}",
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    default.unwrap_or_else(|| "None".to_string()),
                    row.get::<_, i64>(5)?
                ))
            })
            .unwrap();
        out.extend(rows.map(Result::unwrap));
        let mut indexes = conn.prepare("PRAGMA index_list(navtex_log)").unwrap();
        let list: Vec<(String, i64, String, i64)> = indexes
            .query_map([], |row| Ok((row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        for (name, unique, origin, partial) in list {
            let mut info = conn.prepare(&format!("PRAGMA index_info({name})")).unwrap();
            let columns: Vec<String> = info
                .query_map([], |row| row.get::<_, String>(2))
                .unwrap()
                .map(Result::unwrap)
                .collect();
            out.push(format!(
                "index {name} unique={unique} origin={origin} partial={partial} columns={}",
                columns.join(",")
            ));
        }
        let mode: String = conn.query_row("PRAGMA journal_mode", [], |row| row.get(0)).unwrap();
        out.push(format!("journal_mode {mode}"));
        out
    }

    fn feed(sink: &mut SqliteLogSink, at: SystemTime, quality: u8, text: &str) {
        sink.line_start(at, quality).unwrap();
        for ch in text.chars() {
            sink.write(ch).unwrap();
        }
        sink.line_end().unwrap();
    }

    /// The main test: replay the line events Python received for each
    /// recording, with the fixed times, into a Rust database. The rows, and
    /// the layout of the table, must be exactly Python's.
    #[test]
    fn matches_the_python_database_for_every_recording() {
        let want_schema: Vec<String> = std::fs::read_to_string(testutil::vector_path("", "db_schema.txt"))
            .expect("cannot read db_schema.txt")
            .lines()
            .map(str::to_string)
            .collect();
        let sets = testutil::sets_with("dblog.tsv");
        assert!(sets.len() >= 10, "too few sets: {}", sets.len());
        for set in sets {
            let dir = TempDir::new();
            let path = dir.path().join("log.db");
            let mut sink = open_sink(&path);
            testutil::replay(&set, &mut sink);
            sink.close().unwrap();

            let want: Vec<String> = std::fs::read_to_string(testutil::vector_path(&set, "dblog.tsv"))
                .unwrap()
                .lines()
                .map(str::to_string)
                .collect();
            let got = read_rows(&path);
            assert_eq!(got.len(), want.len(), "{set}: number of rows");
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g, w, "{set}: row {k}");
            }
            assert_eq!(describe_schema(&path), want_schema, "{set}: layout");
            println!("{set}: {} rows identical to Python", want.len());
        }
    }

    #[test]
    fn rows_have_the_right_types() {
        let dir = TempDir::new();
        let path = dir.path().join("log.db");
        let mut sink = open_sink(&path);
        feed(&mut sink, start_time(), 83, "ZCZC KA91");
        sink.close().unwrap();
        let conn = Connection::open(&path).unwrap();
        let types: (String, String, String) = conn
            .query_row(
                "SELECT typeof(utc_timestamp), typeof(signal_strength), typeof(text_output) FROM navtex_log",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(types, ("text".to_string(), "integer".to_string(), "text".to_string()));
    }

    #[test]
    fn sessions_append_to_the_same_database() {
        let dir = TempDir::new();
        let path = dir.path().join("log.db");
        let mut first = open_sink(&path);
        feed(&mut first, start_time(), 10, "ONE");
        first.close().unwrap();
        let mut second = open_sink(&path);
        feed(&mut second, start_time() + Duration::from_secs(60), 20, "TWO");
        second.close().unwrap();
        assert_eq!(
            read_rows(&path),
            vec!["1\t2026-10-18 17:36:36\t10\tONE", "2\t2026-10-18 17:37:36\t20\tTWO"]
        );
    }

    #[test]
    fn blank_lines_are_not_stored_and_line_endings_are_removed() {
        let dir = TempDir::new();
        let path = dir.path().join("log.db");
        let mut sink = open_sink(&path);
        feed(&mut sink, start_time(), 1, "");
        feed(&mut sink, start_time(), 1, "\r\n");
        feed(&mut sink, start_time(), 1, "   \r\n");
        feed(&mut sink, start_time(), 5, "A  B\r\n");
        feed(&mut sink, start_time(), 6, "\r");
        sink.close().unwrap();
        assert_eq!(read_rows(&path), vec!["1\t2026-10-18 17:36:36\t5\tA  B"]);
    }

    #[test]
    fn the_unfinished_last_line_is_stored_when_the_session_closes() {
        let dir = TempDir::new();
        let path = dir.path().join("log.db");
        let mut assembler = LineAssembler::new(vec![Box::new(open_sink(&path))]);
        for ch in "DONE\r\nHALF".chars() {
            assembler.feed(ch, || 33);
        }
        assembler.close();
        let rows = read_rows(&path);
        assert_eq!(rows.len(), 2, "{rows:?}");
        assert!(rows[0].ends_with("\t33\tDONE") && rows[1].ends_with("\t33\tHALF"), "{rows:?}");
    }

    #[test]
    fn the_database_is_in_wal_mode_and_can_be_read_while_it_is_being_written() {
        let dir = TempDir::new();
        let path = dir.path().join("log.db");
        let mut sink = open_sink(&path);
        feed(&mut sink, start_time(), 50, "FIRST");
        // The sink is still open, yet another connection sees the committed row.
        assert_eq!(read_rows(&path).len(), 1);
        feed(&mut sink, start_time(), 50, "SECOND");
        assert_eq!(read_rows(&path).len(), 2);
        let mode: String = Connection::open(&path)
            .unwrap()
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        assert_eq!(mode, "wal");
        sink.close().unwrap();
    }

    #[test]
    fn a_missing_folder_is_created() {
        let dir = TempDir::new();
        let path = dir.path().join("a").join("b").join("log.db");
        let mut sink = open_sink(&path);
        feed(&mut sink, start_time(), 1, "X");
        sink.close().unwrap();
        assert_eq!(read_rows(&path).len(), 1);
    }

    #[test]
    fn a_path_that_cannot_be_opened_is_an_error_not_a_panic() {
        let dir = TempDir::new();
        // A folder cannot be opened as a database file.
        assert!(SqliteLogSink::create(dir.path()).is_err());
        // Nor can a folder be made inside a file.
        let blocker = dir.path().join("file");
        std::fs::write(&blocker, "x").unwrap();
        assert!(SqliteLogSink::create(&blocker.join("log.db")).is_err());
    }

    #[test]
    fn an_existing_database_with_the_python_layout_is_used_as_it_is() {
        // The statements exactly as the Python program writes them.
        let dir = TempDir::new();
        let path = dir.path().join("log.db");
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS navtex_log (
                                 id              INTEGER PRIMARY KEY,
                                 utc_timestamp   TEXT    NOT NULL,
                                 signal_strength INTEGER NOT NULL,
                                 text_output     TEXT    NOT NULL);
                 CREATE INDEX IF NOT EXISTS navtex_log_utc_timestamp ON navtex_log (utc_timestamp);
                 INSERT INTO navtex_log (utc_timestamp, signal_strength, text_output)
                     VALUES ('2020-01-01 00:00:00', 77, 'WRITTEN BY PYTHON');",
            )
            .unwrap();
        }
        let mut sink = open_sink(&path);
        feed(&mut sink, start_time(), 5, "WRITTEN BY RUST");
        sink.close().unwrap();
        assert_eq!(
            read_rows(&path),
            vec!["1\t2020-01-01 00:00:00\t77\tWRITTEN BY PYTHON", "2\t2026-10-18 17:36:36\t5\tWRITTEN BY RUST"]
        );
    }

    /// Checks a database made by the Python program, named by the
    /// NAVTEX_COMPAT_DB environment variable (see rust/tools/check_db.py and
    /// the step 3 notes); skipped unless that is set. Run with
    /// `cargo test --release -p navtex-cli -- --ignored`.
    #[test]
    #[ignore = "needs NAVTEX_COMPAT_DB, a database written by the Python program"]
    fn appends_to_a_database_written_by_the_python_program() {
        let path = PathBuf::from(std::env::var("NAVTEX_COMPAT_DB").expect("set NAVTEX_COMPAT_DB"));
        let before = read_rows(&path);
        let mut sink = open_sink(&path);
        feed(&mut sink, start_time(), 9, "RUST WAS HERE");
        sink.close().unwrap();
        let after = read_rows(&path);
        assert_eq!(&after[..before.len()], &before[..], "earlier rows changed");
        assert_eq!(after.len(), before.len() + 1);
        assert!(after.last().unwrap().ends_with("\t9\tRUST WAS HERE"));
    }

    // ----- failures, with a database that can be made to fail -----

    #[derive(Default)]
    struct Fake {
        rows: RefCell<Vec<(String, u8, String)>>,
        fail_inserts: Cell<bool>,
        fail_close: Cell<bool>,
        fail_connect: Cell<bool>,
        connects: Cell<usize>,
        warn_on_connect: Cell<bool>,
    }

    struct FakeDb(Rc<Fake>);

    impl Database for FakeDb {
        fn insert(&mut self, utc_timestamp: &str, quality: u8, text: &str) -> Result<(), String> {
            if self.0.fail_inserts.get() {
                return Err("database is locked".to_string());
            }
            self.0.rows.borrow_mut().push((utc_timestamp.to_string(), quality, text.to_string()));
            Ok(())
        }
        fn close(&mut self) -> Result<(), String> {
            if self.0.fail_close.get() { Err("close failed".to_string()) } else { Ok(()) }
        }
    }

    struct Harness {
        sink: SqliteLogSink,
        fake: Rc<Fake>,
        warnings: Rc<RefCell<Vec<String>>>,
    }

    /// A sink over a fake database. Connecting again after a failure gives a
    /// working connection (unless told otherwise), and clears the failure.
    fn harness() -> Harness {
        let fake = Rc::new(Fake::default());
        let warnings = Rc::new(RefCell::new(Vec::new()));
        let collected = Rc::clone(&warnings);
        let connect_fake = Rc::clone(&fake);
        let connect: Connect = Box::new(move || {
            connect_fake.connects.set(connect_fake.connects.get() + 1);
            if connect_fake.fail_connect.get() {
                return Err("unable to open database file".to_string());
            }
            connect_fake.fail_inserts.set(false);
            let warnings = if connect_fake.warn_on_connect.get() {
                vec!["[database warning] a warning from connecting".to_string()]
            } else {
                Vec::new()
            };
            Ok(Connected { db: Box::new(FakeDb(Rc::clone(&connect_fake))), warnings })
        });
        let sink = SqliteLogSink::from_parts(
            PathBuf::from("fake.db"),
            connect,
            Box::new(move |message| collected.borrow_mut().push(message.to_string())),
        )
        .unwrap();
        Harness { sink, fake, warnings }
    }

    #[test]
    fn a_failed_insert_reconnects_and_stores_the_row() {
        let mut h = harness();
        h.fake.fail_inserts.set(true);
        feed(&mut h.sink, start_time(), 40, "RETRIED");
        assert_eq!(h.fake.connects.get(), 2, "the first connection plus one reconnection");
        assert_eq!(*h.fake.rows.borrow(), vec![("2026-10-18 17:36:36".to_string(), 40, "RETRIED".to_string())]);
        let warnings = h.warnings.borrow();
        assert_eq!(
            *warnings,
            vec![
                "\n[database warning] database write failed (database is locked); attempting to reconnect...",
                "[database warning] database reconnected successfully, continuing."
            ]
        );
        drop(warnings);
        // And it carries on normally afterwards.
        feed(&mut h.sink, start_time(), 41, "NEXT");
        assert_eq!(h.fake.rows.borrow().len(), 2);
    }

    #[test]
    fn a_failed_reconnect_switches_database_logging_off() {
        let mut h = harness();
        h.fake.fail_inserts.set(true);
        h.fake.fail_connect.set(true);
        feed(&mut h.sink, start_time(), 40, "LOST");
        feed(&mut h.sink, start_time(), 40, "ALSO LOST");
        assert!(h.sink.close().is_ok());
        assert_eq!(h.fake.connects.get(), 2, "it should try once, then give up");
        assert!(h.fake.rows.borrow().is_empty());
        let warnings = h.warnings.borrow();
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert_eq!(
            warnings[1],
            "[database warning] could not reconnect to database (unable to open database file); \
             database logging disabled for the rest of this session (other output continues normally)."
        );
    }

    #[test]
    fn the_retry_failing_also_switches_logging_off() {
        let mut h = harness();
        // Reconnecting works, but inserts keep failing.
        let stubborn = Rc::clone(&h.fake);
        h.sink.connect = Box::new(move || {
            stubborn.connects.set(stubborn.connects.get() + 1);
            Ok(Connected { db: Box::new(FakeDb(Rc::clone(&stubborn))), warnings: Vec::new() })
        });
        h.fake.fail_inserts.set(true);
        feed(&mut h.sink, start_time(), 40, "LOST");
        assert!(h.warnings.borrow()[1].contains("could not reconnect to database (database is locked)"));
        feed(&mut h.sink, start_time(), 40, "SKIPPED");
        // The first connection, one reconnection, and none after logging was switched off.
        assert_eq!(h.fake.connects.get(), 2, "a disabled sink does not try again");
    }

    #[test]
    fn a_second_failure_after_a_recovery_recovers_again() {
        let mut h = harness();
        h.fake.fail_inserts.set(true);
        feed(&mut h.sink, start_time(), 1, "ONE");
        h.fake.fail_inserts.set(true);
        feed(&mut h.sink, start_time(), 2, "TWO");
        assert_eq!(h.fake.rows.borrow().len(), 2);
        assert_eq!(h.fake.connects.get(), 3);
    }

    #[test]
    fn warnings_from_connecting_are_passed_on() {
        let mut h = harness();
        h.fake.warn_on_connect.set(true);
        h.fake.fail_inserts.set(true);
        feed(&mut h.sink, start_time(), 1, "ONE");
        assert!(h.warnings.borrow().iter().any(|m| m == "[database warning] a warning from connecting"));
    }

    #[test]
    fn a_failed_close_is_reported_and_ignored() {
        let mut h = harness();
        h.fake.fail_close.set(true);
        assert!(h.sink.close().is_ok());
        assert_eq!(
            h.warnings.borrow()[0],
            "[database warning] error closing database (close failed) -- ignoring, shutting down anyway."
        );
    }

    #[test]
    fn closing_a_real_database_twice_is_harmless() {
        let dir = TempDir::new();
        let mut sink = open_sink(&dir.path().join("log.db"));
        assert!(sink.close().is_ok());
        assert!(sink.close().is_ok());
    }
}
