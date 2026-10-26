//! Turns the stream of decoded characters into line events for the outputs.
//!
//! The decoder produces one long run of characters. The outputs (the
//! console, the text log, the database) want lines, each with the time it
//! started and the signal quality at that moment. The `LineAssembler` sits
//! between them and tells every output when a line starts, each character
//! on it, and when it ends.
//!
//! The rules, all copied from the Python program:
//!
//! * A line starts when its first character arrives. The clock and the
//!   signal quality are read at that moment, once per line.
//! * A carriage return (CR) or a line feed (LF) ends the line. It is part
//!   of the line, written before the line ends.
//! * CR followed by LF is one line ending: the LF is written but starts no
//!   new line.
//! * [`LineAssembler::end_line`] ends the line in progress as if a CR LF had
//!   arrived. It is used when the decoder's lock is released, so the next
//!   message does not carry on a half-finished line stamped with an old time.
//! * [`LineAssembler::close`] ends any unfinished line and closes every output.
//! * An output that returns an error is reported and dropped; the others
//!   carry on. Decoding never stops because a log file filled up.

use std::time::SystemTime;

/// What an output returns: `Err` carries a message for the user.
pub type SinkResult = Result<(), String>;

/// An output: something that receives lines as they are decoded.
///
/// A *trait* is Rust's way of saying "anything that can do these things".
/// The console, the text log and the database will each implement it, and
/// the assembler holds a list of them without caring which is which. Every
/// method except `name` has a default that does nothing, so an output only
/// writes the ones it needs (the console needs only `write`).
pub trait LineSink {
    /// A short name for warning messages.
    fn name(&self) -> &str;

    /// A line has started. `at` is the time and `quality` the 00 to 99 reading.
    fn line_start(&mut self, _at: SystemTime, _quality: u8) -> SinkResult {
        Ok(())
    }

    /// One character of the line, including the CR and LF that end it.
    fn write(&mut self, _ch: char) -> SinkResult {
        Ok(())
    }

    /// The line has ended.
    fn line_end(&mut self) -> SinkResult {
        Ok(())
    }

    /// The session is over: finish and release the file or database.
    fn close(&mut self) -> SinkResult {
        Ok(())
    }
}

pub struct LineAssembler {
    sinks: Vec<Box<dyn LineSink>>,
    clock: Box<dyn FnMut() -> SystemTime>,
    warn: Box<dyn FnMut(&str)>,
    /// A line has started and not yet ended.
    in_line: bool,
    /// The previous character was a CR.
    after_cr: bool,
}

impl LineAssembler {
    /// An assembler that reads the real clock and prints warnings to stderr.
    pub fn new(sinks: Vec<Box<dyn LineSink>>) -> Self {
        LineAssembler {
            sinks,
            clock: Box::new(SystemTime::now),
            warn: Box::new(|message| eprintln!("{message}")),
            in_line: false,
            after_cr: false,
        }
    }

    /// Replaces the clock (tests use a fixed one).
    pub fn with_clock(mut self, clock: impl FnMut() -> SystemTime + 'static) -> Self {
        self.clock = Box::new(clock);
        self
    }

    /// Replaces where warnings go (tests collect them).
    pub fn with_warn(mut self, warn: impl FnMut(&str) + 'static) -> Self {
        self.warn = Box::new(warn);
        self
    }

    /// Adds the next decoded character. `quality` is called only if this
    /// character starts a new line, because that is the only time the
    /// reading is wanted.
    pub fn feed(&mut self, ch: char, quality: impl FnOnce() -> u8) {
        if ch == '\n' && self.after_cr {
            // The second half of a CR LF pair: part of the line ending
            // already reported at the CR.
            self.after_cr = false;
            self.emit(|sink| sink.write(ch));
            return;
        }
        self.after_cr = false;

        if !self.in_line {
            let at = (self.clock)();
            let level = quality();
            self.emit(|sink| sink.line_start(at, level));
            self.in_line = true;
        }

        self.emit(|sink| sink.write(ch));

        if ch == '\r' || ch == '\n' {
            self.emit(|sink| sink.line_end());
            self.in_line = false;
            self.after_cr = ch == '\r';
        }
    }

    /// Ends the line in progress, if there is one, as if a CR LF had arrived.
    pub fn end_line(&mut self) {
        if self.in_line {
            // A line is in progress, so neither call starts a new one and
            // the quality is never read.
            self.feed('\r', || 0);
            self.feed('\n', || 0);
        }
    }

    /// Ends any unfinished line, then closes every output.
    pub fn close(&mut self) {
        if self.in_line {
            self.emit(|sink| sink.line_end());
            self.in_line = false;
        }
        for mut sink in std::mem::take(&mut self.sinks) {
            if let Err(error) = sink.close() {
                (self.warn)(&format!(
                    "[output warning] error closing {} ({error}) -- ignoring.",
                    sink.name()
                ));
            }
        }
    }

    /// Sends one event to every output. An output that fails is reported
    /// and removed.
    fn emit(&mut self, mut call: impl FnMut(&mut dyn LineSink) -> SinkResult) {
        let mut i = 0;
        while i < self.sinks.len() {
            match call(self.sinks[i].as_mut()) {
                Ok(()) => i += 1,
                Err(error) => {
                    let name = self.sinks[i].name().to_string();
                    (self.warn)(&format!(
                        "\n[output warning] {name} failed ({error}); output to it is disabled \
                         for the rest of this session."
                    ));
                    self.sinks.remove(i);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Recorder;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use std::time::{Duration, UNIX_EPOCH};

    /// An assembler with one recording output, and the recording.
    fn setup() -> (LineAssembler, Rc<RefCell<Vec<String>>>) {
        let rows = Rc::new(RefCell::new(Vec::new()));
        let assembler = LineAssembler::new(vec![Box::new(Recorder::new(&rows))]);
        (assembler, rows)
    }

    /// Feeds a string, reading the quality as `quality`.
    fn feed(assembler: &mut LineAssembler, text: &str, quality: u8) {
        for ch in text.chars() {
            assembler.feed(ch, || quality);
        }
    }

    fn rows(rows: &Rc<RefCell<Vec<String>>>) -> String {
        rows.borrow().join(" ")
    }

    #[test]
    fn text_without_a_line_ending_is_one_line_closed_at_the_end() {
        let (mut a, r) = setup();
        feed(&mut a, "AB", 40);
        assert_eq!(rows(&r), "S,40 W,65 W,66");
        a.close();
        assert_eq!(rows(&r), "S,40 W,65 W,66 E close");
    }

    #[test]
    fn a_cr_lf_pair_is_one_line_ending() {
        let (mut a, r) = setup();
        feed(&mut a, "A\r\nB", 7);
        // The LF is written but neither ends the line again nor starts one.
        assert_eq!(rows(&r), "S,7 W,65 W,13 E W,10 S,7 W,66");
    }

    #[test]
    fn a_lone_lf_ends_a_line() {
        let (mut a, r) = setup();
        feed(&mut a, "A\nB", 7);
        assert_eq!(rows(&r), "S,7 W,65 W,10 E S,7 W,66");
    }

    #[test]
    fn a_lone_cr_ends_a_line_and_the_next_text_starts_another() {
        let (mut a, r) = setup();
        feed(&mut a, "A\rB", 7);
        assert_eq!(rows(&r), "S,7 W,65 W,13 E S,7 W,66");
    }

    #[test]
    fn two_line_feeds_in_a_row_make_an_empty_line() {
        let (mut a, r) = setup();
        feed(&mut a, "\n\n", 3);
        assert_eq!(rows(&r), "S,3 W,10 E S,3 W,10 E");
    }

    #[test]
    fn the_quality_is_read_once_per_line_and_only_at_its_start() {
        let (mut a, _r) = setup();
        let reads = Cell::new(0);
        for ch in "AB\r\nCD\r\n".chars() {
            a.feed(ch, || {
                reads.set(reads.get() + 1);
                50
            });
        }
        assert_eq!(reads.get(), 2);
    }

    #[test]
    fn the_clock_is_read_once_per_line() {
        let rows_seen = Rc::new(RefCell::new(Vec::new()));
        let times = Rc::new(RefCell::new(Vec::new()));
        struct TimeSink(Rc<RefCell<Vec<u64>>>);
        impl LineSink for TimeSink {
            fn name(&self) -> &str {
                "TimeSink"
            }
            fn line_start(&mut self, at: SystemTime, _quality: u8) -> SinkResult {
                self.0.borrow_mut().push(at.duration_since(UNIX_EPOCH).unwrap().as_secs());
                Ok(())
            }
        }
        let tick = Rc::new(Cell::new(1000));
        let clock_tick = Rc::clone(&tick);
        let mut a = LineAssembler::new(vec![
            Box::new(TimeSink(Rc::clone(&times))),
            Box::new(Recorder::new(&rows_seen)),
        ])
        .with_clock(move || {
            clock_tick.set(clock_tick.get() + 1);
            UNIX_EPOCH + Duration::from_secs(clock_tick.get())
        });
        feed(&mut a, "AB\r\nCD\r\n", 1);
        // Two lines, so two readings, shared by both outputs.
        assert_eq!(*times.borrow(), vec![1001, 1002]);
    }

    #[test]
    fn end_line_acts_as_a_cr_lf() {
        let (mut a, r) = setup();
        feed(&mut a, "AB", 9);
        a.end_line();
        feed(&mut a, "C", 21);
        assert_eq!(rows(&r), "S,9 W,65 W,66 W,13 E W,10 S,21 W,67");
    }

    #[test]
    fn end_line_does_nothing_between_lines() {
        let (mut a, r) = setup();
        a.end_line();
        feed(&mut a, "A\r\n", 5);
        a.end_line();
        assert_eq!(rows(&r), "S,5 W,65 W,13 E W,10");
    }

    #[test]
    fn closing_twice_closes_each_output_once() {
        let (mut a, r) = setup();
        feed(&mut a, "A", 1);
        a.close();
        a.close();
        assert_eq!(rows(&r), "S,1 W,65 E close");
    }

    #[test]
    fn a_failing_output_is_dropped_and_the_others_carry_on() {
        struct Failing;
        impl LineSink for Failing {
            fn name(&self) -> &str {
                "Failing"
            }
            fn write(&mut self, _ch: char) -> SinkResult {
                Err("disk full".to_string())
            }
        }
        let recorded = Rc::new(RefCell::new(Vec::new()));
        let warnings = Rc::new(RefCell::new(Vec::<String>::new()));
        let collected = Rc::clone(&warnings);
        let mut a = LineAssembler::new(vec![Box::new(Failing), Box::new(Recorder::new(&recorded))])
            .with_warn(move |message| collected.borrow_mut().push(message.to_string()));
        feed(&mut a, "AB", 2);
        assert_eq!(rows(&recorded), "S,2 W,65 W,66", "the other output kept working");
        let warnings = warnings.borrow();
        assert_eq!(warnings.len(), 1, "it should be reported once, then dropped");
        assert!(warnings[0].contains("Failing failed (disk full)"), "{}", warnings[0]);
    }

    #[test]
    fn an_output_that_fails_to_close_is_reported() {
        struct BadClose;
        impl LineSink for BadClose {
            fn name(&self) -> &str {
                "BadClose"
            }
            fn close(&mut self) -> SinkResult {
                Err("locked".to_string())
            }
        }
        let warnings = Rc::new(RefCell::new(Vec::<String>::new()));
        let collected = Rc::clone(&warnings);
        let mut a = LineAssembler::new(vec![Box::new(BadClose)])
            .with_warn(move |message| collected.borrow_mut().push(message.to_string()));
        a.close();
        assert_eq!(warnings.borrow().len(), 1);
        assert!(warnings.borrow()[0].contains("error closing BadClose (locked)"));
    }
}
