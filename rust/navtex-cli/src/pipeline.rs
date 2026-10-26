//! The whole chain from audio to line events:
//!
//! ```text
//! audio -> Demodulator -> bit decisions -> SoftLock -> characters -> LineAssembler -> outputs
//!                              |
//!                              +-> QualityMeter (read when a line starts)
//! ```
//!
//! `navtex-core` has a `Decoder` that joins the first three stages, but it
//! hands back a whole chunk's worth of results at once. This program needs
//! the quality reading *as it was when each character came out*, so the
//! pieces are driven here one bit decision at a time, exactly as the
//! Python session does: update the meter with the bit, give the bit to the
//! lock, and handle any characters that result.

use crate::lines::LineAssembler;
use crate::quality::{DEFAULT_WINDOW, QualityMeter};
use navtex_core::bitsync::DEFAULT_LOOP_GAIN;
use navtex_core::demod::Demodulator;
use navtex_core::frontend::FrontEndConfig;
use navtex_core::params::SoftLockParams;
use navtex_core::softlock::{LockEvent, SoftLock};

pub struct Pipeline {
    demodulator: Demodulator,
    lock: SoftLock,
    meter: QualityMeter,
    assembler: LineAssembler,
    /// The release count of the last character seen. When it changes, the
    /// lock was released in between, and the line in progress is ended.
    releases: usize,
}

impl Pipeline {
    pub fn new(
        config: &FrontEndConfig,
        loop_gain: f64,
        params: SoftLockParams,
        quality_window: usize,
        assembler: LineAssembler,
    ) -> Self {
        Pipeline {
            demodulator: Demodulator::new(config, loop_gain),
            lock: SoftLock::new(params),
            meter: QualityMeter::new(quality_window),
            assembler,
            releases: 0,
        }
    }

    /// The settings the Python decoder uses by default.
    pub fn with_defaults(assembler: LineAssembler) -> Self {
        Pipeline::new(
            &FrontEndConfig::default(),
            DEFAULT_LOOP_GAIN,
            SoftLockParams::default(),
            DEFAULT_WINDOW,
            assembler,
        )
    }

    /// Feeds in the next chunk of audio (mono, `f32`, any length).
    pub fn push(&mut self, chunk: &[f32]) {
        for decision in self.demodulator.push(chunk) {
            self.meter.update(decision.confidence);
            for event in self.lock.push(decision.soft_value) {
                self.handle(event);
            }
        }
    }

    /// Call at the end of the audio: releases the characters still held
    /// back, ends the last line and closes the outputs.
    pub fn finish(&mut self) {
        for event in self.lock.finish() {
            self.handle(event);
        }
        self.assembler.close();
    }

    /// The current signal quality reading, 0 to 99.
    pub fn quality(&self) -> u8 {
        self.meter.level()
    }

    /// True while the decoder holds a lock on a signal.
    pub fn is_locked(&self) -> bool {
        self.lock.is_locked()
    }

    fn handle(&mut self, event: LockEvent) {
        if event.releases != self.releases {
            // The lock was released since the last character: end the line
            // in progress, so the next text starts a fresh one.
            self.releases = event.releases;
            self.assembler.end_line();
        }
        let meter = &self.meter;
        self.assembler.feed(event.ch, || meter.level());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{self, Recorder};
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Decodes a whole signal in chunks of `chunk_len` and returns the line
    /// events as rows of text (without the final "close" row).
    fn run(samples: &[f32], chunk_len: usize) -> Vec<String> {
        let rows = Rc::new(RefCell::new(Vec::new()));
        let mut pipeline =
            Pipeline::with_defaults(LineAssembler::new(vec![Box::new(Recorder::new(&rows))]));
        for chunk in samples.chunks(chunk_len) {
            pipeline.push(chunk);
        }
        pipeline.finish();
        let mut rows = rows.borrow().clone();
        assert_eq!(rows.pop().as_deref(), Some("close"), "the output should be closed last");
        rows
    }

    /// The main test: every line start (with its quality reading), every
    /// character and every line end must match what the Python decoder sent
    /// to its outputs, for every recording.
    #[test]
    fn line_events_match_python_for_every_recording() {
        let sets = testutil::sets_with("lines.csv");
        assert!(sets.len() >= 10, "too few sets: {}", sets.len());
        for set in sets {
            let samples = testutil::read_wav(&testutil::vector_path(&set, "input.wav"));
            let got = run(&samples, 4096);
            let want = testutil::read_line_events(&set);
            assert_eq!(got.len(), want.len(), "{set}: number of events");
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g, w, "{set}: event {k}");
            }
            let lines = want.iter().filter(|row| row.as_str() == "E").count();
            println!("{set}: {} events, {lines} lines, identical to Python", want.len());
        }
    }

    #[test]
    fn the_chunk_size_does_not_change_the_lines() {
        // real_clip1 locks and loses its lock; scen_two_messages breaks a line at a release.
        for set in ["real_clip1", "scen_two_messages"] {
            let samples = testutil::read_wav(&testutil::vector_path(set, "input.wav"));
            let reference = run(&samples, 4096);
            assert!(reference.len() > 20, "{set}: no text decoded");
            for chunk_len in [7, 777, 48_000, samples.len()] {
                assert_eq!(run(&samples, chunk_len), reference, "{set}: chunk length {chunk_len}");
            }
        }
    }

    #[test]
    fn silence_gives_no_lines_and_a_quality_of_zero() {
        let rows = Rc::new(RefCell::new(Vec::new()));
        let mut pipeline =
            Pipeline::with_defaults(LineAssembler::new(vec![Box::new(Recorder::new(&rows))]));
        pipeline.push(&vec![0.0; 48_000 * 20]);
        assert_eq!(pipeline.quality(), 0);
        assert!(!pipeline.is_locked());
        pipeline.finish();
        assert_eq!(*rows.borrow(), vec!["close".to_string()]);
    }
}
