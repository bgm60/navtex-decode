//! The complete decoder: audio in, characters out.
//!
//! This joins the two halves built so far:
//!
//! ```text
//! audio chunks -> Demodulator -> soft bit decisions -> SoftLock -> characters
//! ```
//!
//! Audio can arrive in chunks of any size, so it works the same for a live
//! sound card and for a file read in blocks. Call [`Decoder::finish`] at the
//! end of a recording to get the characters still being held back.

use crate::bitsync::DEFAULT_LOOP_GAIN;
use crate::demod::Demodulator;
use crate::frontend::FrontEndConfig;
use crate::params::SoftLockParams;
use crate::softlock::{LockEvent, SoftLock};

/// The most bit confidences kept waiting for [`Decoder::take_confidences`];
/// if the caller never collects them, the oldest are discarded.
const MAX_CONFIDENCES: usize = 100_000;

pub struct Decoder {
    demodulator: Demodulator,
    lock: SoftLock,
    /// The confidence of each bit decision since the last call to
    /// `take_confidences`.
    confidences: Vec<f64>,
}

impl Decoder {
    pub fn new(config: &FrontEndConfig, loop_gain: f64, params: SoftLockParams) -> Self {
        Decoder {
            demodulator: Demodulator::new(config, loop_gain),
            lock: SoftLock::new(params),
            confidences: Vec::new(),
        }
    }

    /// Feeds in the next chunk of audio (mono, `f32`, any length) and
    /// returns the characters that become ready, which is usually none.
    pub fn push(&mut self, chunk: &[f32]) -> Vec<LockEvent> {
        let mut events = Vec::new();
        for decision in self.demodulator.push(chunk) {
            self.confidences.push(decision.confidence);
            events.extend(self.lock.push(decision.soft_value));
        }
        if self.confidences.len() > MAX_CONFIDENCES {
            let excess = self.confidences.len() - MAX_CONFIDENCES;
            self.confidences.drain(..excess);
        }
        events
    }

    /// Call at the end of the audio: returns what is still held back.
    pub fn finish(&mut self) -> Vec<LockEvent> {
        self.lock.finish()
    }

    /// Returns the confidence (0 to 1) of every bit decision made since the
    /// last call, and forgets them. A program can average these to show
    /// signal strength.
    pub fn take_confidences(&mut self) -> Vec<f64> {
        std::mem::take(&mut self.confidences)
    }

    /// True while the decoder holds a lock on a signal.
    pub fn is_locked(&self) -> bool {
        self.lock.is_locked()
    }

    /// How many times a lock has been released.
    pub fn releases(&self) -> usize {
        self.lock.releases()
    }

    /// How many times the lock has moved to another alignment.
    pub fn switches(&self) -> usize {
        self.lock.switches()
    }
}

/// The settings the Python decoder uses by default.
impl Default for Decoder {
    fn default() -> Self {
        Decoder::new(&FrontEndConfig::default(), DEFAULT_LOOP_GAIN, SoftLockParams::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil;

    /// Runs the whole signal through a fresh decoder in chunks of `chunk_len`.
    fn run(samples: &[f32], chunk_len: usize) -> Vec<LockEvent> {
        let mut decoder = Decoder::default();
        let mut events = Vec::new();
        for chunk in samples.chunks(chunk_len) {
            events.extend(decoder.push(chunk));
        }
        events.extend(decoder.finish());
        events
    }

    #[test]
    fn audio_to_text_matches_python_for_every_audio_set() {
        for set in testutil::reference_sets() {
            let samples = testutil::read_wav(&testutil::vector_path(&set, "input.wav"));
            let events = run(&samples, 4096);

            // Every character, and the bit it came from.
            let want = testutil::read_lock_chars(&testutil::vector_path(&set, "lock_full_chars.csv"));
            let got: Vec<(usize, String)> =
                events.iter().map(|e| (e.bit_index, e.ch.to_string())).collect();
            assert_eq!(got.len(), want.len(), "{set}: number of characters");
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g, w, "{set}: character {k}");
            }

            // The text itself (carriage returns ignored: see the lock tests).
            let text: String = events.iter().map(|e| e.ch).filter(|&c| c != '\r').collect();
            let want_text = std::fs::read_to_string(testutil::vector_path(&set, "decoded_full.txt"))
                .expect("cannot read decoded_full.txt")
                .replace('\r', "");
            assert_eq!(text, want_text, "{set}: decoded text");
            println!("{set}: {} samples, {} characters, identical to Python", samples.len(), events.len());
        }
    }

    #[test]
    fn the_chunk_size_does_not_change_the_text() {
        // Two sets that between them lock, follow a bit slip and drop a
        // run of identical letters.
        for set in ["real_clip1", "scen_idle_gap"] {
            let samples = testutil::read_wav(&testutil::vector_path(set, "input.wav"));
            let reference = run(&samples, 4096);
            assert!(!reference.is_empty(), "{set}: no text decoded");
            for chunk_len in [7, 777, 48_000, samples.len()] {
                assert_eq!(run(&samples, chunk_len), reference, "{set}: chunk length {chunk_len}");
            }
        }
    }

    #[test]
    fn confidences_are_collected_and_handed_over_once() {
        let set = "scen_weak";
        let samples = testutil::read_wav(&testutil::vector_path(set, "input.wav"));
        let want: Vec<f64> = testutil::read_bits_csv(&testutil::vector_path(set, "bits.csv"))
            .iter()
            .map(|&(_, confidence, _)| confidence)
            .collect();

        let mut decoder = Decoder::default();
        let mut got = Vec::new();
        for chunk in samples.chunks(5000) {
            decoder.push(chunk);
            got.extend(decoder.take_confidences());
        }
        assert!(decoder.take_confidences().is_empty(), "they should be handed over only once");
        assert_eq!(got.len(), want.len());
        let worst = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0, f64::max);
        assert!(worst <= 1e-9, "confidence error {worst:e}");
    }

    #[test]
    fn uncollected_confidences_are_capped() {
        let mut decoder = Decoder::default();
        // 1,100 seconds of silence is 110,000 bits.
        decoder.push(&vec![0.0; 48_000 * 1100]);
        assert_eq!(decoder.take_confidences().len(), MAX_CONFIDENCES);
    }

    #[test]
    fn silence_gives_no_text_and_no_lock() {
        let mut decoder = Decoder::default();
        let mut events = decoder.push(&vec![0.0; 48_000 * 20]);
        events.extend(decoder.finish());
        assert!(events.is_empty());
        assert!(!decoder.is_locked());
    }
}
