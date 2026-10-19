//! The demodulator: audio in, one soft decision per bit out.
//!
//! This joins the first three stages of the decoder:
//!
//! ```text
//! audio chunks -> FrontEnd (window + tone detector) -> BitSync -> bit decisions
//! ```
//!
//! Audio can arrive in chunks of any size, so it works the same for a live
//! sound card and for a file read in blocks.

use crate::bitsync::{BitDecision, BitSync, DEFAULT_LOOP_GAIN};
use crate::frontend::{FrontEnd, FrontEndConfig};

pub struct Demodulator {
    front_end: FrontEnd,
    bit_sync: BitSync,
}

impl Demodulator {
    pub fn new(config: &FrontEndConfig, loop_gain: f64) -> Self {
        let front_end = FrontEnd::new(config);
        let bit_sync = BitSync::new(loop_gain, front_end.phase_step());
        Demodulator { front_end, bit_sync }
    }

    /// Feeds in the next chunk of audio (mono, `f32`, any length) and
    /// returns the bit decisions it completes, which may be none.
    pub fn push(&mut self, chunk: &[f32]) -> Vec<BitDecision> {
        self.front_end
            .push(chunk)
            .into_iter()
            .filter_map(|tone| self.bit_sync.push(tone))
            .collect()
    }
}

/// The settings the Python decoder uses by default.
impl Default for Demodulator {
    fn default() -> Self {
        Demodulator::new(&FrontEndConfig::default(), DEFAULT_LOOP_GAIN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil;

    /// Runs the whole signal through a fresh demodulator in chunks of `chunk_len`.
    fn run(samples: &[f32], chunk_len: usize) -> Vec<BitDecision> {
        let mut demod = Demodulator::default();
        let mut bits = Vec::new();
        for chunk in samples.chunks(chunk_len) {
            bits.extend(demod.push(chunk));
        }
        bits
    }

    #[test]
    fn matches_python_bits_for_every_reference_set() {
        for set in testutil::reference_sets() {
            let samples = testutil::read_wav(&testutil::vector_path(&set, "input.wav"));
            let expected = testutil::read_bits_csv(&testutil::vector_path(&set, "bits.csv"));

            let got = run(&samples, 4096);
            assert_eq!(got.len(), expected.len(), "{set}: different number of bits");

            let mut bit_mismatches = Vec::new();
            let mut worst = 0.0_f64;
            for (k, (g, &(bit, confidence, soft_value))) in got.iter().zip(&expected).enumerate() {
                if g.bit != bit {
                    bit_mismatches.push(k);
                }
                worst = worst
                    .max((g.soft_value - soft_value).abs())
                    .max((g.confidence - confidence).abs());
            }
            println!(
                "{set}: {} bits, bit mismatches {}, largest soft_value or confidence error {worst:e}",
                got.len(),
                bit_mismatches.len()
            );
            assert!(bit_mismatches.is_empty(), "{set}: bits differ at {bit_mismatches:?}");
            assert!(worst <= 1e-9, "{set}: error {worst:e} is too large");
        }
    }

    #[test]
    fn the_chunk_size_does_not_change_the_result() {
        let set = &testutil::reference_sets()[0];
        let samples = testutil::read_wav(&testutil::vector_path(set, "input.wav"));
        let reference = run(&samples, 4096);
        for chunk_len in [7, 777, 48_000, samples.len()] {
            assert_eq!(run(&samples, chunk_len), reference, "{set}: chunk length {chunk_len}");
        }
    }

    #[test]
    fn the_first_bit_arrives_with_the_900th_sample() {
        // The first frame needs 480 samples, and each of the next seven
        // frames needs 60 more: 480 + 7 * 60 = 900 samples for a first bit.
        let mut demod = Demodulator::default();
        assert!(demod.push(&vec![0.0; 899]).is_empty(), "only seven frames so far");
        assert_eq!(demod.push(&[0.0]).len(), 1, "the 900th sample completes the eighth frame");
    }
}
