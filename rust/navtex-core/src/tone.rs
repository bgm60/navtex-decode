//! Mark/space tone detection: measures how much of each tone is in a frame.

use std::f64::consts::PI;

/// What the detector reports for one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneSample {
    /// `(mark - space) / (mark + space)`, between -1 and +1. It does not
    /// depend on how loud the signal is.
    pub diff: f64,
    /// Hard decision: `true` = mark (a 1), `false` = space (a 0).
    pub bit: bool,
}

/// Measures the energy at the mark and space frequencies in a frame.
///
/// For each tone this is a single-frequency DFT: multiply the frame by a
/// cosine and a sine at that frequency, add up each product, and square
/// and add the two totals. The cos/sin tables only depend on the frame
/// length, the sample rate and the two frequencies, so they are worked out
/// once, when the detector is created.
pub struct ToneDetector {
    mark_cos: Vec<f64>,
    mark_sin: Vec<f64>,
    space_cos: Vec<f64>,
    space_sin: Vec<f64>,
}

impl ToneDetector {
    pub fn new(frame_len: usize, sample_rate: f64, mark_freq: f64, space_freq: f64) -> Self {
        // The angle at sample `t` of a tone at `freq` Hz, in radians.
        let angle = |freq: f64, t: usize| 2.0 * PI * freq * t as f64 / sample_rate;
        ToneDetector {
            mark_cos: (0..frame_len).map(|t| angle(mark_freq, t).cos()).collect(),
            mark_sin: (0..frame_len).map(|t| angle(mark_freq, t).sin()).collect(),
            space_cos: (0..frame_len).map(|t| angle(space_freq, t).cos()).collect(),
            space_sin: (0..frame_len).map(|t| angle(space_freq, t).sin()).collect(),
        }
    }

    /// Measures one already-windowed frame. The frame must be exactly the
    /// length the detector was created for.
    pub fn process(&self, frame: &[f32]) -> ToneSample {
        assert_eq!(frame.len(), self.mark_cos.len(), "frame has the wrong length");
        let mark = tone_power(frame, &self.mark_cos, &self.mark_sin);
        let space = tone_power(frame, &self.space_cos, &self.space_sin);
        let total = mark + space;
        let diff = if total > 1e-12 { (mark - space) / total } else { 0.0 };
        ToneSample { diff, bit: mark >= space }
    }
}

/// Energy at one frequency: (sum of x*cos)^2 + (sum of x*sin)^2.
/// The samples are `f32`, but the sums are done in `f64`.
fn tone_power(frame: &[f32], cos: &[f64], sin: &[f64]) -> f64 {
    let mut re = 0.0_f64;
    let mut im = 0.0_f64;
    for ((&x, &c), &s) in frame.iter().zip(cos).zip(sin) {
        let x = f64::from(x);
        re += x * c;
        im += x * s;
    }
    re * re + im * im
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil;
    use crate::window::hamming_periodic;

    const SAMPLE_RATE: f64 = 48_000.0;
    const MARK: f64 = 1785.0;
    const SPACE: f64 = 1615.0;
    const FRAME: usize = 480;

    /// One windowed frame of a pure tone.
    fn tone_frame(freq: f64) -> Vec<f32> {
        let window = hamming_periodic(FRAME);
        (0..FRAME)
            .map(|t| {
                let x = (2.0 * PI * freq * t as f64 / SAMPLE_RATE).sin() as f32;
                x * window[t]
            })
            .collect()
    }

    #[test]
    fn silence_gives_zero_diff() {
        let detector = ToneDetector::new(FRAME, SAMPLE_RATE, MARK, SPACE);
        let sample = detector.process(&vec![0.0_f32; FRAME]);
        assert_eq!(sample.diff, 0.0);
    }

    #[test]
    fn a_mark_tone_reads_as_mark_and_a_space_tone_as_space() {
        let detector = ToneDetector::new(FRAME, SAMPLE_RATE, MARK, SPACE);
        let mark = detector.process(&tone_frame(MARK));
        let space = detector.process(&tone_frame(SPACE));
        assert!(mark.bit && mark.diff > 0.9, "mark tone gave {mark:?}");
        assert!(!space.bit && space.diff < -0.9, "space tone gave {space:?}");
    }

    #[test]
    fn matches_the_python_tone_csv() {
        let set = "synth_snr10_seed1";
        let samples = testutil::read_wav(&testutil::vector_path(set, "input.wav"));
        let expected = testutil::read_tone_csv(&testutil::vector_path(set, "tone.csv"));

        let hop = 60;
        let window = hamming_periodic(FRAME);
        let detector = ToneDetector::new(FRAME, SAMPLE_RATE, MARK, SPACE);

        let frame_count = (samples.len() - FRAME) / hop + 1;
        assert_eq!(frame_count, expected.len(), "different number of frames");

        let mut windowed = vec![0.0_f32; FRAME];
        let mut worst_diff = 0.0_f64;
        let mut bit_mismatches = Vec::new();
        for k in 0..frame_count {
            let start = k * hop;
            for i in 0..FRAME {
                windowed[i] = samples[start + i] * window[i];
            }
            let got = detector.process(&windowed);
            let (expected_diff, expected_bit) = expected[k];
            worst_diff = worst_diff.max((got.diff - expected_diff).abs());
            if got.bit != expected_bit {
                bit_mismatches.push(k);
            }
        }
        println!("{frame_count} frames; largest diff error {worst_diff:e}; bit mismatches {}",
                 bit_mismatches.len());
        assert!(bit_mismatches.is_empty(), "bits differ at frames {bit_mismatches:?}");
        assert!(worst_diff <= 1e-9, "diff error {worst_diff:e} is too large");
    }
}
