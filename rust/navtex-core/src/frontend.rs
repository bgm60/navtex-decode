//! The front end: windowing and tone detection together.
//!
//! Audio arrives in chunks of any size. Every `hop` samples a frame one
//! symbol long is cut from it, multiplied by the window, and measured by
//! the tone detector. With the default settings that is eight overlapping
//! frames per bit.

use crate::tone::{ToneDetector, ToneSample};
use crate::window::hamming_periodic;

/// NAVTEX / SITOR-B symbol rate.
const BAUD: f64 = 100.0;

/// Front-end settings. The defaults match the Python decoder's.
#[derive(Debug, Clone)]
pub struct FrontEndConfig {
    pub sample_rate: u32,
    /// Analysis frames produced per bit period.
    pub oversample: u32,
    pub mark_freq: f64,
    pub space_freq: f64,
}

impl Default for FrontEndConfig {
    fn default() -> Self {
        FrontEndConfig {
            sample_rate: 48_000,
            oversample: 8,
            mark_freq: 1785.0,
            space_freq: 1615.0,
        }
    }
}

pub struct FrontEnd {
    window: Vec<f32>,
    detector: ToneDetector,
    /// How far the frame start moves between frames, in samples.
    hop: usize,
    /// Samples received that are not yet finished with.
    buffer: Vec<f32>,
    /// Scratch space for one windowed frame, reused every time.
    frame: Vec<f32>,
}

impl FrontEnd {
    pub fn new(config: &FrontEndConfig) -> Self {
        // One frame is one bit period long: 480 samples at 48 kHz.
        let frame_len = (f64::from(config.sample_rate) / BAUD).round() as usize;
        let hop = (frame_len / config.oversample as usize).max(1);
        FrontEnd {
            window: hamming_periodic(frame_len),
            detector: ToneDetector::new(
                frame_len,
                f64::from(config.sample_rate),
                config.mark_freq,
                config.space_freq,
            ),
            hop,
            buffer: Vec::new(),
            frame: vec![0.0; frame_len],
        }
    }

    /// How far the bit-clock phase advances per frame: the hop divided by
    /// the frame length (0.125 with the default settings).
    pub fn phase_step(&self) -> f64 {
        self.hop as f64 / self.window.len() as f64
    }
    
    /// Feeds in the next chunk of audio (mono, `f32`, any length) and
    /// returns one `ToneSample` for every frame that is now complete.
    pub fn push(&mut self, chunk: &[f32]) -> Vec<ToneSample> {
        self.buffer.extend_from_slice(chunk);
        let frame_len = self.window.len();

        let mut samples = Vec::new();
        let mut start = 0;
        while self.buffer.len() - start >= frame_len {
            for i in 0..frame_len {
                self.frame[i] = self.buffer[start + i] * self.window[i];
            }
            samples.push(self.detector.process(&self.frame));
            start += self.hop;
        }
        // Throw away the samples that no later frame needs.
        self.buffer.drain(..start);
        samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil;

    const SETS: [&str; 5] = ["synth_snr10_seed1", "synth_clean_seed2", "synth_fade_snr15", "real_clip1", "real_clip2"];

    /// Runs the whole signal through a fresh front end in chunks of `chunk_len`.
    fn run(samples: &[f32], chunk_len: usize) -> Vec<ToneSample> {
        let mut front_end = FrontEnd::new(&FrontEndConfig::default());
        let mut out = Vec::new();
        for chunk in samples.chunks(chunk_len) {
            out.extend(front_end.push(chunk));
        }
        out
    }

    #[test]
    fn matches_python_for_every_reference_set() {
        for set in SETS {
            let samples = testutil::read_wav(&testutil::vector_path(set, "input.wav"));
            let expected = testutil::read_tone_csv(&testutil::vector_path(set, "tone.csv"));

            let got = run(&samples, 4096);
            assert_eq!(got.len(), expected.len(), "{set}: different number of frames");

            let mut worst = 0.0_f64;
            for (k, (sample, &(expected_diff, expected_bit))) in
                got.iter().zip(&expected).enumerate()
            {
                assert_eq!(sample.bit, expected_bit, "{set}: bit differs at frame {k}");
                worst = worst.max((sample.diff - expected_diff).abs());
            }
            println!("{set}: {} frames, largest diff error {worst:e}", got.len());
            assert!(worst <= 1e-9, "{set}: diff error {worst:e} is too large");
        }
    }

    #[test]
    fn the_chunk_size_does_not_change_the_result() {
        let set = SETS[0];
        let samples = testutil::read_wav(&testutil::vector_path(set, "input.wav"));
        let reference = run(&samples, 4096);
        for chunk_len in [7, 777, 48_000, samples.len()] {
            assert_eq!(run(&samples, chunk_len), reference, "chunk length {chunk_len}");
        }
    }

    #[test]
    fn too_little_audio_gives_no_frames_and_more_audio_completes_one() {
        let mut front_end = FrontEnd::new(&FrontEndConfig::default());
        assert!(front_end.push(&vec![0.0; 479]).is_empty());
        assert_eq!(front_end.push(&[0.0]).len(), 1); // the 480th sample completes a frame
    }
}
