//! The input-level meter (`--level`): a single line of text that is redrawn
//! about ten times a second, to check that the right audio device is chosen
//! and that its level is neither too low nor clipping.

use crate::pipeline::Pipeline;
use crate::run::run_with;
use crate::source::AudioSource;
use std::io::Write;
use std::sync::atomic::AtomicBool;

/// Samples per reading: 0.1 s at 48 kHz.
pub const READING_SAMPLES: usize = 4800;

/// The level of one block of audio, in decibels relative to full scale
/// (0 dBFS is a sample of +1.0 or -1.0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reading {
    pub rms_db: f64,
    pub peak_db: f64,
}

/// Cuts the audio into blocks of [`READING_SAMPLES`] and measures each one.
pub struct LevelMeter {
    block: usize,
    count: usize,
    sum_squares: f64,
    peak: f32,
}

impl LevelMeter {
    pub fn new(block: usize) -> Self {
        LevelMeter { block, count: 0, sum_squares: 0.0, peak: 0.0 }
    }

    /// Adds audio and returns a reading for every block that is now complete.
    /// A block may be spread over several chunks.
    pub fn push(&mut self, chunk: &[f32]) -> Vec<Reading> {
        let mut readings = Vec::new();
        for &sample in chunk {
            self.sum_squares += f64::from(sample) * f64::from(sample);
            self.peak = self.peak.max(sample.abs());
            self.count += 1;
            if self.count == self.block {
                let mean_square = self.sum_squares / self.block as f64;
                readings.push(Reading { rms_db: to_db(mean_square.sqrt()), peak_db: to_db(f64::from(self.peak)) });
                self.count = 0;
                self.sum_squares = 0.0;
                self.peak = 0.0;
            }
        }
        readings
    }
}

/// Decibels, with silence shown as -120 rather than minus infinity.
fn to_db(amplitude: f64) -> f64 {
    (20.0 * amplitude.log10()).max(-120.0)
}

/// Width of the bar, and the level at its left end (the right end is 0 dBFS).
const BAR_WIDTH: usize = 30;
const BAR_FLOOR_DB: f64 = -60.0;

/// One line of the meter, always the same length so that each redraw covers
/// the last one completely:
///
/// ```text
/// Level [##################------------]  -23.4 dBFS  peak  -10.2   Quality 37  Locked
/// ```
///
/// `CLIP` replaces the spaces after the peak when the peak reaches full scale.
pub fn render(reading: &Reading, quality: u8, locked: bool) -> String {
    let fraction = ((reading.rms_db - BAR_FLOOR_DB) / -BAR_FLOOR_DB).clamp(0.0, 1.0);
    let filled = (fraction * BAR_WIDTH as f64).round() as usize;
    let bar: String = "#".repeat(filled) + &"-".repeat(BAR_WIDTH - filled);
    let clip = if reading.peak_db >= -0.5 { "CLIP" } else { "    " };
    let lock = if locked { "Locked   " } else { "Searching" };
    format!(
        "Level [{bar}] {:6.1} dBFS  peak {:6.1} {clip}  Quality {quality:2}  {lock}",
        reading.rms_db, reading.peak_db
    )
}

/// Decodes like [`crate::run::run`] (nothing is output, because the pipeline
/// has no sinks) and draws the meter on `out` for each tenth of a second of
/// audio. Ends the last meter line when the run finishes.
pub fn run_level(
    source: &mut dyn AudioSource,
    pipeline: &mut Pipeline,
    stop: &AtomicBool,
    out: &mut dyn Write,
) -> Result<(), String> {
    let mut meter = LevelMeter::new(READING_SAMPLES);
    let result = run_with(source, pipeline, stop, &mut |chunk, pipeline| {
        for reading in meter.push(chunk) {
            // A failure to draw the meter (a closed pipe, say) is not worth stopping for.
            let _ = write!(out, "\r{}", render(&reading, pipeline.quality(), pipeline.is_locked()));
            let _ = out.flush();
        }
    });
    let _ = writeln!(out);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lines::LineAssembler;
    use crate::testutil;
    use crate::wav::WavSource;

    #[test]
    fn a_full_scale_square_wave_reads_zero_decibels() {
        let mut meter = LevelMeter::new(4);
        let readings = meter.push(&[1.0, -1.0, 1.0, -1.0]);
        assert_eq!(readings, vec![Reading { rms_db: 0.0, peak_db: 0.0 }]);
    }

    #[test]
    fn half_amplitude_is_six_decibels_down() {
        let mut meter = LevelMeter::new(2);
        let readings = meter.push(&[0.5, -0.5]);
        assert!((readings[0].rms_db + 6.0206).abs() < 1e-3, "{readings:?}");
        assert!((readings[0].peak_db + 6.0206).abs() < 1e-3);
    }

    #[test]
    fn the_rms_and_the_peak_are_different_things() {
        let mut meter = LevelMeter::new(4);
        let r = meter.push(&[0.0, 0.0, 0.0, 1.0])[0];
        assert!((r.rms_db + 6.0206).abs() < 1e-3, "rms of one 1.0 in four is 0.5: {r:?}");
        assert_eq!(r.peak_db, 0.0);
    }

    #[test]
    fn silence_reads_minus_120() {
        let mut meter = LevelMeter::new(3);
        assert_eq!(meter.push(&[0.0; 3]), vec![Reading { rms_db: -120.0, peak_db: -120.0 }]);
    }

    #[test]
    fn a_block_can_span_chunks_and_a_chunk_can_hold_several_blocks() {
        let mut meter = LevelMeter::new(4);
        assert!(meter.push(&[0.1, 0.1]).is_empty());
        assert_eq!(meter.push(&[0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.9]).len(), 2);
        // 1 sample of the next block is waiting; 3 more complete it.
        assert_eq!(meter.push(&[0.9, 0.9, 0.9]).len(), 1);
    }

    #[test]
    fn each_block_starts_afresh() {
        let mut meter = LevelMeter::new(2);
        let readings = meter.push(&[1.0, 1.0, 0.1, 0.1]);
        assert_eq!(readings[0].peak_db, 0.0);
        assert!((readings[1].peak_db + 20.0).abs() < 1e-6, "the peak of 1.0 is not carried over");
    }

    #[test]
    fn the_line_has_the_same_length_whatever_the_reading() {
        let readings = [
            Reading { rms_db: -120.0, peak_db: -120.0 },
            Reading { rms_db: -23.4, peak_db: -10.2 },
            Reading { rms_db: 0.0, peak_db: 0.0 },
        ];
        let lengths: Vec<usize> = readings
            .iter()
            .flat_map(|r| [render(r, 0, false).len(), render(r, 99, true).len()])
            .collect();
        assert!(lengths.iter().all(|&n| n == lengths[0]), "{lengths:?}");
    }

    #[test]
    fn the_line_shows_the_bar_the_clip_warning_the_quality_and_the_lock() {
        let quiet = render(&Reading { rms_db: -60.0, peak_db: -50.0 }, 5, false);
        assert_eq!(quiet, "Level [------------------------------]  -60.0 dBFS  peak  -50.0       Quality  5  Searching");
        let half = render(&Reading { rms_db: -30.0, peak_db: -12.5 }, 37, true);
        assert_eq!(half, "Level [###############---------------]  -30.0 dBFS  peak  -12.5       Quality 37  Locked   ");
        let loud = render(&Reading { rms_db: 0.0, peak_db: 0.0 }, 99, true);
        assert_eq!(loud, "Level [##############################]    0.0 dBFS  peak    0.0 CLIP  Quality 99  Locked   ");
        assert!(!render(&Reading { rms_db: -3.0, peak_db: -0.6 }, 0, false).contains("CLIP"));
        assert!(render(&Reading { rms_db: -3.0, peak_db: -0.4 }, 0, false).contains("CLIP"));
    }

    #[test]
    fn a_recording_gives_ten_readings_a_second_and_the_pipeline_quality() {
        let path = testutil::vector_path("real_clip1", "input.wav");
        let mut source = WavSource::open(&path, 48_000).unwrap();
        let samples = testutil::read_wav(&path).len();
        let mut pipeline = Pipeline::with_defaults(LineAssembler::new(Vec::new()));
        let mut out = Vec::new();
        run_level(&mut source, &mut pipeline, &AtomicBool::new(false), &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let redraws = text.matches('\r').count();
        assert_eq!(redraws, samples / READING_SAMPLES);
        assert!(text.ends_with('\n'));
        // The recording holds a signal, so the meter must have seen the lock and some quality.
        assert!(text.contains("Locked"), "the pipeline should have locked on the recording");
        let last = text.trim_end().rsplit('\r').next().unwrap();
        assert!(last.starts_with("Level ["), "{last}");
    }
}
