//! WAV file input.
//!
//! Reads a recording a block at a time, so a recording many hours long never
//! has to fit in memory, and gives the decoder mono `f32` samples exactly as
//! the Python program's `FileSource` does:
//!
//! * integer samples are divided by 2^(bits-1) (so 16-bit values are
//!   divided by 32768, and the results lie between -1 and just under +1);
//!   8-bit samples are centred on 128 first; 32-bit float samples are used as they are;
//! * more than one channel is averaged to mono, adding in single precision
//!   in channel order and dividing by the channel count, like numpy's `mean`;
//! * the sample rate must already be the decoder's (48 kHz): resampling is
//!   not implemented, so any other rate is an error.
//!
//! Supported: WAV files holding 8, 16, 24 or 32-bit integer samples or 32-bit
//! float samples. Other file types and sample formats are reported as errors.

use crate::source::AudioSource;
use hound::{SampleFormat, WavReader};
use std::fs::File;
use std::io::BufReader;

/// Frames read at a time, as in the Python program.
pub const BLOCK_FRAMES: usize = 4096;

pub struct WavSource {
    path: String,
    reader: WavReader<BufReader<File>>,
    format: SampleFormat,
    bits: u16,
    channels: usize,
    block_frames: usize,
    /// An error found part-way through a block, reported after the samples
    /// read before it have been handed over.
    pending_error: Option<String>,
    finished: bool,
}

impl WavSource {
    /// Opens a WAV file, checking that it is a supported format at
    /// `required_rate` Hz.
    pub fn open(path: &str, required_rate: u32) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
        let reader = WavReader::new(BufReader::new(file)).map_err(|e| describe(path, &e))?;
        let spec = reader.spec();
        let supported = match spec.sample_format {
            SampleFormat::Int => matches!(spec.bits_per_sample, 8 | 16 | 24 | 32),
            SampleFormat::Float => spec.bits_per_sample == 32,
        };
        if !supported {
            return Err(describe(path, &hound::Error::Unsupported));
        }
        if spec.sample_rate != required_rate {
            return Err(format!(
                "{path}: the sample rate is {} Hz, but only {required_rate} Hz is supported \
                 (resampling is not implemented)",
                spec.sample_rate
            ));
        }
        Ok(WavSource {
            path: path.to_string(),
            reader,
            format: spec.sample_format,
            bits: spec.bits_per_sample,
            channels: usize::from(spec.channels),
            block_frames: BLOCK_FRAMES,
            pending_error: None,
            finished: false,
        })
    }

    /// Changes how many frames are read at a time (tests use small blocks).
    pub fn with_block_frames(mut self, frames: usize) -> Self {
        self.block_frames = frames.max(1);
        self
    }

    /// Reads up to one block of frames, turning each sample into `f32` with
    /// `convert` and averaging the channels. Returns the samples and, if the
    /// file turned out to be damaged, the problem. A frame that is only
    /// partly there is dropped.
    fn read_block<S: hound::Sample>(&mut self, convert: impl Fn(S) -> f32) -> (Vec<f32>, Option<String>) {
        let channels = self.channels;
        let mut block = Vec::with_capacity(self.block_frames);
        let mut problem = None;
        let mut samples = self.reader.samples::<S>();
        'frames: for _ in 0..self.block_frames {
            let mut sum = 0.0_f32;
            for channel in 0..channels {
                match samples.next() {
                    Some(Ok(sample)) => {
                        let value = convert(sample);
                        // numpy starts a sum from the first value, not from 0.0.
                        sum = if channel == 0 { value } else { sum + value };
                    }
                    Some(Err(error)) => {
                        problem = Some(format!(
                            "{}: the file ended early or is damaged ({error})",
                            self.path
                        ));
                        break 'frames;
                    }
                    None => break 'frames,
                }
            }
            block.push(if channels == 1 { sum } else { sum / channels as f32 });
        }
        (block, problem)
    }
}

impl AudioSource for WavSource {
    fn description(&self) -> String {
        format!("WAV file '{}'", self.path)
    }

    fn next_chunk(&mut self) -> Result<Option<Vec<f32>>, String> {
        if let Some(error) = self.pending_error.take() {
            self.finished = true;
            return Err(error);
        }
        if self.finished {
            return Ok(None);
        }
        let (block, problem) = match self.format {
            SampleFormat::Float => self.read_block::<f32>(|sample| sample),
            SampleFormat::Int => {
                // A power of two, so the scaling is exact.
                let scale = 1.0 / (1_u64 << (self.bits - 1)) as f32;
                self.read_block::<i32>(move |sample| sample as f32 * scale)
            }
        };
        if let Some(error) = problem {
            if block.is_empty() {
                self.finished = true;
                return Err(error);
            }
            self.pending_error = Some(error);
        } else if block.len() < self.block_frames {
            self.finished = true; // the end of the data
        }
        if block.is_empty() { Ok(None) } else { Ok(Some(block)) }
    }
}

/// A readable message for a problem found while opening a file.
fn describe(path: &str, error: &hound::Error) -> String {
    match error {
        hound::Error::FormatError(detail) => format!("{path}: not a readable WAV file ({detail})"),
        hound::Error::Unsupported => format!(
            "{path}: unsupported WAV format (only 8, 16, 24 and 32-bit integer and 32-bit float \
             samples are supported)"
        ),
        hound::Error::IoError(io) => format!("{path}: cannot read the file ({io})"),
        other => format!("{path}: not a readable WAV file ({other})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{self, TempDir};

    fn read_all(mut source: WavSource) -> Vec<f32> {
        let mut out = Vec::new();
        while let Some(chunk) = source.next_chunk().expect("read error") {
            out.extend(chunk);
        }
        out
    }

    fn format_path(name: &str, extension: &str) -> String {
        testutil::vector_path("wav_formats", &format!("{name}.{extension}"))
    }

    /// The samples Python's FileSource gets from the file, from `<name>.f32`.
    fn python_samples(name: &str) -> Vec<f32> {
        let bytes = std::fs::read(format_path(name, "f32")).expect("cannot read the reference samples");
        bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    const FORMATS: [&str; 9] = [
        "mono_u8", "mono_16", "stereo_16", "three_16", "mono_24", "stereo_24", "mono_32",
        "mono_float", "stereo_float",
    ];

    /// The main test: every sample format and channel count must give exactly
    /// the same float values as Python, bit for bit.
    #[test]
    fn every_format_gives_python_samples_exactly() {
        for name in FORMATS {
            let source = WavSource::open(&format_path(name, "wav"), 48_000).expect(name);
            let got = read_all(source);
            let want = python_samples(name);
            assert_eq!(got.len(), want.len(), "{name}: number of samples");
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "{name}: sample {k}: got {g}, Python {w}");
            }
            println!("{name}: {} samples identical to Python", want.len());
        }
    }

    #[test]
    fn the_block_size_does_not_change_the_samples() {
        for name in ["stereo_16", "three_16", "mono_24"] {
            let want = python_samples(name);
            for block in [1, 7, 1000, 4096, 10_000] {
                let source = WavSource::open(&format_path(name, "wav"), 48_000).unwrap().with_block_frames(block);
                assert_eq!(read_all(source), want, "{name}: block of {block} frames");
            }
        }
    }

    #[test]
    fn the_reference_files_cover_the_extremes() {
        // Sample 0 is zero and sample 1 is full-scale negative in every file.
        for name in FORMATS {
            let want = python_samples(name);
            assert_eq!(want[0], 0.0, "{name}");
            assert_eq!(want[1], -1.0, "{name}");
        }
    }

    #[test]
    fn it_describes_the_file() {
        let source = WavSource::open(&format_path("mono_16", "wav"), 48_000).unwrap();
        assert!(source.description().starts_with("WAV file '"));
        assert!(source.description().ends_with("mono_16.wav'"));
    }

    // ----- problems -----

    /// Writes a WAV with `frames` frames of a ramp, in the given format.
    fn write_wav(path: &std::path::Path, rate: u32, channels: u16, frames: usize) {
        let spec = hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..frames * usize::from(channels) {
            writer.write_sample((i % 3000) as i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn a_wrong_sample_rate_is_an_error_naming_both_rates() {
        let dir = TempDir::new();
        let path = dir.path().join("rate.wav");
        write_wav(&path, 44_100, 1, 100);
        let error = WavSource::open(path.to_str().unwrap(), 48_000).err().expect("should be refused");
        assert!(error.contains("44100 Hz") && error.contains("48000 Hz"), "{error}");
        assert!(error.contains("resampling is not implemented"), "{error}");
    }

    #[test]
    fn a_missing_file_is_an_error() {
        let dir = TempDir::new();
        let path = dir.path().join("nothing.wav");
        let error = WavSource::open(path.to_str().unwrap(), 48_000).err().expect("should fail");
        assert!(error.starts_with("cannot open "), "{error}");
    }

    #[test]
    fn a_file_that_is_not_a_wav_is_an_error() {
        let dir = TempDir::new();
        let path = dir.path().join("text.wav");
        std::fs::write(&path, "this is not audio at all, it is just some text").unwrap();
        let error = WavSource::open(path.to_str().unwrap(), 48_000).err().expect("should fail");
        assert!(error.contains("not a readable WAV file"), "{error}");
        // And an empty file.
        std::fs::write(&path, "").unwrap();
        assert!(WavSource::open(path.to_str().unwrap(), 48_000).is_err());
    }

    #[test]
    fn an_unsupported_sample_format_is_an_error() {
        // 64-bit float, which libsndfile reads but this program does not.
        let dir = TempDir::new();
        let path = dir.path().join("float64.wav");
        let mut bytes = Vec::new();
        let data_len = 8u32 * 10;
        bytes.extend(b"RIFF");
        bytes.extend((36 + data_len).to_le_bytes());
        bytes.extend(b"WAVEfmt ");
        bytes.extend(16u32.to_le_bytes());
        bytes.extend(3u16.to_le_bytes()); // IEEE float
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(48_000u32.to_le_bytes());
        bytes.extend((48_000u32 * 8).to_le_bytes());
        bytes.extend(8u16.to_le_bytes());
        bytes.extend(64u16.to_le_bytes());
        bytes.extend(b"data");
        bytes.extend(data_len.to_le_bytes());
        bytes.extend(vec![0u8; data_len as usize]);
        std::fs::write(&path, bytes).unwrap();
        let error = WavSource::open(path.to_str().unwrap(), 48_000).err().expect("should be refused");
        // The reader library turns this away itself ("bits per sample is not 32");
        // if it ever let it through, our own check says "unsupported WAV format".
        assert!(
            error.contains("bits per sample") || error.contains("unsupported WAV format"),
            "{error}"
        );
    }

    #[test]
    fn an_empty_recording_gives_no_samples() {
        let dir = TempDir::new();
        let path = dir.path().join("empty.wav");
        write_wav(&path, 48_000, 2, 0);
        let mut source = WavSource::open(path.to_str().unwrap(), 48_000).unwrap();
        assert_eq!(source.next_chunk(), Ok(None));
        assert_eq!(source.next_chunk(), Ok(None), "asking again is harmless");
    }

    #[test]
    fn a_recording_shorter_than_one_block_is_delivered_whole() {
        let dir = TempDir::new();
        let path = dir.path().join("short.wav");
        write_wav(&path, 48_000, 1, 10);
        let mut source = WavSource::open(path.to_str().unwrap(), 48_000).unwrap();
        let chunk = source.next_chunk().unwrap().unwrap();
        assert_eq!(chunk.len(), 10);
        assert_eq!(source.next_chunk(), Ok(None));
    }

    #[test]
    fn a_truncated_file_gives_its_whole_frames_and_then_an_error() {
        let dir = TempDir::new();
        let path = dir.path().join("cut.wav");
        write_wav(&path, 48_000, 2, 10_000); // 40,000 bytes of data after a 44-byte header
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(44 + 4 * 6000 + 2); // 6,000 whole frames and half of the next
        std::fs::write(&path, &bytes).unwrap();

        let mut source = WavSource::open(path.to_str().unwrap(), 48_000).unwrap();
        let mut frames = 0;
        let error = loop {
            match source.next_chunk() {
                Ok(Some(chunk)) => frames += chunk.len(),
                Ok(None) => panic!("the damage should have been reported"),
                Err(error) => break error,
            }
        };
        assert_eq!(frames, 6000, "all whole frames, and not the partial one");
        assert!(error.contains("the file ended early or is damaged"), "{error}");
        assert_eq!(source.next_chunk(), Ok(None), "finished after the error");
    }
}
