//! Helpers shared by the tests that read the reference files in `test-vectors/`.
//! Only compiled for `cargo test`.

/// Path of a file inside one reference set, e.g. `vector_path("synth_snr10_seed1", "tone.csv")`.
pub fn vector_path(set: &str, file: &str) -> String {
    format!("{}/../test-vectors/{}/{}", env!("CARGO_MANIFEST_DIR"), set, file)
}

/// Reads a 16-bit mono WAV file as `f32` values (each sample divided by
/// 32768), which is how the Python decoder reads it.
pub fn read_wav(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path)
        .unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "expected a mono file");
    assert_eq!(spec.bits_per_sample, 16, "expected 16-bit samples");
    assert_eq!(spec.sample_format, hound::SampleFormat::Int, "expected integer samples");
    reader
        .samples::<i16>()
        .map(|s| f32::from(s.expect("bad sample")) / 32768.0)
        .collect()
}

/// Reads `tone.csv` (columns: frame, diff, bit) as `(diff, bit)` pairs.
pub fn read_tone_csv(path: &str) -> Vec<(f64, bool)> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    text.lines()
        .skip(1) // header row
        .map(|line| {
            let mut fields = line.split(',');
            let _frame = fields.next();
            let diff = fields.next().expect("missing diff").parse::<f64>().expect("bad diff");
            let bit = fields.next().expect("missing bit") == "1";
            (diff, bit)
        })
        .collect()
}
