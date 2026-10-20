//! Helpers shared by the tests that read the reference files in `test-vectors/`.
//! Only compiled for `cargo test`.

/// Path of a file inside one reference set, e.g. `vector_path("synth_snr10_seed1", "tone.csv")`.
pub fn vector_path(set: &str, file: &str) -> String {
    format!("{}/../test-vectors/{}/{}", env!("CARGO_MANIFEST_DIR"), set, file)
}

/// The names of every reference set in `test-vectors/` that has both a
/// `tone.csv` and a `bits.csv`, in alphabetical order. Any new set made with
/// `dump_reference.py` is picked up automatically.
pub fn reference_sets() -> Vec<String> {
    let root = format!("{}/../test-vectors", env!("CARGO_MANIFEST_DIR"));
    let entries = std::fs::read_dir(&root).unwrap_or_else(|e| panic!("cannot read {root}: {e}"));
    let mut names: Vec<String> = entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().join("tone.csv").is_file() && entry.path().join("bits.csv").is_file()
        })
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no reference sets found in {root}");
    names
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

/// Reads `bits.csv` (columns: bit_index, bit, confidence, soft_value) as
/// `(bit, confidence, soft_value)` triples.
pub fn read_bits_csv(path: &str) -> Vec<(bool, f64, f64)> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    text.lines()
        .skip(1) // header row
        .map(|line| {
            let mut fields = line.split(',');
            let _index = fields.next();
            let bit = fields.next().expect("missing bit") == "1";
            let confidence = fields
                .next()
                .expect("missing confidence")
                .parse::<f64>()
                .expect("bad confidence");
            let soft_value = fields
                .next()
                .expect("missing soft_value")
                .parse::<f64>()
                .expect("bad soft_value");
            (bit, confidence, soft_value)
        })
        .collect()
}

/// Path of `combine_cases.csv`, which sits in the root of `test-vectors/`.
pub fn combine_cases_path() -> String {
    format!("{}/../test-vectors/combine_cases.csv", env!("CARGO_MANIFEST_DIR"))
}

/// One row of `combine_cases.csv`: a pair of received codes and what the
/// Python combiner did with it.
pub struct CombineRow {
    pub seq: usize,
    /// The hard decisions as the Python wrote them, e.g. "1110001".
    pub dx: String,
    pub rx: String,
    pub dx_soft: [f64; 7],
    pub rx_soft: [f64; 7],
    /// The characters produced: empty, or one character.
    pub out: String,
    /// The shift state after this pair.
    pub letters_mode: bool,
}

/// Reads `combine_cases.csv` (columns: seq, step, dx, rx, dx0..dx6, rx0..rx6,
/// out, letters_mode). `out` holds Unicode code points as decimal numbers.
pub fn read_combine_cases(path: &str) -> Vec<CombineRow> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    text.lines()
        .skip(1) // header row
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            assert_eq!(f.len(), 20, "expected 20 fields in {line:?}");
            let floats = |from: usize| -> [f64; 7] {
                std::array::from_fn(|i| f[from + i].parse::<f64>().expect("bad soft value"))
            };
            let out = f[18]
                .split_whitespace()
                .map(|n| char::from_u32(n.parse().expect("bad code point")).expect("bad char"))
                .collect();
            CombineRow {
                seq: f[0].parse().expect("bad seq"),
                dx: f[2].to_string(),
                rx: f[3].to_string(),
                dx_soft: floats(4),
                rx_soft: floats(11),
                out,
                letters_mode: f[19] == "1",
            }
        })
        .collect()
}
