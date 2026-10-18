//! The analysis window applied to each frame before tone detection.

use std::f64::consts::PI;

/// Builds a periodic Hamming window of length `n`, as `f32` values.
///
/// "Periodic" (what scipy calls `fftbins=True`) means the cosine repeats
/// every `n` samples, not every `n - 1`:
///
/// ```text
/// w[i] = 0.54 - 0.46 * cos(2 * pi * i / n)
/// ```
///
/// The values are worked out in `f64` and then rounded to `f32`, which is
/// how the Python decoder builds its window.
pub fn hamming_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.54 - 0.46 * (2.0 * PI * i as f64 / n as f64).cos()) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads the second column of a reference CSV (header row skipped).
    fn read_second_column(path: &str) -> Vec<f64> {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
        text.lines()
            .skip(1)
            .map(|line| {
                let field = line.split(',').nth(1).expect("row has no second column");
                field.parse::<f64>().expect("second column is not a number")
            })
            .collect()
    }

    #[test]
    fn has_the_expected_length_and_shape() {
        let w = hamming_periodic(480);
        assert_eq!(w.len(), 480);
        assert!((w[0] - 0.08).abs() < 1e-6, "first value should be 0.08");
        assert_eq!(w[240], 1.0, "the peak is at the middle, n / 2");
    }

    #[test]
    fn matches_the_python_window() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../test-vectors/synth_snr10_seed1/window.csv"
        );
        let expected = read_second_column(path);
        let ours = hamming_periodic(expected.len());

        let mut worst = 0.0_f64;
        for (i, (&mine, &theirs)) in ours.iter().zip(expected.iter()).enumerate() {
            let diff = (mine as f64 - theirs).abs();
            worst = worst.max(diff);
            assert!(diff <= 1e-7, "window[{i}]: rust {mine} vs python {theirs}");
        }
        println!("largest difference from the Python window: {worst:e}");
    }
}
