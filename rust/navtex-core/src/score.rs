//! Scoring: how well does each possible alignment of the bit stream look
//! like SITOR-B?
//!
//! Every character is sent twice, five slots (35 bits) apart. If we guess
//! the right alignment, a group of 7 bits and the group 35 bits later are
//! two copies of the same character, so their soft values should agree.
//! With the wrong alignment they are unrelated. A score measures that
//! agreement, and adding scores up over a window of comparisons makes the
//! right alignment stand out even in heavy noise.
//!
//! There are 14 hypotheses: 7 possible bit phases (where a group starts)
//! times 2 parities (which of two interleaved groups is the first copy).
//! Hypothesis `h` is phase `h / 2` and parity `h % 2`.

use std::collections::VecDeque;

use crate::charset;
use crate::params::SoftLockParams;

/// Number of alignment hypotheses: 7 bit phases times 2 parities.
pub const HYPOTHESES: usize = 14;

/// Bit decisions per comparison: two 7-bit slots.
pub const BINS_PER_STEP: usize = 14;

/// Distance between the two copies of a character, in bit decisions.
const SPLIT_OFFSET: usize = 35;

/// Decisions needed beyond `14 * b` before comparison `b` can be scored.
const LAST: usize = 55;

/// Added to denominators so that silence (all zeros) does not divide by zero.
const EPS: f64 = 1e-12;

/// How many comparisons can be scored from `n` bit decisions.
pub fn bins_available(n: usize) -> usize {
    if n < LAST { 0 } else { (n - LAST) / BINS_PER_STEP + 1 }
}

/// Score for one pair of groups: `x` is a group of 7 soft values and `y`
/// the group five slots later. High means "these look like two copies of
/// one SITOR-B character".
///
/// It adds two measures, each shifted and scaled by its typical value on
/// pure noise so that they are comparable:
/// * the cosine of the angle between `x` and `y` (do they agree in sign
///   and relative size?), and
/// * the best match of `x + y` against any of the 32 traffic codes (does
///   the sum look like a real character?).
pub fn comparison_score(x: &[f64; 7], y: &[f64; 7], params: &SoftLockParams) -> f64 {
    let mut xy = 0.0;
    let mut xx = 0.0;
    let mut yy = 0.0;
    let mut both = [0.0; 7];
    for k in 0..7 {
        xy += x[k] * y[k];
        xx += x[k] * x[k];
        yy += y[k] * y[k];
        both[k] = x[k] + y[k];
    }
    let cosine = xy / ((xx * yy).sqrt() + EPS);

    let mut both_energy = 0.0;
    for value in both {
        both_energy += value * value;
    }

    // The best agreement with any traffic code. A code is a pattern of
    // +1 and -1, so "agreement" is the sum of `both` with signs applied.
    let mut best = f64::NEG_INFINITY;
    for code in charset::traffic_codes() {
        let mut agreement = 0.0;
        for k in 0..7 {
            let sign = if code & (1 << (6 - k)) != 0 { 1.0 } else { -1.0 };
            agreement += both[k] * sign;
        }
        best = best.max(agreement);
    }
    let correlation = best / (7.0_f64.sqrt() * both_energy.sqrt() + EPS);

    (cosine - params.cosine_null.0) / params.cosine_null.1
        + (correlation - params.correlation_null.0) / params.correlation_null.1
}

/// Seven consecutive soft values starting at decision number `start`.
/// `soft[0]` is decision number `base` (older decisions may have been
/// discarded to save memory).
fn group(soft: &[f64], base: usize, start: usize) -> [f64; 7] {
    std::array::from_fn(|k| soft[start + k - base])
}

/// The 14 scores for comparison number `b`, from the soft values of all the
/// bit decisions so far (index 0 is the first decision).
///
/// Needs at least `14 * b + 55` decisions: see [`bins_available`].
pub fn bin_scores(soft: &[f64], b: usize, params: &SoftLockParams) -> [f64; HYPOTHESES] {
    bin_scores_from(soft, 0, b, params)
}

/// Like [`bin_scores`], for a slice that starts at decision number `base`
/// rather than at the very first decision.
pub fn bin_scores_from(
    soft: &[f64],
    base: usize,
    b: usize,
    params: &SoftLockParams,
) -> [f64; HYPOTHESES] {
    std::array::from_fn(|h| {
        let phase = h / 2;
        let parity = h % 2;
        // Parity 0 takes the odd group of the pair as the earlier copy,
        // parity 1 the even one. This is how the Python is written.
        let first_group = 2 * b + if parity == 0 { 1 } else { 0 };
        let x_start = phase + 7 * first_group;
        let x = group(soft, base, x_start);
        let y = group(soft, base, x_start + SPLIT_OFFSET);
        comparison_score(&x, &y, params)
    })
}

/// The most recent comparison scores, kept so that they can be added up
/// into z-scores over a window.
#[derive(Debug, Clone)]
pub struct ScoreWindow {
    capacity: usize,
    rows: VecDeque<[f64; HYPOTHESES]>,
}

impl ScoreWindow {
    pub fn new(capacity: usize) -> Self {
        ScoreWindow { capacity, rows: VecDeque::with_capacity(capacity) }
    }

    /// Adds the newest row of scores, dropping the oldest if the window is full.
    pub fn push(&mut self, scores: [f64; HYPOTHESES]) {
        if self.rows.len() == self.capacity {
            self.rows.pop_front();
        }
        self.rows.push_back(scores);
    }

    /// Forgets every row, so the window starts again from empty.
    pub fn clear(&mut self) {
        self.rows.clear();
    }

    pub fn is_full(&self) -> bool {
        self.rows.len() == self.capacity
    }

    /// The sum of each hypothesis's scores over the window, oldest first.
    pub fn sums(&self) -> [f64; HYPOTHESES] {
        let mut total = [0.0; HYPOTHESES];
        for row in &self.rows {
            for h in 0..HYPOTHESES {
                total[h] += row[h];
            }
        }
        total
    }

    /// The sum over only the newest `n` rows (or all of them, if there are
    /// fewer), oldest first.
    pub fn sums_last(&self, n: usize) -> [f64; HYPOTHESES] {
        let mut total = [0.0; HYPOTHESES];
        for row in self.rows.iter().skip(self.rows.len().saturating_sub(n)) {
            for h in 0..HYPOTHESES {
                total[h] += row[h];
            }
        }
        total
    }

    /// z-scores over the newest `n` rows. The divisor always assumes `n`
    /// rows, as the Python does.
    pub fn z_last(&self, n: usize, params: &SoftLockParams) -> [f64; HYPOTHESES] {
        let scale = params.score_sd * (n as f64).sqrt();
        self.sums_last(n).map(|sum| sum / scale)
    }

    /// The window sums scaled to z-scores: a sum of `window` independent
    /// noise scores has standard deviation `score_sd * sqrt(window)`.
    pub fn z(&self, params: &SoftLockParams) -> [f64; HYPOTHESES] {
        let scale = params.score_sd * (params.window as f64).sqrt();
        self.sums().map(|sum| sum / scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil;

    fn params() -> SoftLockParams {
        SoftLockParams::default()
    }

    /// Soft values for a code: `strength` for a 1 and `-strength` for a 0.
    fn soft_code(code: u8, strength: f64) -> [f64; 7] {
        std::array::from_fn(|i| if code & (1 << (6 - i)) != 0 { strength } else { -strength })
    }

    const A: u8 = 0b1110001;

    #[test]
    fn bins_available_counts_from_the_55th_decision() {
        assert_eq!(bins_available(0), 0);
        assert_eq!(bins_available(54), 0);
        assert_eq!(bins_available(55), 1);
        assert_eq!(bins_available(68), 1);
        assert_eq!(bins_available(69), 2);
    }

    /// Two identical, clean copies: cosine 1 and correlation 1.
    fn perfect_score() -> f64 {
        let p = params();
        (1.0 - p.cosine_null.0) / p.cosine_null.1
            + (1.0 - p.correlation_null.0) / p.correlation_null.1
    }

    #[test]
    fn two_identical_clean_copies_score_high() {
        let x = soft_code(A, 2.0);
        let score = comparison_score(&x, &x, &params());
        assert!((score - perfect_score()).abs() < 1e-9, "{score} vs {}", perfect_score());
    }

    #[test]
    fn silence_scores_the_noise_floor_without_dividing_by_zero() {
        let zeros = [0.0; 7];
        let p = params();
        let expected = (0.0 - p.cosine_null.0) / p.cosine_null.1
            + (0.0 - p.correlation_null.0) / p.correlation_null.1;
        let score = comparison_score(&zeros, &zeros, &p);
        assert!(score.is_finite());
        assert!((score - expected).abs() < 1e-9);
    }

    #[test]
    fn the_right_hypothesis_wins_on_planted_copies() {
        // Hypothesis 0 (phase 0, parity 0) at comparison 0 compares the
        // group starting at 7 with the one starting at 42.
        let mut soft = vec![0.0; 55];
        soft[7..14].copy_from_slice(&soft_code(A, 2.0));
        soft[42..49].copy_from_slice(&soft_code(A, 2.0));
        let scores = bin_scores(&soft, 0, &params());
        assert!((scores[0] - perfect_score()).abs() < 1e-9);
        for h in 1..HYPOTHESES {
            assert!(scores[h] < scores[0], "hypothesis {h} should score lower");
        }
    }

    #[test]
    fn the_window_keeps_only_the_newest_rows() {
        let mut p = params();
        p.window = 3;
        p.score_sd = 2.0;
        let mut window = ScoreWindow::new(3);
        for value in [1.0, 2.0, 3.0, 4.0] {
            window.push([value; HYPOTHESES]);
        }
        assert!(window.is_full());
        assert_eq!(window.sums()[5], 2.0 + 3.0 + 4.0);
        let expected = 9.0 / (2.0 * 3.0_f64.sqrt());
        assert!((window.z(&p)[5] - expected).abs() < 1e-12);
    }

    #[test]
    fn the_last_n_rows_and_clear_work() {
        let mut window = ScoreWindow::new(5);
        for value in [1.0, 2.0, 3.0, 4.0] {
            window.push([value; HYPOTHESES]);
        }
        assert_eq!(window.sums_last(2)[0], 7.0);
        assert_eq!(window.sums_last(10)[0], 10.0); // fewer rows than asked for
        window.clear();
        assert!(!window.is_full());
        assert_eq!(window.sums()[0], 0.0);
    }

    #[test]
    fn a_trimmed_history_gives_the_same_scores() {
        // Scores from a slice that starts at decision 20 must equal scores
        // from the whole slice, as long as the needed decisions are there.
        let soft: Vec<f64> = (0..200).map(|i| ((i * 37 % 11) as f64) - 5.0).collect();
        let whole = bin_scores(&soft, 3, &params());
        let trimmed = bin_scores_from(&soft[20..], 20, 3, &params());
        assert_eq!(whole, trimmed);
    }

    #[test]
    fn matches_the_python_scores_for_every_reference_set() {
        let params = params();
        let mut checked = 0;
        for set in testutil::reference_sets() {
            let trace_path = testutil::vector_path(&set, "lock_core_trace.csv");
            if !std::path::Path::new(&trace_path).is_file() {
                continue;
            }
            let soft: Vec<f64> =
                testutil::read_bits_csv(&testutil::vector_path(&set, "bits.csv"))
                    .iter()
                    .map(|&(_, _, soft_value)| soft_value)
                    .collect();
            let expected = testutil::read_trace_scores(&trace_path);
            assert_eq!(expected.len(), bins_available(soft.len()), "{set}: number of comparisons");

            let mut worst = 0.0_f64;
            for (b, row) in expected.iter().enumerate() {
                let got = bin_scores(&soft, b, &params);
                for h in 0..HYPOTHESES {
                    worst = worst.max((got[h] - row[h]).abs());
                }
            }
            println!("{set}: {} comparisons, largest score difference {worst:e}", expected.len());
            assert!(worst <= 1e-9, "{set}: score error {worst:e} is too large");
            checked += 1;
        }
        assert!(checked > 0, "no reference set has a lock_core_trace.csv");
    }
}
