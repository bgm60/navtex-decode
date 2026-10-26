//! Signal quality: a 00 to 99 reading of how cleanly the bits are decoding.
//!
//! Every bit decision comes with a confidence between 0 and 1. A single
//! value is noisy, so the meter averages the newest 100 of them (about one
//! second at 100 baud) and maps the average linearly onto 0 to 99. Noise
//! alone averages below `FLOOR` and reads 00; a clean strong signal
//! averages near `CEILING` and reads 99.
//!
//! This is the figure the Python decoder calls "signal strength". Those
//! numbers were calibrated for the Hamming window and 8x oversampling.

use std::collections::VecDeque;

/// Average confidence at which decoding fails: reads 00.
pub const FLOOR: f64 = 0.50;
/// Average confidence of a clean, strong signal: reads 99.
pub const CEILING: f64 = 0.92;
/// How many bit decisions are averaged by default.
pub const DEFAULT_WINDOW: usize = 100;

pub struct QualityMeter {
    window: usize,
    values: VecDeque<f64>,
}

impl QualityMeter {
    pub fn new(window: usize) -> Self {
        QualityMeter { window, values: VecDeque::new() }
    }

    /// Adds the confidence of the next bit decision, forgetting the oldest
    /// one once the window is full.
    pub fn update(&mut self, confidence: f64) {
        self.values.push_back(confidence);
        while self.values.len() > self.window {
            self.values.pop_front();
        }
    }

    /// The current reading, 0 to 99. It is 0 before any bit has arrived.
    pub fn level(&self) -> u8 {
        if self.values.is_empty() {
            return 0;
        }
        let average = python_sum(self.values.iter().copied()) / self.values.len() as f64;
        level_from_average(average)
    }
}

/// Maps an average confidence onto 0 to 99.
pub fn level_from_average(average: f64) -> u8 {
    let scaled = (average - FLOOR) / (CEILING - FLOOR);
    // `round_ties_even` is Python's `round`: exact halves go to the even number.
    let level = (scaled * 99.0).round_ties_even();
    level.clamp(0.0, 99.0) as u8
}

/// Adds up floats the way Python 3.12 and later does inside `sum()`:
/// Neumaier's version of Kahan summation, which corrects for rounding
/// errors as it goes. A plain `+=` can differ in the last digit, and a
/// reading that lands exactly on a rounding boundary would then differ too.
fn python_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut total = 0.0_f64;
    let mut lost = 0.0_f64; // the rounding error so far, added back at the end
    for x in values {
        let t = total + x;
        if total.abs() >= x.abs() {
            lost += (total - t) + x;
        } else {
            lost += (x - t) + total;
        }
        total = t;
    }
    if lost != 0.0 && lost.is_finite() { total + lost } else { total }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil;

    #[test]
    fn a_new_meter_reads_zero() {
        assert_eq!(QualityMeter::new(100).level(), 0);
    }

    #[test]
    fn the_floor_reads_0_and_the_ceiling_reads_99() {
        assert_eq!(level_from_average(FLOOR), 0);
        assert_eq!(level_from_average(CEILING), 99);
    }

    #[test]
    fn readings_are_limited_to_0_to_99() {
        assert_eq!(level_from_average(0.0), 0);
        assert_eq!(level_from_average(0.3), 0);
        assert_eq!(level_from_average(1.0), 99);
    }

    #[test]
    fn the_middle_of_the_range_reads_about_half() {
        // On paper (0.71 - 0.50) / 0.42 * 99 is exactly 49.5. In floating
        // point it comes out as 49.499999999999986, so it rounds down to 49,
        // and Python's round() gives 49 for the same sum.
        assert_eq!(level_from_average(0.71), 49);
    }

    #[test]
    fn only_the_newest_window_of_bits_counts() {
        let mut meter = QualityMeter::new(3);
        for _ in 0..10 {
            meter.update(0.0);
        }
        for _ in 0..3 {
            meter.update(CEILING);
        }
        assert_eq!(meter.level(), 99, "the old zeros should have been forgotten");
    }

    #[test]
    fn a_partly_filled_window_averages_what_it_has() {
        let mut meter = QualityMeter::new(100);
        meter.update(CEILING);
        assert_eq!(meter.level(), 99);
    }

    #[test]
    fn a_window_of_zero_always_reads_zero() {
        // The Python deque with maxlen=0 holds nothing, so it reads 0 too.
        let mut meter = QualityMeter::new(0);
        meter.update(CEILING);
        assert_eq!(meter.level(), 0);
    }

    #[test]
    fn the_sum_is_compensated_like_pythons() {
        // 0.1 added ten times: a plain sum gives 0.9999999999999999,
        // while Python's sum() gives exactly 1.0.
        let sum = python_sum(std::iter::repeat_n(0.1, 10));
        assert_eq!(sum, 1.0);
        assert_ne!(std::iter::repeat_n(0.1, 10).fold(0.0, |a, b| a + b), 1.0);
    }

    /// The main test: feed the confidences the Python decoder produced for
    /// each recording and compare the reading after every single bit.
    #[test]
    fn matches_python_after_every_bit() {
        let sets = testutil::sets_with("quality.csv");
        assert!(sets.len() >= 10, "too few sets: {}", sets.len());
        for set in sets {
            let confidences = testutil::read_confidences(&set);
            let want = testutil::read_quality(&set);
            assert_eq!(confidences.len(), want.len(), "{set}: number of bits");

            let mut meter = QualityMeter::new(DEFAULT_WINDOW);
            let mut nonzero = 0;
            for (k, (&confidence, &level)) in confidences.iter().zip(&want).enumerate() {
                meter.update(confidence);
                assert_eq!(meter.level(), level, "{set}: bit {k}");
                nonzero += usize::from(level > 0);
            }
            println!("{set}: {} readings identical, {nonzero} above zero", want.len());
            assert!(nonzero > 0, "{set}: the reference never leaves zero, so it proves little");
        }
    }
}
