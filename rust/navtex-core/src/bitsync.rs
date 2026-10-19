//! Bit-clock recovery: turns the stream of frame measurements (eight per
//! bit) into one decision per bit, locked to the transmitter's bit timing.
//!
//! A phase counter advances a little with every frame. When it passes 1.0,
//! one bit period has gone by and a decision is made from the frames
//! collected since the last one. Whenever the tone flips between mark and
//! space, the moment of the flip (estimated from where `diff` crosses zero)
//! is compared with where the counter says a bit boundary should be, and
//! the counter is nudged by a small fraction of the error. That fraction is
//! the loop gain.

use crate::tone::ToneSample;

/// The loop gain the Python decoder uses. It was tuned by hand for weak,
/// noisy signals, so do not change it without re-testing on real recordings.
pub const DEFAULT_LOOP_GAIN: f64 = 0.05;

/// One recovered bit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BitDecision {
    /// `true` = mark (1), `false` = space (0).
    pub bit: bool,
    /// Mean of `|diff|` over the frames in this bit, between 0 and 1.
    pub confidence: f64,
    /// Sum of `diff` over the frames in this bit. Its sign is the bit and
    /// its size is how strong the evidence was.
    pub soft_value: f64,
}

/// A running total that corrects for rounding errors as it goes
/// (Neumaier's version of Kahan summation).
///
/// Python 3.12 and later add up floats this way inside `sum()`. The Python
/// decoder uses `sum()`, so the Rust version has to do the same to get
/// exactly the same totals. A plain `+=` can differ in the last digit.
#[derive(Debug, Clone, Copy, Default)]
struct CompensatedSum {
    total: f64,
    /// The rounding error lost so far, added back in at the end.
    lost: f64,
}

impl CompensatedSum {
    fn add(&mut self, x: f64) {
        let t = self.total + x;
        if self.total.abs() >= x.abs() {
            self.lost += (self.total - t) + x;
        } else {
            self.lost += (x - t) + self.total;
        }
        self.total = t;
    }

    fn value(&self) -> f64 {
        if self.lost != 0.0 && self.lost.is_finite() {
            self.total + self.lost
        } else {
            self.total
        }
    }
}

pub struct BitSync {
    loop_gain: f64,
    /// How far the phase counter advances per frame: the hop divided by
    /// the length of one bit, in samples (60 / 480 = 0.125 by default).
    phase_step: f64,
    /// Where we are within the current bit, from 0.0 up to 1.0.
    phase: f64,
    /// The previous frame's bit and `diff`, once there has been one.
    previous: Option<(bool, f64)>,
    /// The frames collected for the bit in progress.
    frames: usize,
    sum: CompensatedSum,
    abs_sum: CompensatedSum,
}

impl BitSync {
    pub fn new(loop_gain: f64, phase_step: f64) -> Self {
        BitSync {
            loop_gain,
            phase_step,
            phase: 0.0,
            previous: None,
            frames: 0,
            sum: CompensatedSum::default(),
            abs_sum: CompensatedSum::default(),
        }
    }

    /// Feeds in the next frame measurement. Returns `Some` on the frames
    /// that complete a bit (about one in eight) and `None` otherwise.
    pub fn push(&mut self, sample: ToneSample) -> Option<BitDecision> {
        self.sum.add(sample.diff);
        self.abs_sum.add(sample.diff.abs());
        self.frames += 1;

        let phase_before_step = self.phase;
        self.phase += self.phase_step;

        // The tone has flipped since the last frame: use that to correct the phase.
        if let Some((previous_bit, previous_diff)) = self.previous {
            if previous_bit != sample.bit {
                // Where between the two frames did diff cross zero? (0 to 1)
                let t = if previous_diff != sample.diff {
                    (previous_diff / (previous_diff - sample.diff)).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                let crossing_phase = phase_before_step + t * self.phase_step;
                // The nearest bit boundary. Halves round to the nearest even
                // number, as Python's round() does.
                let nearest_boundary = crossing_phase.round_ties_even();
                let error = crossing_phase - nearest_boundary;
                self.phase -= self.loop_gain * error;
            }
        }

        let decision = if self.phase >= 1.0 {
            let total = self.sum.value();
            let decision = BitDecision {
                bit: total >= 0.0,
                confidence: self.abs_sum.value() / self.frames as f64,
                soft_value: total,
            };
            self.sum = CompensatedSum::default();
            self.abs_sum = CompensatedSum::default();
            self.frames = 0;
            self.phase -= 1.0;
            Some(decision)
        } else {
            None
        };

        self.previous = Some((sample.bit, sample.diff));
        decision
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil;

    const PHASE_STEP: f64 = 60.0 / 480.0;

    fn sample(diff: f64) -> ToneSample {
        ToneSample { diff, bit: diff >= 0.0 }
    }

    #[test]
    fn the_compensated_sum_beats_a_plain_one() {
        let values = [1e16, 1.0, -1e16];
        let plain: f64 = values.iter().sum();
        let mut compensated = CompensatedSum::default();
        for v in values {
            compensated.add(v);
        }
        assert_eq!(plain, 0.0, "a plain sum loses the 1.0");
        assert_eq!(compensated.value(), 1.0, "the compensated sum keeps it");
    }

    #[test]
    fn a_steady_mark_gives_one_bit_every_eight_frames() {
        let mut sync = BitSync::new(DEFAULT_LOOP_GAIN, PHASE_STEP);
        for _ in 0..7 {
            assert_eq!(sync.push(sample(0.5)), None);
        }
        let bit = sync.push(sample(0.5)).expect("the eighth frame completes a bit");
        assert!(bit.bit);
        assert_eq!(bit.soft_value, 4.0);
        assert_eq!(bit.confidence, 0.5);
        // and the next bit starts afresh
        assert_eq!(sync.push(sample(0.5)), None);
    }

    #[test]
    fn a_steady_space_gives_a_negative_soft_value() {
        let mut sync = BitSync::new(DEFAULT_LOOP_GAIN, PHASE_STEP);
        let mut decisions = Vec::new();
        for _ in 0..16 {
            decisions.extend(sync.push(sample(-0.25)));
        }
        assert_eq!(decisions.len(), 2);
        for d in decisions {
            assert!(!d.bit);
            assert_eq!(d.soft_value, -2.0);
            assert_eq!(d.confidence, 0.25);
        }
    }

    #[test]
    fn matches_the_python_bits_csv() {
        for set in testutil::reference_sets() {
            let tones = testutil::read_tone_csv(&testutil::vector_path(&set, "tone.csv"));
            let expected = testutil::read_bits_csv(&testutil::vector_path(&set, "bits.csv"));

            let mut sync = BitSync::new(DEFAULT_LOOP_GAIN, PHASE_STEP);
            let mut got = Vec::new();
            for &(diff, bit) in &tones {
                got.extend(sync.push(ToneSample { diff, bit }));
            }

            assert_eq!(got.len(), expected.len(), "{set}: different number of bits");
            for (k, (g, &(bit, confidence, soft_value))) in got.iter().zip(&expected).enumerate() {
                assert_eq!(g.bit, bit, "{set}: bit {k} differs");
                assert_eq!(g.soft_value, soft_value, "{set}: soft_value of bit {k} differs");
                assert_eq!(g.confidence, confidence, "{set}: confidence of bit {k} differs");
            }
            println!("{set}: {} bits, all identical to Python", got.len());
        }
    }
}
