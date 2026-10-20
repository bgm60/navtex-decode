//! Soft diversity combining: decides one character from its two copies.
//!
//! SITOR-B sends every character twice, a few slots apart. Instead of
//! deciding each copy separately and comparing, this adds the two copies'
//! soft values bit by bit and decides once. A bit that one copy lost to
//! noise is often rescued by the other.
//!
//! This is the rule only. Working out which received groups are
//! partners, and in what order, is the lock's job (a later step).

use crate::charset::{self, Code, Control};

/// Combines pairs of received codes into characters and keeps the
/// LTRS/FIGS shift state between them.
#[derive(Debug, Clone)]
pub struct Combiner {
    letters_mode: bool,
}

impl Default for Combiner {
    fn default() -> Self {
        Combiner::new()
    }
}

impl Combiner {
    /// A new combiner starts in letters mode.
    pub fn new() -> Self {
        Combiner { letters_mode: true }
    }

    pub fn letters_mode(&self) -> bool {
        self.letters_mode
    }

    /// Combines the two copies of one character, given as seven soft values
    /// each (first bit received first; positive means 1). Returns the
    /// character to print, or `None` if the pair prints nothing: a pair of
    /// phasing signals, a shift, or idle filler.
    ///
    /// `'~'` means an error; `'!'` means the two copies each looked like a
    /// valid, but different, character.
    pub fn combine(&mut self, dx_soft: &[f64; 7], rx_soft: &[f64; 7]) -> Option<char> {
        let dx = charset::hard_code(dx_soft);
        let rx = charset::hard_code(rx_soft);

        if charset::is_phasing(dx) && charset::is_phasing(rx) {
            return None;
        }

        // First choice: add the evidence, then decide.
        let combined: [f64; 7] = std::array::from_fn(|i| dx_soft[i] + rx_soft[i]);
        let code = charset::hard_code(&combined);
        match charset::weight(code) {
            4 => return self.decode(code),
            // One bit away from valid: try to repair it.
            3 | 5 => {
                if let Some(fixed) = correct_single_bit(code, &combined) {
                    return self.decode(fixed);
                }
            }
            _ => {}
        }

        // Otherwise fall back to comparing the two separate decisions.
        let dx_valid = charset::weight(dx) == 4;
        let rx_valid = charset::weight(rx) == 4;
        let chosen = match (dx_valid, rx_valid) {
            (true, true) if dx == rx => dx,
            (true, true) => return Some('!'),
            (true, false) => dx,
            (false, true) => rx,
            (false, false) => return Some('~'),
        };
        self.decode(chosen)
    }

    /// Turns a valid code into its character, updating the shift state.
    fn decode(&mut self, code: Code) -> Option<char> {
        if let Some(control) = charset::control(code) {
            return match control {
                Control::Letters => {
                    self.letters_mode = true;
                    None
                }
                Control::Figures => {
                    self.letters_mode = false;
                    None
                }
                Control::CarriageReturn => Some('\r'),
                Control::LineFeed => Some('\n'),
                Control::Space => Some(' '),
                Control::Blank => None,
            };
        }
        // A weight-4 code with no printable meaning (an unassigned figure,
        // or a phasing signal that got this far) is an error.
        Some(charset::printable(code, self.letters_mode).unwrap_or('~'))
    }
}

/// Repairs a code of weight 3 or 5 by flipping one bit so that it becomes a
/// valid code.
///
/// Usually several flips would work, so the one on the least certain bit
/// wins: `confidence[i]` is the size of the combined soft value there. If
/// two bits are equally uncertain the earlier one is used.
fn correct_single_bit(code: Code, combined: &[f64; 7]) -> Option<Code> {
    let flip_ones = match charset::weight(code) {
        3 => false, // a bit is missing: flip a 0 to a 1
        5 => true,  // a bit is extra: flip a 1 to a 0
        _ => return None,
    };

    let mut best: Option<(f64, Code)> = None;
    for (i, value) in combined.iter().enumerate() {
        let mask: Code = 1 << (6 - i);
        if (code & mask != 0) != flip_ones {
            continue;
        }
        let flipped = code ^ mask;
        if !charset::is_valid(flipped) {
            continue;
        }
        let confidence = value.abs();
        // Strictly less, so ties keep the earlier bit.
        if best.is_none_or(|(lowest, _)| confidence < lowest) {
            best = Some((confidence, flipped));
        }
    }
    best.map(|(_, fixed)| fixed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::charset::{PHASING_1, PHASING_2};
    use crate::testutil;

    /// Soft values for a code: `strength` for a 1 and `-strength` for a 0.
    fn soft(code: Code, strength: f64) -> [f64; 7] {
        std::array::from_fn(|i| if code & (1 << (6 - i)) != 0 { strength } else { -strength })
    }

    const A: Code = 0b1110001;
    const B: Code = 0b0100111;
    const LTRS: Code = 0b0101101;
    const FIGS: Code = 0b0110110;

    #[test]
    fn two_clean_copies_give_the_character() {
        let mut c = Combiner::new();
        assert_eq!(c.combine(&soft(A, 2.0), &soft(A, 3.0)), Some('A'));
    }

    #[test]
    fn two_phasing_signals_print_nothing_and_change_nothing() {
        let mut c = Combiner::new();
        assert_eq!(c.combine(&soft(PHASING_1, 2.0), &soft(PHASING_2, 2.0)), None);
        assert!(c.letters_mode());
    }

    #[test]
    fn shifts_change_the_mode_and_print_nothing() {
        let mut c = Combiner::new();
        assert_eq!(c.combine(&soft(FIGS, 2.0), &soft(FIGS, 2.0)), None);
        assert!(!c.letters_mode());
        assert_eq!(c.combine(&soft(A, 2.0), &soft(A, 2.0)), Some('-'));
        assert_eq!(c.combine(&soft(LTRS, 2.0), &soft(LTRS, 2.0)), None);
        assert_eq!(c.combine(&soft(A, 2.0), &soft(A, 2.0)), Some('A'));
    }

    #[test]
    fn a_bit_one_copy_lost_is_rescued_by_the_other() {
        let mut bad = soft(A, 1.0);
        bad[0] = -0.2; // the first bit flipped, but weakly
        let mut c = Combiner::new();
        assert_eq!(c.combine(&bad, &soft(A, 3.0)), Some('A'));
    }

    #[test]
    fn single_bit_correction_picks_the_least_certain_bit() {
        // 0b1110000 (weight 3) is one bit from several valid codes. Make
        // the last bit the weakest, so it is the one flipped: 0b1110001 = A.
        let mut words = soft(0b1110000, 4.0);
        words[6] = -0.1;
        let mut c = Combiner::new();
        assert_eq!(c.combine(&words, &words), Some('A'));
    }

    #[test]
    fn two_different_valid_characters_give_an_exclamation_mark() {
        // A and B differ in four bits. With equal strengths those bits sum
        // to exactly 0, which counts as a 1, so the sum has weight 6 and
        // cannot be repaired. Each copy on its own is valid but different.
        let mut c = Combiner::new();
        assert_eq!(c.combine(&soft(A, 3.0), &soft(B, 3.0)), Some('!'));
    }

    #[test]
    fn rubbish_gives_a_tilde() {
        let mut c = Combiner::new();
        assert_eq!(c.combine(&[-1.0; 7], &[-1.0; 7]), Some('~'));
    }

    #[test]
    fn an_unassigned_figure_gives_a_tilde() {
        let mut c = Combiner::new();
        c.combine(&soft(FIGS, 2.0), &soft(FIGS, 2.0));
        assert_eq!(c.combine(&soft(0b1100101, 2.0), &soft(0b1100101, 2.0)), Some('~'));
    }

    /// The main test: 6,000 pairs captured from the Python combiner. Every
    /// output character and the shift state after it must match exactly.
    #[test]
    fn matches_python_on_the_captured_cases() {
        let rows = testutil::read_combine_cases(&testutil::combine_cases_path());
        assert!(rows.len() >= 1000, "too few cases: {}", rows.len());

        let mut combiner = Combiner::new();
        let mut last_seq = None;
        let mut failures = Vec::new();
        let (mut printed, mut errors, mut disagreements) = (0, 0, 0);

        for (n, row) in rows.iter().enumerate() {
            if last_seq != Some(row.seq) {
                combiner = Combiner::new(); // each sequence starts afresh
                last_seq = Some(row.seq);
            }
            // The hard decisions in the file must agree with ours.
            let dx = charset::hard_code(&row.dx_soft);
            let rx = charset::hard_code(&row.rx_soft);
            assert_eq!(format!("{dx:07b}"), row.dx, "row {n}: dx code");
            assert_eq!(format!("{rx:07b}"), row.rx, "row {n}: rx code");

            let got: String = combiner.combine(&row.dx_soft, &row.rx_soft).into_iter().collect();
            if got != row.out || combiner.letters_mode() != row.letters_mode {
                failures.push(format!(
                    "row {n} (seq {}): expected {:?} mode {}, got {:?} mode {}",
                    row.seq, row.out, row.letters_mode, got, combiner.letters_mode()
                ));
            }
            match got.as_str() {
                "" => {}
                "~" => errors += 1,
                "!" => disagreements += 1,
                _ => printed += 1,
            }
        }
        println!(
            "{} cases: {printed} characters, {errors} '~', {disagreements} '!', {} mismatches",
            rows.len(),
            failures.len()
        );
        assert!(failures.is_empty(), "{} mismatches, first: {:?}", failures.len(), &failures[..failures.len().min(5)]);
    }
}
