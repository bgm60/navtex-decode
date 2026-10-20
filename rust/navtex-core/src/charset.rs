//! The CCIR 476 / SITOR-B character set (ITU-R M.476-5).
//!
//! A character is sent as a 7-bit code with exactly four 1-bits. Here a
//! code is a `u8` holding those 7 bits, with the bit received FIRST in the
//! highest of the seven positions. So the Python string `"1110001"` (the
//! letter A) is `0b1110001` here, and bit number 0 (the first one received)
//! is the value `0b1000000`.

/// One 7-bit code, in the low 7 bits of a `u8`.
pub type Code = u8;

/// The two phasing signals sent before and between messages. They are
/// never message content.
pub const PHASING_1: Code = 0b1111000;
pub const PHASING_2: Code = 0b0110011;

/// Codes that are the same in letters and figures mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    CarriageReturn,
    LineFeed,
    /// Switch to letters mode.
    Letters,
    /// Switch to figures mode.
    Figures,
    Space,
    /// Idle filler: prints nothing.
    Blank,
}

/// The 26 codes that are a letter in letters mode and something else in
/// figures mode: `(code, letter, figure)`. A figure of `None` means the
/// standard leaves that position unassigned.
const TRAFFIC: [(Code, char, Option<char>); 26] = [
    (0b1110001, 'A', Some('-')),
    (0b0100111, 'B', Some('?')),
    (0b1011100, 'C', Some(':')),
    (0b1100101, 'D', None),
    (0b0110101, 'E', Some('3')),
    (0b1101100, 'F', None),
    (0b1010110, 'G', None),
    (0b1001011, 'H', None),
    (0b1011001, 'I', Some('8')),
    (0b1110100, 'J', Some('\x07')), // the bell
    (0b0111100, 'K', Some('(')),
    (0b1010011, 'L', Some(')')),
    (0b1001110, 'M', Some('.')),
    (0b1001101, 'N', Some(',')),
    (0b1000111, 'O', Some('9')),
    (0b1011010, 'P', Some('0')),
    (0b0111010, 'Q', Some('1')),
    (0b1010101, 'R', Some('4')),
    (0b1101001, 'S', Some('\'')),
    (0b0010111, 'T', Some('5')),
    (0b0111001, 'U', Some('7')),
    (0b0011110, 'V', Some('=')),
    (0b1110010, 'W', Some('2')),
    (0b0101110, 'X', Some('/')),
    (0b1101010, 'Y', Some('6')),
    (0b1100011, 'Z', Some('+')),
];

const CONTROL: [(Code, Control); 6] = [
    (0b0001111, Control::CarriageReturn),
    (0b0011011, Control::LineFeed),
    (0b0101101, Control::Letters),
    (0b0110110, Control::Figures),
    (0b0011101, Control::Space),
    (0b0101011, Control::Blank),
];

/// The number of 1-bits in a code. A valid code has weight 4.
pub fn weight(code: Code) -> u32 {
    code.count_ones()
}

pub fn is_phasing(code: Code) -> bool {
    code == PHASING_1 || code == PHASING_2
}

/// Is this one of the 34 codes the system uses (26 letters, 6 controls
/// and the 2 phasing signals)? Single-bit correction only accepts these.
pub fn is_valid(code: Code) -> bool {
    control(code).is_some()
        || is_phasing(code)
        || TRAFFIC.iter().any(|&(c, _, _)| c == code)
}

/// The control meaning of a code, if it has one.
pub fn control(code: Code) -> Option<Control> {
    CONTROL.iter().find(|&&(c, _)| c == code).map(|&(_, meaning)| meaning)
}

/// The printable character for a code in the given mode. `None` if the code
/// is not a letter code, or is an unassigned figures position.
pub fn printable(code: Code, letters_mode: bool) -> Option<char> {
    let &(_, letter, figure) = TRAFFIC.iter().find(|&&(c, _, _)| c == code)?;
    if letters_mode { Some(letter) } else { figure }
}

/// Turns seven soft values into a code: a value of zero or more is a 1.
/// `soft[0]` is the first bit received.
pub fn hard_code(soft: &[f64; 7]) -> Code {
    let mut code = 0;
    for (i, &value) in soft.iter().enumerate() {
        if value >= 0.0 {
            code |= 1 << (6 - i);
        }
    }
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_32_traffic_codes_are_distinct_and_have_weight_4() {
        let mut codes: Vec<Code> = TRAFFIC.iter().map(|&(c, _, _)| c).collect();
        codes.extend(CONTROL.iter().map(|&(c, _)| c));
        assert_eq!(codes.len(), 32);
        assert!(codes.iter().all(|&c| weight(c) == 4 && c < 128));
        codes.sort();
        codes.dedup();
        assert_eq!(codes.len(), 32, "a code appears twice");
    }

    #[test]
    fn there_are_34_valid_codes_and_all_have_weight_4() {
        let valid: Vec<Code> = (0..128).filter(|&c| is_valid(c)).collect();
        assert_eq!(valid.len(), 34);
        assert!(valid.iter().all(|&c| weight(c) == 4));
        // Of the 35 possible weight-4 codes, exactly one is unused.
        assert_eq!((0..128u8).filter(|&c| weight(c) == 4).count(), 35);
    }

    #[test]
    fn letters_and_figures_look_up_correctly() {
        assert_eq!(printable(0b1110001, true), Some('A'));
        assert_eq!(printable(0b1110001, false), Some('-'));
        assert_eq!(printable(0b1100011, true), Some('Z'));
        assert_eq!(printable(0b1100011, false), Some('+'));
        assert_eq!(printable(0b1110100, false), Some('\x07'));
        // D, F, G and H have no figure.
        assert_eq!(printable(0b1100101, true), Some('D'));
        assert_eq!(printable(0b1100101, false), None);
        let unassigned = TRAFFIC.iter().filter(|t| t.2.is_none()).count();
        assert_eq!(unassigned, 4);
    }

    #[test]
    fn controls_and_phasing_are_not_printable() {
        assert_eq!(control(0b0101101), Some(Control::Letters));
        assert_eq!(control(0b0110110), Some(Control::Figures));
        assert_eq!(printable(0b0101101, true), None);
        assert_eq!(control(PHASING_1), None);
        assert_eq!(printable(PHASING_1, true), None);
        assert!(is_phasing(PHASING_2) && is_valid(PHASING_2));
    }

    #[test]
    fn hard_code_reads_the_first_bit_as_the_highest() {
        let soft = [1.0, 1.0, 1.0, -0.5, -0.5, -0.5, 0.0];
        assert_eq!(hard_code(&soft), 0b1110001); // a soft value of exactly 0 is a 1
        assert_eq!(hard_code(&[-1.0; 7]), 0);
    }
}
