//! Turns a point in time into the UTC date and time fields the log files need.
//!
//! Rust's standard library can say how long it has been since 1970, but it
//! has no calendar. Rather than add a library for it, this does the
//! arithmetic itself: it is about fifteen lines and is checked against
//! Python's own formatting on thousands of timestamps.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A UTC date and time, to the microsecond.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UtcTime {
    pub year: i64,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub microsecond: u32,
}

impl UtcTime {
    /// Converts a system time. A time before 1970 reads as 1970-01-01.
    pub fn from_system_time(time: SystemTime) -> Self {
        let since_epoch = time.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
        let seconds = since_epoch.as_secs();
        let (year, month, day) = civil_from_days((seconds / 86_400) as i64);
        let in_day = seconds % 86_400;
        UtcTime {
            year,
            month,
            day,
            hour: (in_day / 3600) as u32,
            minute: (in_day % 3600 / 60) as u32,
            second: (in_day % 60) as u32,
            microsecond: since_epoch.subsec_micros(),
        }
    }

    /// `YYYYMMDD_HHMMSS`, as used in log file names.
    pub fn file_stamp(&self) -> String {
        format!(
            "{:04}{:02}{:02}_{:02}{:02}{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }

    /// `YYYYMMDD HH:MM:SS`, as used at the start of each log line.
    pub fn line_stamp(&self) -> String {
        format!(
            "{:04}{:02}{:02} {:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }

    /// `YYYY-MM-DD HH:MM:SS`, the time format SQLite uses and the database stores.
    pub fn db_stamp(&self) -> String {
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }

    /// ISO 8601 with a `+00:00` offset, like Python's `isoformat()`: the
    /// fraction of a second is left out when it is exactly zero.
    pub fn iso(&self) -> String {
        let fraction = if self.microsecond == 0 {
            String::new()
        } else {
            format!(".{:06}", self.microsecond)
        };
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{fraction}+00:00",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }
}

/// Day number (days since 1970-01-01) to year, month and day.
///
/// This is Howard Hinnant's well-known "civil from days" method. It counts
/// in 400-year eras (which contain exactly 146,097 days, the length of the
/// Gregorian leap-year cycle) and treats March as the first month, so the
/// odd-length February falls at the end of the year and needs no special case.
fn civil_from_days(days_since_1970: i64) -> (i64, u32, u32) {
    let z = days_since_1970 + 719_468; // days since 0000-03-01
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097); // 0 to 146,096
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_from_march = (5 * day_of_year + 2) / 153; // 0 = March ... 11 = February
    let day = day_of_year - (153 * month_from_march + 2) / 5 + 1;
    let month = if month_from_march < 10 { month_from_march + 3 } else { month_from_march - 9 };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month as u32, day as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at_micros(micros: u64) -> UtcTime {
        UtcTime::from_system_time(UNIX_EPOCH + Duration::from_micros(micros))
    }

    #[test]
    fn the_epoch_is_1970_01_01() {
        let t = at_micros(0);
        assert_eq!((t.year, t.month, t.day, t.hour, t.minute, t.second), (1970, 1, 1, 0, 0, 0));
        assert_eq!(t.iso(), "1970-01-01T00:00:00+00:00");
    }

    #[test]
    fn a_time_before_1970_reads_as_the_epoch() {
        let t = UtcTime::from_system_time(UNIX_EPOCH - Duration::from_secs(5));
        assert_eq!(t.file_stamp(), "19700101_000000");
    }

    #[test]
    fn the_formats_look_like_the_python_ones() {
        let t = at_micros(1_792_344_996_250_000);
        assert_eq!(t.file_stamp(), "20261018_173636");
        assert_eq!(t.line_stamp(), "20261018 17:36:36");
        assert_eq!(t.iso(), "2026-10-18T17:36:36.250000+00:00");
    }

    #[test]
    fn leap_days_are_handled() {
        // 2000 was a leap year (divisible by 400); 2100 will not be (divisible by 100 only).
        assert_eq!(at_micros(951_782_400_000_000).line_stamp(), "20000229 00:00:00");
        assert_eq!(at_micros(4_107_542_400_000_000).line_stamp(), "21000301 00:00:00");
    }

    /// The main test: thousands of timestamps, formatted four ways, must
    /// match the text Python makes from the same instant.
    #[test]
    fn matches_python_on_the_captured_timestamps() {
        let path = format!("{}/../test-vectors/time_cases.csv", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
        let mut count = 0;
        for line in text.lines().skip(1) {
            let fields: Vec<&str> = line.split(',').collect();
            assert_eq!(fields.len(), 5, "bad row {line:?}");
            let t = at_micros(fields[0].parse().expect("bad micros"));
            assert_eq!(t.file_stamp(), fields[1], "file stamp of {}", fields[0]);
            assert_eq!(t.line_stamp(), fields[2], "line stamp of {}", fields[0]);
            assert_eq!(t.iso(), fields[3], "iso of {}", fields[0]);
            assert_eq!(t.db_stamp(), fields[4], "database stamp of {}", fields[0]);
            count += 1;
        }
        assert!(count >= 3000, "too few cases: {count}");
        println!("{count} timestamps identical to Python");
    }
}
