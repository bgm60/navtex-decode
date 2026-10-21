//! The settings of the weak-signal lock.
//!
//! These are internal constants, not user settings. The defaults match the
//! Python decoder's `SoftLockParams`. The thresholds were calibrated
//! together with the window length and the noise statistics, so change
//! none of them in isolation.
//!
//! Not every field is used yet: scoring (step 6b) uses the window and the
//! noise statistics, and the later steps use the rest.

#[derive(Debug, Clone, PartialEq)]
pub struct SoftLockParams {
    /// Lock threshold on the window z-score.
    pub z_star: f64,
    /// Comparisons in the detection window.
    pub window: usize,
    /// How far the best hypothesis must lead the runner-up, in z.
    pub gap: f64,
    /// Standard deviation of one comparison score on noise alone.
    pub score_sd: f64,
    /// (mean, sd) of the cosine term on noise alone.
    pub cosine_null: (f64, f64),
    /// (mean, sd) of the correlation term on noise alone.
    pub correlation_null: (f64, f64),
    pub release_window: usize,
    pub release_z: f64,
    pub release_hold: usize,
    pub output_delay: usize,
    pub replay_margin: usize,
    /// Move the lock to a clearly better alignment.
    pub follow: bool,
    pub switch_window: usize,
    pub slip_z: f64,
    pub slip_margin: f64,
    pub switch_z: f64,
    pub switch_margin: f64,
    pub switch_hold: usize,
    pub own_release_hold: usize,
    /// Drop runs of this many identical letters (0 = keep them).
    pub max_letter_run: usize,
    /// Idle look-alike pairs in a row before they are dropped (0 = off).
    pub ghost_hold: usize,
}

impl Default for SoftLockParams {
    fn default() -> Self {
        SoftLockParams {
            z_star: 5.0,
            window: 45,
            gap: 1.0,
            score_sd: 1.4168,
            cosine_null: (0.0026, 0.3793),
            correlation_null: (0.715, 0.1284),
            release_window: 30,
            release_z: 1.5,
            release_hold: 5,
            output_delay: 15,
            replay_margin: 10,
            follow: true,
            switch_window: 12,
            slip_z: 4.0,
            slip_margin: 2.0,
            switch_z: 5.0,
            switch_margin: 3.0,
            switch_hold: 3,
            own_release_hold: 25,
            max_letter_run: 8,
            ghost_hold: 3,
        }
    }
}
