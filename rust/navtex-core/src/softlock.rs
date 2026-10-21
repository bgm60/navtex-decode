//! The weak-signal lock: finds the alignment of a SITOR-B bit stream from
//! soft evidence, and notices when the signal has gone.
//!
//! This is built up in stages. At this stage it does the lock state
//! machine only: it scores every comparison, detects the alignment, locks
//! to it, and later releases it. Decoding characters from the lock comes
//! in the next step, and following a moving alignment after that.
//!
//! The history of soft values is kept, because the lock is only declared
//! once enough evidence has built up, and decoding then starts from the
//! evidence that came before it.

use crate::params::SoftLockParams;
use crate::score::{self, HYPOTHESES, ScoreWindow};

/// Soft values kept; when more than this have been stored, the oldest are
/// dropped until half this many remain.
const HISTORY_MAX: usize = 6000;

/// The two hypotheses reached from `h` by slipping one bit earlier or
/// later. Slipping past either end of the 7 phases wraps round and swaps
/// the parity.
fn slip_neighbours(h: usize) -> (usize, usize) {
    let (phase, parity) = (h / 2, h % 2);
    let later = if phase == 6 { (0, 1 - parity) } else { (phase + 1, parity) };
    let earlier = if phase == 0 { (6, 1 - parity) } else { (phase - 1, parity) };
    (later.0 * 2 + later.1, earlier.0 * 2 + earlier.1)
}

pub struct SoftLock {
    params: SoftLockParams,
    /// Soft values of the bit decisions, from decision number `base`.
    history: Vec<f64>,
    base: usize,
    /// Bit decisions received so far.
    count: usize,
    /// The next comparison to score.
    next_bin: usize,
    scores: ScoreWindow,
    locked: bool,
    /// The locked hypothesis (meaningful only while locked).
    h: usize,
    neighbours: (usize, usize),
    /// Consecutive comparisons for which the locked alignment and both its
    /// neighbours scored low.
    low_run: usize,
    releases: usize,
}

impl SoftLock {
    /// A new lock, searching. Following a moving alignment is not built
    /// yet, so `params.follow` must be off for now.
    pub fn new(params: SoftLockParams) -> Self {
        assert!(!params.follow, "following is not implemented yet: set follow to false");
        let window = params.window;
        SoftLock {
            params,
            history: Vec::new(),
            base: 0,
            count: 0,
            next_bin: 0,
            scores: ScoreWindow::new(window),
            locked: false,
            h: 0,
            neighbours: (0, 0),
            low_run: 0,
            releases: 0,
        }
    }

    /// Feeds in the soft value of the next bit decision.
    pub fn push(&mut self, soft_value: f64) {
        self.history.push(soft_value);
        self.count += 1;
        while score::bins_available(self.count) > self.next_bin {
            self.step(self.next_bin);
            self.next_bin += 1;
        }
        if self.history.len() > HISTORY_MAX {
            let drop = self.history.len() - HISTORY_MAX / 2;
            self.history.drain(..drop);
            self.base += drop;
        }
    }

    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// The locked hypothesis (`2 * bit phase + parity`), if locked.
    pub fn hypothesis(&self) -> Option<usize> {
        self.locked.then_some(self.h)
    }

    /// How many times a lock has been released.
    pub fn releases(&self) -> usize {
        self.releases
    }

    /// How many comparisons have been scored so far.
    pub fn comparisons(&self) -> usize {
        self.next_bin
    }

    /// Scores comparison `b`, then locks, or checks whether to release.
    fn step(&mut self, b: usize) {
        let row = score::bin_scores_from(&self.history, self.base, b, &self.params);
        self.scores.push(row);

        if self.locked {
            self.maybe_release();
            return;
        }
        if !self.scores.is_full() {
            return;
        }

        // Which hypothesis leads, and by how much?
        let z = self.scores.z(&self.params);
        let mut top = 0;
        for h in 1..HYPOTHESES {
            if z[h] > z[top] {
                top = h;
            }
        }
        let second = (0..HYPOTHESES)
            .filter(|&h| h != top)
            .map(|h| z[h])
            .fold(f64::NEG_INFINITY, f64::max);
        if z[top] >= self.params.z_star && z[top] - second >= self.params.gap {
            self.lock(top);
        }
    }

    fn lock(&mut self, h: usize) {
        self.locked = true;
        self.h = h;
        self.neighbours = slip_neighbours(h);
        self.low_run = 0;
    }

    /// Drops the lock when the locked alignment, or either of its
    /// one-bit-slip neighbours, has scored below `release_z` for
    /// `release_hold` comparisons in a row.
    fn maybe_release(&mut self) {
        let p = &self.params;
        let z = self.scores.z_last(p.release_window, p);
        let best = z[self.h].max(z[self.neighbours.0]).max(z[self.neighbours.1]);
        self.low_run = if best < p.release_z { self.low_run + 1 } else { 0 };

        if self.low_run >= p.release_hold {
            self.locked = false;
            self.releases += 1;
            self.scores.clear(); // start again from empty
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil;

    /// The settings of the Python's `core` variant: no following, no filters.
    fn core_params() -> SoftLockParams {
        SoftLockParams {
            follow: false,
            max_letter_run: 0,
            ghost_hold: 0,
            ..SoftLockParams::default()
        }
    }

    #[test]
    fn slip_neighbours_wrap_round_and_swap_parity() {
        assert_eq!(slip_neighbours(0), (2, 13)); // phase 0, parity 0
        assert_eq!(slip_neighbours(13), (0, 11)); // phase 6, parity 1
        assert_eq!(slip_neighbours(6), (8, 4)); // phase 3, parity 0
    }

    #[test]
    #[should_panic(expected = "following is not implemented")]
    fn following_is_refused_for_now() {
        SoftLock::new(SoftLockParams::default());
    }

    #[test]
    fn silence_never_locks() {
        let mut lock = SoftLock::new(core_params());
        for _ in 0..20_000 {
            lock.push(0.0);
        }
        assert!(!lock.is_locked());
        assert_eq!(lock.releases(), 0);
        // 20,000 decisions score floor((20000 - 55) / 14) + 1 comparisons,
        // and the trimmed history (6,000 at most) did not stop that.
        assert_eq!(lock.comparisons(), 1425);
    }

    #[test]
    fn matches_the_python_lock_state_for_every_reference_set() {
        let mut total_locked = 0;
        let mut total_releases = 0;
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
            let expected = testutil::read_trace_states(&trace_path);

            let mut lock = SoftLock::new(core_params());
            let mut seen = 0;
            let mut locked_bins = 0;
            for &value in &soft {
                lock.push(value);
                if lock.comparisons() > seen {
                    // One new comparison has just been scored.
                    let want = &expected[seen];
                    assert_eq!(lock.is_locked(), want.locked, "{set}: locked, comparison {seen}");
                    assert_eq!(
                        lock.hypothesis().map_or(-1, |h| h as i64),
                        want.h,
                        "{set}: hypothesis, comparison {seen}"
                    );
                    assert_eq!(lock.releases(), want.releases, "{set}: releases, comparison {seen}");
                    locked_bins += usize::from(want.locked);
                    seen += 1;
                }
            }
            assert_eq!(seen, expected.len(), "{set}: number of comparisons");
            println!(
                "{set}: {seen} comparisons, locked for {locked_bins}, {} releases, all match",
                lock.releases()
            );
            total_locked += locked_bins;
            total_releases += lock.releases();
        }
        assert!(total_locked > 0, "the test never saw a lock, so it proves nothing");
        assert!(total_releases > 0, "the test never saw a release, so it proves nothing");
    }
}
