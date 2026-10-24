//! The weak-signal lock: finds the alignment of a SITOR-B bit stream from
//! soft evidence, decodes the characters, and notices when the signal has
//! gone.
//!
//! This is built up in stages. At this stage it scores every comparison,
//! detects the alignment, locks to it, replays the stored evidence from
//! before the lock, combines the two copies of each character, holds the
//! characters back for a short delay, and releases the lock when the
//! signal goes. While locked it also watches for a better alignment and
//! moves to it (a bit slipped in the bit clock, or a second station took
//! over). Finally it drops two kinds of junk that a phasing signal read out
//! of step can produce: long runs of one letter, and "idle look-alike"
//! pairs.
//!
//! The history of soft values is kept, because the lock is only declared
//! once enough evidence has built up, and decoding then starts from the
//! evidence that came before it, so that no text is lost.

use std::collections::VecDeque;

use crate::charset::{self, Code, PHASING_1, PHASING_2};
use crate::combine::Combiner;
use crate::params::SoftLockParams;
use crate::score::{self, BINS_PER_STEP, HYPOTHESES, ScoreWindow};

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

/// Division rounding up, also for negative numbers: `ceil_div(-3, 7)` is 0.
fn ceil_div(a: i64, b: i64) -> i64 {
    -((-a).div_euclid(b))
}

/// Rotates a 7-bit code left by `k` places (the first `k` bits move to the
/// end), as the Python does to its strings.
fn rotate(code: Code, k: u32) -> Code {
    ((code << k) | (code >> (7 - k))) & 0x7F
}

/// How many of the 7 bits differ.
fn hamming(a: Code, b: Code) -> u32 {
    (a ^ b).count_ones()
}

/// Are the two groups each within one bit of an idle pair (the two phasing
/// signals, one after the other) read 1 to 6 bits out of step? Two copies
/// of a real character agree, so real text cannot look like this.
fn is_idle_lookalike(dx: Code, rx: Code) -> bool {
    (1..=6).any(|k| {
        [(PHASING_1, PHASING_2), (PHASING_2, PHASING_1)]
            .iter()
            .any(|&(x, y)| hamming(dx, rotate(x, k)) <= 1 && hamming(rx, rotate(y, k)) <= 1)
    })
}

/// The index of the largest value; the first one if several are equal.
fn first_max(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

/// Where did the signal change from alignment `old` to alignment `new`?
/// The inputs are the two alignments' scores over the same comparisons,
/// oldest first. Returns `k`, the number of leading comparisons that
/// belong to the old alignment: the `k` that maximises the old scores
/// before it plus the new scores after it. If several are equal, the
/// earliest.
fn best_split(old: &[f64], new: &[f64]) -> usize {
    let m = old.len();
    // old_total[j] is the sum of the first j old scores, and likewise new.
    let mut old_total = vec![0.0; m + 1];
    let mut new_total = vec![0.0; m + 1];
    for j in 0..m {
        old_total[j + 1] = old_total[j] + old[j];
        new_total[j + 1] = new_total[j] + new[j];
    }
    let fits: Vec<f64> = (0..=m).map(|j| old_total[j] + (new_total[m] - new_total[j])).collect();
    first_max(&fits)
}

/// One decoded character.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockEvent {
    pub ch: char,
    /// Number of the first bit decision of the repeat group that produced
    /// the character, counting from the start of the stream.
    pub bit_index: usize,
    /// How many locks had been released before this character came out.
    /// The characters of one lock (one message) share a number; the next
    /// lock's characters have a higher one, so a caller can start a new
    /// line when it changes.
    pub releases: usize,
}

/// A character waiting out the output delay.
#[derive(Debug, Clone)]
struct Pending {
    /// The decision just after the repeat group; it is released once the
    /// stream is far enough beyond this point.
    end: usize,
    ch: char,
    bit_index: usize,
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
    /// Consecutive comparisons for which the locked alignment alone scored low.
    own_low_run: usize,
    /// Consecutive comparisons for which the same challenger has been
    /// clearly ahead.
    challenge_run: usize,
    /// Comparisons since the lock started or last moved.
    dwell: usize,
    releases: usize,
    /// Releases caused only by the last-resort rule (the locked alignment
    /// alone low for a long time). A diagnostic; the Python has no counter.
    own_releases: usize,
    switches: usize,
    /// The letter in the current run of identical letters, and its length.
    run_char: Option<char>,
    run_len: usize,
    /// True once a run has been recognised and its remaining letters are dropped.
    run_dropping: bool,
    /// Idle look-alike pairs in a row, the starts of the ones decoded so
    /// far, and whether the run is being dropped.
    ghost_run: usize,
    ghost_keys: Vec<usize>,
    ghost_dropping: bool,
    /// Diagnostics for the tests: how often each filter has fired.
    letter_run_drops: usize,
    ghost_runs_dropped: usize,
    /// Decisions before this belong to a lock that has finished.
    prev_end: i64,
    /// Start of the first 7-bit group read under this lock, and its number
    /// counting from the first possible group (its parity decides which
    /// groups are repeats).
    group_start: usize,
    first_group: usize,
    /// Groups read since the lock began.
    groups_done: usize,
    /// The last six groups read (the copy five groups back, and this one).
    recent: VecDeque<[f64; 7]>,
    combiner: Combiner,
    pending: VecDeque<Pending>,
    emitted: usize,
}

impl SoftLock {
    /// A new lock, searching.
    pub fn new(params: SoftLockParams) -> Self {
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
            own_low_run: 0,
            challenge_run: 0,
            dwell: 0,
            releases: 0,
            own_releases: 0,
            switches: 0,
            run_char: None,
            run_len: 0,
            run_dropping: false,
            ghost_run: 0,
            ghost_keys: Vec::new(),
            ghost_dropping: false,
            letter_run_drops: 0,
            ghost_runs_dropped: 0,
            prev_end: 0,
            group_start: 0,
            first_group: 0,
            groups_done: 0,
            recent: VecDeque::with_capacity(6),
            combiner: Combiner::new(),
            pending: VecDeque::new(),
            emitted: 0,
        }
    }

    /// Feeds in the soft value of the next bit decision. Returns the
    /// characters that become ready, which is usually none.
    pub fn push(&mut self, soft_value: f64) -> Vec<LockEvent> {
        self.history.push(soft_value);
        self.count += 1;
        let mut events = Vec::new();
        while score::bins_available(self.count) > self.next_bin {
            events.extend(self.step(self.next_bin));
            self.next_bin += 1;
        }
        if self.history.len() > HISTORY_MAX {
            let drop = self.history.len() - HISTORY_MAX / 2;
            self.history.drain(..drop);
            self.base += drop;
        }
        self.emitted += events.len();
        events
    }

    /// Call at the end of the stream: returns what is still held back.
    pub fn finish(&mut self) -> Vec<LockEvent> {
        if self.locked {
            self.advance_groups();
        }
        let releases = self.releases;
        let events: Vec<LockEvent> = self
            .pending
            .drain(..)
            .map(|p| LockEvent { ch: p.ch, bit_index: p.bit_index, releases })
            .collect();
        self.emitted += events.len();
        self.locked = false;
        events
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

    /// How many of those releases came from the last-resort rule alone.
    pub fn own_releases(&self) -> usize {
        self.own_releases
    }

    /// Runs of identical letters recognised and dropped so far.
    pub fn letter_run_drops(&self) -> usize {
        self.letter_run_drops
    }

    /// Runs of idle look-alike pairs recognised and dropped so far.
    pub fn ghost_runs_dropped(&self) -> usize {
        self.ghost_runs_dropped
    }

    /// How many times the lock has moved to another alignment.
    pub fn switches(&self) -> usize {
        self.switches
    }

    /// How many comparisons have been scored so far.
    pub fn comparisons(&self) -> usize {
        self.next_bin
    }

    /// Characters decoded but still held back by the output delay.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Characters released so far.
    pub fn emitted(&self) -> usize {
        self.emitted
    }

    /// Scores comparison `b`, then locks, or decodes and checks whether to
    /// release. Returns any characters released.
    fn step(&mut self, b: usize) -> Vec<LockEvent> {
        let row = score::bin_scores_from(&self.history, self.base, b, &self.params);
        self.scores.push(row);

        if self.locked {
            self.advance_groups();
            return self.maybe_release(b);
        }
        if !self.scores.is_full() {
            return Vec::new();
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
            self.lock(top, b);
        }
        Vec::new()
    }

    /// Locks to hypothesis `h`, found at comparison `b`, and decodes
    /// everything from just before the start of the detection window.
    fn lock(&mut self, h: usize, b: usize) {
        self.locked = true;
        self.h = h;
        self.neighbours = slip_neighbours(h);
        self.low_run = 0;
        self.own_low_run = 0;
        self.challenge_run = 0;
        self.dwell = 0;
        self.reset_run();
        self.reset_ghost();
        self.combiner = Combiner::new();

        // The first group to read: the first one that starts at this bit
        // phase on or after the replay point. These are signed numbers
        // because the replay point can come out negative early in a stream.
        let phase = (h / 2) as i64;
        let window = self.params.window as i64;
        let margin = self.params.replay_margin as i64;
        let window_start = (b as i64 - window + 1 - margin) * BINS_PER_STEP as i64;
        let replay_from = window_start.max(self.prev_end).max(self.base as i64);
        let mut first_group = ceil_div(replay_from - phase, 7).max(0);
        let mut group_start = phase + 7 * first_group;
        while group_start < self.base as i64 {
            // never start before the history that was kept
            first_group += 1;
            group_start += 7;
        }
        self.first_group = first_group as usize;
        self.group_start = group_start as usize;
        self.groups_done = 0;
        self.recent.clear();
        self.pending.clear();
        self.advance_groups();
    }

    /// Reads every 7-bit group that has completed since the last call. A
    /// group that is a repeat copy is combined with the group five before
    /// it, and any character that results is queued.
    fn advance_groups(&mut self) {
        let parity = self.h % 2;
        while self.group_start + 7 * self.groups_done + 7 <= self.count {
            let start = self.group_start + 7 * self.groups_done;
            let soft = score::group(&self.history, self.base, start);
            if self.recent.len() == 6 {
                self.recent.pop_front();
            }
            self.recent.push_back(soft);

            let is_repeat = (self.first_group + self.groups_done) % 2 == parity;
            if self.groups_done >= 5 && is_repeat {
                let first_copy = self.recent[0]; // five groups before this one
                let dx = charset::hard_code(&first_copy);
                let rx = charset::hard_code(&soft);
                if self.skip_idle_lookalike(dx, rx, start) {
                    self.groups_done += 1;
                    continue;
                }
                if let Some(ch) = self.combiner.combine(&first_copy, &soft) {
                    self.queue(start + 7, ch, start);
                }
            }
            self.groups_done += 1;
        }
    }

    fn reset_run(&mut self) {
        self.run_char = None;
        self.run_len = 0;
        self.run_dropping = false;
    }

    fn reset_ghost(&mut self) {
        self.ghost_run = 0;
        self.ghost_keys.clear();
        self.ghost_dropping = false;
    }

    /// Tracks idle look-alike pairs. Returns true for a pair that is to be
    /// dropped without decoding. The first `ghost_hold - 1` of a run are
    /// decoded as usual, because they cannot yet be told from text; when
    /// the run reaches `ghost_hold`, the characters they produced, which
    /// are still held back, are removed too.
    fn skip_idle_lookalike(&mut self, dx: Code, rx: Code, key: usize) -> bool {
        let hold = self.params.ghost_hold;
        if hold == 0 {
            return false;
        }
        if charset::is_phasing(dx) && charset::is_phasing(rx) {
            return false; // a genuine idle pair: the combiner prints nothing for it
        }
        if !is_idle_lookalike(dx, rx) {
            self.reset_ghost();
            return false;
        }
        self.ghost_run += 1;
        if self.ghost_run < hold {
            self.ghost_keys.push(key);
            return false;
        }
        if !self.ghost_dropping {
            self.ghost_dropping = true;
            self.ghost_runs_dropped += 1;
            let earlier = std::mem::take(&mut self.ghost_keys);
            self.pending.retain(|item| !earlier.contains(&item.bit_index));
        }
        true
    }

    /// Adds a decoded character to the held-back queue, dropping runs of
    /// `max_letter_run` or more identical letters. Characters are held back
    /// longer than the run limit, so the start of a run is still in the
    /// queue when it is recognised.
    fn queue(&mut self, end: usize, ch: char, bit_index: usize) {
        let limit = self.params.max_letter_run;
        if limit == 0 || !ch.is_alphabetic() {
            self.reset_run();
            self.pending.push_back(Pending { end, ch, bit_index });
            return;
        }
        if self.run_char == Some(ch) {
            self.run_len += 1;
        } else {
            self.run_char = Some(ch);
            self.run_len = 1;
            self.run_dropping = false;
        }
        if self.run_dropping {
            return;
        }
        if self.run_len >= limit {
            self.run_dropping = true;
            self.letter_run_drops += 1;
            for _ in 0..self.run_len - 1 {
                if self.pending.back().is_some_and(|item| item.ch == ch) {
                    self.pending.pop_back();
                }
            }
            return;
        }
        self.pending.push_back(Pending { end, ch, bit_index });
    }

    /// Releases the characters whose delay has passed, then drops the lock
    /// if the locked alignment, or either of its one-bit-slip neighbours,
    /// has scored below `release_z` for `release_hold` comparisons in a row.
    fn maybe_release(&mut self, b: usize) -> Vec<LockEvent> {
        if self.params.follow {
            self.maybe_switch(b);
        }
        let p = &self.params;
        let limit = BINS_PER_STEP as i64 * (b as i64 - p.output_delay as i64) + BINS_PER_STEP as i64;
        let z = self.scores.z_last(p.release_window, p);
        let best = z[self.h].max(z[self.neighbours.0]).max(z[self.neighbours.1]);
        self.low_run = if best < p.release_z { self.low_run + 1 } else { 0 };
        self.own_low_run = if z[self.h] < p.release_z { self.own_low_run + 1 } else { 0 };
        let hold = p.release_hold;
        let own_hold = p.own_release_hold;
        let follow = p.follow;

        let mut events = Vec::new();
        while self.pending.front().is_some_and(|item| item.end as i64 <= limit) {
            if let Some(item) = self.pending.pop_front() {
                // Counted before this step's release, as the Python lock does.
                events.push(LockEvent {
                    ch: item.ch,
                    bit_index: item.bit_index,
                    releases: self.releases,
                });
            }
        }

        if self.low_run >= hold || (follow && self.own_low_run >= own_hold) {
            self.pending.clear(); // what is left is the stray tail
            self.locked = false;
            self.releases += 1;
            if self.low_run < hold {
                self.own_releases += 1;
            }
            self.prev_end = self.prev_end.max(limit);
            self.scores.clear(); // start again from empty
        }
        events
    }
}

impl SoftLock {
    /// Moves the lock to another alignment that is clearly better.
    fn maybe_switch(&mut self, b: usize) {
        let s = self.params.switch_window;
        self.dwell += 1;
        if self.scores.len() < s || self.dwell < s {
            self.challenge_run = 0;
            return;
        }

        // Which alignment has scored best over the last `s` comparisons?
        let z = self.scores.z_last(s, &self.params);
        let challenger = first_max(&z);
        if challenger == self.h {
            self.challenge_run = 0;
            return;
        }

        // A one-bit slip needs less evidence than an unrelated alignment.
        let p = &self.params;
        let near = challenger == self.neighbours.0 || challenger == self.neighbours.1;
        let (need_z, need_margin) =
            if near { (p.slip_z, p.slip_margin) } else { (p.switch_z, p.switch_margin) };
        let clearly_ahead = z[challenger] >= need_z && z[challenger] - z[self.h] >= need_margin;
        let hold = p.switch_hold;

        self.challenge_run = if clearly_ahead { self.challenge_run + 1 } else { 0 };
        if self.challenge_run >= hold {
            self.switch(challenger, b, near);
        }
    }

    /// Re-aligns to hypothesis `c` from the point where the stored scores
    /// say the signal changed from the old alignment to the new one.
    fn switch(&mut self, c: usize, b: usize, near: bool) {
        let m = self.scores.len();
        let k = best_split(&self.scores.column(self.h), &self.scores.column(c));
        // The first decision that belongs to the new alignment.
        let cut = BINS_PER_STEP as i64 * (b as i64 - m as i64 + 1 + k as i64) + 35;
        let cut = cut.max(self.base as i64);

        // Characters still held back from after the change are dropped.
        self.pending.retain(|item| item.end as i64 <= cut);

        self.h = c;
        self.neighbours = slip_neighbours(c);
        if !near {
            // An unrelated alignment: the letters/figures state cannot be trusted.
            self.combiner = Combiner::new();
        }
        let phase = (c / 2) as i64;
        let mut first_group = ceil_div(cut - phase, 7).max(0);
        let mut group_start = phase + 7 * first_group;
        while group_start < self.base as i64 {
            first_group += 1;
            group_start += 7;
        }
        self.first_group = first_group as usize;
        self.group_start = group_start as usize;
        self.groups_done = 0;
        self.recent.clear();
        self.reset_run();
        self.reset_ghost();
        self.low_run = 0;
        self.own_low_run = 0;
        self.challenge_run = 0;
        self.dwell = 0;
        self.switches += 1;
        // Read the groups already received under the new alignment.
        self.advance_groups();
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
    fn ceil_div_rounds_up_even_for_negative_numbers() {
        assert_eq!(ceil_div(-3, 7), 0);
        assert_eq!(ceil_div(-7, 7), -1);
        assert_eq!(ceil_div(-8, 7), -1);
        assert_eq!(ceil_div(0, 7), 0);
        assert_eq!(ceil_div(1, 7), 1);
        assert_eq!(ceil_div(7, 7), 1);
        assert_eq!(ceil_div(8, 7), 2);
    }

    /// The settings of the Python's `follow` variant: following, no filters.
    fn follow_params() -> SoftLockParams {
        SoftLockParams { follow: true, ..core_params() }
    }

    #[test]
    fn first_max_picks_the_earliest_of_equal_values() {
        assert_eq!(first_max(&[1.0, 3.0, 2.0, 3.0]), 1);
        assert_eq!(first_max(&[5.0]), 0);
        assert_eq!(first_max(&[-2.0, -1.0, -1.0]), 1);
    }

    #[test]
    fn best_split_finds_where_the_scores_change_over() {
        // The old alignment scores well for the first three comparisons,
        // the new one for the last two.
        let old = [1.0, 1.0, 1.0, 0.0, 0.0];
        let new = [0.0, 0.0, 0.0, 1.0, 1.0];
        assert_eq!(best_split(&old, &new), 3);
        // The new alignment was ahead throughout: the split is at the start.
        assert_eq!(best_split(&[0.0; 4], &[1.0; 4]), 0);
        // The old one was ahead throughout: the split is at the end.
        assert_eq!(best_split(&[1.0; 4], &[0.0; 4]), 4);
        // No difference at all: the earliest split.
        assert_eq!(best_split(&[1.0; 4], &[1.0; 4]), 0);
    }

    #[test]
    fn rotating_and_comparing_codes() {
        assert_eq!(rotate(0b1111000, 1), 0b1110001);
        assert_eq!(rotate(0b0110011, 2), 0b1001101);
        assert_eq!(rotate(0b1000001, 6), 0b1100000);
        assert_eq!(hamming(0b1111000, 0b1111000), 0);
        assert_eq!(hamming(0b1111000, 0b0110011), 4);
    }

    #[test]
    fn idle_look_alikes_are_recognised_but_real_pairs_are_not() {
        for k in 1..=6 {
            assert!(is_idle_lookalike(rotate(PHASING_1, k), rotate(PHASING_2, k)));
            assert!(is_idle_lookalike(rotate(PHASING_2, k), rotate(PHASING_1, k)));
        }
        // one bit wrong in each group is still a look-alike
        assert!(is_idle_lookalike(rotate(PHASING_1, 3) ^ 0b0000100, rotate(PHASING_2, 3) ^ 0b0100000));
        // two copies of the letter A
        assert!(!is_idle_lookalike(0b1110001, 0b1110001));
    }

    #[test]
    fn a_run_of_eight_identical_letters_is_dropped_with_the_seven_before_it() {
        let mut lock = SoftLock::new(SoftLockParams::default());
        for i in 0..7 {
            lock.queue(i, 'K', i);
        }
        assert_eq!(lock.pending_len(), 7);
        lock.queue(7, 'K', 7); // the eighth
        assert_eq!(lock.pending_len(), 0);
        lock.queue(8, 'K', 8); // the rest of the run is dropped
        assert_eq!(lock.pending_len(), 0);
        assert_eq!(lock.letter_run_drops(), 1);
        lock.queue(9, 'A', 9); // a different letter starts a new run
        lock.queue(10, 'K', 10);
        assert_eq!(lock.pending_len(), 2);
    }

    #[test]
    fn something_that_is_not_a_letter_breaks_a_run() {
        let mut lock = SoftLock::new(SoftLockParams::default());
        for i in 0..7 {
            lock.queue(i, 'K', i);
        }
        lock.queue(7, '~', 7);
        for i in 8..15 {
            lock.queue(i, 'K', i);
        }
        assert_eq!(lock.pending_len(), 15);
        assert_eq!(lock.letter_run_drops(), 0);
    }

    #[test]
    fn the_third_look_alike_in_a_row_removes_the_earlier_characters_too() {
        let mut lock = SoftLock::new(SoftLockParams::default()); // ghost_hold is 3
        let dx = rotate(PHASING_1, 2);
        let rx = rotate(PHASING_2, 2);
        lock.queue(10, 'X', 100);
        lock.queue(20, 'Y', 200);
        lock.queue(30, 'Z', 300); // from a pair that is not part of the run
        assert!(!lock.skip_idle_lookalike(dx, rx, 100));
        assert!(!lock.skip_idle_lookalike(dx, rx, 200));
        assert_eq!(lock.pending_len(), 3);
        assert!(lock.skip_idle_lookalike(dx, rx, 400)); // the third: dropped
        assert_eq!(lock.pending_len(), 1); // X and Y removed, Z kept
        assert!(lock.skip_idle_lookalike(dx, rx, 500)); // and the run carries on
        assert_eq!(lock.ghost_runs_dropped(), 1);
        // a real pair ends the run
        assert!(!lock.skip_idle_lookalike(0b1110001, 0b1110001, 600));
        assert!(!lock.skip_idle_lookalike(dx, rx, 700)); // a new run starts again
    }

    #[test]
    fn a_genuine_idle_pair_is_left_alone_and_does_not_end_a_run() {
        let mut lock = SoftLock::new(SoftLockParams::default());
        let dx = rotate(PHASING_1, 2);
        let rx = rotate(PHASING_2, 2);
        assert!(!lock.skip_idle_lookalike(dx, rx, 1));
        assert!(!lock.skip_idle_lookalike(PHASING_1, PHASING_2, 2));
        assert!(!lock.skip_idle_lookalike(dx, rx, 3));
        assert!(lock.skip_idle_lookalike(dx, rx, 4)); // the run was not reset
    }

    #[test]
    fn silence_never_locks_and_gives_no_text() {
        let mut lock = SoftLock::new(core_params());
        let mut events = 0;
        for _ in 0..20_000 {
            events += lock.push(0.0).len();
        }
        events += lock.finish().len();
        assert_eq!(events, 0);
        assert!(!lock.is_locked());
        assert_eq!(lock.releases(), 0);
        // 20,000 decisions score floor((20000 - 55) / 14) + 1 comparisons,
        // and the trimmed history (6,000 at most) did not stop that.
        assert_eq!(lock.comparisons(), 1425);
    }

    #[test]
    fn matches_the_python_lock_for_every_reference_set() {
        let variants: [(&str, SoftLockParams); 3] = [
            ("core", core_params()),
            ("follow", follow_params()),
            ("full", SoftLockParams::default()),
        ];
        let mut total_chars = 0;
        let mut total_releases = 0;
        let mut total_switches = 0;
        let mut total_own_releases = 0;
        let mut total_run_drops = 0;
        let mut total_ghost_drops = 0;
        for (variant, params) in variants {
            for set in testutil::lock_sets() {
                let trace_path = testutil::vector_path(&set, &format!("lock_{variant}_trace.csv"));
                let soft: Vec<f64> =
                    testutil::read_bits_csv(&testutil::vector_path(&set, "bits.csv"))
                        .iter()
                        .map(|&(_, _, soft_value)| soft_value)
                        .collect();
                let expected = testutil::read_trace_states(&trace_path);

                let mut lock = SoftLock::new(params.clone());
                let mut events = Vec::new();
                let mut seen = 0;
                for &value in &soft {
                    let releases_before = lock.releases();
                    let new_events = lock.push(value);
                    // Characters come out before the step's own release is counted.
                    assert!(
                        new_events.iter().all(|e| e.releases == releases_before),
                        "{variant}, {set}: release number of the characters"
                    );
                    events.extend(new_events);
                    if lock.comparisons() > seen {
                        // One new comparison has just been scored.
                        let want = &expected[seen];
                        let what = format!("{variant}, {set}, comparison {seen}");
                        assert_eq!(lock.is_locked(), want.locked, "locked: {what}");
                        assert_eq!(
                            lock.hypothesis().map_or(-1, |h| h as i64),
                            want.h,
                            "hypothesis: {what}"
                        );
                        assert_eq!(lock.releases(), want.releases, "releases: {what}");
                        assert_eq!(lock.switches(), want.switches, "switches: {what}");
                        assert_eq!(lock.pending_len(), want.pending, "pending: {what}");
                        assert_eq!(lock.emitted(), want.emitted, "emitted: {what}");
                        seen += 1;
                    }
                }
                assert_eq!(seen, expected.len(), "{variant}, {set}: number of comparisons");
                let flushed = lock.finish();
                assert!(flushed.iter().all(|e| e.releases == lock.releases()));
                events.extend(flushed);
                assert!(
                    events.windows(2).all(|w| w[0].releases <= w[1].releases),
                    "{variant}, {set}: release numbers should never go down"
                );

                // Every character, and the bit it came from, must match.
                let want_chars = testutil::read_lock_chars(&testutil::vector_path(
                    &set,
                    &format!("lock_{variant}_chars.csv"),
                ));
                let got_chars: Vec<(usize, String)> =
                    events.iter().map(|e| (e.bit_index, e.ch.to_string())).collect();
                assert_eq!(got_chars.len(), want_chars.len(), "{variant}, {set}: number of characters");
                for (k, (got, want)) in got_chars.iter().zip(&want_chars).enumerate() {
                    assert_eq!(got, want, "{variant}, {set}: character {k}");
                }

                // And the text itself, as the Python wrote it out. Git may have
                // changed the line endings of the text file (Windows checkouts
                // often do), so carriage returns are ignored in this comparison;
                // the character list above already checked every one of them.
                let text: String =
                    events.iter().map(|e| e.ch).filter(|&c| c != '\r').collect();
                let want_text: String = std::fs::read_to_string(testutil::vector_path(
                    &set,
                    &format!("decoded_{variant}.txt"),
                ))
                .expect("cannot read the decoded text")
                .replace('\r', "");
                assert_eq!(text, want_text, "{variant}, {set}: decoded text");

                println!(
                    "{variant:6} {set}: {seen} comparisons, {} characters, {} releases ({} by the own-alignment rule), {} switches, {} letter runs and {} idle runs dropped, all match",
                    events.len(),
                    lock.releases(),
                    lock.own_releases(),
                    lock.switches(),
                    lock.letter_run_drops(),
                    lock.ghost_runs_dropped()
                );
                total_chars += events.len();
                total_releases += lock.releases();
                total_switches += lock.switches();
                total_own_releases += lock.own_releases();
                total_run_drops += lock.letter_run_drops();
                total_ghost_drops += lock.ghost_runs_dropped();
            }
        }
        assert!(total_chars > 0, "the test never saw a character, so it proves nothing");
        assert!(total_releases > 0, "the test never saw a release, so it proves nothing");
        assert!(total_switches > 0, "the test never saw a switch, so it proves nothing");
        assert!(total_own_releases > 0, "the test never saw the own-alignment release rule");
        assert!(total_run_drops > 0, "the test never saw the letter-run filter");
        assert!(total_ghost_drops > 0, "the test never saw the idle look-alike filter");
    }
}
