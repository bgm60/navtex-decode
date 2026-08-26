# NAVTEX Decoder — Tuning Reference

This documents every constant in the pipeline that affects decode quality,
what it controls, and what changing it trades off against. It exists so
you can experiment against real weak/fading signals without having to
read the source to figure out what each number does first.

**Context for all of this:** every value currently in the code was tuned
against one strong, clean recording (`Sample_C.wav`) plus synthetic
Gaussian noise. Real HF/MF fading doesn't look like Gaussian noise — it's
bursty, has deep fades, sometimes phase discontinuities. So treat every
"current value" below as a reasonable starting point validated under
different conditions than the ones you actually care about (weak/fading
DX), not as a value that's known to be right for that use case. (Update:
`BitSync.loop_gain` is the one exception to this — see below, it *has*
now been tuned directly against real weak/fading recordings.)

**Your stated goal** is to favor catching a weak, fading signal over
avoiding the occasional corrupt decode — so in general, the directions
below that say "faster/looser" are the ones aligned with what you're
trying to do. But every one of them trades against *something*, spelled
out per-parameter below.

**How to experiment safely:** change one parameter at a time. After each
change, run `python test_step4_roundtrip.py` — it should still pass
cleanly; if it doesn't, the change broke something structural, not just
shifted a trade-off. Then test against a real recording. For anything
touching Step 3/4 timing or locking behavior, `navtex_confidence_plot.py`
is useful for seeing *where* in a recording confidence drops, which helps
tell a real weak-signal problem apart from an acquisition/churn problem.

---

## Do not tune these

A few values are fixed by the protocol itself, not free parameters:

- **`baud` (100.0)** and **`sample_rate` (48000)** — NAVTEX is defined as
  100 baud; sample rate is a property of your audio hardware, not a
  decoding choice.
- **`FecCombiner.LAG` (5)** — this is the SITOR-B DX/RX time-diversity
  spacing itself (see Step 4's docstrings for how this was derived and
  verified against a real transmitter implementation). Changing it
  doesn't "tune" anything — it makes DX/RX comparison structurally wrong.
- **`mark_freq` / `space_freq`** — these aren't tuning parameters, they're
  a calibration specific to your receiver's audio center frequency and
  tone orientation. Re-run `calibrate_tone_frequencies.py` against a
  fresh recording if these ever need to change (new receiver, new SDR
  software, different tuning offset) rather than hand-adjusting them.

---

## Step 1 — Sampling & windowing (`navtex_step1_sampling_windowing.py`)

### `NavtexConfig.oversample` (currently 8)

**What it controls:** how many overlapping analysis frames are produced
per symbol period. Frame length (`window_size`) stays fixed at one full
symbol period regardless of this value — oversample only controls the
*hop* between frames (`hop_size = samples_per_symbol / oversample`), i.e.
how many independent "looks" downstream stages get per bit.

**Increase it:** more looks per bit for Step 3's bit-clock recovery to
work with — potentially finer timing resolution and a smoother `diff`
signal for the PLL to track. Costs: more computation (proportional to
oversample), and every downstream stage that counts "how many frames per
symbol" (notably `BitSync`'s phase step) scales with it — this is a
foundational value, not a local one, so changing it means re-validating
the whole pipeline, not just one stage.

**Decrease it:** less computation, but coarser timing resolution — Step
3's PLL has fewer, further-apart samples to correct against, which likely
hurts under fading (where you want the loop reacting quickly to a
signal that's coming and going).

**My honest take:** I would not start here. It's the most invasive value
to change (everything downstream assumes it), and the PLL/loop-gain
territory in Step 3 is a more targeted place to address the "holding
lock under fading" issue you're seeing.

### `NavtexConfig.window_type` (currently `"hamming"`)

**What it controls:** the window function applied before Goertzel tone
detection. Any name `scipy.signal.get_window` accepts works (`"hann"`,
`"blackman"`, `"rectangular"`, etc.).

**Effect:** windowing trades spectral leakage (how much energy from one
tone bleeds into the other's detector — matters since mark/space are only
170 Hz apart) against effective noise bandwidth and mainlobe width.
Hamming is a reasonable general-purpose choice. A window with lower
sidelobes (e.g. Blackman) would reduce mark/space crosstalk further at
the cost of a wider mainlobe (slightly worse frequency resolution) —
plausibly worth trying since your BPF already removes most out-of-band
interference before it reaches the detector, so crosstalk *between the
two NAVTEX tones themselves* may be more of a limiting factor for you
than broadband noise rejection.

---

## Step 3 — Bit-clock recovery (`navtex_step3_bit_sync.py`)

### `BitSync.loop_gain` (currently 0.05)

**What it controls:** how strongly each detected bit transition corrects
the PLL's phase estimate. This is the main knob for "how well does it
hold lock."

**Status: already hand-tuned against real weak-signal recordings.** This
started at 0.25, an earlier pass lowered it to 0.19, and it has since
been tuned further down to 0.05 through direct experimentation against
real weak/fading DX — and measured as the best-performing value found so
far for that specific goal. This is *outside* the range that would
normally be recommended from first principles (see the trade-off
description below) — at 0.05 the loop barely reacts to any single
transition's timing-error measurement. That it still performs best in
practice suggests noise-induced spurious transitions are a bigger
problem for this receiver/signal chain than slow reacquisition is, at
least for the recordings tested so far. Don't reflexively "correct" this
back toward 0.19–0.35 without re-testing against real weak-signal audio
— the earlier reasoning below (written before this tuning pass) no
longer reflects the validated-best value, only the general shape of the
trade-off.

**Increase it:** faster correction per transition — the loop reacts more
aggressively to each new piece of timing evidence. This is the
"widen the loop bandwidth" direction — more responsive to a signal that's
fading in and out, at the cost of being more easily perturbed by
noise-induced spurious transitions (a bit error can look like a
transition that isn't real, and a higher gain reacts to it more).

**Decrease it:** smoother, more noise-resistant tracking, but slower to
correct real timing drift — in principle this could make "holding lock
through a fade" worse, since the loop is sluggish to react once good
signal returns. In practice, for this project's real recordings, pushing
gain lower (down to 0.05) has outperformed the higher values tried
earlier — so this theoretical downside doesn't appear to dominate here,
though it's worth keeping an eye on if you test against a fresh batch of
recordings with different fading characteristics.

Note the known limitation already documented in the code: this loop is
proportional-only (Type-I) — it has no memory of an ongoing frequency
error, only reacts to instantaneous phase error. Under a *constant*
clock-rate mismatch this leaves a small steady-state error that no
amount of gain-tuning fully removes (a Type-II loop, tracking frequency
and not just phase, would be a structural fix rather than a tuning one —
worth keeping in mind if gain-tuning alone doesn't close the gap with
SeaTTY).

**Not the cause of the phasing/lock issue on specific stations:** now
confirmed by direct diagnosis (see "Open issues" below) — that issue was
never a PLL/loop_gain problem at all. It was a `FecCombiner` DX/RX parity
bug, now fixed. Left this note in place since the earlier confirmation
(that it predates and is unaffected by this tuning work) is still
accurate background.

---

## Step 4 — Character sync (`CharacterGrouper` in `navtex_step4_character_decode.py`)

These four work together as a group — see "Coupled parameters" below
before changing just one.

### `ACQUIRE_THRESHOLD` (currently 0.6)

**What it controls:** the minimum weight-4 hit rate a candidate phase
must show before `CharacterGrouper` will commit to it (both for initial
acquisition and for `_try_acquire` after a phase drop).

**Lower it:** faster/easier acquisition — useful if real acquisition
never gets a chance to reach 0.6 under your conditions. Costs: less
statistical separation from the ~27% pure-chance baseline (random noise
occasionally producing a "valid-looking" 7-bit group), meaning more risk
of locking onto pure noise and treating it as signal.

**Raise it:** more conservative, slower to acquire, less false-lock risk.

### `DROP_THRESHOLD` (currently 0.46)

**What it controls:** how far the *currently active* phase's score has to
fall before `CharacterGrouper` even considers abandoning it (see
`_reconsider`). Note this only *considers* dropping — actually switching
still requires `SWITCH_MARGIN` to be satisfied too (see below).

**Status: already raised from the original 0.4 to 0.46**, per the
suggestion below, favoring willingness to attempt recovery once a lock
starts struggling.

**Raise it further** (closer to `ACQUIRE_THRESHOLD`): more willing to
give up on a struggling lock and try to re-acquire. Given your priority
of holding onto weak/fading signal rather than defending against
occasional corruption, this is one of the more directly relevant knobs —
but see the important caveat under `SWITCH_MARGIN`.

**Lower it:** more tolerant of a lock that's performing poorly, sticking
with it longer before reconsidering — this is the "stubborn" direction,
generally more aligned with what SeaTTY appears to be doing well
(holding a lock through degraded conditions rather than abandoning it).

### `SWITCH_MARGIN` (currently 0.2)

**What it controls:** once a drop is being considered, how much better an
alternative phase has to look before `CharacterGrouper` actually switches
to it, rather than staying put. This exists specifically because
repetitive filler (the phasing signal, or mid-message resync bursts) can
score deceptively well at more than one phase simultaneously — without
this margin, the decoder would churn between phases that are merely tied,
not genuinely one correct and one wrong.

**Lower it:** switches more readily once a drop is being considered —
faster to jump to a genuinely better phase, but more willing to switch
based on a smaller, less certain difference, which is closer to the kind
of churn this parameter was added to prevent in the first place.

**Raise it:** stickier — needs much clearer evidence before abandoning
the current phase for another. Given the phasing-signal tie problem this
was built to solve, I would be cautious lowering this much below its
current value without testing carefully against a recording that
actually contains phasing/idle bursts, not just against a single clean
in-message stretch.

### `MIN_GROUPS_FOR_ACQUIRE` (currently 30)

**What it controls:** minimum sample size before *any* phase's score is
trusted for acquisition, regardless of how good it looks. This exists
because small samples spike above threshold by pure chance far more
often than the false-acquisition risk suggests — see the code comment
for the actual statistics (n=8 was only ~1.6 standard deviations of
separation from the phasing-preamble's noise ceiling; n=30 gives >3).

**Lower it:** faster acquisition, directly costs acquisition-time on
every cold start and every re-acquisition after a drop — this is likely
your single biggest lever for "acquires faster," but the further you
lower it below ~20-25, the more you're eroding the statistical safety
margin this value was specifically chosen to provide. I would not go
below the ~n=15-20 range without directly re-running the false-lock
statistics the way this value was originally derived (a rough recipe is
in the code comment) — going in blind here risks reintroducing the
"locks onto pure noise" failure mode this project spent real effort
fixing.

**Raise it:** more conservative, slower everywhere lock is needed.

*(Earlier notes here suggested this was a leading hypothesis for the
specific-station phasing bug. It wasn't — see "Open issues" below. Real
diagnosis showed `CharacterGrouper`'s phase lock survived the affected
recording fine throughout; the actual bug was in `FecCombiner`.)*

### `sync_window` (`CharacterGrouper.__init__`, currently 250 bits)

**What it controls:** how many trailing bits `CharacterSync` keeps to
score phases against. Must comfortably exceed `MIN_GROUPS_FOR_ACQUIRE * 7`
bits (30 × 7 = 210) — see coupling note below.

**Effect:** a larger window gives more stable, less noisy phase scoring
but reacts more slowly to a real change in alignment (e.g. after a
genuine resync). A smaller window (down toward the 210-bit floor) reacts
faster but with less averaging to smooth over noise.

---

## Step 4 — FEC parity locking (`FecCombiner` in `navtex_step4_character_decode.py`)

Same shape as the CharacterGrouper parameters above, applied to DX/RX
parity instead of character phase. One structural difference worth
knowing: **`FecCombiner` has no drop-threshold at all** — it never
abandons an already-locked parity just because it's performing poorly on
its own merits; it only switches when `_maybe_switch` finds the *other*
parity clearly, decisively better (governed by `SWITCH_MARGIN`, same
concept as above). So there's no direct FEC equivalent of
`DROP_THRESHOLD` to loosen.

### `ACQUIRE_THRESHOLD` (currently 0.6) and `SWITCH_MARGIN` (currently 0.2)

Same trade-offs as CharacterGrouper's versions, applied to parity instead
of phase.

### `MIN_SAMPLES_FOR_RATE` (currently 5)

**What it controls:** minimum comparisons before trusting a parity's
match-rate estimate at all.

**Context that matters for weak-signal tuning:** this was already lowered
once, from an original 10 down to 5, after checking directly against real
data — once the phasing period is properly excluded, the correct parity
shows a clean ~100% match starting from essentially the first possible
comparison, with zero ambiguity to average out. That check was against a
*strong* recording, though. Under weak/fading signal, individual bit
errors are more frequent, so the "clean, unambiguous separation"
assumption this value relies on is less safe — I would not lower this
further without re-running the same kind of false-lock check (inject bit
errors at realistic weak-signal rates, confirm no false locks) rather
than assuming the earlier result still holds.

### `RATE_WINDOW` (currently 30) and `lock_window` (currently 15)

**What they control:** how much codeword history is kept for computing
rolling match rates (`RATE_WINDOW`) and, jointly with it, sizing the
underlying history buffer (see coupling note below). Similar trade-off
shape to `sync_window` above — larger means smoother/more stable but
slower to react to a genuine parity change; smaller reacts faster but
with less averaging.

---

## `decode_bit_stream`'s phasing-burst detection threshold

In `navtex_step4_character_decode.py`, `decode_bit_stream` treats a run
of **4 or more** consecutive phasing codewords as a genuine phasing
burst. This 4-codeword threshold isn't a class constant, just a literal
in the function, but it's tunable the same way.

**As of the fix below, crossing this threshold now UNCONDITIONALLY
resets `FecCombiner`** (forces a fresh, statistically-gated DX/RX parity
re-acquisition), regardless of whether FEC was already locked coming in.
This used to be conditional on `fec._parity is None` (only reset if
unlocked) — see "Open issues" for why that was wrong and how this was
confirmed against a real recording. Separately, inside `_combine`, this
threshold-independent phasing-vs-phasing detection still suppresses
`*`/`!` output entirely for phasing-vs-phasing comparisons, unchanged.

**Lower it:** recognizes shorter bursts as phasing sooner — could matter
if a weak/fading signal only lets a few phasing codewords through cleanly
before corruption sets in, causing genuine phasing to go unrecognized.
Now also means FEC re-acquires slightly more eagerly after shorter
filler stretches, mid-message or between messages.

**Raise it:** more conservative about what counts as "definitely
phasing," at the cost of needing a longer clean run to trigger the
phasing-aware behavior — under fading, a longer clean run might simply
not happen often enough. Now also means FEC holds onto a (possibly
stale) parity lock longer before being forced to re-earn it.

---

## Coupled parameters — change these together, not independently

- **`MIN_GROUPS_FOR_ACQUIRE` (bits: ×7) must stay comfortably below
  `sync_window`.** Currently 30×7=210 bits vs a 250-bit window — only 40
  bits of margin. If you lower `sync_window` without also lowering
  `MIN_GROUPS_FOR_ACQUIRE`, acquisition could stall entirely (never
  enough groups in the window to satisfy the minimum).
  Trying 25 to see if faster acquisition produces a noticeable improvement
  in the real world.
- **`RATE_WINDOW` and `lock_window`, together with `LAG`, size
  `FecCombiner`'s history buffer** (`history_len = LAG + max(lock_window,
  RATE_WINDOW)`). Changing one without the other can shrink the
  effective history available for the *other* mechanism than the one you
  meant to adjust.
- **`ACQUIRE_THRESHOLD` and `SWITCH_MARGIN` interact multiplicatively,
  not independently**, in both `CharacterGrouper` and `FecCombiner`: a
  candidate must clear the threshold *and* beat the runner-up by the
  margin. Lowering one while leaving the other high may not actually
  change acquisition speed much if the other is now the binding
  constraint — check which one is actually limiting you before assuming
  a change will have the effect you expect.

---

## Open issues (as of this update)

1. **Phasing/lock failure on specific stations — ROOT-CAUSED AND FIXED,
   pending field confirmation.** Diagnosed against a real recording
   (`Sync_failure.wav`, two HAMBURG messages ~90s apart on one
   continuous carrier) with per-codeword instrumentation. Findings:
   - `CharacterGrouper`'s bit-level phase lock was **not** the problem —
     it acquired once, early, and never needed to reconsider for the
     entire 152s recording; both messages decoded with correct 7-bit
     framing throughout.
   - The actual bug was in `FecCombiner`'s DX/RX parity lock. The
     ~13-second inter-message phasing burst is period-2 content and
     structurally can't carry parity information (dx never equals rx
     during phasing, regardless of which parity is "true"). `FecCombiner`
     kept its parity locked from message 1 straight through the burst
     (per the old `fec._parity is None` guard). Whatever caused the true
     parity to differ across the burst (this recording didn't resolve
     whether that's a normal protocol characteristic of inter-message
     phasing timing, or an artifact of `BitSync`'s very low `loop_gain`
     giving fewer correction opportunities during a long run of
     non-alternating phasing tone — both remain open questions, but
     don't change the fix) meant every DX/RX comparison after the burst
     compared two unrelated-but-individually-valid real codewords —
     textbook '!' output — for ~20 characters (the opening `ZCZC SA02`
     and `NCC-HAMBURG` identification), until `FecCombiner`'s own passive
     match-rate tracking noticed and self-corrected.
   - **Fix:** `decode_bit_stream` now resets `FecCombiner` unconditionally
     after every ≥4-codeword phasing burst, not only when it was
     previously unlocked. Verified against `Sync_failure.wav`: message 2
     now decodes as `*ZCZC SA02` / `NCC-HAMBURG` / ... in full (down from
     20 lost characters to 1 stray leading `*`); message 1 is
     byte-for-byte unchanged; `test_step4_roundtrip.py` still passes.
   - **Still worth watching:** this was confirmed against one recording
     from one station (HAMBURG). The fix's mechanism doesn't depend on
     which station it is, so it should generalize, but please flag it
     if the "opening text missing" symptom recurs on any station after
     this update — that would mean there's a second contributing cause.
2. **Weak-signal optimization is ongoing.** `BitSync.loop_gain` has been
   tuned down to 0.05 with good real-world results; `DROP_THRESHOLD` has
   been raised to 0.46. Remaining candidates from "Suggested first
   experiments" below are still open.

## Suggested first experiments, given what you've told me

1. ~~**`BitSync.loop_gain`** — try raising it modestly (e.g. 0.25 → 0.35)
   first...~~ **Superseded.** This has since been tuned directly against
   real recordings and landed at 0.05 (lower, not higher, than the
   original value) — see the `loop_gain` section above for why, and
   don't re-run this specific experiment without new data.
2. ~~**`DROP_THRESHOLD`** (CharacterGrouper) — raising this slightly...~~
   **Done.** Raised from 0.4 to 0.46.
3. Everything else — I'd treat as second-tier until they've been tried
   against real weak/fading recordings and shown whether they close
   meaningful ground against SeaTTY, since they're higher-risk (touch
   statistical safety margins directly) for less clearly-targeted
   benefit given your specific complaint.

