# HANDOFF — Weft 2

**Project:** Weft Inner PAM (continuous-trajectory associative memory, post-architectural-rethink)
**Repo:** `/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/`
**Status:** Fresh repo, bootstrapped. No code yet. Awaiting encoder substrate verification.

---

## What this repo is

This is a fresh repository for the Weft project, built around the architecture articulated in `WEFT_INNER_PAM_v0_Spec.md`. The previous repo at `/mnt/c/Users/Jason/Desktop/Eridos/Weft/` contains four iterations of negative results that established the previous architecture (next-frame prediction with cosine retrieval) was building the wrong thing. The new architecture is path-prediction with Gaussian negative-log-likelihood loss, learning trajectory shapes through repetition. See the spec for full claims.

The previous repo stays in place as historical record. This repo does not edit it or share state with it.

---

## What's been done

- Repo bootstrapped per `instructions/` setup batch.
- `CODING_STANDARDS.md`, `research_operations_v1.md`, `WEFT_INNER_PAM_v0_Spec.md` carried forward.
- Encoder substrate verification (per `instructions/ENCODER_SUBSTRATE_VERIFICATION.md`) **complete — verdict FAIL.**

---

## Encoder substrate verification — verdict FAIL (2026-04-30)

Read-only protocol from `WEFT_INNER_PAM_v0_Spec.md` §5 against the
seed-7 furniture-run bank in the previous repo. Headline numbers from
`results/encoder_verification/verification_data.json`; full breakdown
in `results/encoder_verification/ENCODER_VERIFICATION_REPORT.md`.

| check | aggregate | starting threshold | result |
|---|---:|---|---|
| 1. cross-instance stability (mean cosine, n = 250 pairs across 5 items) | `1.0000` | `> 0.75` | PASS (degenerate — see report §7) |
| 2. cross-element distinguishability (mean cosine, n = 1000 pairs across 20 ordered pairs) | `0.8697` | `< 0.60` | **FAIL** (load-bearing) |
| 3. combined gap (Check 1 − Check 2) | `0.1303` | `≥ 0.15` | FAIL |

**Verdict: FAIL.** Encoder does not meet the protocol on this bank.

**Why FAIL is the right call (load-bearing finding):** Check 2 is the
real failure — V-JEPA 2 mean-pool produces cross-element cosines
ranging `0.8347` (Bed ↔ Dresser) to `0.9210` (DiningTable ↔ Sofa) for
the 5 furniture items in seed 7's house. All 10 distinct cross-pair
values are far above the 0.60 starting threshold. This is consistent
with the prior Stage 0b room-distinctness diagnostics: V-JEPA 2 mean-
pool's geometry is dominated by scene context, not the recurring
unit. Check 2's failure does *not* depend on Check 1.

**Caveat — Check 1 is degenerate, not informative.** The seed-7
furniture-run dwell mechanism teleports the agent to the *exact same
pose* every dwell frame, every loop, so AI2-THOR renders bit-identical
pixels and V-JEPA 2 (deterministic, frozen) produces bit-identical
embeddings. The within-instance cosine of `1.0000` with std `0.0000`
across all 50 sampled pairs at all 5 items reflects this — it is
measuring rendering determinism, not encoder stability under natural
instance variation. Spec §5.1 was written assuming instances would
carry natural variation (different angles, lighting, etc.); this bank
does not provide that. Same artifact appears in §3's per-pair std =
`0.0000` for every ordered pair: with bit-identical embeddings within
an item, sampling 50 pairs reduces to one cosine repeated 50 times.

The verdict therefore stands on Check 2 alone. Check 3's gap of
`0.1303` corroborates rather than adds independent signal: it is the
1.0 (degenerate) minus the 0.87 (real). A non-degenerate Check 1 (on
varied instances) would lower its mean and shrink the gap further.

**Per spec §5.5,** v0 implementation does not proceed on this encoder
without substrate work. The decision (alternative frozen encoder,
fine-tuning, redefining the recurring unit) is human review, not
autonomous.

---

## Next immediate action

**Session 2: launch Phase 1 main training run.** Code scaffolding is complete and unit-tested; precomputed embeddings file is full-stream and verified; reviewer has approved DINOv2 as the v0 encoder (see Session-1 outcomes below). Before launching:

1. **Host-protection prerequisites (carryover from session 1 — user-side actions).** Two items must be set on the Windows host before training launches:
   - **Disk sleep:** Settings → Power → Additional power settings → Change plan settings → Change advanced power settings → Hard disk → "Turn off hard disk after" → set to **0 (Never)** on AC. Current state at end of session 1 was 20 min on AC (`0x000004b0`), which would interrupt a ~30-90-min Phase 1 run.
   - **Windows Update pause:** Settings → Windows Update → Pause updates for at least the planned batch window. No pause was active at end of session 1.
   - The screen-sleep setting was already "Never" — no action needed.
2. **tmux decision.** Optional for session 2 (training launches with `nohup` so it survives shell death anyway). Recommended if you plan to watch live: `tmux new -s weft` before launching the CC session.
3. **Launch:** `nohup python3.12 -u scripts/run_phase1_train.py > logs/phase1_main_$(date +%Y%m%d_%H%M%S).log 2>&1 &` then poll the log per CODING_STANDARDS §5.3. Expected wall-clock ≈ 27-30 min on RTX 4080 Super (smoke run measured 60 grad steps/sec, ~95.7k training steps).
4. **Shuffle control runs sequentially after main** (per instr §7.5, VRAM contention). Same script with `run_phase1_shuffle.py`.
5. **Per-checkpoint eval** with `scripts/run_eval.py --checkpoint ... --probes phase1 --output ...` paused-for-training per instr §6.4.
6. **End of session 2:** evaluate G1.1-G1.5 gates, update HANDOFF with results, do not proceed to Phase 2 without explicit gate review.

---

## Operational state (end of session 1)

- Working tree: clean. Eleven commits added in session 1 (see "Session 1 outcomes" below).
- Push hold: in effect.
- No running jobs.
- Embeddings file at `data/dinov2_embeddings/embeddings.npy`: 100,000 × 1024 fp32, all rows L2-normed (min cosine 1.000000 vs archived dwell-only file).
- Archived dwell-only file: `data/dinov2_embeddings/embeddings_dwell_only_v1.npy`.

---

## Session 1 outcomes — 2026-05-13

**Goal.** Build the v0 code scaffolding ready for session-2 Phase 1 training launch. Not training itself.

**DINOv2 reviewer approval — recorded.** The reviewer approved DINOv2 ViT-L/14 CLS as the v0 encoder on 2026-05-12, citing the substrate-verification + stability batch results: **Check 1 = `0.9260`, Check 2 = `0.4422`, Check 3 gap = `0.4838` — all PASS** against the §5 starting thresholds. The "human review of the DINOv2 stability PASS verdict" gate from the previous next-immediate-action is closed. v0 proceeds on DINOv2 ViT-L/14 CLS.

**STOP caught and resolved at pre-flight: embeddings file was dwell-only.** Pre-flight verification of `data/dinov2_embeddings/embeddings.npy` found that the file had the expected shape `(100000, 1024)` but **only 32,760 of the 100,000 rows were L2-normalised; the remaining 67,240 rows had norm = 0.0** (transit frames). The substrate-verification batch only needed dwell frames; transit frames were never encoded. Phase 1 training requires a contiguous stream (spec §2.3, instr §7.2). Stop reported, full-stream re-encode authorised by reviewer with one tightening (consistency threshold 0.999 → 0.9999).

Re-encode (`scripts/run_dinov2_encode_full_stream.py`, commit `a86c6f0`):
- Protocol: facebook/dinov2-large, frozen, fp16 eval, 256→224 center crop, ImageNet mean/std, CLS token, L2-normalise (same as substrate-verification).
- Wall-clock: 218.6 s on RTX 4080 Super, fp16, batch 64 (~457 frames/s).
- Norm check on all 100,000 rows: PASS (norms in [1−1e-5, 1+1e-5]).
- Consistency check on 50 random dwell frames against `embeddings_dwell_only_v1.npy`: **min cosine = 1.000000** (threshold 0.9999) — DINOv2 forward is bit-identical-deterministic on identical pixels.
- Report: [`data/dinov2_embeddings/encode_full_stream_report.json`](data/dinov2_embeddings/encode_full_stream_report.json) (gitignored).

**Documentation corrections caught at pre-flight.** Two items in `WEFT_INNER_PAM_v0_EXPERIMENT_INSTRUCTIONS.md` were inconsistent with actual repo state:

| location | original | corrected | how caught |
|---|---|---|---|
| §0 Environment Header | "Python 3.10 (target match to previous repo)" | Python 3.12.3, matching the previous repo's `requirements.txt` header which explicitly says "WSL2, Python 3.12.3, CUDA 12.8 via WSL2 passthrough" | comparing §0 against the old repo's `requirements.txt` comments at pre-flight |
| §0 venv | `Active venv: .venv at repo root` | "none; uses system Python 3.12.3, matching the previous repo's verified-working pattern" | no `.venv` exists; the old repo also used system Python |
| §1.2 / §7.2 (embeddings precondition) | "100,000 frames × 1024 dim, fp32, L2-normalised" — implicitly assumed all 100k rows populated | (now true after the session-1 re-encode; no doc change needed, but the gap was load-bearing and is captured in this entry) | direct inspection of the file's norm distribution |

The §0 corrections are committed in `cc0a6a8`. Both errors were caught **before** any training launch, which is the design intent of the §7.2 / §4.7 norm checks.

**Code scaffolding delivered.** Eleven commits stand up the full Phase 1 pipeline:

| commit | scope | files |
|---|---|---|
| `e640dde` | infra | `requirements.txt`, `.gitignore`, `src/config.py`, all `src/*/__init__.py`, `src/env/explorer_phase{1,2,3}.py` stubs |
| `a86c6f0` | encoding | `scripts/run_dinov2_encode_full_stream.py` |
| `3016f23` | memory | `src/memory/memory_bank.py` (FAISS, hard cap, BankCapExceededError) |
| `a820ce1` | predictor | `src/predictor/inner_pam.py` (4-layer transformer, K*(d+1) head, Gaussian NLL) |
| `11e3f41` | mixing | `src/mixing/recall_mixer.py` (confidence threshold, τ calibration helper) |
| `567799f` | trainer | `src/trainer/online_trainer.py` (single-pass loop + §4.7 init-time checks) |
| `2dd3ae9` | eval | `src/eval/probes.py`, `metrics.py` (M1-M7), `controls.py` (C1 + S1-S4) |
| `4938a50` | encoder | `src/encoder/dinov2_encoder.py` (Phase 2/3 wrapper) |
| `715ba21` | scripts | `scripts/run_phase1_train.py`, `run_phase1_shuffle.py`, `run_eval.py` (with `--developmental` flag wired) |
| `b03062d` | tests | 21 tests across predictor / memory / mixer / probes / embeddings-norm invariant (all pass) |
| `cc0a6a8` | docs | `WEFT_INNER_PAM_v0_EXPERIMENT_INSTRUCTIONS.md` §0 correction |

**Verification before commit (per instr §4.7 / instr §15-style review-cycle equivalents):**

- **21 unit tests pass** on system Python 3.12.3 / pytest 9.0.3: predictor shapes + param count (21,555,728 trainable params, within 2.6% of the 21M target — well inside the 10% tolerance), Gaussian-NLL closed-form sanity, log_var clamp at [−10, 10], target-detached-from-grad, memory-bank append + retrieve + hard-cap + FIFO + save/load round-trip, mixer routing + tau median calibration, probe construction (held-out boundary, steady-state uniform-dwell, cue dwell-to-transit), and an explicit "no-zero-rows" guard on `embeddings.npy` that would catch the dwell-only failure mode if it ever recurs.
- **§4.7 init-time smoke run on real Phase 1 data** (300-step budget): all four §4.7 checks pass — encoder frozen-equivalent (DINOv2 not loaded at training time; embeddings are precomputed), predictor trainable (21.6M params), forward pass produces correct shapes `(2, K, d)` + `(2, K)`, embedding norm check passes on 1000 sampled rows. 270 gradient steps in 4.3 s, no NaN/Inf, loss trended monotonically downward (first-50 mean ≈ −13,985 → last-50 mean ≈ −30,265 — Gaussian NLL is unbounded below; only the trend is informative). Bank populated correctly. Smoke artefact deleted before commits.

**Estimated session-2 budget.** At ~60 grad steps/sec, full Phase 1 (~95,700 training steps) is ~27 min plus checkpoint I/O. Shuffle control adds another ~27 min sequentially. Eval at 10 checkpoints × ~2 min/ckpt ≈ 20 min. Total session-2 wall-clock ≈ 75-90 min before gate review.

**Operational divergences from the instructions that are now resolved or recorded:**

1. **Python 3.12.3, not 3.10.** Doc corrected (commit `cc0a6a8`). System Python directly; no `.venv`. Matches the substrate-verification batch.
2. **`.env_snapshot.txt` written** (`pip freeze`, 207 packages) and gitignored per CODING_STANDARDS §8.4.
3. **`requirements.txt`** is pinned to the substrate-verification batch's stack plus `scipy==1.17.1` (used by G1.4 / G2.3 / G3.3 t-test + Wilcoxon fallback).
4. **Bug fix during session-1 encoding:** the encode script's temp-file rename relied on `Path.with_suffix(".npy.tmp")` which produced `.npy.tmp`, but `np.save` auto-appends `.npy`, so the file actually landed at `.tmp.npy` and the rename-to-final failed at the end of the encode. The work (encode + checks) had already completed cleanly before the rename; manual rename completed the artefact handover. Script fixed in the same commit so what's in git is what would work clean on a re-run.
5. **GPU has 3.4 GB used by the Windows desktop compositor.** Acceptable (12.6 GB free for training). No compute processes; no other ML jobs.

**Push hold remains in effect.**

---

## DINOv2 substrate verification batch — STOP (2026-04-30)

The DINOv2 substrate verification batch was issued to re-run the §5
protocol against DINOv2 ViT-L/14 on the seed-7 furniture-run frames,
for direct comparison to the V-JEPA 2 result.

**Stop trigger.** Per the batch §2.1 and §6, the seed-7 furniture-run
**source RGB frames are not retained** in the previous repo. Only
encoded V-JEPA 2 embeddings, per-frame annotations, and metadata are
present. The original training script encoded each frame in the
forward pass and discarded the pixels. DINOv2 cannot be evaluated
without re-encoding, and re-encoding requires source pixels.

**Resolution:** The reviewer authorised a full re-render (next entry).

*STOP commit:* `aefa1bc`.

---

## DINOv2 cross-instance stability under per-frame jitter — PASS (2026-05-12)

Fills the Check 1 gap left by the prior DINOv2 verification, whose
aggregate `1.0000` was a tautology (bit-identical pixels → bit-
identical embeddings). New collection: one full loop of the seed-7
furniture route with **per-frame** position+heading jitter applied
inside the explorer's dwell teleport, so every dwell frame has a
genuinely different pose.

**Spec interpretation decision (documented per CODING_STANDARDS §9.2).**
Batch §3 reads "apply per-loop jitter … the agent then dwells at the
jittered pose" (one jitter per visit) but §5 expects "~30 unique
frames per viewing position from one jittered loop" and §9 stops on
"fewer than 15 dwell frames per viewing position". One-jitter-per-
visit on a single loop gives 1 unique pose per item → Check 1 is
degenerate again, exactly the failure mode the batch was built to
fix. Per-frame jitter is the only interpretation consistent with §5's
sample-count expectation, so per-frame is what was implemented. RNG
seeded once with `jitter_seed=7`, drawn sequentially in frame order
for reproducibility. Flagging for review.

**Collection** (previous repo, `scripts/run_furniture_stability_collect.py`):

  - 458 frames total, one loop. Wall-clock 25.0 s.
  - 30 dwell frames at each of items 1..5 (150 total dwell); 308
    transit.
  - Jitter: `position_m=0.2` per horizontal axis, `heading_deg=10.0`,
    fallback ladder 100% → 50% → 25% → unjittered for NavMesh-
    unreachable poses. **Zero fallbacks** — all 150 jittered teleports
    succeeded at full 100% magnitude.
  - frames at [`data/seed7_dinov2_stability_frames/`](data/seed7_dinov2_stability_frames/)
    (PNG, ~12 MB total, gitignored); annotations at
    [`data/seed7_dinov2_stability_annotations.jsonl`](data/seed7_dinov2_stability_annotations.jsonl).
  - Modification in previous repo: `src/env/furniture_route_explorer.py`
    (jitter logic with fallback ladder, opt-in via constructor args),
    `src/env/ai2thor_furniture_env.py` (pass-through), and new
    `scripts/run_furniture_stability_collect.py` (pure data
    extraction, no V-JEPA 2 / predictor / trainer).

**DINOv2 stability test** (new repo, `scripts/run_dinov2_stability_test.py`):

| viewing_position_id | object | n pairs | mean | std | min | max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Bed | 50 | `0.9467` | `0.0289` | `0.8475` | `0.9889` |
| 2 | DiningTable | 50 | `0.9447` | `0.0189` | `0.9037` | `0.9755` |
| 3 | Dresser | 50 | `0.9317` | `0.0453` | `0.7834` | `0.9847` |
| 4 | Sofa | 50 | `0.8524` | `0.0969` | `0.6682` | `0.9749` |
| 5 | Television | 50 | `0.9547` | `0.0223` | `0.9034` | `0.9846` |

**Aggregate**: mean **`0.9260`** (n=250), std `0.0635`, min `0.6682`,
max `0.9889`. Pass criterion (>0.75): **PASS** with margin `0.176`.

**Pattern noted, not a finding.** Sofa is the least stable item (mean
0.8524, std 0.0969, min 0.6682). Sofa also produced the highest
cross-element pair in the prior Check 2 (DiningTable↔Sofa = 0.6709).
Coincidence is plausible; the report flags but does not interpret
the pattern.

**DINOv2 full §5 status (combining Check 1 from this batch with
Checks 2/3 from the prior DINOv2 verification on the same encoder):**

| check | DINOv2 | starting threshold | result |
|---|---:|---|---|
| 1 (cross-instance stability, non-degenerate jitter substrate) | `0.9260` | `> 0.75` | PASS |
| 2 (cross-element distinguishability, prior) | `0.4422` | `< 0.60` | PASS |
| 3 (combined gap, `0.9260 − 0.4422`) | `0.4838` | `≥ 0.15` | PASS |

Full report:
[`results/encoder_verification_dinov2_stability/STABILITY_REPORT.md`](results/encoder_verification_dinov2_stability/STABILITY_REPORT.md);
raw cosines + jitter summary in
[`results/encoder_verification_dinov2_stability/stability_data.json`](results/encoder_verification_dinov2_stability/stability_data.json).

**Caveat (recorded in the report §6).** Jitter magnitudes `0.2 m` /
`10°` are SCAFFOLDING values per the batch's §3 — verdict is
conditional on this magnitude. A non-trivially different magnitude
could produce a different aggregate; the protocol does not first-
principle-derive the jitter range from a model of natural agent-
instance variation. Flagged for reviewer.

*Stability commit: pending in both repos.*

---

## DINOv2 substrate verification on rerendered seed-7 frames — PASS (2026-05-12)

DINOv2 ViT-L/14 CLS, frozen, fp16 eval, encoded over the rerender's
32 760 dwell frames at items 1..5 (224×224 center crop of the 256×256
source, ImageNet mean/std). Same protocol, same seeds (7 / 8), same
pair counts, same sampling procedure as the V-JEPA 2 verification —
encoder is the only variable.

| check | DINOv2 (this batch) | starting threshold | V-JEPA 2 (prior) | DINOv2 result |
|---|---:|---|---:|---|
| 1. cross-instance stability (mean cosine, 250 pairs) | `1.0000` | `> 0.75` | `1.0000` | PASS (degenerate — see below) |
| 2. cross-element distinguishability (mean cosine, 1000 pairs) | **`0.4422`** | `< 0.60` | `0.8697` (FAIL) | **PASS** (load-bearing) |
| 3. combined gap (Check 1 − Check 2) | **`0.5578`** | `≥ 0.15` | `0.1303` (FAIL) | **PASS** |

**Verdict: PASS** (no recalibration applied; empirical values are not
within ±0.05 of the starting thresholds). Per-pair Check 2 means span
`0.2547` (DiningTable ↔ Television) to `0.6709` (DiningTable ↔ Sofa);
DiningTable ↔ Sofa is the only ordered pair above 0.60, and the
aggregate is still 0.16 below the threshold. Full per-pair matrix and
V-JEPA 2 side-by-side in
[`results/encoder_verification_dinov2/ENCODER_VERIFICATION_DINOV2_REPORT.md`](results/encoder_verification_dinov2/ENCODER_VERIFICATION_DINOV2_REPORT.md);
raw cosines in
[`results/encoder_verification_dinov2/verification_data.json`](results/encoder_verification_dinov2/verification_data.json).

**Check 1 carries the same degeneracy caveat as the V-JEPA 2 result.**
Within-position dwell frames are bit-identical across loops within the
rerender, so DINOv2 (deterministic eval-mode forward) produces bit-
identical embeddings — std `0.0000` across all 50 within-instance
pairs at all 5 items. The verdict stands on Check 2, which is genuine
encoder discrimination on bit-identical pixels and is not a sampling
artifact: every per-pair std is `0.0000` for the same reason (one
distinct cosine value per ordered pair), but the *values themselves*
are how DINOv2 separates the 5 items. The 10 distinct cross-pair
values range `0.2547`–`0.6709`, against V-JEPA 2's `0.8347`–`0.9210`
on the same items — a ~0.43 reduction in aggregate cross-element
similarity.

**Caveat (from RERENDER_REPORT).** Items 3 (Dresser) and 4 (Sofa) —
both LivingRoom — have constant per-item offsets from the original
V-JEPA 2 bank at the cosine `0.0005`–`0.0008` level *when read by
V-JEPA 2*. DINOv2 re-encodes the rerender's frames directly, so its
numbers are internally consistent. Recorded in case downstream
analysis surfaces an unexplained discrepancy at that magnitude; not
load-bearing on the verdict.

**Compute:** ~75 s of GPU forward (RTX 4080 Super, fp16, batch 64) +
~15 s for sampling / I/O. 32 760 dwell frames; one embedding per
frame; encoded once and saved to
`data/dinov2_embeddings/embeddings.npy` (391 MB, gitignored).

*Verification commit: pending.*

---

## Seed-7 furniture re-render with frames saved — PASS-AFTER-RECALIBRATION (2026-05-12)

**Final verdict updated 2026-05-12.** Reviewer applied a one-time
threshold recalibration from 0.9999 to 0.999 under spec §5.5; all 50
sampled frames pass the recalibrated threshold (`cos_min = 0.999188`).
Recalibration justification and final report:
[`results/frame_rerender/RERENDER_REPORT.md`](results/frame_rerender/RERENDER_REPORT.md).
Original stop record preserved in
[`STOP_REPORT.md`](STOP_REPORT.md) (commit `56050cc`), now marked
superseded. Frames at [`data/seed7_furniture_frames/`](data/seed7_furniture_frames/)
are usable substrate for downstream encoder verification.

The audit trail below is preserved from the original 2026-05-01 entry.

---

### Original entry (2026-05-01) — superseded by the 2026-05-12 recalibration above

The reviewer authorised re-rendering the seed-7 furniture run with
frames written to disk so DINOv2 (and any future encoder) could be
verified on the same substrate the V-JEPA 2 verification analysed.

**The re-render itself completed cleanly:**
  - 100 000 frames saved as `frame_{idx:08d}.png` to
    `data/seed7_furniture_frames/` (~5.2 GB, gitignored).
  - 218 loops completed — matching the original run's loop count
    exactly.
  - Wall-clock 11 219 s (~3.1 hr); ~5 min slower than the original
    due to PNG-write overhead.
  - `frame_annotations.jsonl` is **bit-identical** to the original
    run's (same md5 `6f241260...`); the explorer's trajectory and
    per-frame metadata are deterministic.
  - Modified script committed in previous repo as `98578d3`
    (`feat(furniture-rerun): save frames during forward pass for
    verification reuse`) — opt-in flags only; original behaviour
    preserved when neither flag is set.

**Determinism check FAILED at the spec'd 0.9999 threshold.** Re-encoded
50 sampled frames (10 per viewing position) through the same V-JEPA 2
checkpoint; compared cosine to original bank entries.

| viewing position | object type | room | n samples | cos (mean = min = max) | < 0.9999 |
|---:|---|---|---:|---:|---:|
| 1 | Bed | Bedroom | 10 | `1.000000` | 0/10 |
| 2 | DiningTable | Bedroom | 10 | `1.000000` | 0/10 |
| 3 | Dresser | LivingRoom | 10 | `0.999188` | **10/10** |
| 4 | Sofa | LivingRoom | 10 | `0.999481` | **10/10** |
| 5 | Television | Bedroom | 10 | `1.000000` | 0/10 |

**Pattern:** Bedroom items render bit-identically across runs (cos =
1.000000 exactly). LivingRoom items 3 and 4 differ from the original
by a small, item-specific, run-constant amount — every sampled
frame at item 3 has cos `0.999188` exactly; every sampled frame at
item 4 has cos `0.999481` exactly. The re-render is deterministic
*within* a run (frames at the same item across loops are bit-
identical, consistent with the V-JEPA 2 verification's degenerate
Check 1) but differs from the original *between* runs at the two
LivingRoom items.

**Most plausible cause:** scene-state-dependent rendering on first
entry to LivingRoom (shader compilation order, asset upload,
physics settling on instantiation). Bedroom is the spawn room and
warms before LivingRoom is ever rendered, so its rendering is stable
across runs. Once LivingRoom is "warm" within a run, it renders
deterministically — explaining the within-run consistency.
Numerically, the cosines correspond to L2 distances of 0.040 / 0.032
between unit vectors, ≈14–17× closer to "identical" than typical
inter-furniture cross-element distances (~0.55) — but the threshold
is 0.9999 and the protocol's stop trigger is "any sample below". Per
spec §5.5 / batch §9, recalibration is reviewer-only; the script does
not recalibrate the threshold autonomously.

**Per the batch §5 and §8, this is an unconditional stop.**

**Full evidence + four reviewer options** in `STOP_REPORT.md` at the
project root. Options range from a one-time threshold relaxation
(items 3 and 4 cluster near `0.999`, well above any plausible
"different content" floor) to investigating AI2-THOR non-determinism,
running DINOv2 on the re-render with the caveat documented, or
treating the V-JEPA 2 result as final and skipping alternative-encoder
verification on this bank.

**Operational state.**
  - Working tree: clean modulo this stop's commits.
  - `data/seed7_furniture_frames/` (5.2 GB), `data/seed7_furniture_rerender_aux/` (411 MB) gitignored.
  - Push hold: in effect.
  - No running jobs.

*STOP commit: pending.*
