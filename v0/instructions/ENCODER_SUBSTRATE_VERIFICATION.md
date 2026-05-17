# Encoder Substrate Verification — V-JEPA 2 Mean-Pool on Existing Bank

**Goal:** Run the Encoder Substrate Verification Protocol from `WEFT_INNER_PAM_v0_Spec.md` §5 against the existing seed 7 furniture run memory bank. Answer the precondition question: does frozen V-JEPA 2 mean-pool produce embeddings that meet the architectural requirements of Inner PAM v0?

**Outcome:** Pass / Fail / Borderline-with-recalibration verdict, with the empirical distributions reported. This is a one-shot diagnostic on existing artifacts, not an experiment with parallel arms or unconditional progression.

**Scope:** Read-only analysis of the existing memory bank and `frame_annotations.jsonl`. No retraining. No new training data. No new probes.

**Authority:** `CODING_STANDARDS.md` governs discipline. `WEFT_INNER_PAM_v0_Spec.md` §5 is the authoritative protocol — this document operationalises it for the existing bank, it does not redefine the criteria.

---

## 0. Read first

In order:

1. `CODING_STANDARDS.md`
2. `WEFT_INNER_PAM_v0_Spec.md` §5 (the Encoder Substrate Verification Protocol)
3. `HANDOFF.md` — current state
4. `results/stage_0b_furniture/HOUSE_SELECTION.md` — the 5 furniture items and their viewing positions
5. `results/stage_0b_furniture/STAGE_0B_FURNITURE_REPORT.md` — context on what was trained
6. This document

If any are missing or contradict each other, stop and report.

---

## 1. Scope lock — explicit

In scope:
- Loading the existing seed 7 furniture-run memory bank and `frame_annotations.jsonl`.
- Computing the three checks specified in §5 of the spec (cross-instance stability, cross-element distinguishability, combined gap criterion).
- Reporting empirical distributions and pass/fail verdict against the spec's quantitative criteria.
- Recording any threshold recalibration with explicit justification per §5.5.

Out of scope (do NOT do):
- Retraining anything.
- Running the encoder forward on new frames.
- Trying alternative encoders (DINOv2, SigLIP, etc.) in this batch — that's the next decision *after* the verdict, not part of the verdict itself.
- Modifying any document in `/mnt/project/` or the spec.
- Pushing commits.
- Recommending architectural changes. The output of this batch is a verdict on the encoder substrate, not a recommendation about what to do next.

---

## 2. The protocol, operationalised

The spec's §5 specifies three checks. This section translates each into a concrete computation against the existing data.

### 2.1 Setup

Load:
- The seed 7 furniture-run memory bank (path documented in HANDOFF; verify before computing).
- `results/stage_0b_furniture/main/frame_annotations.jsonl` — provides per-frame `viewing_position_id` (1–5 for the five furniture items, 0 for transit) and `loop_index`.

Filter: keep only frames with `viewing_position_id ∈ {1, 2, 3, 4, 5}` (the dwell frames at the five furniture viewing positions). Exclude transit frames. The architecture's claim is about recurring elements; transit frames are not recurring elements in the sense the spec requires.

Document: total frames retained, breakdown by `viewing_position_id`, breakdown by `loop_index`.

### 2.2 Check 1 — Cross-instance stability (§5.1)

For each of the 5 viewing positions, compute the mean cosine similarity between embeddings across loop instances of that position.

Sampling:
- For each viewing_position_id, sample at most 100 dwell frames per loop.
- Across all loops, randomly sample 50 *pairs* of frames where both members of the pair are at the same viewing_position_id but from different `loop_index` values.
- Compute cosine similarity on L2-normalised embeddings.
- Report: mean, std, min, max across the 50 pairs. Then aggregate across all five viewing positions to produce an overall mean for the check.
- Per-viewing-position breakdown is required (some furniture items may stabilise better than others).

Pass criterion (per spec): aggregated mean cosine > 0.75.

### 2.3 Check 2 — Cross-element distinguishability (§5.2)

For each ordered pair of distinct viewing positions (e.g., position 1 vs position 2, position 1 vs position 3, …), compute the mean cosine similarity between embeddings sampled from the two positions.

Sampling:
- For each ordered pair (i, j) where i ≠ j, randomly sample 50 frame pairs where one frame is from viewing_position_id = i and the other from viewing_position_id = j (across any loops).
- Compute cosine similarity on L2-normalised embeddings.
- Report: mean, std, min, max across the 50 pairs per ordered pair.
- Aggregate across all 20 ordered pairs (5 × 4) to produce an overall mean for the check.
- Per-pair breakdown is required (some furniture items may be more confusable than others).

Pass criterion (per spec): aggregated mean cosine < 0.60.

### 2.4 Check 3 — Combined gap criterion (§5.3)

Gap = (Check 1 aggregated mean) − (Check 2 aggregated mean).

Pass criterion (per spec): gap ≥ 0.15.

### 2.5 Reproducibility

Random sampling uses seed 7 (matching the original training run for symmetry). Document the seed and the exact sampling procedure in the report so the analysis can be re-run.

---

## 3. Recalibration discipline

Per spec §5.1, §5.2, and §5.5, the threshold values (0.75, 0.60, 0.15) are starting targets. They may be recalibrated if the empirical distribution warrants, but recalibration must be reported with explicit justification.

**The discipline:**

- If the empirical aggregated values comfortably pass all three checks, no recalibration. Report Pass.
- If the empirical values comfortably fail all three checks, no recalibration. Report Fail.
- If the empirical values are borderline (within ~0.05 of any threshold), or the empirical distributions suggest the starting threshold was poorly chosen for V-JEPA 2's actual cosine-similarity range, recalibration is *considered*. Report:
  - The empirical distribution (histogram or summary statistics).
  - The proposed recalibration with justification (e.g., "the within-instance distribution clusters between 0.65 and 0.78; threshold 0.75 cuts through the distribution unhelpfully; proposing 0.68 to align with the observed cluster centre").
  - Whether the recalibrated thresholds would be passed.
  - The gap criterion held at 0.15 unless explicitly justified otherwise.

**What is forbidden:** recalibrating thresholds downward repeatedly to make the data pass. The first round of recalibration is allowed if justified by the empirical distribution. A second round on the same dataset is a process violation.

If you're not sure whether recalibration is justified, do not recalibrate. Report the empirical values and let the human review decide.

---

## 4. Required artifacts

- `results/encoder_verification/verification_data.json` — raw per-pair cosines for both checks, full distributions.
- `results/encoder_verification/ENCODER_VERIFICATION_REPORT.md` — human-readable report including:
  - Setup: total frames retained per viewing position, sampling seed, sampling procedure.
  - Check 1 results: aggregated mean, per-viewing-position breakdown, distribution summary, pass/fail against starting threshold.
  - Check 2 results: aggregated mean, per-pair breakdown, distribution summary, pass/fail against starting threshold.
  - Check 3 result: gap value, pass/fail against starting threshold.
  - Recalibration decision (if any) with justification.
  - **Verdict:** one of the following, named clearly:
    - **PASS** — encoder meets the protocol on the existing bank without recalibration.
    - **PASS-AFTER-RECALIBRATION** — encoder meets the protocol after a single justified recalibration. Recalibrated thresholds and rationale recorded.
    - **BORDERLINE** — empirical values are close to thresholds but the distribution is messy enough that calling pass or fail is judgment, not data. Stops for human review.
    - **FAIL** — encoder does not meet the protocol and recalibration cannot reasonably bring it within criteria.
  - Honest interpretation: what the verdict means for Weft Inner PAM v0. (No recommendations on next architecture steps — verdict only.)

---

## 5. Operational rules

- Stop after 5 failed tool calls in sequence.
- Operate in away mode: make reasonable decisions, document in HANDOFF, flag for review.
- Every number traces to a file. No remembered numbers.
- One commit at the end with descriptive message.
- Update HANDOFF.md at the end with the verdict and the headline numbers.
- **Do not push commits.** Push hold remains in effect.
- One CC context per working tree.

---

## 6. Unconditional stops

Stop and report if:

- The seed 7 furniture-run memory bank cannot be loaded.
- `frame_annotations.jsonl` is missing or inconsistent with memory bank frame indices.
- Any viewing_position_id ∈ {1, 2, 3, 4, 5} has fewer than 100 frames after filtering (insufficient sample size for the protocol).
- Any computed cosine produces NaN or undefined values that suggest a structural problem (e.g., zero-norm embeddings).
- An implementation decision is needed that this document doesn't cover.
- Five failed tool calls in sequence.
- Disk usage exceeds 80% of quota.
- Any document in `/mnt/project/` or the spec would need modification to proceed.
- The empirical results are so unexpected that the verdict cannot be assigned within the four named outcomes (e.g., negative cosines suggesting embeddings aren't being normalised; aggregated within-instance < cross-element).

Stop format: write `STOP_REPORT.md` at project root with triggering condition, evidence, analysis. Update HANDOFF. Commit. Wait.

---

## 7. What NOT to do

- Do not run alternative encoders in this batch.
- Do not propose architectural changes in the report.
- Do not interpret the verdict as gating the next experiment — the human decides what to do based on the verdict.
- Do not skip the per-viewing-position or per-pair breakdowns. Aggregate-only reports hide failure modes.
- Do not recalibrate more than once on the same dataset.
- Do not modify any project document or the spec.
- Do not push commits.

---

## 8. Done state

End of batch:
- `verification_data.json` and `ENCODER_VERIFICATION_REPORT.md` written.
- One commit: `exp(verification): encoder substrate protocol on seed 7 bank`.
- HANDOFF.md updated with verdict and headline numbers.
- Working tree clean. No commits pushed.

Hard stop. The next action is a human read of the verdict.

---

*End of batch instructions.*
