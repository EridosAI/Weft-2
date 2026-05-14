# HANDOFF — Weft 2

**Project:** Weft Inner PAM (continuous-trajectory associative memory, post-architectural-rethink)
**Repo:** `/mnt/c/Users/Jason/Desktop/Eridos/Weft 2/`
**Status at end of session 7 (2026-05-14, v0 verdict RECORDED):** **V2 — Shape-learning falsified, with coupling-mechanism caveat.** Per the reviewer-chat-locked branch logic, `reading_i_supported` → V2 with caveat. The path-prediction mechanism fit and learned (loss decreased, cluster structure formed on TV/Dresser/Sofa, variance head learned); the failure is specifically that the per-K-step isotropic scalar variance's gradient propagates uniformly across the predictor's state rather than tracking the inputs that produced surprise. Decisive evidence: ord 9 and ord 10 of Bed's close-up are pixel-MD5 identical across loops 30/50/75/100 with cos(loop_30, loop_100) = 1.000000, yet their variance drifts (-0.44, -0.41 nat) sit inside Bed's 2σ uniform-drift band — variance change without input change can only come from gradient updates on other frames. Supplementary Pearson r = -0.0128 (p = 0.97) corroborates. Sharper diagnosis than the spec's existing §11 V2 framing anticipates: the architectural lever v1 should target is uniform gradient propagation in the variance head. Sofa ord-1's +0.056 widening (only positive drift across all 44 (item, ordinal) pairs) carried forward as a v1 disambiguation question — localised architectural signal vs n=1 noise; V2 stands independently because Bed's coupling result doesn't depend on Sofa-ord-1's status. Spec §11 verdict block + §5.8 cross-scope-locality protocol committed at `c757d67`. Working tree clean; push hold in effect.

**Status at end of session 7 (post-per-ordinal-input-variation addendum, superseded by verdict-recording above):** Ninth STOP, **verdict-branch pointer = `reading_i_supported`**. Reviewer-chat-authorised disambiguator on existing data: at Bed's close-up ords 9 and 10 — established as pixel-MD5 identical across Stage B loops 50/75/100 — the loop-30 frames are ALSO pixel-MD5 identical to those, and `cos(loop_30 embedding, loop_100 embedding) = 1.000000` exactly. Yet variance drift at those ordinals = **-0.4356** and **-0.4059** nat, both inside Bed's 2σ uniform-drift band [-0.488, -0.358]. Drift-under-zero-input-variation observed on perfectly invariant Bed poses. Supplementary Pearson r = -0.0128 (p = 0.97), far outside any non-zero correlation. The mechanism (i) interpretation — cross-item coupling shifting Bed's variance estimate via global gradient flow regardless of Bed's per-pose inputs — is supported. The mechanism (ii) interpretation — predictor responding to input variation — is incompatible with these specific ordinals' data (zero input variation but non-zero drift). All three sanity checks PASS (frame indices match `variance_by_ordinal.json`; embedding L2 norms in tolerance; ord 9/10 MD5s recomputed from PNGs match `within_loop_invariance.json`). **CC does not record the verdict; the pointer is a mechanical summary for the fresh reviewer chat that closes v0.** Working tree clean; push hold in effect.

**Status at end of session 7 (post-disaggregating-diagnostics, superseded by per-ordinal-input addendum above):** Ninth STOP, **v0 verdict still deferred; verdict-recording goes to a fresh reviewer chat once these diagnostics are in.** Reviewer authorised two disaggregating diagnostics on existing data after the cross-loop substrate-determinism finding: (1) variance trajectory by close-up ordinal across items × loops; (2) within-loop input-invariance check on Bed in Stage B. **Headline new finding:** Bed's cross-loop drift loop 30 → 100 is **uniform across all 11 close-up ordinals** (std 0.034, range 0.129 nat) while TV, Dresser, Sofa show **non-uniform per-ordinal drifts** (std 0.12–0.19, range 0.30–0.56). Bed has LARGER within-loop pose-driven view variation than TV (cos_mean 0.74 vs 0.85), so input-magnitude alone doesn't explain why Bed's drift uniform while TV's is ordinal-monotonic. The data is mixed across items: TV/Dresser/Sofa's behaviour supports input-driven per-pose response (consistent with reviewer's account ii); Bed's uniform drift is consistent with cross-item coupling (consistent with account i). **No autonomous verdict.** Loop-170 safety check NOT run per directive. Working tree clean; push hold in effect.

**Status at end of session 7 (post-substrate-determinism addendum, superseded by disaggregating-diagnostics addendum above):** Ninth STOP, **verdict deferred pending substrate-determinism diagnostic findings now in.** Reviewer surfaced a question after the loop-100 G2.T2 verdict: §8.4 reported control gaps (Bed 0.0045, TV 0.0068) that are inconsistent with the deterministic-substrate assumption (expected gap = 0). Cross-loop invariance diagnostic on existing data (commit `91a66fb`): **Bed Stage A close-up frames are pixel-MD5 identical across loops** (Case A); **Television Stage A close-up frames are NOT pixel-identical** but DINOv2 cosines are 0.99983–1.0 (Case B — per-call rendering non-determinism specific to TV, visually negligible). Poses match exactly. **Decisive new finding:** Bed's Stage A within-cosine is exactly 1.000000 (consistent with bit-identicity), but the §8.4 within-Stage-B cosine is 0.9954, so the entire Bed gap comes from Stage B — `RandomizeMaterials(inRoomTypes=["LivingRoom"])` produces visual variation in Bedroom item frames despite Bedroom not being in API scope. **Cross-room visual leakage from the perturbation API itself** — a substrate-level finding the §8.4 framing didn't surface. Reframes the loop-100 V2 reading: control items DO have real Stage B input variation (small but nonzero); Bed's predictor-narrowing despite that variation is still anomalous, but not the same as "predictor doesn't see perturbation". **No autonomous verdict.** Working tree clean; push hold in effect.

**Status at end of session 7 (initial, pre-diagnostic; superseded by addendum above):** Ninth STOP. G2.T2 restructured (instr §8.7a) per session-7 authorisation: three-part criterion at loop 100 — (a) trajectory direction (monotonic non-decreasing, ≤ 3 dips), (b) trajectory shape (descriptive), (c) differential perturbed/control ≥ 2.0. Original 0.5 absolute threshold dropped per the same pattern as §8.4's 0.05 floor → Wilcoxon Reading C restructuring in session 6 (research_operations §15 corollary added). Trainer extended-mode added: on resume, loads prior diagnostic JSON, skips the session-6 boundary auto-trip, and re-evaluates G2.T1/T3 at every checkpoint over the extended window. Training resumed from `ckpt_12000.pt` with `--max_loops 100`; ran 286.3s, 24,360 gradient steps. **G2.T1 and G2.T3 clean across the entire loops-36..100 window** (no in-flight trip at the 4 new checkpoints). **G2.T2 restructured FAILS all three gated criteria:** (a) 34 dips vs ≤ 3 allowed; (b) shape "mixed" — widened by loop 50 then partially retracted; (c) ratio 0.639 vs ≥ 2.0 required (controls drift MORE than perturbed widen — Bed log_var actually NARROWED by 0.21 over loops 30→100). This is the documented session-7 STOP-for-review condition. Per session-7 authorisation language, the restructured-G2.T2 failure puts option (iv) (reframe verdict toward V2) in the reviewer's frame as a live decision based on loop-100 evidence. **No autonomous decision on the verdict.** Working tree clean; push hold in effect.

**Status at end of session 6 (2026-05-14):** Eighth STOP. §8.4 gate restructured and PASSED on the existing collected data (ratio = 2.107 vs threshold 2.0 with clean controls = {Bed, Television}; Wilcoxon corrected_p < 1e-300 on both perturbed items vs 0.001 threshold). Trainer extended with `--max_loops` and `--resume_from`; resume smoke-tested on real data. Phase 2 training launched with `--max_loops 35` and ran cleanly to step 12,960 (loop 36 boundary). **§8.7a G2.T2 TRIPPED**: perturbed-item log_var widening = +0.022 between loop 30 and loop 35, vs required ≥ 0.5 SCAFFOLDING threshold. Trainer wrote `transition_diagnostic_TRIPPED.txt`, exit code 3. G2.T1 (loss-spike) and G2.T3 (control-drift) both clean. This is the documented session-6 STOP-for-review condition — training STOPPED, reviewer judgement requested. Working tree clean; push hold in effect.

**Status at end of session 5 (2026-05-14):** Seventh-STOP §8.4 verification reviewed; reviewer authorised the next session's restructuring and (conditional) Phase 2 training launch but stopped THIS session for context-budget reasons (`CODING_STANDARDS.md` §9.6 — compacted context can produce fabricated numbers during statistical-test computation). Phase 2 collected data on the corrected substrate is on disk (65k frames + annotations + materials metadata; `encode_report.json` committed at `6d6e58d`; embeddings.npy not yet written — pending the restructured §8.4 gate). Phase 2 training NOT launched. Next session has authorised tasks captured in detail in the "Next immediate action" block below. Working tree clean; push hold in effect.

---

## What this repo is

This is a fresh repository for the Weft project, built around the architecture articulated in `WEFT_INNER_PAM_v0_Spec.md`. The previous repo at `/mnt/c/Users/Jason/Desktop/Eridos/Weft/` contains four iterations of negative results that established the previous architecture (next-frame prediction with cosine retrieval) was building the wrong thing. The new architecture is path-prediction with Gaussian negative-log-likelihood loss, learning trajectory shapes through repetition. See the spec for full claims.

The previous repo stays in place as historical record. This repo does not edit it or share state with it. The previous repo is referenced once at runtime via Python sys.path for the AI2-THOR explorer's `route.json` (the seed-7 furniture metadata, which is data, not code); the explorer + env wrapper themselves were copied into `src/env/` per CODING_STANDARDS §2.3 one-source-of-truth.

---

## What's been done (end of session 4)

- **v0 code scaffolding:** predictor, online trainer, memory bank, recall mixer, eval probes/metrics/controls, DINOv2 encoder wrapper, continuous-motion explorer + env wrapper, Phase 1 train/shuffle scripts, run_eval (with `--developmental` flag), gate report analysis. 21 unit tests passing.
- **Encoder substrate verification:** DINOv2 ViT-L/14 CLS PASS on the §5 protocol (Check 1 = 0.9260, Check 2 = 0.4422, gap = 0.4838). Approved by reviewer 2026-05-12 as the v0 encoder.
- **DINOv2 full-stream encoding:** `data/dinov2_embeddings/embeddings.npy` is 100,000 × 1024 fp32, all rows L2-normalised (consistency cosine = 1.000000 vs the verified dwell-only archive on 50 sampled frames).
- **Phase 1 main + shuffle (substrate-degenerate baseline):** trained, evaluated, gates reported. G1.1 PASS, G1.2 PASS, G1.5 PASS @ scaffolding; G1.3 FAIL @ absolute scaffolding (treated as substrate artefact); G1.4 FAIL @ k=8 (treated as rank-512-architecture-limit candidate diagnosis; substrate-degenerate baseline anyway). Per session-3 reviewer directive: Phase 1 substrate declared substrate-degenerate; v0 evidence base restarts at Phase 2 on the new continuous-motion substrate.
- **Continuous-motion substrate implemented + 5-loop calibrated.** Trajectory: per-item "close-up" segment (2 m perpendicular pass through viewing_position, heading locked at viewing_heading), NavMesh-densified transit between items. 316 frames/loop empirical. Within-loop motion-continuity PASSES the DINOv2 spot-check (0 bit-identical consecutive pairs in 255 close_up→close_up + 1,275 transit→transit). Cross-loop apex bit-identicity FAILS on 4 of 5 items (item 5 is the lone non-deterministic-render exception); variation strategy needed to break across-loop pose-determinism floor.
- **All spec / instructions / research_operations docs updated** to lock the substrate change in. §0 Python-version corrections. §1.3 substrate revision. §1.5 "dwell as pause is not part of the architecture." §4.6 cadence recomputed for the 316-frame loop. §8.3 / §9.3 frame budget tables. research_operations_v1.md §15 substrate-assumption drift check added.

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

## v0 commit session close (2026-05-14)

v0 is closed. Three artifact commits landed:

- `c757d67` — `docs(spec): §5.8 cross-scope perturbation locality + §11 V0 verdict recorded`
- `05d1d30` — `docs(handoff): v0 verdict recorded — V2 with coupling-mechanism caveat`
- `7fd2cae` — `docs(closing): WEFT_INNER_PAM_v0_CLOSING.md — v0-closing companion to spec/instructions/handoff`

Working tree clean. No untracked files at repo root. Push hold remains in effect.

**Divergences from the commit-mechanics instruction surfaced for the record (not unwound per the instruction's no-retry rule):** (1) commit order was spec → HANDOFF → closing rather than the prescribed spec → closing → HANDOFF; (2) commit messages diverged in wording from the prescribed templates, content equivalent. The three artifacts each got their own commit per §2.5–2.6 of CODING_STANDARDS; no bundling. The closing document at `7fd2cae` is 22 KB (the instruction expected 11–13 KB); structural checks (10 numbered sections + terminator line) pass.

v1 design begins in a separate chat with its own spec discipline.

---

## v0 verdict recorded (2026-05-14)

**V2 — Shape-learning falsified, with coupling-mechanism caveat.**

Per the reviewer-chat-locked branch logic: `reading_i_supported` → V2 with coupling-mechanism caveat. Spec §11 verdict block updated at commit `c757d67` (alongside the new §5.8 cross-scope-perturbation-locality protocol).

### Primary discriminator

Ord 9 and ord 10 of Bed's close-up are **pixel-MD5 identical across all four sampled loops** {30, 50, 75, 100}; **cos(loop_30, loop_100) = 1.000000** at both ordinals; variance drifts (**−0.4356, −0.4059**) both inside Bed's 11-ordinal **2σ band [−0.4881, −0.3578]**. Supplementary **Pearson r = −0.0128, p = 0.97** — across Bed's 11 ordinals, per-ordinal cross-loop input variation has no relationship to per-ordinal variance drift.

### Architectural reading

The predictor's variance moved by ~0.42 nat on frames whose inputs did not change at all over 70 Stage B loops. The only mechanism in the architecture that can produce variance change without input change is gradient updates from training on other frames. The predictor's variance representation is therefore coupled across the training stream rather than isolated per (item, pose). Spec §2.2's prediction — that variance responds to per-item surprise — does not survive on this substrate with this loss formulation: the variance head is operating (loss decreased, structure formed elsewhere), but the gradient updates it produces propagate uniformly across the predictor's state regardless of which inputs were the source of surprise.

This is V2 (architectural claim not supported), with a more specific failure mode than the spec's existing §11 V2 framing: the path-prediction mechanism fits and learns; the failure is specifically how the per-K-step isotropic scalar variance's gradient propagates across the predictor's state. This is a sharper diagnosis than V2's spec language anticipates, and the specific mechanism identified (**uniform gradient propagation in the variance head**) is the architectural lever v1 should target.

### Localised-widening note (Sofa ord-1)

From [variance_by_ordinal.json](results/inner_pam_v0/phase2_main/variance_by_ordinal.json): Sofa ord-1 drift loop 30 → 100 is **+0.056** — the only positive (widening) drift across all (item, ordinal) pairs in the disaggregated data. Bed / TV / Dresser have monotonic-narrowing trajectories at every ordinal; Sofa narrows at ords 0 and 2–10 but widens at ord 1.

Two readings:

- **(a) Localised architectural signal.** The variance mechanism may operate at finer (item, ordinal) granularity than the aggregate G2.T2 measured.
- **(b) Noise at n=1.** Below §3.6 small-sample threshold.

Distinguishing (a) from (b) requires v1 evaluation at per-(item, ordinal) granularity from design time, not aggregate-then-disaggregate; aggregate gates will continue to average over locally-specific signals. V2 stands either way (Bed coupling result is independent of Sofa ord-1's status). Carried forward as a v1 disambiguation question in the closing doc.

### What v0 ships

- Spec changes committed at `c757d67`: §5.8 (cross-scope perturbation locality at the rendered-/observation-frame level — broadened from §5.6's per-item-stability framing to cover rendering / physics / audio / policy substrate-state coupling) and §11 V0 verdict (the V2-with-caveat record above).
- All Phase 2 collected data + trained checkpoints + diagnostic outputs preserved on disk (per the prior addendum's enumeration; no artefacts were removed for the verdict-recording step).
- Working tree clean. Push hold remains in effect.

### What is deferred to v1 / the closing doc

- **WEFT_INNER_PAM_v0_CLOSING.md**: not yet written. Referenced from spec §11 for the Sofa-ord-1 disambiguation question and any longer-form v0-closing narrative.
- **v1 architectural lever**: variance head's uniform gradient propagation. Concrete v1 design choices (per-(item, ordinal) variance heads? coupled-but-shaped variance? alternative loss formulations that isolate per-item surprise gradients?) are scoped in the closing doc / v1 design chat, not here.
- **Substrate question**: whether the coupling result generalises beyond AI2-THOR + DINOv2-frozen. v1 design decision.

### Working-tree state at end of v0

Working tree clean. Push hold in effect. Commits since the prior addendum (`8a48336`):
- `c757d67` — `docs(spec): §5.8 cross-scope perturbation locality + §11 V0 verdict recorded`.
- (this commit) — HANDOFF verdict-recording entry.

No running jobs. GPU clear. Disk: ~230 GB free.

---

### (Earlier — per-ordinal-input-variation diagnostic; verdict now recorded above)

The reviewer-chat-authorised per-ordinal cross-loop input variation diagnostic on Bed produced a decisive result on the primary discriminator: ords 9 and 10 have literally zero cross-loop input variation across the full loop-30-to-100 span (pixel-MD5 identical across all four sampled loops; embedding `cos(loop_30, loop_100) = 1.000000` exactly), yet their variance drifts (-0.4356 and -0.4059 nat) fall inside Bed's 2σ uniform-drift band. Drift-under-zero-input-variation is architecturally impossible under reading (ii) and expected under reading (i). The supplementary Pearson r = -0.0128 (p = 0.97) corroborates: essentially no relationship between per-ordinal input variation and per-ordinal drift across the 11 ordinals.

### Per-ordinal cross-loop input variation diagnostic (verdict-disambiguator)

**Script:** [scripts/run_phase2_per_ordinal_cross_loop_input.py](scripts/run_phase2_per_ordinal_cross_loop_input.py).
**Report:** [results/inner_pam_v0/phase2_main/per_ordinal_cross_loop_input.json](results/inner_pam_v0/phase2_main/per_ordinal_cross_loop_input.json).
**Method:** for Bed at the same four sample loops used by `variance_by_ordinal.json` ({30, 50, 75, 100}), resolve the 11 close-up frame indices per loop (cross-checked against `variance_by_ordinal.json` target_frame_idx — all 44 indices match exactly), pull pairwise cosines across the 4 loops per ordinal (6 loop pairs × 11 ordinals = 66 cosines), recompute pixel-MD5 at loop 30 from PNGs, cross-reference ord-9/ord-10 MD5s for loops 50/75/100 against `within_loop_invariance.json` (recomputed-from-PNGs match).

**Sanity checks (all PASS):**

1. Frame index resolution matches `variance_by_ordinal.json`'s `target_frame_idx` at all 4 loops × 11 ordinals.
2. All 44 embedding L2 norms in [0.9999999, 1.0000001] (well inside the 1e-5 tolerance).
3. Ord 9/10 pixel-MD5 at loops 50/75/100, recomputed from PNGs in this run, match the values stored in `within_loop_invariance.json`.

**Primary discriminator (ords 9 and 10):**

| ord | MD5 identical across 4 loops? | cos(loop_30, loop_100) | variance drift loop 30 → 100 | within 2σ band? |
|---:|:---:|---:|---:|:---:|
| 9 | **TRUE** | **1.000000** | -0.4356 | TRUE |
| 10 | **TRUE** | **1.000000** | -0.4059 | TRUE |

Bed's drift distribution across all 11 ordinals: mean = -0.4229, std = 0.0326, 2σ band = [-0.4881, -0.3578]. Both ord-9 and ord-10 drifts are inside the band.

**The frame at Bed close-up ord 9 in loop 30 is bit-identical (pixel-MD5 and DINOv2-embedding) to the same frame in loop 100.** Same for ord 10. There is no input variation to attribute drift to at these ordinals; the predictor's variance estimate on those frames moved by ~0.42 nat over 70 Stage B loops with zero changes in the input. The only mechanism that can produce variance change without input change is gradient updates from training on OTHER frames — cross-item coupling. Reading (i) is supported; reading (ii) is incompatible with the data at these specific ordinals.

**Supplementary Pearson r across the 11 ordinals:**

`x_per_ord = 1 − mean(pairwise cosines across the 4 loops)`, per ordinal.
`y_per_ord = |drift|`, per ordinal.

`r = -0.0128, p = 0.9702`. r is far outside the non-load-bearing band [0.3, 0.7] — specifically, well below 0.3, with a sign that's effectively zero. No relationship between per-ordinal input variation magnitude and per-ordinal drift magnitude. This corroborates the primary discriminator: variance drift is not tracking input variation across Bed's ordinals.

**Verdict-branch logic evaluation:**

- **`reading_i_supported`** conditions (all required):
  - Ord 9 and ord 10 MD5 identical across 4 loops: **TRUE**
  - Ord 9 and ord 10 cos(loop_30, loop_100) ≥ 0.9999: **TRUE** (both 1.000000)
  - Ord 9 and ord 10 drift within mean ± 2σ band: **TRUE**
  → overall TRUE.

- **`reading_ii_supported`** conditions (all required):
  - Ord 9 or ord 10 cos < 0.999: **FALSE** (both are 1.000000)
  - Pearson r ≥ 0.7 with p < 0.05: **FALSE** (r = -0.0128, p = 0.97)
  → overall FALSE.

**Verdict branch pointer = `reading_i_supported`.**

### Notes for the fresh reviewer chat

- The embedding-path entry in the diagnostic spec (`data/dinov2_embeddings/embeddings.npy`) appears to be a typo for `data/phase2_embeddings/embeddings.npy`. The former is the 100k Phase-1 substrate-degenerate baseline whose frame indices don't align with `phase2_annotations.jsonl` (65k Phase-2 frames). All prior Phase-2 diagnostics (variance_by_ordinal, within_loop_invariance, encode_report) used the Phase-2 file; the per-ordinal-input diagnostic does too. Flagged in the output JSON's `notes` field and again here so the reviewer chat can confirm the path interpretation. If the reviewer intended a Phase-1 comparison, that would be a different diagnostic that this script does not implement.
- All thresholds (cos ≥ 0.9999, Pearson [0.3, 0.7], 2σ band) are per the reviewer-chat spec and were not adjusted.
- `verdict_branch_pointer` is a mechanical summary computed from the spec's logic; CC has not edited the spec or recorded a verdict. The pointer is in the output JSON for the reviewer chat to pick up.

### Stop conditions / scope-of-this-session

- No v0 verdict recorded.
- No loop-170 safety check.
- No further training, encoding, or AI2-THOR.
- Diagnostic completed in 5.3 seconds (well under the 5-minute STOP threshold).
- No SCAFFOLDING thresholds adjusted.
- Working tree clean.
- Push hold remains in effect.

### Working-tree state at end of this addendum

Working tree clean. Commits since `981b09d` (prior session-7 HANDOFF):
- `64b3e36` — `exp(phase2): per-ordinal cross-loop input variation diagnostic on Bed (verdict-disambiguation)`.
- (this commit) — HANDOFF entry with discriminator results, Pearson, branch pointer.

No running jobs. GPU clear. Disk: 231 GB free (88% used).

---

### (Earlier — initial disaggregating-diagnostics addendum; superseded by the per-ordinal-input addendum above)

**Ninth STOP — v0 verdict still deferred. Two disaggregating diagnostics now in (commit `b49e80f`); verdict-recording goes to a fresh reviewer chat with this data.** The reviewer-authorised disaggregating diagnostics (variance-by-ordinal, within-loop input invariance) ran on existing data and produced a mixed result: TV/Dresser/Sofa show input-driven per-ordinal differentiation (consistent with the architecture's variance response operating), while Bed shows uniform-across-ordinals drift inconsistent with input-driven variance. The data does NOT cleanly select between the V2-stands account (i) and the V2-doesn't-stand account (ii). Loop-170 safety check NOT run per directive.

### Disaggregating diagnostics (session-7 reviewer-directed)

**Method 1: variance trajectory by close-up ordinal** ([scripts/run_phase2_variance_by_ordinal.py](scripts/run_phase2_variance_by_ordinal.py), report at [results/inner_pam_v0/phase2_main/variance_by_ordinal.json](results/inner_pam_v0/phase2_main/variance_by_ordinal.json)). For each item ∈ {Bed=1, Television=5, Dresser=3, Sofa=4} and each loop ∈ {30, 50, 75, 100}, loaded the predictor from the checkpoint nearest after the end of that loop (`ckpt_12000`, `ckpt_20000`, `ckpt_30000`, `ckpt_36360`) and evaluated the predictor on the W=16-frame window targeting each of the 11 close-up ordinal positions. Captured mean(log_var across K=16 prediction steps) per (item, loop, ordinal).

**Cross-loop drift loop 30 → 100, per ordinal** (negative = log_var narrowed, predictor more confident):

| item | ord0 | ord1 | ord2 | ord3 | ord4 | ord5 | ord6 | ord7 | ord8 | ord9 | ord10 | std | range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bed | -0.430 | -0.494 | -0.447 | -0.389 | -0.411 | -0.421 | -0.406 | -0.447 | -0.365 | -0.436 | -0.406 | **0.034** | 0.129 |
| Television | -0.198 | -0.311 | -0.262 | -0.365 | -0.360 | -0.342 | -0.489 | -0.585 | -0.637 | -0.729 | -0.755 | **0.193** | 0.557 |
| Dresser | -0.317 | -0.368 | -0.353 | -0.318 | -0.301 | -0.363 | -0.582 | -0.580 | -0.606 | -0.564 | -0.504 | **0.124** | 0.305 |
| Sofa | -0.240 | +0.056 | -0.067 | -0.067 | -0.086 | -0.250 | -0.192 | -0.232 | -0.441 | -0.295 | -0.345 | **0.144** | 0.497 |

**Reading per reviewer framing:** if cross-loop drift is driven by per-ordinal input variation, drifts should differ by ordinal. If driven by cross-item coupling independent of input, drifts should be uniform. **Bed's drifts are uniform** (std 0.034 — 3× smaller than the next item's 0.124). **TV, Dresser, Sofa show non-uniform per-ordinal drift**, with TV showing a striking monotonic ordinal-dependent trend (ord 0 drift -0.20, ord 10 drift -0.76). Sofa even has a positive (widening) drift at ord 1 (+0.056) alongside narrowing at later ordinals.

**Method 2: within-loop input invariance** ([scripts/run_phase2_within_loop_invariance.py](scripts/run_phase2_within_loop_invariance.py), report at [results/inner_pam_v0/phase2_main/within_loop_invariance.json](results/inner_pam_v0/phase2_main/within_loop_invariance.json)). For Bed at each Stage B loop in {50, 75, 100}, computed pairwise pixel-MD5 hashes and pairwise DINOv2 cosines across the 11 close-up ordinals within each single loop.

Bed within-loop: 11 unique pixel-MD5 hashes per loop (no two ordinals identical). Pairwise cosine min ~0.32, max ~0.98, mean ~0.74. **The reviewer's "Expected: bit-identical or very close (cosine ~1.0)" hypothesis is firmly refuted** — the close-up sweep is a 2 m perpendicular pass that produces substantial pose-driven view changes. Adjacent ordinals (cos ~0.98) are similar; far-apart ordinals (cos ~0.32) look quite different.

Cross-check across items (Stage B, 11 close-up ordinals within one loop, averaged over loops 50/75/100):

| item | within-loop cos_min | within-loop cos_mean |
|---|---:|---:|
| Bed | 0.32 | 0.74 |
| Television | 0.60 | 0.85 |
| Dresser | 0.31 | 0.68 |
| Sofa | 0.23 | 0.71 |

**Bed has LARGER pose-driven within-loop view variation than TV** (cos_mean 0.74 vs 0.85), yet Bed's per-ordinal drift across loops is more uniform than TV's. Input-variation magnitude alone does not explain the uniformity-of-drift differential between items.

### Two interpretive accounts of the loop-100 G2.T2 verdict, with the new data anchoring each

**Account (i) — V2 stands. The predictor failed §2.2's per-item perturbation response prediction.**

- Bed's cross-loop drift (loop 30 → 100, mean -0.42 across ordinals) is essentially uniform across the 11 ordinals (std 0.034). The 11 ordinals correspond to 11 *different* pose-driven views of Bed within each loop, with cos min 0.32 — substantial input variation per pose. If the predictor were learning a per-pose variance representation, each ordinal would drift by a different amount over 70 loops; instead they all drift by ~-0.42 within ±0.06.
- This pattern is consistent with cross-item coupling: gradient updates from training on Dresser+Sofa+TV close-ups (and from transit frames between items) propagate into Bed's variance estimate as a global shift, without engaging Bed's per-pose structure. The predictor's per-K-step scalar isotropic variance (spec §3.3) gives every Bed pose the same variance update, regardless of which specific Bed pose is the target.
- Bed has the smallest Stage B input variation among the four items (cross-loop within-Stage-B cosine 0.9954) and the largest uniformity-of-drift. The drift magnitude (-0.42) is ~85× larger than the cross-loop input variation magnitude (1 − 0.9954 = 0.0046 cosine). If drift were proportional to input variation, Bed should have the smallest drift; instead it has comparable drift to the perturbed items (Dresser -0.44, Sofa -0.20). The drift on Bed cannot be explained by Bed's input variation.
- The architectural prediction "variance widens specifically on perturbed items in response to per-item surprise" requires per-item variance updates. The data shows per-item updates ARE happening on TV/Dresser/Sofa (non-uniform per-ordinal drifts there) but NOT on Bed (uniform drift). The architecture is partly functional, partly broken — with the breakage concentrated on the cleanest-input item, where cross-item coupling dominates the observable signal.
- V2 stands under this reading. The §11 V2 evidence package would document: "the architecture's confidence-graded variance mechanism does not cleanly isolate per-item updates; Bed-specific evidence of cross-item coupling at this signal magnitude."

**Account (ii) — V2 doesn't stand. The predictor responded normally to inputs that had per-loop variation on all items.**

- The cross-room visual leakage finding (commit `91a66fb`) established that Stage B's `RandomizeMaterials(inRoomTypes=["LivingRoom"])` produces visual variation in Bedroom item frames despite Bedroom not being in API scope. So the predictor sees per-loop variation on ALL items in Stage B, including Bed.
- TV/Dresser/Sofa's non-uniform per-ordinal drifts are consistent with the architecture learning per-pose variance responses driven by per-pose input variation. The architectural mechanism is operating on those items.
- Bed's uniform-across-ordinals drift could reflect that Bed's per-pose inputs all see *the same kind* of Stage B variation (the LivingRoom-retexture cross-room leakage propagates uniformly across Bed's poses), and the predictor responds with a uniform variance shift. TV's monotonic ordinal-dependent drift could reflect TV's specific view geometry producing different magnitudes of variation per pose.
- Under this reading, the G2.T2 restructured-gate's "controls drift more than perturbed widen" finding is just an artifact of the controls having more accumulated input variation than the gate's framing assumed. Bed's -0.42 drift is the predictor narrowing as it absorbs Bed-specific input variation; not anomalous, just a side-effect of input variation the original gate framing didn't account for.
- The architectural claim "shape representations widen but survive under visual variation" (spec §1) holds: the predictor's variance estimates are tracking real input variation per item; the drift magnitudes happen to put controls and perturbed items in the same rough range, but the *pattern* of per-pose differentiation is operating correctly.
- V2 doesn't stand under this reading. The data is consistent with architectural success under coupled inputs (where coupling is at the renderer level, not the predictor's).

### Where the data is decisive and where it isn't

Decisive:
- The close-up sweep is NOT input-invariant within a loop; it is a 2 m pass with cos min 0.23–0.32 across items.
- Bed's cross-loop drift is uniform across ordinals (std 0.034); TV/Dresser/Sofa's is non-uniform (std 0.124–0.193).
- The non-uniformity differential between Bed and the other three items is not explainable by within-loop input variation magnitude (Bed has more within-loop variation than TV).

Indeterminate:
- Whether Bed's uniform drift reflects (i) cross-item coupling (architecture's per-item isolation failing on Bed specifically) or (ii) Bed's specific kind of cross-loop variation (LivingRoom-retexture cross-room leakage) being uniform across Bed's poses while TV's is not. Both readings are consistent with the present data.
- The interpretive choice between (i) and (ii) depends on: do we assert that Bed's poses should receive *differentiated* per-pose updates because the input variation across loops differs by pose, and verify that empirically; OR do we assert that uniform-across-poses input variation produces uniform-across-poses drift, and verify *that*?

A third decisive diagnostic would be: for each Bed pose (ordinal), compute the cross-loop input variation magnitude at that ordinal (e.g. for ord 0, compute cosine across loops 30/50/75/100 at Bed's frame_3-of-close-up, vs the same at ord 5, vs ord 10). If per-ordinal cross-loop input variation is itself uniform, account (ii) is supported. If it's non-uniform yet drift is uniform, account (i) is strongly supported. This is a cheap follow-up (~minute on existing data) — but per the directive ("After both diagnostics complete, write a HANDOFF interpretation entry presenting the data and the two interpretive accounts. Do not commit to a v0 verdict yet"), it goes to a fresh reviewer chat, not autonomous action.

### What is preserved on disk for the verdict-recording session

- Encoder embeddings + §8.4 report: `data/phase2_embeddings/embeddings.npy` + `encode_report.json`.
- All Phase 2 frames (65k PNGs) at `data/phase2_frames/`.
- All Phase 2 annotations at `data/phase2_annotations.jsonl`.
- Phase 2 training checkpoints at `results/inner_pam_v0/phase2_main/ckpt_*.pt` + `ckpt_*/` bank dirs (steps 1000, 2000, 4000, 6500, 10000, 12000, 15000, 20000, 30000, 36360).
- Transition diagnostic: `results/inner_pam_v0/phase2_main/transition_diagnostic.json` (per-loop log_var for loops 0..101).
- Restructured G2.T2 report: `transition_g2t2_restructured.json`.
- Cross-loop invariance: `cross_loop_invariance_check.json`.
- Variance-by-ordinal: `variance_by_ordinal.json`.
- Within-loop invariance: `within_loop_invariance.json`.
- Session-6 trip marker preserved as `transition_diagnostic_TRIPPED.session6.txt`.

### Stop conditions, scope-of-this-session

Per directive: no v0 verdict recorded; no loop-170 safety check run; no further training. The verdict-recording decision and the v0-closing-document work go to a fresh reviewer chat.

Working-tree state at end of this addendum: clean. Push hold in effect. Commits since session-7's HANDOFF:
- `91a66fb` — cross-loop invariance diagnostic (Stage A Case A/B per item; cross-room visual leakage finding).
- `f811146` — HANDOFF addendum on substrate-determinism diagnostic.
- `b49e80f` — variance-by-ordinal + within-loop invariance diagnostics.
- (this commit) — HANDOFF interpretation entry with both accounts.

No running jobs. GPU clear. Disk: ~100 GB free.

---

### (Earlier — initial post-substrate-determinism addendum; superseded by the disaggregating-diagnostics addendum above)

**Ninth STOP — v0 verdict deferred pending reviewer interpretation of the post-verdict substrate-determinism diagnostic.** The §8.7a G2.T2 restructured-gate fails at loop 100 (recorded below), but a reviewer-surfaced question about substrate determinism produced a follow-up diagnostic whose findings change how the verdict is read. The verdict has NOT been recorded. The loop-170 safety check has NOT been run.

### Cross-loop invariance diagnostic (post-session-7, reviewer-directed) — findings

**Question:** §8.4 reported control-item Stage A vs Stage B gaps of 0.0045 (Bed) and 0.0068 (Television). On a corrected-substrate, deterministic-renderer assumption, Stage A close-up control-item frames should be bit-identical across loops (same seed, same trajectory, same pose, no perturbation in scope). Expected gap on truly invariant inputs to a frozen deterministic encoder = 0, not 0.0045.

**Method.** From existing `data/phase2_frames/`, sampled Stage A loops {1, 5, 10, 15, 20, 25, 28}; for each (item, loop), collected close-up frames (11 per loop, identical poses confirmed: max XZ displacement across loops = 0.0, max rotation deviation = 0.0°); at each within-close-up ordinal index, computed pairwise pixel-MD5 hashes (on decoded RGB arrays) and pairwise DINOv2 cosines (from the saved `embeddings.npy`). Same for the apex frame. Report at [results/inner_pam_v0/phase2_main/cross_loop_invariance_check.json](results/inner_pam_v0/phase2_main/cross_loop_invariance_check.json).

**Bed (vp=1) — Case A (pixel-deterministic in Stage A).**
- All 11 ordinals × 7 loops: **pixel-MD5 ALL identical** at every ordinal. `n_unique_pixel_md5 = 1` everywhere.
- DINOv2 cosine off-diagonal range [0.99999994, 1.00000012] — numerical noise around 1.0.
- Apex: pixel-identical across all 7 loops; cosine 0.99999996.
- Substrate determinism holds for Bed at this pose. Stage A is, as designed, bit-identical.

**Television (vp=5) — Case B (per-call non-determinism, visually negligible).**
- All 11 ordinals × 7 loops: pixel-MD5 NOT identical. `n_ordinals_with_all_pixel_md5_identical = 0` (zero out of 11 ordinals have all-identical hashes across the 7 loops).
- DINOv2 cosine off-diagonal range [0.99983, 1.00000], mean of means 0.99999.
- Apex: 2 unique pixel-MD5 hashes across 7 loops; cosines 0.99998–1.0.
- TV-specific renderer non-determinism — most likely shader cache state, animation frame counters, anti-aliasing sample order, or similar non-deterministic GPU state that varies per `controller.step` call even with identical pose. Visually negligible (cosine > 0.9998) but pixel-distinct.

### Where the §8.4 control gaps actually come from

The §8.4 report's per-stage cosines (read from `data/phase2_embeddings/encode_report.json`):

| item | within_A | within_B | cross | gap |
|---|---:|---:|---:|---:|
| Bed | **1.000000** | 0.995422 | 0.993194 | 0.0045 |
| Television | 0.999987 | 0.989502 | 0.988223 | 0.0065 |
| DiningTable | 0.999969 | 0.980352 | 0.976574 | 0.0136 |

**Bed's within_A is exactly 1.000000** (consistent with the bit-identical Stage A frames confirmed above). The entire 0.0045 gap comes from **within_B = 0.9954** — Stage B has within-stage visual variation that Stage A does not. The Stage A → Stage B drift then accounts for the additional 0.0023 from `within_avg - cross`.

**The only mechanism that can produce within-Stage-B variation on Bed is the per-loop `RandomizeMaterials(inRoomTypes=["LivingRoom"])` call producing visual side-effects in Bedroom frames.** Bedroom is not in the API's room scope, but the rendering after the call differs from the rendering before. **Cross-room visual leakage from the perturbation API itself** — a substrate-level finding the §8.4 framing didn't surface and that re-frames the loop-100 verdict.

For TV: within_A = 0.999987 (consistent with renderer non-determinism observed in the Stage A diagnostic). within_B = 0.9895 — the TV-specific renderer non-determinism contributes a small floor, plus the same cross-room leakage Bed sees, plus possibly more accumulated non-determinism over 150 Stage B loops vs 31 Stage A loops.

For DT (skipped in the diagnostic per directive, but the §8.4 numbers confirm the noisy-control framing): within_A = 0.999969 (essentially deterministic — DT's doorway-bleed is a Stage B effect, not a Stage A determinism issue); within_B = 0.9804 — DT's FOV catches enough LivingRoom backdrop that `RandomizeMaterials` moves DT's embedding by 4× more than Bed's and 2× more than TV's.

### Implications for the loop-100 G2.T2 verdict

The cross-loop diagnostic doesn't directly invalidate the G2.T2 restructured-gate failure, but it materially changes the reading of it. The reading-pre-diagnostic was:

> "Bed log_var NARROWED by 0.21 over loops 30→100 — predictor became more confident on Bed during Stage B than at Stage A end. Controls drift MORE than perturbed widen. The architectural prediction (variance widens specifically on perturbed items in response to surprise, while controls remain stable) is not observed."

The reading-post-diagnostic refines this:

1. **Controls are NOT input-invariant in Stage B.** Bed's input has Stage B within-cosine 0.9954 (real variation, small but nonzero). TV's has 0.9895. DT's has 0.9804. These are real visual changes the predictor sees on items that the perturbation API's scope claims not to affect.

2. **The "controls should remain stable" prior was wrong.** The architectural prediction "variance widens specifically on perturbed items, controls remain stable" assumed pure perturbation locality at the input. The locality holds at the API level (`inRoomTypes=["LivingRoom"]`) but does NOT hold at the rendered-frame level. The predictor sees real Stage B variation on Bed, TV, AND DT, just smaller magnitudes than on Dresser and Sofa.

3. **Bed's narrowing (-0.21) is still anomalous, but the framing changes.** Bed has the smallest Stage B input variation (within_B 0.9954 → about 0.005 cosine drop) yet the predictor's log_var moved by 0.21 in the *narrowing* direction. This is not "predictor doesn't see perturbation"; it's "predictor's variance estimate on Bed is moving by ~50× the magnitude of Bed's input variation, in the wrong direction". Possible mechanisms: (i) gradient-descent's global parameter updates couple item-specific variance estimates through the shared transformer body — updates on Dresser+Sofa batches affect Bed's variance estimate as a side-effect; (ii) the per-K-step scalar isotropic variance (spec §3.3) averages across the embedding dimension and across K steps, smearing item-specific surprise into a global term that doesn't track per-item input variation; (iii) some combination.

4. **The differential ratio (c) needs to be re-read against the actual per-item input variation.** A naïve fix would be "weight the control drift by the inverse of input variation", but that's recalibrating the gate post-hoc against the same data it's being tested on — exactly the failure mode §15 guards against. The clean reading is: with `RandomizeMaterials` producing cross-room leakage, the §8.7a "{Bed, DT, TV} = controls" framing was never fully clean. The Bed/TV portion is *cleaner* than DT (smaller magnitude) but is not zero-input-variation as the original gate assumed.

5. **The §8.4 ratio gate at 2.107 still holds:** clean controls (Bed, TV) sum to mean gap 0.0055, perturbed items to 0.01165, ratio 2.107. The perturbation produces 2× more visual variation on perturbed items than the cross-room leakage produces on clean controls. That's a real differential signal at the encoder level, just not as clean as "perturbed = X variation, controls = 0".

### Three readings the reviewer might pick from

(A) **The diagnostic doesn't change the verdict.** The architectural claim is "variance responds preferentially to per-item surprise"; if the predictor's variance changes are dominated by global gradient coupling rather than per-item input variation, that's still the architecture's mechanism not operating as the spec predicts. V2 stands; cross-room leakage is a substrate-level caveat in the V2 report.

(B) **The diagnostic invalidates the G2.T2.c gate's interpretation, not the verdict.** The "control widening MORE than perturbed widening" finding is now legibly a consequence of: (i) Bed's input *does* vary in Stage B, and (ii) the predictor's variance updates are not per-item-isolated. (B-i) re-frames G2.T2.c; (B-ii) is the load-bearing architectural finding. Verdict still V2-leaning but the framing in §11 changes from "predictor doesn't see perturbation" to "predictor sees perturbation but its variance estimate is coupled across items and direction-noisy".

(C) **The diagnostic is enough to pause for substrate-revision before declaring a verdict.** `RandomizeMaterials`'s cross-room leakage was unexpected; the assumption that scoped perturbation produces zero side-effects on out-of-scope rooms is empirically false on this AI2-THOR build. A revised perturbation mechanism (per-object material setting, or asset replacement, or hand-built texture swaps) might produce cleaner locality at the rendered-frame level. The loop-100 evidence is then conditional on a substrate that doesn't isolate the perturbation as the design assumed; a different substrate could produce a different verdict.

I don't have a strong recommendation among A/B/C — they're each defensible. (A) is the strictest reading of the architectural claim and treats the substrate finding as a caveat. (B) is the most precise framing — it acknowledges that the gate's interpretation of the data is now updated while the underlying observation (controls drift more than perturbed widen) stands. (C) is the most conservative — wait, fix the substrate, then re-test. (C) is also the most expensive (re-collection + re-training).

What is **NOT** decided autonomously: which reading to take; the V2 declaration; substrate revision; loop-170 safety check. All defer to experiment-chat review.

### Stop conditions if continuation is authorised

- (A) authorised: write V2 evidence package per §11, with the substrate caveat documented. No further training.
- (B) authorised: same as (A) but with the precise framing — clean controls (Bed, TV) DO have input variation in Stage B due to `RandomizeMaterials` cross-room leakage; the architectural finding is "predictor's variance estimate is coupled across items, not per-item isolated". V2 evidence + caveat.
- (C) authorised: stop training; substrate-revision path becomes live. The substrate-revision design (per-object material setting, asset replacement, etc.) goes to a new substrate-verification cycle.
- Loop-170 safety check (originally option (ii) in session-7's reviewer options): still available but explicitly NOT to be run before A/B/C is decided per the current directive.

### Working-tree state at end of this addendum

Working tree clean. Push hold in effect. Commits since session-7's HANDOFF:
- `91a66fb` — exp(phase2): cross-loop invariance diagnostic — Bed pixel-identical, TV non-deterministic; §8.4 control gaps explained by cross-room visual leakage from `RandomizeMaterials`.
- (this commit) — docs(handoff): post-verdict substrate-determinism diagnostic findings; verdict deferred.

No running jobs. GPU clear. Disk: ~100 GB free.

---

### (Earlier — initial session-7 ninth-STOP framing; superseded by the post-diagnostic addendum above)

### Session 7 outcomes (load-bearing summary)

**§8.7a G2.T2 restructuring (commit `28003ee`).** Dropped the original 0.5 absolute-widening threshold. New three-part criterion:
- **(a) Trajectory direction** (gated): perturbed-item mean log_var monotonically non-decreasing across loops 30 → 100, with up to 3 stochastic dips tolerated.
- **(b) Trajectory shape** (descriptive, not gated): characterise the curve at {30, 35, 50, 75, 100} as accelerating / decelerating / linear / flat. Flat fails the architectural claim; the other three are interpretable.
- **(c) Differential** (gated): `perturbed_widening_at_100 / mean(|control_drift_at_100|) ≥ 2.0`, or controls essentially unmoved.

Audit-trail rationale recorded in §8.7a: the original 0.5 was anchored to "what a clearly-visible response looks like" (a 30–40% cross-stage cosine drop); the empirical drop is only 1–2% per §8.4. Wrong perturbation scale. Same pattern as §8.4's 0.05 → Wilcoxon Reading C restructuring in session 6. SCAFFOLDING inventory at §12 updated.

**research_operations §15 corollary added.** Fifth instance of the absolute-magnitude pattern in the v0 batch and first inside an architectural-strength check. The pattern is now strong enough to promote from recurring-correction to default discipline: **for any architectural-strength gate, default to trajectory-shape or relative-criterion formulations rather than absolute-magnitude, unless an empirical baseline exists from prior runs on the same substrate.**

**Trainer extended-mode (commits `28003ee` + `19d464b`).** `TrainerConfig.transition_diagnostic_extended_mode` auto-enables on resume. Trainer (1) seeds `_diag_per_loop_summary` from the existing diagnostic JSON so prior loops 0..35 from session 6 are preserved; (2) skips the session-6 boundary G2.T2 auto-trip (the restructured G2.T2 is post-hoc at loop 100); (3) re-evaluates G2.T1 + G2.T3 at every checkpoint past `post_end`, breaking on any trip. Drops partial-loop preseed entries (`n_train_steps_attributed < 50`) so the session-6 partial loop 36 (1 step) gets re-recorded fully. Session-6 trip marker renamed to `transition_diagnostic_TRIPPED.session6.txt` so a session-7 trip would write a fresh `*TRIPPED.txt` unambiguously.

**Resume training run (commit `98446dc`).** Launched `python3.12 scripts/run_phase2_train.py --resume_from results/inner_pam_v0/phase2_main/ckpt_12000.pt --max_loops 100`. Ran 286.3s on the RTX 4080 Super, 24,360 gradient steps (steps 12,001 → 36,360). Bank loaded from disk (size 11,986 entries). New checkpoints at scheduled steps {15,000, 20,000, 30,000} plus the final stop step 36,360 (first step of loop 101). Tau calibration skipped (handled in patched script — resume's confidence log starts at step 12,001 ≫ the calibration window [5,000, 10,000); session-6's `tau_calibration.json` at step ~10,000 with τ = 7.7215 stands).

### Restructured G2.T2 verdict — the full data

**Report:** [results/inner_pam_v0/phase2_main/transition_g2t2_restructured.json](results/inner_pam_v0/phase2_main/transition_g2t2_restructured.json).

**(a) Trajectory direction: FAIL** — 34 dips across the loop-30..100 window vs ≤ 3 allowed. Perturbed log_var fluctuates loop-to-loop without a sustained monotonic trend. Concrete dip examples (selected from the 34): loop 32 → 33 (-7.957 → -8.065, -0.108), loop 36 → 37 (-7.915 → -8.061, -0.146), loop 44 → 45 (-7.926 → -8.010, -0.084), loop 87 → 88 (-8.049 → -8.107, -0.058), loop 93 → 94 (-7.891 → -8.112, -0.222), loop 94 → 95 (-8.112 → -8.151, -0.039).

**(b) Trajectory shape: "mixed"** — perturbed log_var at the key loops:

| loop | mean log_var (Dresser + Sofa) | delta from prior | cumulative from loop 30 |
|---:|---:|---:|---:|
| 30 | -8.0404 | — | 0 |
| 35 | -8.0179 | +0.0225 | +0.0225 |
| 50 | -7.9498 | +0.0681 | +0.0906 |
| 75 | -7.9615 | -0.0117 | +0.0789 |
| 100 | -7.9751 | -0.0137 | +0.0653 |

Reading: perturbed log_var widened from loop 30 → 50 (Δ +0.091), then partially retracted from loop 50 → 100 (Δ -0.025). Net widening at loop 100 vs loop 30 = +0.0653 nat — equivalent to σ × √(e^0.065) ≈ σ × 1.033, a 3.3% σ increase. NOT a monotone-accelerating, decelerating, linear, or flat trajectory. Closer to "widened modestly, then plateaued / slightly returned toward baseline".

**(c) Differential at loop 100: FAIL** — perturbed_widening = +0.0653. Per-control drift loop 30 → 100:

| item | log_var(30) | log_var(100) | drift | |drift| |
|---|---:|---:|---:|---:|
| Bed (vp=1) | -8.0291 | -8.2382 | **-0.2091** | 0.2091 |
| DiningTable (vp=2) | -7.9927 | -7.9212 | +0.0715 | 0.0715 |
| Television (vp=5) | -7.9239 | -7.9497 | -0.0258 | 0.0258 |

`control_widening_mean_abs = mean(0.2091, 0.0715, 0.0258) = 0.1021`.
`ratio = 0.0653 / 0.1021 = 0.639` vs threshold 2.0 → FAIL.

**The control-set behaviour is the load-bearing finding.** Bed's log_var NARROWED by 0.21 nat over loops 30 → 100 — the predictor became *more confident* on Bed during Stage B than it was at the end of Stage A. DT widened by 0.07 (modest). TV essentially flat. The perturbed items widened by 0.065 — *less than Bed's narrowing in magnitude.* The architecturally-meaningful prediction ("variance widens specifically on perturbed items in response to surprise, while controls remain stable") is not observed.

### G2.T1 / G2.T3 in-flight verdicts

G2.T1 and G2.T3 evaluated at each new checkpoint in the extended window. No trips:

| step | loop | G2.T1 status | G2.T3 status |
|---:|---:|---|---|
| 15,000 | ~42 | PASS (post_loss_max ≪ 3× baseline) | PASS (max abs drift < 0.3) |
| 20,000 | ~56 | PASS | PASS |
| 30,000 | ~83 | PASS | PASS |
| 36,360 | 101 | PASS | PASS |

`transition_diagnostic.json` `gate_tripped: false` at end of training.

### Reading

The encoder signal is real and statistically distinguishable from controls (§8.4 PASSED with margin). The predictor receives that signal — its variance moves loop-to-loop by ~0.1 nat on multiple items — but the *pattern* of that movement does NOT match the architectural prediction:

1. **Perturbed items don't preferentially widen.** Bed's |drift| over loops 30 → 100 (0.21) is more than 3× the perturbed-item widening (0.065). The predictor's variance changes are dominated by per-item idiosyncrasy rather than perturbation status.

2. **The variance trajectory is highly noisy.** 34 non-monotonic transitions in 70 loops — roughly one dip every other loop. The predictor's variance estimate is unstable at this signal magnitude, not a smooth response surface.

3. **The widening that does happen on perturbed items partially retracts.** Peak widening occurs at loop 50 (+0.091) and partially reverses by loop 100 (+0.065). If the architectural claim were operating, repetition should drive *sustained* widening, not transient.

4. **Bed specifically becomes MORE confident during Stage B.** -0.21 in log_var = σ shrinks by ~9%. There's no architectural reason for the predictor to become more confident on an item the perturbation doesn't visually affect. Most plausibly: the predictor's per-item representations are coupled through the shared transformer body, so updates on perturbed-frame batches affect control-item representations as a side-effect of gradient descent's global parameter updates. The per-K-step scalar isotropic variance (spec §3.3) may be inadequate to isolate item-specific surprise.

Combined with session 6's reading (which observed the +0.022 widening at loop 35 as "the predictor just starting to notice"), the loop-100 evidence says: it *did* notice, modestly, and then mostly forgot — and meanwhile the variance estimates on items the perturbation doesn't visually affect drifted by larger magnitudes in idiosyncratic directions.

### Four options for the reviewer (no autonomous resolution; same set as session 6 plus the loop-100 evidence updating their relative weights)

(i) **Further restructure G2.T2** (third gate re-design). E.g. (a) require widening at *any* loop in 30..100 to exceed control drift, not the loop-100 endpoint specifically. The shape-(b) data shows widening peaks at loop 50 then retracts; a "peak widening" formulation would pass on the perturbed items (+0.091 at loop 50) but probably also pass on Bed (0.21 narrowing). Doesn't really help. Considered low-yield.

(ii) **Continue training to Phase end (~loop 170)** and re-evaluate. The shape-(b) peak-then-retraction at loop 50→100 suggests the predictor is finding a different equilibrium, not progressively widening. Another 70 loops is unlikely to reverse that — but is cheap to run (~5 min compute) and removes the "5 more loops would have changed everything" argument. Defensible diagnostic; resume from `ckpt_36360.pt` if authorised.

(iii) **Switch to a stronger perturbation mechanism.** Texture-variation magnitude increase, multiple `RandomizeMaterials` calls per loop, or per-loop asset replacement at Dresser + Sofa. The loop-100 data narrows the question: it's specifically whether the architectural mechanism can respond to *this* magnitude of perturbation, or whether it needs a fundamentally larger signal. Re-collection cost (~few hours).

(iv) **Reframe verdict toward V2 (Shape-learning falsified).** The architectural prediction is testable as "variance widens specifically on perturbed items in response to repetition + surprise". The loop-100 evidence is consistent with that prediction failing — controls drift more than perturbed widen, perturbed widening peaks early then partially retracts, no monotonic widening over 70 Stage B loops. Per the V2 criteria in §11 of the instructions: "Multiple subsequent gates fail in ways consistent with 'the predictor never represented shapes as recurring patterns.'" Two restructured gates (the 0.5 absolute threshold AND the three-part restructured) have now failed; the second was specifically designed to test the architectural claim more honestly than the first.

**My read:** the loop-100 evidence is stronger for (iv) than session 6's was. Session 6's reading included "+0.022 widening at loop 35 may be the predictor just starting to notice" — that hypothesis is now testable: did it notice? Yes, modestly, peaked at loop 50, then partially retracted. The peak-and-retract pattern is not what "variance responds to surprise and tightens with repetition" predicts; that prediction would have widening *continue* under continued surprise (which Stage B provides every loop). The control behaviour (Bed narrowing more than perturbed widen) also doesn't fit. (ii) is a cheap safety check; (iii) is the "but maybe we just need a bigger signal" path; (iv) is the "let's read what the evidence says" path.

What is **NOT** decided autonomously: which option to take; the V2 verdict declaration; further restructuring of G2.T2; re-collection. All four go to experiment-chat review.

### Stop conditions for the next session (if continuation is authorised)

- If option (i) is authorised: implement the new G2.T2 formulation in docs + analysis script; re-run analysis on the existing JSON (no new training). Standard CC stop discipline applies.
- If option (ii) is authorised: resume from `ckpt_36360.pt` (loop 101 start) to Phase 2 end (~loop 171 at 65k frames, held-out 10 loops). Trainer continues to handle extended-mode diagnostic accumulation; G2.T1 + G2.T3 monitored at the remaining scheduled checkpoints (40k, 55k, final). New G2.T2 evaluation point would need to be specified (loop 170? final?).
- If option (iii) is authorised: STOP current line; substrate-revision path becomes live (re-collection of Phase 2 frames with stronger perturbation mechanism per §9.2-equivalent preflight).
- If option (iv) is authorised: stop training; produce the V2 evidence package per §11; no further Phase 2 work.

### Working-tree state at end of session 7

Working tree clean. Push hold in effect. Commits this session:
- `28003ee` — docs(instructions) + docs(ops) + feat(trainer): §8.7a G2.T2 restructured + research_operations §15 corollary + trainer extended-mode diagnostic.
- `19d464b` — prep(session-7): preseed only fully-recorded loops; preserve session-6 trip marker as historical record.
- `98446dc` — exp(phase2): training resumed loop 33→100 + restructured G2.T2 verdict (FAIL all gated).
- (this commit) — exp(phase2) + docs(handoff): HANDOFF entry with full loop-30..100 trajectory and restructured-G2.T2 verdict.

No running jobs. GPU clear (training exited cleanly after 286.3s; tau-calibration crash post-training was a script-level bug fixed in the trainer commit above, no data loss). Disk: 101 GB free at training-launch time.

---

### (Earlier eighth STOP — session 6's G2.T2 trip; superseded by the session 7 restructuring + loop-100 verdict above)

**Eighth STOP (session 6 outcome).** §8.4 restructured and PASSED on the existing collected data; Phase 2 training launched with `--max_loops 35` per the session-6 authorisation; the in-flight transition diagnostic's **G2.T2 (perturbed log_var widening) tripped** with widening = 0.022 versus the 0.5 SCAFFOLDING threshold. G2.T1 (loss spike) and G2.T3 (control drift) both clean. This is the explicit STOP-for-review condition recorded in the session-5 handoff's "Stop conditions during the next session's execution" block. Training stopped at loop 36 boundary (step 12,960); `transition_diagnostic_TRIPPED.session6.txt` (renamed from `.txt` at the start of session 7) written; trainer exited non-zero (status 3). The session-7 restructuring of G2.T2 supersedes the session-6 trip's gate logic; the per-loop trajectory data from session 6 (loops 0..35) is preserved in the current `transition_diagnostic.json` and carried into the loop-100 analysis.

### Session 6 outcomes (load-bearing summary)

**§8.4 restructured (commit `58548d0`).** Encode script rewritten so embeddings.npy is saved unconditionally (save-first / gate-second) and the gate suite is:
- **Ratio gate** with clean controls = {Bed, Television}, DT recorded as record-only noisy-control diagnostic. Threshold ≥ 2.0.
- **Wilcoxon signed-rank gate** (Reading C per session-6 authorisation): per perturbed item, run `scipy.stats.wilcoxon((1 − cross_stage_cosine), alternative='greater', method='approx')` over the full n_a × n_b apex-pair grid; Bonferroni-correct by 2 (perturbed items); gate at corrected p < 0.001. Bed, TV, DT each run record-only.

Spec/instructions §8.4 + SCAFFOLDING inventory §12 updated to match.

**§8.4 verdict on existing data (commit `c127457`).** Both gates PASS:
| metric | value | threshold | verdict |
|---|---:|---|---|
| `perturbed_mean_gap` | 0.01165 (Dresser 0.0097, Sofa 0.0136) | — | — |
| `clean_control_mean_gap` | 0.00553 (Bed 0.0045, TV 0.0065) | — | — |
| ratio | **2.107** | ≥ 2.0 | PASS (~5% margin) |
| DT noisy-control gap | 0.01362 | record-only | (≈ Sofa magnitude; doorway-bleed unchanged from session-5) |
| Wilcoxon Dresser corrected_p | < 1e-300 (floored) | < 0.001 | PASS |
| Wilcoxon Sofa corrected_p | < 1e-300 (floored) | < 0.001 | PASS |
| Wilcoxon Bed corrected_p | < 1e-300 (record-only) | n/a | drift exists but small (median 0.0065) |
| Wilcoxon TV corrected_p | < 1e-300 (record-only) | n/a | drift exists but small (median 0.0137) |
| Wilcoxon DT corrected_p | < 1e-300 (record-only) | n/a | drift larger (median 0.0232) — consistent with noisy-control framing |

Encode used the saved (65000, 1024) fp32 matrix at `data/phase2_embeddings/embeddings.npy`. Encoding took 141.8s on the RTX 4080 Super (458 f/s).

**Trainer changes (commit `4435696`).** Added `--max_loops` and `--resume_from` to `scripts/run_phase2_train.py`; corresponding `TrainerConfig.max_loops` and `TrainerConfig.resume_step` in `src/trainer/online_trainer.py`. Smoke-tested both on the real Phase 2 data (fresh run with `--max_loops 1` stopped at step 720; resume from `ckpt_720.pt` with `--max_loops 2` continued for 360 more gradient steps and stopped at step 1080; RNG-restore fixed to handle GPU→CPU map_location). Resume disables the §8.7a diagnostic by construction (its evaluation window has already passed at any resume point).

**Phase 2 training launch (no further commit needed; results in `results/inner_pam_v0/phase2_main/`).** Launched at 2026-05-14, `--max_loops 35`. Ran 124.2s, 12,930 gradient steps, stopped at step 12,960 (first step of loop 36). Predictor params: 21,555,728 (within tolerance). Tau calibrated to 7.721485 over the 5k–10k window. Checkpoints at steps {1000, 2000, 4000, 6500, 10000, 12000} on disk (`*.pt` + `ckpt_<step>/` bank dirs); no checkpoint at the actual stop step 12,960 because the trainer breaks on §8.7a trip without writing a final checkpoint. **Last reliable resume point is `ckpt_12000.pt`** (loop 33; 960 steps / ~2.5 loops before the trip).

### G2.T2 trip — the full data

**Trip record** at [results/inner_pam_v0/phase2_main/transition_diagnostic_TRIPPED.txt](results/inner_pam_v0/phase2_main/transition_diagnostic_TRIPPED.txt):
```json
{
  "gate_tripped": true,
  "gate_name": "G2.T2_perturbed_widening_insufficient",
  "perturbed_items_vp_ids": [3, 4],
  "log_var_at_baseline_end_loop": 30,
  "log_var_baseline_end": -8.040,
  "log_var_at_post_end_loop": 35,
  "log_var_post_end": -8.018,
  "delta_observed": 0.022,
  "delta_required_min": 0.5
}
```

**Per-loop trajectory (loops 25..36):** `mean_log_var` columns are perturbed (D = Dresser/vp=3, S = Sofa/vp=4) and controls (B = Bed/vp=1, DT = DiningTable/vp=2, TV = Television/vp=5). Stage A baseline = loops 25–30; perturbation onset at loop 31; Stage B post-onset window = loops 31–35.

| loop | mean_loss | lv3 D | lv4 S | lv1 B | lv2 DT | lv5 TV |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | -55251 | -7.96 | -7.73 | -8.02 | -7.93 | -7.75 |
| 26 | -55687 | -8.11 | -7.95 | -8.13 | -7.82 | -7.92 |
| 27 | -55449 | -8.14 | -7.88 | -8.22 | -7.89 | -7.72 |
| 28 | -55562 | -8.19 | -7.95 | -8.15 | -8.02 | -7.61 |
| 29 | -55626 | -8.18 | -7.99 | -8.17 | -7.94 | -7.91 |
| 30 | -55407 | -8.11 | -7.97 | -8.03 | -7.99 | -7.92 |
| **31** | -55588 | -8.10 | -7.83 | -8.23 | -8.08 | -7.80 | ← Stage B onset
| 32 | -55756 | -8.04 | -7.87 | -8.12 | -7.88 | -7.85 |
| 33 | -55946 | -8.19 | -7.94 | -8.19 | -7.99 | -7.70 |
| 34 | -56043 | -8.23 | -7.92 | -8.11 | -8.04 | -7.83 |
| 35 | -55798 | -8.09 | -7.94 | -8.26 | -7.78 | -7.70 |
| 36 | -56482 | nan   | nan   | -8.05 | nan   | nan   | ← partial; one Bed step before max_loops break

G2.T1: baseline_loss_mean(loops 25–30) = -55,497.02; post_loss_max(loops 31–35) = -55,588.48; spike_threshold (3× sign-safe form) = -55,497.02 + 2 × |baseline| = +55,497.02; post_loss_max ≪ spike_threshold → **G2.T1 PASS**.
G2.T3: control-item |drift| across loops 30→35: Bed = 0.229 (lv30=-8.029, lv35=-8.258), DT = 0.218 (lv30=-7.993, lv35=-7.775), TV = 0.229 (lv30=-7.924, lv35=-7.695) — all below the 0.3 threshold → **G2.T3 PASS**.
G2.T2: perturbed mean log_var at loop 30 = -8.0404; at loop 35 = -8.0179; delta = +0.0225 ≪ 0.5 → **G2.T2 FAIL**.

### Reading

**The perturbation produces a measurable encoder-level signal (§8.4 Wilcoxon corrected_p < 1e-300 on Dresser and Sofa, ratio 2.1 vs clean controls), but the predictor is not absorbing that signal as a widening of its predicted variance.** The predictor's log_var on perturbed items at loop 30 sits at -8.04 (σ ≈ 0.018) — already very tight after 30 loops of Stage A repetition. Across the five post-onset loops, that log_var barely moves (+0.022), staying inside the same -8.0..-8.2 band that controls also occupy.

A few mechanisms could be in play, all worth the reviewer's judgement:

1. **The architecture's variance is per-K-step scalar, isotropic, averaged over the embedding dimension** (spec §3.3). The Stage B perturbation magnitude at the encoder (cross-stage cosine ~0.98–0.99, equivalent to per-frame L2 ~0.18–0.20 against unit vectors) is small relative to the predictor's typical step-to-step path geometry. The Gaussian NLL's surprise signal from such a small drift may be too diffuse to drive a 0.5-unit log_var widening within 5 loops × 360 steps = 1,800 gradient updates.

2. **The 0.5 log_var-widening threshold may be wrong-scale** (analogous to the 0.05 §8.4 case caught in session 5). 0.5 in log_var units = `e^0.5 ≈ 1.65×` widening in σ². Whether that's the architecturally-correct expectation for *this* magnitude of perturbation has not been empirically grounded — the SCAFFOLDING was set pre-substrate-empirical-data. Per `research_operations_v1.md` §15's absolute-magnitude-threshold pattern, this is the same class of structural vulnerability.

3. **The post-onset window may be too short.** 5 loops (~1,800 gradient steps) might be insufficient for the predictor to register and accumulate variance updates against perturbation evidence; spec §3.4's "variance responds to surprise" mechanism may be slower than the SCAFFOLDING window assumed.

4. **The architecture's confidence-graded mechanism may not be operating as the spec predicts on this signal magnitude** (the framing recorded in session-5's session-6 stop conditions). This is the architecturally-load-bearing interpretation — if the predictor simply *can't* widen variance on small-but-real perturbation, the whole curriculum's premise needs reframing.

### Four options for the reviewer (no autonomous resolution)

(i) **Recalibrate the G2.T2 threshold against the empirical loop-level distribution and continue training.** Loops 25–35 give an empirical baseline for "what log_var width does this predictor produce, and how much can it move per N loops". An empirically-grounded threshold (e.g. "widening ≥ 1σ of the loop-level log_var noise") would be a more architecturally-honest gate than the pre-empirical 0.5. Per `research_operations_v1.md` §15 ("SCAFFOLDING thresholds get evaluated against what they're protecting"), the 0.5 was meant to detect "predictor absorbs perturbation as widening" — if the empirical signal *exists* but at a smaller scale, the threshold needs recalibrating rather than the experiment failing. Recommended if the reviewer judges that mechanism 1 or 3 above is the load-bearing one.

(ii) **Extend the post-onset observation window and re-evaluate at a later loop.** Resume from `ckpt_12000.pt` with the diagnostic disabled, run another N loops (e.g. to loop 60 or loop 100), then re-evaluate G2.T2 on that extended window. Predicted log_var changes can be slow under continuous training; the 5-loop window may simply be too small to read.

(iii) **Reframe: switch to a stronger perturbation mechanism.** This is option (iii) from the seventh-STOP options carried forward. Texture-variation magnitude increase, multiple `RandomizeMaterials` calls per loop, or per-loop asset replacement at Dresser+Sofa. Highest cost — requires re-running Phase 2 collection on a new substrate parameterisation. Reserve for if (i) and (ii) leave the architectural-strength question unresolved.

(iv) **Reframe deeper: declare the architecture's confidence-graded mechanism falsified at this signal magnitude.** Move to V2 (Shape-learning falsified) territory with the documented evidence. Earliest possible declaration; consult the verdict structure in §11 of the instructions.

I'd lean **(i) and (ii) before (iii) or (iv)**: the §8.4 gate confirms the encoder signal is real and statistically distinguishable, so the question is purely whether the predictor's variance-response mechanism is registering it at this magnitude. (i) tests "is the threshold wrong-scale"; (ii) tests "is the window too short". Either can be done without re-collection. (iii) is much more expensive; (iv) is a strong claim that the reviewer should make only after exhausting the cheaper investigations.

What is **NOT** decided autonomously: recalibrating the G2.T2 threshold, extending the training window past loop 35, switching the perturbation mechanism, or declaring a verdict. All four go to experiment-chat review.

### Stop conditions for the next session (if the reviewer authorises continuation)

- If option (i) is authorised: re-run the diagnostic with the new threshold on the existing JSON (no training; record-only). Document the empirical loop-level log_var noise in a HANDOFF entry. Standard CC stop discipline (`CODING_STANDARDS.md` §9.4) applies to anything beyond that.
- If option (ii) is authorised: resume from `ckpt_12000.pt` with `--resume_from results/inner_pam_v0/phase2_main/ckpt_12000.pt` and no `--max_loops` (full phase) or a new max_loops cap. The §8.7a diagnostic is disabled in resume mode by construction; the trainer will run to phase end. Monitor the per-loop log_var trajectory (the trainer keeps logging via `transition_diagnostic.json` in the existing run dir — but note that on resume the diag is disabled so `per_loop` will not be appended to; if extended logging is wanted, that needs a small code change first).
- If option (iii) is authorised: stop the current line; the seventh-STOP option-(iii) re-collection path becomes live. Substantial work.
- If option (iv) is authorised: produce the V2 evidence package per §11; no further training.

### Working-tree state at end of session 6

Working tree clean. Push hold in effect per project standing instruction (no `git push` without explicit authorisation). Commits this session:
- `58548d0` — feat(encode) + docs(instructions): §8.4 restructured per session-6 directive.
- `c127457` — exp(phase2): §8.4 restructured gate verdict — PASS on existing collected data.
- `4435696` — feat(trainer): --max_loops + --resume_from for session-6 STOP-and-resume.

Plus the session-6 closing commit will add this HANDOFF update.

No running jobs. GPU clear (`nvidia-smi --query-gpu=utilization.gpu` returned 9% at launch; encode + training both completed cleanly). Disk: 112 GB free on `/mnt/c` at training-launch time (well above the 50 GB session floor). PID 160900 (training) exited at completion; PID 158359 (encode) exited at completion.

---

### (Earlier — session 5 directive that authorised session 6; superseded by the session 6 outcome above)

**STOP for experiment-chat review (seventh STOP in session 5).** Phase 2 collection completed cleanly on the corrected substrate (commit `9a3d636` launched; `6d6e58d` recorded the encoding + §8.4 outcome). The §8.4 verification — the load-bearing locality test the modified-(i) gate moved the magnitude question to — **FAILS on both gates with real data**. Embeddings.npy NOT written; Phase 2 training NOT launched.

**§8.4 results (per-item Stage B vs Stage A gap at apex frames, DINOv2 CLS cosines, 25 samples/stage):**

| item | room | role | gap |
|---|---|---|---:|
| Bed | Bedroom | control | 0.0045 |
| **DiningTable** | **Bedroom** | **control** | **0.0144** |
| Television | Bedroom | control | 0.0068 |
| Dresser | LivingRoom | perturbed | 0.0097 |
| Sofa | LivingRoom | perturbed | 0.0136 |

**Gate verdicts:**

- **Absolute gate (per-perturbed-item gap > 0.05):** FAIL — Dresser 0.0097 and Sofa 0.0136 are both well below 0.05.
- **Differential ratio gate (perturbed_mean_gap / control_mean_gap ≥ 2.0):** FAIL — ratio = 0.0116 / 0.0086 = **1.357**.

**Two findings:**

(1) **The DINOv2 apex-embedding signal from `RandomizeMaterials(inRoomTypes=["LivingRoom"])` is small.** Perturbed items move 0.010–0.014 in apex-cosine space between Stage A and Stage B, on average across 25 samples per stage and 150 Stage B loops. That's a real signal (well above zero) but well below the 0.05 SCAFFOLDING threshold I'd written into instructions §8.4. The threshold was a pre-empirical guess; the empirical magnitude is now known.

(2) **DiningTable's residual doorway-bleed at corrected eye-height is the same magnitude as Sofa's perturbation signal.** DT gap = 0.0144 ≈ Sofa gap = 0.0136. The substrate fix improved DT substantially (gap was 0.045 at the original pose; now 0.014) but didn't eliminate the bleed — DT's FOV at the h118 pose still catches enough of the LivingRoom backdrop that LivingRoom retexturing perturbs DT's DINOv2 embedding by the same amount the actual LivingRoom items' embeddings shift. If DT is excluded from the control set:

  - `control_mean_gap (Bed + Television only)` = (0.0045 + 0.0068) / 2 = **0.00565**
  - `ratio` = 0.0116 / 0.00565 = **2.06** — right at the 2.0 threshold.

**Sample-size SCAFFOLDING bug noted + fixed mid-run (committed `6d6e58d`).** First encode pass hit the insufficient-frames branch because I'd hardcoded `_PERTURBATION_SAMPLE_N = 50` (inherited from spec §5.1/5.2's pattern) but Stage A only contains 31 loops → 31 apex frames per item per stage. Reduced to 25 (well below Stage A's 31-loop ceiling), instructions §8.4 updated. This change is mechanical bookkeeping (not authorised gate-tuning) — 25 still gives ~500 pairs to the gap estimator and is bounded by the substrate's natural Stage A length.

**Three options for the reviewer (no autonomous resolution per directive's "Not authorised to tune further"):**

(i) **Accept DT as a "noisy control"; redefine the control set as {Bed, Television} only.** This is option (a) from the third STOP — already authorised by the user as the framing-of-record for the §8.4 differential metric. The ratio with that control set is 2.06, right at the 2.0 threshold. The directive language ("≥2× criterion") naturally accommodates this since the original framing already acknowledged DT as a noisy control. **The absolute gate still fails** (both Dresser and Sofa below 0.05); the 0.05 threshold itself would need recalibration against the empirical distribution to interpret the result.

(ii) **Recalibrate both thresholds against the empirical distribution.** Empirical perturbed gaps sit in 0.01–0.014 range; the 0.05 absolute threshold was a guess that the actual perturbation magnitude doesn't reach. The §15 principle ("SCAFFOLDING thresholds get evaluated against what they're protecting") applies — 0.05 was a placeholder; the data now defines the achievable scale. A recalibrated threshold (e.g., absolute > 0.005 + ratio ≥ 1.5) would pass with the current control set, or absolute > 0.005 + ratio ≥ 2.0 would pass without DT.

(iii) **Apply a stronger perturbation mechanism.** Increase the texture variation magnitude per RandomizeMaterials draw, or use multiple perturbation calls per loop, or switch to a perturbation that affects more pixels (e.g., furniture replacement). This addresses the small-DINOv2-signal directly rather than recalibrating the gate. Highest cost — would require re-running collection + encoding.

(iv) **Apply another substrate fix to DT** (further pose change to remove the residual doorway-bleed). The h118 pose reduced bleed from 0.045 to 0.014; another iteration might get it to ~0.005. But the pose-search already identified h118 as the only candidate passing the prior G_M2 > 0.98 threshold; finding a candidate with even better DT-locality would require relaxing the 0.98 stability gate or accepting a different trade-off. Substantial effort with uncertain payoff.

I'd recommend **(i) + (ii) combined**: redefine controls as {Bed, Television} (DT moves to a per-item-disaggregation-only read), recalibrate the absolute threshold from 0.05 to ~0.005 (10× lower, justified by empirical apex-gap distribution), keep the ratio at 2.0. With those changes the §8.4 verdict on the existing collected data would be PASS (ratio 2.06 ≥ 2.0; per-perturbed-item gap 0.0097/0.0136 ≥ 0.005). The §15 discipline says SCAFFOLDING thresholds get evaluated against their protective purpose; the empirical data now defines the scale. (iii) and (iv) are bigger interventions; reserve for if (i)+(ii) leaves the reviewer unsatisfied with the locality signal strength.

What is NOT decided autonomously: lowering the 0.05 absolute or 2.0 ratio thresholds; redefining the control set; changing the perturbation mechanism; re-running collection.

---

### (Earlier sixth STOP in session 5) — resolved by the substrate y-fix; superseded by the seventh STOP above

The sixth STOP raised the `forceAction`/y-bob camera-elevation bug. Resolved by the user's "Apply the three-change fix" authorisation (2026-05-14); fix landed in commit `4b38d42`; v3 verifications PASSed (commit `3eb89c5`); Phase 2 collection on the corrected substrate launched (`9a3d636`) and completed cleanly (`6d6e58d`'s precursor). The §8.4 verdict on the actual collected stream (this STOP) is the next gate.

---

**STOP for experiment-chat review (sixth STOP in session 5).** Phase 2 launch (commit `e3feaa2`) was paused 2026-05-14 ~04:00Z after a reviewer-directed trajectory diagnostic surfaced a substrate-rendering bug. Collection process (PID 109417) killed at frame 6,800; partial Phase 2 frames removed from `data/phase2_frames/`. Diagnostic + recommended fix at [results/phase2_calibration_v2/trajectory_diagnostic.json](results/phase2_calibration_v2/trajectory_diagnostic.json) (committed `29478a2`).

**The bug:** Agent base y oscillates between close-up steps (y = 0.9010) and transit steps (y = 0.0065) at every close_up↔transit phase boundary — a 0.894 m vertical bounce. Because the rendered camera sits at a fixed offset above the agent base in AI2-THOR, the camera elevation oscillates between ~1.8 m (close-up; bird's-eye view of furniture from above) and ~0.9 m (transit; normal eye-height). The reviewer's two observations have a common cause:

- "Camera height bobs unexpectedly" → the literal 0.9 m vertical leap of the camera at every phase boundary.
- "Agent jumps through walls" → no horizontal wall-jump was found (zero displacement pairs in the [0.25, 0.50), [0.50, 1.00), [1.00, 2.00) bins for x/z-only motion); the "wall-jump" effect is visual artefact of the camera leaping 0.9 m in elevation, which looks like teleportation past a floor/ceiling.

**Diagnostic evidence:**

| diagnostic | finding |
|---|---|
| 1. Per-frame y range | **0.0065 to 0.9010**, 0.8945 m range, exactly 2 unique values rounded to 4 decimal places. |
| 1. y per phase | Close-up frames: y = 0.9010 (from route.json's `viewing_position.y`). Transit frames: y = 0.0065 (from NavMesh-planned waypoints at floor level). |
| 2. 3D displacement | Bimodal: 1000 pairs near-zero (corner rotations, no x/z displacement), 635 in [0.15, 0.20) (densified motion), 120 in [0.20, 0.25), and **29 outliers at exactly 0.8945 m, all with dx=dz=0 and \|dy\|=0.894**. |
| 2. x/z wall-jumps | **Zero pairs in [0.25, 0.50), [0.50, 1.00), or [1.00, 2.00) for x/z-only**. No horizontal wall traversal. |
| 3. Contact sheets | 29 PNG strips at `results/phase2_calibration_v2/discontinuity_frames/` show close-up frames rendered from a bird's-eye angle and transit frames from eye-height — the elevation drop is visually unmistakable. |

**Root cause traced:** `src/env/continuous_motion_explorer.py:426` calls AI2-THOR's `Teleport` with `forceAction=True` AND the input position dict's y value. `forceAction=True` bypasses NavMesh + floor-snap validation, so the agent base lands at whatever y is supplied. The y value comes from two different sources depending on the trajectory phase:

- **Close-up steps** copy y from `route.json` items' `viewing_position.y` (0.9010, which is the prior repo's stage_0b standing agent.position.y, mistakenly persisted into the route).
- **Transit steps** copy y from `controller.step("GetShortestPathToPoint")` waypoints (~0.0065, NavMesh floor level).

`git blame` shows `forceAction=True` has been in the production explorer since the session-4 commit `ec172c7` — **not** a session-5 leak from pose-search reachability testing; the pose-search and motion-continuity scripts (which I added this session) independently use `forceAction=True`, but the production path predates them.

**Recommended fix (3 changes, ~5 lines):**

1. In `ContinuousMotionExplorer.__init__`, query `controller.step("GetReachablePositions")` once and store a representative floor y on `self._agent_floor_y` (e.g., median of the reported y values).
2. In `ContinuousMotionExplorer._teleport`, **drop the input position's y** and pass `self._agent_floor_y` to `Teleport`. Keep `forceAction=True` — it's still needed for the close-up path's 0.20 m off-grid x/z steps which don't align to AI2-THOR's 0.25 m grid.
3. Optionally add `standing=True` to the Teleport call to make the agent's standing posture explicit.

**Downstream invalidation scope:**

- v1 calibration (session-4, 316 fpl, 1,580 frames) — rendered with camera bobbing; close-up apex frames at 1.8 m camera elevation. Used as the substrate baseline that determined 316 fpl + cross-loop apex bit-identicity findings. The bit-identicity finding is still valid (deterministic rendering at fixed pose is deterministic regardless of elevation); the apex visual framing is not "agent eye-level" as the spec intended.
- v2 calibration (session-5, 360 fpl, 1,800 frames) — same bug; the +13.9 % loop length finding is still valid (transit paths are unchanged), but the rendered frames have the same elevation artefact.
- Preflight runs (multiple, original-pose + adjusted-pose) — DINOv2 cosines were computed on bird's-eye close-up captures; the per-item Bedroom/LivingRoom contrasts are valid for the data as captured but the substrate they captured isn't the intended one.
- Pose-search candidate frames — same; the DT_h118 pose was selected as the lone candidate with DINOv2 stability > 0.98 against an out-of-scope perturbation. The stability finding holds for the captures as taken; the DT_h118 verdict should be re-verified at the corrected camera height after the fix.
- Motion-continuity sweep frames — same; the consec-cosine values were computed at the elevated close-up sweep heights.
- Aborted Phase 2 collection (6,800 frames) — discarded; not used.

**After the fix lands:** re-run v2 calibration, re-run preflight on adjusted geometry, re-run pose-search verification (or accept the DT_h118 verdict pending re-verification once frames are rendered correctly), then re-launch Phase 2 collection. The §8.4 differential gate threshold (0.01) is SCAFFOLDING that may need recalibration after the corrected frames produce a meaningful empirical distribution.

**STOP for review.** No code changes to the explorer; no re-runs. The diagnostic-and-fix recommendation is the deliverable; the reviewer signs off on the fix before CC implements it.

---

### (Earlier fifth STOP in session 5) — superseded by the sixth STOP above

---

### (Earlier fifth STOP in session 5) — resolved by the modified-(i) authorisation; superseded by the Phase 2 launch above

**STOP for experiment-chat review (fifth STOP in session 5).** The user's fourth authorisation (option (a): accept +13.9% loop length, hold 65k budget, recalibrate dependent arithmetic transparently; run motion-continuity + preflight on adjusted geometry; if both pass, launch Phase 2) was implemented through doc updates (commits `71f7693`, `b305aaa`, `21829f3`) and verifications (commits `45aca0e`, `4b1408c`). Within-loop motion-continuity on v2 calibration **PASSED** with the curriculum-aligned verdict. The preflight on adjusted geometry produced a **mixed verdict** that needs reviewer judgement:

**Adjusted-geometry preflight summary:**

| gate | result | observation |
|---|---|---|
| G_M1 mechanism | ✓ PASS | `RandomizeMaterials(inRoomTypes=['LivingRoom'])` returns success |
| G_M2 Bedroom locality | ✓ PASS | Bedroom DINOv2 3-call mean = **0.9822** (> 0.98); DT now in-line with Bed and TV rather than the outlier |
| G_M3 contrast | ✗ S1 trip | Contrast = +0.0062 (locality-correct direction; < 0.02 SCAFFOLDING threshold) |

**Per-item DINOv2 3-call means at adjusted geometry:**

| item | room | adjusted | original (3rd STOP) | delta |
|---|---|---:|---:|---:|
| Bed | Bedroom | 0.9847 | 0.9884 | −0.004 |
| DiningTable | Bedroom | **0.9768** | **0.9448** | **+0.032** |
| Television | Bedroom | 0.9852 | 0.9945 | −0.009 |
| Dresser | LivingRoom | 0.9806 | 0.9884 | −0.008 |
| Sofa | LivingRoom | 0.9713 | 0.9757 | −0.004 |
| **Bedroom mean** | | **0.9822** | 0.9759 | **+0.006** (PASS) |
| **LivingRoom mean** | | 0.9760 | 0.9820 | −0.006 |
| **Contrast** | | **+0.0062** | **−0.0061** | **+0.012** (right-sign flip) |

**The substrate fix worked.** DT improved by +0.032 in DINOv2 mean cosine. Bedroom mean rose above 0.98 cleanly. The contrast flipped sign (from −0.006 wrong-direction to +0.006 right-direction). All five items now sit in a ~0.97–0.99 band rather than one item being a 0.94 outlier.

**Why S1 still trips.** The S1 = 0.02 threshold was a SCAFFOLDING guess written *before* any empirical contrast data existed. Across the two preflight runs (original pose and adjusted pose), the empirical contrast magnitudes are |−0.006| and |+0.006| — both small, both in the same order of magnitude. The 0.02 threshold was sized for a larger DINOv2 signal than `RandomizeMaterials` actually produces: most pixels in each frame are unchanged geometry, walls, lighting, and background; texture swaps move per-item embeddings by ~0.02–0.04, and the perturbed-vs-control mean contrast is the small difference of two small numbers.

**Reading.** The locality fix is operating correctly at the substrate level (DT no longer doorway-bleeding into the LivingRoom; Bedroom mean above 0.98). The contrast magnitude trips a threshold that was sized without empirical grounding, and §15's just-added principle ("SCAFFOLDING thresholds get evaluated against what they're protecting, not adjusted by margin") applies — S1 was meant to catch "perturbation is not producing a meaningful localisation signal" and the empirical contrast *is* producing such a signal in the right direction.

**Three options for the reviewer:**

(i) **Accept the +0.006 contrast as evidence of locality and lower S1 to a value the empirical distribution supports** (e.g., S1 = 0.005, well below the observed magnitude). The locality claim is being tested by the direction of the contrast (positive = LivingRoom moves more); the magnitude tells us how strong that signal is but not whether it exists. The §8.4 verification on the actual collected stream is the load-bearing test; the preflight is the early-warning sanity check, and the warning is now: "the signal is small but in the right direction." Recommended.

(ii) **Accept the substrate fix but override the preflight S1 trip in writing**, without changing the threshold. Document the empirical contrast distribution and proceed with the §8.4 verification at the load-bearing site. This is the "the preflight is doing its job by flagging the small magnitude; we proceed with our eyes open" reading. Slightly weaker than (i) because it leaves the next preflight run vulnerable to the same trip.

(iii) **Fall back to option (d) from the third STOP**: revert `data/route_phase2.json` use, accept the original DT locality breach, document it via the §8.4 differential metric. Most conservative; gives up the substrate fix entirely.

I recommend **(i)**. The substrate fix is real (Bedroom mean cleanly above 0.98 with DT no longer an outlier). The S1 threshold was a SCAFFOLDING guess that empirical data now contradicts; recalibrating it from the empirical distribution is exactly the workflow §15 prescribes. (ii) is defensible if you'd rather keep S1 = 0.02 as a "we tried" flag for future runs.

What is NOT decided autonomously: any change to S1 or to the contrast criterion. STOP for review per the directive.

---

### (Earlier fourth STOP in session 5) — resolved by option (a) authorisation; superseded by the fifth STOP above

The fourth STOP raised four options for handling the +13.9% loop length shift. The user authorised **option (a)** — accept the shift, hold the 65k budget, recalibrate the dependent arithmetic transparently, then run motion-continuity + preflight verifications. The recalibration was committed (`71f7693`, `b305aaa`, `21829f3`, `45aca0e`, `4b1408c`). Within-loop motion-continuity passed; preflight tripped S1 (this STOP).

---

**STOP for experiment-chat review (fourth STOP in session 5).** The user's third authorisation (run extended diagnostic → pose search → adjusted geometry → re-calibrate; STOP if loop length shifts > 10%) was implemented end-to-end (commit `3bd341b`). The pose search and motion-continuity check produced a clean DiningTable pose (DiningTable_h118 — heading 117.6°, position 8.25/4.75, DINOv2 stability mean 0.9827, motion-continuity consec cosine min 0.7645 with 0 bit-identical pairs). However:

**The v2 calibration shifts the loop length by +13.9%, exceeding the directive's 10% STOP threshold.**

| metric | baseline (session-4) | v2 (adjusted DT pose) | shift |
|---|---:|---:|---|
| frames per loop | 316 | **360** | **+13.9%** ✗ |
| close-up frames per item | ~11 per loop | ~11 per loop (55/5) | ~0% |
| transit frames per loop | ~260 | 305 | +17.3% |
| within-loop continuity | clean | (not run; stopped at the loop-length check) | — |
| Phase-2 budget impact at 65k | 195 trained loops | ~171 trained loops, ~42 reps into the 100+ bin (vs ~65 prior) | budget margin halved |

The DT shift extends the transit paths: the new DT position at (8.25, 4.75) is further from Bed (11.75, 2.25) AND requires a longer NavMesh route to Dresser (6.0, 3.0) than the original pose did. Close-up frames are unchanged in count; the extra ~44 frames/loop are almost entirely in transit.

**Per the directive: STOP at >10% loop length shift.** The preflight on the adjusted geometry was NOT re-run. The directive structure ("Re-run preflight ... if it passes, proceed to Phase 2 launch. If not, stop and report") gates preflight on calibration succeeding; calibration STOPped first.

**Three options for the reviewer:**

(a) **Accept the +13.9% loop length increase, recalibrate the §8.3 budget arithmetic, and proceed.** The 65k frame budget at 360 frames/loop yields ~171 trained loops, ~141 Stage B loops, ~42 reps into the 100+ bin — still adequate to test §2.2 but with about half the margin we had at 316. The reviewer might re-derive the budget to a slightly higher value (~73k = 200 trained loops, ~70 reps margin) if they want to preserve the original margin. This is the lowest-effort path that preserves the locality-fix.

(b) **Constrain the pose search to candidates that preserve loop length more closely.** The h118 pose works for locality but the +3.4m increase on Bed→DT transit is structural to its location. Other candidates that didn't pass the > 0.98 stability bar (h147 at 0.972, h206 at 0.971) might preserve loop length better; could re-run with a relaxed stability threshold (e.g., > 0.97) and pick the loop-length-cheapest candidate. Trades off some locality cleanness for transit-budget preservation.

(c) **Switch house seed** (the disruptive option (c) from the third STOP). A different seed-house might have DT, Bed, and Dresser arrangement that doesn't force long transits when DT is repositioned to avoid the doorway. Requires re-running session-4 substrate verification + encoder verification protocol from scratch.

(d) **Accept the locality breach** (option (d) from the third STOP). Don't change the substrate; treat DT as a non-clean control; document the §8.4 differential metric per-item so the reviewer can read DT's behaviour separately. Preserves substrate + loop length; weakens the architectural claim.

I'd recommend **(a)** if the +13.9% loop length cost is acceptable for the locality fix it buys. The trained-loops count at 65k drops from 195 to 171 (-12%); the 100+ bin reps drop from 65 to 42 (-35%); the budget arithmetic still works but with less margin. **(b)** is the principled middle ground if budget margin matters more than DINOv2-stability strictness. **(c)** and **(d)** are the substrate-disruptive and substrate-accepting extremes respectively.

What is NOT being decided autonomously: lowering the 10% STOP threshold to admit the +13.9% shift; lowering the 0.98 DINOv2 stability threshold to admit other candidate poses; auto-bumping the 65k frame budget to compensate for the loop-length increase. All of those go past the directive.

---

### (Earlier third STOP in session 5) — resolved by the substrate fix path; superseded by the fourth STOP above

The third STOP raised four options (a–d) for handling DiningTable's DINOv2 bleed. The user authorised **option (b)** — adjust DT's viewing pose to remove the doorway view — and provided the detailed workflow (extended diagnostic → pose search → motion-continuity check → adjusted route → re-run session-4 calibration → re-run preflight). All of those steps ran cleanly (commit `3bd341b`) but the v2 calibration loop length tripped the 10% STOP threshold (this STOP).

---

**STOP for experiment-chat review (third STOP in session 5).** The user's second authorisation (option (a): switch the preflight to DINOv2-embedding contrast; G_M2 at Bedroom DINOv2 cosine > 0.98; G_M3 at 40–60% of observed mean DINOv2 contrast; STOP-and-report if calibration surfaces unexpected pattern, no auto-tuning) was implemented (commit `1dcf387`) and the calibration run was executed. **Both STOP-and-report conditions tripped on the calibration data**, surfacing a substrate-level finding that goes beyond preflight thresholds and has implications for the §8.4 verification + the curriculum's control-item framing.

**DINOv2 calibration outcome (single 3-call run):**

| item | room | DINOv2 3-call mean | per-call cosines |
|---|---|---:|---|
| Bed | Bedroom (control) | 0.9884 | 0.989 / 0.990 / 0.986 |
| **DiningTable** | **Bedroom (control)** | **0.9448** | 0.937 / 0.965 / 0.932 |
| Television | Bedroom (control) | 0.9945 | 0.997 / 0.993 / 0.993 |
| Dresser | LivingRoom (perturbed) | 0.9884 | 0.991 / 0.996 / 0.977 |
| Sofa | LivingRoom (perturbed) | 0.9757 | 0.962 / 0.987 / 0.978 |

- Bedroom mean: **0.9759** (below G_M2's 0.98 threshold — S2 tripped)
- LivingRoom mean: 0.9820
- Contrast: **−0.0061** (negative; below S1's 0.02 threshold — S1 tripped)

**DiningTable, a Bedroom control item, moves more in DINOv2 cosine under LivingRoom retexturing than either LivingRoom perturbed item.** This isn't metric noise — it's a substrate property. The DiningTable viewing pose in the seed-7 house has line-of-sight into the LivingRoom (corroborating the doorway-bleed hypothesis from the prior pixel-cosine runs), and DINOv2 representations are sensitive enough to the visible LivingRoom backdrop that the DiningTable's embedding shifts more when LivingRoom textures change than the LivingRoom items' own embeddings do.

Per the directive: no auto-tuning of G_M2 or G_M3. Stopping for review.

**The finding has implications beyond the preflight:**

The §8.4 verification on the actual Phase 2 stream (perturbed-item Stage B vs Stage A apex embedding gap, plus the differential against control items per the 2026-05-14 directive) also depends on the locality claim holding at the DINOv2 level. The numbers above predict that DiningTable Stage A vs Stage B in the collected stream will likely show *more* separation than Dresser or Sofa Stage A vs Stage B, because every Stage B loop fires a fresh `RandomizeMaterials` call that changes the LivingRoom backdrop visible from the DiningTable pose. The curriculum's "LivingRoom items widen into tubes; Bedroom items stay on Stage A lines as within-experiment controls" framing is therefore undermined for at least one of the controls.

This is an architectural decision for the experiment chat, not a CC-level autonomy question. Options I see:

(a) **Reframe the curriculum to acknowledge DiningTable as a non-clean control.** Keep Bed and Television as the within-experiment Bedroom controls; treat DiningTable as a "partially-perturbed-via-background" item and either report it separately or exclude it from the control comparison. Smallest change; preserves the rest of the design. The §8.4 differential metric (already wired in commit `1dcf387`) handles this naturally because control-item gaps are reported per-item.

(b) **Adjust the route's DiningTable viewing pose so its camera FOV does not include the doorway into the LivingRoom.** Re-derive `viewing_position` and/or `viewing_heading` for item 2 so the scene visible in the DiningTable frame is fully inside the Bedroom. Changes the substrate; requires re-running the session-4 calibration to verify within-loop continuity and apex bit-identicity properties still hold. Cleanest preservation of the curriculum's control framing but adds substrate work.

(c) **Switch house seed.** Pick a seed where the Bedroom and LivingRoom geometry does not produce inter-room line-of-sight from any chosen viewing pose. Most disruptive — would invalidate session-4's calibration artefacts and require fresh substrate verification. Only consider if (a) and (b) prove unworkable.

(d) **Accept the locality breach as a substrate property and reformulate the architectural claim.** The "LivingRoom-scoped perturbation affects only LivingRoom items at the encoder level" claim is empirically wrong for this house seed. Rather than re-engineering the substrate, document the finding and let the §8.4 verification report it explicitly; the architectural reading shifts from "locality" to "scoped-perturbation-with-known-leakage-pattern". The predictor still has a chance to learn the perturbed-item shapes; the within-experiment control story is just weaker.

I'd recommend **(a)**. The locality claim is broken specifically and predictably at DiningTable (the per-call DINOv2 numbers say so cleanly); reframing the curriculum's control set to {Bed, Television} preserves the design with minimal change. The §8.4 differential metric is already in place (commit `1dcf387`) and reports gaps per-item, so the reviewer can read both the full control set and the cleaned-up {Bed, Television} subset from the same encoded data.

(b) is the most principled but adds substrate work. (c) is most disruptive. (d) is the "accept reality" path; reasonable but loses the clean-control story.

**What is NOT decided autonomously:** lowering G_M2 below 0.98 to admit DiningTable's 0.945; widening G_M3's tolerance to admit a negative contrast; switching the preflight to "Dresser+Sofa only" without an explicit reviewer override. All of those would be tuning thresholds to fit and violate the directive.

The originally-planned session-5 STOP point (end of Phase 2 collection + encoding) is no longer the next stop — the Phase 2 collection is gated on whichever option (a–d) the reviewer chooses. After that decision, the preflight is either re-run (b/c), overridden in writing (a/d), or the substrate is changed (b/c). Then collection launches; then the §8.4 verification stops for review per the user's prior directive.

### (Earlier first STOP in session 5) — superseded

Resolved by the user on 2026-05-14 with option (a): recalibrate to three-criterion pixel-RGB gate. Implemented (commit `1da95ba`).

### (Earlier second STOP in session 5) — superseded by the third STOP above

The pixel-RGB three-criterion gate tripped on per-run material lottery + DiningTable doorway view-through after two re-runs (commits `1da95ba` for the recalibration, `1870c02` for the second-STOP HANDOFF entry). Resolved by the user on 2026-05-14 with option (a) from that round: switch the preflight to DINOv2-embedding contrast (G_M2=0.98 fixed, G_M3=40-60% of observed contrast, STOP-and-report on unexpected pattern, no auto-tuning). That switch was implemented (commit `1dcf387`) and the calibration tripped both STOP conditions — see the current "Next immediate action" block above.

---

### (Earlier session-4 STOP) — superseded by 2026-05-14 experiment-chat directive

Session 4's STOP listed a session-5 reviewer-action gate on variation strategy and three other open decisions. All four were resolved by the experiment chat on 2026-05-14:

1. **Variation strategy form** — jitter withdrawn entirely; phase structure is the variation source.
2. **Variation magnitude** — N/A (no jitter).
3. **Television-item anomaly** — accepted as item-specific quirk; recorded in known-substrate-properties.
4. **`transit → close_up` 1-frame duplication** — accepted as cosmetic (not blocking Phase 2 launch); fix in a polish commit at convenience or carry as documented quirk.

The full decision text is in the user's 2026-05-14 message that opened session 5. Session 5 implemented the resulting curriculum framing and the in-flight transition diagnostic.

---

### (Historical) Earlier session-4 STOP text follows below for the audit trail.

Session 4 implemented the continuous-motion substrate per the session-3 reviewer directive, ran a 5-loop calibration, and ran DINOv2 motion-continuity diagnostics. Two findings to review before the full Phase 2 collection begins:

1. **Within-loop motion-continuity PASSES.** All 255 consecutive close_up→close_up pairs and all 1,275 consecutive transit→transit pairs are non-bit-identical (cos < 0.9999 throughout). Mean consecutive cosine across the full 1,579-pair stream is 0.92. The 30-frame static dwell pattern is eliminated.

2. **Cross-loop apex comparison FAILS for 4 of 5 items** at the bit-identical level. Apex frames at items 1 (Bed), 2 (DiningTable), 3 (Dresser), and 4 (Sofa) have **cosine = 1.0000** across all 10 pairs (5 loops × choose 2). The reason is structural: the apex pose for each item is the same `viewing_position + heading=viewing_heading`, repeated each loop; AI2-THOR's frame rendering is deterministic on this stack. Item 5 (Television) is the lone exception (mean cosine 0.97, std 0.015, no bit-identical pairs) — likely TV display dynamics rendered by AI2-THOR at some non-deterministic level. The continuous-motion substrate fixes within-loop targets but does not, by itself, break across-loop bit-identicity.

This means the §2.2 "repetition tightens within-cluster representations" claim is *still* untestable on this substrate unless across-loop variation is introduced. **Proposed variation strategy (load-bearing decision the reviewer needs to sign off on before the full Phase 2 collection):** re-introduce per-frame pose jitter at a **smaller magnitude than the prior Stage 0b stability batch** (0.05 m position, 2° heading vs the prior 0.20 m / 10°). The motion now supplies the bulk of variation; jitter exists only to break the across-loop pose-determinism floor. Reviewer alternatives below.

Until reviewer signs off on the variation strategy, the trajectory implementation and calibration are committed but the full Phase 2 collection does not start. After sign-off, run a second calibration (5 loops) with jitter to verify cross-loop apex variation is now non-degenerate (target: mean cosine 0.92-0.98 across loops at the apex), then launch the full 65k-frame Phase 2 collection.

---

### (Historical) Earlier session-3 STOP — superseded by session 4 substrate change

The session-3 STOP listed three open decisions on G1.3 / G1.4 / S4. These are largely **superseded** by the session-4 substrate change:

- The Phase 1 substrate is now declared **substrate-degenerate** (per reviewer directive); its specific G1.3 / G1.4 / S4 results are not inherited as findings about the architecture and the v0 evidence base restarts at Phase 2 on the new substrate.
- The verified-working items from Phase 1 — predictor scaffolding works (no NaN/Inf, loss decreased monotonically, 21.5M params well within tolerance, gradient flow healthy), encoder pipeline works (DINOv2 deterministic, 100k frames all L2-normed), and cross-cluster discriminability rises with training (M3 trajectory 0.008 → 0.325) — remain valid.
- Phase 1 is not being re-run with the new substrate. The v0 evidence base starts at Phase 2.

For audit, the session-3 G1.3 / G1.4 / S4 disaggregations stay in `results/inner_pam_v0/phase1_main/gate_report.json` and the session-3 HANDOFF entry below documents them. They are not subject to a pending verdict any longer.

---

### Open decisions / proposals for the reviewer (session 4)

1. **Variation strategy.** Three candidate forms; recommend (a):
   - **(a) Per-frame independent jitter at 0.05 m / 2°** drawn fresh each frame. Simplest. Breaks across-loop bit-identicity directly: apex poses across loops would differ by ~0.05 m position and ~2° heading. ContinuousMotionExplorer needs a jitter parameter added; ~30 lines of code.
   - **(b) Per-loop pose offset.** One small `(dx, dz, dh)` offset drawn per loop, applied as a constant shift to all frames in that loop. More principled (whole loop traverses a slightly different path) but requires per-loop state in the explorer.
   - **(c) Hybrid (per-loop base + per-frame micro-noise).** Both signals.
2. **Acceptable jitter magnitude.** 0.05 m / 2° is a starting proposal. The prior stability batch (Stage 0b) used 0.20 m / 10°, but that was the only variation source. Now motion provides most of the variation; jitter only breaks the floor. The reviewer may want it even smaller (0.02 m / 1°) or larger (0.10 m / 5°).
3. **Television-item anomaly.** Item 5 shows non-zero across-loop variation (mean 0.97, std 0.015) without any explicit pose jitter — likely the AI2-THOR Television's rendered display has internal variation. Reviewer may want this investigated separately, or accept it as an item-specific quirk and treat it as another known property of the substrate.
4. **`transit → close_up` boundary frame duplication.** 8 of 24 transit→close_up consecutive pairs have cosine > 0.9999. This is because the final corner-rotation step in transit lands at exactly close_up_start with heading exactly viewing_heading, which is also the first close_up frame's pose — a 1-frame duplication at each phase boundary. Cosmetic; affects 0.5% of consecutive pairs. Fix candidates: drop the last transit-rotation step, or start the close-up one densification step further along. Reviewer call on whether to fix in session 4 or accept for now.
5. **Phase 2 frame budget.** 65k stays valid for the 316-frame loop: 65k/316 ≈ 205 collected loops, 195 trained loops (minus 10 held-out), comfortably into the 100+ rep bin with 95 reps inside. **Item 1 in the held-out region.** The last loop may end partial (depending on where 65k lands), giving (1,2) ~10 cue probes and the other four transitions ~9 each (same partial-loop pattern as Phase 1, plus or minus 1 depending on truncation point); resolves under the same not-a-bug logic as session 2.
6. **Checkpoint cadence for §4.6 recomputed against the 316-frame loop length** (table in the session-4 outcomes entry below). All five rep bins covered, 100+ bin gets 3 checkpoints inside.

Reviewer-action gate: when the variation strategy and magnitude are decided, I'll implement, re-run the 5-loop calibration with jitter, verify cross-loop apex variation is non-degenerate, then launch the full 65k Phase 2 collection.

1. **G1.4 verdict at the gated horizon (k=8).** Main wins at k=1 (mean_diff +0.039, p=1.7e-05) and at k=16 by rank (Wilcoxon p=0.004 even though mean_diff is −0.020). **Main loses at k=8** (mean_diff −0.132, Wilcoxon p=1.0). The mid-horizon failure is real, not an artefact of v1's wrong-shuffle. Diagnosis below points to **rank-512 limited predictor architecture** as a candidate: cosine at the cluster boundary is bounded by the output projection's column space, while squared-error (which the loss actually optimises) shows main beating shuffle decisively. Reviewer call: declare FAIL @ k=8 and pause to investigate, accept the rank-limited reading and recalibrate the gate to squared-error or a different cosine threshold, or proceed with a documented caveat.

2. **G1.3 verdict.** FAIL against the +0.3 absolute scaffolding threshold (separation cue−steady = −0.14, i.e., the wrong direction). But **main has structure shuffle doesn't**: shuffle's separation is 0.0002 (≈ zero, as expected from a temporally-destroyed control), while main's is −0.14 in a stable consistent direction. Main is also ~0.8 log_var more confident than shuffle on both probe types. Substrate-artefact diagnostic is below. Reviewer call: FAIL @ absolute scaffolding stands; PASS-at-relative-baseline depending on how the reviewer weighs "wrong-direction separation but real structure" vs "absolute spec direction".

3. **S4 quantitative thresholds need reviewer-authorised recalibration.** Shuffle did NOT collapse to the form S4 anticipated (`||μ|| < 0.15` AND `log σ² > 0.4`) but DID collapse to a different specific form: `||μ|| ≈ 0.75` AND `log σ² ≈ −7.48` — i.e., predicting the marginal-mean direction with low (calibrated-to-residual) variance. The empirical shuffle distribution is now visible; per instr §6.5 the thresholds are SCAFFOLDING explicitly subject to "recalibrate after observing the empirical shuffle distribution at end of Phase 1." Recommended new thresholds: `||μ|| > 0.6` OR `M3 sharpness < 0.05` (either captures the collapse-to-marginal-mean signature observed). Reviewer chooses the recalibration.

**Mechanical gates remain passed.** G1.1 PASS, G1.2 PASS. **G1.5 trajectory PASS @ scaffolding** (unchanged from session 2; main predictor's M3 sharpness 0.008 → 0.325, 8 of 9 transitions non-decreasing, floor 0.10 cleared).

**Phase 2 entry remains blocked** until the reviewer assigns G1.3 / G1.4 / S4 verdicts.

---

## Operational state (end of session 4)

- Working tree: clean modulo this HANDOFF entry + the session-4 commits (continuous-motion explorer, env wrapper, calibration script, analysis script, calibration data + report, doc updates). 25 commits expected on `main` after this session lands.
- Push hold: in effect.
- No running jobs.
- Phase 1 artefacts: unchanged from session 3 (substrate-degenerate baseline; not being re-run).
- Phase 2 substrate: new continuous-motion explorer + env wrapper at `src/env/continuous_motion_*.py`. 5-loop calibration data at `data/phase2_calibration/` (frames gitignored; annotations + embeddings gitignored per .gitignore rules; report committed). Full Phase 2 collection has NOT begun and will not begin until reviewer signs off on the variation strategy.

---

## Session 5 outcomes — 2026-05-14

**Goal.** Implement the curriculum framing the experiment chat approved on 2026-05-14 (phase structure as the variation source; no jitter; Stage A baseline in Phase 2 loops 1–30, Stage B from loop 31 onward; in-flight transition diagnostic with three SCAFFOLDING gates), commit the doc and code changes, run the Phase 2 preflight, then launch the 65k Phase 2 collection and DINOv2 encoding, STOP at the end of collection+encoding for review (training deferred). Encountered preflight FAIL on threshold-calibration grounds and stopped early instead.

### Documentation updates committed (commit `5104811`)

- **Spec §2.3** extended: continuous-training commitment carries across phase boundaries (predictor + bank not re-initialised at Phase 2 → Phase 3). The original user directive referenced "Spec §1.4" but the spec has no §1.4 (phase structure lives in the instructions); the carry-across-phases reaffirmation went into §2.3 where it fits the existing online-single-pass framing.
- **Instructions §1.3** cross-loop variation framing replaced. Jitter is ruled out; bit-identical Stage A apex frames are the curriculum's baseline state, not a substrate degeneracy.
- **Instructions §1.4** four locked decisions updated with the Stage A → Stage B → Stage C curriculum framing. Phase 2 internal structure recorded: loops 1–30 unperturbed (Stage A), loops 31+ with per-loop `RandomizeMaterials(inRoomTypes=["LivingRoom"])` (Stage B). Phase 2 starts from a freshly-initialised predictor (Phase 1 discarded); Phase 3 resumes from Phase 2's final checkpoint and bank state without reset.
- **Instructions §1.5** "no jitter, no per-frame perturbation, no Stage A inside Phase 3, predictor + bank carry across Phase 2 → Phase 3" added to the "What NOT to change" list.
- **Instructions §8.2** the persistence check (capture loop 1, run loop, capture loop 2 → expect identical) replaced with a per-loop re-application check that mirrors the new Phase 2 collection pattern (call N+1 produces fresh textures, including a third call to rule out a 2-state cycle).
- **Instructions §8.3** collection script takes `--perturbation_start_loop` (default 31); annotations carry `perturbation_active`. Frame budget table re-derived against the Stage A/B split (30 Stage A loops + ~165 Stage B trained loops, ~65 reps of margin past the 100+ bin).
- **Instructions §8.4** perturbation-effect check reframed: Stage B vs Stage A apex embeddings within Phase 2, NOT vs Phase 1 (which is on the substrate-degenerate baseline).
- **Instructions §8.5** Phase 2 training starts from a freshly-initialised predictor (Phase 1 discarded). Stale 458-frame-loop cadence corrected to the 316-frame schedule (1k / 2k / 4k / 6.5k / 10k / 15k / 20k / 30k / 40k / 55k / end).
- **Instructions §8.7** G2.2 and G2.3 reframed against the Stage A baseline inside Phase 2.
- **Instructions §8.7a** new: in-flight Stage A → Stage B transition diagnostic with three SCAFFOLDING gates:
  - **G2.T1** loss spike check: `max(post_loss[31..35]) > baseline_mean[25..30] * 3.0` (sign-safe form documented in the trainer; for non-positive baselines uses `+ (ratio - 1) * |baseline_mean|` so the directional meaning of "loss spiked upward" is preserved in the high-confidence regime).
  - **G2.T2** perturbed-item log_var widening: Dresser + Sofa mean log_var at end of loop 35 minus end of loop 30 must be ≥ 0.5.
  - **G2.T3** control-item log_var drift: Bed, DiningTable, Television each must have `|loop35 - loop30| ≤ 0.3` (any single item over the bound trips).

  Trip behaviour: trainer writes `transition_diagnostic_TRIPPED.txt`, sets `gate_tripped: true` in the JSON, exits non-zero, halts training. Pass behaviour: training continues, per-loop records keep appending to the diagnostic JSON through end of phase as record-only.
- **Instructions §9.1** Phase 3 starts immediately with perturbation active from loop 1; no internal Stage A baseline (would dilute the Stage C signal). Phase 3 resumes from Phase 2's final predictor checkpoint and bank state per spec §2.3.
- **Instructions §12 (scaffolding inventory)** stale "0.2 m / 10°" stability-batch jitter entry removed; new entries for `PHASE_2_PERTURBATION_START_LOOP = 31` and the three transition-diagnostic SCAFFOLDING thresholds.
- **research_operations_v1.md §15** strengthened with the "every inherited mechanism needs re-justification when the framing shifts" rule, capturing the 5-catch pattern across the v0 batch (Python version, embeddings coverage, static dwell, cross-loop apex determinism, jitter necessity).

### Code committed

- **`src/config.py`** (commit `08cce4f`). Added `PHASE_2_PERTURBATION_START_LOOP = 31`, the three transition-diagnostic SCAFFOLDING thresholds, perturbed-item and control-item vp_id tuples, baseline + post-onset loop windows. Removed `JITTER_POSITION_M` / `JITTER_HEADING_DEG`. `PhaseConfig.loop_length_estimate` default now 316 (continuous-motion substrate). `PHASE2.loaded_from_phase = None` (Phase 2 starts fresh).
- **`src/env/explorer_phase2.py`** (commit `08cce4f`). `Phase2RetextureEnv` wraps `ContinuousMotionEnv` and fires `RandomizeMaterials(inRoomTypes=["LivingRoom"], useTrainMaterials=True)` once at the start of every loop with `loop_index >= start_loop`. Records per-loop applied materials for the audit trail. Per-frame `perturbation_active_for_current_frame()` returns True from loop 31 onward. No jitter logic.
- **`src/trainer/online_trainer.py`** (commit `f94410e`). `TrainerConfig` gains seven new optional fields for the transition diagnostic. `OnlineTrainer.__init__` initialises per-loop aggregator dicts. The training loop accumulates per-loop loss and per-(loop, viewing_position_id) mean log_var (close-up frames only); flushes per-loop stats to `transition_diagnostic.json` as each loop completes; at the end of loop 35 evaluates the three gates; trips write a marker file and break out of the training loop with non-zero exit. Smoke-tested end-to-end on a synthetic 5-loop stream. 21 unit tests still pass.
- **`scripts/run_phase2_preflight.py`** (commit `057e711`). Five-check preflight per §8.2 against the seed-7 house. Writes a preflight report + JSON + side-by-side before/after frame captures.
- **`scripts/run_phase2_collect.py`** (commit `057e711`). 65k-frame Phase 2 collection on the continuous-motion substrate, wrapping `ContinuousMotionEnv` in `Phase2RetextureEnv`. Annotations carry `phase: "phase2"`, `perturbation: "livingroom_retexture"`, `perturbation_active: bool`. Per-loop materials written to `data/phase2_collection_metadata.json`.
- **`scripts/run_phase2_encode.py`** (commit `057e711`). DINOv2 encoding of the Phase 2 stream. §8.4 perturbation-effect check: Stage B vs Stage A Dresser-apex (and Sofa-apex) embeddings; within - cross cosine gap must exceed 0.05.
- **`scripts/run_phase2_train.py`** (commit `057e711`). Phase 2 main training entry point. Wires the in-flight transition diagnostic with the SCAFFOLDING thresholds. Committed but NOT launched this session (per the session-5 STOP point).

### Phase 2 preflight recalibration outcome — second STOP (commit `1da95ba`)

After the user authorised option (a) (drop absolute LivingRoom + per-loop-re-application gates; replace with `G_M1` mechanism / `G_M2` Bedroom > 0.97 / `G_M3` contrast ≥ 0.02), the preflight was recalibrated and re-run.

**Run 2 (single-call recalibrated, 2026-05-14 05:59):** `G_M3` tripped at contrast +0.012. Sofa's single-call before-vs-after cosine was 0.996 on this run — the random material draw happened to land on a near-identical-looking texture, pulling the LivingRoom mean up to 0.979.

**3-call averaging fix.** The preflight already makes three `RandomizeMaterials` calls (calls 1/2/3) for the record-only per-loop re-application data. Averaging each item's before-vs-after cosine across the three random draws is the smallest fix that removes the per-run material lottery — and it matches what the actual Phase 2 collection does (fresh material draw per loop). Implemented in the same commit as the recalibration.

**Run 3 (3-call-averaged, 2026-05-14 06:01):** `G_M3` still tripped, contrast +0.008. The per-call breakdown surfaced two structural issues with flat-RGB cosine as the metric:

(1) Sofa is genuinely less affected by the available `useTrainMaterials=True` pool than Dresser is — across 3 calls in run 3 it landed at cosines 0.995 / 0.994 / 0.983, averaging 0.991. Dresser averages 0.966 across the same 3 calls. The asymmetry between the two LivingRoom items is large enough that averaging across them does not produce a clean LivingRoom mean.

(2) DiningTable (a Bedroom control item) showed a 3-call mean of 0.976 — *more change than Sofa, a LivingRoom perturbed item.* The most plausible explanation is that the DiningTable viewing pose looks through into the LivingRoom and the re-textured LivingRoom furniture changes background pixels in the DiningTable frame. (The seed-7 ProcTHOR house has the Bedroom adjacent to the LivingRoom; the route's DiningTable viewing_position and viewing_heading are both in the Bedroom but the camera FOV may catch the doorway.)

The mechanism is doing the right thing — Dresser moves consistently (3-6% per draw across all three runs); the LivingRoom mean before-vs-after cosine is always lower than the Bedroom mean (positive contrast in all three runs). But the magnitude of the contrast is hostage to the material lottery on Sofa and the view-through bleed on DiningTable, both of which are properties of the pixel-cosine metric rather than the underlying perturbation mechanism.

The "Next immediate action" section above presents three options for the reviewer; I recommend (a) — switch the preflight to a DINOv2-embedding-distance contrast that uses the same metric the load-bearing §8.4 verification uses. (b) — override the FAIL with written justification — is also defensible.

### Phase 2 encode script update — docstring (commit `1da95ba`)

`scripts/run_phase2_encode.py` docstring updated to describe the **absolute + differential** metric reporting the experiment chat requested for the §8.4 verification: alongside the gated perturbed-item gap (within - cross) on Dresser and Sofa, the same Stage B vs Stage A comparison is computed on control items (Bed, DiningTable, Television). The "contrast" = perturbed_mean_gap − control_mean_gap is the load-bearing differential read. Implementation of the contrast computation deferred until after the preflight question resolves; the current encode script still gates on the perturbed-item absolute gap only.

### DINOv2-contrast preflight calibration outcome — third STOP (commit `1dcf387`)

Per the user's authorisation of option (a) (switch the preflight to DINOv2-embedding contrast), the preflight was rewritten to encode all 20 captured frames (5 items × {before, call1, call2, call3}) through frozen DINOv2-large and compute the contrast gate at the embedding level. Thresholds reviewer-authorised:

- G_M2 = 0.98 (fixed; Bedroom items should be near-identical in DINOv2 space if locality holds)
- G_M3 = 0.5 × observed_contrast (midpoint of the 40–60% range)
- S1 contrast < 0.02 → STOP-and-report
- S2 bedroom_mean < 0.98 → STOP-and-report

The calibration run executed cleanly (encoder norm check passed; mechanism G_M1 passed). Per-item DINOv2 3-call means:

| item | room | DINOv2 3-call mean |
|---|---|---:|
| Bed | Bedroom | 0.9884 |
| DiningTable | Bedroom | **0.9448** |
| Television | Bedroom | 0.9945 |
| Dresser | LivingRoom | 0.9884 |
| Sofa | LivingRoom | 0.9757 |

Bedroom mean = 0.9759 (< 0.98, **S2 trips**). LivingRoom mean = 0.9820. Contrast = **−0.0061** (negative; < 0.02, **S1 trips**).

The dominant signal is DiningTable's strong DINOv2 shift under LivingRoom retexturing. DINOv2 is sensitive enough to the LivingRoom backdrop visible through the doorway from the DiningTable viewing pose that DiningTable behaves more like a "perturbed" item than a control. Dresser and Sofa show smaller shifts than DiningTable does.

Per the directive: no auto-tuning of thresholds. The finding is a substrate / house-geometry issue with implications for the §8.4 verification + the curriculum's control-item framing, which the experiment chat needs to weigh in on. Four options laid out in the Next-immediate-action section above.

### Phase 2 encode script — absolute + differential metrics (commit `1dcf387`)

`scripts/run_phase2_encode.py` extended to compute both the absolute perturbed-item Stage B vs Stage A gap (gated; passes if Dresser and Sofa each have within - cross > 0.05) and the differential control-item gap (record-only; same calculation applied to Bed, DiningTable, Television). The contrast = mean(perturbed gaps) − mean(control gaps) is reported in the JSON for the experiment-chat review at the §8.4 STOP point. With the DINOv2 calibration prediction that DiningTable's DINOv2 representation is sensitive to LivingRoom retexturing, the differential metric is likely to show DiningTable's gap on the actual collected stream being non-trivial — the reviewer will see this directly when §8.4 runs.

### Phase 2 preflight outcome — FAIL (initial run; superseded by the recalibration above)

Preflight ran at 2026-05-14 05:26 UTC; log at `logs/phase2_preflight_20260514_052627.log`. Process completed cleanly, all five checks ran, three reported FAIL against my chosen thresholds. Empirical numbers are in `results/inner_pam_v0/phase2_preflight/preflight_report.{md,json}`; side-by-side frame captures at `results/inner_pam_v0/phase2_preflight/frames/`.

**Verdicts and observed cosines:**

| check | verdict | observed | criterion |
|---|---|---|---|
| 1 action_exists | PASS | RandomizeMaterials returns success | API verified |
| 2 per_loop_re_application | FAIL | 0.974 / 0.982 / 0.981 / 0.996 | all four < 0.95 |
| 3 perturbation_locality | FAIL | Bed=0.996, DiningTable=0.983, TV=0.995 | all ≥ 0.999 |
| 4 livingroom_visually_changed | FAIL | Dresser=0.969, Sofa=0.947 | both < 0.9 |
| 5 session_determinism | PASS (per-run) | sessions A↔B cosine 0.984 / 0.988 | accept per-run |

**Reading.** The mechanism is working — LivingRoom items move ~3.3% more in pixel cosine than Bedroom items (LivingRoom mean 0.958 vs Bedroom mean 0.991), the right direction. Per-loop re-application produces visibly different textures (cosines drop to 0.974, not 1.0). My thresholds were calibrated for a "big" pixel-space change; the actual change `RandomizeMaterials` produces at 300×300 resolution is smaller because most of each frame's pixels are unchanged geometry, walls, lighting, and background. The §8.4 perturbation-effect check on DINOv2 embeddings (run at encoding time, not preflight time) is the load-bearing verification; pixel cosines are just an early-warning sanity check.

**Stopped before launching the Phase 2 collection** per `CODING_STANDARDS.md` §9.4 ("identify a substantially cleaner approach — don't implement silently"). The clean recalibration of the preflight thresholds (whether by contrast-criterion or by switching the preflight to DINOv2-embedding cosines) is the reviewer's call.

### Working-tree contents committed this session

| commit | scope | files |
|---|---|---|
| `5104811` | curriculum framing across docs | spec §2.3; instructions §1.3, §1.4, §1.5, §8.2, §8.3, §8.4, §8.5, §8.7, §8.7a, §9.1, §12; research_operations_v1.md §15 |
| `08cce4f` | phase 2 env wrapper + config | `src/config.py`, `src/env/explorer_phase2.py` |
| `f94410e` | trainer transition diagnostic | `src/trainer/online_trainer.py` |
| `057e711` | phase 2 scripts | `scripts/run_phase2_preflight.py`, `scripts/run_phase2_collect.py`, `scripts/run_phase2_encode.py`, `scripts/run_phase2_train.py` |
| `f447b01` | first session-5 HANDOFF + initial preflight artefacts | `HANDOFF.md`, initial `results/inner_pam_v0/phase2_preflight/preflight_report.{md,json}`, 14 side-by-side preflight frames |
| `1da95ba` | pixel-RGB preflight recalibration + 3-call averaging + encode docstring | `scripts/run_phase2_preflight.py`, `scripts/run_phase2_encode.py`, updated preflight report + frames |
| `1870c02` | session-5 second-STOP HANDOFF entry | `HANDOFF.md` |
| `1dcf387` | DINOv2-contrast preflight + encode differential metrics + calibration artefacts | `scripts/run_phase2_preflight.py`, `scripts/run_phase2_encode.py`, updated preflight report + frames |
| `c71867f` | session-5 third-STOP HANDOFF entry | `HANDOFF.md` |
| `3bd341b` | pose search + adjusted DT pose + v2 calibration loop-length STOP | `src/env/material_perturbation_probe.py`, `scripts/run_phase2_extended_diagnostic.py`, `scripts/run_phase2_pose_search.py`, `scripts/run_phase2_motion_continuity_check.py`, `data/route_phase2.json`, `results/inner_pam_v0/phase2_{extended_diagnostic,pose_search,motion_continuity}/`, `results/phase2_calibration_v2/calibration_summary.json` |
| `444d0ae` | session-5 fourth-STOP HANDOFF entry | `HANDOFF.md` |
| `71f7693` | §4.6 cadence + §8.3/§9.3 tables + config PHASE_2_3_CKPT_STEPS (v2 substrate) | `WEFT_INNER_PAM_v0_EXPERIMENT_INSTRUCTIONS.md`, `src/config.py` |
| `b305aaa` | spec §5.6 per-item stability gate + Phase 2 scripts default to `route_phase2.json` | `WEFT_INNER_PAM_v0_Spec.md`, `scripts/run_phase2_preflight.py`, `scripts/run_phase2_collect.py` |
| `21829f3` | research_operations §15 two new universal principles | `research_operations_v1.md` |
| `45aca0e` | analyse script verdict refactor + v2 within-loop motion-continuity PASS | `scripts/run_phase2_calibration_analyse.py`, `results/phase2_calibration_v2/continuity_report.json` |
| `4b1408c` | adjusted-geometry preflight (G_M2 PASS, S1 trip on contrast magnitude) | `results/inner_pam_v0/phase2_preflight/` |
| `ceee028` | session-5 fifth-STOP HANDOFF entry | `HANDOFF.md` |
| `fc95301` | modified preflight gate (S1/S2 dropped, encode differential go/no-go) + adjusted-geometry preflight PASS | `scripts/run_phase2_preflight.py`, `scripts/run_phase2_encode.py`, `WEFT_INNER_PAM_v0_EXPERIMENT_INSTRUCTIONS.md`, `results/inner_pam_v0/phase2_preflight/` |
| `e3feaa2` | Phase 2 launch marker (aborted 4 minutes later at frame 6,800 per the sixth STOP) | `HANDOFF.md` |
| `29478a2` | trajectory diagnostic + recommended fix (sixth STOP) | `scripts/run_phase2_trajectory_diagnostic.py`, `results/phase2_calibration_v2/trajectory_diagnostic.json`, 29 discontinuity contact sheets |
| pending this entry | session-5 sixth-STOP HANDOFF entry | `HANDOFF.md` |

### Pose search + motion-continuity outcome (commit `3bd341b`)

Extended diagnostic with 6 RandomizeMaterials draws confirmed DiningTable as the sole affected Bedroom item (DINOv2 mean 0.9564, std 0.0091 — consistent leak, not lottery noise). Bed and Television sit cleanly above 0.98.

Pose search generated 12 candidate angles around DT's centroid, snapped each to the nearest NavMesh-reachable grid point at the right distance, screened for teleport-ok, then ran a 3-draw DINOv2 stability check. Eleven candidates screened in; only **DiningTable_h118** (heading 117.6°, position (8.25, 4.75)) passed the 0.98 stability bar (mean 0.9827, std 0.0082). The agent approaches DT from the northwest looking southeast; the original pose approached from the southeast looking northwest. The doorway to the LivingRoom is now behind the agent rather than in the FOV.

Motion-continuity check on the top 3 candidates: h118 sweep consec cosine range [0.7645, 0.9785], 0 bit-identical pairs — clean. h147 and h206 also motion-continuity-clean but failed the stability threshold.

`data/route_phase2.json` writes the adjusted route with DT's new pose, viewing_distance_m updated to 1.682 (from 1.754, since the snapped position is slightly closer to the centroid than the original target), plus provenance fields recording the original pose, the workflow, and the metric rationale.

### v2 calibration outcome — fourth STOP (commit `3bd341b`)

5-loop calibration on the adjusted route produced **1800 frames over 5 loops = 360 frames/loop**, vs the 316 baseline. Loop length shift = **+13.9%**, exceeding the reviewer's 10% STOP threshold.

The shift is structural to the new DT position. Both DT-adjacent transits change:
- Bed (11.75, 2.25) → DT new (8.25, 4.75): straight-line ~4.3 m vs old ~0.9 m
- DT new (8.25, 4.75) → Dresser (6.0, 3.0): straight-line ~2.85 m vs old ~5.0 m

Close-up frames per item are unchanged (~11 per loop per item). The extra ~44 frames/loop are almost all in transit: 305 transit frames/loop in v2 vs ~260 in v1.

Within-loop motion-continuity on the adjusted geometry was NOT checked — stopped at the loop-length boundary per the directive's "if it shifts more than 10%, stop." Preflight on the adjusted geometry was also NOT re-run for the same reason.

### Reviewer-action items before session 6

1. **Approve the trajectory-bug fix** (3 changes to `src/env/continuous_motion_explorer.py` per the diagnostic's `recommended_fix` field — drop input y in `_teleport`, snap to NavMesh floor queried at init, optionally add `standing=True`). The fix is small and conservative; it does not change the trajectory geometry in x/z, only normalises the agent base y to floor level.
2. After the fix lands:
   - Re-run the session-4 5-loop calibration (this is required — every prior rendered frame has the camera-elevation bug; we need fresh frames at corrected elevation to verify the fix and to populate the v3 substrate calibration).
   - Re-run the trajectory diagnostic on the v3 calibration to confirm the y-bob is gone (single y value, no displacement outliers at 0.894 m).
   - Re-run the preflight on `route_phase2.json` (gates G_M1 / G_M2 / G_M3 against the corrected-elevation captures).
   - Re-verify the DT_h118 pose stability — the pose-search verdict was made on bird's-eye captures; at correct eye-height the FOV looks different and the DINOv2-stability ranking might shift. Pose-search can be re-run cheaply (one AI2-THOR session, ~2-3 minutes).
3. After verifications pass: re-launch the 65k Phase 2 collection; encoding + §8.4 verification (absolute + differential metrics, threshold 0.01 SCAFFOLDING) follows; then STOP for review of §8.4 results before Phase 2 training.

### Earlier fifth-STOP reviewer-action items (superseded)

The fifth-STOP question (whether to recalibrate S1, override, or fall back) was resolved by the user's "modified (i)" authorisation 2026-05-14 (drop S1, keep G_M2 and G_M3, promote §8.4 differential to go/no-go). That decision is implemented in commit `fc95301` and remains valid post-fix — the modified gate is independent of the camera-elevation bug.

### Sixth STOP — substrate y-bob trajectory bug surfaced; Phase 2 launch aborted (commits `e3feaa2` → `29478a2`)

After the modified-gate authorisation (commit `fc95301`) cleared the preflight and the 65k Phase 2 collection launched at 2026-05-14 03:49:32Z (commit `e3feaa2`, PID 109417), the reviewer reported two visual concerns from inspection of v2 calibration renders: (1) camera height appears to bob unexpectedly, (2) agent appears to jump through walls. Phase 2 was paused at frame 6,800 (~4 min into the 60-min run) for a targeted diagnostic.

Trajectory diagnostic (committed `29478a2`):
- Diagnostic 1 (per-frame y): two unique y values (0.0065 floor + 0.9010 eye-height); 0.8945 m range; the agent base oscillates between values at every close_up↔transit phase boundary.
- Diagnostic 2 (3D displacement): 29 outliers at exactly 0.8945 m with dx=dz=0; no horizontal wall-jumps anywhere in the histogram.
- Diagnostic 3 (contact sheets): 29 4-frame strips confirm close-up frames render at bird's-eye camera elevation (~1.8 m) and transit frames at normal eye-height (~0.9 m).
- Root cause: `forceAction=True` in `ContinuousMotionExplorer._teleport` (line 426) bypasses AI2-THOR's floor-snap validation; the agent base lands at whatever y is supplied. Present since session-4 commit `ec172c7` — not a session-5 leak.

Recommended fix (3 lines in the explorer; details in trajectory_diagnostic.json's `recommended_fix` field). Downstream invalidation scope: all rendered frames to date have the elevation artefact, but most analyses are computed on within-phase pairs or apex-only frames where y is constant; the headline impact is that all close-up framings have been bird's-eye rather than eye-height. Phase 2 collection re-runs after the fix lands.

Aborted Phase 2 collection (6,800 frames in `data/phase2_frames/`, partial annotations) cleaned up — gitignored anyway.

### v2 substrate doc + verification pass — fifth STOP (commits `71f7693` → `4b1408c`)

After the user authorised option (a) (accept +13.9% loop length, hold 65k budget, recalibrate arithmetic transparently), the doc pass (§4.6 / §8.3 / §9.3 / §1.3 / spec §5.6 / research_operations §15) landed across `71f7693`, `b305aaa`, `21829f3`. The analyse-script verdict was refactored to the curriculum-aligned form (in-motion pairs gated; boundary cosmetic + cross-loop apex informational) in `45aca0e` along with the v2 calibration analyse: within-loop motion-continuity **PASS** (0 bit-identical in close_up→close_up and 0 in transit→transit). Preflight on adjusted geometry in `4b1408c`: G_M2 PASSes at Bedroom mean 0.9822 (vs 0.9759 at the original DT pose — the locality fix worked); G_M3 trips S1 at contrast +0.0062 (right direction, below the 0.02 SCAFFOLDING threshold guess).

The substrate-fix-and-verification cycle is complete. The remaining decision is the S1 threshold (recommended option (i) above: recalibrate from empirical distribution).

### Operational state (end of session 5)

- Working tree: clean modulo this HANDOFF entry. 36 commits on `main` after this entry lands (15 session-5 commits + the pending HANDOFF entry).
- Push hold: in effect.
- No running jobs (Phase 2 collection was killed at frame 6,800 per the sixth STOP).
- Phase 1 artefacts: unchanged from session 4 (substrate-degenerate baseline; not re-run).
- Phase 2 substrate: adjusted route at `data/route_phase2.json` is the active configuration. v2 calibration analyse confirmed within-loop motion-continuity PASS at the curriculum-aligned verdict; trajectory diagnostic revealed the y-bob camera-elevation bug (sixth STOP). Preflight at adjusted geometry passes the modified gate (G_M1+G_M2+G_M3, S1/S2 dropped). The explorer's `_teleport` is awaiting the floor-snap fix; once fixed, v3 calibration + re-run preflight + Phase 2 launch follow.
- Aborted Phase 2 partial collection (6,800 frames) cleaned up from `data/phase2_frames/` and `data/phase2_annotations.jsonl` (both gitignored).

---

## Session 4 outcomes — 2026-05-13

**Goal.** Implement the continuous-motion substrate per the session-3 reviewer directive ("the agent moves through each loop as one continuous trajectory with no zero-velocity frames"), run a 5-loop calibration, verify motion-continuity with DINOv2 diagnostics, propose a variation strategy informed by the empirical findings, update the spec and instructions to lock the new substrate in, and STOP for review before any full Phase 2 collection.

### Trajectory design — `ContinuousMotionExplorer`

New explorer at `src/env/continuous_motion_explorer.py` replaces `FurnitureRouteExplorer`. State machine has two phases per item: `close_up` (continuous motion through the item) and `transit` (continuous motion between items).

**Close-up segment** (per item N):
- Direction: perpendicular to `viewing_heading_N` (CCW from forward in the top-down screen sense, consistent across all 5 items so the item visually slides in a consistent direction).
- Endpoints: `viewing_position_N ± (close_up_length_m / 2) * perpendicular_unit_ccw`. With `close_up_length_m = 2.0 m` (SCAFFOLDING), endpoints are at ±1.0 m from the viewing position.
- Heading: locked at `viewing_heading_N` throughout the close-up (so the item enters from one side of the frame, centres at the apex, slides out the other side).
- Densification: 0.20 m steps (SCAFFOLDING), yielding ~10-12 frames per close-up.
- Apex frame: the densified step closest to `viewing_position_N` is tagged `close_up_apex_flag = True`.

**Transit segment** (item N → N+1):
- NavMesh-planned path from `close_up_end_N` to `close_up_start_{N+1}` (different from the old explorer's viewing_position → viewing_position transit).
- Densified at 0.20 m steps + corner rotations at 5° step. Same mechanism as the prior FurnitureRouteExplorer.
- Heading: along walking direction within each NavMesh segment; rotates at corners; final rotation aligns to next item's viewing_heading.

**No static dwell at any pose.** Every consecutive frame pair has a non-zero pose delta.

### 5-loop calibration

`scripts/run_phase2_calibration_collect.py` ran 5 loops with the new explorer; no perturbation, no jitter.

| metric | value |
|---|---:|
| frames collected | **1,580** |
| loops | **5** |
| **frames per loop** | **316** |
| wall-clock | 89.5 s (~17.7 fps) |
| close-up frames per item per loop | 11, 12, 11, 11, 11 (avg ~11) |
| transit frames per loop | 260 |
| transitions planned (5 loops × 5 transitions) | 25 |
| transitions using NavMesh fallback | 10 of 25 (40 %) |
| teleport failures | 0 |

**316 frames/loop** is higher than the 200-250 target the reviewer flagged as a starting point, but the rep-bin coverage arithmetic (below) still works at the 65k Phase 2 budget with comfortable margin. Tunable in the v0 SCAFFOLDING inventory; the close-up length (2 m) or density (0.20 m) can be reduced if the reviewer wants the loop shorter.

### DINOv2 motion-continuity diagnostic

`scripts/run_phase2_calibration_analyse.py` encoded all 1,580 frames via the verified DINOv2 protocol and computed consecutive-frame cosines, disaggregated by phase, plus same-item cross-loop apex comparisons.

**Embedding sanity:** all 1,580 frames have L2 norms in [1−1e-5, 1+1e-5]. ✓

**Consecutive-frame cosines (1,579 pairs):**

| phase pair | n | mean | std | min | max | bit-identical (>0.9999) |
|---|---:|---:|---:|---:|---:|---:|
| close_up → close_up | 255 | **0.9304** | 0.126 | 0.315 | 0.991 | **0** ✓ |
| transit → transit | 1,275 | **0.9202** | 0.106 | 0.107 | 0.992 | **0** ✓ |
| close_up → transit | 25 | 0.8034 | 0.151 | 0.616 | 0.985 | 0 |
| transit → close_up | 24 | 0.9232 | 0.075 | 0.802 | 1.000 | **8** ⚠ |
| **aggregate** | 1,579 | 0.9201 | 0.111 | 0.107 | 1.000 | **8** |

The 8 bit-identical pairs are all `transit → close_up`, at the boundary between transit and close-up: the final corner-rotation step of transit lands at exactly `close_up_start` with heading exactly `viewing_heading`, which is also the first close-up frame's pose. **Cosmetic 1-frame duplication at the boundary; 0.5 % of all pairs.** Open decision (4) above proposes a fix.

**Within-loop continuity verdict: PASS.** Zero bit-identical pairs in close_up→close_up or transit→transit (the two "during motion" categories). The 30-frame static-dwell pattern is eliminated.

**Cross-loop apex comparison (5 apex frames per item × 10 pairs = 10 per item):**

| item | object | n pairs | mean cosine | std | bit-identical (>0.9999) |
|---:|---|---:|---:|---:|---:|
| 1 | Bed | 10 | **1.0000** | 0.000 | **10/10** ✗ |
| 2 | DiningTable | 10 | **1.0000** | 0.000 | **10/10** ✗ |
| 3 | Dresser | 10 | **1.0000** | 0.000 | **10/10** ✗ |
| 4 | Sofa | 10 | **1.0000** | 0.000 | **10/10** ✗ |
| 5 | Television | 10 | 0.9695 | 0.016 | **0/10** ✓ |

**Cross-loop apex verdict: FAIL on items 1-4.** Apex poses across loops are bit-identical, because each loop visits the same `viewing_position + viewing_heading` and AI2-THOR renders deterministically on this stack (confirmed by the substrate-verification batch's session-1 consistency check). Item 5 is the lone exception with non-zero across-loop variation, likely because the Television's rendered display has internal dynamics that AI2-THOR doesn't make deterministic — an item-specific quirk, not a designed feature.

**Implication.** The continuous-motion substrate fixes within-loop static dwell (the original session-3 finding) but does NOT, by itself, break across-loop pose-determinism. Any M3 cluster-sharpness measurement that compares predictor outputs across loops at the same pose (the way Phase 1's M3 worked) will still be substrate-floored at cosine = 1.0 within-cluster.

This is a partial substrate fix. Full resolution requires re-introducing across-loop variation, per the variation-strategy proposal above.

### Proposed variation strategy

**Recommendation: per-frame independent jitter at 0.05 m / 2°** (option (a) above). Rationale:

- **0.05 m position jitter** is much smaller than the 0.20 m densification step, so it doesn't dominate consecutive-frame motion; it just adds enough offset to break the across-loop pose-determinism floor.
- **2° heading jitter** is small enough not to swing the item out of frame at the apex (where the item is ~1.75 m from the agent).
- **Per-frame independent draws** (each frame's jitter is fresh, seeded RNG) means every frame has a unique offset. Cross-loop apex frames at item N would differ by ~0.05 m / 2° drawn from independent distributions, producing non-bit-identical embeddings.

Implementation cost: ~30 lines in `ContinuousMotionExplorer` (analogous to the prior `_apply_jittered_teleport` in `FurnitureRouteExplorer`, but no fallback ladder needed at the smaller magnitude — NavMesh tolerance should accept ±0.05 m at most poses).

Alternative options were considered (per-loop offset, hybrid); recorded in the "Open decisions" list above. The reviewer chooses.

### Recomputed checkpoint cadence — §4.6 update for 316-frame loop

The prior §4.6 cadence was derived from a 458-frame loop. The new substrate has 316-frame loops, so the phase-relative-step → rep-count map shifts:

| bin (perturbed-shape rep count) | first frame into phase | last frame into phase |
|---|---:|---:|
| 1–5 | 316 | 1,580 |
| 6–19 | 1,896 | 6,004 |
| 20–50 | 6,320 | 15,800 |
| 51–99 | 16,116 | 31,284 |
| 100+ | 31,600 | (end at ~61,840) |

**Proposed new checkpoint schedule** (phase-relative steps): **1,000 / 2,000 / 4,000 / 6,500 / 10,000 / 15,000 / 20,000 / 30,000 / 40,000 / 55,000 / end** — 10 checkpoints plus end-of-phase. Coverage:

| step | loops elapsed | bin |
|---:|---:|---|
| 1,000 | 3.2 | 1-5 ✓ |
| 2,000 | 6.3 | 6-19 ✓ |
| 4,000 | 12.7 | 6-19 |
| 6,500 | 20.6 | 20-50 ✓ |
| 10,000 | 31.6 | 20-50 |
| 15,000 | 47.5 | 20-50 |
| 20,000 | 63.3 | 51-99 ✓ |
| 30,000 | 94.9 | 51-99 |
| 40,000 | 126.6 | 100+ ✓ |
| 55,000 | 174.1 | 100+ |
| end | 195.7 | 100+ |

All five bins covered; 100+ bin gets three checkpoints. Cadence updated in instr §4.6 in the same commit as the substrate change.

### Frame budget — §8.3 / §9.3 update

Phase 2/3 budgets stay at 65k each. At 316 frames/loop:

| budget | collected loops | trained loops (−10 held-out) | last bin reached |
|---|---:|---:|---|
| 65k | ~205.7 | ~195 | 100+ with 95 reps inside |

Comfortable margin. Updated §8.3 / §9.3 derivation tables in the same commit.

### Phase 1 disposition

Per reviewer directive: Phase 1's substrate is declared substrate-degenerate. Its findings — predictor scaffolding works, encoder pipeline works, parameter counts within tolerance, gradient flow healthy, M3 cross-cluster discriminability rises with training — are kept as substrate-pipeline-validation evidence. Its substrate-degenerate results (G1.3 inversion, G1.4 k=8 dip, S1-S4 sanity check verdicts, variance-saturation patterns) are not inherited as findings about the architecture and are not used as priors for Phase 2 interpretation. **Phase 1 is not being re-run with the new substrate.** The v0 evidence base starts at Phase 2.

### Drift-detection note added to research_operations_v1.md §15

The 30-frame static dwell survived three drafts of the v0 experiment instructions, an adversarial review pass, and session-1's CC pre-flight before the session-3 reviewer asked "what is dwell?" and the substrate-architecture mismatch surfaced. Added as a universal drift check:

> Re-read foundational substrate assumptions whenever the architectural framing shifts. Inherited collection parameters from prior projects are SCAFFOLDING by default and require explicit re-justification against the current architecture's claims.

### Working-tree contents committed this session

| commit | scope | files |
|---|---|---|
| `feat(env): continuous-motion explorer + env wrapper` | new substrate | `src/env/continuous_motion_explorer.py`, `src/env/continuous_motion_env.py`, `src/env/procthor_house.py` (copy from previous repo) |
| `feat(scripts): phase2 calibration collect + analyse` | calibration tooling | `scripts/run_phase2_calibration_collect.py`, `scripts/run_phase2_calibration_analyse.py` |
| `exp(calibration): 5-loop continuous-motion calibration + DINOv2 continuity report` | calibration data | `results/phase2_calibration/calibration_summary.json`, `results/phase2_calibration/continuity_report.json` |
| `docs(spec/instructions): continuous-motion substrate change` | doc updates | spec §1.3, §2.3; instructions §0/§1.3, §1.5, §4.6, §8.3, §9.3; research_operations_v1.md §15 |
| `docs(handoff): session 4 outcomes — STOP for review` | this entry | `HANDOFF.md` |

---

## Session 3 outcomes — 2026-05-13

**Goal.** Resolve the session-2 STOPs on the reviewer's protocol: (1) re-implement shuffle per spec §10.1's "temporal structure destroyed" rationale, (2) re-train shuffle and re-compute G1.4 + S1-S4, (3) augment G1.3 with main-vs-shuffle log_var comparison, (4) document substrate-artefact diagnostic for G1.3, M3 cross-cluster framing, and cue-probe count confirmation. Pause again for review before any Phase 2 step.

### Cue-probe count: confirmed 46 is the natural maximum, not a bug

After the session-2 cue-probe fix (commit `ea84df1`), 46 cue probes are constructed from the held-out region. The expected ceiling is 10 held-out loops × 5 transitions = 50; the actual ceiling is **(1→2): 10, (2→3): 9, (3→4): 9, (4→5): 9, (5→1): 9 = 46**.

Reason: the seed-7 stream has 100,000 frames over 219 loops (loop_idx 0..218), but loop 218 is **partial** — it completes the dwell at item 1 (Bed) and starts the transit toward item 2 (DiningTable), but doesn't finish the loop. So in the held-out region (loops 209–218), transition (1, 2) has 10 valid windows (one per loop including the partial one) but the other four transitions have only 9. Not a bug.

**Confirmation that all gate metrics used the 46-probe set:** the gate_report.json was first produced (and re-produced this session) AFTER the cue-probe fix commit `ea84df1`. The build_probes call inside the gate report uses the fixed build_cue_probes implementation. No stale 9-probe numbers were used in any reported gate verdict.

### Shuffle re-design — spec §10.1 / §6.3 / §7.5

Replaced the session-2 visit-order-only shuffle with the spec-correct implementation. Single commit `a24d92c`:

- `scripts/run_phase1_shuffle.py`: permutation seed=0 applied via `np.random.default_rng(0).permutation(N_train)` to **`embeddings[:N_train]` AND `annotations[:N_train]` in lockstep** before training begins. Held-out region `[N_train, N)` is preserved unshuffled.
- `src/trainer/online_trainer.py`: `shuffle_seed` parameter removed from `TrainerConfig`. The trainer always traverses its input stream in sequential order. Shuffle is solely a property of the input stream the trainer receives.
- `WEFT_INNER_PAM_v0_EXPERIMENT_INSTRUCTIONS.md` §7.5: wording tightened from "applies permutation to training indices" to "applies `np.random.default_rng(0).permutation(N_train)` to the training portion of the embedding stream itself" with a paragraph documenting why the earlier wording was ambiguous.
- `.gitignore`: `results/**/ckpt_*/` (bank-state dirs are large binary; not part of audit trail).

First 5 permuted indices applied: `[62740, 36416, 33605, 36404, 55777]` — confirms embeddings at positions 0..4 in the new stream now hold the original frames 62740, 36416, 33605, 36404, 55777 (random unrelated frames). Window construction `embeddings[t-W+1 : t+1]` then yields 16 random unrelated embeddings, destroying temporal structure at source.

### Phase 1 shuffle v2 — re-train

| metric | v1 (visit-order, session 2) | v2 (spec-correct, this session) |
|---|---:|---:|
| wall-clock | 862.7 s | 871.4 s |
| gradient steps | 95,660 | 95,691 |
| final mean_loss_last_1k | **−69,540** | **−53,526** |
| final mean_log_var | **−9.48** | **−7.48** |
| final predictor weight L2 norm | 1,116.51 | 1,237.88 |

The v2 shuffle's loss is **less negative** than main's (−53,526 vs main's −62,607), and its mean_log_var is **higher** than main's (−7.48 vs main's −8.66). Both directions are the spec-correct expectations: shuffle is less optimised than main, and less confident than main. The v1 reversal (shuffle better than main on both) is now resolved.

### Gate verdicts (v2, against the new shuffle baseline)

| gate | criterion | v1 result | v2 result | verdict (vs scaffolding) |
|---|---|---|---|---|
| G1.1 | No NaN/Inf | PASS | PASS (unchanged) | **PASS** |
| G1.2 | Loss decreased | PASS | PASS (unchanged) | **PASS** |
| G1.3 | mean(log_var, steady) + 0.3 < mean(log_var, cue) | FAIL @ scaffolding (sep −0.14) | FAIL @ scaffolding (sep unchanged: −0.14); but main HAS structure (≈ −0.14) that shuffle does not (≈ 0.0002) | **FAIL @ absolute, pending reviewer call on relative** |
| G1.4 | Paired test main > shuffle at k=8 steady (p<0.01) | HELD (S1-S4 unexpected, control invalid) | **FAIL @ k=8** (mean_diff −0.13, Wilcoxon p=1.0); PASS @ k=1 (p=1.7e-5); PASS @ k=16 (Wilcoxon p=0.004 rank-based even though mean diff is −0.02) | **FAIL @ gated horizon** |
| G1.5 | M3 trajectory + floor | PASS (unchanged) | PASS (unchanged) | **PASS** |

#### G1.4 detail (v2)

Paired Wilcoxon on 250 steady-state probes (Shapiro-Wilk rejected normality at p < 1e-15 on every horizon → Wilcoxon used per the spec's fallback rule):

| horizon k | mean_diff (main − shuffle) | shapiro p | Wilcoxon p (one-sided greater) | pass at p < 0.01? |
|---:|---:|---:|---:|---|
| 1 | **+0.0395** | 1.4e-15 | **1.74e-05** | **PASS** |
| 8 (gated) | **−0.1318** | 9.4e-15 | 1.0 | **FAIL** |
| 16 | −0.0200 | 1.2e-19 | 0.0041 | PASS (rank-based) |

Main beats shuffle at the near horizon (k=1), loses at the mid horizon (k=8 gated), and wins at the far horizon (k=16) only by the rank-based test (mean diff is negative but the rank distribution favours main).

**Diagnostic for the k=8 failure (load-bearing for the reviewer's decision):**

Main's per-step M1 (cosine) has a characteristic shape that is below shuffle's flat baseline at mid-K:

| k | main M1 aggregate | shuffle aggregate (≈) |
|---:|---:|---:|
| 1 | 0.69 | 0.65 |
| 2 | 0.62 | 0.66 |
| 3-4 | **0.55** | 0.66 |
| 5-7 | 0.52 | 0.66 |
| 8 | **0.56** | 0.69 |
| 9-12 | 0.58–0.62 | 0.66 |
| 13-16 | 0.62–0.63 | 0.65 |

(Main aggregate over all 296 probes from `eval_95721.json`. Shuffle aggregate ≈ 0.66 is roughly flat because shuffle's predictions are near-constant marginal-mean outputs, see S3 below.)

Two candidate explanations the reviewer should weigh:

**(A) Rank-512 limited predictor architecture.** The output projection is `Linear(512, K*(d+1))`. Each μ_k slot is a 1024×512 sub-matrix mapping the 512-d transformer state to a 1024-d mean. The column space is rank ≤ 512 in the 1024-d output space. Cosine to a 1024-d target is bounded by the alignment of that target with the rank-512 subspace. Main is being penalised on cosine for an architectural reason that the loss (which optimises **squared error scaled by variance**, not cosine) doesn't see. Main's `mean_log_var = −8.66` vs shuffle's `−7.48` means main's squared error is ~half shuffle's (e^−8.66 ≈ 1.7e-4, e^−7.48 ≈ 5.6e-4) — main is winning on the LOSS objective by a large margin. The cosine gate is testing something the loss doesn't optimise.

**(B) Substrate-induced mid-horizon failure mode.** With the un-jittered Phase 1 stream, steady-state probes have bit-identical 16-frame targets; cue probes have a smooth dwell→transit transition. Main may have learned to predict "the first target frame matches the last window frame" (good at k=1, hence the 0.69 peak) and "the eventual stable trajectory matches the average direction" (good at k=16, hence the 0.62 recovery), but the middle of the predicted path drifts. Shuffle, predicting the marginal mean of all training targets, has a flat cosine across k that happens to land above main's mid-K dip.

Under (A), the architecture is fine and the gate is mis-specified; the reviewer would either raise predictor hidden dim (a SCAFFOLDING change per §12) or change the gate metric. Under (B), the substrate's bit-identical-dwell degeneracy plus the rank-limit conspire to produce a real failure of multi-step path prediction that Phase 2/3's jittered substrate may or may not resolve. The diagnostic that separates (A) from (B) is whether main's k=8 cosine improves substantially when the substrate is jittered (Phase 2/3) or whether the same dip persists.

The G1.4 verdict-as-computed stands at **FAIL @ k=8**. I am not declaring this autonomously per the reviewer protocol; the reviewer assigns the verdict.

#### G1.3 detail (v2) — main has structure shuffle doesn't

| stat | main (final ckpt) | shuffle (final ckpt) | main − shuffle |
|---|---:|---:|---:|
| steady mean log_var | −8.26 | −7.48 | **−0.78** |
| cue mean log_var | −8.40 | −7.48 | **−0.92** |
| separation (cue − steady) | **−0.14** | **+0.0002** | — |

Main is `−0.78 / −0.92` log_var **more confident** than shuffle on steady / cue probes respectively. Shuffle's separation across probe types is ≈ 0 (`0.0002`) — exactly what a temporally-destroyed control should show (shuffle has no way to tell steady from cue, so it predicts the same marginal-mean output for both).

So main DOES learn variance structure that the shuffle doesn't:

- Both magnitudes of confidence (main more confident than shuffle on each probe type).
- And differential structure across probe types (main −0.14 vs shuffle 0).

The structure runs in the **opposite direction** from the spec's specific hypothesis (which expected cue more variance than steady). The reviewer decides whether "main learns variance structure but not the specific direction §2.2 hypothesised" counts as a partial pass against the relative baseline, or whether the wrong-direction structure indicates a real architecture/substrate failure mode.

**Substrate-artefact diagnostic note for G1.3 (per session-2 reviewer item 2).** The un-jittered Phase 1 stream produces bit-identical 16-frame target embeddings for steady-state probes (all 16 future frames at the same viewing position are identical to floating-point precision). The optimal predictor for "predict 16 copies of a constant" under Gaussian NLL is "predict the constant with arbitrarily small variance" — log_var → −∞ (clamped to −10). Empirically main lands at log_var ≈ −8.26 for steady, well above the clamp, indicating the predictor hasn't fully saturated even on this trivial sub-problem. Possible reasons: the **rank-512 architecture limit** prevents perfect prediction of arbitrary 1024-d targets (squared error nonzero → variance has to absorb it), or the predictor's loss is dominated by other (transit, cue) targets it can't predict as well, dragging steady log_var up via shared weights. Cue probes are slightly *easier* in this substrate for an unintuitive reason: the dwell→transit smooth motion gives the predictor a directional cue (the last few window frames already moving), and the K=16 transit-frame targets share spatial structure (smooth path), so the predictor's variance fits a wider but consistent region. The inversion (cue more confident than steady) is consistent with this substrate-driven story. The reviewer should consider whether to wait for the Phase 2/3 jittered substrate (which adds genuine within-cluster variance to steady-state probes) before drawing architectural conclusions from G1.3.

#### S1-S4 sanity check (v2)

| check | criterion | observed | direction |
|---|---|---|---|
| S1 | shuffle log_var > main log_var (less confident) | shuffle −7.48 > main −8.66 | **expected ✓** |
| S2 | shuffle aggregate M1 < main aggregate M1 | shuffle 0.656 > main 0.592 | unexpected |
| S3 | shuffle |sharpness| << main |sharpness| | shuffle 0.011 << main 0.325 | **expected ✓** |
| S4 | quantitative collapse-to-mean: `||μ|| < 0.15` AND `log σ² > 0.4` | shuffle `||μ|| = 0.75`, `log σ² = −7.48` | unexpected |

**Aggregate verdict (with the existing S4 thresholds): unexpected (2 of 4 individual checks fail).**

But the qualitative interpretation has shifted from session 2:

- **S3 PASS is load-bearing.** Shuffle's cluster sharpness is **0.011** vs main's **0.325** — shuffle has essentially no item-discriminability, exactly what a temporally-destroyed control should show. Within-cluster cosines are 1.0 (shuffle outputs are deterministic on inputs) and cross-cluster cosines are 0.989 (shuffle outputs are nearly identical regardless of which item the input window is from). This is the **canonical "collapse to a single output direction" pattern** — just not the specific (norm < 0.15, log_var > 0.4) form S4's SCAFFOLDING thresholds were predicting.
- **S2's "unexpected" is the cosine artefact discussed in G1.4 above.** Shuffle's per-k aggregate is flat (predicting a fixed direction across all probes); main's per-k dips at mid-K. The aggregate-over-k comparison conflates the unfair-to-main mid-K dip with the favourable k=1 and k=16 endpoints.
- **S4's failure is a SCAFFOLDING-threshold mis-specification, not a sanity-check failure.** Instr §6.5 explicitly says: *"Starting thresholds (SCAFFOLDING — recalibrate after observing the empirical shuffle distribution at end of Phase 1)"*. The empirical shuffle distribution is now visible (||μ|| ≈ 0.75, log_var ≈ −7.48); the recalibrated thresholds the reviewer can authorise are e.g. `||μ|| > 0.6` (capturing the marginal-mean-direction signature) OR `sharpness < 0.05` (re-using S3's signal). Either captures the observed form of collapse cleanly.

Sanity check report at `results/inner_pam_v0/phase1_shuffle/sanity_check.json`.

### M3 trajectory framing — what spec claim Phase 1 supports

Per the session-2 reviewer note on item 3, the M3 sharpness trajectory 0.008 → 0.325 is real learning but with a specific interpretation given the substrate:

- **Within-cluster cosine ≈ 1.000** by construction: bit-identical pixels at the same viewing position across loops → bit-identical DINOv2 embeddings → bit-identical predictor outputs → within-cluster cosine = 1.0. This is a substrate floor, not a learned property of the predictor.
- **Cross-cluster cosine** is what evolves: at the first checkpoint (10k steps), shuffle-baseline-aligned ≈ 0.992; at the final checkpoint (95,721 steps), 0.675. Sharpness = 1.0 − 0.675 = 0.325.
- The trajectory is therefore measuring **cross-cluster discriminability**: how distinguishably the predictor outputs different vectors at different items. That is real learning, supported by the Phase 1 evidence.
- The **§2.2 "repetition tightens within-cluster representations" claim** is *not* tested in Phase 1 because within-cluster cosine is floored at 1.0 by the substrate. That test moves to Phases 2/3 where jittered collection produces non-identical dwell embeddings, giving within-cluster cosine room below 1.0 to tighten as repetition accumulates.

Architectural-claim status as of Phase 1:

| claim | status after Phase 1 | resolves in |
|---|---|---|
| Predictor learns cross-cluster discriminability | **supported** | (this phase) |
| Within-cluster representations tighten with repetition | **pending** (substrate-floored) | Phase 2/3 (jittered) |
| Predictor learns multi-step trajectory (cosine at mid-K beats shuffle) | **partial / unresolved** | depends on architecture limit diagnosis |
| Predictor learns differential variance structure across probe types | **supported (in direction)**, **wrong (in sign)** vs spec §2.2 | clarifies in Phase 2/3 with jitter |
| Predictor learns the specific cue-more-variance-than-steady pattern | **failed against spec direction** | reviewer call on whether to retain expectation |

### Reviewer-action items before session 4

1. **G1.3 verdict:** FAIL @ absolute scaffolding stands. PASS-at-relative-baseline is the reviewer's call given that main has structure shuffle doesn't, even in the wrong direction. (Item 2 from the session-2 review handed back.)
2. **G1.4 verdict:** FAIL @ k=8 stands as computed. Diagnostic suggests rank-512 architecture limit OR substrate dip; pre-Phase-2 fix candidates include raising PRED_HIDDEN, switching the gate metric to squared error (which main wins decisively), or accepting cosine-at-mid-K as a known weakness with the bit-identical substrate. (Item 1 from session-2.)
3. **S4 threshold recalibration:** empirical shuffle distribution is now known (||μ|| ≈ 0.75, log_var ≈ −7.48). Reviewer authorises new thresholds; recommendation in the §G1.4 section above.
4. **Phase 2 entry depends on items 1–3.** No autonomous progression.

If the reviewer chooses to investigate rather than continue:

- The cheapest single intervention for the cosine-at-mid-K issue is raising PRED_HIDDEN from 512 to ≥ 1024 (rank-unconstrained), which would put the predictor's mu output in the full 1024-d space. Single-variable, ~30 min retrain. Worth doing once before Phase 2 if the verdict is "investigate."
- The G1.3 direction inversion is most cleanly diagnosed by Phase 2/3 evidence; running them and looking at the relative log_var pattern under jittered substrate would settle whether the substrate or the architecture drove the inversion.

---

## Session 2 outcomes — 2026-05-13

**Goal.** Launch Phase 1 main training, sequentially launch shuffle control, run per-checkpoint eval, compute G1.1–G1.5 + S1–S4 disaggregations, STOP for review. Mechanical gates auto-declared; SCAFFOLDING-threshold gates pause for review per reviewer protocol.

### Phase 1 main training

| metric | value |
|---|---|
| wall-clock | 869.9 s (~14.5 min) on RTX 4080 Super, fp16 |
| gradient steps | 95,691 (≈ 110 steps/sec, faster than session-1 smoke's 60 steps/sec) |
| n_train | 95,722 (last 10 of 218 loops held out) |
| held-out region | frames [95,722, 100,000) |
| τ (calibrated at step 10k from steps 5k–10k median) | **8.125** |
| final mean_loss_last_1k | −62,606.72 |
| first mean_loss_last_1k (ckpt 10k) | −59,541.71 |
| final mean_log_var | −8.66 (predictor confidence rising over training, as expected) |
| final predictor weight L2 norm | 1,270.53 |
| final bank size | 95,707 |
| NaN/Inf | none |

Checkpoints at steps 10k, 20k, …, 90k, 95721; predictor + optimizer + bank state at each.

### Phase 1 shuffle control training

| metric | value |
|---|---|
| wall-clock | 862.7 s (~14.4 min) |
| gradient steps | 95,660 |
| shuffle_seed | 0 |
| final mean_loss_last_1k | **−69,540.23** (more negative than main's −62,607; load-bearing finding — see below) |
| final mean_log_var | **−9.48** (predictor MORE confident than main, opposite of S1 expectation) |
| final predictor weight L2 norm | 1,116.51 |

### Cue-probe construction bug found and fixed mid-session

Initial Phase 1 main eval produced only **9 cue probes** (expected ~50). Root cause in `src/eval/probes.py:build_cue_probes`: the destination-item label was identified by scanning only the next K=16 frames after the window, but seed-7 transit segments are typically 60+ frames long, so the next dwell almost never appeared inside the window and almost every cue candidate was skipped.

Fix (commit `ea84df1`): scan forward through the stream until the next dwell frame appears, regardless of distance. The to_item field is metadata only (which transition the probe tags); the K=16 target frames are unchanged.

After fix: 46 cue probes constructed (10 held-out loops × 5 transitions = 50 max, minus 4 that fail other constraints). Probe tests still pass. Main eval re-run after the fix. Shuffle training was launched only after the fix was verified.

### Gate verdicts

The reviewer protocol: G1.1 / G1.2 are MECHANICAL (auto pass/fail); G1.3 / G1.4 / G1.5 carry SCAFFOLDING thresholds and the script does not autonomously declare pass. Verdicts below are reported "against the documented scaffolding threshold" — recalibration is the reviewer's call.

| gate | criterion | result | verdict (vs scaffolding) |
|---|---|---|---|
| G1.1 | No NaN/Inf in predictor weights + final loss finite | no non-finite parameters; loss = −62,606.72 | **PASS** (mechanical) |
| G1.2 | Final mean_loss_last_1k < first mean_loss_last_1k | −62,606.72 < −59,541.71 (decreased by 3,065.0) | **PASS** (mechanical) |
| G1.3 | mean(log_var, steady) + 0.3 < mean(log_var, cue) | steady: −8.256, cue: −8.399, sep: **−0.14** | **FAIL** at scaffolding +0.3 threshold |
| G1.4 | Paired t-test main > shuffle at k=8 steady (p<0.01) | Shapiro-Wilk p=3.3e-8 → Wilcoxon, p_value=1.0, mean_diff=−0.359 | **HELD** (S1–S4 unexpected per §6.5) |
| G1.5 | M3 trajectory rises across Phase 1, floor > 0.10 | sharpness 0.008 → 0.325, 8 of 9 transitions non-decreasing, floor cleared | **PASS** at scaffolding |

#### G1.5 trajectory detail

| step | cluster sharpness | non-dec from prev? |
|---:|---:|---|
| 10,000 | 0.0084 | — |
| 20,000 | 0.0225 | ✓ |
| 30,000 | 0.0252 | ✓ |
| 40,000 | 0.0669 | ✓ |
| 50,000 | 0.0857 | ✓ |
| 60,000 | 0.1841 | ✓ |
| 70,000 | 0.2335 | ✓ |
| 80,000 | 0.3587 | ✓ |
| 90,000 | 0.2966 | ✗ (dip) |
| 95,721 | 0.3249 | ✓ |

Criterion (≥ 7 of last 9 non-decreasing, allowing 2 dips): **8 / 9 satisfied**. Floor 0.10 cleared comfortably at 0.325. Trajectory and floor both pass.

**Determinism caveat that affects M3 interpretation.** Phase 1 trains on the un-jittered rerender stream (per session-1 HANDOFF). Dwell frames at the same viewing position are bit-identical across loops (confirmed by the substrate verification Check 1 degenerate result and by the 1.000000 consistency cosines in the session-1 re-encode). Therefore, the predictor's outputs at steady-state probes within the same viewing position are bit-identical → within-cluster cosine = 1.0 (deterministic). The M3 cluster sharpness then becomes `1.0 − cross_cluster_mean`, i.e., it is functionally measuring **cross-cluster discriminability** (how different the predictor's outputs are at different items) rather than within-cluster tightening over training. The rising trajectory is still a valid signal — it shows the predictor learning to discriminate items over training — but it does not test §2.2's "repetition tightens clusters" claim in the strong form intended. That test will become meaningful in Phases 2/3, where perturbed-shape rep counts start at 0 and accumulate, and where the jittered collection introduces actual within-cluster variance.

#### G1.3 detail

Across the 10 main checkpoints, the steady-vs-cue mean-log-var separation evolved as follows (steady mean − cue mean per checkpoint):

| step | steady mean log_var | cue mean log_var | sep (cue − steady) |
|---:|---:|---:|---:|
| 10,000 | −8.1811 | −8.0978 | +0.0832 |
| 20,000 | −8.1089 | −8.0956 | +0.0132 |
| 30,000 | −8.6761 | −8.8524 | −0.1762 |
| 40,000 | −8.2375 | −8.5147 | −0.2772 |
| 50,000 | −8.5113 | −8.7847 | −0.2734 |
| 60,000 | −7.9887 | −8.2547 | −0.2660 |
| 70,000 | −8.3213 | −8.6742 | −0.3529 |
| 80,000 | −8.2402 | −8.3690 | −0.1287 |
| 90,000 | −8.7949 | −8.9601 | −0.1651 |
| 95,721 | −8.2561 | −8.3987 | −0.1426 |

(Verified against `results/inner_pam_v0/phase1_main/log_var_trajectory.json`.)

The separation evolves in the **wrong direction** for the gate (towards cue being more confident than steady). The expectation in instr §7.7 G1.3 was `cue_log_var − steady_log_var > 0.3` (cue more variance = less confident). The empirical pattern is `cue_log_var − steady_log_var < 0` (cue less variance = more confident). At early checkpoints (10k, 20k) the separation is small and positive (steady marginally noisier than cue); by 30k it has already flipped negative and stays negative through 95,721. Magnitude bounces in the −0.13 to −0.35 range — the final value of −0.14 is not a steady-state value, it's where the trajectory happens to land at the last checkpoint.

Possible reasons (informational only, not for autonomous recalibration):

- *Substrate artefact.* With un-jittered Phase 1 data, steady probes have bit-identical inputs across loops but the predictor's bias terms may produce a non-zero residual error. With "trivial" target, the optimum log_var goes very low — but if there's a constant tiny error, log_var settles where the gradient balances. Cue probes have less trivial targets but smooth predictable transit-frame trajectories; their convergence might be sharper.
- *Architecture / loss artefact.* The loss formulation may permit / reward an unintended local minimum at cue probes.
- *Calibration mis-specification.* The +0.3 threshold may simply not be the right number; reviewer recalibration is on the table per §12 SCAFFOLDING discipline.

Disaggregated per-step k log_var values are in `log_var_trajectory.json`. The numbers are reported; the verdict is the reviewer's.

#### G1.4 detail — shuffle re-design candidate

The S1-S4 shuffle sanity check returns **"unexpected" on all four individual checks**:

| check | expectation | observed (final ckpt) | direction |
|---|---|---|---|
| S1 | shuffle log_var > main log_var (less confident) | shuffle −9.48 < main −8.66 | inverted |
| S2 | shuffle M1 < main M1 (lower centreline accuracy) | shuffle 0.97+ > main 0.97-ε | inverted |
| S3 | shuffle |sharpness| << main |sharpness| | shuffle similar/higher than main | inverted |
| S4 | quantitative collapse-to-mean | shuffle did not collapse | inverted |

Paired Wilcoxon at three horizons (250 steady-state probes, one-sided main > shuffle, Shapiro-Wilk rejected normality at p < 0.05 → Wilcoxon used):

| horizon k | mean_diff (main − shuffle) | shapiro p | test | p_value | pass at p < 0.01? |
|---:|---:|---:|---|---:|---|
| 1 | −0.302 | 6.6e-20 | Wilcoxon | 1.000 | no |
| 8 (gated) | −0.359 | 3.3e-08 | Wilcoxon | 1.000 | **no** |
| 16 | −0.250 | 2.9e-10 | Wilcoxon | 1.000 | no |

Across all three horizons, **shuffle predicts the held-out continuation better than main**, with mean cosine differences of 0.25–0.36. This is a HUGE divergence in the wrong direction, not noise.

**Root-cause diagnosis (load-bearing for the re-design decision).** The current shuffle implementation matches the literal wording of instr §7.5 — "applies `np.random.default_rng(0).permutation(N_train)` to the training indices" — interpreted as permuting the **visit order** of (window, target) pairs during training. Each pair, however, is still a real coherent W=16-frame window followed by a real K=16-frame continuation drawn from consecutive stream positions. Temporal structure within each pair is preserved.

This contradicts the spec rationale at §10.1 ("Should fail because the temporal structure required for shape learning is destroyed") and §6.3 C2 ("temporally-shuffled version of the same stream"). My current implementation is just SGD-with-random-batches, which is the standard ML optimization heuristic — and in fact it *helps* convergence relative to sequential SGD (consecutive sequential batches are highly correlated because of overlapping windows; random ordering decorrelates them). So shuffle ended up being a *better-optimized* version of main, not a worse one.

Per the `WEFT_INNER_PAM_v0_EXPERIMENT_INSTRUCTIONS.md` preamble — "If this document and the spec disagree, the spec wins and this document is wrong — flag in `HANDOFF.md` and stop" — the spec wins. The shuffle should be re-implemented to actually destroy temporal structure (e.g., permute the embeddings themselves *before* building windows, so each window's contents are random unrelated frames). Cost: ~15 min re-train + ~2 min re-analysis.

If the reviewer agrees, I'll fix the shuffle (preferred path), re-run both shuffle training and the gate analysis, and re-write the §G1.4 section. The G1.4 verdict-as-computed stands at "HELD" pending this re-design.

### S1-S4 sanity check verdict

`results/inner_pam_v0/phase1_shuffle/sanity_check.json`:
- S1 (log_var distribution): shuffle mean −9.48, main mean −8.66. Shuffle MORE confident. **unexpected**.
- S2 (M1 distribution): shuffle aggregate cosine higher than main. **unexpected**.
- S3 (cluster sharpness): shuffle sharpness comparable to main; not zero. **unexpected**.
- S4 (quantitative collapse-to-mean): shuffle ||μ|| not below 0.15 floor; log σ² not above 0.4 floor. **non-collapse**, **unexpected**.

Aggregate verdict: **unexpected**. Per instr §6.5, the gate G1.4 is held; this entry is the documented flag.

### What's in the working tree

All Phase 1 result JSONs are committed in `results/inner_pam_v0/phase1_{main,shuffle}/`:
- `training_summary.json`, `init_report.json`, `tau_calibration.json` (main only).
- `checkpoint_{step}.json` (×10 each in main + shuffle).
- `eval_{step}.json` (×10 in main; shuffle eval was not run separately — the gate-report script computed shuffle predictions directly).
- `m3_trajectory.json`, `log_var_trajectory.json`, `gate_report.json` (main).
- `sanity_check.json` (shuffle).

Predictor checkpoints (`ckpt_*.pt`, ~258 MB each = ~2.5 GB / phase) and bank state dirs (`ckpt_*/`) are gitignored.

### Reviewer-action items before session 3

1. **Resolve the shuffle interpretation** (recommend: re-implement spec-correctly).
2. **Decide G1.3 verdict** (accept the surprising cue-more-confident-than-steady finding for now, or pause to diagnose the inversion). Recalibration of the +0.3 threshold is on the table per §12 SCAFFOLDING discipline.
3. **G1.5 PASS confirmation** at the scaffolding threshold (the trajectory criterion is the primary content; the 0.10 floor is met).
4. **If both shuffle and G1.3 are resolved as PASS / acceptable**: proceed to Phase 2 (LivingRoom RandomizeMaterials wrapper, preflight verification per §8.2). Session 3 setup.

### Host-protection decision recorded

Reviewer empirically confirmed the device stays up indefinitely (10 days of uninterrupted uptime under the current power-plan configuration). Settings verified by my probes:

| item | spec'd | actual | note |
|---|---|---|---|
| AC device sleep | Never | Never | ✓ confirmed by UI screenshot |
| AC screen sleep | (not load-bearing) | 10 min | does not affect WSL2 / training |
| AC hard-disk sleep | Never | **20 min** | not changed; low practical risk given continuous log/checkpoint writes resetting the idle timer |
| WU pause | active | **no pause keys** | low practical risk given active-hours [3, 21) defers reboots |

Recorded as a **deliberate deviation from §0.1** with the reviewer's empirical-evidence rationale. Both items proved out: the ~30-min training runs completed without disk-sleep interruption, and no forced reboot occurred during the session. The §0.1 wording is overspecified for this workload profile; if a future phase ran sub-process-idle (e.g., long pure-Python data loading without disk hits), the disk-sleep item would warrant revisiting.

### DINOv2 determinism observation (carryover note from reviewer)

The 1.000000 consistency cosine on the 50-frame re-encode sample in session 1 is strong evidence that DINOv2 in this environment (fp16 eval mode on RTX 4080 Super, transformers 5.3.0, ImageNet preprocessing pipeline) is genuinely bit-identical-deterministic on identical pixels. If Phase 2/3 substrate sanity-checks ever surface anomalies, re-running the §5 substrate verification (or a smaller spot-check) is a high-confidence first-line diagnostic — encoder behaviour is **not** a run-to-run noise source.

---

## Operational state (end of session 1)

(Historical — preserved for audit.)

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
