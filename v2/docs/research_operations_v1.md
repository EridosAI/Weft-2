# Research Operations — Experiment-Agnostic Process & Discipline

**Purpose:** Operational guardrails for how experiments are run, reviewed, and handed off. This document is deliberately architecture-agnostic — it does not assume any specific research direction, model architecture, training regime, or environment. It captures *how we work*, not *what we build*.

**Scope:** Applies to every experiment in every project, independent of research line. When starting a new project, this document carries over unchanged. Architecture-specific principles and findings live in project-specific docs alongside this one.

**Rule for changing this document:** Any item can be challenged, but changes must be stated explicitly with rationale before implementation — never silently drifted into. If you find yourself operating in a way that conflicts with an item below, stop and flag it. It might be the right move, but it needs to be a deliberate decision, not an accident.

---

## 1. Core Philosophy

### 1.1 Hours of design per minutes of execution
The right ratio for research work is design-heavy. Spec → review → resolve open questions → CC instructions → code. When bugs appear, they tend to appear in whichever stage was under-designed. The cost of an hour of design is small; the cost of a bad CC run is days.

### 1.2 Experiments are single-variable
Never bundle interventions that share training dynamics. Capacity, learning rate, and loss function all interact through shared parameters. If you're not testing the LR, don't change the LR. When three rounds of fixing produce lateral movement (fix one thing, break another), the problem is structural, not parametric — stop tuning and diagnose the mechanism.

### 1.3 Honest negative results contain the prescription for the fix
"We tested X, it failed because Y, this suggests Z." Every major negative result should drive an architectural improvement. A negative result without root cause analysis is a waste — the value is in the Y, not the "it failed."

### 1.4 The cheapest diagnostic comes first
Before committing to expensive compute, run the minimum test that can close the path. A 1-hour diagnostic that kills a bad direction saves a week of data collection. A 5-minute analysis on existing artifacts often out-informs a 2-hour retraining run. If a diagnostic triggers a training loop, it is not a quick diagnostic.

### 1.5 Lessons-learned documents are institutional memory, not canonical truth
Past findings are treated as conditional guidance, not permanent rules. They were established under specific architectures, environments, and training regimes. When any of those change, revisit the finding. Don't carry forward a lesson as settled when the context it was learned in no longer applies.

---

## 2. Spec-Driven Development

### 2.1 Write the spec before writing code
The spec is a separate document that the experiment chat produces before any implementation begins. It states:
- The evaluation question (what does "correct behaviour" mean for this phase?)
- The hypothesis being tested
- The architecture or intervention being evaluated
- The metrics and thresholds that determine success
- The ablation controls designed up front
- The regression canaries (metrics that must not degrade)
- Scaffolding labels on every fixed parameter, with removal plans
- The specific files to be created or modified

No code gets written until the spec exists. No CC instructions are drafted until the spec has been reviewed.

### 2.2 Adversarial review before CC implementation
Before CC runs any substantive experiment, the spec passes through:
1. **Primary reviewer** (Claude oracle chat or equivalent) — alignment with existing principles, drift detection, closed-path check.
2. **Secondary reviewer** (a different model — ChatGPT, Grok, or another instance) — fresh perspective on the same material.
3. **Resolution pass** — both reviewers' findings addressed or explicitly overridden with rationale.

Different models have different blind spots. The union of two reviews is strictly better than either alone. Budget time for this — it is not optional. This step has caught reviewer-killable issues in every document it has been applied to.

### 2.3 Label every fixed parameter as ARCHITECTURE or SCAFFOLDING
Every fixed value in the design is tagged. The test is simple: can you remove it and the architecture still works in principle? If yes, it is SCAFFOLDING and requires a removal plan plus an architectural target it is standing in for. If no, it is ARCHITECTURE and the design depends on it.

Scaffolding that survives three experiments without a removal plan is calcifying into architecture by default. This is a silent drift pattern — review the scaffolding inventory at the start and end of every experiment.

### 2.4 Derive parameters from data, not hardcoded values
Every hardcoded number is a hidden assumption that breaks when the data changes. Thresholds, tolerances, bin boundaries, temporal windows, retrieval k values — all should be computed from the data or labelled as SCAFFOLDING pending a principled derivation. Hardcode only when a fixed value is provably correct across all conditions.

---

## 3. Evaluation Discipline

### 3.1 Define the evaluation question before designing metrics
"Correct behaviour" must be stated explicitly and in writing before any metric is designed. The evaluation question changes as the architecture evolves — revisit it at the start of every phase. Metrics designed without a clear evaluation question tend to measure what is convenient rather than what matters.

### 3.2 Controls are part of the design, not afterthoughts
Every experiment ships with its ablations. The specific controls depend on the system under test, but the principle is fixed. At minimum:
- A baseline that removes the mechanism being tested entirely
- A baseline that provides random or shuffled input (proves content matters, not just extra signal)
- A "does the mechanism help?" diagnostic that isolates the intervention's contribution

Design these on day one. Controls added after the fact are systematically weaker — they tend to confirm what the experiment already showed rather than challenge it.

### 3.3 Run the simple baseline early
If a simple baseline matches the learned system, find out at phase 3, not phase 5. The simplest model (MLP, nearest-neighbour, cosine lookup, random retrieval) answers "is the benchmark complex enough to differentiate?" Run it alongside the main system from day one. If the learned system doesn't beat it, either the benchmark is too easy or the learned mechanism isn't earning its keep — both are useful findings, and both are worth knowing early.

### 3.4 Set regression canaries before interventions
Before changing anything, identify the metric that must not regress and set an explicit floor. Every subsequent intervention is measured against whether it preserves this result. When the canary fires, it prevents a bad checkpoint from becoming the new baseline.

### 3.5 Disaggregate before diagnosing
Summary metrics hide failure modes. Log per-instance predictions, per-class accuracy, per-condition breakdowns, retrieval statistics. The most informative number in a failed experiment is often one that wasn't in the summary — run it on existing output before running another experiment.

### 3.6 Small samples in high-variance domains are unreliable
Do not draw conclusions from n<20 in any domain with meaningful per-instance variance. Specific numbers can swing dramatically between n=3 and n=5; if the effect is real, it survives larger samples. Budget data collection to reach statistical stability, not just to observe an effect.

### 3.7 Evaluation subsets must be meaningful
Before applying a threshold, verify the denominator. A 40% success rate is failure on the whole population but can be success on the relevant subset, or vice versa. State the evaluation subset explicitly in the spec.

### 3.8 Calibrate the event rate before committing to data collection
Run a short calibration collection (typically 50 episodes or less) before committing to a full run. Verify the policy triggers the target events at sufficient rates. Gate on event counts, not on total transitions. A policy that produces 400k transitions of empty actions is not producing data.

---

## 4. Number Discipline

### 4.1 Every number traces to a specific output file
Every number in a result, handoff, spec, or paper traces to a specific file on disk. Before any claim is made, verify the number against the raw output. Do not mix numbers from different configs or runs in the same summary.

### 4.2 Do not trust post-compaction summaries
Summaries produced after context compaction have produced fabricated numbers. Always verify the number against the raw JSON/CSV/log output before quoting it. This applies to both your own summaries and any summary produced by CC.

### 4.3 The summary is not the evidence
Loss curves, accuracy headlines, and aggregate metrics are the tip of the iceberg. The evidence is in the per-instance logs, the diagnostic dumps, the retrieval statistics. A surprising result is not confirmed until it is traced to the raw data.

---

## 5. Single-Variable Discipline

### 5.1 One change at a time
When testing an intervention, change exactly one thing. If three changes are bundled and the result improves, it is impossible to know which change caused the improvement. If the result degrades, you have three suspects instead of one.

### 5.2 If fixes cause lateral movement, the problem is structural
When three rounds of tuning produce "fix one concept, break another" patterns, stop tuning. The architecture itself has a structural issue that parameter changes cannot resolve. Parametric tuning compensates for deeper problems up to a point, then fails in different ways each time.

### 5.3 Fix what feeds the model before making the model bigger
Scaling model capacity before fixing input/data quality conflates variables. Adding capacity to compensate for bad signal makes the experiment more expensive without resolving the underlying problem. Always diagnose which component is failing before changing the whole system.

### 5.4 Retrieval and prediction are separable problems
When a system combines retrieval (finding the right context) and prediction (generating the right output), diagnose which component is failing before changing either one. A prediction failure can be caused by bad retrieval; improving the predictor will not fix it.

---

## 6. Pre-Experiment Gates

### 6.1 Calculate what random achieves before running
A random-baseline analysis takes minutes and prevents spending hours discovering that your "good" result is chance. Action space structure, class imbalance, and evaluation protocol quirks can all produce deceptively high random-baseline scores. Compute the expected random performance analytically before running.

### 6.2 Calibrate physical scales before training
Any threshold compared against a computed value must verify that the scales match. Thresholds and measurements on different orders of magnitude produce silent failures — everything scores at chance because no example ever crosses the threshold.

### 6.3 Verify signal structure before training on it
If the system under test depends on a specific property of the data (signal separation, temporal structure, cross-boundary associations, etc.), verify that property exists before training. A pre-training diagnostic that takes an hour can prevent a week of training on data that never had the signal.

The specific diagnostic depends on the architecture. But the principle is always: measure what the system needs to see, before asking whether it can see it.

### 6.4 Balance training data for binary outcomes
Any binary target (positive/negative, success/failure, etc.) requires explicit balance verification. Severe imbalance (< 10% minority class) produces majority-class priors that look like learning. Verify the class distribution before training; if imbalanced, address it explicitly rather than discovering it in post-hoc analysis.

### 6.5 The environment must contain the structure the mechanism needs
Before running an experiment, confirm the environment actually produces the signal the system is designed to exploit. If the mechanism's distinctive claim is "handles X," and the environment contains no X, the experiment can't evaluate the claim — the mechanism will look indistinguishable from a simpler baseline not because it failed, but because there was nothing to demonstrate. This recurs across research lines; every new environment gets checked for the specific structural property that differentiates the system under test.

### 6.6 The evaluation paradigm must distinguish the intended mechanism from plausible shortcuts
The evaluation protocol has to be able to answer: "is the system doing the thing we designed it to do, or is it succeeding via a shortcut that has the same surface outputs?" This is not the same as "does the system work." Controls, shuffle tests, and mechanism-specific diagnostics exist to separate the two. Design these into the evaluation before training — they are much harder to add after the fact, when results already exist.

### 6.7 The simple baseline must be beatable and must be beaten
Every experiment runs a simple baseline alongside the learned system. The baseline must be capable of succeeding on the task if a shortcut exists (otherwise it doesn't test for shortcuts). And the learned system must beat it somewhere meaningful, or the added complexity is not earning its keep. "Somewhere meaningful" can be a subset, a complexity regime, or a specific metric — but it has to be identifiable in advance and tested explicitly.

---

## 7. Scaffolding Discipline

### 7.1 Every fixed parameter is labelled
ARCHITECTURE or SCAFFOLDING — no third category. Every fixed value, every simplified mechanism, every "for now" choice gets a label. SCAFFOLDING items carry a removal plan: what is it standing in for, what is the architectural target, and how will we know when we can replace it?

### 7.2 Scaffolding inventory is reviewed every phase
Start of phase: what scaffolding exists? End of phase: is any of it ready to be removed? Which is calcifying? Scaffolding that survives three phases without a removal plan or challenge has become architecture by default — either accept it as architecture explicitly, or remove it.

### 7.3 If you're adding a hardcoded number, label it before writing the code
Any new fixed parameter, hardcoded threshold, or simplified mechanism gets a SCAFFOLDING label with a one-line removal plan *before* it enters the codebase. This prevents the calcification pattern where temporary choices become permanent by accident.

---

## 8. CC Operational Rules

These go in the preamble of every CC instruction document. They are compute-safety rules, not research rules — they prevent wasted compute and lost work.

### 8.1 Never kill a running training process
Under any circumstances. If a training run is in progress, wait for it to complete. Interrupted training runs may leave inconsistent state, and the cost of restarting often exceeds the cost of waiting. If an experiment needs to be aborted, the decision is made in the experiment chat, not by CC.

### 8.2 Use nohup for long-running scripts
Any script expected to take more than a few minutes is launched with nohup, redirected to a log file, and backgrounded. CC polls the log file for progress rather than waiting on stdout.

### 8.3 Poll logs, do not wait on stdout
CC reads log files to determine progress. It does not sit blocking on a running process. This lets other work continue while long jobs run.

### 8.4 Stop after 5 failed tool calls
If CC fails the same tool call 5 times in sequence, it stops and reports the failure rather than continuing to retry. Repeated failures usually indicate a structural problem that more attempts will not solve.

### 8.5 Operate in "away mode" — no clarifying questions
The user is not watching. CC makes reasonable decisions based on the spec and instructions, documents those decisions, and flags them in HANDOFF.md for later review. A CC chat that stops to ask a question wastes hours of potential compute time.

### 8.6 Environment header in every instruction document
Every CC instruction document begins with the environment header:
- OS / shell / Python / CUDA / PyTorch versions
- GPU (local or rented, with VRAM)
- Working directory
- Active virtual environment
- Any required env vars (e.g. `CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000`)

### 8.7 HANDOFF.md at every session boundary
Every CC session ends by updating HANDOFF.md. The handoff records:
- What was attempted this session
- What worked, what failed, what is in progress
- Every important decision and its rationale
- The state of the working tree (uncommitted changes, running jobs)
- The next immediate action for the next session

### 8.8 Git commit after every task
Not every change, every *task*. A completed bug fix, a completed experiment run, a completed refactor — each gets its own commit with a descriptive message. The commit message records what was done and why. History reconstruction is impossible without this.

### 8.9 Numbers trace to files
CC verifies every number it reports against the actual output file. No remembered numbers. No mental arithmetic. If CC cannot find the source of a number, the number is not quoted.

### 8.10 STOP at gate failures unless explicitly overridden
Every spec has gates — thresholds that must be met before proceeding. When a gate fails, CC stops and reports. It does not silently continue to the next phase. Override requires explicit justification in the experiment chat, recorded in HANDOFF.md.

### 8.11 "What NOT to change" section in every CC instruction document
Lists settled decisions and anti-patterns specific to this work. Prevents CC from re-litigating decisions already made, or introducing patterns that have been explicitly ruled out.

### 8.12 Phase gates: "STOP after phase N"
Long specs are broken into phases with explicit stop points. "Complete Phase 2, do not proceed to Phase 3 until review." This gives the experiment chat a chance to verify results before CC commits more compute in the wrong direction.

---

## 9. Session & Context Management

### 9.1 Context limits are a real constraint
Treat context as a scarce resource. Budget it like compute. Losing context mid-session costs as much as losing a GPU-hour.

### 9.2 Phase boundaries are session boundaries
Each CC session corresponds to a phase (or sub-phase). Each ends with HANDOFF.md updated and git committed. Do not run multiple phases in a single session — context bloat accumulates and accuracy degrades.

### 9.3 Aggressive scoping: give CC only what it needs
CC gets the spec, HANDOFF.md, and the specific files required for the current task. Not the whole repo, not all the project docs, not every file that might be relevant. The smaller the context, the more reliable the work.

### 9.4 Sub-agents for bounded tasks
When context starts to fill, delegate to a sub-agent with a narrow, bounded task. The sub-agent returns its result and its context is discarded. This extends effective context budget significantly for orchestration-heavy work.

### 9.5 Do not run parallel CC instances on the same codebase
Concurrent edits produce race conditions and merge conflicts that are difficult to untangle. One CC at a time per working tree.

### 9.6 Compacted context cannot be trusted with numbers
After context compaction, CC can produce fabricated numbers in summaries. Any claim from a compacted session is verified against the raw output files before it is treated as true.

---

## 10. Oracle / Review Chat Pattern

### 10.1 Every experiment chat has a companion review chat
The experiment chat does design, implementation via CC, and result interpretation. The review chat does drift detection only — checking that specs, decisions, and results stay aligned with established principles and don't re-explore closed paths.

### 10.2 The review chat is not an implementer
It doesn't write specs, suggest architectures, or propose features. It reads what the experiment chat produces and flags divergence. "This is fine" is valuable. "This contradicts Lesson X" is more valuable.

### 10.3 Minimum review inputs
The review chat has access to:
- Current principles document (this doc + any project-specific principles)
- Lessons-learned document (evidence base)
- Any architecture descriptions specific to the current research line
- The review protocol itself

The review chat does NOT need code, data files, or CC instructions. It reviews specs, design decisions, and result summaries that are pasted in.

### 10.4 What gets pasted in
- Specs before CC runs them
- Design decisions from the experiment chat
- Results that seem surprising
- Proposed pivots
- External model review (ChatGPT, Grok, etc.) — the review chat cross-checks external feedback against principles
- CC output summaries

### 10.5 Review chat response format
- **Alignment assessment**: ALIGNED / CONCERN / DRIFT per item.
- **Specific findings**: what the issue is, which principle/lesson it violates, what the consequence would be, suggested fix or flag.
- **What's fine**: brief confirmation of aligned elements — "the review checked it and it's fine" is a useful signal.

### 10.6 Err toward flagging
A false alarm costs a minute of discussion. A missed drift costs days of wasted CC runs. When in doubt, flag it.

### 10.7 The review chat reads the full docs every session
Not summaries. The full principles and lessons-learned documents. Details matter, drift patterns are specific, and closed paths have specific evidence that summaries lose.

---

## 11. Compute & Infrastructure

### 11.1 Local vs rented compute
Local compute (personal GPU) for development, fast iteration, and debugging. Rented compute (vast.ai or equivalent) for long training runs that exceed local capacity or would block the development machine. Don't train for hours on the development machine — it blocks everything else.

### 11.2 Overallocate disk on rented instances
Always provision more disk than you think you need. Many providers do not allow disk resize after instance creation, and running out of disk mid-run kills the experiment. 500GB minimum for any non-trivial work.

### 11.3 Data collection is versioned and archived
Training data, trajectory collections, and preprocessed datasets are versioned. Name includes collection date and parameters. Never overwrite a previous collection — always write to a new versioned path. Disk is cheap; re-collecting data is expensive.

### 11.4 Model checkpoints are versioned and tagged
Every saved model is tagged with the experiment version, phase, and date. Commit hashes in checkpoint metadata when possible. A checkpoint you can't re-produce from a commit is a checkpoint you don't really have.

### 11.5 Archive old code, don't delete it
Move old phases into archive folders. Keep the root directory clean with only current-phase files. When doing a major version change, create a fresh repo (tag the old one) rather than evolving in place. Establish directory structure before file count grows — flat directories with 50+ files become impossible to navigate.

---

## 12. Documentation

### 12.1 Living documents
Principles and lessons-learned documents are living references. Update them when a finding is validated, invalidated, or refined — with explicit rationale for each change. The change log matters; it lets future sessions reconstruct why a principle evolved.

### 12.2 Paper reviews and external literature
External papers relevant to the work get a structured review note: source, summary of what it does, ADOPT items (things to take into our work), TAKE NOTE items (findings worth knowing but not immediately relevant), AVOID items (things the paper reveals to be bad ideas), Open Questions raised. This prevents the pattern where a paper is read, forgotten, and re-read three times.

### 12.3 Session-level notes
Each working session produces a dated note capturing the session's question, what was tried, what worked or failed, and what the next session should pick up. These accumulate into the evidence base that lessons-learned draws from.

### 12.4 Don't carry institutional memory as canonical truth
Accumulated notes are conditional guidance, not settled fact. When the context changes (new architecture, new data domain, new training regime), old lessons require re-verification, not automatic inheritance.

---

## 13. Process Checklist: Starting a New Phase

Use at the start of every experiment phase.

- [ ] Read this document
- [ ] Read project-specific principles and lessons-learned
- [ ] Read HANDOFF.md from previous session
- [ ] State the evaluation question explicitly (what does "correct behaviour" mean?)
- [ ] Write the spec
- [ ] Confirm ablation controls are included in the spec
- [ ] Confirm simple baseline is included as a validation gate
- [ ] Set regression canaries from previous phase results
- [ ] Label every fixed parameter as ARCHITECTURE or SCAFFOLDING
- [ ] Run pre-experiment gates (signal structure, event rate, random baseline, scale calibration, environment structural check, evaluation-shortcut check, simple-baseline check)
- [ ] Submit spec to primary review chat
- [ ] Submit spec to secondary reviewer (different model)
- [ ] Resolve all review findings or document explicit overrides
- [ ] Draft CC instructions with environment header and "what NOT to change" section
- [ ] Set phase gates with explicit STOP points
- [ ] Git commit spec and CC instructions before implementation begins

---

## 14. Process Checklist: Ending a Phase

Use at the end of every experiment phase.

- [ ] All numbers in summaries traced to output files
- [ ] HANDOFF.md updated with session results and next-action
- [ ] Git committed with descriptive message
- [ ] Scaffolding inventory reviewed — any calcification?
- [ ] Regression canaries checked — did anything degrade?
- [ ] Spec gates evaluated — pass, fail, or conditional?
- [ ] Results submitted to review chat for sign-off
- [ ] Lessons-learned document updated with new findings
- [ ] If the phase produced a negative result, root cause analysis included

---

## 15. Drift Detection — Universal Checks

These checks apply to any research line. Architecture-specific drift checks live in project-specific principles docs.

- [ ] Does the plain-English description of the pipeline match the intended mechanism? (Describe the pipeline stripped of all jargon and architecture names — does it still do what you think it does?)
- [ ] Are you about to recommend something that was already tried? Check lessons-learned before proposing any "obvious" improvement.
- [ ] Have you checked the empirical history before making a theoretical recommendation? Theoretically beautiful ideas have been empirically falsified before.
- [ ] Is a downstream component (evaluation metric, grounding module, generator, etc.) becoming the thing everything else serves? That's a dependency inversion — the foundation should serve the research claim, not the other way around.
- [ ] Does the design give the model an easy path that bypasses the mechanism being tested? Shortcuts (residual connections in contrastive paths, identity mappings, geometric proxies for temporal structure) can make the model succeed without actually learning the intended thing.
- [ ] Could a simple baseline achieve the same result through a shortcut the system inadvertently exposes? If yes, the system's claimed contribution is not what it appears to be.
- [ ] **Re-read foundational substrate assumptions whenever the architectural framing shifts.** Inherited collection parameters from prior projects are SCAFFOLDING by default and require explicit re-justification against the current architecture's claims. Added 2026-05-13 after the 30-frame static dwell from a prior Stage-0b experiment survived three drafts of v0 instructions, an adversarial review, and a session-1 CC pre-flight before being caught in session 3 by literal scrutiny of what each collection step was actually doing. Pattern: substrate assumptions can survive abstract reading and only surface under "what is this step actually producing?" questioning when the upstream architecture changes.

- [ ] **When an architectural framing shifts, every inherited mechanism needs re-justification — not just the directly-affected ones.** Strengthened 2026-05-14 after the fifth substrate-or-variation catch in the v0 batch: (1) Python 3.10 → 3.12 environment-version inheritance, (2) embeddings-coverage assumption (dwell-only vs full-stream), (3) 30-frame static dwell, (4) cross-loop apex pose-determinism floor, and (5) per-frame jitter necessity. The pattern: assumption N+1 is shielded from review by assumption N — once "we need to fix the substrate's within-loop determinism" was accepted, "we therefore also need jitter to fix across-loop determinism" rode along uninspected. When framing changes, walk the full chain of inherited mechanisms and ask of each independently: "does this still solve a problem the new framing creates?" The session-4 jitter proposal solved a problem that the session-3 substrate-change *and* the curriculum framing together had already eliminated; it survived a full session of design + calibration before the experiment chat caught that the phase structure itself was the variation source the architecture's claim actually needed.

- [ ] **Per-item encoder-stability under out-of-scope perturbation is a standard preflight gate for any new substrate.** Added 2026-05-14 after the seventh substrate-or-variation catch in the v0 batch (sixth: preflight pixel-RGB cosine compressed into a too-tight dynamic range to discriminate locality at 300×300; switched to DINOv2-embedding contrast. Seventh: DiningTable viewing pose in seed-7 had line-of-sight into the LivingRoom; under LivingRoom-scoped `RandomizeMaterials` the DiningTable's DINOv2 representation shifted *more* than the LivingRoom items' did — the locality claim of the scoped perturbation broke at the encoder level, even though it held in the API's `inRoomTypes` parameter). When a curriculum relies on a scoped perturbation, the substrate verification must include: for each item the perturbation is scoped *away from*, measure the encoder's mean cosine across N draws of the perturbation; require it to sit above a threshold that admits ordinary lighting/shadow noise but rejects actual cross-scope perturbation propagation. In practice for DINOv2 CLS on a non-perturbed item under small-scope material swap, > 0.98 (cosine, n ≥ 3 draws) is workable; thresholds are SCAFFOLDING per project. Failure to pass the gate is a substrate problem, not a perturbation-mechanism problem; the response is usually a substrate-side fix (a viewing-pose change to remove line-of-sight) rather than a perturbation-side relaxation. Spec-level note for this v0 batch lives in `WEFT_INNER_PAM_v0_Spec.md` §5.6.

- [ ] **SCAFFOLDING thresholds get evaluated against what they're protecting, not adjusted by margin.** Added 2026-05-14 after a round-number guess at "loop length must stay within ±10% of the previous calibration" was tripped by a substrate-fix that pushed loop length by +13.9%. The threshold was originally chosen to protect the downstream budget arithmetic (repetition-bin coverage at the planned frame budget); when the actual arithmetic was recomputed against the +13.9% shift, the rep-bin coverage still cleared the architectural-test requirement (≥ 20 reps in the 100+ bin) by a comfortable margin. The threshold wasn't protecting against the specific harm we'd worried about; the harm didn't materialise. The right move was to evaluate the threshold against its protective purpose and accept the shift; the wrong move would have been to relax the threshold "because 13.9% feels close to 10%". Round-number thresholds are SCAFFOLDING by default; SCAFFOLDING tripped by empirical data gets evaluated against its protective purpose, not adjusted by margin.

- [ ] When a SCAFFOLDING threshold has the form 'absolute magnitude ≥ K' for an architectural property where K was guessed before empirical data on the substrate existed, the threshold is structurally vulnerable to scale mismatch. Prefer: (a) statistical-distinguishability tests over absolute-magnitude thresholds, (b) relative criteria against an observed baseline, (c) downstream architectural checks that test the actual question the threshold was approximating.

  **Corollary added 2026-05-14 (session 7), fifth instance of the absolute-magnitude pattern in the v0 batch and the first inside an *architectural-strength check* rather than a preflight / substrate-verification gate.** Instances 1–4 were preflight pixel-RGB cosine compression (caught with DINOv2 contrast substitute), the §8.4 0.05 absolute floor (restructured to ratio + Wilcoxon Reading C in session 6), the loop-length ±10% threshold (re-evaluated against the budget it was protecting), and the per-item DINOv2 stability gate's 0.98 placeholder. The fifth, in session 6's Phase 2 training, was the §8.7a G2.T2 "perturbed-item log_var widening ≥ 0.5 by loop 35" gate — anchored to "what a clearly-visible response looks like" without empirical contact with the actual perturbation magnitude. Restructured in session 7 to a trajectory-direction + shape + differential formulation at loop 100. **For any architectural-strength gate, default to trajectory-shape or relative-criterion formulations rather than absolute-magnitude formulations, unless an empirical baseline already exists from prior runs on the same substrate.** The pattern has surfaced enough times — and across enough distinct gate categories — to promote from recurring-correction to default-discipline. The cost of writing a relative or trajectory gate first is small; the cost of an absolute gate that turns out to be wrong-scale is at minimum a STOP-and-restructure cycle and at worst (when the gate looks like it passed) a silent measurement of the wrong thing.

- [ ] **Bypass flags (`forceAction`, `--no-verify`, `--force`, `ignore_errors`, etc.) are dangerous; they must be paired with explicit replacement validation, not used in lieu of it.** Added 2026-05-14 after a `forceAction=True` flag in an AI2-THOR `Teleport` call (added in session 4 to support off-grid 0.20 m close-up sweep steps that wouldn't pass AI2-THOR's 0.25 m navigation grid check) silently bypassed the controller's NavMesh + floor-snap validation. The agent base was placed at whatever y the calling code supplied — which differed between trajectory phases (close-up y = 0.901 from route metadata, transit y = 0.006 from NavMesh waypoints), producing a 0.894 m camera-elevation oscillation that survived three sessions of substrate verification (motion-continuity, cross-loop apex, DINOv2 stability — none of those checks measured raw 3D agent pose continuity; they all gated on properties that happened to be invariant to the elevation bug). The lesson: when a flag bypasses a built-in safety check, replace what the check was doing with an explicit equivalent. In this case, the floor-snap was replaced with an explicit query of the NavMesh at init time and a hard-coded y override at every Teleport — same protective property, asserted on our side instead of the framework's. Pattern to apply: whenever a `force_*` / `--no-*` / `ignore_*` flag enters the codebase, write down (a) what validation the flag is bypassing, (b) which property of the data the validation was protecting, (c) the explicit replacement check that now does that protection. Step (c) is the deliverable; without it the bypass is a silent unsafety.

---

## 16. When This Document Should Be Updated

Update this document when:
- A process pattern has been tested across three or more experiments and is ready for promotion
- An existing process item has failed and needs revision
- A new paradigm-independent discipline emerges from practice
- External tools or infrastructure change in ways that affect workflow

Do NOT update this document with:
- Architecture-specific findings — those go in project principles docs
- Specific numerical thresholds tied to a particular setup
- Environment-specific closures
- Model-specific behavioural observations

The test: does this apply to every experiment, regardless of research line? If yes, it belongs here. If no, it belongs in a project-specific doc.

---

*This document is a living reference. Update it when process items are validated, invalidated, or refined — with explicit rationale for each change.*

*Last updated: 2026-04-18*
*Source: extracted from ALAN/PAM project discipline (v1–v14.x experience) and generalised to be experiment-agnostic.*
