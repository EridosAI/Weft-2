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
- No implementation code yet.

---

## Next immediate action

Run the encoder substrate verification per `instructions/ENCODER_SUBSTRATE_VERIFICATION.md`. This is a read-only diagnostic on the previous repo's seed 7 furniture-run memory bank — no new training, no new data.

The verification's verdict (Pass / Pass-after-recalibration / Borderline / Fail) determines whether v0 implementation can proceed with frozen V-JEPA 2 mean-pool as the encoder, or whether encoder substitution is needed first.

Path to previous bank (read-only access): `/mnt/c/Users/Jason/Desktop/Eridos/Weft/results/stage_0b_furniture/main/` and the corresponding memory bank snapshot location documented in the previous repo's HANDOFF.md.

---

## Operational state

- Working tree: clean except newly-created files in this bootstrap.
- Push hold: in effect (carried forward from previous repo discipline).
- No running jobs.
