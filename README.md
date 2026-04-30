# Weft 2

Continuous-trajectory associative memory architecture (Inner PAM).

## Status

Fresh repository, post-architectural-rethink. The architecture is specified in `WEFT_INNER_PAM_v0_Spec.md`. No implementation yet.

## Layout

- `WEFT_INNER_PAM_v0_Spec.md` — canonical architecture specification.
- `CODING_STANDARDS.md` — operational discipline for implementation work.
- `research_operations_v1.md` — research process discipline (project-agnostic).
- `HANDOFF.md` — current state and next-action.
- `instructions/` — active batch instructions for CC sessions.
- `src/` — implementation code (empty).
- `tests/` — test code (empty).
- `results/` — experiment outputs (empty).
- `logs/` — training and process logs (empty).
- `archive/` — historical artifacts (empty).

## Relationship to previous repo

The previous repo at `Eridos/Weft/` contains four iterations of negative results that established the prior architecture was building the wrong thing. It stays in place as historical record. This repo is independent of it; the only shared resource is the seed 7 memory bank from the prior furniture experiment, which is read by the encoder substrate verification batch as input.
