# Weft 2

Continuous-trajectory associative memory architecture (Inner PAM). This repo holds two experimental versions side by side, with shared infrastructure between them.

## Status

- **v0** is closed (verdict V2, with coupling-mechanism caveat). Archived institutional memory; see `WEFT_INNER_PAM_v0_CLOSING.md` at root and `v0/` for the source, scripts, data, and results.
- **v1** is in-progress (spec drafted; implementation pending). See `WEFT_INNER_PAM_v1_Spec_pass{1,2}_*.md` at root and `v1/` for the working directory.

## Layout

```
Weft 2/
├── README.md, HANDOFF.md, CODING_STANDARDS.md, research_operations_v1.md   # discipline at root
├── WEFT_INNER_PAM_v0_CLOSING.md, WEFT_INNER_PAM_v1_DESIGN_INTAKE.md,       # design + closing docs at root
│   WEFT_INNER_PAM_v1_Spec_pass1_*.md, WEFT_INNER_PAM_v1_Spec_pass2_*.md,
│   WEFT_INNER_PAM_v2_DESIGN_INTAKE.md, PREDICTOR_FORWARD_EXCERPT.md
├── shared/                # infrastructure used by both v0 and v1
│   ├── encoder/           # DINOv2-Large frozen encoder
│   ├── substrate/         # AI2-THOR + ProcTHOR scene + item bank
│   └── substrate/verification/   # v0-§5 substrate-verification scripts
├── v0/                    # archived: v0 source, scripts, tests, data, results, ancillary docs
│   ├── src/, scripts/, tests/, data/, results/
│   ├── WEFT_INNER_PAM_v0_Spec.md, WEFT_INNER_PAM_v0_EXPERIMENT_INSTRUCTIONS.md
│   └── archive/, instructions/, logs/, STOP_REPORT.md
└── v1/                    # in-progress: v1 working directory
    ├── src/predictor/, src/training/, src/evaluation/
    ├── scripts/, data/, results/, tests/
    └── WEFT_INNER_PAM_v1_Spec_pass{1,2}_*.md (mirrors of root copies for self-contained reference)
```

## Imports

- v0 internal: `from v0.src.config import ...`
- shared: `from shared.encoder.dinov2_encoder import ...`, `from shared.substrate.continuous_motion_env import ...`
- v1 internal (TBD): `from v1.src.predictor.inner_pam_v1_primary import ...`

The repo root must be on `sys.path` for these imports to resolve. `v0/tests/conftest.py` handles this for the v0 test suite.

## Relationship to previous repo

The previous repo at `Eridos/Weft/` contains four iterations of negative results that established the prior architecture was building the wrong thing. It stays in place as historical record. This repo is independent of it; the only shared resource is the seed-7 memory bank from the prior furniture experiment, which is read by the encoder substrate verification batch as input.
