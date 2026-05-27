# Weft 2

Continuous-trajectory associative memory architecture (Inner PAM). This repo holds
three experimental iterations side by side (v0, v1, v2) with shared infrastructure
between them, and is the record of the Inner PAM research line.

## Status

- **v0** — closed (verdict V2, with coupling-mechanism caveat). Archived institutional
  memory; see `WEFT_INNER_PAM_v0_CLOSING.md` at root and `v0/` for source, scripts, data, results.
- **v1** — complete and frozen: the predictor scaffold that v2 imports unchanged.
  See `WEFT_INNER_PAM_v1_Spec_pass{1,2}_*.md` at root and `v1/`.
- **v2** — closed after Stage 1 of the recalibration phase. See `v2/HANDOFF.md` for the
  running record and `WEFT_INNER_PAM_v2_CLOSING.md` for the closing account.

`v0/`, `v1/`, and `shared/` are frozen at commit `58e91d7`.

## Layout

```
Weft 2/
├── README.md, HANDOFF.md, CODING_STANDARDS.md, research_operations_v1.md   # discipline at root
├── WEFT_INNER_PAM_v0_CLOSING.md, WEFT_INNER_PAM_v1_*.md, WEFT_INNER_PAM_v2_*.md   # design + closing docs
├── shared/                # frozen infrastructure used by v0/v1
│   ├── encoder/           # DINOv2-Large frozen encoder
│   └── substrate/         # AI2-THOR + ProcTHOR scene + item bank (+ verification)
├── v0/                    # archived: source, scripts, tests, data, results, ancillary docs
├── v1/                    # complete, frozen: predictor scaffold imported by v2
│   └── src/predictor/, src/training/, src/evaluation/, scripts/, results/, tests/
└── v2/                    # Inner PAM v2: synthetic substrate + property measurement,
    ├── src/               #   Phase 0/1 preflight, and the recalibration phase (Stage 1)
    ├── scripts/, tests/, results/
    └── HANDOFF.md, config.py
```

## Imports

The repo root must be on `sys.path` for intra-repo imports to resolve. v2 imports the
frozen v1 scaffold rather than copying it:

- v0 internal: `from v0.src.config import ...`
- shared: `from shared.encoder.dinov2_encoder import ...`
- v1 internal: `from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary`
- v2 internal: `from v2.src.substrate.stream_builder import build_stream`

`v0/tests/conftest.py` handles the path setup for the v0 test suite; scripts insert the
repo root themselves.

## Relationship to previous repo

The previous repo at `Eridos/Weft/` contains earlier iterations of negative results that
established the prior architecture was building the wrong thing. It stays in place as
historical record. This repository is the independent record of the Inner PAM v0/v1/v2
research line; the only shared resource is the seed-7 memory bank from the prior furniture
experiment, read by the encoder substrate-verification batch as input.
