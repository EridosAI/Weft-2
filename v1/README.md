# Weft Inner PAM v1

v1 experimental implementation. See `WEFT_INNER_PAM_v1_Spec_pass1_sections_1_to_6.md` and `WEFT_INNER_PAM_v1_Spec_pass2_sections_7_to_11.md` in this directory for architecture and implementation specification. v1 instructions document follows the spec and specifies operational procedures.

Companion documents at repo root:
- `WEFT_INNER_PAM_v0_CLOSING.md` — v0 institutional memory.
- `WEFT_INNER_PAM_v2_DESIGN_INTAKE.md` — v2 design intake.
- `WEFT_INNER_PAM_v1_DESIGN_INTAKE.md` — v1 design intake brief.

Shared infrastructure at `../shared/`:
- `shared/encoder/` — DINOv2-Large frozen encoder.
- `shared/substrate/` — AI2-THOR environment setup, ProcTHOR scenes, item bank.
- `shared/pipeline/` — frame loading, encoding pipeline, loss implementations.
