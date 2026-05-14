# Phase 2 Extended Diagnostic Report

Timestamp: 2026-05-14T02:15:12Z
House seed: 7
Number of RandomizeMaterials draws: 6

## Per-item summary

| item | room | dinov2 mean | dinov2 std | dinov2 min | dinov2 max | flat-RGB mean |
|---|---|---:|---:|---:|---:|---:|
| Bed | Bedroom | 0.9873 | 0.0019 | 0.9834 | 0.9891 | 0.9945 |
| DiningTable | Bedroom | 0.9564 | 0.0091 | 0.9467 | 0.9739 | 0.9869 |
| Dresser | LivingRoom | 0.9843 | 0.0049 | 0.9802 | 0.9946 | 0.9759 |
| Sofa | LivingRoom | 0.9778 | 0.0054 | 0.9704 | 0.9860 | 0.9832 |
| Television | Bedroom | 0.9889 | 0.0033 | 0.9853 | 0.9939 | 0.9901 |

## Aggregate

- Bedroom DINOv2 mean: 0.9775
- LivingRoom DINOv2 mean: 0.9811
- Bedroom − LivingRoom contrast: -0.0035

## Affected Bedroom control items (mean DINOv2 < 0.98)

- **DiningTable** (vp_id=2): mean=0.9564, std=0.0091, range [0.9467, 0.9739]

## Stable Bedroom control items (mean DINOv2 ≥ 0.98)

- **Bed** (vp_id=1): mean=0.9873, std=0.0019
- **Television** (vp_id=5): mean=0.9889, std=0.0033
