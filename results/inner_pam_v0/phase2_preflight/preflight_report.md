# Phase 2 Preflight Report

Timestamp: 2026-05-13T21:26:27Z
House seed: 7

## Verdict: FAIL

### 1_action_exists: PASS
- **action_return_snapshot_call1**: NoneType

### 2_per_loop_re_application: FAIL
- **criterion**: all four pairwise cosines < 0.95
- **cos_dresser_call1_call2**: 0.9739882652134874
- **cos_dresser_call2_call3**: 0.9806273313665728
- **cos_sofa_call1_call2**: 0.9815498285301999
- **cos_sofa_call2_call3**: 0.996158601203959

### 3_perturbation_locality: FAIL
- **criterion**: Bedroom items before-vs-after-LivingRoom-RandomizeMaterials cosine >= 0.999
- **bedroom_cosines**: {'Bed': 0.9955103403995501, 'DiningTable': 0.9830370902772648, 'Television': 0.9954017685831508}

### 4_livingroom_visually_changed: FAIL
- **criterion**: LivingRoom items before-vs-after cosine < 0.9
- **livingroom_cosines**: {'Dresser': 0.9688528310932186, 'Sofa': 0.9472508107165718}

### 5_session_determinism: PASS
- **deterministic_across_sessions**: False
- **cos_dresser_sessionA_sessionB**: 0.9837878278179012
- **cos_sofa_sessionA_sessionB**: 0.9880948293885979
- **note**: Materials are per-run; per-loop applied materials will be recorded in phase2_collection_metadata.json.
