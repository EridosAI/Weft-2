# Phase 2 Preflight Report (DINOv2 contrast, 2026-05-14)

Timestamp: 2026-05-14T03:21:51Z
House seed: 7

## Verdict: FAIL

## STOP-AND-REPORT TRIGGERED
- S1: observed_contrast=0.0062 < 0.02; perturbation is not producing a meaningful DINOv2-level localisation signal. STOP and report to the experiment chat rather than chasing the threshold.

## Gate criteria

### G_M1_mechanism_fires: PASS
- **criterion**: RandomizeMaterials(inRoomTypes=['LivingRoom']) lastActionSuccess == True

### G_M2_bedroom_dinov2_locality: FAIL
- **reason**: STOP-and-report triggered; gate not finalised

### G_M3_dinov2_contrast: FAIL
- **reason**: STOP-and-report triggered; gate not finalised

## DINOv2 measurements (gate-relevant)

- **per_item_3call_mean_before_vs_after**: {'Bed': 0.9846576849619547, 'DiningTable': 0.9767722288767496, 'Dresser': 0.980617622534434, 'Sofa': 0.9713314572970072, 'Television': 0.9852180679639181}
- **per_item_per_call_before_vs_after**: {'Bed': {'call1': 0.9836145639419556, 'call2': 0.9854795932769775, 'call3': 0.9848788976669312}, 'DiningTable': {'call1': 0.9785262942314148, 'call2': 0.9741145372390747, 'call3': 0.9776758551597595}, 'Dresser': {'call1': 0.9857444167137146, 'call2': 0.9812469482421875, 'call3': 0.9748615026473999}, 'Sofa': {'call1': 0.9676117897033691, 'call2': 0.9647364616394043, 'call3': 0.9816461205482483}, 'Television': {'call1': 0.979432225227356, 'call2': 0.9877017140388489, 'call3': 0.9885202646255493}}
- **bedroom_dinov2_mean**: 0.9822159939342074
- **livingroom_dinov2_mean**: 0.9759745399157206
- **observed_contrast**: 0.006241454018486858
- **encoder_norm_check_passed**: True

## Record-only

### per_item_re_application_dinov2
- **Dresser_call1_call2_dinov2**: 0.989314615726471
- **Dresser_call2_call3_dinov2**: 0.9925086498260498
- **Sofa_call1_call2_dinov2**: 0.9687811136245728
- **Sofa_call2_call3_dinov2**: 0.9595118761062622

### flat_rgb_per_item_per_call
- **Bed**: {'call1': 0.9913299372956844, 'call2': 0.9947561224112592, 'call3': 0.9946104804087039}
- **DiningTable**: {'call1': 0.9945208222472047, 'call2': 0.9891982900939734, 'call3': 0.9856094914058497}
- **Dresser**: {'call1': 0.9788544850136378, 'call2': 0.9762853899266423, 'call3': 0.9631581602176426}
- **Sofa**: {'call1': 0.9444331009662521, 'call2': 0.988791344915978, 'call3': 0.994904448052894}
- **Television**: {'call1': 0.9966239525726265, 'call2': 0.9901240529030889, 'call3': 0.9942925858305162}

### flat_rgb_per_item_3call_mean
- **Bed**: 0.9935655133718825
- **DiningTable**: 0.9897762012490093
- **Dresser**: 0.9727660117193077
- **Sofa**: 0.9760429646450414
- **Television**: 0.9936801971020772

### session_determinism
- **deterministic_across_sessions**: False
- **cos_dresser_sessionA_sessionB_flatrgb**: 0.9591186456644238
- **cos_sofa_sessionA_sessionB_flatrgb**: 0.9878902809837619
- **note**: Materials may be per-run; per-loop applied materials are recorded in phase2_collection_metadata.json by the collection script.
