# Phase 2 Preflight Report (DINOv2 contrast, 2026-05-14)

Timestamp: 2026-05-14T01:46:44Z
House seed: 7

## Verdict: FAIL

## STOP-AND-REPORT TRIGGERED
- S1: observed_contrast=-0.0061 < 0.02; perturbation is not producing a meaningful DINOv2-level localisation signal. STOP and report to the experiment chat rather than chasing the threshold.
- S2: bedroom_dinov2_mean=0.9759 < 0.98; Bedroom items moved more than the locality claim allows; suggests the LivingRoom-scoped RandomizeMaterials call is having a global effect. STOP and report; do not lower G_M2 to fit.

## Gate criteria

### G_M1_mechanism_fires: PASS
- **criterion**: RandomizeMaterials(inRoomTypes=['LivingRoom']) lastActionSuccess == True

### G_M2_bedroom_dinov2_locality: FAIL
- **reason**: STOP-and-report triggered; gate not finalised

### G_M3_dinov2_contrast: FAIL
- **reason**: STOP-and-report triggered; gate not finalised

## DINOv2 measurements (gate-relevant)

- **per_item_3call_mean_before_vs_after**: {'Bed': 0.9884311358133951, 'DiningTable': 0.9447953899701437, 'Dresser': 0.9883499145507812, 'Sofa': 0.9756728212038676, 'Television': 0.9944967826207479}
- **per_item_per_call_before_vs_after**: {'Bed': {'call1': 0.9892984628677368, 'call2': 0.9896981716156006, 'call3': 0.9862967729568481}, 'DiningTable': {'call1': 0.937103807926178, 'call2': 0.9651550650596619, 'call3': 0.9321272969245911}, 'Dresser': {'call1': 0.9914512634277344, 'call2': 0.996125340461731, 'call3': 0.9774731397628784}, 'Sofa': {'call1': 0.9618289470672607, 'call2': 0.9872576594352722, 'call3': 0.9779318571090698}, 'Television': {'call1': 0.9972003698348999, 'call2': 0.9928362369537354, 'call3': 0.9934537410736084}}
- **bedroom_dinov2_mean**: 0.9759077694680957
- **livingroom_dinov2_mean**: 0.9820113678773243
- **observed_contrast**: -0.006103598409228694
- **encoder_norm_check_passed**: True

## Record-only

### per_item_re_application_dinov2
- **Dresser_call1_call2_dinov2**: 0.9913298487663269
- **Dresser_call2_call3_dinov2**: 0.9777700901031494
- **Sofa_call1_call2_dinov2**: 0.9646561741828918
- **Sofa_call2_call3_dinov2**: 0.9716571569442749

### flat_rgb_per_item_per_call
- **Bed**: {'call1': 0.996695228104227, 'call2': 0.996696285724291, 'call3': 0.9955103403995501}
- **DiningTable**: {'call1': 0.9860194772320566, 'call2': 0.9822976506724176, 'call3': 0.9785346704488898}
- **Dresser**: {'call1': 0.9661136865472268, 'call2': 0.9924700759420299, 'call3': 0.966809340184994}
- **Sofa**: {'call1': 0.9237674794746255, 'call2': 0.9970440275442823, 'call3': 0.9910325143262259}
- **Television**: {'call1': 0.9984701478003913, 'call2': 0.9878063293220158, 'call3': 0.9886813536225909}

### flat_rgb_per_item_3call_mean
- **Bed**: 0.9963006180760227
- **DiningTable**: 0.9822839327844547
- **Dresser**: 0.9751310342247502
- **Sofa**: 0.9706146737817112
- **Television**: 0.9916526102483326

### session_determinism
- **deterministic_across_sessions**: False
- **cos_dresser_sessionA_sessionB_flatrgb**: 0.959158256912879
- **cos_sofa_sessionA_sessionB_flatrgb**: 0.946389032202203
- **note**: Materials may be per-run; per-loop applied materials are recorded in phase2_collection_metadata.json by the collection script.
