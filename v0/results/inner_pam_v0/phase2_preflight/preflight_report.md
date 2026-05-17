# Phase 2 Preflight Report (DINOv2 contrast, 2026-05-14)

Timestamp: 2026-05-14T05:06:17Z
House seed: 7

## Verdict: PASS

## Gate criteria

### G_M1_mechanism_fires: PASS
- **criterion**: RandomizeMaterials(inRoomTypes=['LivingRoom']) lastActionSuccess == True

### G_M2_bedroom_dinov2_locality: PASS
- **criterion**: mean Bedroom DINOv2 before-vs-after CLS cosine > 0.98 (averaged across 3 RandomizeMaterials draws and 3 Bedroom items)
- **bedroom_per_item_dinov2_3call_mean**: {'Bed': 0.9869462251663208, 'DiningTable': 0.9816572864850363, 'Television': 0.9893996119499207}
- **bedroom_mean_cosine**: 0.986001041200426
- **threshold**: 0.98

### G_M3_dinov2_contrast: PASS
- **criterion**: (Bedroom mean - LivingRoom mean) DINOv2 cosine >= 0.5 * observed_contrast
- **livingroom_per_item_dinov2_3call_mean**: {'Dresser': 0.9837676684061686, 'Sofa': 0.977067232131958}
- **livingroom_mean_cosine**: 0.9804174502690632
- **bedroom_mean_cosine**: 0.986001041200426
- **observed_contrast**: 0.005583590931362736
- **calibration_ratio**: 0.5
- **calibrated_threshold**: 0.002791795465681368
- **note**: Threshold is set to 50% of the observed contrast (midpoint of the reviewer-authorised 40-60% range). This run trivially passes by construction; the threshold's purpose is to gate future preflight runs (or substrate changes) against a downward drop of more than 50% in the contrast.

## DINOv2 measurements (gate-relevant)

- **per_item_3call_mean_before_vs_after**: {'Bed': 0.9869462251663208, 'DiningTable': 0.9816572864850363, 'Dresser': 0.9837676684061686, 'Sofa': 0.977067232131958, 'Television': 0.9893996119499207}
- **per_item_per_call_before_vs_after**: {'Bed': {'call1': 0.9832339286804199, 'call2': 0.9898388385772705, 'call3': 0.987765908241272}, 'DiningTable': {'call1': 0.9784488677978516, 'call2': 0.9806676506996155, 'call3': 0.9858553409576416}, 'Dresser': {'call1': 0.9918203353881836, 'call2': 0.9798579216003418, 'call3': 0.9796247482299805}, 'Sofa': {'call1': 0.9711427688598633, 'call2': 0.9809942245483398, 'call3': 0.9790647029876709}, 'Television': {'call1': 0.9821088314056396, 'call2': 0.9925571084022522, 'call3': 0.9935328960418701}}
- **bedroom_dinov2_mean**: 0.986001041200426
- **livingroom_dinov2_mean**: 0.9804174502690632
- **observed_contrast**: 0.005583590931362736
- **encoder_norm_check_passed**: True

## Record-only

### per_item_re_application_dinov2
- **Dresser_call1_call2_dinov2**: 0.9782956838607788
- **Dresser_call2_call3_dinov2**: 0.9845829606056213
- **Sofa_call1_call2_dinov2**: 0.9743362665176392
- **Sofa_call2_call3_dinov2**: 0.9823113083839417

### flat_rgb_per_item_per_call
- **Bed**: {'call1': 0.9922162449646366, 'call2': 0.9967712326711758, 'call3': 0.9972288490138694}
- **DiningTable**: {'call1': 0.9742189247549113, 'call2': 0.9758177873061095, 'call3': 0.994141717256954}
- **Dresser**: {'call1': 0.9858276333369602, 'call2': 0.9525247927882241, 'call3': 0.963416670907835}
- **Sofa**: {'call1': 0.957604492688341, 'call2': 0.9916138949606419, 'call3': 0.995187686534963}
- **Television**: {'call1': 0.9891645768149775, 'call2': 0.9831706580601074, 'call3': 0.9905695779012601}

### flat_rgb_per_item_3call_mean
- **Bed**: 0.9954054422165607
- **DiningTable**: 0.9813928097726583
- **Dresser**: 0.967256365677673
- **Sofa**: 0.9814686913946487
- **Television**: 0.987634937592115

### session_determinism
- **deterministic_across_sessions**: False
- **cos_dresser_sessionA_sessionB_flatrgb**: 0.9908697885216307
- **cos_sofa_sessionA_sessionB_flatrgb**: 0.9652292440042105
- **note**: Materials may be per-run; per-loop applied materials are recorded in phase2_collection_metadata.json by the collection script.
