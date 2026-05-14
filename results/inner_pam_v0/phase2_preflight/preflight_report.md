# Phase 2 Preflight Report (DINOv2 contrast, 2026-05-14)

Timestamp: 2026-05-14T03:47:55Z
House seed: 7

## Verdict: PASS

## Gate criteria

### G_M1_mechanism_fires: PASS
- **criterion**: RandomizeMaterials(inRoomTypes=['LivingRoom']) lastActionSuccess == True

### G_M2_bedroom_dinov2_locality: PASS
- **criterion**: mean Bedroom DINOv2 before-vs-after CLS cosine > 0.98 (averaged across 3 RandomizeMaterials draws and 3 Bedroom items)
- **bedroom_per_item_dinov2_3call_mean**: {'Bed': 0.9892024993896484, 'DiningTable': 0.9790758689244589, 'Television': 0.9889542460441589}
- **bedroom_mean_cosine**: 0.9857442047860888
- **threshold**: 0.98

### G_M3_dinov2_contrast: PASS
- **criterion**: (Bedroom mean - LivingRoom mean) DINOv2 cosine >= 0.5 * observed_contrast
- **livingroom_per_item_dinov2_3call_mean**: {'Dresser': 0.9842591484387716, 'Sofa': 0.9710326989491781}
- **livingroom_mean_cosine**: 0.9776459236939748
- **bedroom_mean_cosine**: 0.9857442047860888
- **observed_contrast**: 0.008098281092113968
- **calibration_ratio**: 0.5
- **calibrated_threshold**: 0.004049140546056984
- **note**: Threshold is set to 50% of the observed contrast (midpoint of the reviewer-authorised 40-60% range). This run trivially passes by construction; the threshold's purpose is to gate future preflight runs (or substrate changes) against a downward drop of more than 50% in the contrast.

## DINOv2 measurements (gate-relevant)

- **per_item_3call_mean_before_vs_after**: {'Bed': 0.9892024993896484, 'DiningTable': 0.9790758689244589, 'Dresser': 0.9842591484387716, 'Sofa': 0.9710326989491781, 'Television': 0.9889542460441589}
- **per_item_per_call_before_vs_after**: {'Bed': {'call1': 0.9891659021377563, 'call2': 0.9886592626571655, 'call3': 0.9897823333740234}, 'DiningTable': {'call1': 0.9810835123062134, 'call2': 0.9777792692184448, 'call3': 0.9783648252487183}, 'Dresser': {'call1': 0.9918650388717651, 'call2': 0.981531023979187, 'call3': 0.9793813824653625}, 'Sofa': {'call1': 0.975105345249176, 'call2': 0.963584303855896, 'call3': 0.9744084477424622}, 'Television': {'call1': 0.9925450086593628, 'call2': 0.9811898469924927, 'call3': 0.9931278824806213}}
- **bedroom_dinov2_mean**: 0.9857442047860888
- **livingroom_dinov2_mean**: 0.9776459236939748
- **observed_contrast**: 0.008098281092113968
- **encoder_norm_check_passed**: True

## Record-only

### per_item_re_application_dinov2
- **Dresser_call1_call2_dinov2**: 0.9763787984848022
- **Dresser_call2_call3_dinov2**: 0.9921088814735413
- **Sofa_call1_call2_dinov2**: 0.965919017791748
- **Sofa_call2_call3_dinov2**: 0.9725021719932556

### flat_rgb_per_item_per_call
- **Bed**: {'call1': 0.9963987164576537, 'call2': 0.9952042810803563, 'call3': 0.9967770957882544}
- **DiningTable**: {'call1': 0.9898880908272318, 'call2': 0.9922979272142491, 'call3': 0.9873182483671147}
- **Dresser**: {'call1': 0.9919526237027293, 'call2': 0.9502218549124879, 'call3': 0.9469578346165547}
- **Sofa**: {'call1': 0.9848414025289093, 'call2': 0.9506815823236477, 'call3': 0.9822147156754382}
- **Television**: {'call1': 0.9896464582798518, 'call2': 0.996233768025457, 'call3': 0.9884017446711558}

### flat_rgb_per_item_3call_mean
- **Bed**: 0.9961266977754214
- **DiningTable**: 0.9898347554695319
- **Dresser**: 0.9630441044105907
- **Sofa**: 0.9725792335093318
- **Television**: 0.9914273236588215

### session_determinism
- **deterministic_across_sessions**: False
- **cos_dresser_sessionA_sessionB_flatrgb**: 0.9871675450235738
- **cos_sofa_sessionA_sessionB_flatrgb**: 0.9929293276358897
- **note**: Materials may be per-run; per-loop applied materials are recorded in phase2_collection_metadata.json by the collection script.
