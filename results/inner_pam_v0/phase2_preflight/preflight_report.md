# Phase 2 Preflight Report (recalibrated 2026-05-14)

Timestamp: 2026-05-13T22:01:06Z
House seed: 7

## Verdict: FAIL

## Gate criteria

### G_M1_mechanism_fires: PASS
- **criterion**: RandomizeMaterials(inRoomTypes=['LivingRoom']) lastActionSuccess == True
- **action_return_snapshot_call1**: NoneType

### G_M2_bedroom_scope_locality: PASS
- **criterion**: mean Bedroom before-vs-after cosine > 0.97 (averaged across 3 random draws and 3 Bedroom items)
- **bedroom_per_item_mean_cosines**: {'Bed': 0.9947295559186943, 'DiningTable': 0.9760574480325297, 'Television': 0.9890220938407551}
- **bedroom_mean_cosine**: 0.9866030325973263
- **threshold**: 0.97

### G_M3_livingroom_bedroom_contrast: FAIL
- **criterion**: (Bedroom mean before-vs-after) - (LivingRoom mean before-vs-after) >= 0.02, each side averaged across 3 random RandomizeMaterials draws
- **livingroom_per_item_mean_cosines**: {'Dresser': 0.9663611499842634, 'Sofa': 0.9906245696415196}
- **livingroom_mean_cosine**: 0.9784928598128915
- **bedroom_mean_cosine**: 0.9866030325973263
- **contrast**: 0.008110172784434821
- **threshold**: 0.02

## Record-only measurements

### per_loop_re_application_cosines
- **criterion_note**: Not gated. Captures whether RandomizeMaterials genuinely re-randomises across consecutive calls; a value near 1.0 across all four pairs would indicate hard-cached materials.
- **cos_dresser_call1_call2**: 0.9286815533600105
- **cos_dresser_call2_call3**: 0.9622180436796011
- **cos_sofa_call1_call2**: 0.9960044795939615
- **cos_sofa_call2_call3**: 0.989406046566814

### per_item_before_vs_after_per_call
- **Bed**: {1: 0.995326877198989, 2: 0.9936370762082954, 3: 0.9952247143487984}
- **DiningTable**: {1: 0.9610685068304924, 2: 0.9792286114377647, 3: 0.987875225829332}
- **Dresser**: {1: 0.9379446729190325, 2: 0.9973846274796729, 3: 0.9637541495540848}
- **Sofa**: {1: 0.9948163435852801, 2: 0.9941491866348766, 3: 0.982908178704402}
- **Television**: {1: 0.992784088097194, 2: 0.9877926608069069, 3: 0.9864895326181646}

### per_item_before_vs_after_3call_mean
- **Bed**: 0.9947295559186943
- **DiningTable**: 0.9760574480325297
- **Dresser**: 0.9663611499842634
- **Sofa**: 0.9906245696415196
- **Television**: 0.9890220938407551

### session_determinism
- **deterministic_across_sessions**: False
- **cos_dresser_sessionA_sessionB**: 0.9457827008279531
- **cos_sofa_sessionA_sessionB**: 0.9385325830550233
- **note**: Materials are per-run; per-loop applied materials are recorded in phase2_collection_metadata.json.
