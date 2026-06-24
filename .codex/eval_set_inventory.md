# Evaluation Set Inventory (2026-06-24)

Snapshot of what each track currently evaluates vs. how far it can expand, with
per-scene query yields. Use this to plan benchmark expansion. Counts are from the
local NAS data; "buildable" = the source data needed to build memory + queries is
present locally.

## Summary table

| Track | Metric | **Currently built (2026-06-24 expansion)** | **Expandable to (local data)** | Per-scene query yield |
|---|---|---|---|---|
| 1 object_location | scenes | **10** ScanNet++ (036bce3393 + 9) | **208** ScanNet++ scenes | — |
| 1 | queries | **121** dc (936 all) across 10 scenes | ~**2,500** dc / ~**9,100** all | mean 12 dc / 44 all per scene (range 3–37 / 14–65) |
| 2 referring | scenes | **10** ScanEnts3D (shared w/ T3) | **141** ScanEnts3D scenes (37 with local `.sens`) | mean 67, range 9–189 |
| 2 | queries | **750** (100% GT bbox resolved) | **9,508** total (≈2,500 in the 37 local-`.sens` scenes) | as above |
| 3 openeqa | scenes | **10** ScanNet (shared w/ T2) + scene0709_00 | **89** ScanNet episodes (79 with local `.sens`) + 63 HM3D (no local data) | mean 12, range 8–14 |
| 3 | questions | **121** across the 10 (+13 scene0709_00) | **1,079** ScanNet (≈960 in the 79 local scenes) + 557 HM3D | as above |

### 2026-06-24 expansion (first batch)

- **Track 1**: 10 ScanNet++ scenes `036bce3393, 076c822ecc, 079a326597, 07f5b601ee,
  07ff1c45bb, 08bbbdcc3d, 09bced689e, 0a184cf634, 0a5c013435, 0a76e06478`
  (`benchmarks/track1/scannetpp/<scene>/`). 121 detector_coverable queries.
- **Track 2 + 3 share 10 ScanNet scenes**: `scene0015_00, scene0050_00, scene0077_00,
  scene0084_00, scene0131_00, scene0193_00, scene0207_00, scene0222_00, scene0256_00,
  scene0314_00`. Downloaded the 4 ScanNet annotation files per scene from kaldir
  (now under `scannet/scans/<scene>/`); all 10 have local `.sens`. T2 =
  `benchmarks/track2/scanents3d/<scene>/` (750 referring queries, all GT-bbox
  resolved). T3 = `benchmarks/track3/openeqa/<scene>/` (121 questions).
- Track 1 uses ScanNet++ (its native dataset, separate scenes); Track 2/3 share the
  same 10 ScanNet scenes so one `.sens` frame extraction per scene serves both.
- NOT YET RUN: these are benchmark *sets* (queries + GT). Running methods over them
  (build memory per scene + tool_llm/fixed_api eval) is the next step.

## Track 1 — object location (ScanNet++)

- Source GT: `<scannetpp>/data/<scene>/scans/segments_anno.json` (`segGroups`).
  Memory build input: `<scene>/iphone/rgb.mkv` + `depth.bin` + `pose_intrinsic_imu.json`.
- **208 / 209 scene dirs** have BOTH `segments_anno.json` and `iphone/rgb.mkv`
  → fully usable. (Currently only `036bce3393` is built.)
- Query model (`track1/data.py`): **one "where is the <label>?" query per distinct
  canonical object label per scene**, in two splits:
  - `all_annotated`: every annotated label (mean ~44/scene, range 14–65).
  - `detector_coverable`: labels in the shared 37-label OV list
    (`spatial_memory_evaluation/assets/class_lists/detector_coverable.txt`) — the
    formal split (mean ~12/scene, range 3–37). `036bce3393` = 37 dc queries.
- **Expansion potential**: 208 scenes × ~12 dc labels ≈ **~2,500 detector_coverable
  queries** (≈9,100 all_annotated). No download needed — all local.
- Build: `python scripts/build_track1_data.py --scene-id <scene>`.

## Track 2 — instance referring (ScanEnts3D, ScanNet geometry for GT bbox)

- Source: `scanents3d/ScanRefer_filtered_val_ScanEnts3D.json` — **9,508 referring
  queries across 141 ScanNet scenes** (mean 67/scene, range 9–189; biggest:
  scene0645_00=189, scene0207_00=169, scene0231_00=166).
- Query model (`track2/data.py`): **one query per referring utterance** (target
  object_id/name + anchor names; GT 3D bbox resolved from ScanNet geometry via
  `track2/scannet_bbox.py`). Currently only `scene0207_00` built (169 + a 15
  distinct-object-type subset for cross-method comparison).
- **Local-data bottleneck**: scoring needs ScanNet per-scene geometry
  (`scannet/scans/<scene>/{_vh_clean_2.ply,.aggregation.json,segs.json,.txt}`) and
  memory needs `.sens` frames. Locally: **37 / 141** ScanEnts3D scenes have `.sens`
  (≈2,500 of the 9,508 queries), but only **1** (`scene0207_00`) has the `.ply`
  geometry downloaded so far. Others are per-file downloadable from
  `kaldir.vc.in.tum.de/scannet/v2/scans/<scene>/...` (gating is just a keypress;
  see `scannet_bbox.py` / memory `daaam-native-build-env`).
- **Expansion potential (no new download)**: the 37 local-`.sens` scenes
  (≈2,500 queries) once their 4 ScanNet annotation files are fetched. Full 141
  scenes (9,508 queries) needs `.sens` + geometry downloads for the other 104.
- Build: `python scripts/build_track2_data.py --scene-id <scene>`.

## Track 3 — OpenEQA general QA (ScanNet + HM3D)

- Source: `/home/robin_wang/open-eqa/data/open-eqa-v0.json` — **1,636 questions**:
  **1,079 ScanNet** (89 episodes, mean 12/episode, range 8–14) + **557 HM3D**
  (63 episodes). 7 balanced categories (object/attribute/spatial/state/functional/
  world-knowledge/localization, ~150 each).
- Query model (`track3/data.py`): **the OpenEQA questions as-is per episode**,
  filtered to one scene. Currently only `scene0709_00` built (13 Qs).
- **Local-data status**: extracted frames exist for only **1** episode
  (`002-scannet-scene0709_00`), but **79 / 89** ScanNet episodes have local `.sens`
  → frames extractable with `scripts/methods/daaam/extract_sens_frames.py`
  (≈960 questions). HM3D half (557 Qs) has **no local scene data** — needs HM3D
  download before it can be built.
- **Expansion potential (no new download)**: 79 ScanNet episodes ≈ **960 questions**
  via `.sens` frame extraction. +557 HM3D once HM3D scenes are acquired.
- Build: `scripts/build_track3_data.py` (builds all scannet Qs) then filter per scene.

## Cross-track reuse (key for cheap expansion)

- **37 ScanNet scenes** have BOTH ScanEnts3D referring queries AND OpenEQA questions
  AND a local `.sens` (e.g. scene0015_00, scene0050_00, scene0077_00, scene0084_00,
  scene0131_00, scene0193_00, **scene0207_00**, scene0222_00, scene0256_00,
  scene0314_00, ...). Extracting one scene's `.sens` frames once serves **both**
  Track 2 and Track 3 — these 37 are the cheapest high-leverage expansion set.
- Track 1 is a separate dataset (ScanNet++, not ScanNet), so its 208 scenes don't
  overlap with Track 2/3 scenes.

## Recommended expansion order (cheapest -> most setup)

1. **Track 1**: scale to N ScanNet++ scenes now — all 208 are local, no download
   (build_track1_data per scene; ~2,500 dc queries available).
2. **Track 2+3 on the 37 shared local-`.sens` ScanNet scenes**: extract `.sens`
   frames once per scene (serves both tracks); for Track 2 also fetch the 4 ScanNet
   annotation files per scene for GT bbox. ≈2,500 referring + ≈480 QA from these 37.
3. **Track 3 remaining ScanNet** (42 more local-`.sens` episodes, ≈480 more Qs).
4. **Full Track 2** (104 more scenes): download `.sens` + geometry.
5. **Track 3 HM3D** (557 Qs): acquire HM3D scenes first.

Acquisition pointers in `path_registry.md`; per-scene build commands above.
