# ReMEmbR 3-Track Adaptation Results

ReMEmbR is adapted as **caption memory + LLM tool-calling**: build a caption
memory from a posed RGB-D scene (`build_memory_package.py`), then evaluate via the
`tool_llm` path where an LLM calls the native `retrieve_from_text` /
`retrieve_from_position` tools over the captions and produces the answer. The
captioner and the agent LLM both use the local Claude CLI (stand-in for ReMEmbR's
VILA captioner; faithful to ReMEmbR's `ReMEmbRAgent` retrieval loop). Generated
packages/results live under gitignored `memories/` and `results/`.

Capabilities: family `caption_memory`, `agent_access.mode = tool_llm`, all
fixed-API tracks `invalid` (ReMEmbR has no deterministic native fixed API) — it is
scored only through the tool-LLM path.

## Track 1 — Object-Level Location Query (ScanNet++ scene 036bce3393)

- Memory: 24 captioned frames (Claude captioner) from the prepared HOV-SG layout,
  `memory/captions.jsonl` in ReMEmbR `MemoryItem` shape (caption/time/position/theta).
- Eval: `--mode tool_llm`, local Claude CLI, max 3 tool calls/query, on a
  representative 8-label subset of the detector-coverable queries
  (bottle, box, cabinet, chair, keyboard, monitor, sofa, table).
- Result: `success@5 = 0.375` (cabinet, sofa, table found; bottle/box/chair/
  keyboard/monitor missed), `recall@1 = 0.0`, `mrr = 0.39`,
  `mean_first_hit_distance = 1.30 m`, mean latency ~53 s/query.
- Reading: caption memory gives coarse object location (via "where was the robot
  when it saw X") for large salient furniture, but misses small/ambiguous objects
  and has ~1.3 m position error — the expected weakness of non-geometric caption
  memory at precise object localization.

### Reproduce

```bash
PKG=$(pwd)/memories/remembr/scannetpp/036bce3393/remembr-track1-036bce3393

# build memory (Claude captioner)
python scripts/methods/remembr/build_memory_package.py \
  --layout-dir data/hovsg_layouts/scannetpp_036bce3393/<run> \
  --dataset scannetpp --scene-id 036bce3393 --captioner claude --max-frames 24 \
  --run-id remembr-track1-036bce3393

# eval (tool_llm); use an ABSOLUTE --output so {prompt_path} resolves under the
# per-query cwd
python scripts/evaluate_track1.py "$PKG" --scene-id 036bce3393 --mode tool_llm \
  --llm-command 'claude -p "$(cat {prompt_path})" --output-format text --permission-mode bypassPermissions > {output_path}' \
  --output "$(pwd)/results/remembr/track1-tool_llm/remembr-track1-036bce3393/eval_summary.json"
```

## Track 2 — Instance-Level Referring (ScanEnts3D val, scene0207_00)

ScanRefer's filtered train/val is gated; **ScanEnts3D** (https://scanents3d.github.io/)
is its public superset and is used instead. The val json
(`ScanRefer_filtered_val_ScanEnts3D.json`, on NAS at
`semantic_mapping/scanents3d/`) has GT target `object_id`/`object_name`, the
referring `description`, and an `entities` array grounding every phrase to ScanNet
instance ids (target + anchors). The json has no 3D bbox, but `object_id` is a
ScanNet instance id whose **3D box is derivable from the scan geometry**
(`track2/scannet_bbox.py`: aggregation `objectId`->segments -> segs.json
seg->verts -> `_vh_clean_2.ply` verts -> axis-aligned min/max; verified bit-exact
against ScanRefer's `load_scannet_data.py`). The benchmark builder resolves each
target's GT bbox into `target_bbox_3d`, so Track 2 reports both **target
object-name accuracy** and **distance-based localization** (`acc@0.25m` /
`acc@0.5m` = top-1 predicted position within X m of the GT object center, plus
`mean_center_distance_m`). Distance, not IoU, is used: caption-memory methods
emit a viewpoint position, not an instance bbox.

- Scene: `scene0207_00` (28 distinct object types, 169 val referring queries) -
  chosen for object diversity so name-level referring is non-trivial. Frames
  extracted from NAS `.sens` to `semantic_mapping/scanents3d_frames/scene0207_00`.
  GT bboxes resolved from the 4 ScanNet annotation files for the scene on NAS
  (`scannet/scans/scene0207_00/`: `.aggregation.json`,
  `_vh_clean_2.0.010000.segs.json`, `_vh_clean_2.ply`, `.txt`).
- Memory: 24 Claude-captioned frames,
  `memories/remembr/scanents3d/scene0207_00/remembr-track2-scene0207_00`.
- Benchmark: `scripts/build_track2_data.py --scene-id scene0207_00` ->
  `benchmarks/track2/scanents3d/scene0207_00/{referring_queries.jsonl,scene_objects.jsonl}`
  (all 169 queries get `target_bbox_3d`).
- Eval: `--mode tool_llm` over a 15-query subset spanning distinct object types
  (window, bathtub, cabinet, toilet, door, mirror, desk, monitor, ...). The LLM
  calls `retrieve_from_text` over the captions and returns referring predictions.
- Result (15-query subset): **referring_acc@1 = 0.87** (13/15; missed only
  `bathtub` and `rack`, which the captions did not name), **acc@0.25m = acc@0.5m
  = 0.0**, **mean_center_distance_m = 2.20 m**. Mean latency ~126 s/query
  (multi-step tool loops). Reading: name-level referring is strong, but
  localization is far off — caption memory emits the *robot viewpoint* position
  ("where the robot stood when it saw X"), ~2.2 m from the object itself, and has
  no per-instance 3D output, so it cannot disambiguate same-class instances or
  hit a sub-meter distance threshold. This is the expected weakness of
  non-geometric caption memory; the distance metric (and the resolved GT bboxes)
  become discriminative for instance-emitting object-memory methods
  (ConceptGraphs/DAAAM) later. See
  `results/remembr/track2-tool_llm/remembr-track2-scene0207_00/`.

### Reproduce (Track 2)

```bash
# extract a ScanEnts3D val scene's frames from NAS .sens (24 sampled frames)
#   -> data/scanents3d_layouts/scene0207_00/{color,pose}
# build referring benchmark (resolves GT target_bbox_3d from ScanNet geometry on NAS)
python scripts/build_track2_data.py --scene-id scene0207_00
python scripts/methods/remembr/build_memory_package.py \
  --layout-dir data/scanents3d_layouts/scene0207_00 \
  --dataset scanents3d --scene-id scene0207_00 --captioner claude --max-frames 24 \
  --run-id remembr-track2-scene0207_00
PKG=$(pwd)/memories/remembr/scanents3d/scene0207_00/remembr-track2-scene0207_00
python scripts/evaluate_track2.py "$PKG" --mode tool_llm \
  --benchmark-dir benchmarks/track2/scanents3d/scene0207_00 \
  --llm-command 'claude -p "$(cat {prompt_path})" --output-format text --permission-mode bypassPermissions > {output_path}' \
  --output "$(pwd)/results/remembr/track2-tool_llm/remembr-track2-scene0207_00/eval_summary.json"

# (optional) re-score a prior run's persisted predictions against the GT bboxes
# without re-invoking the LLM (deterministic; adds acc@0.25m/acc@0.5m + distance):
python scripts/methods/remembr/rescore_track2_distance.py \
  --results-dir results/remembr/track2-tool_llm/remembr-track2-scene0207_00 \
  --benchmark-dir benchmarks/track2/scanents3d/scene0207_00
```

## Track 3 — OpenEQA General QA (ScanNet scene0709_00)

- Memory: reuses the scene0709_00 caption memory (24 Claude-captioned frames from
  the OpenEQA NAS frames `openeqa_frames/scannet-v0/002-scannet-scene0709_00`),
  copied to `memories/remembr/openeqa/scene0709_00/remembr-track3-scene0709_00`.
- Benchmark: the 13 OpenEQA questions for `scene0709_00` (all 7 categories), built
  from `open-eqa-v0.json` via `scripts/build_track3_data.py` and filtered to the
  scene.
- Eval: `--mode tool_llm`, local Claude CLI as the answering agent, max 4 tool
  calls/question. The LLM calls `retrieve_from_text` over caption memory and
  returns a short answer + evidence. Scored by an LLM-Match judge (local Claude
  CLI, separate from the answering call; OpenEQA-style 1-5 rating mapped to [0,1]).
- Result: 13/13 answered. **LLM-Match = 0.65** (LLM judge, `llm_judge_available=true`;
  the transparent exact/substring fallback judge gave 0.54). By category: attribute
  1.0, world-knowledge 0.88, object-recognition 0.75, object-state 0.63,
  functional / spatial-understanding / object-localization 0.5 — caption memory +
  tool-LLM answers attribute / world-knowledge / recognition questions well but is
  weaker on spatial and localization. See `results/remembr/track3-tool_llm-judged/...`.

### Reproduce (Track 3)

```bash
PKG=$(pwd)/memories/remembr/openeqa/scene0709_00/remembr-track3-scene0709_00
python scripts/build_track3_data.py --dataset scannet          # builds all scannet Qs
# filter to scene0709_00 -> a benchmark dir with questions.jsonl + answers.jsonl
python scripts/evaluate_track3.py "$PKG" --dataset scannet --mode tool_llm \
  --benchmark-dir <scene0709-benchmark-dir> \
  --llm-command 'claude -p "$(cat {prompt_path})" --output-format text --permission-mode bypassPermissions > {output_path}' \
  --judge-command 'claude -p "$(cat {prompt_path})" --output-format text' \
  --output "$(pwd)/results/remembr/track3-tool_llm-judged/remembr-track3-scene0709_00/eval_summary.json"
```

## Unified 10-scene ScanNet results (2026-06-25)

Full results in `.codex/scannet_10scene_results.md`. ReMEmbR (caption memory) +
the two controls:
- ReMEmbR T1 success@5 0.094 / T2 acc@0.5m 0.0 (caption memory stores viewpoints,
  not object centers — strict localization ~0) BUT proximity_top1@3m 0.92 (points to
  the right region). **T3 OpenEQA LLM-Match 0.498** (beats geometric methods' 0.34-0.37).
- LLM-with-captions control: **T3 0.520** (highest), T1/T2 localization ~0.
- Multi-frame VLM control: T3 0.337, T1/T2 ~0.
- Build (faithful): qwen3.5:4b captioner (VILA substitute, ~1 caption/3s native
  cadence = stride 18), qwen3-embedding:0.6b caption embeddings, retrieve_from_text
  = embedding cosine (Milvus substitute). 23 captions/scene, 0.3 MB. No Claude in memory.
- Finding: caption memory wins recognition/QA, loses precise localization — the core
  memory-type trade-off vs DAAAM/ClawS geometric memory.
