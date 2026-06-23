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
instance ids (target + anchors). Objects are referenced by ScanNet instance id
(no 3D bbox in the json), so a caption-memory method like ReMEmbR is scored at the
**target object-name level** (does the resolved object's label match the GT
`object_name`); the IoU path stays available for any future bbox-bearing split.

- Scene: `scene0207_00` (28 distinct object types, 169 val referring queries) -
  chosen for object diversity so name-level referring is non-trivial. Frames
  extracted from NAS `.sens` to `semantic_mapping/scanents3d_frames/scene0207_00`.
- Memory: 24 Claude-captioned frames,
  `memories/remembr/scanents3d/scene0207_00/remembr-track2-scene0207_00`.
- Benchmark: `scripts/build_track2_data.py --scene-id scene0207_00` ->
  `benchmarks/track2/scanents3d/scene0207_00/referring_queries.jsonl`.
- Eval: `--mode tool_llm` over a 15-query subset spanning distinct object types
  (window, bathtub, cabinet, toilet, door, mirror, desk, monitor, ...). The LLM
  calls `retrieve_from_text` over the captions and returns referring predictions.
- Result (15-query subset): **referring_acc@1 = 0.87** (13/15; missed only
  `bathtub` and `rack`, which the captions did not name). Mean latency ~126 s/query
  (multi-step tool loops). Caveat: this is **target object-name level** accuracy -
  it measures whether the resolved object's class matches the GT `object_name`, not
  whether the correct *instance* was localized. Caption memory has no per-instance
  3D output, so it cannot disambiguate among same-class instances; a stronger,
  instance/bbox-level metric would require GT bboxes (resolvable from ScanNet
  instance annotations via the ScanEnts3D instance ids) and a method that emits
  instance predictions. See `results/remembr/track2-tool_llm/remembr-track2-scene0207_00/`.

### Reproduce (Track 2)

```bash
# extract a ScanEnts3D val scene's frames from NAS .sens (24 sampled frames)
#   -> data/scanents3d_layouts/scene0207_00/{color,pose}
python scripts/build_track2_data.py --scene-id scene0207_00      # build referring benchmark
python scripts/methods/remembr/build_memory_package.py \
  --layout-dir data/scanents3d_layouts/scene0207_00 \
  --dataset scanents3d --scene-id scene0207_00 --captioner claude --max-frames 24 \
  --run-id remembr-track2-scene0207_00
PKG=$(pwd)/memories/remembr/scanents3d/scene0207_00/remembr-track2-scene0207_00
python scripts/evaluate_track2.py "$PKG" --mode tool_llm \
  --benchmark-dir benchmarks/track2/scanents3d/scene0207_00 \
  --llm-command 'claude -p "$(cat {prompt_path})" --output-format text --permission-mode bypassPermissions > {output_path}' \
  --output "$(pwd)/results/remembr/track2-tool_llm/remembr-track2-scene0207_00/eval_summary.json"
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
