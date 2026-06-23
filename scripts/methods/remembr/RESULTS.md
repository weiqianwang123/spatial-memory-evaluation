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

## Track 2 — ScanRefer / ScanEnts3D (deferred)

Deferred by request, pending a proper data source.

- The public ScanRefer **benchmark/test** json
  (`ScanRefer_filtered_test.json`) is on NAS at
  `/data/mondo-training-dataset/semantic_mapping/scanrefer/`; it has
  `scene_id`/`object_id`/`object_name`/`description` but **no 3D bbox** (test
  split is for leaderboard submission), so it only supports object-name-level
  referring scoring. The full filtered train/val (with bbox-resolvable instance
  ids) is gated behind a Google form.
- Preferred replacement: **ScanEnts3D** (https://scanents3d.github.io/), a public
  superset of ScanRefer with the same json fields plus an `entities` array giving
  every phrase->instance-id correspondence (target + anchors). Download is public:
  `https://scanents3d.github.io/ScanEnts3D_ScanRefer.zip` (~3.9 MB) and
  `.../ScanEnts3D_Nr3D.csv`. Objects are ScanNet **instance ids** (`"45_toaster_oven"`),
  resolvable to 3D bboxes via ScanNet instance annotations. The Track 2 data
  builder (`spatial_memory_evaluation/track2/data.py:build_track2_data`) already
  parses the shared `scene_id`/`object_id`/`object_name`/`description` fields.
- Track 2 needs a ScanNet scene with extracted frames (to build the ReMEmbR
  memory). `scene0709_00` is both an OpenEQA scene (frames on NAS) and a ScanRefer
  test scene, so it is a ready candidate once GT-with-bbox referring data lands.

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
