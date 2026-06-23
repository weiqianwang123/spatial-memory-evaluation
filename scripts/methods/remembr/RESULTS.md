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

## Track 2 — ScanRefer (pending data)

ScanRefer annotations (`ScanRefer_filtered_*.json`) are gated and not yet
downloaded under `/home/robin_wang/ScanRefer/data/`; ScanNet scenes on NAS are raw
`.sens`. See the Track 2 section below once the data lands.

## Track 3 — OpenEQA (ScanNet scene0709_00)

OpenEQA ScanNet frames for `scene0709_00` are extracted on NAS
(`openeqa_frames/scannet-v0/002-scannet-scene0709_00`, 936 frames + 4x4 poses) and
questions are in `open-eqa-v0.json`. See the Track 3 section below once run.
