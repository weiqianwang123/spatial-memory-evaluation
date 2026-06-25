# ClawS SpatialRAG 3-Track Adaptation Results

ClawS SpatialRAG is an **object spatial-memory** method (per-frame YOLO+ByteTrack
detection → depth/pose 3D projection → sqlite-vec memory of object records with
labels + 3D positions). Unlike DAAAM/ReMEmbR, ClawS exposes **native
non-interactive query APIs**, so it is scored two ways:

1. **`fixed_api`** — ClawS's native deterministic object query
   (`tools/query_object.py`, backed by `SpatialStorage` reads). This is its real
   distinguishing capability (no LLM in the loop).
2. **`tool_llm`** — the agentic comparison shared with DAAAM/ReMEmbR: an LLM
   calls ClawS's native tools over the packaged spatial memory. The exposed
   native tools are `query_spatial_memory` (semantic), `get_entity_anchor`
   (name→most-recent 3D), `retrieve_by_location` (radius spatial query), and
   `get_all_objects` (full listing) — see `tool_llm/native_tools.py`.

Packages (gitignored under `memories/`):
- T1 `036bce3393`: 183 objects from the native sqlite-vec DB
  (`build_memory_package.py` over `outputs/scannetpp_memory_036bce3393_ollama_vlm.db`).
- T2 `scene0207_00` (11 objects) / T3 `scene0709_00` (12 objects): built from
  ScanNet RGB-D layouts by driving ClawS's own `SpatialPipeline.process_frame`
  (`build_scannet_memory.py`; ollama `qwen3-embedding:0.6b` dim-1024 embeddings,
  VLM describer off, label from YOLO11). Sparse because ClawS uses a COCO-class
  detector without VLM refinement on these scenes.

## fixed_api — native deterministic query (ClawS's distinguishing capability)

| Track | Status | Result |
|---|---|---|
| 1 — object location (036bce3393, 37 q) | **supported** | success@5=0.216, success@1=0.162, MRR=0.875, **first-hit 0.12 m**, **2.3 ms/query** |
| 2 — referring (scene0207_00) | invalid | no native referring-expression resolver in ClawS |
| 3 — OpenEQA (scene0709_00) | invalid | no native QA / answer-synthesis API in ClawS |

Reading: ClawS's native API is deterministic and ~5 orders of magnitude faster
than the tool-LLM loop (2.3 ms vs minutes/query), and very precise when it hits
(0.12 m first-hit) — but recall is bounded by the COCO-class detector's coverage
of the detector-coverable label set. Track 2/3 are honestly `invalid`: ClawS has
no native referring/QA entrypoint (matches `.codex/baseline_registry.md`).

### Reproduce (fixed_api)

```bash
PKG=$(pwd)/memories/claws/scannetpp/036bce3393/claws-track1-036bce3393
python scripts/evaluate_track1.py "$PKG" --scene-id 036bce3393 --mode fixed_api \
  --output "$(pwd)/results/claws/track1-fixed_api/claws-track1-036bce3393/eval_summary.json"
```

## tool_llm — agentic comparison (LLM + ClawS native tools)

The LLM calls `query_spatial_memory` / `get_entity_anchor` /
`retrieve_by_location` / `get_all_objects` over the packaged ClawS memory.

| Track | Scene | Result |
|---|---|---|
| 1 | 036bce3393 (37 q) | _(pending — results/claws/track1-tool_llm/)_ |
| 2 | scene0207_00 (15-q subset) | _(pending — results/claws/track2-tool_llm/)_ |
| 3 | scene0709_00 (13 q) | _(pending — results/claws/track3-tool_llm-judged/)_ |

### Reproduce (tool_llm)

```bash
# build ScanNet DBs for Track 2/3 (drives ClawS SpatialPipeline.process_frame):
export LD_LIBRARY_PATH="/home/robin_wang/miniforge3/envs/spatial-rag/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
python scripts/methods/claws/build_scannet_memory.py \
  --layout-dir data/daaam_layouts/scannet_scene0207_00/daaam-track2-scene0207_00 \
  --scene-id scene0207_00 --db-path data/claws_scannet/scannet_memory_scene0207_00.db \
  --rag-config data/claws_scannet/claws_scannet_config.yaml --no-vlm
python scripts/methods/claws/build_memory_package.py --scene-id scene0207_00 \
  --db-path data/claws_scannet/scannet_memory_scene0207_00.db --run-id claws-scene0207_00 --no-crops

PKG=$(pwd)/memories/claws/scannetpp/036bce3393/claws-track1-036bce3393
python scripts/evaluate_track1.py "$PKG" --scene-id 036bce3393 --mode tool_llm \
  --llm-command 'claude -p "$(cat {prompt_path})" --output-format text --permission-mode bypassPermissions > {output_path}' \
  --output "$(pwd)/results/claws/track1-tool_llm/claws-track1-036bce3393/eval_summary.json"
```

## Unified 10-scene ScanNet results (2026-06-25)

Full results in `.codex/scannet_10scene_results.md`. ClawS (object_map):
- T1 tool_llm: success@5 0.418, **first-hit 0.330 m** (tightest), prox_top1@1m 0.583.
- T1 fixed_api: **success@5 0.472** (> tool_llm), 0.6 ms/query — its distinguishing strength.
- T2 tool_llm: acc@0.5m **0.351**, mean dist **1.458 m** (best localizer), prox@3m 0.837.
- T3 OpenEQA: LLM-Match 0.340, answered 0.916.
- Build: YOLO-World-L + set_classes(ScanNet200) + qwen3.5:4b VLM describer,
  ~0.18 s/frame, ~44 objects/scene, 5.1 MB. VLM describes only new confirmed tracks.
