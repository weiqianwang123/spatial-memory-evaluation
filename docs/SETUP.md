# Setup Guide

How to configure the benchmark and each baseline so the evaluators run correctly.
There are **three layers**: (1) the evaluation env, (2) shared perception/LLM modules,
(3) each baseline method's own repo/env. The auto-designed memory only needs layers 1–2.

> Paths below are the ones used on the development machine
> (`/home/robin_wang/...`, NAS at `/data/mondo-training-dataset/...`). On another
> machine, keep the **relative layout** the same and update the absolute roots in
> `.codex/path_registry.md` + the env files. Do not hardcode a new user's `$HOME`
> into committed configs.

---

## Layer 1 — Evaluation environment

The evaluators + the auto-design harness run in one conda env. The dev machine's
env is named **`spatial-rag`** (a pre-existing SpatialRAG env that the deps were
installed into); `environment.evaluation.yml` calls it `spatial-memory-eval` for a
clean-room install. Either name works — just be consistent.

```bash
# Option A — fresh env from the spec (recommended for a new machine)
conda env create -f environment.evaluation.yml     # creates env "spatial-memory-eval", python 3.10
conda activate spatial-memory-eval
pip install -e .                                     # installs the spatial_memory_evaluation package

# Option B — install into an existing env (what the dev machine did)
conda activate spatial-rag
pip install -r requirements.evaluation.txt
pip install -e .
```

**Caveat about the env files (read this):** `environment.evaluation.yml` /
`requirements.evaluation.txt` currently pin two baseline repos by **absolute local
path** (`-e /home/robin_wang/ClawS-SpatialRAG`, `-e /home/robin_wang/HOV-SG`). Those
lines are only needed if you run the ClawS / HOV-SG baselines (Layer 3) — for the
**benchmark evaluators + the auto-designed memory alone, they are NOT required**.
On a new machine, clone those repos first (Layer 3) and edit the paths, or comment
them out if you are not running those baselines.

Core deps (independent of any baseline): numpy, scipy, torch≥2.2, transformers,
sentence-transformers, ultralytics≥8.2, opencv, pillow, matplotlib, sqlite-vec,
openai, tenacity, pyyaml, tqdm, natsort.

---

## Layer 2 — Shared modules + LLM stack (fair comparison)

All detector-based methods share ONE open-vocab detector and ONE class list so
comparisons are fair. Shared modules live on NAS:

```
/data/mondo-training-dataset/semantic_mapping/modules/
  yolo/yolo_world/yolov8l-world.pt      # shared OV detector (formal). yolov8s-world.pt = smoke fallback
  sam/vit_h/sam_vit_h_4b8939.pth        # SAM (HOV-SG / ConceptGraphs / DAAAM); vit_b/sam_b.pt = smoke
  fastsam/x/FastSAM-x-640x480.engine    # DAAAM native-fast segmenter (TensorRT)
  dam/nvidia_DAM-3B                     # DAAAM grounding describer (cached)
  embeddings/sentence-transformers_sentence-t5-large
```
Shared class list (repo-controlled): `spatial_memory_evaluation/assets/class_lists/scannet200.txt`
(and `detector_coverable.txt` for the Track 1 detector-coverable split).

**ENV quirk:** YOLO-World's GPU forward needs cudnn disabled *before* importing
ultralytics: `import torch; torch.backends.cudnn.enabled = False`.

**Local LLM stack (Ollama)** — used by the agent-designed memory + ReMEmbR captioner:
```bash
ollama pull qwen3.5:4b            # 4.7B vision describer (VILA substitute), Q4_K_M
ollama pull qwen3-embedding:0.6b  # 0.6B text embeddings, 1024-d, Q8_0
# ollama serves at http://localhost:11434
```

**Cloud LLM (Bedrock)** — the per-query answering agent + the Track 3 LLM-Match judge:
```bash
export CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-west-2
#   answering agent : us.anthropic.claude-haiku-4-5-20251001-v1:0
#   Track 3 judge   : us.anthropic.claude-sonnet-4-6
# (model ids + command templates in scripts/methods/llm_presets.sh)
```

---

## Layer 3 — Baselines

Each baseline is an **external method repo** cloned next to this one; we adapt its
*native* memory onto the package contract (no re-implementation). Clone them, then
build a package per scene, then evaluate.

Data prep for all ScanNet baselines (once per scene): extract the RGB-D layout
`scripts/methods/prepare_scannet_layout.sh <scene> <stride>` and build the track
GT with `scripts/build_track{1,2,3}_data.py`.

### ClawS / SpatialRAG (`object_map`)
- Repo: `/home/robin_wang/ClawS-SpatialRAG` (installed `-e` into the eval env).
- Extra dep: `sqlite-vec` (already in the eval env).
- Build: `scripts/methods/claws/build_scannet_memory.py` drives ClawS's real
  `SpatialPipeline.process_frame` over the layout (YOLO-World-L + set_classes(ScanNet200)
  + qwen3.5:4b VLM describer, via Ollama) → sqlite-vec DB; then
  `scripts/methods/claws/build_memory_package.py` packages it. No separate env.

### ReMEmbR (`caption`)
- Repo: `/home/robin_wang/remembr` (NaVQA/CODa data + reference eval live here).
- ReMEmbR's native stack (VILA captioner + Milvus + LangGraph) is **not installed**;
  we keep its exact `MemoryItem(caption,time,position,theta)` shape and reproduce its
  `retrieve_from_text/position` tools. Captioner backend is pluggable
  (`--captioner claude` default, or `ollama`, or `none`). No separate env.
- Build: `scripts/methods/remembr/build_memory_package.py --layout-dir ... --captioner claude`.

### HOV-SG (`get_object` baseline)
- Repo: `/home/robin_wang/HOV-SG` (installed `-e`). Uses SAM + OpenCLIP + the shared
  detector. Smoke default: SAM vit_b, OpenCLIP ViT-B-32/laion2b; formal: SAM vit_h,
  ViT-H-14. Build via `scripts/methods/hovsg/build_memory_smoke.py`.

### DAAAM (`scene_graph`) — the hard one, needs a SEPARATE env
DAAAM builds a Hydra Dynamic Scene Graph and needs a dedicated conda env plus a
colcon-built Hydra/Spark-DSG workspace and a TensorRT FastSAM engine. This is the
most involved baseline.

**Repos / workspaces:**
```
/home/robin_wang/DAAAM                 # DAAAM source (PYTHONPATH: DAAAM/src)
/home/robin_wang/daaam_colcon_ws       # colcon workspace with hydra_python bindings
/home/robin_wang/miniforge3/envs/daaam # dedicated conda env (python 3.10, torch cu128)
```

**Required env exports BEFORE any DAAAM build/eval** (the build subprocess copies
`os.environ`, so these must be set in the launching shell):
```bash
DAAAM_ENV=/home/robin_wang/miniforge3/envs/daaam
export MPLCONFIGDIR=/tmp/matplotlib-daaam XDG_CACHE_HOME=/tmp/daaam-cache PYTHONNOUSERSITE=1
export PYTHONPATH=/home/robin_wang/DAAAM/src:/home/robin_wang/daaam_colcon_ws/src/hydra/python/src:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$DAAAM_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/robin_wang/daaam_colcon_ws/install/lib:$DAAAM_ENV/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=$DAAAM_ENV/lib/libstdc++.so.6:$DAAAM_ENV/lib/libjpeg.so.8${LD_PRELOAD:+:$LD_PRELOAD}
```
Why each:
- **cuDNN on LD_LIBRARY_PATH** — torch 2.11+cu128 raises `CUDNN_STATUS_NOT_INITIALIZED`
  on the first conv unless the env's bundled cuDNN-9 libdir is on the path.
- **hydra_python on PYTHONPATH** — DAAAM's `HydraPipelineRunner` needs the colcon-built
  `hydra_python` bindings (NOT pip-installed; sourcing `install/setup.bash` does not set
  PYTHONPATH). Their C++ deps (`libhydra.so`/`libkhronos.so`/gtsam in
  `daaam_colcon_ws/install/lib`, `libyaml-cpp.so.0.8` in the env lib) must be on
  LD_LIBRARY_PATH. Duplicate-type-registration `[ERROR]` lines on import are non-fatal.

**Models (on NAS):** FastSAM-x TensorRT engine + DAM-3B grounding describer + SAM +
sentence-t5-large (see Layer 2). DAM-3B is cached; the grounding worker also needs
`fastapi`/`gradio` in the DAAAM env (installed).

**ScanNet build flow** (DAAAM's native exporter is ScanNet++-only, so for ScanNet):
```bash
# 1. extract .sens -> frames, then to a posed RGB-D layout
python scripts/methods/daaam/extract_sens_frames.py --sens <scene>.sens --output-dir <frames>
python scripts/methods/daaam/export_scannet_layout.py ...     # -> rgb/depth/pose/intrinsic
# 2. native DSG build (FastSAM-x + DAM grounding + Hydra), skipping the flaky preflight
python scripts/methods/daaam/build_memory_smoke.py --layout-dir <layout> --scene-id <scene> \
    --daaam-python $DAAAM_ENV/bin/python --shared-module-profile formal \
    --skip-dependency-preflight
```
Known quirk: after "Results saved", the native process can deadlock joining the
CUDA-holding grounding worker — the `out_*/` artifacts are complete at that point, so
`scripts/methods/daaam/build_scannet_scene.sh` watches for the save marker, kills the
hung process, and packages from `out_*/`. (Also: for OC-NaVQA / Track 4 on CODa, DAAAM
needs stereo depth from cam0+cam1 via FoundationStereo — see `.codex/track4_oc_navqa_findings.md`.)

### Controls (no separate repo)
- **Multi-frame VLM** and **LLM-with-captions** are `explicit_memory=false` controls
  built by `scripts/methods/multiframe_vlm/` and `scripts/methods/remembr/build_caption_control_package.py`.

---

## Run it

```bash
# validate a built package
python -m spatial_memory_evaluation.memory_package_validator <package_dir>

# all methods × 10 held-out ScanNet scenes × 3 tracks (tool_llm)
bash scripts/methods/eval_all_scannet.sh all tool_llm daaam,claws,remembr,remembr_captions,multiframe_vlm

# the auto-designed memory (Layers 1–2 only) — see the root README "auto-design" section
```

See `.codex/path_registry.md` for the authoritative path list, `.codex/modules.md` for
the shared-module policy, and `.codex/baseline_registry.md` for per-baseline API support.
