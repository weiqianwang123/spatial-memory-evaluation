# Method Runtime And Smoke Runbook

Last updated: 2026-06-18

This note records how method adapters are configured, where shared model
artifacts live, and how to run smoke builds/evals without modifying external
method repos.

## Core Paths

Project repo:

```text
/home/robin_wang/spatial-memory-evaluation
```

ScanNet++ data root:

```text
/data/mondo-training-dataset/semantic_mapping/scannetpp
```

Canonical shared modules / NAS root:

```text
/data/mondo-training-dataset/semantic_mapping/modules
```

Default smoke scene:

```text
036bce3393
```

Method roots:

```text
/home/robin_wang/ClawS-SpatialRAG
/home/robin_wang/HOV-SG
/home/robin_wang/concept-graphs
/home/robin_wang/DAAAM
/home/robin_wang/Hydra
/home/robin_wang/remembr
```

Runtime envs currently used:

```text
/home/robin_wang/miniforge3/envs/spatial-rag
/home/robin_wang/miniforge3/envs/daaam
```

## Shared Modules Policy

**Shared OV detector + class list (all detector-based methods).** Any method that
runs a detector must use the shared strongest OV detector AND the single shared
class prompt/eval list = the Track 1 `detector_coverable.txt`
(`spatial_memory_evaluation/assets/class_lists/detector_coverable.txt`, 37 labels;
the canonical path from `common/labels.py`). The shared-module registry already
hands this same file to daaam/hovsg/conceptgraphs as `class_names`. ClawS is wired
to it in `scripts/methods/claws/build_scannet_memory.py` (defaults: YOLO-World-L
`modules/yolo/yolo_world/yolov8l-world.pt` + `set_classes(detector_coverable)` —
ClawS's `UltralyticsBackend` does not call `set_classes` itself, so an OV model
needs the prompt applied by the driver). Method-native detector/vocabulary
overrides are `module_ablation` only, never the formal main-table result.
Detector-free methods (ReMEmbR captioner, caption/multiframe controls) are exempt.

External method repos should not be edited to load shared modules. Method
scripts under `scripts/methods/` read
`spatial_memory_evaluation/shared_modules/registry.py` and translate registry
entries into each method's native CLI/config format.

All model artifacts/checkpoints used by shared method adapters should live under
the NAS root:

```text
/data/mondo-training-dataset/semantic_mapping/modules
```

Python/C++ runtime dependencies are not checkpoints. Examples: `spark_dsg`,
`daaam`, Hydra bindings, `open_clip`, `sentence_transformers`, `ultralytics`,
`segment_anything`, `cvxpy`, `fastapi`, and `langchain_*` belong in the
selected conda env.

Current/target shared layout. Some entries are target paths and may still need
the artifact copied or exported before a run:

```text
/data/mondo-training-dataset/semantic_mapping/modules/
  sam/
    vit_b/sam_b.pt
    vit_h/sam_vit_h_4b8939.pth
  yolo/
    yolo_world/yolov8s-world.pt
    yolo_world/yolov8l-world.pt
  fastsam/
    s/FastSAM-s.pt
    s/FastSAM-s-640x480.engine
    x/FastSAM-x.pt
    x/FastSAM-x-640x480.engine
  openclip/
    ViT-B-32/laion2b_s34b_b79k/hf_cache/
    ViT-H-14/laion2b_s32b_b79k/hf_cache/
  dam/
    nvidia_DAM-3B/
  embeddings/
    sentence-transformers_sentence-t5-large/
  groundingdino/
  llm/
```

Smoke runs can use weaker shared artifacts such as SAM ViT-B and YOLO-World-S.
Formal runs should use the strongest shared route that all relevant methods can
actually run, currently targeted as SAM ViT-H and YOLO-World-L.

FastSAM also belongs under shared modules. Do not leave FastSAM `.pt` or
TensorRT `.engine` files only inside an external method repo; method adapters
should load the NAS/shared path and pass it into the external method.

Current FastSAM artifacts were created on 2026-06-18 from Ultralytics assets
v8.4.0 and TensorRT 10.16.1.11:

```text
/data/mondo-training-dataset/semantic_mapping/modules/fastsam/s/FastSAM-s.pt
/data/mondo-training-dataset/semantic_mapping/modules/fastsam/s/FastSAM-s-640x480.engine
/data/mondo-training-dataset/semantic_mapping/modules/fastsam/x/FastSAM-x.pt
/data/mondo-training-dataset/semantic_mapping/modules/fastsam/x/FastSAM-x-640x480.engine
```

Check available shared artifacts:

```bash
ROOT=/data/mondo-training-dataset/semantic_mapping/modules

find "$ROOT" -maxdepth 5 -type f \( \
  -name '*.pt' -o -name '*.pth' -o -name '*.safetensors' -o -name '*.bin' \
  -o -name '*.engine' \
\) -printf '%s %p\n' | sort -nr | head -100
```

Download missing HF artifacts:

```bash
DAAAM_ENV=/home/robin_wang/miniforge3/envs/daaam

PYTHONNOUSERSITE=1 $DAAAM_ENV/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/DAM-3B",
    local_dir="/data/mondo-training-dataset/semantic_mapping/modules/dam/nvidia_DAM-3B",
    local_dir_use_symlinks=False,
)

snapshot_download(
    repo_id="sentence-transformers/sentence-t5-large",
    local_dir="/data/mondo-training-dataset/semantic_mapping/modules/embeddings/sentence-transformers_sentence-t5-large",
    local_dir_use_symlinks=False,
)
PY
```

If `nvidia/DAM-3B` is gated, run `huggingface-cli login` first.

Prepare FastSAM shared artifacts for DAAAM native-fast runs:

```bash
ROOT=/data/mondo-training-dataset/semantic_mapping/modules
DAAAM_ROOT=/home/robin_wang/DAAAM
DAAAM_ENV=/home/robin_wang/miniforge3/envs/daaam

mkdir -p "$ROOT/fastsam/s" "$ROOT/fastsam/x" "$DAAAM_ROOT/checkpoints/fastsam"

# Put the source .pt under shared modules first. If you already have a trusted
# local copy, copy it into $ROOT/fastsam/{s,x}/. Otherwise download it yourself
# from the upstream FastSAM/Ultralytics source you trust.
cp /path/to/FastSAM-s.pt "$ROOT/fastsam/s/FastSAM-s.pt"
cp /path/to/FastSAM-x.pt "$ROOT/fastsam/x/FastSAM-x.pt"

# DAAAM's export script expects the .pt under its checkpoints/fastsam folder.
# Symlink from shared modules so the canonical artifact remains on NAS.
ln -sf "$ROOT/fastsam/s/FastSAM-s.pt" "$DAAAM_ROOT/checkpoints/fastsam/FastSAM-s.pt"
ln -sf "$ROOT/fastsam/x/FastSAM-x.pt" "$DAAAM_ROOT/checkpoints/fastsam/FastSAM-x.pt"

PYTHONNOUSERSITE=1 "$DAAAM_ENV/bin/python" "$DAAAM_ROOT/scripts/export_fastsam_trt.py" \
  --model_name FastSAM-s \
  --imgsz 480 640 \
  --half \
  --simplify \
  --device cuda:0

PYTHONNOUSERSITE=1 "$DAAAM_ENV/bin/python" "$DAAAM_ROOT/scripts/export_fastsam_trt.py" \
  --model_name FastSAM-x \
  --imgsz 480 640 \
  --half \
  --simplify \
  --device cuda:0

cp -f "$DAAAM_ROOT/checkpoints/fastsam/FastSAM-s-640x480.engine" \
  "$ROOT/fastsam/s/FastSAM-s-640x480.engine"
cp -f "$DAAAM_ROOT/checkpoints/fastsam/FastSAM-x-640x480.engine" \
  "$ROOT/fastsam/x/FastSAM-x-640x480.engine"
```

## Common Evaluation Commands

All memory packages should validate before evaluation:

```bash
cd /home/robin_wang/spatial-memory-evaluation

/home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/package/validate_memory_package.py \
  memories/<method>/scannetpp/036bce3393/<run-id>
```

> Track renaming (3-track refactor): the eval entrypoints are now
> `scripts/evaluate_track1.py` (`track1_object_location`: object-location query +
> build cost), `scripts/evaluate_track2.py` (`track2_scanrefer`), and
> `scripts/evaluate_track3.py` (`track3_openeqa`). The old separate
> memory-construction vs object-location entrypoints are merged into Track 1.

Track 1 (object location + build cost), fixed API:

```bash
/home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/evaluate_track1.py \
  memories/<method>/scannetpp/036bce3393/<run-id> \
  --scene-id 036bce3393 \
  --mode fixed_api
```

Track 1 tool-LLM (methods with native retrieval tools only):

```bash
/home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/evaluate_track1.py \
  memories/<method>/scannetpp/036bce3393/<run-id> \
  --scene-id 036bce3393 \
  --mode tool_llm \
  --llm-command '<llm transport command>'
```

Track 2 (ScanRefer) and Track 3 (OpenEQA) use `scripts/evaluate_track2.py` and
`scripts/evaluate_track3.py` with the same `--mode {fixed_api,tool_llm}` switch.
They emit a `data_unavailable` result until their datasets are acquired (see
`path_registry.md`).

Each eval writes:

```text
eval_summary.json
eval_details.json
eval_report.md
```

## HOV-SG

Root repo:

```text
/home/robin_wang/HOV-SG
```

Runtime env:

```text
/home/robin_wang/miniforge3/envs/spatial-rag
```

HOV-SG smoke defaults:

- shared profile: `smoke`
- frame stride default: `120`
- max frames default: `24`
- shared SAM: `sam.vit_b`
- shared OpenCLIP: `openclip.vit_b_32`
- no detector; object memory comes from SAM masks + CLIP features + HOV-SG graph.

Prepare layout only:

```bash
cd /home/robin_wang/spatial-memory-evaluation

RUN_ID=hovsg-smoke-$(date +%Y%m%d-%H%M%S)

CUDA_VISIBLE_DEVICES=0 /home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/methods/hovsg/build_memory_smoke.py \
  --scene-id 036bce3393 \
  --run-id "$RUN_ID" \
  --frame-stride 10 \
  --max-frames 200 \
  --cuda-visible-devices 0 \
  --prepare-only
```

Build memory:

```bash
CUDA_VISIBLE_DEVICES=0 /home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/methods/hovsg/build_memory_smoke.py \
  --scene-id 036bce3393 \
  --run-id "$RUN_ID" \
  --frame-stride 10 \
  --max-frames 200 \
  --cuda-visible-devices 0
```

Expected outputs:

```text
data/hovsg_layouts/scannetpp_036bce3393/$RUN_ID/
data/hovsg_native/scannetpp_036bce3393/$RUN_ID/
memories/hovsg/scannetpp/036bce3393/$RUN_ID/
```

Notes:

- HOV-SG has CUDA hard-coding in feature extraction paths; use a healthy GPU.
- The adapter uses `run_semantic_segmentation_patched.py` by default to guard
  empty SAM crops from crashing OpenCV resize.
- Sequential 3D mask merging is slow and mostly CPU-bound.

HOV-SG eval wrapper:

```bash
/home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/methods/hovsg/eval_memory_smoke.py \
  --package-dir memories/hovsg/scannetpp/036bce3393/$RUN_ID
```

## DAAAM

Root repo:

```text
/home/robin_wang/DAAAM
```

Runtime env:

```text
/home/robin_wang/miniforge3/envs/daaam
```

Hydra/Spark-DSG colcon workspace:

```text
/home/robin_wang/daaam_colcon_ws
```

Use these exports before DAAAM build/eval commands:

```bash
cd /home/robin_wang/spatial-memory-evaluation

DAAAM_ENV=/home/robin_wang/miniforge3/envs/daaam

export MPLCONFIGDIR=/tmp/matplotlib-daaam
export XDG_CACHE_HOME=/tmp/daaam-cache
export PYTHONPATH=/home/robin_wang/DAAAM/src:/home/robin_wang/daaam_colcon_ws/src/hydra/python/src:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/robin_wang/daaam_colcon_ws/install/lib:$DAAAM_ENV/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=$DAAAM_ENV/lib/libstdc++.so.6:$DAAAM_ENV/lib/libjpeg.so.8${LD_PRELOAD:+:$LD_PRELOAD}
```

Use `PYTHONNOUSERSITE=1` so the DAAAM env does not silently import packages from
`/home/robin_wang/.local`.

Core import check:

```bash
PYTHONNOUSERSITE=1 $DAAAM_ENV/bin/python - <<'PY'
import spark_dsg
import daaam
import open_clip
import sentence_transformers
import torch
import ultralytics
import segment_anything
import boxmot
print("core DAAAM deps OK")
PY
```

DAM grounding import check:

```bash
PYTHONNOUSERSITE=1 $DAAAM_ENV/bin/python - <<'PY'
import fastapi
import gradio
import daaam.grounding.workers.dam_grounding
print("DAM grounding imports OK")
PY
```

Known dependency fixes that were needed locally:

```bash
PYTHONNOUSERSITE=1 $DAAAM_ENV/bin/python -m pip install \
  fastapi \
  uvicorn \
  spaces \
  einops \
  cvxpy \
  'huggingface-hub>=0.34.0,<1.0' \
  langchain \
  langchain-openai \
  langchain-anthropic \
  langchain-google-genai \
  langgraph
```

Do not run the full DAAAM `requirements.txt` unless intentionally rebuilding
the env. It pins Torch/CUDA versions and may replace the current working PyTorch
install.

Prepare layout only:

```bash
RUN_ID=daaam-smoke-$(date +%Y%m%d-%H%M%S)

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 $DAAAM_ENV/bin/python \
  scripts/methods/daaam/build_memory_smoke.py \
  --scene-id 036bce3393 \
  --run-id "$RUN_ID" \
  --frame-stride 5 \
  --max-frames 50 \
  --daaam-python $DAAAM_ENV/bin/python \
  --cuda-visible-devices 0 \
  --prepare-only
```

Build memory:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 $DAAAM_ENV/bin/python \
  scripts/methods/daaam/build_memory_smoke.py \
  --scene-id 036bce3393 \
  --run-id "$RUN_ID" \
  --frame-stride 5 \
  --max-frames 50 \
  --daaam-python $DAAAM_ENV/bin/python \
  --cuda-visible-devices 0
```

This default route uses `--daaam-segmenter shared_sam`: the adapter passes the
shared SAM checkpoint from `shared_modules`. It is useful for cross-method smoke
consistency, but it is not DAAAM's realtime path.

DAAAM native FastSAM/TensorRT route:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 $DAAAM_ENV/bin/python \
  scripts/methods/daaam/build_memory_smoke.py \
  --scene-id 036bce3393 \
  --run-id "$RUN_ID" \
  --frame-stride 5 \
  --max-frames 120 \
  --target-fps 0.2 \
  --daaam-python $DAAAM_ENV/bin/python \
  --cuda-visible-devices 0 \
  --daaam-segmenter native_fastsam_trt \
  --config-overrides grounding.query_interval_frames=10 \
  --config-overrides workers.assignment_config.min_obs_per_track=2 \
  --config-overrides workers.dam_grounding_config.multi_image_min_n_masks=8
```

The native-fast route expects the shared FastSAM TensorRT engine at:

```text
/data/mondo-training-dataset/semantic_mapping/modules/fastsam/x/FastSAM-x-640x480.engine
```

For a smaller debug ablation, pass a shared S engine explicitly:

```bash
--allow-shared-module-override \
--native-fastsam-model /data/mondo-training-dataset/semantic_mapping/modules/fastsam/s/FastSAM-s-640x480.engine
```

If the run finishes with `Corrections applied: 0` and the log shows the
`GroundingWorker` was still `Loading checkpoint shards`, the RGB-D stream ended
before DAM-3B finished loading. Use a longer/slower smoke:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 $DAAAM_ENV/bin/python \
  scripts/methods/daaam/build_memory_smoke.py \
  --scene-id 036bce3393 \
  --run-id "$RUN_ID" \
  --frame-stride 5 \
  --max-frames 120 \
  --target-fps 0.2 \
  --daaam-python $DAAAM_ENV/bin/python \
  --cuda-visible-devices 0 \
  --config-overrides grounding.query_interval_frames=10 \
  --config-overrides workers.assignment_config.min_obs_per_track=2 \
  --config-overrides workers.dam_grounding_config.multi_image_min_n_masks=8
```

The adapter injects shared DAAAM artifacts from the registry:

```text
workers.dam_grounding_config.dam_model_path=/data/mondo-training-dataset/semantic_mapping/modules/dam/nvidia_DAM-3B
--sentence-embedding-model /data/mondo-training-dataset/semantic_mapping/modules/embeddings/sentence-transformers_sentence-t5-large
--daaam-segmenter native_fastsam_trt reads /data/mondo-training-dataset/semantic_mapping/modules/fastsam/x/FastSAM-x-640x480.engine
```

Expected outputs:

```text
data/daaam_layouts/scannetpp_036bce3393/$RUN_ID/
data/daaam_native/scannetpp_036bce3393/$RUN_ID/
memories/daaam/scannetpp/036bce3393/$RUN_ID/
```

The adapter runs DAAAM in headless/no-logging mode so DAAAM respects the
adapter's `--output-dir`. It also normalizes DAAAM/Hydra artifacts back into the
native output root, including:

```text
data/daaam_native/scannetpp_036bce3393/$RUN_ID/dsg.json
data/daaam_native/scannetpp_036bce3393/$RUN_ID/corrections.yaml
data/daaam_native/scannetpp_036bce3393/$RUN_ID/hydra_output/backend/dsg.json
```

If `run_pipeline.py` exits nonzero after producing a current-run DSG, the smoke
adapter normalizes the partial native output and continues by default. The
package `build_log.json` records this as an adapter warning. Add
`--strict-native-run` when you want native DAAAM failures to abort immediately.
Add `--native-verbose` when debugging the native stack traceback.

Package existing native output:

```bash
PYTHONNOUSERSITE=1 $DAAAM_ENV/bin/python \
  scripts/methods/daaam/build_memory_smoke.py \
  --scene-id 036bce3393 \
  --run-id "$RUN_ID" \
  --native-output-dir data/daaam_native/scannetpp_036bce3393/$RUN_ID \
  --skip-daaam-run \
  --daaam-python $DAAAM_ENV/bin/python
```

DAAAM eval wrapper:

```bash
PYTHONNOUSERSITE=1 $DAAAM_ENV/bin/python \
  scripts/methods/daaam/eval_memory_smoke.py \
  --package-dir memories/daaam/scannetpp/036bce3393/$RUN_ID
```

Common issues:

- Missing `fastapi`, `langchain_openai`, or incompatible `huggingface-hub`:
  install the targeted packages above.
- `No module named 'cvxpy'`: install `cvxpy` in the DAAAM env. DAAAM's
  `min_frames_max_size` assignment worker imports it when the pipeline starts.
- `TraversabilityVisualizer` or `RosMetaDataListener` registration errors:
  these come from the Hydra config, e.g.
  `/home/robin_wang/daaam_colcon_ws/src/daaam_ros/config/hydra_config/clio_dataset_khronos.yaml`.
  In the observed smoke run they were noisy non-fatal messages and Hydra still
  reported pipeline initialization success. Treat them as a headless-config
  cleanup item unless they become the final exception.
- Missing DAM-3B: cache `nvidia/DAM-3B` under shared modules and pass the local
  `workers.dam_grounding_config.dam_model_path`. The old
  `grounding.dam_grounding_config.dam_model_path` key is wrong for DAAAM's
  `PipelineConfig` dataclass and will leave the run on `nvidia/DAM-3B`.
- Missing ReID engine: keep default ReID-disabled smoke route, or pass
  `--with-reid --reid-weights <real shared artifact>` for an explicit ablation.
- Missing FastSAM engine with `--daaam-segmenter native_fastsam_trt`: export the
  TensorRT engine from the shared FastSAM `.pt` and copy it back under
  `/data/mondo-training-dataset/semantic_mapping/modules/fastsam/`. Do not
  make `/home/robin_wang/DAAAM/checkpoints/fastsam` the only source of truth.

### DAAAM 3-track build notes (2026-06-24)

- **cuDNN regression**: torch 2.11.0+cu128 in the daaam env raises
  `CUDNN_STATUS_NOT_INITIALIZED` on the first conv unless the env's bundled cuDNN
  9 libdir is on `LD_LIBRARY_PATH` (matmul works without it; only conv fails).
  Prepend it before the colcon lib dir:
  `export LD_LIBRARY_PATH=$DAAAM_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/robin_wang/daaam_colcon_ws/install/lib:$DAAAM_ENV/lib:$LD_LIBRARY_PATH`
- **Dependency preflight false-negative**: the build's `_preflight_python_deps`
  subprocess can spuriously report `ultralytics`/`boxmot` missing even though they
  import fine from the DAAAM cwd. Pass `--skip-dependency-preflight` (imports
  verified separately) to proceed.
- **postprocess regression**: `postprocess_scene_graph.py` fails on the installed
  `sentence_transformers` (rejects DAAAM's multimodal `{'description': ...}` dict).
  Build with `--skip-postprocess`. The DSG is still complete and packageable; its
  objects land in `BACKGROUND_OBJECTS` (object_count may be 0, background_object_count
  large) and `get_matching_subjects` still returns them with positions. This means
  the Track 2 deterministic semantic index isn't built, so the package falls back
  to the label index (Track 2 fixed_api still reported `supported` per-package, but
  the 3-track comparison uses tool_llm anyway).
- **ScanNet scenes (Track 2 scene0207_00, Track 3 scene0709_00)**: DAAAM's native
  exporter is ScanNet++-only, so prepare the layout with the new helpers:
  `scripts/methods/daaam/extract_sens_frames.py` (.sens -> `{idx}-rgb.png`/
  `-depth.png`/`.txt`) then `scripts/methods/daaam/export_scannet_layout.py`
  (-> `rgb/depth/pose/intrinsic/camera_info.json`, color intrinsic scaled to depth
  res), then build with `--layout-dir <prepared> --scene-id <scene>`. scene0709_00
  frames already exist at `openeqa_frames/scannet-v0/002-scannet-scene0709_00`
  (936 RGB+depth). Native scratch for these lands under `data/daaam_native_scannet/`.
- A native DSG for `036bce3393` already exists under
  `data/daaam_native/scannetpp_036bce3393/daaam-native-fastsam-full-nativegraph-20260618-153522/`
  (package Track 1 from it with `--skip-daaam-run --native-output-dir`).
- 2026-06-24 results (tool_llm, Opus 4.8): T1 success@5=0.51 / first-hit 0.32 m /
  MRR 0.97; T2 referring@1=0.40 / acc@0.5m=0.20; T3 LLM-Match=0.60. fixed_api T1
  success@5=0.27 / first-hit 0.50 m / 2.4 ms/query. See
  `scripts/methods/daaam/RESULTS.md`.

## ClawS

Root repo:

```text
/home/robin_wang/ClawS-SpatialRAG
```

Adapter entrypoints:

```text
scripts/methods/claws/build_memory_package.py   # package an existing ClawS sqlite-vec DB
scripts/methods/claws/build_scannet_memory.py   # build a DB for a plain ScanNet scene
```

ClawS is an object spatial-memory method with **native non-interactive query
APIs**, so it is scored two ways (report both):

- `--mode fixed_api`: native deterministic `tools/query_object.py:query_object`
  over `memory/object_table.jsonl`. Track 1 = supported; Track 2/3 = `invalid`
  (no native referring/QA API). Instant (~2.3 ms/query), precise (first-hit 0.12 m
  on 036bce3393). This is ClawS's distinguishing capability.
- `--mode tool_llm`: agent + ClawS native tools (`query_spatial_memory`,
  `get_entity_anchor`, `retrieve_by_location`, `get_all_objects`).

Build/package routes:

```bash
# (a) Package an existing ClawS DB (Track 1 036bce3393 has one shipped in the repo):
/home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/methods/claws/build_memory_package.py --scene-id 036bce3393 \
  --run-id claws-track1-036bce3393

# (b) Build a DB for a plain ScanNet scene (Track 2/3). Drives ClawS's own
#     SpatialPipeline.process_frame over a prepared DAAAM RGB-D layout; ollama
#     qwen3-embedding:0.6b dim-1024 embeddings. Defaults to the SHARED OV detector
#     YOLO-World-L (modules/yolo/yolo_world/yolov8l-world.pt) prompted with the
#     Track 1 detector_coverable.txt class list via set_classes (ClawS's backend
#     does not call set_classes itself), and the VLM describer ON (qwen3.5:4b,
#     since the config default qwen3.5:35b is not pulled) so it stores rich
#     `**label** description` snapshots like DAAAM. Needs spatial-rag env + cuDNN fix:
export LD_LIBRARY_PATH=/home/robin_wang/miniforge3/envs/spatial-rag/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
/home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/methods/claws/build_scannet_memory.py \
  --layout-dir data/daaam_layouts/scannet_scene0207_00/<run> --scene-id scene0207_00 \
  --db-path data/claws_scannet/scannet_memory_scene0207_00.db \
  --rag-config data/claws_scannet/claws_scannet_config.yaml
#     (--no-vlm for a fast label-only build; --detector-model / --class-list to override.)
/home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/methods/claws/build_memory_package.py --scene-id scene0207_00 \
  --db-path data/claws_scannet/scannet_memory_scene0207_00.db --run-id claws-scene0207_00 --no-crops
```

ClawS records its detector, embeddings, and SQLite/sqlite-vec versions in the
package manifest/build log. Note ScanNet DBs are sparse (COCO-class YOLO, VLM
off): 11 objects for scene0207_00, 12 for scene0709_00 vs 183 for the
036bce3393 native DB. 2026-06-24 results in `scripts/methods/claws/RESULTS.md`.

## ConceptGraphs

Root repo:

```text
/home/robin_wang/concept-graphs
```

Current status: adapter task is still in progress. Formal Track 1/2 runs should
use the shared strongest OV detector route once the exporter is implemented.
Do not compare method-native GroundingDINO/RAM/Tag2Text or different prompt
lists as the main fair-comparison result; treat those as module ablations.

## Hydra And ReMEmbR

Root repos:

```text
/home/robin_wang/Hydra
/home/robin_wang/remembr
```

Current Track 1 (`track1_object_location`) fixed API status:

- Hydra standalone: invalid/declaration route unless a native object-memory
  package is produced.
- ReMEmbR: invalid for Track 1/2/3 fixed-API object memory; caption-memory has no
  object table or object-location output. Do not force a synthetic object table
  with an LLM wrapper to satisfy fixed API.

ReMEmbR's runnable value is the **tool_llm** path (its native `ReMEmbRAgent`
retrieval loop), adapted across all 3 tracks (2026-06-24):
- Build caption memory with `scripts/methods/remembr/build_memory_package.py`
  (`--captioner claude` stands in for VILA; MemoryItem caption/time/position/theta).
- Native tools `retrieve_from_text`/`retrieve_from_position`/`retrieve_from_time`.
- Results (Opus 4.8): T1 success@5=0.375 / first-hit 1.30 m; T2 referring@1=0.87
  but acc@0.25m=acc@0.5m=0.0 (caption memory emits the robot viewpoint, no object
  position); T3 LLM-Match=0.65. See `scripts/methods/remembr/RESULTS.md`.
- This contrasts with DAAAM/ClawS geometric memory: caption memory wins on
  name-level recognition, loses on precise localization. Both run on the same
  Track 2/3 scenes (scene0207_00 / scene0709_00) for comparability.

The two no-explicit-memory controls (Multi-frame VLM `raw_frame_control`,
LLM-with-captions `caption_control`) also run via tool_llm (build scripts
`scripts/methods/multiframe_vlm/build_control_package.py`,
`scripts/methods/remembr/build_caption_control_package.py`) while keeping
`explicit_memory=false` — real metrics, but never promoted to object-memory
baselines (fixed_api stays `invalid` / `control_no_explicit_memory`).

## Tool-LLM Eval

Tool-LLM mode (`--mode tool_llm`) is shared by Track 1/2/3. The evaluator creates a
trace/sandbox directory containing only per-query prompts, tool specs, raw/native
memory links, and original method source links needed by the native tool runtime.
It does not copy fixed-API views, evaluation adapters, build code, benchmark GT, or
raw frames.

### 3-track adaptation pattern (2026-06-24)

The runnable adaptation for object/scene-graph/caption methods is **build memory
-> per-query LLM tool-calling** over the method's own native retrieval tools, on
one scene per track:

- Track 1 object location: ScanNet++ `036bce3393`,
  `benchmarks/track1/scannetpp/036bce3393` (37 detector_coverable queries).
- Track 2 referring: ScanEnts3D `scene0207_00`,
  `benchmarks/track2/scanents3d/scene0207_00` (+ a 15-distinct-object-type subset
  `scene0207_00_subset15` used for cross-method comparison). Distance-based
  `acc@0.25m`/`acc@0.5m` (top-1 predicted position within X m of the GT object
  center) plus `referring_acc@1/@5`; GT bboxes resolved from ScanNet geometry
  (`track2/scannet_bbox.py`).
- Track 3 OpenEQA QA: ScanNet `scene0709_00`,
  `benchmarks/track3/openeqa/scene0709_00` (13 Qs filtered from the scannet split);
  scored by an LLM-Match judge via `--judge-command`.

Per-method native tools are declared in
`spatial_memory_evaluation/tool_llm/native_tools.py`. **Principle: expose ALL of a
method's native interfaces that the packaged artifact can faithfully back, and let
the agent choose** (do not collapse to one tool, do not invent capability the
package can't serve). Current tool surfaces:

| Method (family) | Native tools exposed | Backed by |
|---|---|---|
| DAAAM (`scene_graph`) | `get_matching_subjects`, `get_objects_in_radius` | `memory/native/` corrections+object_positions+background_objects / dsg.json |
| ReMEmbR / caption_control | `retrieve_from_text`, `retrieve_from_position`, `retrieve_from_time` | `memory/captions.jsonl` |
| ClawS (`object_map`) | `query_spatial_memory`, `get_entity_anchor`, `retrieve_by_location`, `get_all_objects` | `memory/object_table.jsonl` |
| Multi-frame VLM (`raw_frame_control`) | `retrieve_frames` | `raw_links/sampled_frames.jsonl` (frame paths + pose) |

DAAAM's `get_region_information` / `get_objects_in_view` / trajectory tools are
intentionally NOT exposed: the package has no region-summary / camera-pose /
agent-layer data to back them. Tool returns are bounded (`top_k`) except ClawS
`get_all_objects`, which returns the full listing by explicit choice (fidelity over
speed). With many objects (ClawS: 183) the full dump bloats the next prompt and
the agent chains more tool steps, so ClawS tool_llm is ~2x slower than DAAAM(71
objs)/ReMEmbR(24 captions) — inherent, not a bug.

Methods with a native deterministic query API (ClawS, DAAAM via `query_object.py`)
are ALSO run in `--mode fixed_api` (instant, no LLM); report both modes. Methods
without one (ReMEmbR, controls) are `invalid` on fixed_api and only run tool_llm.

### LLM model selection (Bedrock)

The CLI runs on Bedrock (`CLAUDE_CODE_USE_BEDROCK=1`), so `--model` needs the
`us.anthropic.…` prefix. Bare `claude -p` uses the `~/.claude/settings.json`
default (currently `us.anthropic.claude-opus-4-8[1m]`). `scripts/methods/llm_presets.sh`
provides cost/speed tiers (verified on this account):

```bash
source scripts/methods/llm_presets.sh
#   haiku  -> us.anthropic.claude-haiku-4-5-20251001-v1:0  (cheapest; ~9x faster: 26s vs ~230s/query)
#   sonnet -> us.anthropic.claude-sonnet-4-6               (mid)
#   opus   -> us.anthropic.claude-opus-4-8[1m]             (default; all 2026-06-24 runs used this)
python scripts/evaluate_track1.py "$PKG" --scene-id 036bce3393 --mode tool_llm \
  --llm-command "$(llm_cmd haiku)" --output "$OUT"
```

Keep one model across a comparison set (the committed DAAAM/ClawS/ReMEmbR runs are
all Opus 4.8). Running 3+ tool_llm chains concurrently can hang Claude CLI calls;
keep <=2 concurrent.

Evaluator example (explicit Opus, mirrors committed runs):

```bash
/home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/evaluate_track2.py \
  memories/<method>/scannetpp/036bce3393/<run-id> \
  --scene-id 036bce3393 \
  --mode tool_llm \
  --llm-command 'claude -p "$(cat {prompt_path})" --output-format text --permission-mode bypassPermissions > {output_path}'
```

## Unified 10-scene ScanNet evaluation (2026-06-25)

Major pivot: ALL THREE TRACKS now share the **same 10 ScanNet scenes**, so one
memory build per (method, scene) serves Track 1+2+3. Track 1 keeps its ScanNet++
support but gained a ScanNet path. Branch `eval-10scene-unified`.

**Shared scenes:** scene0015_00, scene0050_00, scene0077_00, scene0084_00,
scene0131_00, scene0193_00, scene0207_00, scene0222_00, scene0256_00, scene0314_00.

### Track 1 ScanNet support
- `build_track1_scannet_data()` in `track1/data.py`; CLI `--dataset scannet` on
  `scripts/build_track1_data.py` + `scripts/evaluate_track1.py`. GT object
  inventory from the ScanNet aggregation (objectId->label) + axis-aligned instance
  bbox via `track2/scannet_bbox.py` (same GT geometry all 3 tracks share). Drops
  structure labels (wall/floor/ceiling) + generic tags (object/objects). Benchmarks
  under `benchmarks/track1/scannet/<scene>/`. 10 scenes => 148 detector_coverable queries.

### Shared RGB-D layout (one extraction serves every method)
- `scripts/methods/prepare_scannet_layout.sh <scene> <stride>`: extract .sens at
  stride 5 -> DAAAM layout (rgb jpg / depth png / pose / intrinsic) + `color`->`rgb`
  symlink. data/scannet_layouts/<scene>/layout/. 6 fps effective (30fps/stride5).
- FRAME EXTRACTION FIX: write color as JPEG via cv2 (~18ms) not full-res PNG
  (~450ms) -> ~6x faster (extract_sens_frames.py). depth stays uint16 PNG.

### Build configs (faithful + fair; all describers/captioners LOCAL, no Claude)
- **DAAAM**: `scripts/methods/daaam/build_scannet_scene.sh <scene>`. FastSAM-x TRT
  segmenter + DAM grounding ON + `--shared-module-profile formal` (SAM vit_h,
  OpenCLIP **ViT-H-14**, YOLO-World-L). ViT-H-14 was NOT cached + env is HF-offline
  -> downloaded to shared modules AND copied to local SSD
  `/home/robin_wang/.cache/spatial_eval_openclip` (NAS cold-read stalls the worker);
  build reads it via `SPATIAL_EVAL_OPENCLIP_CACHE_ROOT`. run_pipeline_patched.py
  patches `wait_for_workers_ready` to honor `SPATIAL_EVAL_WORKER_READY_TIMEOUT`
  (default 600s) so DAM fully loads before the stream (stock 60s timeout SIGINT'd
  it -> 0 objects). After "Results saved" the native run DEADLOCKS joining the CUDA
  grounding worker; the wrapper watches for the marker + complete out_*/dsg.json,
  kills the hang, packages from out_*/ (--skip-daaam-run). Objects land in
  BACKGROUND_OBJECTS (object_count=0, bg>0) due to --skip-postprocess; query_object.py
  falls back to background_object_table.jsonl. build_log records native per-frame
  COMPUTE cost (cv_avg+hydra_avg ~0.10-0.16 s/frame), not throttled wall-clock.
  `--dataset-tag scannet` -> memories/daaam/scannet/<scene>.
- **ClawS**: build_scannet_memory.py (YOLO-World-L + `set_classes(scannet200.txt)` +
  qwen3.5:4b VLM describer) then build_memory_package.py `--dataset-tag scannet`.
  VLM-describes only NEW confirmed tracks (~1/object), not every frame -> fast.
- **ReMEmbR** (native VILA not installed): build_memory_package.py
  `--captioner ollama --ollama-model qwen3.5:4b` (local VLM, ~VILA-3B scale, no
  Claude) + `--embed-model qwen3-embedding:0.6b` (precomputes caption embeddings)
  + `--frame-stride 18` = native ~1 caption/3s cadence (6fps layout). retrieve_from_text
  now does EMBEDDING cosine (faithful to Milvus), lexical fallback.
- **caption-control**: build_caption_control_package.py reuses ReMEmbR captions
  (+embeddings). **multiframe-VLM**: build_control_package.py `--frame-stride 18`
  (follows ReMEmbR cadence; retrieve_frames returns ALL sampled frames).

### Metrics (per user, fair to viewpoint-based caption memory)
- T1: success@{1,5}, recall@{1,5}, mrr, mean_first_hit_distance_m, **proximity@{1,3,5}m**
  (best-of-top5) + **proximity_top1@{1,3,5}m** (primary answer). Strict match
  threshold 0.5-2m size-scaled; proximity is threshold-free nearest pred->GT center.
- T2: **acc@{0.25,0.5}m** (top-1 precision) + **acc_top5@{0.25,0.5}m** (recall) +
  **proximity@{1,3,5}m** (top-1) + **proximity_top5@{1,3,5}m**. Distance-only;
  referring_acc/name-match removed.
- T3: llm_match (sonnet judge), answered_rate.

### Eval models + harness (per user)
- agent = haiku (`llm_cmd haiku`, fastest), judge = sonnet (`judge_cmd sonnet`, T3).
  ALL describers/captioners/embedders = local qwen (no Claude in memory construction).
- **DESIGN DECISION: per-query INDEPENDENT agent** (fresh context per query) — a
  persistent per-scene session was rejected as unfair (later queries see earlier
  retrievals; order-dependent; cross-track contamination).
- `scripts/methods/eval_all_scannet.sh <track> <mode> <methods-csv>` (T2 uses
  `_subset15` benchmark dirs, 15 distinct-target queries/scene). T1 fixed_api for
  ClawS+DAAAM. Aggregate: `scripts/methods/aggregate_scannet_results.py`.
- CONCURRENCY: per-scene-cell parallelism at cap ~12 (each cell = one scene's
  queries serially). cap-40 caused per-query stalls (16-min hung agents); cap-12
  is healthy (~36 agents, ~4min max/query). >~12 concurrent risks CLI hangs.

### Scope (user decision)
- T1 full (148 dc q), T2 SUBSET15 (150 q), T3 full (121 q), x5 methods, stride 5.
- Results gitignored under results/<m>/track<N>-<mode>/scannet-<scene>/.
