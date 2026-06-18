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
/home/robin_wang/DualMap
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
TensorRT `.engine` files only inside `/home/robin_wang/DAAAM` or
`/home/robin_wang/DualMap`; method adapters should load the NAS/shared path and
pass it into the external method.

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

Track 1 fixed API:

```bash
/home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/evaluate_track1.py \
  memories/<method>/scannetpp/036bce3393/<run-id> \
  --scene-id 036bce3393 \
  --mode fixed_api
```

Track 2 fixed API:

```bash
/home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/evaluate_track2.py \
  memories/<method>/scannetpp/036bce3393/<run-id> \
  --scene-id 036bce3393 \
  --mode fixed_api
```

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

## DualMap

Root repo:

```text
/home/robin_wang/DualMap
```

Runtime env:

```text
/home/robin_wang/miniforge3/envs/spatial-rag
```

DualMap smoke defaults:

- shared profile: `smoke`
- frame stride default: `5`
- max frames default: `200`
- shared OV detector: YOLO-World-S smoke fallback
- shared SAM: `sam.vit_b`
- shared OpenCLIP: `openclip.vit_b_32`
- FastSAM disabled by default.

Build memory:

```bash
cd /home/robin_wang/spatial-memory-evaluation

RUN_ID=dualmap-smoke-$(date +%Y%m%d-%H%M%S)

CUDA_VISIBLE_DEVICES=0 /home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/methods/dualmap/build_memory_smoke.py \
  --scene-id 036bce3393 \
  --run-id "$RUN_ID" \
  --frame-stride 5 \
  --max-frames 200 \
  --cuda-visible-devices 0
```

Full sampled run:

```bash
RUN_ID=dualmap-full-stride5-$(date +%Y%m%d-%H%M%S)

CUDA_VISIBLE_DEVICES=0 /home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/methods/dualmap/build_memory_smoke.py \
  --scene-id 036bce3393 \
  --run-id "$RUN_ID" \
  --frame-stride 5 \
  --max-frames 0 \
  --cuda-visible-devices 0
```

Expected outputs:

```text
data/dualmap_layouts/scannetpp_036bce3393/$RUN_ID/
data/dualmap_native/scannetpp_036bce3393/$RUN_ID/
memories/dualmap/scannetpp/036bce3393/$RUN_ID/
```

DualMap eval wrapper:

```bash
/home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/methods/dualmap/eval_memory_smoke.py \
  --package-dir memories/dualmap/scannetpp/036bce3393/$RUN_ID
```

Common issue:

- `CUDNN_STATUS_NOT_INITIALIZED`: GPU/driver/cuDNN runtime is unhealthy for this
  process. Do not disable cuDNN for formal runs; restart node/driver or run on a
  healthy GPU node.

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

## ClawS

Root repo:

```text
/home/robin_wang/ClawS-SpatialRAG
```

Current adapter entrypoint:

```text
scripts/methods/claws/build_memory_package.py
```

Use this route only after confirming the ClawS native DB/memory output exists.
ClawS should record its detector, embeddings, SQLite/sqlite-vec versions, and
any VLM/caption modules in the package manifest/build log.

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

Current Track 1/2 fixed API status:

- Hydra standalone: invalid/declaration route unless a native object-memory
  package is produced.
- ReMEmbR: invalid for Track 1/2 fixed API object memory; caption-memory control
  semantics are separate from main object-memory methods.

Do not force these methods into a synthetic object table with an LLM wrapper just
to satisfy fixed API. Agentic evaluation can still receive native memory/code
when the package honestly declares its capabilities.

## Agentic Eval

Track 1/2 agentic modes:

```text
agentic_memory_only
agentic_full_access
```

For Claude Code through Bedrock:

```bash
CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-west-2 claude \
  -p "$(cat {prompt_path})" \
  --permission-mode bypassPermissions \
  --output-format text \
  --max-budget-usd 5 \
  > {output_path}
```

Evaluator example:

```bash
/home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/evaluate_track2.py \
  memories/<method>/scannetpp/036bce3393/<run-id> \
  --scene-id 036bce3393 \
  --mode agentic_full_access \
  --sandbox-root /tmp/<method>_track2_agentic_claude \
  --agent-command 'CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-west-2 claude -p "$(cat {prompt_path})" --permission-mode bypassPermissions --output-format text --max-budget-usd 5 > {output_path}'
```

Agentic full access should include:

- the memory package;
- method adapter code;
- `spatial_memory_evaluation/shared_modules`;
- the external root repo source code when available.

The agent may design its own interaction with the memory, but it must return the
required evaluator JSON.
