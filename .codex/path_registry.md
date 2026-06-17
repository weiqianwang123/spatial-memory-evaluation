# Path Registry

Last updated: 2026-06-15

本文件集中记录当前 spatial-memory-evaluation 会用到的数据、repo、
checkpoint、prepared intermediate、memory package、result output 和工具环境路径。
路径变更时先更新这里，再改脚本/config。

## Repository Roots

| Name | Path | Purpose | Git policy |
|---|---|---|---|
| evaluation repo | `/home/robin_wang/spatial-memory-evaluation` | 当前独立 evaluation harness | commit source/config/docs only |
| old nested repo path | `/home/robin_wang/open-eqa/spatial-memory-evaluation` | 已迁移旧路径 | do not use |
| HOV-SG repo | `/home/robin_wang/HOV-SG` | HOV-SG method-native code | external method repo |
| ClawS repo | `/home/robin_wang/ClawS-SpatialRAG` | ScanNet++ depth/layout reader currently reused by HOV-SG prepare | external method repo |
| DualMap repo | `/home/robin_wang/DualMap` | source of current SAM smoke checkpoint and YOLO-World-S smoke checkpoint | external method repo |

## Data Roots

| Name | Path | Purpose | Git policy |
|---|---|---|---|
| NAS root | `/data/mondo-training-dataset` | shared heavy data | never commit |
| semantic mapping data root | `/data/mondo-training-dataset/semantic_mapping` | spatial-memory datasets/checkpoints/modules | never commit |
| ScanNet++ root | `/data/mondo-training-dataset/semantic_mapping/scannetpp` | ScanNet++ dataset root | never commit |
| current ScanNet++ scene | `/data/mondo-training-dataset/semantic_mapping/scannetpp/data/036bce3393` | current HOV-SG smoke/full scene | never commit |
| current scene RGB | `/data/mondo-training-dataset/semantic_mapping/scannetpp/data/036bce3393/iphone/rgb.mkv` | source RGB video for HOV-SG prepare | never commit |
| current scene depth | `/data/mondo-training-dataset/semantic_mapping/scannetpp/data/036bce3393/iphone/depth.bin` | source depth stream for HOV-SG prepare | never commit |
| current scene poses/intrinsics | `/data/mondo-training-dataset/semantic_mapping/scannetpp/data/036bce3393/iphone/pose_intrinsic_imu.json` | source poses/intrinsics for HOV-SG prepare | never commit |

## Shared Checkpoints And Modules

Formal runs should centralize reusable modules under:

```text
/data/mondo-training-dataset/semantic_mapping/modules/
```

See `modules.md` for module-level policy. Current known HOV-SG-related paths:

| Module | Path or identifier | Status | Notes |
|---|---|---|---|
| SAM smoke checkpoint | `/home/robin_wang/DualMap/sam_b.pt` | present | current HOV-SG smoke default, `models.sam.type=vit_b` |
| SAM formal target | `/data/mondo-training-dataset/semantic_mapping/modules/sam/vit_h/sam_vit_h_4b8939.pth` | missing | target shared formal checkpoint |
| YOLO-World smoke checkpoint | `/home/robin_wang/DualMap/yolov8s-world.pt` | present | smoke fallback only |
| YOLO-World formal target | `/data/mondo-training-dataset/semantic_mapping/modules/yolo/yolo_world/yolov8l-world.pt` | missing | formal strongest shared OV detector target; referenced by DualMap and ConceptGraphs native configs |
| DualMap default YOLO target | `/home/robin_wang/DualMap/model/yolov8l-world.pt` | missing | upstream config default; smoke overrides to `yolov8s-world.pt` |
| DualMap default FastSAM target | `/home/robin_wang/DualMap/model/FastSAM-s.pt` | missing | smoke disables FastSAM unless explicitly enabled |
| OpenCLIP smoke model | `ViT-B-32` | available through `open_clip` | current HOV-SG smoke default |
| OpenCLIP smoke pretrained tag | `laion2b_s34b_b79k` | available through `open_clip` | current HOV-SG smoke default |
| HOV-SG default OpenCLIP target | `ViT-H-14 / laion2b_s32b_b79k` | uncentralized | HOV-SG config default, heavier than smoke setup |
| shared OV prompt/evaluation label list | `spatial_memory_evaluation/assets/class_lists/detector_coverable.txt` | present | repo-controlled prompt/eval labels for shared OV detector and detector-coverable split; must match `DEFAULT_DETECTOR_COVERABLE_LABELS` |
| HOV-SG native HM3D labels | `/home/robin_wang/HOV-SG/hovsg/labels/HM3D_CountsOfObjectTypes.csv` | present | HOV-SG native `HM3DSEM_LABELS`, 1624 object types plus header |

## Python And Runtime

| Name | Path / value | Purpose |
|---|---|---|
| spatial-rag Python | `/home/robin_wang/miniforge3/envs/spatial-rag/bin/python` | current HOV-SG/ScanNet++ prepare/build runtime |
| HOV-SG device | `cuda` | HOV-SG upstream has hard-coded `.cuda()` paths |
| DualMap smoke device | `cuda` | DualMap native detection/mapping is expected to run on GPU |
| CUDA selection flag | `--cuda-visible-devices <ids>` | passed through to HOV-SG/DualMap subprocess as `CUDA_VISIBLE_DEVICES` |
| Hydra error mode | `HYDRA_FULL_ERROR=1` | set by HOV-SG build wrapper |
| HOV-SG safe crop launcher | `scripts/methods/hovsg/run_semantic_segmentation_patched.py` | wraps HOV-SG semantic segmentation and guards empty SAM crops |

## HOV-SG Step Outputs

All generated outputs below are ignored by Git.

### Step 1: Prepare HOV-SG Eval Layout

Script:

```text
scripts/methods/hovsg/prepare_eval_layout.py
```

Default output:

```text
data/hovsg_layouts/scannetpp_<scene-id>/<run-id>/
```

For the current scene:

```text
data/hovsg_layouts/scannetpp_036bce3393/<run-id>/
```

The prepared layout directory contains:

```text
color/*.jpg
depth/*.png
pose/*.txt
intrinsic/intrinsic_color.txt
intrinsic/intrinsic_depth.txt
layout_summary.json
```

`layout_summary.json` is the completion marker. Do not use a prepared layout
for HOV-SG build until this file exists.

Full-scene prepare command:

```bash
RUN_ID=hovsg-full-$(date +%Y%m%d-%H%M%S)
/home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/methods/hovsg/prepare_eval_layout.py \
  --scene-id 036bce3393 \
  --run-id "$RUN_ID" \
  --frame-stride 1 \
  --max-frames 0
```

### Step 2: Build HOV-SG Memory Package

Script:

```text
scripts/methods/hovsg/build_memory_smoke.py
```

Input prepared layout:

```text
data/hovsg_layouts/scannetpp_036bce3393/<run-id>/
```

Native HOV-SG output:

```text
data/hovsg_native/scannetpp_036bce3393/<run-id>/scannet/
```

Memory package output:

```text
memories/hovsg/scannetpp/036bce3393/<run-id>/
```

Build command:

```bash
CUDA_VISIBLE_DEVICES=0 /home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/methods/hovsg/build_memory_smoke.py \
  --scene-id 036bce3393 \
  --run-id "$RUN_ID" \
  --layout-dir "data/hovsg_layouts/scannetpp_036bce3393/$RUN_ID" \
  --cuda-visible-devices 0
```

By default this uses `scripts/methods/hovsg/run_semantic_segmentation_patched.py`
instead of calling HOV-SG `application/semantic_segmentation.py` directly. The
launcher does not change CUDA/cuDNN behavior; it only replaces empty SAM bbox
crops with blank crops so OpenCV does not fail on `cv2.resize(empty)`. Pass
`--disable-safe-crop-patch` only when testing upstream HOV-SG without this guard.

### Step 3: Eval HOV-SG Memory Package

Script:

```text
scripts/methods/hovsg/eval_memory_smoke.py
```

Input package:

```text
memories/hovsg/scannetpp/036bce3393/<run-id>/
```

Result output:

```text
results/hovsg/hovsg-memory-smoke-full/<run-id>/eval_summary.json
```

Eval command:

```bash
/home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/methods/hovsg/eval_memory_smoke.py \
  "memories/hovsg/scannetpp/036bce3393/$RUN_ID" \
  --query object \
  --query chair \
  --query table \
  --query monitor \
  --top-k 10 \
  --output "results/hovsg/hovsg-memory-smoke-full/$RUN_ID/eval_summary.json"
```

## DualMap Smoke Outputs

All generated outputs below are ignored by Git.

### Step 1: Prepare DualMap Eval Layout

Script:

```text
scripts/methods/dualmap/build_memory_smoke.py --prepare-only
```

Default output:

```text
data/dualmap_layouts/scannetpp_<scene-id>/<run-id>/
```

For the current scene:

```text
data/dualmap_layouts/scannetpp_036bce3393/<run-id>/
```

The prepared layout root contains:

```text
exported/scannetpp_036bce3393/color/*.jpg
exported/scannetpp_036bce3393/depth/*.png
exported/scannetpp_036bce3393/pose/*.txt
exported/scannetpp_036bce3393/intrinsic/intrinsic_color.txt
exported/scannetpp_036bce3393/intrinsic/intrinsic_depth.txt
scannetpp_036bce3393_dataset.yaml
layout_summary.json
```

Smoke prepare command:

```bash
RUN_ID=dualmap-smoke-$(date +%Y%m%d-%H%M%S)
/home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/methods/dualmap/build_memory_smoke.py \
  --scene-id 036bce3393 \
  --run-id "$RUN_ID" \
  --frame-stride 5 \
  --prepare-only
```

By default smoke prepare samples every 5th frame and caps at 200 sampled frames.
Pass `--max-frames 0` to use all frames selected by `--frame-stride 5`.

### Step 2: Build DualMap Memory Package

Script:

```text
scripts/methods/dualmap/build_memory_smoke.py
```

Input prepared layout:

```text
data/dualmap_layouts/scannetpp_036bce3393/<run-id>/
```

Native DualMap output:

```text
data/dualmap_native/scannetpp_036bce3393/<run-id>/scannet_scannetpp_036bce3393/map/
```

Memory package output:

```text
memories/dualmap/scannetpp/036bce3393/<run-id>/
```

Build command:

```bash
CUDA_VISIBLE_DEVICES=0 /home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/methods/dualmap/build_memory_smoke.py \
  --scene-id 036bce3393 \
  --run-id "$RUN_ID" \
  --skip-layout-export \
  --cuda-visible-devices 0
```

Smoke defaults override DualMap's missing upstream checkpoints to:

```text
yolo.model_path=/home/robin_wang/DualMap/yolov8s-world.pt
sam.model_path=/home/robin_wang/DualMap/sam_b.pt
use_fastsam=false
clip.model_name=ViT-B-32
clip.pretrained=laion2b_s34b_b79k
```

If PyTorch CUDA is visible but cuDNN conv initialization fails with
`CUDNN_STATUS_NOT_INITIALIZED`, fix the GPU/cuDNN runtime or use
`--skip-cuda-preflight` only when intentionally letting DualMap fail inside its
own runtime. Formal runs should not disable cuDNN.

### Step 3: Eval DualMap Memory Package

Script:

```text
scripts/methods/dualmap/eval_memory_smoke.py
```

Input package:

```text
memories/dualmap/scannetpp/036bce3393/<run-id>/
```

Result output:

```text
results/dualmap/dualmap-memory-smoke/<timestamp>/eval_summary.json
```

Eval command:

```bash
/home/robin_wang/miniforge3/envs/spatial-rag/bin/python \
  scripts/methods/dualmap/eval_memory_smoke.py \
  "memories/dualmap/scannetpp/036bce3393/$RUN_ID" \
  --query object \
  --query chair \
  --query table \
  --top-k 10
```

## Current Incomplete Run

The old combined build command created a partial prepared layout at:

```text
data/hovsg_layouts/scannetpp_036bce3393/smoke-full-20260615-174504/
```

It currently lacks `layout_summary.json`, so treat it as incomplete and do not
use it as a build input unless it is regenerated or manually completed and
validated.

## Path Rules

- `.codex/*.md` records paths and policy only; generated data stays outside
  `.codex`.
- Do not commit `data/`, `memories/`, `results/`, native HOV-SG outputs,
  checkpoints, frames, logs, or metric outputs.
- Every memory package should record the concrete data/checkpoint paths it used
  in `manifest.json`, `build_log.json`, or both.
- If a checkpoint is copied or symlinked into the shared module root, update both
  this file and `modules.md`.
