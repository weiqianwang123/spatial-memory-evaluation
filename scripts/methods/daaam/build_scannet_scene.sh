#!/usr/bin/env bash
# Build ONE DAAAM memory package for a ScanNet scene from its prepared layout.
#
# Uses DAAAM's real-time FastSAM-x TensorRT segmenter (compute ~0.095 s/frame).
# DAAAM's DAM grounding describer is a separate async worker that needs ~70-90s
# to load. The patched runner (run_pipeline_patched.py) WAITS for all workers to
# report ready before the frame stream starts (SPATIAL_EVAL_WORKER_READY_TIMEOUT,
# default 600s), so the stream runs at NATIVE speed (--no-throttle) with DAM
# fully loaded -> objects get real descriptions, and the per-frame COMPUTE cost
# (processing_stats cv_avg+hydra_avg) is undistorted by any artificial delay.
#
# Usage: build_scannet_scene.sh <scene_id>
set -uo pipefail
cd /home/robin_wang/spatial-memory-evaluation

SCENE="$1"
RUN_ID="daaam-track-$SCENE"
LAYOUT="data/scannet_layouts/$SCENE/layout"
NATIVE_ROOT="data/daaam_native_scannet"
DAAAM_ENV=/home/robin_wang/miniforge3/envs/daaam

export MPLCONFIGDIR=/tmp/matplotlib-daaam XDG_CACHE_HOME=/tmp/daaam-cache PYTHONNOUSERSITE=1
export PYTHONPATH=/home/robin_wang/DAAAM/src:/home/robin_wang/daaam_colcon_ws/src/hydra/python/src:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$DAAAM_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/robin_wang/daaam_colcon_ws/install/lib:$DAAAM_ENV/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=$DAAAM_ENV/lib/libstdc++.so.6:$DAAAM_ENV/lib/libjpeg.so.8${LD_PRELOAD:+:$LD_PRELOAD}
export SPATIAL_EVAL_WORKER_READY_TIMEOUT="${SPATIAL_EVAL_WORKER_READY_TIMEOUT:-600}"

NFRAMES=$(ls "$LAYOUT/rgb" 2>/dev/null | wc -l)
if [ "$NFRAMES" -eq 0 ]; then echo "[ERR] no frames in $LAYOUT/rgb"; exit 2; fi

echo "[$(date +%H:%M:%S)] DAAAM $SCENE: $NFRAMES frames, native speed, wait-for-workers<=${SPATIAL_EVAL_WORKER_READY_TIMEOUT}s"
START=$(date +%s)
$DAAAM_ENV/bin/python scripts/methods/daaam/build_memory_smoke.py \
  --scene-id "$SCENE" \
  --layout-dir "$LAYOUT" \
  --run-id "$RUN_ID" \
  --native-output-root "$NATIVE_ROOT" \
  --daaam-python $DAAAM_ENV/bin/python \
  --daaam-segmenter native_fastsam_trt \
  --frame-stride 1 --max-frames 0 \
  --shared-module-profile formal \
  --skip-dependency-preflight --skip-postprocess \
  --package-root memories
RC=$?
echo "[$(date +%H:%M:%S)] [DAAAM_DONE] scene=$SCENE rc=$RC elapsed=$(( $(date +%s) - START ))s"
