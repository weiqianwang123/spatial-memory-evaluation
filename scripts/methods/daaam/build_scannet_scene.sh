#!/usr/bin/env bash
# Build ONE DAAAM memory package for a ScanNet scene from its prepared layout,
# faithfully (DAAAM defaults, DAM grounding ON, formal modules), then package.
#
# DAAAM is a realtime SLAM-style system: the DAM grounding describer is an async
# worker. The patched runner waits for all workers to fully load before streaming
# (SPATIAL_EVAL_WORKER_READY_TIMEOUT), then runs the FastSAM-x TRT segmenter at
# native speed; the async DAM worker keeps pace (backpressure throttles the feed
# naturally) and writes rich descriptions into out_*/dsg.json + corrections.yaml.
# The fair per-frame COMPUTE cost is processing_stats cv_avg+hydra_avg.
#
# KNOWN ISSUE: after "Results saved", the native process can deadlock joining the
# CUDA-holding grounding worker subprocess. The native artifacts (out_*/) are
# already COMPLETE at that point, so we watch for the save marker, then kill the
# hung native run and package from the out_* dir (mirrors the reference build).
#
# Usage: build_scannet_scene.sh <scene_id>
set -uo pipefail
cd /home/robin_wang/spatial-memory-evaluation

SCENE="$1"
RUN_ID="daaam-track-$SCENE"
LAYOUT="data/scannet_layouts/$SCENE/layout"
NATIVE_ROOT="data/daaam_native_scannet"
NATIVE_DIR="$NATIVE_ROOT/scannet_${SCENE}/$RUN_ID"
DAAAM_ENV=/home/robin_wang/miniforge3/envs/daaam
PKG="memories/daaam/scannet/$SCENE/$RUN_ID"

export MPLCONFIGDIR=/tmp/matplotlib-daaam XDG_CACHE_HOME=/tmp/daaam-cache PYTHONNOUSERSITE=1
export PYTHONPATH=/home/robin_wang/DAAAM/src:/home/robin_wang/daaam_colcon_ws/src/hydra/python/src:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$DAAAM_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/robin_wang/daaam_colcon_ws/install/lib:$DAAAM_ENV/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=$DAAAM_ENV/lib/libstdc++.so.6:$DAAAM_ENV/lib/libjpeg.so.8${LD_PRELOAD:+:$LD_PRELOAD}
export SPATIAL_EVAL_WORKER_READY_TIMEOUT="${SPATIAL_EVAL_WORKER_READY_TIMEOUT:-600}"
# Fast local-SSD OpenCLIP cache (NAS cold reads of the 3.9GB ViT-H-14 stall the
# grounding worker). Falls back to the shared NAS cache if this root is absent.
export SPATIAL_EVAL_OPENCLIP_CACHE_ROOT="${SPATIAL_EVAL_OPENCLIP_CACHE_ROOT:-/home/robin_wang/.cache/spatial_eval_openclip}"

NFRAMES=$(ls "$LAYOUT/rgb" 2>/dev/null | wc -l)
if [ "$NFRAMES" -eq 0 ]; then echo "[ERR] no frames in $LAYOUT/rgb"; exit 2; fi

rm -rf "$NATIVE_DIR" "$PKG" 2>/dev/null
echo "[$(date +%H:%M:%S)] DAAAM $SCENE: $NFRAMES frames, native FastSAM-x + DAM grounding ON"
START=$(date +%s)

# --- Phase 1: native run (background; watch for completion marker) ---
LOG="/tmp/daaam_native_${SCENE}.log"
$DAAAM_ENV/bin/python scripts/methods/daaam/build_memory_smoke.py \
  --scene-id "$SCENE" \
  --dataset-tag scannet \
  --layout-dir "$LAYOUT" \
  --run-id "$RUN_ID" \
  --native-output-root "$NATIVE_ROOT" \
  --daaam-python $DAAAM_ENV/bin/python \
  --daaam-segmenter native_fastsam_trt \
  --frame-stride 1 --max-frames 0 \
  --shared-module-profile formal \
  --skip-dependency-preflight --skip-postprocess \
  --package-root memories \
  > "$LOG" 2>&1 &
BUILD_PID=$!

# Watch for the native "Results saved" marker + a complete out_*/dsg.json. Once
# present, the package data exists; if the process then hangs on worker teardown,
# kill it and package from out_*/.
SAVED=0
for _ in $(seq 1 240); do  # up to ~60 min for the largest scenes
  if ! kill -0 "$BUILD_PID" 2>/dev/null; then echo "[$(date +%H:%M:%S)] native process exited on its own"; break; fi
  if grep -q "Results saved to:" "$LOG" 2>/dev/null; then
    OUT=$(ls -d "$NATIVE_DIR"/out_* 2>/dev/null | head -1)
    if [ -n "$OUT" ] && [ -f "$OUT/dsg.json" ] && [ -f "$OUT/corrections.yaml" ]; then
      SAVED=1
      sleep 8  # let any final save flush
      if kill -0 "$BUILD_PID" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] native artifacts complete; killing hung teardown (PID $BUILD_PID)"
        pkill -9 -P "$BUILD_PID" 2>/dev/null
        kill -9 "$BUILD_PID" 2>/dev/null
        pkill -9 -f "run_pipeline_patched.py.*$SCENE" 2>/dev/null
      fi
      break
    fi
  fi
  sleep 15
done
wait "$BUILD_PID" 2>/dev/null
sleep 3

# --- Phase 2: package from the completed out_*/ native dir (no model reload) ---
OUT=$(ls -d "$NATIVE_DIR"/out_* 2>/dev/null | head -1)
if [ -z "$OUT" ] || [ ! -f "$OUT/dsg.json" ]; then
  echo "[$(date +%H:%M:%S)] [DAAAM_FAIL] scene=$SCENE: no native out_/dsg.json produced"; exit 3
fi
rm -rf "$PKG" 2>/dev/null
$DAAAM_ENV/bin/python scripts/methods/daaam/build_memory_smoke.py \
  --scene-id "$SCENE" --dataset-tag scannet \
  --native-output-dir "$OUT" --skip-daaam-run --skip-track2-index \
  --daaam-python $DAAAM_ENV/bin/python --run-id "$RUN_ID" --package-root memories \
  >> "$LOG" 2>&1
RC=$?
OBJ=$($DAAAM_ENV/bin/python -c "import json;d=json.load(open('$PKG/build_log.json'));print(d.get('object_count'),d.get('background_object_count'))" 2>/dev/null)
echo "[$(date +%H:%M:%S)] [DAAAM_DONE] scene=$SCENE rc=$RC elapsed=$(( $(date +%s) - START ))s objects=($OBJ)"
