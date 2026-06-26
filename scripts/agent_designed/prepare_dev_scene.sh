#!/usr/bin/env bash
# Prepare ONE dev scene for the agent-designed auto-research framework, end to end:
#   1. download the 4 ScanNet GT files from kaldir (if missing)
#   2. extract the .sens -> shared RGB-D layout (stride 5)
#   3. build the Track 1 (object-location) + Track 2 (referring) benchmarks
#      (Track 3 OpenEQA is built once for all scannet scenes; see note below)
#
# These benchmarks are the metric-faithful GT the designer's self-tests are scored
# against (the seed for loop_fixed_tests; the reference tooling for auto_research).
# DEV scenes must be OUTSIDE the held-out 10 (enforced by splits.py at run time).
#
# Usage: prepare_dev_scene.sh <scene_id> [sens_stride]
set -uo pipefail
cd /home/robin_wang/spatial-memory-evaluation
HERE="$(dirname "$0")"
PY=/home/robin_wang/miniforge3/envs/spatial-rag/bin/python
SCANS_ROOT="${SCANNET_SCANS_ROOT:-/data/mondo-training-dataset/semantic_mapping/scannet/scans}"

SCENE="${1:?usage: prepare_dev_scene.sh <scene_id> [sens_stride]}"
STRIDE="${2:-5}"

echo "==[1/3] GT files for $SCENE =="
bash "$HERE/download_scannet_gt.sh" "$SCENE"

echo "==[2/3] RGB-D layout for $SCENE (stride $STRIDE) =="
bash scripts/methods/prepare_scannet_layout.sh "$SCENE" "$STRIDE"

echo "==[3/3] Track 1 + Track 2 benchmarks for $SCENE =="
$PY scripts/build_track1_data.py \
  --scene-id "$SCENE" --dataset scannet \
  --scannet-scans-root "$SCANS_ROOT" \
  --output-dir "benchmarks/track1/scannet/$SCENE"

$PY scripts/build_track2_data.py \
  --scene-id "$SCENE" \
  --scannet-scans-root "$SCANS_ROOT" \
  --output-dir "benchmarks/track2/scanents3d/$SCENE"

echo "==[3b] Track 3 (OpenEQA) per-scene dir for $SCENE =="
# build_track3_data writes a flat scannet/ dir for ALL scenes; build it once if
# missing, then split this scene into benchmarks/track3/openeqa/<scene>/.
if [ ! -f "benchmarks/track3/openeqa/scannet/questions.jsonl" ]; then
  $PY scripts/build_track3_data.py --dataset scannet >/dev/null
fi
$PY scripts/agent_designed/split_track3_by_scene.py "$SCENE"

echo "[$SCENE] dev-scene prep done."
