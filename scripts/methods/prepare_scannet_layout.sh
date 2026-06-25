#!/usr/bin/env bash
# Prepare a single ScanNet scene into the shared RGB-D layout that ALL methods
# consume: DAAAM (rgb/depth/pose/intrinsic), ClawS (rgb/), ReMEmbR + caption
# control (color/+pose/), multiframe VLM control (frames-dir+pose-dir).
#
# One extraction per scene at the chosen .sens stride; export_scannet_layout then
# emits the DAAAM layout (frame-stride 1 = keep all extracted). A color->rgb
# symlink lets the ReMEmbR/caption builders read the same images.
#
# Usage: prepare_scannet_layout.sh <scene_id> [sens_stride]
set -euo pipefail
cd /home/robin_wang/spatial-memory-evaluation

SCENE="$1"
STRIDE="${2:-5}"
PY=/home/robin_wang/miniforge3/envs/spatial-rag/bin/python
SCANS_ROOT=/data/mondo-training-dataset/semantic_mapping/scannet/scans
SENS="$SCANS_ROOT/$SCENE/$SCENE.sens"

FRAMES_DIR="data/scannet_layouts/$SCENE/frames"
LAYOUT_DIR="data/scannet_layouts/$SCENE/layout"

echo "[$(date +%H:%M:%S)] $SCENE: extracting .sens at stride $STRIDE -> $FRAMES_DIR"
$PY scripts/methods/daaam/extract_sens_frames.py \
  --sens "$SENS" \
  --output-dir "$FRAMES_DIR" \
  --frame-skip "$STRIDE" \
  --max-frames 0

echo "[$(date +%H:%M:%S)] $SCENE: exporting DAAAM layout -> $LAYOUT_DIR"
$PY scripts/methods/daaam/export_scannet_layout.py \
  --frames-dir "$FRAMES_DIR" \
  --output-dir "$LAYOUT_DIR" \
  --scene-id "$SCENE" \
  --frame-stride 1 \
  --max-frames 0

# color/ alias for ReMEmbR/caption builders (they read color/*.jpg + pose/*.txt).
if [ ! -e "$LAYOUT_DIR/color" ]; then
  ln -s rgb "$LAYOUT_DIR/color"
fi

N=$(ls "$LAYOUT_DIR/rgb" | wc -l)
echo "[$(date +%H:%M:%S)] $SCENE: layout ready, $N frames"
