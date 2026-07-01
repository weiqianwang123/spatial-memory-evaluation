#!/usr/bin/env bash
# Build the FROZEN run3 agent-designed memory (HEAD 079cfa7 — the agent's final
# deliverable: deferred-caption relational object graph, stride2, anchor-T2,
# spatial-T3) on all 10 held-out ScanNet scenes, into the main-eval package
# location so eval_all_scannet.sh scores it via the fair per-query tool_llm path.
#
# run2's packages (already scored; run2 vs run3 vs baselines held-out comparison is
# in .codex/agent_designed_run3_analysis.md) are moved aside to
# memories/agent_designed_run2_heldout/ first, so nothing is lost.
# The design is NOT modified — we only run its build_memory.py on the 10 scenes.
set -uo pipefail
cd /home/robin_wang/spatial-memory-evaluation
PY=/home/robin_wang/miniforge3/envs/spatial-rag/bin/python
SB=/home/robin_wang/agent_designed_sandbox_loop_fixed_tests_run3
LAYOUTS=/home/robin_wang/spatial-memory-evaluation/data/scannet_layouts
DEST=memories/agent_designed/scannet
RUN2_BAK=memories/agent_designed_run2_heldout/scannet
HELD="scene0015_00 scene0050_00 scene0077_00 scene0084_00 scene0131_00 scene0193_00 scene0207_00 scene0222_00 scene0256_00 scene0314_00"

echo "[heldout-build run3] frozen design at $(cd $SB && git rev-parse --short HEAD)"

# Preserve run2's held-out packages once (idempotent).
if [ ! -d "$RUN2_BAK" ] && [ -f "$DEST/scene0015_00/agent-designed-track-scene0015_00/manifest.json" ]; then
  mkdir -p "$RUN2_BAK"
  echo "[preserve] moving run2 held-out packages -> $RUN2_BAK"
  mv "$DEST"/* "$RUN2_BAK"/ 2>/dev/null || true
fi

for s in $HELD; do
  layout="$LAYOUTS/$s/layout"
  out="$DEST/$s/agent-designed-track-$s"
  if [ -f "$out/manifest.json" ]; then echo "[skip] $s (run3 package exists)"; continue; fi
  if [ ! -d "$layout/rgb" ]; then echo "[MISSING-LAYOUT] $s"; continue; fi
  mkdir -p "$out"
  echo "[$(date +%H:%M:%S)] build $s ($(ls $layout/rgb|wc -l) frames) -> $out"
  ( cd "$SB" && PYTHONPATH="$SB:/home/robin_wang/spatial-memory-evaluation" \
      "$PY" starter/build_memory.py \
        --layout-dir "$layout" --scene-id "$s" \
        --out "/home/robin_wang/spatial-memory-evaluation/$out" ) \
    > "$DEST/$s.build.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ] && [ -f "$out/manifest.json" ]; then
    valid=$($PY -m spatial_memory_evaluation.memory_package_validator "$out" 2>&1 | grep -o '"valid": [a-z]*' | head -1)
    echo "[ok] $s built + $valid"
  else
    echo "[FAIL] $s rc=$rc (see $DEST/$s.build.log)"; tail -4 "$DEST/$s.build.log"
  fi
done
echo "[heldout-build run3 DONE] $(find $DEST -name manifest.json 2>/dev/null|wc -l)/10 packages at $(date +%H:%M:%S)"
