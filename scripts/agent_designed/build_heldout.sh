#!/usr/bin/env bash
# Build the FROZEN agent-designed memory (run2 best, commit 1d977bf) on all 10
# held-out ScanNet scenes, into the main-eval package location so
# eval_all_scannet.sh can score it via the fair per-query tool_llm path.
#
# This is the "freeze then score once on held-out" step — the design is NOT
# modified; we just run its build_memory.py on the 10 benchmark scenes.
set -uo pipefail
cd /home/robin_wang/spatial-memory-evaluation
PY=/home/robin_wang/miniforge3/envs/spatial-rag/bin/python
SB=/home/robin_wang/agent_designed_sandbox_loop_fixed_tests_run2
LAYOUTS=/home/robin_wang/spatial-memory-evaluation/data/scannet_layouts
HELD="scene0015_00 scene0050_00 scene0077_00 scene0084_00 scene0131_00 scene0193_00 scene0207_00 scene0222_00 scene0256_00 scene0314_00"

echo "[heldout-build] frozen design at $(cd $SB && git rev-parse --short HEAD)"
for s in $HELD; do
  layout="$LAYOUTS/$s/layout"
  out="memories/agent_designed/scannet/$s/agent-designed-track-$s"
  if [ -f "$out/manifest.json" ]; then echo "[skip] $s (package exists)"; continue; fi
  if [ ! -d "$layout/rgb" ]; then echo "[MISSING-LAYOUT] $s"; continue; fi
  mkdir -p "$out"
  echo "[$(date +%H:%M:%S)] build $s ($(ls $layout/rgb|wc -l) frames) -> $out"
  # Run the frozen design's builder from the sandbox (its sm_core/pkg_tools are
  # importable there); write the package into the main-eval location.
  ( cd "$SB" && PYTHONPATH="$SB:/home/robin_wang/spatial-memory-evaluation" \
      "$PY" starter/build_memory.py \
        --layout-dir "$layout" --scene-id "$s" \
        --out "/home/robin_wang/spatial-memory-evaluation/$out" ) \
    > "memories/agent_designed/scannet/$s.build.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ] && [ -f "$out/manifest.json" ]; then
    echo "[ok] $s built + $($PY -m spatial_memory_evaluation.memory_package_validator "$out" 2>&1 | grep -o '"valid": [a-z]*' | head -1)"
  else
    echo "[FAIL] $s rc=$rc (see $s.build.log)"; tail -3 "memories/agent_designed/scannet/$s.build.log"
  fi
done
echo "[heldout-build DONE] $(find memories/agent_designed/scannet -name manifest.json 2>/dev/null|wc -l)/10 packages"
