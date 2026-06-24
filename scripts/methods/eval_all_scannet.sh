#!/usr/bin/env bash
# Evaluate all built methods on the 10 shared ScanNet scenes across Track 1/2/3.
#
#   tool_llm  agent  = haiku  (fastest)        -> $(llm_cmd haiku)
#   tool_llm  judge  = sonnet (medium, T3 QA)  -> $(judge_cmd sonnet)
#   fixed_api: ClawS + DAAAM on Track 1 (instant, native deterministic query)
#
# One (method, scene, track, mode) per call; results under
# results/<method>/track<N>-<mode>/scannet-<scene>/eval_*.{json,md}.
#
# Args: $1 = track (1|2|3|all), $2 = mode (tool_llm|fixed_api|both), $3 = methods csv (optional)
set -uo pipefail
cd /home/robin_wang/spatial-memory-evaluation
source scripts/methods/llm_presets.sh
PY=/home/robin_wang/miniforge3/envs/spatial-rag/bin/python

TRACK="${1:-all}"
MODE="${2:-tool_llm}"
METHODS_CSV="${3:-daaam,claws,remembr,remembr_captions,multiframe_vlm}"
SCENES="scene0015_00 scene0050_00 scene0077_00 scene0084_00 scene0131_00 scene0193_00 scene0207_00 scene0222_00 scene0256_00 scene0314_00"
IFS=',' read -ra METHODS <<< "$METHODS_CSV"

AGENT_CMD="$(llm_cmd haiku)"
JUDGE_CMD="$(judge_cmd sonnet)"

# method -> packaged run-id dir name + memory dataset subdir
pkg_dir() {
  local m="$1" s="$2"
  case "$m" in
    daaam)            echo "memories/daaam/scannet/$s/daaam-track-$s" ;;
    claws)            echo "memories/claws/scannet/$s/claws-track-$s" ;;
    remembr)          echo "memories/remembr/scannet/$s/remembr-track-$s" ;;
    remembr_captions) echo "memories/remembr_captions/scannet/$s/captions-track-$s" ;;
    multiframe_vlm)   echo "memories/multiframe_vlm/scannet/$s/multiframe-vlm-track-$s" ;;
  esac
}

run_one() {
  local track="$1" mode="$2" m="$3" s="$4"
  local pkg; pkg="$(pkg_dir "$m" "$s")"
  if [ ! -f "$pkg/manifest.json" ]; then echo "[SKIP] $m/$s: no package"; return; fi
  local out="results/$m/track${track}-${mode}/scannet-$s/eval_summary.json"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ]; then echo "[DONE-cached] T$track $mode $m $s"; return; fi
  local START; START=$(date +%s)
  case "$track" in
    1)
      if [ "$mode" = "tool_llm" ]; then
        $PY scripts/evaluate_track1.py "$pkg" --dataset scannet --scene-id "$s" \
          --mode tool_llm --llm-command "$AGENT_CMD" --output "$out" > "${out%.json}.log" 2>&1
      else
        $PY scripts/evaluate_track1.py "$pkg" --dataset scannet --scene-id "$s" \
          --mode fixed_api --output "$out" > "${out%.json}.log" 2>&1
      fi ;;
    2)
      if [ "$mode" = "tool_llm" ]; then
        $PY scripts/evaluate_track2.py "$pkg" --benchmark-dir "benchmarks/track2/scanents3d/$s" \
          --mode tool_llm --llm-command "$AGENT_CMD" --output "$out" > "${out%.json}.log" 2>&1
      else
        $PY scripts/evaluate_track2.py "$pkg" --benchmark-dir "benchmarks/track2/scanents3d/$s" \
          --mode fixed_api --output "$out" > "${out%.json}.log" 2>&1
      fi ;;
    3)
      $PY scripts/evaluate_track3.py "$pkg" --benchmark-dir "benchmarks/track3/openeqa/$s" \
        --dataset scannet --mode tool_llm --llm-command "$AGENT_CMD" --judge-command "$JUDGE_CMD" \
        --output "$out" > "${out%.json}.log" 2>&1 ;;
  esac
  local RC=$?
  echo "[EVAL_DONE] T$track $mode $m $s rc=$RC elapsed=$(( $(date +%s)-START ))s"
}

TRACKS=(); [ "$TRACK" = "all" ] && TRACKS=(1 2 3) || TRACKS=("$TRACK")
for t in "${TRACKS[@]}"; do
  for m in "${METHODS[@]}"; do
    for s in $SCENES; do
      run_one "$t" "$MODE" "$m" "$s"
    done
  done
done
echo "[ALL_EVAL_DONE track=$TRACK mode=$MODE methods=$METHODS_CSV]"
