#!/usr/bin/env bash
# Download the 4 per-scene ScanNet GT annotation files needed for Track 1/2 GT
# bbox derivation (track2/scannet_bbox.py) into the local scans root.
#
# Files (per scene): <scene>_vh_clean_2.ply, <scene>.aggregation.json,
#                    <scene>_vh_clean_2.0.010000.segs.json, <scene>.txt
# Source: https://kaldir.vc.in.tum.de/scannet/v2/scans/<scene>/...  (un-gated;
# a direct GET returns the file). This is the same acquisition used for the 10
# held-out scenes (see .codex/eval_set_inventory.md).
#
# Usage: download_scannet_gt.sh <scene_id> [<scene_id> ...]
set -uo pipefail

SCANS_ROOT="${SCANNET_SCANS_ROOT:-/data/mondo-training-dataset/semantic_mapping/scannet/scans}"
BASE="https://kaldir.vc.in.tum.de/scannet/v2/scans"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <scene_id> [<scene_id> ...]" >&2
  exit 2
fi

fetch_one() {
  local scene="$1" name="$2" dest="$3"
  if [ -s "$dest" ]; then
    echo "  [skip] $name (exists, $(du -h "$dest" | cut -f1))"
    return 0
  fi
  local url="$BASE/$scene/$name"
  local code
  code=$(curl -sS -m 600 -L -o "$dest.partial" -w "%{http_code}" "$url" 2>/dev/null)
  if [ "$code" = "200" ] && [ -s "$dest.partial" ]; then
    mv "$dest.partial" "$dest"
    echo "  [ok]   $name ($(du -h "$dest" | cut -f1))"
    return 0
  fi
  rm -f "$dest.partial"
  echo "  [FAIL] $name (HTTP $code) <- $url" >&2
  return 1
}

rc=0
for scene in "$@"; do
  dir="$SCANS_ROOT/$scene"
  mkdir -p "$dir"
  echo "[$scene] -> $dir"
  fetch_one "$scene" "${scene}.txt"                              "$dir/${scene}.txt" || rc=1
  fetch_one "$scene" "${scene}.aggregation.json"                 "$dir/${scene}.aggregation.json" || rc=1
  fetch_one "$scene" "${scene}_vh_clean_2.0.010000.segs.json"    "$dir/${scene}_vh_clean_2.0.010000.segs.json" || rc=1
  fetch_one "$scene" "${scene}_vh_clean_2.ply"                   "$dir/${scene}_vh_clean_2.ply" || rc=1
done

if [ "$rc" -ne 0 ]; then echo "[download_scannet_gt] some files failed" >&2; fi
exit "$rc"
