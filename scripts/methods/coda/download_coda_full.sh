#!/usr/bin/env bash
# Download the FULL CODa sequences needed for Track 4 (OC-NaVQA), so that every
# method (ReMEmbR text-memory, DAAAM with LiDAR/depth, future methods) can eval on
# the same data. 7 sequences: 0,3,4,6,16,21,22 (~546 GB total) -> NAS.
#
# Each per-sequence zip is downloaded with curl (resumable via -C -), then
# extracted in place, then the zip is deleted to reclaim space. Idempotent: a
# sequence whose extracted poses/timestamps already exist is skipped.
#
# Usage: download_coda_full.sh [out_dir] [seq ...]
set -uo pipefail
OUT="${1:-/data/mondo-training-dataset/semantic_mapping/coda/raw}"
shift || true
SEQS=("$@"); [ ${#SEQS[@]} -eq 0 ] && SEQS=(0 3 4 6 16 21 22)
BASE="https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/sequences"
mkdir -p "$OUT"
LOG="$OUT/download.log"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
log "CODa full download START -> $OUT | seqs: ${SEQS[*]}"

for s in "${SEQS[@]}"; do
  # skip if already extracted (timestamps file is a cheap marker that lives in every seq)
  if ls "$OUT/$s/timestamps/$s.txt" >/dev/null 2>&1 || ls "$OUT/$s"*/timestamps/"$s.txt" >/dev/null 2>&1; then
    log "[seq $s] already extracted — skip"; continue
  fi
  zip="$OUT/$s.zip"
  url="$BASE/$s.zip"
  sz=$(curl -sS -m 60 -I -L "$url" 2>/dev/null | grep -i content-length | tail -1 | tr -dc '0-9')
  log "[seq $s] downloading ($(python3 -c "print(f'{${sz:-0}/1e9:.1f} GB')" 2>/dev/null)) -> $zip"
  # resumable download
  curl -sS -L -C - -o "$zip" "$url" 2>>"$LOG"
  rc=$?
  if [ "$rc" -ne 0 ]; then log "[seq $s] curl rc=$rc — will retry once"; curl -sS -L -C - -o "$zip" "$url" 2>>"$LOG"; fi
  if [ ! -s "$zip" ]; then log "[seq $s] FAILED (no zip)"; continue; fi
  log "[seq $s] extracting $(du -h "$zip" | cut -f1) ..."
  ( cd "$OUT" && unzip -q -o "$zip" ) 2>>"$LOG"
  if [ $? -eq 0 ]; then
    rm -f "$zip"
    log "[seq $s] DONE extracted, zip removed"
  else
    log "[seq $s] EXTRACT FAILED (zip kept at $zip)"
  fi
done
log "CODa full download ALL DONE"
echo "CODA_DOWNLOAD_DONE $(date +%H:%M:%S)" | tee -a "$LOG"
