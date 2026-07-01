#!/usr/bin/env bash
# Download CODa sequences for Track 4, LOCAL-FIRST then move to NAS.
#
# Why: the TACC server is ~6 MB/s and the NAS writes at ~5 MB/s; piping curl
# straight to NAS stalls. So per sequence we:
#   1. curl the full zip to FAST local SSD (/home, ~1 GB/s write, resumable -C -)
#   2. unzip ONLY the modalities we need (cam0 + cam1 RGB, poses, calib,
#      timestamps, metadata, 3d_bbox) — skip LiDAR (3d_comp ~12GB) + cam3
#   3. delete the zip
#   4. rsync the slim per-seq dir to the NAS
#   5. delete the local slim dir
# Net: download is bounded only by TACC bandwidth; NAS sees a single sequential
# rsync (no random small writes); local peak ≈ one zip (≤119 GB) + extract.
#
# Resumable: a sequence already present on NAS (poses marker) is skipped; a
# partially-downloaded zip resumes via curl -C -.
#
# Usage: download_coda_local_then_nas.sh [seq ...]   (default: 0 3 4 6 16 21 22)
set -uo pipefail
LOCAL=/home/robin_wang/coda_local
NAS=/data/mondo-training-dataset/semantic_mapping/coda/seqs
BASE="https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/sequences"
SEQS=("$@"); [ ${#SEQS[@]} -eq 0 ] && SEQS=(0 3 4 6 16 21 22)
mkdir -p "$LOCAL" "$NAS"
LOG=/home/robin_wang/spatial-memory-evaluation/_run_logs/coda_local_then_nas.log
# keep only these top-level zip dirs (everything else is skipped at extract time)
KEEP=("2d_rect/cam0" "2d_rect/cam1" "poses" "timestamps" "calibrations" "metadata" "3d_bbox")

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== CODa local-then-NAS START | seqs ${SEQS[*]} | local=$LOCAL nas=$NAS ==="

for s in "${SEQS[@]}"; do
  if [ -s "$NAS/$s/poses/dense_global/$s.txt" ]; then log "[seq $s] on NAS already — skip"; continue; fi
  zip="$LOCAL/$s.zip"; sdir="$LOCAL/$s"
  url="$BASE/$s.zip"
  sz=$(curl -sS -m 60 -IL "$url" 2>/dev/null | grep -i content-length | tail -1 | tr -dc '0-9')
  log "[seq $s] downloading full zip ($(python3 -c "print(f'{${sz:-0}/1e9:.1f} GB')")) to LOCAL ..."
  # resumable single-connection download to fast local disk
  for try in 1 2 3 4 5; do
    curl -sS -L -C - --connect-timeout 60 -o "$zip" "$url" 2>>"$LOG" && break
    log "[seq $s] curl attempt $try interrupted (rc=$?), resuming in 15s ..."; sleep 15
  done
  if [ ! -s "$zip" ]; then log "[seq $s] DOWNLOAD FAILED — skip"; continue; fi
  log "[seq $s] downloaded $(du -h "$zip" | cut -f1); extracting needed modalities ..."
  rm -rf "$sdir"; mkdir -p "$sdir"
  # build unzip include patterns
  inc=()
  for k in "${KEEP[@]}"; do inc+=("$k/*"); done
  ( cd "$sdir" && unzip -q -o "$zip" "${inc[@]}" ) 2>>"$LOG"
  rc=$?
  rm -f "$zip"
  if [ "$rc" -ne 0 ] && [ ! -d "$sdir/poses" ]; then log "[seq $s] EXTRACT FAILED"; continue; fi
  log "[seq $s] extracted slim dir $(du -sh "$sdir" | cut -f1); rsync -> NAS ..."
  rsync -a --remove-source-files "$sdir/" "$NAS/$s/" 2>>"$LOG"
  rm -rf "$sdir"
  log "[seq $s] DONE on NAS: $(du -sh "$NAS/$s" 2>/dev/null | cut -f1) | cam0 $(ls "$NAS/$s/2d_rect/cam0/$s" 2>/dev/null|wc -l) cam1 $(ls "$NAS/$s/2d_rect/cam1/$s" 2>/dev/null|wc -l)"
done
log "=== CODA_LOCAL_THEN_NAS ALL DONE ==="
echo "CODA_DOWNLOAD_DONE $(date '+%H:%M:%S')" | tee -a "$LOG"
