#!/usr/bin/env bash
# Keep the CODa local-then-NAS download alive until all 7 seqs land on NAS.
# Relaunches the downloader if it dies; the downloader itself skips done seqs
# and resumes partial zips (curl -C -).
set -uo pipefail
SCRIPT=/home/robin_wang/spatial-memory-evaluation/scripts/methods/coda/download_coda_local_then_nas.sh
LOG=/home/robin_wang/spatial-memory-evaluation/_run_logs/coda_dl_supervisor.log
NAS=/data/mondo-training-dataset/semantic_mapping/coda/seqs
for i in $(seq 1 200); do
  done=0
  for s in 0 3 4 6 16 21 22; do [ -s "$NAS/$s/poses/dense_global/$s.txt" ] && done=$((done+1)); done
  echo "[$(date '+%m-%d %H:%M:%S')] supervisor check: $done/7 seqs on NAS" >> "$LOG"
  [ "$done" -ge 7 ] && { echo "ALL_7_DONE" >> "$LOG"; break; }
  if ! ps -eo args 2>/dev/null | grep -q '[d]ownload_coda_local_then_nas'; then
    echo "[$(date '+%m-%d %H:%M:%S')] downloader not running — relaunching" >> "$LOG"
    nohup bash "$SCRIPT" >> /home/robin_wang/spatial-memory-evaluation/_run_logs/coda_dl_nohup.log 2>&1 &
  fi
  sleep 300
done
