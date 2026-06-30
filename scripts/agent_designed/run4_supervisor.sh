#!/usr/bin/env bash
# run4 supervisor: keep a designer agent working on the run4 sandbox for a HARD
# 10-hour wall, RELAUNCHING it every time it exits early (finished / max-turns /
# crash) until the wall is hit. Each relaunch is a fresh `claude -p` process, but
# the SANDBOX PERSISTS (git history + DESIGN_NOTES.md + history.jsonl), so a new
# agent reads its own prior progress and continues the keep/revert loop — it does
# NOT start the design from scratch.
#
# Usage: run4_supervisor.sh <sandbox_dir> [wall_seconds]
set -uo pipefail

SANDBOX="${1:?usage: run4_supervisor.sh <sandbox_dir> [wall_seconds]}"
WALL="${2:-36000}"            # 10h hard wall
PROMPT_FILE="$SANDBOX/TASK_PROMPT.txt"
LOG_DIR="$SANDBOX/_run_logs"
SUP_LOG="$LOG_DIR/run4_supervisor.log"
MODEL="us.anthropic.claude-opus-4-8[1m]"
mkdir -p "$LOG_DIR"

# Absolute deadline. We pass start/now via env (Date.now() etc. are fine in bash).
START=$(date +%s)
DEADLINE=$(( START + WALL ))

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUP_LOG"; }

log "run4 supervisor START — sandbox=$SANDBOX wall=${WALL}s deadline=$(date -d @"$DEADLINE" +%H:%M:%S)"

attempt=0
while :; do
  NOW=$(date +%s)
  REMAIN=$(( DEADLINE - NOW ))
  if [ "$REMAIN" -le 120 ]; then
    log "WALL REACHED (remaining ${REMAIN}s ≤ 120s) — stopping. total attempts=$attempt"
    break
  fi
  attempt=$(( attempt + 1 ))
  ALOG="$LOG_DIR/run4_attempt_${attempt}.log"
  log "ATTEMPT $attempt — launching designer with timeout ${REMAIN}s (remaining budget), log -> $(basename "$ALOG")"

  # Launch the designer. timeout = remaining budget, so a single long-lived agent
  # that runs to the wall is killed exactly at the deadline; an agent that exits
  # early frees the loop to relaunch.
  (
    cd "$SANDBOX" || exit 97
    CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-west-2 \
    timeout "${REMAIN}s" claude -p "$(cat "$PROMPT_FILE")" \
      --model "$MODEL" \
      --permission-mode bypassPermissions \
      --add-dir . \
      --max-turns 2000 \
      --output-format text
    echo "ATTEMPT_EXIT rc=$?"
  ) > "$ALOG" 2>&1
  RC=$(grep -oE "ATTEMPT_EXIT rc=[0-9]+" "$ALOG" | tail -1 | grep -oE "[0-9]+$")
  log "ATTEMPT $attempt EXITED rc=${RC:-?} after $(( $(date +%s) - NOW ))s"

  # tiny backoff so a fast-crash loop doesn't spin
  sleep 10
done

log "run4 supervisor DONE — $(date +%H:%M:%S). Final sandbox state left intact for held-out eval."
echo "RUN4_SUPERVISOR_DONE attempts=$attempt at $(date +%H:%M:%S)" | tee -a "$SUP_LOG"
