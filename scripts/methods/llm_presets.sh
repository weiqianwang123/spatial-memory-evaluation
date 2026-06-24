#!/usr/bin/env bash
# Shared LLM command presets for tool_llm / judge evaluation.
#
# The eval scripts take an --llm-command (and --judge-command) template with
# {prompt_path} / {output_path} placeholders. This file exposes ready-made
# templates per Claude model tier on this machine's Bedrock CLI, so runs can
# trade cost/speed against quality without retyping the long model ids.
#
# Usage:
#   source scripts/methods/llm_presets.sh
#   llm_cmd haiku      # cheapest/fastest   -> echoes the --llm-command template
#   llm_cmd sonnet     # mid
#   llm_cmd opus       # most capable (CLI default; what the first runs used)
#   judge_cmd sonnet   # judge template (no bypass redirect quirks)
#   model_id haiku     # just the bedrock model id
#
# Example:
#   python scripts/evaluate_track1.py "$PKG" --scene-id 036bce3393 --mode tool_llm \
#     --llm-command "$(llm_cmd haiku)" --output "$OUT"
#
# Model ids verified working on this Bedrock account (CLAUDE_CODE_USE_BEDROCK=1):
#   haiku  -> us.anthropic.claude-haiku-4-5-20251001-v1:0   (cheapest)
#   sonnet -> us.anthropic.claude-sonnet-4-6                (mid)
#   opus   -> us.anthropic.claude-opus-4-8[1m]              (default, most capable)

model_id() {
  case "$1" in
    haiku)  echo "us.anthropic.claude-haiku-4-5-20251001-v1:0" ;;
    sonnet) echo "us.anthropic.claude-sonnet-4-6" ;;
    opus)   echo "us.anthropic.claude-opus-4-8[1m]" ;;
    "")     echo "us.anthropic.claude-opus-4-8[1m]" ;;   # default
    *)      echo "$1" ;;                                  # pass through a raw id
  esac
}

# Answering/agent command: reads the prompt file, writes raw text to output file.
llm_cmd() {
  local mid; mid="$(model_id "${1:-opus}")"
  printf 'claude -p "$(cat {prompt_path})" --model %s --output-format text --permission-mode bypassPermissions > {output_path}' "$mid"
}

# Judge command: same model selection, no output redirect (evaluator captures stdout).
judge_cmd() {
  local mid; mid="$(model_id "${1:-opus}")"
  printf 'claude -p "$(cat {prompt_path})" --model %s --output-format text' "$mid"
}
