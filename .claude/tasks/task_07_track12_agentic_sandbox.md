# Task 07: Track 1/2 Agentic Sandbox

## Goal

Generalize the Track 1/2 agentic runner so every memory package can be evaluated by a sandboxed Claude agent with memory package access plus source-code access.

## Scope

Implement runner/prompt/output handling only. Do not change fixed API scoring or memory package schema unless required for agentic metadata.

## Context Files

- `.codex/agentic_eval_plan.md`
- Existing Track 1/2 evaluators
- Existing agentic code paths, if any
- `.claude/settings.local.json`

Known local Claude smoke command:

```bash
CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-west-2 claude -p "$(cat {prompt_path})" --permission-mode bypassPermissions --output-format text --max-budget-usd 5 > {output_path}
```

## Implementation Rules

- Copy memory package into sandbox; do not mutate the original package.
- Copy evaluation adapter code from `scripts/methods/<method>/`.
- Copy shared module code from `spatial_memory_evaluation/shared_modules/` and `scripts/methods/shared_modules.py`.
- Copy the original method root repo source code from `manifest.method.repo_path`.
- Source-code copies may exclude `.git`, generated data, checkpoints, caches, memories, and result artifacts, but must keep source, configs, scripts, schema, and README/docs.
- GT answers must not enter sandbox.
- Default agentic mode is package + source-code full access.
- raw frames and unrestricted OV must be explicit ablations.
- Agent output parser should tolerate Claude text wrapping but must extract a valid JSON payload.
- Record latency/cost/model/backend metadata when available.
- Prompt must clearly tell the agent it may design temporary interfaces, parsers, or query scripts inside the sandbox to interact with memory artifacts.

## Deliverables

- Track 1 agentic full-access runner or mode.
- Track 2 agentic full-access runner or mode.
- Prompt templates and JSON output schema.
- Claude Bedrock command example in docs.
- Smoke command using an existing package.

## Acceptance Checks

- Agentic mode writes summary/details/report.
- Invalid JSON from agent becomes `status: error`, not a silent success.
- Sandbox package copy contains no GT answers.
- Sandbox source context includes the method adapter code, shared_modules code, and original method root source code.
- Fixed API behavior remains unchanged.

## PR Title

`feat: add track12 agentic sandbox evaluation`
