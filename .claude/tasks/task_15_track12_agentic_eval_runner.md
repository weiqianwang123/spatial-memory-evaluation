# Task 15: Track 1/2 Agentic Eval Runner

## Goal

Integrate Track 1/2 agentic full-access evaluation using the sandbox packager,
Claude Bedrock command path, structured prompts, output parsing, and reports.

## Scope

Implement runner/prompt/output handling for Track 1 and Track 2. Do not change
fixed API scoring or memory package schema.

## Context Files

- `.codex/agentic_eval_plan.md`
- Existing Track 1/2 evaluators
- Agentic sandbox packager from Task 14
- `.claude/settings.local.json`

Known local Claude command shape:

```bash
CODE_USE_BEDROCK=1 CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-west-2 claude -p "$(cat {prompt_path})" --permission-mode bypassPermissions --output-format text --max-budget-usd 5 > {output_path}
```

## Implementation Rules

- Agent prompt must state that the agent may design temporary interfaces,
  parsers, or query scripts inside the sandbox.
- Agent gets memory package, adapter code, shared module code, and method root
  source code.
- GT answers, raw frames, and external paths remain unavailable
  unless an explicit ablation enables them.
- Agent output parser should tolerate text wrapping but must extract a valid
  JSON payload.
- Invalid JSON becomes `status: error`, not silent success.
- Record backend/model/cost/latency metadata when available.
- Every run writes summary, details, and markdown report.

## Deliverables

- Track 1 agentic full-access mode.
- Track 2 agentic full-access mode.
- Prompt templates and JSON output schema.
- Claude Bedrock command docs/example.
- Smoke command using an existing package.

## Acceptance Checks

- Agentic mode writes summary/details/report.
- Sandbox does not contain GT answers.
- Output validation distinguishes memory missing, schema unclear, artifact not
  found, and reasoning/query failure when possible.
- Fixed API behavior remains unchanged.

## PR Title

`feat: add track12 agentic eval runner`
