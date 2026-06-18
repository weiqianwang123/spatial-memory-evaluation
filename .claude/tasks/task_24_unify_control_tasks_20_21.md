# Task 24: Unify Multi-Frame VLM And Caption Control Branches

## Goal

Create a clean combined control branch that contains the useful work from Task
20 and Task 21 together, with conflicts resolved and consistent control semantics.

## Current Problem

Task 20 and Task 21 both completed, but they touch shared docs:

- `.codex/baseline_registry.md`
- `README.md`

Task 20 also adds shared evaluator semantics for no-explicit-memory controls.
Task 21 adds the caption-control package builder and fixture. These should be
reviewed together so `raw_frame_control` and `caption_control` behave
consistently.

Existing branches:

- Task 20: `claude/task-20-multiframe-vlm-control` at `412da6d`
- Task 21: `claude/task-21-llm-captions-control` at `b3bb60d`

## Scope

Work in a new combined worktree/branch created from Task 20 or main plus both
task changes. Keep the final branch focused on control semantics and examples.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/memory_package_spec.md`
- `.codex/baseline_registry.md`
- `.codex/modules.md`
- `README.md`
- `spatial_memory_evaluation/common/package_io.py`
- `spatial_memory_evaluation/common/reporting.py`
- `spatial_memory_evaluation/track1/evaluator.py`
- `spatial_memory_evaluation/track2/evaluator.py`
- `scripts/methods/remembr/build_caption_control_package.py`
- `examples/multiframe_vlm_control/`
- `examples/caption_control_package/`

## Implementation Rules

- Do not modify external method repos.
- Preserve Task 20's evaluator behavior:
  - no-explicit-memory controls get `reason_code:
    control_no_explicit_memory`;
  - summaries/reports expose `control`, `explicit_memory`, and
    `method_family`;
  - normal object-memory packages still use `unsupported_fixed_api`.
- Preserve Task 21's caption-control builder and fixture, but align it with Task
  20 semantics. If the validator/evaluator now emits `control_no_explicit_memory`
  for `caption_control`, update docs/tests that still expect
  `unsupported_fixed_api`.
- Keep raw-frame and caption controls out of the Track 1/2 object-memory fixed
  API table.
- Do not track generated `memories/`, `results/`, or `data/` outputs.
- Resolve `.codex/baseline_registry.md` and `README.md` conflicts into a single
  coherent story:
  - Multi-frame VLM = raw-frame no-explicit-memory control.
  - LLM-with-captions = caption-control / no object-memory fixed API.
  - Both are agentic/control ablations, not object-memory baselines.

## Deliverables

- A clean combined branch with both Task 20 and Task 21 useful changes.
- Consistent docs for `raw_frame_control` and `caption_control`.
- Fixtures/examples validate.
- Track 1/2 fixed API on each fixture returns readable invalid/control reports.

## Acceptance Checks

- `git status --short` is clean after commit.
- `git diff --name-only main...HEAD` contains no paths under `memories/`,
  `results/`, or `data/`.
- Validator passes for both example control packages.
- Track 1/2 evaluator reports for both examples use consistent control semantics.
- No external method repo is modified.

## PR Title

`feat: unify track12 no-memory control semantics`
