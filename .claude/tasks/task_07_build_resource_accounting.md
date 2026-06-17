# Task 07: Build Resource Accounting

## Goal

Add build runtime, time per frame, native memory size, package size, and peak
RAM/VRAM accounting to memory package build outputs.

## Scope

Apply to package metadata helpers and existing method exporters/build smoke
scripts where feasible. Do not change scoring semantics.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/memory_package_spec.md`
- Existing method scripts under `scripts/methods/`
- `spatial_memory_evaluation/memory_package_validator.py`
- Track 1 report/evaluator code

## Implementation Rules

- `native_memory_size_bytes` is the primary memory-size metric.
- `package_size_bytes` is reported separately.
- `memory_artifact_size_bytes` is package `memory/` size.
- `time_per_frame_seconds = build_runtime_seconds / frame_count` when
  `frame_count > 0`.
- Peak RAM/VRAM should be measured only when reliable; otherwise write null and
  a reason.
- Do not include GT/query files in size accounting.

## Deliverables

- Shared accounting helper if repeated across methods.
- Build log/manifest fields for runtime, frame count, sizes, peak RAM/VRAM.
- Validator checks for field presence and null peak metric reasons.
- At least one smoke command/output showing new fields.

## Acceptance Checks

- Existing packages still validate or fail with actionable missing-field
  messages.
- Track 1 can read memory size and build runtime fields.
- Native artifact size and package wrapper size are not confused.

## PR Title

`feat: add memory build resource accounting`
