# Task 08: Build Resource Accounting

## Goal

Add build runtime, time per frame, native memory size, package size, and peak RAM/VRAM accounting to memory build outputs.

## Scope

Apply to existing method exporters/build smoke scripts and package metadata. Do not change scoring semantics.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/memory_package_spec.md`
- Existing method scripts under `scripts/methods/`
- `spatial_memory_evaluation/memory_package_validator.py`

## Implementation Rules

- `native_memory_size_bytes` is the primary memory-size metric.
- `package_size_bytes` is reported separately.
- `memory_artifact_size_bytes` is the size of package `memory/`.
- `time_per_frame_seconds = build_runtime_seconds / frame_count` when frame_count > 0.
- Peak RAM/VRAM should be measured when reliable; otherwise write null and a reason.
- Do not include GT/query files in size accounting.

## Deliverables

- Build log fields for runtime, frame count, time per frame, sizes, peak RAM/VRAM.
- Helper utilities if repeated across methods.
- Validator updates that check field presence and allow null peak metrics with reasons.
- At least one smoke output showing the new fields.

## Acceptance Checks

- Existing packages still validate or fail with actionable missing-field messages.
- Track 1 reports can read memory size and build runtime.
- The native artifact size and package wrapper size are not confused.

## PR Title

`feat: add memory build resource accounting`
