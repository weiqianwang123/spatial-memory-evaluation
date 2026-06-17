# Task 02: Baseline Audit - HOV-SG

## Goal

Audit `/home/robin_wang/HOV-SG` and update the registry with evidence-backed
Track 1/2 fixed API eligibility for HOV-SG.

## Scope

This is an audit-only PR. Focus on HOV-SG native graph construction, native
artifacts, query/read tools, detector/SAM/CLIP stack, and whether a formal
closed-vocabulary eval variant is feasible.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- `.codex/modules.md`
- `/home/robin_wang/HOV-SG`

## Implementation Rules

- Use root repo evidence only and cite `relative/path:line`.
- Track 1 support requires an object/graph memory with object labels and 3D
  positions or a thin export from native graph artifacts.
- Track 2 support requires a native query/read path that can answer object
  location queries without benchmark-specific logic.
- Formal Track 1/2 must be closed-vocabulary; unrestricted open-vocabulary
  behavior is `ov_ablation`.
- Do not edit HOV-SG itself.

## Deliverables

- Update `.codex/baseline_registry.md` HOV-SG row/section.
- Record evidence for native build command/config, artifact paths/formats,
  graph/object schema, query/read capability, and modules/checkpoints.
- Record whether HOV-SG Track 2 fixed API is supported, candidate, or invalid.

## Acceptance Checks

- HOV-SG has a documented CV formal route or a clear blocker.
- Any Track 2 decision has root repo evidence or an explicit missing reason.
- OV behavior is not mixed into formal Track 1/2 support.

## PR Title

`docs: audit hovsg baseline capabilities`
