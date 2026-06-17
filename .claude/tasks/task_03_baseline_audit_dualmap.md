# Task 03: Baseline Audit - DualMap

## Goal

Audit `/home/robin_wang/DualMap` and update the registry with evidence-backed
Track 1/2 fixed API eligibility for DualMap.

## Scope

This is an audit-only PR. Focus on DualMap native concrete/global maps, object
artifacts, query/read paths, detector/SAM/CLIP stack, and the formal
closed-vocabulary eval variant.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- `.codex/modules.md`
- `/home/robin_wang/DualMap`

## Implementation Rules

- Use root repo evidence only and cite `relative/path:line`.
- Track 1 support can rely on native object map artifacts plus a thin object
  table export.
- Track 2 support is allowed only if DualMap has a native or thin
  non-interactive object/query bridge.
- Formal Track 1/2 must use canonical closed-vocabulary labels. Unrestricted
  OV output remains `ov_ablation`.
- Do not implement exporter changes in this audit task.

## Deliverables

- Update `.codex/baseline_registry.md` DualMap row/section.
- Record evidence for concrete map/global map artifacts, native build path,
  object fields, query/read capability, and perception modules.
- Record clear Track 1/2 fixed API status and CV/OV distinction.

## Acceptance Checks

- DualMap formal CV status is explicit.
- Track 2 is not marked supported unless a native bridge is identified.
- The registry explains concrete map vs global map relevance for evaluation.

## PR Title

`docs: audit dualmap baseline capabilities`
