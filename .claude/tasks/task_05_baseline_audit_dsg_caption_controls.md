# Task 05: Baseline Audit - DSG, Caption, And Controls

## Goal

Audit DAAAM, Hydra, ReMEmbR, Multi-frame VLM control, and LLM-with-captions
control against the Track 1/2 fixed API gate.

## Scope

This is an audit/design PR. It should decide which methods are object-memory
fixed API candidates, which are agentic-only, and which are controls.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- `.codex/modules.md`
- `/home/robin_wang/DAAAM`
- `/home/robin_wang/Hydra`
- `/home/robin_wang/remembr`

## Implementation Rules

- Treat DAAAM, Hydra, ReMEmbR, and controls separately.
- Use root repo evidence and cite `relative/path:line`.
- DAAAM/Hydra Track 1 support requires native DSG/object nodes with labels and
  3D positions.
- Track 2 support requires native object-location querying, not a benchmark
  LLM wrapper.
- ReMEmbR and caption/frame controls should not be promoted to object-memory
  fixed API unless native object memory exists.
- Human owns final decisions for ambiguous DSG/caption cases.

## Deliverables

- Update `.codex/baseline_registry.md` for DAAAM, Hydra, ReMEmbR, and controls.
- Record evidence paths or missing reasons for Track 1/2.
- Add a concise recommendations section: fixed API, agentic-only, control-only,
  or deferred.

## Acceptance Checks

- No-memory and caption controls are not treated as spatial-memory fixed API
  baselines.
- DAAAM and Hydra are not conflated.
- Ambiguous support remains `candidate` or `invalid`, with human-review notes.

## PR Title

`docs: audit dsg caption control baselines`
