# Task 03: ClawS Track 1/2 Fixed API

## Goal

Implement or complete the ClawS memory package path for Track 1 and Track 2 fixed API smoke evaluation.

## Scope

ClawS should be the first full fixed API baseline because it has native spatial-memory construction and query services.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- Existing scripts under `scripts/methods/`
- Existing Track 1/2 evaluators under `spatial_memory_evaluation/track1/` and `spatial_memory_evaluation/track2/`

Root repo:

- `/home/robin_wang/ClawS-SpatialRAG`

## Implementation Rules

- Use ClawS native APIs and artifacts; do not invent evaluator-side behavior.
- Track 1 should read native stored objects or a thin exported object table.
- Track 2 should call a native or thinly wrapped ClawS query path.
- Use the canonical closed-vocabulary label list for formal Track 1/2.
- Record modules, build command, runtime, memory sizes, and capability declarations.

## Deliverables

- ClawS exporter/build smoke script if missing.
- Package `manifest.json`, `capabilities.json`, `schema.md`, and fixed API tools.
- A smoke command that builds/evaluates one small scene.
- Documentation update in `.codex/baseline_registry.md` if support status changes.

## Acceptance Checks

- A ClawS package validates.
- Track 1 fixed API produces metrics, not invalid.
- Track 2 fixed API produces metrics if the native query bridge works; otherwise it is explicitly invalid with reason.
- No GT annotations or benchmark answers enter the package.

## PR Title

`feat: add claws track12 fixed api smoke`
