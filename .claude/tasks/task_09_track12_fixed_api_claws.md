# Task 09: ClawS Track 1/2 Fixed API

## Goal

Implement or complete the ClawS memory package path for Track 1 and Track 2
fixed API smoke evaluation.

## Scope

ClawS is the first full fixed API baseline if Task 01 confirms native spatial
memory construction and query support.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- `.codex/modules.md`
- Existing scripts under `scripts/methods/`
- `spatial_memory_evaluation/track1/`
- `spatial_memory_evaluation/track2/`
- `/home/robin_wang/ClawS-SpatialRAG`

## Implementation Rules

- Use ClawS native APIs/artifacts; do not invent evaluator-side behavior.
- Track 1 should export/read native stored objects or a thin object table view.
- Track 2 should call a native or thin non-interactive ClawS query/read path.
- Use canonical shared OV-detector labels for formal Track 1/2.
- Record modules, build command, runtime, memory sizes, and capability
  declarations.

## Deliverables

- ClawS exporter/build smoke script if missing.
- Package `manifest.json`, `capabilities.json`, `schema.md`, and fixed API
  entrypoints/tools.
- Smoke command for one small scene.
- Registry/doc update if support status changes.

## Acceptance Checks

- ClawS package validates.
- Track 1 fixed API produces metrics, not invalid.
- Track 2 fixed API produces metrics if native bridge works; otherwise invalid
  with reason.
- No GT annotations or benchmark answers enter the package.

## PR Title

`feat: add claws track12 fixed api smoke`
