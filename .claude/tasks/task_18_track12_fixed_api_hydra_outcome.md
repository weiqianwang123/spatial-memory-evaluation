# Task 18: Hydra Track 1/2 Fixed API Outcome

## Goal

Determine and implement the honest Track 1/2 fixed API outcome for Hydra as a
standalone baseline.

## Scope

Hydra is treated separately from DAAAM. Inspect the Hydra root repo and native
artifacts, but only change this evaluation repo.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- `.codex/modules.md`
- Existing method scripts under `scripts/methods/`
- `/home/robin_wang/Hydra`
- `/home/robin_wang/daaam_colcon_ws/src/hydra` if present

## Implementation Rules

- Do not modify `/home/robin_wang/Hydra` or the colcon workspace.
- Inspect Hydra native outputs, config, Python bindings, and examples for
  object-level memory and object-location query support.
- Track 1 support requires object-level memory with labels and 3D positions.
- Track 2 support requires a deterministic native object-location query/read
  capability.
- If Hydra only provides geometry/places/rooms or generic DSG infrastructure
  without labeled object-location memory for this benchmark, mark Track 1/2
  fixed API as `invalid`.
- Do not build a benchmark-only LLM wrapper or object-query heuristic and call
  it fixed API.

## Deliverables

- Hydra fixed API support decision in `.codex/baseline_registry.md` with root
  repo evidence paths.
- Minimal invalid package/result route or documented package declaration if
  fixed API is unsupported.
- Any thin reader/prototype only if Hydra native artifacts clearly expose
  object labels and positions.
- Risk/follow-up notes for agentic evaluation if Hydra memory is better suited
  to sandbox full-access evaluation.

## Acceptance Checks

- Hydra Track 1/2 support is not overstated.
- Invalid result has `reason_code: unsupported_fixed_api` and a specific
  message.
- Evidence paths point to root repo files or native artifact schema, not to
  evaluation adapters.
- No external repo files are modified.

## PR Title

`docs: finalize hydra track12 fixed api outcome`
