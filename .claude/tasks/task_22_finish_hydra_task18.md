# Task 22: Finish Hydra Task 18

## Goal

Finish the incomplete Hydra standalone Track 1/2 fixed-API outcome work from
`/home/robin_wang/spatial-memory-evaluation-task18` and commit it cleanly.

## Current Problem

Task 18 produced useful uncommitted changes but did not commit. The Claude
session ended with an API error and also stopped because it inspected
`/home/robin_wang/daaam_colcon_ws/src/hydra`, which is dirty from earlier local
DAAAM runtime pybind/hydra_python patches. That colcon workspace fork is not the
Hydra standalone root repo for this task.

Current Task 18 worktree state:

- Worktree: `/home/robin_wang/spatial-memory-evaluation-task18`
- Branch: `claude/task-18-hydra-outcome`
- Uncommitted files:
  - `.codex/baseline_registry.md`
  - `scripts/methods/README.md`
  - `scripts/methods/hydra/declare_invalid_package.py`

## Scope

Continue in the existing Task 18 worktree. Do not restart from main.

Use `/home/robin_wang/Hydra` as the standalone Hydra root repo evidence source.
Do not inspect or rely on `/home/robin_wang/daaam_colcon_ws/src/hydra` unless
absolutely necessary; it is a known dirty runtime fork for DAAAM, not the
standalone Hydra evidence root for this task.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- `.codex/modules.md`
- `scripts/methods/README.md`
- `scripts/methods/hydra/declare_invalid_package.py`
- `/home/robin_wang/Hydra`

## Implementation Rules

- Do not modify `/home/robin_wang/Hydra` or any external repo.
- Track 1/2 fixed API for Hydra standalone should be finalized as `invalid`
  unless the root repo itself provides Hydra-produced object labels and a native
  deterministic object-location query API.
- The current conclusion is expected to remain:
  - Track 1 invalid: Hydra object `semantic_label` comes from an external labels
    stream, not from Hydra.
  - Track 2 invalid: no native object-location query API.
  - Track 3/4 invalid: no referring resolver / QA API.
- Keep `scripts/methods/hydra/declare_invalid_package.py` as a generator for an
  ignored declaration package/results; do not commit generated `memories/`,
  `results/`, or `data/` outputs.
- If the script writes generated packages/results during tests, verify they are
  ignored by git.
- If `/home/robin_wang/Hydra` is dirty, stop and report. The known dirty
  `/home/robin_wang/daaam_colcon_ws/src/hydra` fork should not block this task
  if you did not inspect it.

## Deliverables

- Clean committed Task 18 branch.
- `scripts/methods/hydra/declare_invalid_package.py` checked for robustness,
  path handling, and clear invalid reasons.
- `.codex/baseline_registry.md` and `scripts/methods/README.md` updated with
  Hydra standalone final status and evidence paths.
- No generated `memories/`, `results/`, or `data/` files tracked.

## Acceptance Checks

- `git status --short` in the Task 18 worktree is clean after commit.
- `git diff --name-only main...HEAD` contains no paths under `memories/`,
  `results/`, or `data/`.
- The Hydra declaration script runs with `--no-result` or a temporary output dir
  and validates the package, or else reports an actionable error.
- `/home/robin_wang/Hydra` remains clean.

## PR Title

`docs: finalize hydra fixed api invalid outcome`
