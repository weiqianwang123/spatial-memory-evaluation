# Task 13: DAAAM Track 1/2 Fixed API Finish And Smoke

## Goal

Finish the existing DAAAM Track 1/2 adapter work and get a local smoke memory
build/eval as far as the current runtime allows. This is a continuation task,
not a rewrite.

## Scope

Work only inside this evaluation repo. `/home/robin_wang/DAAAM` and
`/home/robin_wang/daaam_colcon_ws` are read-only evidence/runtime sources.

Existing partial implementation to continue from:

- `scripts/methods/daaam/build_memory_smoke.py`
- `scripts/methods/daaam/eval_memory_smoke.py`
- `scripts/methods/shared_modules.py`
- `spatial_memory_evaluation/shared_modules/registry.py`
- `.codex/baseline_registry.md`
- `.codex/modules.md`

Known local runtime context:

- DAAAM conda env: `/home/robin_wang/miniforge3/envs/daaam`
- DAAAM repo: `/home/robin_wang/DAAAM`
- Hydra/Spark workspace: `/home/robin_wang/daaam_colcon_ws`
- Hydra config candidate:
  `/home/robin_wang/daaam_colcon_ws/src/daaam_ros/config/hydra_config/clio_dataset_khronos.yaml`
- Test scene: ScanNet++ `036bce3393`

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- `.codex/modules.md`
- `scripts/methods/daaam/`
- `scripts/methods/shared_modules.py`
- `spatial_memory_evaluation/shared_modules/`
- `spatial_memory_evaluation/track1/`
- `spatial_memory_evaluation/track2/`
- `/home/robin_wang/DAAAM`
- `/home/robin_wang/daaam_colcon_ws`

## Implementation Rules

- Do not edit, create, delete, format, patch, checkout, reset, or commit files
  under `/home/robin_wang/DAAAM` or `/home/robin_wang/daaam_colcon_ws`.
- Preserve both build routes:
  - package from existing DAAAM native output;
  - prepare ScanNet++ layout and optionally run DAAAM.
- Fix adapter/runtime issues in this repo only. Recent blockers included
  repo-relative paths being passed into DAAAM while its cwd is the DAAAM repo,
  plus DAAAM env dynamic-library ordering for `ultralytics`, `hydra_python`,
  and `spark_dsg`.
- Prefer robust absolute paths in adapter commands and manifest/build logs.
- Track 1 fixed API may be supported only through exported DAAAM DSG object
  inventory with canonical labels and 3D positions.
- Track 2 fixed API may be supported only if a deterministic native DAAAM
  semantic object-location index exists. Do not use
  `SceneUnderstandingAgent.answer_query` or any LLM-orchestrated path for fixed
  API.
- If Track 2 native index is missing, declare Track 2 `invalid` with a clear
  reason while keeping Track 1 usable when object inventory exists.
- External model/checkpoint artifacts remain shared-module/NAS artifacts; Python
  import/runtime failures are conda/runtime issues.

## Deliverables

- Hardened DAAAM build/package adapter in `scripts/methods/daaam/`.
- DAAAM package validation path for package-from-output and raw-build routes.
- Track 1 fixed API object export smoke result if native DAAAM produces a DSG.
- Track 2 supported-or-invalid decision with evidence.
- Updated `.codex/baseline_registry.md` and `.codex/modules.md` entries.
- A short runbook section or comments with the exact local env variables needed
  for DAAAM smoke if they are still required.
- If the native build still cannot complete, a concise blocker note with the
  exact command, log excerpt, and next action.

## Acceptance Checks

- `python scripts/methods/daaam/build_memory_smoke.py --help` works.
- `--prepare-only` can create or read a ScanNet++ layout for scene `036bce3393`.
- Package-from-output gives actionable errors if no DSG is present.
- If a DSG is available, `scripts/package/validate_memory_package.py` passes.
- Track 1 eval writes summary/details/report or reports `invalid` honestly.
- Track 2 eval uses only native deterministic index or reports `invalid`.
- No files under external method/runtime repos are modified.

## PR Title

`feat: finish daaam track12 fixed api smoke`
