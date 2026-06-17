# Task 04: DualMap Track 1/2 CV Fixed API

## Goal

Lock DualMap formal Track 1/2 evaluation to a closed-vocabulary variant. Unrestricted open-vocabulary DualMap results must be recorded only as `ov_ablation`.

## Scope

Use DualMap native memory artifacts while ensuring formal object labels and queries are constrained to the canonical closed-vocabulary class list.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/modules.md`
- `.codex/memory_package_spec.md`
- Existing DualMap scripts under `scripts/methods/dualmap/`

Root repo:

- `/home/robin_wang/DualMap`

## Implementation Rules

- Formal Track 1/2 must use `vocabulary_mode=closed`.
- Record class list path, detector/checkpoint, SAM/checkpoint, CLIP/checkpoint, and preprocessing.
- If DualMap internally uses YOLO-World or CLIP, restrict formal output/query labels to the canonical class list.
- Track 1 may export object inventory from `map/*.pkl`.
- Track 2 is supported only if the non-interactive query bridge is grounded in native DualMap object/query logic.
- Do not score unrestricted OV output in the main results.

## Deliverables

- DualMap package/export updates for CV formal mode.
- Manifest/build log fields for vocabulary and module metadata.
- Track 1 fixed API smoke.
- Track 2 fixed API support or explicit invalid reason.
- Optional `ov_ablation` note if unrestricted OV remains available.

## Acceptance Checks

- DualMap formal package declares `vocabulary_mode=closed`.
- Track 1 object labels are from the canonical class list.
- Track 2 uses structured `target_label` when supported.
- The report does not mix unrestricted OV results into formal metrics.

## PR Title

`feat: make dualmap track12 eval closed vocabulary`
