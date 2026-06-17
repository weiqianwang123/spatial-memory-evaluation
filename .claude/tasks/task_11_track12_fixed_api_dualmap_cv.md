# Task 11: DualMap Track 1/2 CV Fixed API

## Goal

Implement or complete DualMap formal Track 1/2 evaluation as a
closed-vocabulary variant. Unrestricted OV results remain `ov_ablation`.

## Scope

Use DualMap native memory artifacts while ensuring formal object labels and
queries are constrained to the canonical closed-vocabulary class list.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/modules.md`
- `.codex/memory_package_spec.md`
- Existing DualMap scripts under `scripts/methods/dualmap/`
- `/home/robin_wang/DualMap`

## Implementation Rules

- Formal package must declare `vocabulary_mode=closed`.
- Record class list path, detector/checkpoint, SAM/checkpoint, CLIP/checkpoint,
  and preprocessing.
- If DualMap internally uses OV modules, restrict formal output/query labels to
  the canonical class list.
- Track 1 may export object inventory from native map artifacts.
- Track 2 is supported only if the non-interactive query bridge is grounded in
  native DualMap object/query logic.
- Do not score unrestricted OV output in main results.

## Deliverables

- DualMap package/export updates for CV formal mode.
- Manifest/build log fields for vocabulary and module metadata.
- Track 1 fixed API smoke.
- Track 2 fixed API support or invalid reason.
- Optional `ov_ablation` note if unrestricted OV remains available.

## Acceptance Checks

- DualMap formal package declares closed vocabulary.
- Track 1 object labels are canonical.
- Track 2 uses structured `target_label` only if supported.
- Reports do not mix OV ablation with formal metrics.

## PR Title

`feat: make dualmap track12 eval closed vocabulary`
