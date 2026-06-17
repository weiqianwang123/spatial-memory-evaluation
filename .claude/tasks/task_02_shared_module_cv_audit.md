# Task 02: Shared Module CV Audit

## Goal

Confirm the formal closed-vocabulary module stack for Track 1/2 and update the shared module registry with exact paths and missing checkpoints.

## Scope

Audit detector, segmenter, feature encoder, class list, and relevant preprocessing used by methods that build object memories.

## Context Files

- `.codex/modules.md`
- `.codex/agentic_eval_plan.md`
- `spatial_memory_evaluation/shared_modules/`
- `scripts/methods/shared_modules.py`
- `spatial_memory_evaluation/common/labels.py`
- `spatial_memory_evaluation/assets/class_lists/detector_coverable.txt`

Root repos to inspect:

- `/home/robin_wang/ClawS-SpatialRAG`
- `/home/robin_wang/DualMap`
- `/home/robin_wang/HOV-SG`
- `/home/robin_wang/concept-graphs`
- `/home/robin_wang/DAAAM`

## Implementation Rules

- Formal Track 1/2 uses closed vocabulary.
- If a method uses an open-vocabulary detector or CLIP query path, constrain its formal eval to the canonical class list.
- Use the strongest shared module/checkpoint that all relevant methods can realistically run.
- If the strongest model is missing, mark it missing. Do not silently substitute.
- Keep smoke fallback and formal target separate.
- External method repos must not be edited. Add or adjust adapters under `scripts/methods/` so shared modules are translated into native method args.

## Deliverables

- Update `.codex/modules.md` and `spatial_memory_evaluation/shared_modules/` with exact known checkpoint paths, missing checkpoints, and formal/smoke defaults.
- Update method adapters under `scripts/methods/` when a method needs shared detector/SAM/CLIP settings.
- Record the canonical closed-vocabulary class list and how OV methods become CV eval variants.
- Add unresolved items for downloads/symlinks that require human action.

## Acceptance Checks

- The registry states that unrestricted OV results are `ov_ablation` only.
- The registry names the formal CV detector/class list policy.
- Missing checkpoints are explicit and not treated as available.

## PR Title

`docs: define shared closed vocabulary module policy`
