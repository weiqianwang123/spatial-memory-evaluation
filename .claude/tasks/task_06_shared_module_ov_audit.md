# Task 06: Shared Module Open-Vocabulary Detector Audit

## Goal

Freeze the formal shared OV detector module policy for Track 1/2 and
record exact checkpoints, class lists, and method adapter implications.

## Scope

Audit detector, segmenter, feature encoder, vocabulary, and preprocessing for
methods that build object memories. This task may update shared module registry
code and docs, but must not edit external method repos.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/modules.md`
- `.codex/baseline_registry.md`
- `spatial_memory_evaluation/shared_modules/`
- `scripts/methods/shared_modules.py`
- `spatial_memory_evaluation/common/labels.py`
- `spatial_memory_evaluation/assets/class_lists/detector_coverable.txt`
- `/home/robin_wang/ClawS-SpatialRAG`
- `/home/robin_wang/DualMap`
- `/home/robin_wang/HOV-SG`
- `/home/robin_wang/concept-graphs`
- `/home/robin_wang/DAAAM`

## Implementation Rules

- Formal Track 1/2 uses `vocabulary_mode=open_vocabulary` with the shared strongest OV detector.
- Prompt/evaluation labels and normalization must come from one repo-owned list.
- If multiple methods use the same module type, formal eval should use the same
  strongest feasible module/checkpoint/preprocess.
- Missing checkpoints must be marked missing, not silently substituted.
- External method repos must not be edited. Add translation/adaptation code
  under `scripts/methods/` or `spatial_memory_evaluation/shared_modules/`.
- Keep formal target and smoke fallback separate.

## Deliverables

- Update `.codex/modules.md`.
- Update `spatial_memory_evaluation/shared_modules/` if registry fields are
  missing.
- Update method adapter notes where shared modules must be translated into
  native CLI/Hydra args.
- Add unresolved items requiring human download/symlink/checkpoint action.

## Acceptance Checks

- Formal shared OV detector/class-list policy is explicit.
- Detector-backed methods state how they use the shared OV detector route.
- Missing checkpoints and fallback smoke choices are clearly separated.

## PR Title

`docs: freeze shared open vocabulary detector modules`
