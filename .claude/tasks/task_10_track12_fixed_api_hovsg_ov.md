# Task 10: HOV-SG Track 1/2 OV Fixed API

## Goal

Implement or complete HOV-SG shared OV-detector memory package export and Track
1/2 fixed API smoke evaluation.

## Scope

Use HOV-SG native graph/artifacts while constraining formal Track 1/2 labels and
queries to the shared OV prompt/evaluation label list.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- `.codex/modules.md`
- Existing HOV-SG scripts under `scripts/methods/hovsg/`
- `spatial_memory_evaluation/track1/`
- `spatial_memory_evaluation/track2/`
- `/home/robin_wang/HOV-SG`

## Implementation Rules

- Formal package must declare `vocabulary_mode=open_vocabulary`.
- Preserve native HOV-SG artifacts and export only thin readable views needed by
  fixed API.
- Track 1 may read graph/object node exports if labels and 3D locations are
  available.
- Track 2 is supported only if a native or thin non-interactive object-location
  query path is valid.
- Record SAM/CLIP/detector/checkpoint/preprocess metadata from shared modules.
- Unrestricted OV behavior is `module_ablation` only.

## Deliverables

- HOV-SG package/export updates for formal shared OV mode.
- Manifest/build log fields for vocabulary and module metadata.
- Track 1 fixed API smoke.
- Track 2 fixed API support or explicit invalid reason.
- Registry/doc updates.

## Acceptance Checks

- HOV-SG formal package validates and declares shared OV detector.
- Track 1 labels are canonical.
- Track 2 is not marked supported unless a native bridge exists.
- OV output is not mixed into formal metrics.

## PR Title

`feat: add hovsg shared ov track12 smoke`
