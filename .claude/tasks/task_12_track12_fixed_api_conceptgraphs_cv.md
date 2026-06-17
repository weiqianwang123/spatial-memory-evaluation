# Task 12: ConceptGraphs Track 1/2 CV Fixed API

## Goal

Implement or complete ConceptGraphs closed-vocabulary Track 1 object export and
determine whether Track 2 fixed API is honestly supported.

## Scope

ConceptGraphs formal Track 1/2 evaluation must use a CV variant. Track 2 should
remain invalid unless a stable non-interactive native query bridge exists.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/modules.md`
- `.codex/memory_package_spec.md`
- Existing method scripts under `scripts/methods/`
- `/home/robin_wang/concept-graphs`

## Implementation Rules

- Inspect ConceptGraphs native object map and query scripts before editing.
- Formal output labels must be canonical closed-vocabulary labels.
- Preserve native object map/scene graph artifacts in the package.
- Exporting an object table for Track 1 is allowed as a thin readable view.
- Do not create a generic CLIP/LLM answerer and call it fixed API.

## Deliverables

- ConceptGraphs CV object export smoke script or package path.
- Track 1 package validation and fixed API smoke.
- Track 2 support decision with root repo evidence.
- Registry/docs updates.

## Acceptance Checks

- Package declares closed vocabulary for formal Track 1/2.
- Track 1 labels are canonical.
- Track 2 is either supported by native query bridge or invalid with reason.

## PR Title

`feat: add conceptgraphs closed vocabulary track1 export`
