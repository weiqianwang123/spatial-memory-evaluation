# Task 09: Evidence Contract

## Goal

Define what evidence means in memory packages and evaluation reports, then add lightweight validation/reporting support.

## Scope

Evidence is provenance/debug/grounding. It is not a second answer source and must not introduce benchmark leakage.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/memory_package_spec.md`
- Existing memory package validator
- Existing Track 1/2 report code

## Implementation Rules

- Evidence may reference frame ids, timestamps, crops, keyframes, masks, point clouds, bboxes, graph nodes, native artifact paths, or feature ids.
- Evidence must be generated during memory construction or package export.
- Evidence must not contain GT answer labels, benchmark query answers, or hand-written query rules.
- Track scoring should not change because evidence exists.

## Deliverables

- Evidence section in `.codex/memory_package_spec.md`.
- Lightweight evidence schema or conventions.
- Validator checks for broken evidence references when possible.
- Report fields indicating evidence availability/usage.

## Acceptance Checks

- Packages without evidence can still be valid if the capability declares why.
- Broken evidence paths are reported clearly.
- Track 1/2 metric values are unchanged by evidence validation.

## PR Title

`docs: define memory evidence contract`
