# Task 08: Evidence Contract

## Goal

Define the evidence contract for memory packages and add lightweight validation
and reporting support.

## Scope

Evidence is provenance/debug/grounding. It is not an extra answer source and
must not introduce benchmark leakage.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/memory_package_spec.md`
- Existing memory package validator
- Existing Track 1/2 report code

## Implementation Rules

- Evidence may reference frame ids, timestamps, crops, keyframes, masks, point
  clouds, bboxes, graph nodes, native artifact paths, or feature ids.
- Evidence must be produced during memory construction or package export.
- Evidence must not contain GT answers, benchmark query answers, or handwritten
  query rules.
- Track 1/2 metric values must not change because evidence exists.
- Packages without evidence may remain valid if they honestly declare why.

## Deliverables

- Evidence section in `.codex/memory_package_spec.md`.
- Lightweight evidence schema/conventions.
- Validator checks for broken evidence references when feasible.
- Report fields for evidence availability/usage.

## Acceptance Checks

- Broken evidence paths produce clear validation/report messages.
- Packages without evidence can still be valid with a declared reason.
- Track 1/2 metrics remain unchanged by evidence validation.

## PR Title

`docs: define memory evidence contract`
