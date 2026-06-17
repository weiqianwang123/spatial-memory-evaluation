# Task 16: Track 3 ScanRefer Design

## Goal

Draft the implementation design for ScanRefer adaptation after Track 1/2 are
stable.

## Scope

Design only unless explicitly asked to implement. The final evaluator design is
human-owned.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- Existing Track 1/2 evaluator/report patterns

## Implementation Rules

- Do not put ScanRefer GT target labels or answers into agent sandbox.
- Fixed API support requires native referring-expression capability or a clearly
  equivalent native object query API.
- Methods without such capability should be invalid for fixed API and still
  eligible for agentic eval.
- Keep closed-vocabulary Track 1/2 policy separate from referring-expression
  evaluation details.

## Deliverables

- Design doc or section covering query format, package requirements, fixed API
  eligibility, agentic prompt shape, metrics, and leakage policy.
- Recommended smoke subset and acceptance criteria.
- List of implementation tasks for future PRs.

## Acceptance Checks

- Design explains scene/object id and coordinate alignment.
- Design distinguishes fixed API from agentic access.
- Design states what invalid means for methods without referring resolvers.

## PR Title

`docs: design scanrefer track3 adaptation`
