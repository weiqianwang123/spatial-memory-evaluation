# Task 11: Track 4 OpenEQA Design

## Goal

Draft the implementation design for OpenEQA adaptation after Track 1/2 are stable.

## Scope

Design only unless explicitly asked to implement. The final evaluator and LLM judge setup are human-owned.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- Existing Track 1/2 evaluator/report patterns

## Implementation Rules

- GT answers go only to scorer/judge, never to fixed API entrypoint or agent sandbox.
- Fixed API support requires a native QA/retrieval API.
- Do not add a generic object-table-to-LLM answerer and count it as method support.
- LLM judge must be separate from method memory construction modules.

## Deliverables

- A design doc section or new doc describing OpenEQA package mapping, fixed API eligibility, agentic prompt shape, scoring, evidence audit, and leakage policy.
- Recommended smoke scene/question subset.
- List of implementation tasks that can become future PRs.

## Acceptance Checks

- Design distinguishes native fixed QA from agentic full-access QA.
- Design states how invalid fixed API results are produced.
- Design records judge/model isolation requirements.

## PR Title

`docs: design openeqa track4 adaptation`
