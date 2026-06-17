# Task 13: DSG, Caption, And Control Fixed API Outcomes

## Goal

Produce honest Track 1/2 fixed API outcomes for DAAAM, Hydra, ReMEmbR, and
controls: supported only with native evidence, otherwise invalid/control-only.

## Scope

This task may add minimal readers/prototypes for clear native DSG/object
artifacts, but it should not force unsupported methods into fixed API.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- `.codex/modules.md`
- Existing method scripts under `scripts/methods/`
- `/home/robin_wang/DAAAM`
- `/home/robin_wang/Hydra`
- `/home/robin_wang/remembr`

## Implementation Rules

- Treat DAAAM, Hydra, ReMEmbR, Multi-frame VLM, and LLM-with-captions
  separately.
- Track 1 support requires object-level memory with labels and 3D positions.
- Track 2 support requires native object-location query/read capability.
- Caption/raw-frame controls should be control-only or agentic-only unless they
  have explicit object memory.
- Invalid fixed API results are formal results, not errors.
- Human owns final judgment for ambiguous DSG cases.

## Deliverables

- Registry update with final/recommended Track 1/2 outcomes.
- Invalid result/package stubs where appropriate.
- Optional minimal prototype reader for clear DSG object artifacts.
- Risk list for human review.

## Acceptance Checks

- No unsupported method is marked supported.
- Each invalid result has a clear reason code/message.
- Controls are not mixed with object-memory baselines.

## PR Title

`docs: finalize dsg caption control fixed api outcomes`
