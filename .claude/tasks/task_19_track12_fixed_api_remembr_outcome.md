# Task 19: ReMEmbR Track 1/2 Fixed API Outcome

## Goal

Determine and implement the honest Track 1/2 fixed API outcome for ReMEmbR.

## Scope

ReMEmbR may be more relevant to temporal or QA-style memory than Track 1/2
object inventory/location. This task should verify its native capabilities and
record the correct fixed API status.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- `.codex/modules.md`
- Existing method scripts under `scripts/methods/`
- `/home/robin_wang/remembr`
- `/home/robin_wang/ReMEmbR` if present

## Implementation Rules

- Do not modify external ReMEmbR repo files.
- Inspect native memory artifacts, scripts, examples, and query interfaces.
- Track 1 support requires object-level memory with labels and 3D positions.
- Track 2 support requires deterministic object-location query/read capability.
- If ReMEmbR has temporal/event/caption/text memory but no object inventory or
  object-location API, declare Track 1/2 fixed API `invalid` and note likely
  Track 4 or deferred temporal relevance.
- Do not create a generic text/caption-to-location LLM wrapper for fixed API.

## Deliverables

- ReMEmbR registry update with Track 1/2 fixed API decision and root evidence.
- Minimal invalid package/result declaration if unsupported.
- Notes on which future track, if any, ReMEmbR should enter first.

## Acceptance Checks

- Track 1/2 status matches native capability.
- Invalid result reason is specific and reproducible.
- Agentic/future-track notes are separated from fixed API support.
- No external repo files are modified.

## PR Title

`docs: finalize remembr track12 fixed api outcome`
