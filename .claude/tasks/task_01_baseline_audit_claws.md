# Task 01: Baseline Audit - ClawS SpatialRAG

## Goal

Audit `/home/robin_wang/ClawS-SpatialRAG` and update the registry with
evidence-backed Track 1/2 fixed API eligibility for ClawS.

## Scope

This is an audit-only PR. Inspect ClawS native build, native memory artifacts,
native query/read APIs, and perception stack. Do not implement exporters.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- `.codex/modules.md`
- `/home/robin_wang/ClawS-SpatialRAG`

## Implementation Rules

- Use root repo evidence only. Do not cite evaluation adapters as capability
  proof.
- Every supported/candidate/invalid claim must include `relative/path:line`.
- Track 1 support requires native object/memory records with labels and 3D
  position, or a thin deterministic export from those records.
- Track 2 support requires a native or thin non-interactive query/read path.
- Do not promote general LLM behavior to fixed API support unless the root repo
  already exposes it as a native memory query capability.
- Leave implementation follow-ups for a later fixed API task.

## Deliverables

- Update `.codex/baseline_registry.md` ClawS row/section.
- Record evidence for native build/ingest, memory artifact/schema, query/read
  interface, perception modules, and Track 1/2 support decision.
- Add unresolved questions if any capability remains `candidate`.

## Acceptance Checks

- ClawS Track 1/2 fixed API status is evidence-backed.
- No adapters are used as support evidence.
- The registry distinguishes native memory/query capability from future package
  export work.

## PR Title

`docs: audit claws baseline capabilities`
