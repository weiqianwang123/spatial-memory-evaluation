# Task 01: Baseline Capability Audit

## Goal

Audit every baseline from its root repo and update the Track 1/2 fixed API support matrix with evidence paths.

## Scope

Methods:

- ClawS SpatialRAG
- DualMap
- HOV-SG
- ConceptGraphs
- DAAAM
- Hydra standalone
- ReMEmbR
- Multi-frame VLM control
- LLM-with-captions control

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- `.codex/modules.md`

Root repos to inspect:

- `/home/robin_wang/ClawS-SpatialRAG`
- `/home/robin_wang/DualMap`
- `/home/robin_wang/HOV-SG`
- `/home/robin_wang/concept-graphs`
- `/home/robin_wang/DAAAM`
- `/home/robin_wang/Hydra`
- `/home/robin_wang/remembr`

## Implementation Rules

- Do not use `adapters/` as proof that a baseline supports an API.
- A fixed API is supported only if the root repo or native artifact already exposes the required information.
- Mark unsupported fixed API as `invalid`; do not propose an LLM wrapper.
- Formal Track 1/2 support must assume closed-vocabulary detector-coverable evaluation.
- Open-vocabulary-only behavior should be recorded as `ov_ablation`, not as the main result.

## Deliverables

- Update `.codex/baseline_registry.md`.
- For each method, record root repo evidence paths for native build, native memory artifact, native query/read interface, and Track 1/2 eligibility.
- Add a concise unresolved-questions section for methods that remain `candidate`.

## Acceptance Checks

- Every Track 1/2 support decision has an evidence path or an explicit missing reason.
- DualMap/HOV-SG/ConceptGraphs are described as needing CV eval variants for formal Track 1/2.
- ReMEmbR and no-memory controls are not treated as object-memory fixed API baselines.

## PR Title

`docs: audit baseline fixed api capabilities`
