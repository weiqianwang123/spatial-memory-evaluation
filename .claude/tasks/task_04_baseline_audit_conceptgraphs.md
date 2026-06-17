# Task 04: Baseline Audit - ConceptGraphs

## Goal

Audit `/home/robin_wang/concept-graphs` and update the registry with
evidence-backed Track 1/2 fixed API eligibility for ConceptGraphs.

## Scope

This is an audit-only PR. Focus on native object map artifacts, graph/object
fields, query/read scripts, perception modules, and the formal
shared OV-detector variant.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- `.codex/modules.md`
- `/home/robin_wang/concept-graphs`

## Implementation Rules

- Use root repo evidence only and cite `relative/path:line`.
- Track 1 support requires native object map artifacts with labels and 3D
  geometry or a thin deterministic export.
- Track 2 support requires a stable native query/read path. Do not add generic
  CLIP/LLM querying and count it as fixed API.
- Formal Track 1/2 must be shared OV-detector; method-native detector/OV override is `module_ablation`.
- Do not implement exporter changes in this audit task.

## Deliverables

- Update `.codex/baseline_registry.md` ConceptGraphs row/section.
- Record evidence for build path, artifact format, object schema, query/read
  capability, and detector/SAM/CLIP modules.
- Record Track 1/2 status and unresolved blockers.

## Acceptance Checks

- ConceptGraphs Track 1/2 status is evidence-backed.
- Track 2 remains candidate/invalid unless a native non-interactive bridge is
  identified.
- shared OV vs OV behavior is documented clearly.

## PR Title

`docs: audit conceptgraphs baseline capabilities`
