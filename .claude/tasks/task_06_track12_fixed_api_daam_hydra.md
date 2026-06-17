# Task 06: DAAAM/Hydra Track 1/2 Fixed API Exploration

## Goal

Explore whether DAAAM and standalone Hydra can honestly support Track 1/2 fixed API through native DSG/object artifacts.

## Scope

This is an exploration and design PR. Implement only minimal read/export code if native artifacts are clear. Final support judgment is human-owned.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/modules.md`
- `.codex/memory_package_spec.md`

Root repos:

- `/home/robin_wang/DAAAM`
- `/home/robin_wang/Hydra`

## Implementation Rules

- Treat DAAAM and Hydra separately.
- For Track 1, look for stable DSG object/semantic nodes with 3D position and label.
- For Track 2, look for native subject matching or object-location query tools.
- If the method only has a live agent/LLM query path, do not call that Track 1/2 fixed API unless it is native and deterministic enough for the track.
- Use closed-vocabulary formal eval when object labels are produced by detector/model modules.

## Deliverables

- Update `.codex/baseline_registry.md` with evidence paths and recommended status.
- Optional prototype reader/exporter if native artifact format is clear.
- A short risk list for human review.

## Acceptance Checks

- DAAAM and Hydra are not conflated.
- Each Track 1/2 status has evidence or a missing reason.
- Ambiguous support remains `candidate` or `invalid`; it is not promoted to supported.

## PR Title

`docs: assess daam hydra track12 fixed api support`
