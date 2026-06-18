# Claude Task Index

This directory contains focused PR-sized tasks for Claude agents. Tasks 01-11
were completed and retired; the active queue now starts from the remaining
Track 1/2 fixed API work, then agentic evaluation, then Track 3/4 design.

## Global Rules

- Use one branch and one PR per task.
- Read `.codex/agentic_eval_plan.md` before starting.
- Do not edit external method repos under `/home/robin_wang`; inspect them and
  adapt from this repo only.
- Do not run shell write operations inside external method repos, including
  `sed -i`, Python file writes, `apply_patch`, formatter commands,
  `git checkout`, `git reset`, or `git commit`. If an external repo is dirty,
  report it instead of modifying or reverting it.
- Do not use evaluation adapters as proof that a baseline supports a fixed API.
  Fixed API evidence must come from the method root repo or native artifacts.
- Do not invent LLM wrappers for fixed API support. Unsupported methods must be
  declared `invalid` with an evidence-backed reason.
- Formal Track 1/2 uses the shared strongest open-vocabulary detector setup and
  reports the detector-coverable split. Closed-detector or method-native
  detector variants are `module_ablation` only.
- Keep memory package schema, GT/query builders, and scoring semantics
  unchanged unless the task explicitly asks for them.
- Do not spawn broad subagents that exceed the task scope. If a task needs more
  work than fits one PR, write clear follow-up items instead.

## Active Queue

| Phase | Task | Owner | Purpose |
|---|---|---|---|
| 1 Fixed API | `tasks/task_12_track12_fixed_api_conceptgraphs_ov.md` | Claude | Implement ConceptGraphs shared OV-detector object export. |
| 1 Fixed API | `tasks/task_13_track12_fixed_api_daaam_finish_smoke.md` | Claude | Finish and smoke the partially implemented DAAAM Track 1/2 adapter. |
| 1 Fixed API | `tasks/task_18_track12_fixed_api_hydra_outcome.md` | Claude draft, human final | Determine Hydra Track 1/2 fixed API support or invalid status. |
| 1 Fixed API | `tasks/task_19_track12_fixed_api_remembr_outcome.md` | Claude draft, human final | Determine ReMEmbR Track 1/2 fixed API support or invalid status. |
| 1 Follow-up | `tasks/task_22_finish_hydra_task18.md` | Claude | Finish and commit incomplete Hydra Task 18 work. |
| 1 Follow-up | `tasks/task_23_fix_remembr_task19_no_memories.md` | Claude | Repair ReMEmbR Task 19 so no `memories/` files are in branch history. |
| 1 Follow-up | `tasks/task_24_unify_control_tasks_20_21.md` | Claude | Combine Task 20 and 21 control semantics into one clean branch. |
| 2 Agentic | `tasks/task_14_agentic_sandbox_packager.md` | Claude | Build reusable sandbox/source-context packager. |
| 2 Agentic | `tasks/task_15_track12_agentic_eval_runner.md` | Claude | Integrate Track 1/2 agentic runner, prompts, output validation, reports. |
| 3 Design | `tasks/task_16_track3_scanrefer_design.md` | Claude draft, human final | Draft ScanRefer Track 3 design. |
| 4 Design | `tasks/task_17_track4_openeqa_design.md` | Claude draft, human final | Draft OpenEQA Track 4 design. |

## Launch Notes

Recommended worktree naming:

```bash
git worktree add -b claude/task-13-daaam-finish-smoke ../spatial-memory-evaluation-task13
python scripts/tools/run_claude_task.py 13 --worktree ../spatial-memory-evaluation-task13 --background --stream-json
```

Use one worktree per task. The dashboard reads `.claude/session_logs/` inside
each worktree.

## Human-Owned Final Decisions

- Final fixed API support judgment for ambiguous methods.
- Strongest OV detector/checkpoint and shared module policy for formal Track 1/2
  eval.
- Whether DSG/caption methods can enter fixed API or should be agentic-only.
- Track 3/4 metric and leakage-policy tradeoffs.
- Paper/report wording for memory-form fairness and evidence meaning.
