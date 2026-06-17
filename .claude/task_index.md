# Claude Task Index

This directory contains standalone PR tasks for Claude agents. Each task should be done on its own branch and opened as a focused PR.

Global rules for every task:

- Do not use evaluation adapters as evidence for baseline capability. Inspect root method repos under `/home/robin_wang`.
- Do not mix unrestricted open-vocabulary results into formal Track 1/2 metrics. Formal Track 1/2 uses the closed-vocabulary detector-coverable setup.
- Do not change benchmark GT, memory package schema, or scoring semantics unless the task explicitly asks for it.
- Keep PRs narrow. Update docs/tests/smoke commands that directly prove the task.
- If a method cannot honestly support fixed API, declare `invalid` with a clear reason instead of adding a fake wrapper.

## Task List

| Task | Owner | Purpose |
|---|---|---|
| `tasks/task_01_baseline_capability_audit.md` | Claude | Audit native memory/API support from root repos and update the support matrix. |
| `tasks/task_02_shared_module_cv_audit.md` | Claude | Confirm the closed-vocabulary formal eval module stack and checkpoint paths. |
| `tasks/task_03_track12_fixed_api_claws.md` | Claude | Implement/smoke ClawS Track 1/2 fixed API package path. |
| `tasks/task_04_track12_fixed_api_dualmap_cv.md` | Claude | Lock DualMap formal Track 1/2 to the CV variant; OV only as ablation. |
| `tasks/task_05_track12_fixed_api_conceptgraphs_cv.md` | Claude | Add ConceptGraphs CV object export; Track 2 only if native bridge is real. |
| `tasks/task_06_track12_fixed_api_daam_hydra.md` | Claude draft, human final | Explore DAAAM/Hydra DSG object support and propose fixed API eligibility. |
| `tasks/task_07_track12_agentic_sandbox.md` | Claude | Generalize Track 1/2 agentic sandbox runner and Claude command path. |
| `tasks/task_08_build_resource_accounting.md` | Claude | Add build runtime, time per frame, memory size, peak RAM/VRAM accounting. |
| `tasks/task_09_evidence_contract.md` | Claude | Define evidence semantics/schema/validation without changing answer logic. |
| `tasks/task_10_track3_scanrefer_design.md` | Claude draft, human final | Draft ScanRefer adaptation design before implementation. |
| `tasks/task_11_track4_openeqa_design.md` | Claude draft, human final | Draft OpenEQA adaptation design before implementation. |

Human-owned final decisions:

- Final fixed API support judgment for ambiguous methods.
- Strongest shared detector/checkpoint choice for formal CV eval.
- Track 3/4 benchmark design and metric tradeoffs.
- Paper/report wording for memory-form fairness and evidence meaning.
