# Claude Task Index

This directory contains focused PR-sized tasks for Claude agents. The task order
matches `.codex/agentic_eval_plan.md`: audit first, then shared modules and
package contracts, then Track 1/2 fixed API, then Track 1/2 agentic, then
Track 3/4 design.

## Global Rules

- Use one branch and one PR per task.
- Read `.codex/agentic_eval_plan.md` before starting.
- Do not edit external method repos under `/home/robin_wang`; inspect them and
  adapt from this repo only.
- Do not use evaluation adapters as proof that a baseline supports a fixed API.
  Fixed API evidence must come from the method root repo or native artifacts.
- Do not invent LLM wrappers for fixed API support. Unsupported methods must be
  declared `invalid` with an evidence-backed reason.
- Formal Track 1/2 uses closed-vocabulary detector-coverable evaluation.
  Unrestricted open-vocabulary results are `ov_ablation` only.
- Keep memory package schema, GT/query builders, and scoring semantics unchanged
  unless the task explicitly asks for them.
- Do not spawn broad subagents that exceed the task scope. If a task needs more
  work than fits one PR, write clear follow-up items instead.

## Recommended Order

| Phase | Task | Owner | Purpose |
|---|---|---|---|
| 0 Audit | `tasks/task_01_baseline_audit_claws.md` | Claude | Audit ClawS native memory/build/query capability. |
| 0 Audit | `tasks/task_02_baseline_audit_hovsg.md` | Claude | Audit HOV-SG native memory/build/query capability. |
| 0 Audit | `tasks/task_03_baseline_audit_dualmap.md` | Claude | Audit DualMap native memory/build/query capability. |
| 0 Audit | `tasks/task_04_baseline_audit_conceptgraphs.md` | Claude | Audit ConceptGraphs native memory/build/query capability. |
| 0 Audit | `tasks/task_05_baseline_audit_dsg_caption_controls.md` | Claude draft, human final | Audit DAAAM, Hydra, ReMEmbR, and controls. |
| 0 Policy | `tasks/task_06_shared_module_cv_audit.md` | Claude draft, human final | Freeze formal closed-vocabulary shared module policy. |
| 1 Infra | `tasks/task_07_build_resource_accounting.md` | Claude | Add build runtime, size, and peak resource accounting. |
| 1 Infra | `tasks/task_08_evidence_contract.md` | Claude | Define evidence schema/conventions and validation. |
| 1 Fixed API | `tasks/task_09_track12_fixed_api_claws.md` | Claude | Implement ClawS Track 1/2 fixed API package smoke. |
| 1 Fixed API | `tasks/task_10_track12_fixed_api_hovsg_cv.md` | Claude | Implement HOV-SG closed-vocabulary Track 1/2 package smoke. |
| 1 Fixed API | `tasks/task_11_track12_fixed_api_dualmap_cv.md` | Claude | Implement DualMap closed-vocabulary Track 1/2 route. |
| 1 Fixed API | `tasks/task_12_track12_fixed_api_conceptgraphs_cv.md` | Claude | Implement ConceptGraphs closed-vocabulary object export. |
| 1 Fixed API | `tasks/task_13_track12_fixed_api_dsg_caption_invalids.md` | Claude draft, human final | Add invalid/prototype handling for DAAAM, Hydra, ReMEmbR, controls. |
| 2 Agentic | `tasks/task_14_agentic_sandbox_packager.md` | Claude | Build reusable sandbox/source-context packager. |
| 2 Agentic | `tasks/task_15_track12_agentic_eval_runner.md` | Claude | Integrate Track 1/2 agentic runner, prompts, output validation, reports. |
| 3 Design | `tasks/task_16_track3_scanrefer_design.md` | Claude draft, human final | Draft ScanRefer Track 3 design. |
| 4 Design | `tasks/task_17_track4_openeqa_design.md` | Claude draft, human final | Draft OpenEQA Track 4 design. |

## Human-Owned Final Decisions

- Final fixed API support judgment for ambiguous methods.
- Strongest detector/checkpoint and shared module policy for formal CV eval.
- Whether DSG/caption methods can enter fixed API or should be agentic-only.
- Track 3/4 metric and leakage-policy tradeoffs.
- Paper/report wording for memory-form fairness and evidence meaning.
