# Agent-Designed Memory Baseline

Last updated: 2026-06-23

This is the project's centerpiece baseline. Instead of evaluating a fixed, human-
engineered spatial memory, we let a **coding agent design its own memory**: it reads
the shared modules, a contract, and some training cases, then writes the memory
construction code, the schema, and the query/tool interface. The harness builds the
agent's memory, runs it through the same Track 1/2/3 evaluators as every other
method, and (in the iterative variant) feeds errors back so the agent can improve.

If an agent-designed memory does well, hand-built scene graphs are not the only
answer. If it does poorly, that is also a result: current agents cannot yet design
stable, geometry-consistent, generalizable spatial memory.

This doc is the design and the contract for the `spatial_memory_evaluation/
agent_designed/` harness skeleton. Skeleton-first: interfaces and data flow are
fixed here; the agent-invocation internals are stubs to fill in Phase 4.

## 1. Position in the benchmark

- The agent-designed memory is just another method, with `method.family =
  "agent_designed"`. Its output is a normal memory package (see
  `memory_package_spec.md`) and it is scored by the unchanged Track 1/2/3
  evaluators, in both `fixed_api` and `tool_llm` modes.
- It uses the same shared modules registry as hand-built methods, so the
  perception stack (detector/segmenter/CLIP/VLM) is fair. What the agent designs is
  the **memory representation and the query/tool interface**, not a stronger
  detector.
- After it works on our own benchmark, it is a zero-shot transfer candidate for
  SG3D and OC-NaVQA (Phase 5), exactly like the other agentic baselines.

## 2. Roles and isolation

Three LLM/agent roles must stay isolated to avoid leakage and confounds:

1. **Designer agent** (coding agent): writes memory construction + query code.
   Sees: shared-module docs, training scenes/queries (no answers), contract,
   example packages, the metric definitions, and — in iterative mode — aggregate
   scores and failure summaries on the *training* split only.
2. **Method LLM/VLM** (optional): any LLM/VLM the designed memory itself calls at
   build or query time (e.g. a captioner). Declared as a shared module.
3. **Evaluator judge LLM**: scores Track 3 answers / evidence. Must be a different
   model instance/config from roles 1 and 2, and never sees the designer's code.

The designer never sees: test answers, test GT labels, held-out queries, or the
judge's prompts.

## 3. Data splits

```text
train  : scenes + queries WITH answers visible only to the harness, never to the
         designer. The designer may see the queries and a small number of WORKED
         examples (depending on variant), plus its own training-split scores.
heldout: scenes + queries used to score the final package. Never shown to the
         designer in any form.
transfer (Phase 5): SG3D / OC-NaVQA, zero-shot.
```

The split manifest lives under `benchmarks/agent_designed/<dataset>/splits.json`
(created in Phase 4). For ScanNet++ the current single scene `036bce3393` is too
small for a real split; the first skeleton uses it as a smoke "train==heldout"
degenerate split, clearly flagged, until more scenes are prepared.

## 4. Inputs given to the designer

The workspace builder (`agent_designed/workspace.py`) assembles a sandbox dir:

```text
<workspace>/
  CONTRACT.md            # what to build, the package contract, the rules
  shared_modules.md      # how to call shared detector/segmenter/CLIP/VLM/embeddings
  metrics.md             # Track 1/2/3 metric definitions (from this repo's docs)
  examples/              # 1-2 example memory packages (minimal + one real)
  train/                 # training scenes (RGB-D/pose/intrinsics links) + queries
  starter/               # empty build_memory.py / query_*.py templates to fill in
  README.md              # entrypoints the harness will call + how to run locally
```

What the workspace must NOT contain: test answers, held-out queries, GT label files,
evaluator adapter code, or the judge prompt. The workspace builder enforces this and
records a manifest of what was provided.

Shared modules are exposed through the same registry adapter as the methods, so the
designer calls a documented Python API (or CLI) rather than re-downloading models.

## 5. What the designer must output

A directory that the harness packages (or that already is a package) providing:

1. `build_memory.py` — builds memory from posed RGB-D for one scene/episode and
   writes a memory package under `memories/agent_designed/<dataset>/<scene>/<run>/`.
2. memory schema (`schema.md`) and any `schemas/*.json`.
3. query/tool interface implementing the fixed-API entrypoints it claims to support:
   - `query_object` for Track 1,
   - `resolve_referring_expression` for Track 2,
   - `answer_question` for Track 3,
   and/or native tools for `tool_llm` mode.
4. `capabilities.json` declaring which tracks are `supported` vs `invalid`.
5. `README.md` describing how to build and query.
6. evidence format (what each prediction returns as provenance).

The designer is free to choose the memory form (object table, graph, vector DB,
captions, hybrid) — that is the point of the baseline.

## 6. Constraints (anti-leakage / fairness)

- No access to test answers, GT labels, or held-out queries.
- No test-query-specific hardcoded rules. Enforced two ways: (a) the held-out split
  is never shown; (b) a leakage validator scans the designed code for embedded
  answer strings / scene-id-conditioned branches over held-out scenes.
- Same raw inputs and the same shared modules as hand-built methods.
- A compute/token budget per build and per iterative round (recorded).
- The produced memory must be saveable and reproducible: re-running `build_memory.py`
  on the same scene must yield an equivalent package.
- Every query must return evidence.

## 7. Harness control flow

`agent_designed/harness.py` (skeleton) orchestrates one run:

```text
1. workspace = build_workspace(variant, dataset, train_split)      # workspace.py
2. design    = invoke_designer(workspace, feedback=None)           # designer.py (STUB)
3. for scene in train_split.scenes (or heldout in final scoring):
       package = run_build(design, scene)                          # builds a package
       report  = validate_package(package)                         # existing validator
4. leakage   = scan_for_leakage(design, heldout_split)             # leakage.py (STUB)
5. scores    = evaluate(package, tracks=[1,2,3], modes=[...])      # existing evaluators
6. if variant == "iterative" and round < max_rounds:
       feedback = summarize_training_failures(scores)              # train split only
       goto 2 with feedback
7. final     = evaluate_on_heldout(best_design)                    # heldout scoring
8. write run report under results/agent_designed/<variant>/<timestamp>/
```

Steps 3/5/7 reuse the unchanged Track 1/2/3 evaluators. Steps 2/4/6 are the
agent-designed-specific pieces and start as stubs with clear TODOs.

## 8. Variants (increasing capability)

| Variant | Designer sees | Iterates? | Tests |
|---|---|---|---|
| `prompt_only` | only contract + metrics + shared modules | no | can a spec alone yield usable memory? |
| `few_example` | + a few worked training examples + error feedback | no | does a little supervision help? |
| `coding_agent` | + full training scenes/queries, writes full build+query code | no | full coding-agent design |
| `iterative` | + its own training-split scores each round | yes | self-improvement loop |

All variants share one workspace contract; they differ only in what
`build_workspace` exposes and whether `harness` loops.

## 9. Metrics and reporting

Primary score is the agent-designed package's performance on the **held-out** split
across Track 1/2/3 (same metrics as other methods). Secondary, baseline-specific:

- build cost the agent incurred (size / time-per-frame / peak — from Track 1
  accounting);
- designer cost (tokens / wall-clock / number of iterations);
- robustness: variance across scenes; reproducibility check pass/fail;
- transfer score on SG3D / OC-NaVQA (Phase 5).

Run report: `results/agent_designed/<variant>/<timestamp>/{run_summary.json,
design_manifest.json, eval_report.md}` plus the per-track eval outputs.

## 10. Skeleton scope (this refactor)

Implemented as importable skeleton with stable signatures and TODOs:

- `agent_designed/__init__.py`
- `agent_designed/contract.py` — workspace contract constants, allowed/forbidden
  inputs, the entrypoint names the harness will call.
- `agent_designed/workspace.py` — `build_workspace(...)` assembles the sandbox and
  the provided-inputs manifest; enforces anti-leakage on what gets copied in.
- `agent_designed/designer.py` — `invoke_designer(...)` STUB that documents how a
  coding agent (default Claude Code / Bedrock) is launched and what it returns.
- `agent_designed/leakage.py` — `scan_for_leakage(...)` STUB for embedded-answer /
  held-out-scene-conditioning checks.
- `agent_designed/harness.py` — `run_agent_designed(...)` orchestrating the control
  flow above by calling the existing validator + Track 1/2/3 evaluators.
- `scripts/agent_designed/run_baseline.py` — thin CLI over the harness.

Not in scope this refactor (Phase 4+): the real designer invocation, the real
leakage scanner, multi-scene splits, and SG3D/NaVQA transfer.

## 11. Open decisions (human-owned)

- Final train/heldout scene split for ScanNet++ (need more scenes than `036bce3393`).
- Compute/token budget per variant.
- Whether the designer may call a method LLM/VLM at query time, and how it is
  metered separately from the evaluator judge.
- Exact leakage-scan heuristics vs a held-out-only guarantee.
- How `agent_designed` rows are presented next to hand-built methods in the main
  table (separate section vs unified, to avoid implying the agent "won" unfairly).
