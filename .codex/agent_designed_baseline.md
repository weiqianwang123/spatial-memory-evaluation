# Agent-Designed Memory Baseline (Auto-Research Self-Improvement)

Last updated: 2026-06-25

This is the project's centerpiece baseline. Instead of evaluating a fixed, human-
engineered spatial memory, we let a **coding agent design and continuously improve its
own semantic-map memory**. The agent reads the shared perception modules, the package
contract, and a few dev scenes; it writes the memory-construction code, the memory
schema, and the query/tool interface; it builds the memory; it **evaluates itself on
its own test cases**; it reads its own failures and logs; it edits its own code; and it
repeats — an *auto-research* self-improvement loop — until a compute budget or a
convergence criterion is hit. Only then is the frozen result scored, once, on the
held-out 10-scene benchmark by the unchanged Track 1/2/3 evaluators.

If an agent-designed memory does well, hand-built scene graphs are not the only answer.
If it does poorly, that is also a result: current coding agents cannot yet self-improve
their way to stable, geometry-consistent, generalizable spatial memory.

This doc is the design and the contract for the `spatial_memory_evaluation/
agent_designed/` harness. Skeleton-first: the interfaces, data flow, and the loop's
control structure are fixed here; the agent-invocation internals are stubs to fill in
when we implement (we are **not** implementing yet — this doc is the spec).

## 1. Intellectual lineage (what we borrow, adapted to semantic mapping)

Two recent projects define the shape of the loop. We adapt both, in service of
*semantic-map memory for spatial QA/localization* rather than their original domains.

- **AutoResearch** (Karpathy, `github.com/karpathy/autoresearch`) — a compact
  autonomous research loop. A coding agent is handed a real training setup and a
  human-owned instruction file (`program.md`); overnight it *modifies the code, runs
  under a fixed time budget, checks whether the metric improved, keeps or discards, and
  repeats* (~100 experiments/night), leaving a journal of experiments and a better
  artifact. **What we borrow:** the fixed per-round build+eval budget, the
  keep-or-discard-against-best decision rule, the human-owned "skill"/instruction file,
  and an append-only experiment journal as the loop's memory.
- **Learning Beyond Gradients** (Jiayi Weng,
  `trinkle23897.github.io/learning-beyond-gradients`) — "Heuristic Learning": a coding
  agent improves a *Heuristic System* (rules, state detectors, tests, memory) **without
  any weight updates**, by reading failures/logs, editing code/tests/memory, rerunning,
  and writing results back into trials and summaries. Continual learning is kept
  **explicit**: old capabilities live as regression tests, golden traces, and version
  diffs (readable, deletable, refactorable), and a healthy system must both *absorb
  feedback* **and** *compress history*. **What we borrow:** the no-gradient code/memory-
  editing update rule, the discipline of carrying capabilities as explicit regression
  tests + golden traces, and the absorb-feedback / compress-history balance so the
  semantic-map codebase stays maintainable across rounds.

Neither original targets 3D semantic memory. Here the "artifact" being improved is the
**semantic-map memory + its query interface for ScanNet scenes**; the "metric" is the
agent's *own* dev-split self-evaluation (aligned to the Track 1/2/3 metrics but owned by
the agent); and the held-out benchmark plays the role of a one-shot, never-seen test.

## 2. Position in the benchmark

- The agent-designed memory is just another method, `method.family = "agent_designed"`.
  Its frozen output is a normal memory package (see `memory_package_spec.md`) scored by
  the **unchanged** Track 1/2/3 evaluators in both `fixed_api` and `tool_llm` modes, on
  the **same 10 held-out ScanNet scenes** as DAAAM / ClawS / ReMEmbR / the controls
  (`scene0015_00, 0050_00, 0077_00, 0084_00, 0131_00, 0193_00, 0207_00, 0222_00,
  0256_00, 0314_00`). This is what makes its number directly comparable.
- It uses the same shared modules registry as the hand-built methods, so the perception
  stack (detector / segmenter / CLIP / VLM / embeddings) is fair. What the agent designs
  and self-improves is the **memory representation and the query/tool interface**, not a
  stronger detector. All build/query-time describers/captioners/embedders must be the
  **local** stack (qwen via ollama), exactly as the other methods — see §6.
- After it works on our benchmark it is a zero-shot transfer candidate for SG3D /
  OC-NaVQA (deferred), like the other agentic baselines.

## 3. The self-improvement loop (centerpiece)

```text
  ┌─ round r (fixed build+eval budget B_r) ───────────────────────────────────┐
  │ 1. PROPOSE   designer agent edits build_memory.py / query_*.py / memory     │
  │              schema / its own dev tests, given: contract, shared modules,    │
  │              the dev scenes, last round's eval report + failure cases + logs,│
  │              and the running experiment journal. (no held-out scenes, ever)  │
  │ 2. BUILD     run build_memory.py on each DEV scene -> a memory package        │
  │              (validated by the existing package validator).                  │
  │ 3. SELF-EVAL run the agent's OWN test cases over the package via the          │
  │              unchanged Track 1/2/3 evaluators -> dev score + per-case failures│
  │ 4. REFLECT   designer reads the dev report, failing cases, evidence, and      │
  │              build logs; writes a short rationale into the journal.           │
  │ 5. KEEP/DISCARD  if dev score improves over best-so-far -> promote to best    │
  │              and snapshot a version diff + refresh golden traces; else revert │
  │              the code to best-so-far (keep the journal note about why).       │
  └──> repeat until budget exhausted OR K consecutive rounds with no dev gain ────┘
  FREEZE best-so-far design ── then, ONCE: score on the 10 held-out scenes.
```

Mapping to the lineage: step 1/5 is AutoResearch's *edit → keep-or-discard-vs-best*;
the no-gradient *edit code/tests/memory* update and the journal/golden-trace bookkeeping
are Learning Beyond Gradients. The freeze-then-score-once on held-out is our fairness
guarantee — the loop never touches the 10 benchmark scenes.

## 4. Roles and isolation

Three LLM/agent roles stay isolated to avoid leakage and confounds:

1. **Designer agent** (the coding agent, default Claude Code / Bedrock): the only role
   that *writes and edits code*. It runs the loop in §3. It sees: shared-module docs,
   the package contract, the metric definitions, the **dev** scenes + the dev cases it
   authors, its **own dev-split** scores/failures/logs, and the experiment journal.
   The designer is a *development-time* component, **not** a runtime memory component —
   so using a strong coding LLM here does not violate the "local describers only" rule
   (§6); it is analogous to the human engineer who wrote DAAAM/ClawS.
2. **Method LLM/VLM** (runtime): any model the *designed memory itself* calls at build
   or query time (captioner, describer, embedder). Must be the shared **local** stack
   (qwen3.5:4b / qwen3-embedding via ollama), declared as a shared module and metered.
3. **Evaluator judge LLM**: scores Track 3 answers/evidence (sonnet, medium tier).
   A different model/config from roles 1–2; never sees the designer's code or journal.

The designer never sees: any held-out (10-scene) data, the held-out GT, the held-out
queries, or the judge's prompts.

## 5. Data splits

```text
dev      : ~3 ScanNet scenes OUTSIDE the held-out 10. The designer has FULL access
           (RGB-D/pose/intrinsics + GT-derivation tooling) and AUTHORS ITS OWN test
           cases here (object-location / referring / QA, aligned to T1/T2/T3 metrics).
           This is the only data the self-improvement loop ever evaluates on.
held-out : the 10 ScanNet scenes shared by all methods. Used ONCE, after FREEZE, for
           the reported score. Never shown to the designer in any form.
transfer : SG3D / OC-NaVQA, zero-shot (deferred).
```

**Why the agent authors its own dev test cases.** The user's framing is auto-research:
the agent improves against *its own* evaluation, not a held-out leaderboard. On the dev
scenes the agent has GT geometry, so it can generate faithful self-tests (e.g. "where is
the <label>?" with GT bbox centers, referring utterances with GT targets, QA pairs) and
maintain them as regression tests + golden traces (Learning Beyond Gradients). The
harness *provides the GT-derivation tooling* (e.g. `track2/scannet_bbox.py`, the Track 1
data builder) on the dev scenes so the agent's self-tests are metric-faithful, but the
agent decides which cases to keep and chase. Self-eval uses the unchanged Track 1/2/3
evaluators so dev scores are on the same scale as the held-out report.

**Dev-scene acquisition (open item, confirmed needed).** A scan of the local ScanNet
mirror found **27** candidate scenes that appear in BOTH ScanEnts3D (referring) and
OpenEQA-ScanNet (QA), have a local `.sens`, and are NOT in the held-out 10 — e.g.
`scene0356_00, scene0406_00, scene0426_00, scene0435_00, scene0462_00, scene0500_00,
…`. These have frames (`.sens`) locally but **not** their GT geometry: the 4 ScanNet
annotation files per scene (`_vh_clean_2.ply`, `.aggregation.json`,
`_vh_clean_2.0.010000.segs.json`, `.txt`) must be downloaded from
`kaldir.vc.in.tum.de/scannet/v2/scans/<scene>/...`, exactly as was done for the 10
held-out scenes (gating is a keypress; see `eval_set_inventory.md` /
`daaam-native-build-env`). Pick ~3 of the 27 spanning small/medium/large room sizes;
final choice is an open decision (§11).

The split manifest will live under `benchmarks/agent_designed/scannet/splits.json`.

## 6. Constraints (anti-leakage / fairness)

- **Held-out isolation**: the loop never builds, evals, reads, or is told about the 10
  scenes. Enforced structurally — `build_workspace` only ever copies dev scenes in.
- **No per-scene hardcoding**: a leakage validator scans the designed code for embedded
  answer strings and scene-id-conditioned branches (over both dev and held-out ids).
- **Same perception + vocabulary** as hand-built methods (shared modules registry, same
  OV class list). The agent designs *representation + query logic*, not perception.
- **Local runtime models only**: every build/query-time describer/captioner/embedder is
  the local qwen stack via ollama — no Claude/Bedrock inside the *memory* (the designer
  agent is development-time only, §4).
- **Budget**: a compute/token budget per round and per full loop, recorded (designer
  tokens + wall-clock + #rounds, and the runtime build cost the package incurs).
- **Reproducible**: re-running the frozen `build_memory.py` on a scene yields an
  equivalent package; every query returns evidence.

## 7. What the designer maintains and outputs

A working directory that *is* (or that the harness packages into) a memory package,
plus the loop's bookkeeping — its "Heuristic System":

1. `build_memory.py` — builds memory from posed RGB-D for one scene and writes a package
   under `memories/agent_designed/scannet/<scene>/<run>/`.
2. memory schema (`schema.md`) + any `schemas/*.json`.
3. query/tool interface for the tracks it claims: `query_object` (T1),
   `resolve_referring_expression` (T2), `answer_question` (T3), and/or native tools for
   `tool_llm` mode.
4. `capabilities.json` declaring which tracks are `supported` vs `invalid`.
5. `README.md` — how to build and query.
6. evidence format (provenance returned per prediction).
7. **dev test cases** (`dev_tests/`) — the agent's self-authored cases + GT, kept as
   regression tests.
8. **experiment journal** (`journal.jsonl`) — one row per round: what changed, dev score
   before/after, keep/discard decision, rationale, budget spent.
9. **golden traces / version diffs** of best-so-far, so a regression in a later round is
   detectable and revertible (Learning Beyond Gradients' "compress history").

The designer is free to choose the memory form (object table, graph, vector DB,
captions, hybrid) — that is the point of the baseline.

## 8. Harness control flow (skeleton)

`agent_designed/harness.py` orchestrates one loop:

```text
1. workspace   = build_workspace(dataset, dev_split)              # workspace.py
2. best        = None
3. for r in range(max_rounds):
4.     design  = invoke_designer(workspace, history=journal)      # designer.py (STUB)
5.     for scene in dev_split.scenes:
6.         package = run_build(design, scene)                     # build_memory.py
7.         report  = validate_package(package)                    # existing validator
8.     dev_score   = evaluate(packages, design.dev_tests,         # existing T1/2/3
                              tracks=[1,2,3], modes=[...])         #   evaluators
9.     leakage     = scan_for_leakage(design, heldout_ids)        # leakage.py (STUB)
10.    journal.append(round_record(design, dev_score, ...))
11.    if improved(dev_score, best): best = snapshot(design)      # keep
12.    else: revert(design, best)                                 # discard
13.    if converged(journal, K) or budget_exhausted(): break
14. final = evaluate_on_heldout(best, the_10_scenes)              # ONCE, frozen
15. write run report under results/agent_designed/<timestamp>/
```

Steps 6/7/8/14 reuse the unchanged validator + Track 1/2/3 evaluators. Steps 4/9 (and
the keep/discard policy 11–13) are the agent-designed-specific pieces, stubbed with
clear TODOs. Step 14 is the only time the held-out 10 scenes are touched.

## 9. Variants (ablations of the loop)

The headline result is the **full self-improvement loop** (`auto_research`). The others
are ablations that isolate where the gains come from:

| Variant | Loops? | Self-authored dev tests? | Tests the question |
|---|---|---|---|
| `one_shot` | no (1 round) | n/a (provided dev cases) | can a coding agent design usable memory in one pass? |
| `loop_fixed_tests` | yes | no — harness-provided dev cases | does iteration help, holding the eval fixed? |
| `auto_research` | yes | **yes** — agent owns its dev tests | full auto-research self-improvement (centerpiece) |

All variants share one workspace contract; they differ only in whether `harness` loops
and whether the agent authors its own dev cases.

## 10. Metrics and reporting

Primary score is the **frozen** package's performance on the **held-out 10 scenes**
across Track 1/2/3 (same metrics as every other method — T1 success@{1,5} + proximity,
T2 acc@{0.25,0.5}m + proximity, T3 llm_match + answered_rate). Secondary,
baseline-specific:

- **dev-score trajectory**: per-round dev score (the loop's learning curve) — the
  AutoResearch "progress" artifact;
- **designer cost**: tokens / wall-clock / number of rounds to convergence;
- **runtime build cost** the produced memory incurs (size / time-per-frame, from the
  package's build_log.json, comparable to the other methods);
- **robustness**: variance across held-out scenes; reproducibility check pass/fail;
  dev↔held-out generalization gap (overfitting to self-authored tests is itself a
  finding);
- transfer score on SG3D / OC-NaVQA (deferred).

Run report: `results/agent_designed/<variant>/<timestamp>/{run_summary.json,
journal.jsonl, design_manifest.json, eval_report.md}` plus the per-track eval outputs.
To avoid implying the agent "won" unfairly, agent_designed rows are reported in their
own section alongside — not merged into — the hand-built-method table.

## 11. Skeleton scope (when we implement)

Importable skeleton with stable signatures and TODOs:

- `agent_designed/__init__.py`
- `agent_designed/contract.py` — workspace contract constants, allowed/forbidden
  inputs, entrypoint names, the held-out scene-id blocklist.
- `agent_designed/workspace.py` — `build_workspace(...)` assembles the dev-only sandbox
  + the GT-derivation tooling for self-test authoring + provided-inputs manifest;
  enforces held-out isolation on what gets copied in.
- `agent_designed/designer.py` — `invoke_designer(workspace, history)` STUB documenting
  how the coding agent is launched and what it returns each round.
- `agent_designed/leakage.py` — `scan_for_leakage(...)` STUB for embedded-answer /
  held-out-scene-id conditioning checks.
- `agent_designed/journal.py` — append-only round journal + best-so-far snapshot /
  revert + golden-trace refresh.
- `agent_designed/harness.py` — `run_agent_designed(...)` orchestrating §8 via the
  existing validator + Track 1/2/3 evaluators.
- `scripts/agent_designed/run_baseline.py` — thin CLI over the harness.

Not in this skeleton: the real designer invocation, the real leakage scanner, the
SG3D/NaVQA transfer.

## 12. Open decisions (human-owned)

- **Which ~3 dev scenes** (of the 27 candidates) and confirmation to download their 4
  ScanNet GT files each from kaldir (frames `.sens` already local).
- Whether the agent **authors its own dev test cases** (centerpiece intent) vs uses
  harness-provided dev cases — and, if self-authored, any guardrail that its dev cases
  stay metric-faithful (e.g. GT pulled only via the provided tooling).
- Compute/token budget per round and per loop; `max_rounds` and the no-gain
  convergence window `K`.
- Whether the designer may call a runtime method LLM/VLM and how it is metered vs the
  evaluator judge.
- Exact leakage-scan heuristics vs the structural held-out-only guarantee.
- How the loop persists across days (resume from journal) and how much human shaping the
  `program.md`-style instruction file is allowed to carry.
