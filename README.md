# 🔥 MemForge

**A benchmark for spatial memory as an external resource for embodied agents — and an
auto-research loop where a coding agent *forges* its own memory and self-improves against it.**

> **MemForge** = **Mem**ory + **Forge**: the project both *evaluates* spatial-memory methods and
> lets an AI *forge* (design + iteratively refine) a new one against real metrics.

Robots that operate over long horizons need a *spatial memory*: a persistent, queryable
representation of everything they have seen. MemForge asks two questions:

1. **How good is a given spatial-memory method** when an agent must actually *use* it to
   answer questions? We evaluate each method's **native memory format** — no re-implementation,
   no guessed APIs — through either its declared Python API or a per-query LLM + native-tool loop.
2. **Can an AI design a better spatial memory than hand-built systems?** A coding agent is
   given the metrics, the perception stack, and a blank slate, and runs a git-driven
   *keep/revert* auto-research loop: propose code → build memory → self-evaluate → keep if it
   improved, revert if not → repeat for hours.

Everything is **package-first**: every method (hand-built or agent-designed) exports a minimal
*memory package*, and the same evaluators score them all on the same scenes.

---

## The Benchmark — three tracks (+ a fourth in progress)

| Track | Capability key | What it tests | Primary metric | Data |
|---|---|---|---|---|
| **1** | `track1_object_location` | locate a named object in 3D | success@1 | ScanNet |
| **2** | `track2_scanrefer` | resolve a referring expression to one instance | acc@0.5m | ScanRefer / ScanNet |
| **3** | `track3_openeqa` | answer open-ended spatial questions | LLM-Match | OpenEQA (ScanNet) |
| **4** *(WIP)* | `track4_oc_navqa` | long-horizon robot-trajectory QA | per-type* | OC-NaVQA / CODa |

\* Track 4 scores per question type: position (L2), binary (accuracy), time/duration (min-error), text (LLM judge).

**Two evaluation interfaces**, both preserving the method's real memory:
- `fixed_api` — call the package's declared native query entrypoint (deterministic, fast).
- `tool_llm` — a per-query LLM agent calls the method's native retrieval tools and produces the
  answer. This is the fair cross-method protocol (same agent, same scenes; only the memory differs).

Unsupported APIs are reported as `invalid` with a reason — never silently approximated.

---

## Auto-designed memory (the centerpiece)

Instead of hand-building a memory, a coding agent designs one from scratch under a fixed contract,
then improves it with a git-driven keep/revert loop (Karpathy-style AutoResearch):

```
propose code → build memory (all dev scenes) → score on the real Track 1/2/3 evaluators
   → loop_objective improved?  ─ yes → git commit (kept)
                               └ no  → git reset  (reverted)
   → repeat until the time budget runs out
```

The git log *is* the experiment journal — every commit is a genuine improvement. The most recent
run (**run4**: blind/pure-discovery, 6 dev scenes, 14 h budget, T2 on the held-out-matched 15-query
subset) climbed steadily on all three tracks:

![run4 auto-design progress](docs/run4_autodesign_progress.png)

*Green step = best-so-far `loop_objective`; ● keep, × revert. run4 discovered — with no prior
hints — an object-centric 3D semantic map (YOLO-World detect → depth back-project → multi-view
fuse → amortized qwen caption/embed), then self-improved it: spatial-relation re-ranking for T2,
synonym-expanded coverage for T1, and bbox-volumetric-center prediction (which matches the GT
scoring center) lifting T1 **0.72→0.82** and T2 **0.37→0.51**.*

---

## Environment setup

**➡️ Full, machine-agnostic instructions are in [docs/SETUP.md](docs/SETUP.md)** — it covers the
three layers (eval env, shared modules + LLM stack, and each baseline incl. the fiddly DAAAM env).
Quick version for the **benchmark evaluators + the auto-designed memory** (no baseline repos needed):

```bash
# 1. Evaluation env
conda env create -f environment.evaluation.yml      # env: spatial-memory-eval, python 3.10
conda activate spatial-memory-eval
pip install -e .
#   (environment.evaluation.yml pins two baseline repos by LOCAL PATH — only needed for the
#    ClawS/HOV-SG baselines; comment them out for the evaluators-only setup. See docs/SETUP.md.)

# 2. Local LLM stack (Ollama): agent-designed captioner + embeddings
ollama pull qwen3.5:4b            # 4.7B vision describer (VILA substitute)
ollama pull qwen3-embedding:0.6b  # 0.6B, 1024-d text embeddings

# 3. Shared perception modules on NAS (YOLO-World-L + ScanNet200 class list); see
#    .codex/modules.md + path_registry.md. NB: torch.backends.cudnn.enabled=False before ultralytics.

# 4. Cloud LLM (Bedrock): answering agent + Track 3 judge
export CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-west-2   # haiku agent + sonnet judge
```

Baselines (DAAAM / ClawS / ReMEmbR / HOV-SG) each need their own repo/env — **see
[docs/SETUP.md](docs/SETUP.md) Layer 3** (DAAAM in particular needs a separate conda env + a
colcon-built Hydra workspace + TensorRT FastSAM). Generated artifacts (`memories/`, `results/`,
`benchmarks/`, `data/`) are gitignored; use the NAS paths in `.codex/path_registry.md`.

---

## Running the baselines

Five hand-built / control methods are adapted onto the package contract: **DAAAM** (Hydra 3D scene
graph), **ClawS** (object-map + sqlite-vec), **ReMEmbR** (caption memory), plus **LLM-with-captions**
and **Multi-frame-VLM** controls.

**Build the benchmark data** (per dataset; done once):

```bash
python scripts/build_track1_data.py --scene-id <scene> --dataset scannet
python scripts/build_track2_data.py --scene-id <scene>
python scripts/build_track3_data.py --dataset scannet
```

**Evaluate one package** on one track/scene:

```bash
python scripts/evaluate_track1.py <package_dir> --dataset scannet --scene-id <scene> \
    --mode tool_llm --llm-command "<agent CLI template>" --output <out.json>
```

**Run everything** — all methods × 10 held-out scenes × 3 tracks (the driver wires up the shared
detector, the Haiku agent, and the Sonnet judge):

```bash
# scripts/methods/eval_all_scannet.sh <track|all> <fixed_api|tool_llm> <methods_csv>
bash scripts/methods/eval_all_scannet.sh all tool_llm daaam,claws,remembr,remembr_captions,multiframe_vlm
```

### Held-out results (10 ScanNet scenes, `tool_llm`)

| Method | T1 success@1 | T2 acc@0.5m | T3 LLM-Match | real-time build |
|---|---|---|---|---|
| **agent-designed** (run2, frozen) | **0.774** | **0.360** | 0.502 | ~0.9 s/f |
| DAAAM (scene_graph) | 0.386 | 0.330 | 0.367 | 0.012 s/f |
| ClawS (object_map) | 0.290 | 0.351 | 0.340 | 0.095 s/f |
| ReMEmbR (caption) | 0.045 | 0.000 | 0.498 | n/a (sparse) |

The agent-designed memory is best-or-tied-best on every track. Full analysis, the run3 deep-dive
(where a *more aggressive* design under-performed the plain object map — a real negative result),
and the baseline fairness audit are in `.codex/agent_designed_run3_analysis.md` and
`.codex/scannet_10scene_results.md`.

---

## Running the auto-design loop yourself

```bash
# 1. Prepare dev scenes (download GT + extract RGB-D layout + build track1/2/3 dev tests)
bash scripts/agent_designed/prepare_dev_scene.sh <scene_id>

# 2. Create a fresh, self-contained sandbox (blank starter/, dev scenes, docs, harness)
python scripts/agent_designed/make_sandbox.py --variant loop_fixed_tests \
    --sandbox-root ~/my_autodesign_run \
    --dev-scene-id <scene_a> --dev-scene-id <scene_b> ...

# 3. Drop in the task prompt + init git, then let the coding agent run the loop.
#    Each round the agent edits starter/ and calls:
python autoresearch_round.py --build-cmd "python starter/build_memory.py" \
    --message "round N: <what changed and why>"
#    -> builds all dev scenes, scores on the fixed dev tests, and keeps or reverts automatically.
```

For a long unattended run, `scripts/agent_designed/run4_supervisor.sh <sandbox> <wall_seconds>`
keeps a designer agent working to a hard time budget, relaunching it if it exits early (the sandbox
persists, so a relaunch reads its own DESIGN_NOTES / history / git log and continues).

The scored objective:

```
loop_objective = success@1[T1] + acc@0.5m[T2] + llm_match[T3] − cost_penalty
```

`cost_penalty` keeps the build near real-time (≤0.2 s/frame) and the memory compact (≤50 MB/scene).
End-to-end query latency is measured and reported (but not scored) so the loop can't cheat by
deferring heavy compute to query time.

---

## The minimal memory package

Every method exports the same shape, consumed by the same evaluators:

```text
manifest.json        # method + dataset metadata
capabilities.json    # which fixed APIs / agent tools are supported (else "invalid" + reason)
memory/              # the method's NATIVE memory (object table, DB, captions, DSG, ...)
tools/               # package-local Python entrypoints (query_object, resolve_referring, ...)
schema.md  schemas/  evidence/  raw_links/  build_log.json
```

Validate a package:

```bash
python -m spatial_memory_evaluation.memory_package_validator <package_dir>
```

The full contract is in [.codex/memory_package_spec.md](.codex/memory_package_spec.md).

---

## Repository layout

- `spatial_memory_evaluation/track{1,2,3,4}/` — per-track benchmark builders + evaluators.
- `spatial_memory_evaluation/tool_llm/` — per-query LLM + native-tool runner (the fair protocol).
- `spatial_memory_evaluation/agent_designed/` — auto-design harness: dev-eval scorer, per-scene
  session eval, contract/workspace.
- `spatial_memory_evaluation/common/`, `schemas/`, `assets/` — shared IO, schemas, class lists.
- `scripts/build_track*_data.py`, `scripts/evaluate_track*.py` — data + eval CLIs.
- `scripts/methods/` — per-baseline adapters + `eval_all_scannet.sh`; `scripts/methods/coda/` for Track 4.
- `scripts/agent_designed/` — sandbox maker, auto-research round controller, supervisor, scorer.
- `.codex/` — design notes, specs, registries, results, and analyses (start at `.codex/README.md`).
- `examples/` — small valid package fixtures.

## Documentation

Start with [.codex/README.md](.codex/README.md). Key docs: `agentic_eval.md` (vision),
`memory_package_spec.md` (contract), `baseline_registry.md` (methods), `path_registry.md` (paths),
`modules.md` (shared modules), `agent_designed_baseline.md` (auto-design), and the results/analysis
docs listed in the `.codex` index.
