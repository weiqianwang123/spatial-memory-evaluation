# Project Context

This repository is a standalone spatial-memory evaluation harness at:

```text
/home/robin_wang/spatial-memory-evaluation
```

Work from the repository root. Do not use the old nested path under
`/home/robin_wang/open-eqa`.

For the agentic spatial-memory benchmark direction, read:

1. `agentic_eval.md`
2. `agentic_eval_plan.md`
3. `memory_package_spec.md`
4. `agent_designed_baseline.md`
5. `baseline_registry.md`
6. `path_registry.md`
7. `modules.md`

The benchmark has three tracks: `track1_object_location` (object-level location
query + build cost), `track2_scanrefer` (instance-level referring), and
`track3_openeqa` (general spatial QA on ScanNet + HM3D). The agent-designed
memory baseline is the project's centerpiece. OC-NaVQA and SG3D are deferred
zero-shot transfer targets, not main-line tracks.

## Core Workflow

1. A method-native repo or exporter builds the method's minimal spatial memory.
2. The exporter writes a memory package under
   `memories/<method>/<dataset>/<scene>/<run-id>/`.
3. The package includes `manifest.json`, `capabilities.json`, `schema.md`, and
   method-native artifacts.
4. The package validator checks that declared capabilities are honest and that
   unsupported tracks are marked invalid instead of being forced into a shared API.
5. Fixed-API evaluation and agentic full-access evaluation both start from the
   exported package.
6. Evaluation outputs go under `results/<method>/<evaluation>/<timestamp>/`.

Two evaluation modes per package (see `method_runtime_runbook.md` > Tool-LLM Eval):
- `--mode fixed_api`: the package's declared deterministic entrypoint (instant, no
  LLM). Used where a method has a native non-interactive query API (ClawS Track 1,
  DAAAM Track 1 via `query_object.py`).
- `--mode tool_llm`: per-query LLM (local Claude CLI, Bedrock) calling the method's
  native retrieval tools. This is the runnable adaptation for object/scene-graph/
  caption methods across all 3 tracks (one scene per track). Methods done this way
  (2026-06-24): DAAAM, ClawS, ReMEmbR, + two no-explicit-memory controls. Expose
  ALL native tools the package can back and let the agent choose. Model presets
  (haiku/sonnet/opus) in `scripts/methods/llm_presets.sh`.

The current data source is the NAS mount:

```text
/data/mondo-training-dataset
```

Local `data/` is only a cache or fallback and is ignored by Git.
Concrete data, checkpoint, repo, runtime, intermediate, memory, and result paths
are tracked in `path_registry.md`.

## Memory Package Contract

The package contract is defined in `memory_package_spec.md`. Keep package
metadata explicit enough that another agent can understand:

- where the method-native memory lives,
- what tracks the memory can support through fixed APIs,
- which tracks are invalid for this memory form,
- which shared modules/checkpoints were used to build it,
- how to reproduce or inspect the memory.

Track support is declared in `capabilities.json` under the three keys
`track1_object_location`, `track2_scanrefer`, `track3_openeqa`; a method is
allowed to declare `invalid` for a track when its memory form genuinely cannot
answer that API.

## Shared Modules

Use `modules.md` whenever a method uses reusable components such as SAM, YOLO,
OpenCLIP, GroundingDINO, VLMs, embeddings, vector stores, or LLM judges. If two
methods use the same functional module, formal runs should use the same strongest
common version and checkpoint. Any method-native exception must be recorded as a
module override in package metadata.

## Path Rules

- Use `PYTHONPATH=.:...` from this repo root.
- Put generated memory under `memories/`.
- Put evaluation predictions and metrics under `results/<method>/<evaluation>/<timestamp>/`.
- Use `_data` as the method folder only for data-prep or data-check outputs that do not belong to a method.
- Keep heavy raw data on NAS when possible.
- Do not commit generated DBs, maps, model checkpoints, frames, logs, or metric outputs.

Result examples:

```text
results/claws/memory-qa/20260612-153012/predictions.json
results/dualmap/object-recall/20260612-153245/metrics.json
results/_data/data-check/20260612-153500/report.json
```
