# Spatial Memory Evaluation

Standalone framework for evaluating spatial-memory methods as an external
resource for embodied agents.

The design is package-first: every method exports a minimal memory package, then
evaluators consume that package through either declared fixed Python APIs or a
per-query LLM + native-tool agentic loop. The method's native memory format is
preserved inside the package; unsupported fixed APIs are reported as `invalid`
instead of guessed or silently approximated.

The benchmark has **three tracks** (refactor 2026-06-23) plus an agent-designed
memory baseline:

| Track | Capability key | Tests | Dataset |
|---|---|---|---|
| Track 1 | `track1_object_location` | object-level location query + build cost | ScanNet++ |
| Track 2 | `track2_scanrefer` | instance-level referring query | ScanRefer / ScanNet |
| Track 3 | `track3_openeqa` | general spatial QA | OpenEQA (ScanNet + HM3D) |

The **agent-designed memory baseline** (the project's centerpiece) lets a coding
agent design its own memory + tools under a fixed contract, then scores it with
the same Track 1/2/3 evaluators. OC-NaVQA and SG3D are deferred zero-shot transfer
targets, not main-line tracks. See `.codex/agent_designed_baseline.md`.

## Current Workflow

```text
method repo / native outputs
  -> exporter
  -> memories/<method>/<dataset>/<scene-or-episode>/<run-id>/
  -> package validator
  -> fixed API eval or tool-LLM agentic eval (Track 1/2/3)
  -> results/<method>/<evaluation>/<timestamp>/
```

The package contract is defined in
[.codex/memory_package_spec.md](.codex/memory_package_spec.md).

## Project Layout

- `.codex/`: planning notes, baseline registry, package spec, agent-designed
  baseline design, and agent context.
- `spatial_memory_evaluation/common/`: shared package loading, label
  normalization, JSONL, object matching, build accounting, reporting.
- `spatial_memory_evaluation/track1/`: object-level location query + build-cost
  benchmark builder and evaluator.
- `spatial_memory_evaluation/track2/`: ScanRefer referring-query evaluator
  (skeleton; emits `data_unavailable` until ScanRefer is acquired).
- `spatial_memory_evaluation/track3/`: OpenEQA general-QA evaluator (ScanNet +
  HM3D; LLM-Match judge pluggable).
- `spatial_memory_evaluation/agent_designed/`: agent-designed memory baseline
  harness skeleton (contract, workspace, designer stub, leakage stub, harness).
- `spatial_memory_evaluation/tool_llm/`: per-query LLM + native-tool runner.
- `spatial_memory_evaluation/shared_modules/`: detector/segmenter/CLIP registry.
- `spatial_memory_evaluation/schemas/memory_package/`: JSON schemas for the
  minimal memory package.
- `examples/minimal_memory_package/`: small valid package fixture
  (`track1_object_location` supported via `query_object`).
- `examples/multiframe_vlm_control/`, `examples/caption_control_package/`:
  no-explicit-memory control fixtures (`explicit_memory=false`, all fixed APIs
  `invalid`).
- `benchmarks/`, `memories/`, `results/`, `data/`, `sandboxes/`: generated;
  ignored by Git.
- `scripts/`: build-data, evaluate, package, method, and agent-designed CLIs.

## Minimal Package

A valid package contains:

```text
manifest.json
capabilities.json
schema.md
memory/
evidence/
raw_links/
schemas/
tools/
build_log.json
```

The fixed API capabilities are declared in `capabilities.json`:

- `track1_object_location`
- `track2_scanrefer`
- `track3_openeqa`

Each track is either `supported` with a package-local Python entrypoint, or
`invalid` with a reason. A valid package does not need to support every track.

## Validate A Package

```bash
python scripts/package/validate_memory_package.py examples/minimal_memory_package
# or
python -m spatial_memory_evaluation.memory_package_validator examples/minimal_memory_package
```

## Build Benchmarks And Evaluate

```bash
# Track 1: object-level location query + build cost (ScanNet++).
python scripts/build_track1_data.py --scene-id 036bce3393
python scripts/evaluate_track1.py memories/<method>/scannetpp/036bce3393/<run-id> --mode fixed_api

# Track 2: ScanRefer referring query (data acquisition pending).
python scripts/build_track2_data.py --scannet-split val
python scripts/evaluate_track2.py memories/<method>/.../<run-id> --mode fixed_api

# Track 3: OpenEQA general QA (ScanNet now; HM3D pending).
python scripts/build_track3_data.py --dataset scannet
python scripts/evaluate_track3.py memories/<method>/.../<run-id> --dataset scannet --mode fixed_api
```

Each evaluator supports `--mode fixed_api` and `--mode tool_llm` (per-query LLM +
method-native tools, for methods that declare native retrieval tools). If a
package honestly declares an unsupported fixed API, the result is `invalid` for
that method/track rather than coerced into an approximate score. Track 2/3 emit a
`data_unavailable` result until their datasets are built (see
`.codex/path_registry.md`).

No-explicit-memory controls (Multi-frame VLM, LLM-with-captions;
`explicit_memory=false`) are control-only for fixed API: their result is a
distinct `invalid` with `reason_code=control_no_explicit_memory` and
`control=true`.

Track 1 fair-comparison runs should use the shared detector vocabulary in
`spatial_memory_evaluation/assets/class_lists/detector_coverable.txt`. Any
detector/segmenter/CLIP/checkpoint/class-list override is a module ablation
unless all compared methods use the same setting.

## Agent-Designed Memory Baseline

```bash
python scripts/agent_designed/run_baseline.py --variant coding_agent --dataset scannetpp \
  --train-scene-id 036bce3393 --heldout-scene-id <held-out-scene>
```

This assembles a designer workspace, (Phase 4) invokes a coding agent to design a
memory, scans for leakage, then builds and scores the designed package on Track
1/2/3. The designer invocation and per-scene build are Phase-4 stubs; see
`.codex/agent_designed_baseline.md`.

## Next Work

1. Acquire ScanRefer + HM3D data; turn Track 2/3 skeletons into real evals.
2. Freeze method packages (ClawS, ConceptGraphs, DualMap, HOV-SG, DAAAM) on the
   3-track contract; run Track 1 fixed_api + tool_llm.
3. Implement the agent-designed designer invocation + leakage scanner (Phase 4).
4. Zero-shot transfer to SG3D / OC-NaVQA (Phase 5).
