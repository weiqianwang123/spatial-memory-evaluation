# Spatial Memory Evaluation

Standalone framework for evaluating spatial-memory methods.

The current design is package-first: every method exports a minimal memory
package, then evaluators consume that package through either declared fixed
Python APIs or an agentic sandbox. The method's native memory format is
preserved inside the package; unsupported fixed APIs are reported as `invalid`
instead of being guessed or silently approximated.

## Current Workflow

```text
method repo / native outputs
  -> exporter
  -> memories/<method>/<dataset>/<scene-or-episode>/<run-id>/
  -> package validator
  -> fixed API eval or agentic eval
  -> results/<method>/<evaluation>/<timestamp>/
```

The package contract is defined in
[.codex/memory_package_spec.md](.codex/memory_package_spec.md).

## Project Layout

- `.codex/`: planning notes, baseline registry, package spec, and agent context.
- `spatial_memory_evaluation/common/`: shared package loading, label normalization,
  JSONL, and object matching helpers.
- `spatial_memory_evaluation/track1/`: object-inventory benchmark data builder
  and evaluator.
- `spatial_memory_evaluation/track2/`: object-location query builder and
  evaluator.
- `spatial_memory_evaluation/schemas/memory_package/`: JSON schemas for the
  minimal memory package.
- `examples/minimal_memory_package/`: small valid package fixture for smoke
  tests.
- `benchmarks/`: generated Track 1 and Track 2 benchmark files.
- `scripts/package/`: package-level utilities.
- `scripts/methods/`: future method-specific scripts, grouped by method.
- `memories/`: generated memory packages; ignored by Git.
- `results/`: evaluation outputs; ignored by Git.
- `data/`: optional local data cache; ignored by Git.

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

- `track1_memory_construction`
- `track2_object_location`
- `track3_scanrefer`
- `track4_openeqa`

Each track is either `supported` with a package-local Python entrypoint, or
`invalid` with a reason. A valid package does not need to support every track.

## Validate A Package

Run from the repository root:

```bash
python scripts/package/validate_memory_package.py examples/minimal_memory_package
```

Equivalent module entrypoint, if the package is installed or `PYTHONPATH` points
to the repository:

```bash
python -m spatial_memory_evaluation.memory_package_validator examples/minimal_memory_package
```

The validator checks package layout, manifest/capability honesty, artifact
paths, declared entrypoints, `schema.md`, and leakage-related flags. It does not
force methods to implement unsupported tracks.

## Track 1 And Track 2

Track 1 and Track 2 are intentionally separate. Build their data independently
and evaluate them with separate entrypoints:

```bash
python scripts/build_track1_data.py --scene-id 036bce3393
python scripts/evaluate_track1.py memories/<method>/scannetpp/036bce3393/<run-id>

python scripts/build_track2_queries.py --scene-id 036bce3393
python scripts/evaluate_track2.py memories/<method>/scannetpp/036bce3393/<run-id>
```

Track 1 evaluates object-memory construction: object inventory precision/recall,
duplicate objects, false-memory ratio, center error, memory package size, native
linked size when available, and build runtime when declared.

Track 2 evaluates category-level object-location queries against the exported
memory: recall/success at 1/5/10, MRR, first-hit distance, query latency, total
query runtime, and QPS.

Both tracks support `--mode fixed_api` and `--mode agentic_memory_only`. If a
package honestly declares an unsupported fixed API, the result is `invalid` for
that method/track rather than coerced into an approximate score.

Formal fair-comparison runs should use the shared detector vocabulary in
`spatial_memory_evaluation/assets/class_lists/detector_coverable.txt`, which is
checked against `spatial_memory_evaluation/common/labels.py`. Any detector,
segmenter, CLIP model, checkpoint, or class-list override should be treated as a
module ablation unless all compared methods use the same setting.

## Next Work

1. Freeze more method packages with the minimal memory package spec.
2. Run Track 1/2 on HOV-SG and DualMap smoke packages, then broaden to Hydra and
   ReMEmbR.
3. Add Track 3 ScanRefer package evaluation after the Track 1/2 API is stable.
4. Add OpenEQA fixed API and agentic evaluation after ScanRefer.
