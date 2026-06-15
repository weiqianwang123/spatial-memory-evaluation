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
- `spatial_memory_evaluation/`: shared Python package and validator logic.
- `spatial_memory_evaluation/schemas/memory_package/`: JSON schemas for the
  minimal memory package.
- `examples/minimal_memory_package/`: small valid package fixture for smoke
  tests.
- `scripts/package/`: package-level utilities.
- `scripts/methods/`: future method-specific scripts, grouped by method.
- `memories/`: generated memory packages; ignored by Git.
- `results/`: evaluation outputs; ignored by Git.
- `data/`: optional local data cache; ignored by Git.

Some legacy adapters and configs may still exist during the transition, but the
new evaluation path should go through exported memory packages.

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
export PYTHONPATH=/home/robin_wang/spatial-memory-evaluation:$PYTHONPATH

python scripts/package/validate_memory_package.py examples/minimal_memory_package
```

Equivalent module entrypoint:

```bash
python -m spatial_memory_evaluation.memory_package_validator examples/minimal_memory_package
```

The validator checks package layout, manifest/capability honesty, artifact
paths, declared entrypoints, `schema.md`, and leakage-related flags. It does not
force methods to implement unsupported tracks.

## Next Work

1. Add exporter stubs under `scripts/methods/<method>/`.
2. Export packages for ClawS, DualMap, HOV-SG, ConceptGraphs, DAAAM, Hydra, and
   ReMEmbR.
3. Implement fixed API evaluators for Track 1 and Track 2 first.
4. Add agentic sandbox evaluation once packages validate reliably.
