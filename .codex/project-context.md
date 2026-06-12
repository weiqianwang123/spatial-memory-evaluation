# Project Context

This repository is a standalone spatial-memory evaluation harness at:

```text
/home/robin_wang/spatial-memory-evaluation
```

Work from the repository root. Do not use the old nested path under
`/home/robin_wang/open-eqa`.

## Core Workflow

1. An adapter in `adapters/` reads RGB-D episode data or prebuilt method output.
2. The adapter builds or loads method-specific spatial memory.
3. Generated memory artifacts are written under `memories/`.
4. Evaluation scripts read the adapter or exported memory and write predictions,
   metrics, reports, and logs under `results/<method>/<evaluation>/<timestamp>/`.

The current data source is the NAS mount:

```text
/data/mondo-training-dataset
```

Local `data/` is only a cache or fallback and is ignored by Git.

## Adapter Contract

Each method should expose a factory such as:

```text
adapters.my_method:create_method
```

The loaded object must provide:

- `get_memory_text(question: str) -> str`
- `get_object(query: str) -> Sequence[ObjectPrediction | Mapping]`

If a method can export reusable structured memory, prefer adding:

- `export_spatial_memory_db(path: Path) -> Path`

Keep adapter-specific config in `configs/`. Store generated memory in a stable
method-first subdirectory such as:

```text
memories/<method>/<dataset-or-evaluation>/<scene-or-episode>/
```

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
