# Method Scripts

Place future method-specific scripts under one folder per method:

```text
scripts/methods/claws/
scripts/methods/dualmap/
scripts/methods/hovsg/
scripts/methods/conceptgraphs/
scripts/methods/daaam/
scripts/methods/hydra/
scripts/methods/remembr/
```

Exporter scripts should write validated packages to:

```text
memories/<method>/<dataset>/<scene-or-episode>/<run-id>/
```

## Shared Modules

External method repos should not be edited just to share detectors, SAM, CLIP, or
class lists. Method scripts must load shared module settings from:

```text
spatial_memory_evaluation/shared_modules/
```

and translate them into the external repo's native CLI/Hydra/config overrides
inside this evaluation repo.

Current adapter:

```text
scripts/methods/shared_modules.py
```

Current smoke profiles cover HOV-SG, DualMap, ConceptGraphs, and DAAAM. Inspect
them with:

```bash
python scripts/package/inspect_shared_modules.py --method hovsg --profile smoke --check
python scripts/package/inspect_shared_modules.py --method dualmap --profile smoke --check
python scripts/package/inspect_shared_modules.py --method daaam --profile smoke --check
```

Current HOV-SG entrypoints:

```bash
python scripts/methods/hovsg/prepare_eval_layout.py --scene-id 036bce3393 --run-id <run-id>
python scripts/methods/hovsg/build_memory_smoke.py --scene-id 036bce3393 --run-id <run-id> --layout-dir data/hovsg_layouts/scannetpp_036bce3393/<run-id>
python scripts/methods/hovsg/eval_memory_smoke.py memories/hovsg/scannetpp/036bce3393/<run-id>
```

`prepare_eval_layout.py` is method-specific data preparation. It converts
ScanNet++ iPhone RGB-D frames into the ScanNet-style layout expected by HOV-SG.
Memory build and eval scripts should consume that prepared layout instead of
re-exporting it implicitly.

Current DualMap smoke entrypoints:

```bash
python scripts/methods/dualmap/build_memory_smoke.py --scene-id 036bce3393 --run-id <run-id> --frame-stride 5 --prepare-only
python scripts/methods/dualmap/build_memory_smoke.py --scene-id 036bce3393 --run-id <run-id> --skip-layout-export --cuda-visible-devices 0
python scripts/methods/dualmap/eval_memory_smoke.py memories/dualmap/scannetpp/036bce3393/<run-id>
```

DualMap smoke prepare writes a ScanNet-style layout under
`data/dualmap_layouts/scannetpp_<scene-id>/<run-id>/exported/scannetpp_<scene-id>/`.
The build step calls DualMap `applications/runner_dataset.py`, packages native
`map/*.pkl`, and keeps fixed API eval separate from memory construction.
Formal runs should keep cuDNN enabled.

Current DAAAM smoke entrypoints:

```bash
python scripts/methods/daaam/build_memory_smoke.py \
  --scene-id 036bce3393 \
  --run-id <run-id> \
  --frame-stride 5 \
  --max-frames 200 \
  --prepare-only

python scripts/methods/daaam/build_memory_smoke.py \
  --scene-id 036bce3393 \
  --run-id <run-id> \
  --layout-dir data/daaam_layouts/scannetpp_036bce3393/<run-id> \
  --hydra-config-path <hydra-config.yaml> \
  --daaam-python <python-with-daaam-and-spark-dsg> \
  --cuda-visible-devices 0

python scripts/methods/daaam/build_memory_smoke.py \
  --scene-id 036bce3393 \
  --native-output-dir <existing-daaam-output-dir> \
  --daaam-python <python-with-daaam-and-spark-dsg>

python scripts/methods/daaam/eval_memory_smoke.py \
  memories/daaam/scannetpp/036bce3393/<run-id>
```

DAAAM smoke prepare writes an `ImageSequenceDataset` layout under
`data/daaam_layouts/scannetpp_<scene-id>/<run-id>/` with `rgb/`, `depth/`,
`pose/`, and `camera_info.json`. The build step either launches DAAAM's native
`scripts/run_pipeline.py` or packages an existing DAAAM output directory with
`dsg_updated.json`/`dsg.json`. Track 1 fixed API is exported from DSG objects
and background objects. Track 2 fixed API is declared `supported` only when the
builder can export a deterministic DAAAM semantic index from native
scene-understanding embeddings; otherwise the package declares Track 2
`invalid`.

DAAAM dependency policy:

- Environment dependencies belong in the conda/runtime env passed through
  `--daaam-python`. This includes `spark_dsg`, `daaam`, `torch`, `open_clip`,
  `sentence_transformers`, `ultralytics`, and `segment_anything`.
- Model artifacts/checkpoints belong in shared modules/NAS, not inside the
  DAAAM repo. This includes SAM checkpoints, YOLO-World checkpoints, FastSAM
  weights/engines, BotSort/ReID weights/engines, and optional HF/OpenCLIP model
  caches.
- If preflight reports missing `spark_dsg` or `daaam`, fix the conda/env first.
  If it reports a missing checkpoint path, place or symlink that artifact under
  `/data/mondo-training-dataset/semantic_mapping/modules/`.

Current ReMEmbR caption-control entrypoint:

```bash
# Build from a native ReMEmbR caption JSON (read-only input):
python scripts/methods/remembr/build_caption_control_package.py \
  --captions-json /home/robin_wang/remembr/data/captions/<seq>/captions/captions_<captioner>_<secs>_secs.json \
  --dataset oc-navqa --episode-id <seq> --run-id remembr-captions-<date>

# Rebuild the committed offline example fixture (synthetic captions, deterministic):
python scripts/methods/remembr/build_caption_control_package.py \
  --synthetic --output-dir examples/caption_control_package \
  --dataset example --episode-id seq0 \
  --started-at 2026-06-17T00:00:00+08:00 --finished-at 2026-06-17T00:00:00+08:00
```

This builds the **LLM-with-captions Track 1/2 control** package. Caption-only
memory is packaged honestly as `memory/captions.jsonl` (plus the verbatim native
caption JSON under `memory/native/` when built from a real source), but the
package sets `manifest.method.family = "caption_control"` and
`manifest.explicit_memory = false` and declares **all four fixed-API tracks
`invalid`**. Each `invalid` reason points at the native capability gap in the
ReMEmbR root repo (no object inventory for Track 1; no deterministic native
object-location query for Track 2; an LLM caption answerer is never fixed-API
support). See `.codex/baseline_registry.md` →
`## LLM-With-Captions Track 1/2 Control Outcome (Task 21)`.
