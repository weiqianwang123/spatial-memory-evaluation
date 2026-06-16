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
