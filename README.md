# Spatial Memory Evaluation

This folder is an independent harness for evaluating a spatial-memory method on
ScanNet RGB-D episode histories with two outputs:

- `get_memory_text(question)`: text used as the OpenEQA answer and scored with
  the internal OpenEQA-compatible LLM-Match scorer.
- `get_object(query)`: object predictions exported to JSON and optionally passed
  to your existing ScanNet object evaluator.

The harness does not import or change the upstream `openeqa` package. Run
commands from the repository root and add this folder to `PYTHONPATH`.

```bash
export PYTHONPATH=spatial-memory-evaluation:$PYTHONPATH
```

## Shared Conda Environment

Use one evaluation environment for Spatial Memory Evaluation and ClawS SpatialRAG. The
fastest path on this machine is to extend the existing `spatial-rag` env:

```bash
source /home/robin_wang/miniforge3/etc/profile.d/conda.sh
conda activate spatial-rag
pip install -r spatial-memory-evaluation/requirements.evaluation.txt
export PYTHONPATH=/home/robin_wang/open-eqa/spatial-memory-evaluation:/home/robin_wang/ClawS-SpatialRAG:$PYTHONPATH
```

Or create a clean combined env:

```bash
conda env create -f spatial-memory-evaluation/environment.evaluation.yml
conda activate spatial-memory-eval
```

`spatial-rag` already has the heavy SpatialRAG pieces here (`cv2`,
`sqlite_vec`, `ultralytics`, `torch`, `transformers`). It needs `openai` and
`tenacity` for the internal LLM-Match scorer.

## ClawS SpatialRAG Adapter

The built-in adapter is:

```bash
--method adapters.claws_spatial_rag:create_method
```

For the current ScanNet++ scene used by the ClawS repo, use:

```bash
--method-kwargs spatial-memory-evaluation/configs/claws_current_scene_method_kwargs.json
```

This points at:

```text
/home/robin_wang/ClawS-SpatialRAG/outputs/scannetpp_memory_036bce3393_ollama_vlm.db
```

The adapter exposes:

- `get_memory_text(question)`: retrieves relevant SpatialRAG memory snippets.
- `get_object(query)`: returns matching objects from the current scene memory DB.

For a full OpenEQA ScanNet run, set `build_from_sequence: true` in the kwargs so
the adapter builds a SpatialRAG DB from each OpenEQA RGB-D sequence before
answering. For the current scene smoke run, it reuses the existing DB.

## Current Scene Smoke Run

This runs `get_object`, writes a few memory-context examples, and then runs the
same existing ScanNet++ metric script on scene `036bce3393`.

```bash
PYTHONPATH=spatial-memory-evaluation:/home/robin_wang/ClawS-SpatialRAG:$PYTHONPATH \
python3 spatial-memory-evaluation/scripts/run_claws_current_scene.py
```

Outputs:

```text
spatial-memory-evaluation/results/claws-current-scene-objects.json
spatial-memory-evaluation/results/claws-current-scene-memory.json
spatial-memory-evaluation/results/claws-current-scene-object-metrics.json
spatial-memory-evaluation/results/claws-current-scene-object-metrics.md
```

## DualMap Current Scene Baseline

The base DualMap adapter only supports `get_object(query)`. For OpenEQA memory
answering, use the LLM-with-memory wrapper below; it exports the DualMap object
map as scene-graph-like memory before answering.

On this machine, run DualMap through the shared `spatial-rag` environment with
user-site packages disabled. The default Python path otherwise picks up
`~/.local`'s CUDA 13 PyTorch, which is incompatible with the installed NVIDIA
driver.

```bash
source /home/robin_wang/miniforge3/etc/profile.d/conda.sh
conda activate spatial-rag

PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/robin_wang/open-eqa/spatial-memory-evaluation:/home/robin_wang/DualMap:/home/robin_wang/ClawS-SpatialRAG:$PYTHONPATH \
python spatial-memory-evaluation/scripts/build_dualmap_current_scene_map.py \
  --run-stride 1

PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/robin_wang/open-eqa/spatial-memory-evaluation:/home/robin_wang/DualMap:/home/robin_wang/ClawS-SpatialRAG:$PYTHONPATH \
python spatial-memory-evaluation/scripts/run_dualmap_current_scene.py
```

Outputs:

```text
/data/mondo-training-dataset/semantic_mapping/dualmap/scannetpp_036bce3393/map/*.pkl
spatial-memory-evaluation/results/dualmap-current-scene-objects.json
spatial-memory-evaluation/results/dualmap-current-scene-memory.db
spatial-memory-evaluation/results/dualmap-current-scene-object-metrics.json
spatial-memory-evaluation/results/dualmap-current-scene-object-metrics.md
```

For full scene recall over every exported DualMap object:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/robin_wang/open-eqa/spatial-memory-evaluation:/home/robin_wang/DualMap:/home/robin_wang/ClawS-SpatialRAG:$PYTHONPATH \
python spatial-memory-evaluation/scripts/evaluate_dualmap_current_scene_recall.py
```

## HOV-SG Current Scene Baseline

The base HOV-SG adapter only supports `get_object(query)`. For OpenEQA memory
answering, use the LLM-with-memory wrapper below; it exports the HOV-SG object
map as scene-graph-like memory before answering.

The adapter expects a precomputed HOV-SG feature map:

```text
<hovsg-result-path>/
  mask_feats.pt
  objects/
    pcd_0.ply
    pcd_1.ply
    ...
```

The HOV-SG repo is expected at:

```text
/home/robin_wang/HOV-SG
```

Evaluate the current ScanNet++ scene once the HOV-SG feature map exists:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/robin_wang/open-eqa/spatial-memory-evaluation:/home/robin_wang/HOV-SG:/home/robin_wang/ClawS-SpatialRAG:$PYTHONPATH \
python spatial-memory-evaluation/scripts/evaluate_hovsg_current_scene_recall.py \
  --hovsg-result-path /data/mondo-training-dataset/semantic_mapping/hovsg/scannetpp_036bce3393/scannet
```

To build that feature map from the already exported ScanNet++ RGB-D sequence:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/robin_wang/open-eqa/spatial-memory-evaluation:/home/robin_wang/HOV-SG:$PYTHONPATH \
python spatial-memory-evaluation/scripts/build_hovsg_current_scene_map.py \
  --skip-frames 1
```

For a fast environment smoke test, use `--skip-frames 2000 --merge-type sequential`.

Visualize the saved HOV-SG map:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/robin_wang/open-eqa/spatial-memory-evaluation:$PYTHONPATH \
python spatial-memory-evaluation/scripts/visualize_hovsg_map.py \
  --mode masked
```

Useful modes:

```bash
# Raw RGB point cloud
python spatial-memory-evaluation/scripts/visualize_hovsg_map.py --mode full

# Random-colored object masks
python spatial-memory-evaluation/scripts/visualize_hovsg_map.py --mode masked

# Highlight get_object results
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/robin_wang/open-eqa/spatial-memory-evaluation:$PYTHONPATH \
python spatial-memory-evaluation/scripts/visualize_hovsg_map.py \
  --mode query \
  --query chair \
  --show-full-context
```

Outputs:

```text
spatial-memory-evaluation/results/hovsg-current-scene-objects.json
spatial-memory-evaluation/results/hovsg-full-recall-memory.db
spatial-memory-evaluation/results/hovsg-full-recall.json
spatial-memory-evaluation/results/hovsg-full-recall.md
```

The default HOV-SG CLIP settings in this machine-local config are `ViT-B-32`
with `laion2b_s34b_b79k`, because those weights are already cached here. If the
map was generated with a different CLIP backbone, update
`spatial-memory-evaluation/configs/hovsg_current_scene_method_kwargs.json`
before evaluating.

## OpenEQA ScanNet Run

OpenEQA `scannet-v0` requires the original ScanNet `.sens` files. RGB videos or
viewer MP4s are not enough for SpatialRAG because depth and camera poses are
only recovered from `.sens`.

Check the local data state:

```bash
PYTHONPATH=spatial-memory-evaluation:/home/robin_wang/ClawS-SpatialRAG:$PYTHONPATH \
python spatial-memory-evaluation/scripts/check_openeqa_scannet_data.py \
  --scannet-root data/raw/scannet \
  --frames-root data/frames
```

The pasted official ScanNet downloader is Python 2, so this harness includes a
Python 3 downloader for the same `.sens` URLs. It downloads only the ScanNet
scenes referenced by OpenEQA `scannet-v0`:

```bash
python spatial-memory-evaluation/scripts/download_openeqa_scannet_sens.py \
  --out-dir data/raw/scannet \
  --agree-tos
```

On this machine, the SpatialRAG dataset NAS is mounted at:

```text
/data/mondo-training-dataset
```

To keep ScanNet off local disk, put the raw `.sens` and converted RGB-D frames
there instead:

```bash
python spatial-memory-evaluation/scripts/download_openeqa_scannet_sens.py \
  --out-dir /data/mondo-training-dataset/semantic_mapping/scannet \
  --agree-tos

python spatial-memory-evaluation/scripts/prepare_scannet_sens_rgbd.py \
  --scannet-root /data/mondo-training-dataset/semantic_mapping/scannet \
  --frames-root /data/mondo-training-dataset/semantic_mapping/openeqa_frames \
  --layout-root /data/mondo-training-dataset/semantic_mapping/openeqa_scannet_rgbd
```

For a non-downloading preview:

```bash
python spatial-memory-evaluation/scripts/download_openeqa_scannet_sens.py \
  --out-dir /data/mondo-training-dataset/semantic_mapping/scannet \
  --dry-run
```

If raw ScanNet exists, convert each available `.sens` into both useful views:

- SpatialRAG/OpenEQA flat frames under `<frames-root>/scannet-v0/<episode>/`.
- HOV-SG/DualMap ScanNet-style frames under
  `<layout-root>/exported/<scene_id>/`.

```bash
python spatial-memory-evaluation/scripts/prepare_scannet_sens_rgbd.py \
  --scannet-root data/raw/scannet \
  --frames-root data/frames \
  --layout-root data/openeqa_scannet_rgbd
```

By default this exports every frame in each `.sens` so SpatialRAG, HOV-SG, and
DualMap see the same complete RGB-D stream. Pass `--max-frames N` only for an
explicit debugging run.

Then run ClawS SpatialRAG on the OpenEQA ScanNet split:

```bash
PYTHONPATH=spatial-memory-evaluation:/home/robin_wang/ClawS-SpatialRAG:$PYTHONPATH \
python spatial-memory-evaluation/scripts/run_openeqa_scannet_memory.py \
  --frames-root data/frames \
  --output spatial-memory-evaluation/results/openeqa-scannet-memory.json
```

With NAS-backed frames and per-episode SpatialRAG DBs:

```bash
PYTHONPATH=spatial-memory-evaluation:/home/robin_wang/ClawS-SpatialRAG:$PYTHONPATH \
python spatial-memory-evaluation/scripts/run_openeqa_scannet_memory.py \
  --scannet-root /data/mondo-training-dataset/semantic_mapping/scannet \
  --frames-root /data/mondo-training-dataset/semantic_mapping/openeqa_frames \
  --method-kwargs spatial-memory-evaluation/configs/claws_openeqa_scannet_nas_method_kwargs.json \
  --output spatial-memory-evaluation/results/openeqa-scannet-memory.json
```

For a quick first episode smoke test:

```bash
PYTHONPATH=spatial-memory-evaluation:/home/robin_wang/ClawS-SpatialRAG:$PYTHONPATH \
python spatial-memory-evaluation/scripts/run_openeqa_scannet_memory.py \
  --dry-run
```

The per-episode ClawS DBs are written under:

```text
spatial-memory-evaluation/results/openeqa-scannet-dbs/
```

## OpenEQA Scene0709 LLM-With-Memory Run

The prepared one-scene full-frame ScanNet RGB-D sequence is:

```text
/data/mondo-training-dataset/semantic_mapping/openeqa_frames/scannet-v0/002-scannet-scene0709_00
```

The matching ScanNet-style layout for HOV-SG and DualMap is:

```text
/data/mondo-training-dataset/semantic_mapping/openeqa_scannet_rgbd/exported/scene0709_00
```

This scene has 936 RGB/depth/pose frames. Formal runs should not pass
`--max-frames`, `--frame-stride`, `--run-stride`, or `--skip-frames` values that
drop frames.

Build or load method memory first:

```bash
# ClawS SpatialRAG builds its scene DB lazily during the LLM run below.

# DualMap: build a concrete map from the full ScanNet-style RGB-D sequence.
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/robin_wang/open-eqa/spatial-memory-evaluation:/home/robin_wang/DualMap:/home/robin_wang/ClawS-SpatialRAG:$PYTHONPATH \
python spatial-memory-evaluation/scripts/build_dualmap_current_scene_map.py \
  --skip-export \
  --scene-id scene0709_00 \
  --dualmap-dataset-root /data/mondo-training-dataset/semantic_mapping/openeqa_scannet_rgbd \
  --dataset-root /data/mondo-training-dataset/semantic_mapping/openeqa_scannet_rgbd \
  --output-dir /data/mondo-training-dataset/semantic_mapping/dualmap/openeqa_scannet_scene0709_00 \
  --run-stride 1

# HOV-SG: build a feature map from the same full ScanNet-style RGB-D sequence.
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/robin_wang/open-eqa/spatial-memory-evaluation:/home/robin_wang/HOV-SG:$PYTHONPATH \
python spatial-memory-evaluation/scripts/build_hovsg_current_scene_map.py \
  --scene-id scene0709_00 \
  --dataset-path /data/mondo-training-dataset/semantic_mapping/openeqa_scannet_rgbd/exported/scene0709_00 \
  --output-root /data/mondo-training-dataset/semantic_mapping/hovsg/openeqa_scannet_scene0709_00 \
  --skip-frames 1
```

Run OpenEQA answer prediction with the LLM-with-memory wrapper. The wrapper
turns each method's memory into scene-graph-like text and gives that memory to
the LLM before answering:

```bash
export OPENAI_API_KEY=...

# ClawS SpatialRAG memory -> LLM answer
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/robin_wang/open-eqa/spatial-memory-evaluation:/home/robin_wang/ClawS-SpatialRAG:$PYTHONPATH \
python -m spatial_memory_evaluation.run_memory \
  --method adapters.llm_with_memory:create_method \
  --method-kwargs spatial-memory-evaluation/configs/llm_openeqa_scene0709_claws_kwargs.json \
  --frames-root /data/mondo-training-dataset/semantic_mapping/openeqa_frames \
  --episode-history scannet-v0/002-scannet-scene0709_00 \
  --output spatial-memory-evaluation/results/openeqa-scene0709-claws-llm.json

# DualMap object memory -> LLM answer
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/robin_wang/open-eqa/spatial-memory-evaluation:/home/robin_wang/DualMap:$PYTHONPATH \
python -m spatial_memory_evaluation.run_memory \
  --method adapters.llm_with_memory:create_method \
  --method-kwargs spatial-memory-evaluation/configs/llm_openeqa_scene0709_dualmap_kwargs.json \
  --frames-root /data/mondo-training-dataset/semantic_mapping/openeqa_frames \
  --episode-history scannet-v0/002-scannet-scene0709_00 \
  --output spatial-memory-evaluation/results/openeqa-scene0709-dualmap-llm.json

# HOV-SG object memory -> LLM answer
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/robin_wang/open-eqa/spatial-memory-evaluation:/home/robin_wang/HOV-SG:$PYTHONPATH \
python -m spatial_memory_evaluation.run_memory \
  --method adapters.llm_with_memory:create_method \
  --method-kwargs spatial-memory-evaluation/configs/llm_openeqa_scene0709_hovsg_kwargs.json \
  --frames-root /data/mondo-training-dataset/semantic_mapping/openeqa_frames \
  --episode-history scannet-v0/002-scannet-scene0709_00 \
  --output spatial-memory-evaluation/results/openeqa-scene0709-hovsg-llm.json
```

Score the predictions with the internal OpenEQA-compatible LLM-Match evaluator:

```bash
PYTHONPATH=spatial-memory-evaluation:$PYTHONPATH \
python -m spatial_memory_evaluation.evaluate_memory \
  spatial-memory-evaluation/results/openeqa-scene0709-claws-llm.json \
  --dataset data/open-eqa-v0.json \
  --output spatial-memory-evaluation/results/openeqa-scene0709-claws-llm-match.json
```

For a no-API wiring check, set `answer_mode` to `context` in the relevant
`llm_openeqa_scene0709_*_kwargs.json`; the output will be the memory context
that would be passed to the LLM.

## Required Method Adapter

Wrap Spatial RAG with a small adapter. The adapter receives one `RGBDSequence`
per `episode_history` and must expose the two methods below.

```python
from spatial_memory_evaluation import ObjectPrediction, RGBDSequence


class MySpatialRAGAdapter:
    def __init__(self, sequence: RGBDSequence):
        self.rag = build_spatial_rag_from_rgbd(sequence)

    def get_memory_text(self, question: str) -> str:
        # Return the final answer text for OpenEQA.
        # If your system returns raw memory context, call your answerer here.
        return self.rag.answer(question)

    def get_object(self, query: str):
        objects = self.rag.retrieve_objects(query)
        return [
            ObjectPrediction(
                label=obj.label,
                score=obj.score,
                object_id=str(obj.id),
                bbox_3d=obj.bbox_3d,
                position_3d=obj.center,
            )
            for obj in objects
        ]


def create_method(sequence: RGBDSequence):
    return MySpatialRAGAdapter(sequence)
```

The loader accepts `module:attribute`. For example, if the adapter is
`my_spatial_rag/eval_adapter.py` with `create_method`, use:

```bash
--method my_spatial_rag.eval_adapter:create_method
```

Adapter factories may also accept `rgbd_sequence=...`, or build an object first
and then expose one of these setup hooks: `build_memory(sequence)`,
`load_rgbd_sequence(sequence)`, or `set_rgbd_sequence(sequence)`.

## RGB-D Input

OpenEQA ScanNet frames are expected under:

```text
data/frames/scannet-v0/<episode>/
  000000-rgb.png
  000000-depth.png
  000000.txt
  intrinsic_color.txt
  intrinsic_depth.txt
```

Create them with the existing OpenEQA extractor:

```bash
python3 data/scannet/extract-frames.py
```

Use `--rgb-only` only if your method does not require depth or poses.

## 1. Run OpenEQA Memory Predictions

This runs `get_memory_text(question)` for each ScanNet OpenEQA question and
writes a normal OpenEQA prediction file.

```bash
PYTHONPATH=spatial-memory-evaluation:$PYTHONPATH \
python3 -m spatial_memory_evaluation.run_memory \
  --method my_spatial_rag.eval_adapter:create_method \
  --dataset data/open-eqa-v0.json \
  --frames-root data/frames \
  --episode-prefix scannet-v0 \
  --output spatial-memory-evaluation/results/memory-predictions.json
```

Useful options:

- `--dry-run`: first 5 questions.
- `--max-frames 200`: debug-only cap; omit it for formal evaluation.
- `--frame-stride 5`: pass every fifth frame.
- `--method-kwargs '{"key": "value"}'`: pass JSON config into the factory.

## 2. Evaluate Memory With Internal LLM-Match

This uses `spatial_memory_evaluation.llm_match`, which mirrors OpenEQA's
LLM-Match prompt and 1-5 scoring but does not import the upstream `openeqa`
package. It requires `OPENAI_API_KEY` or `--openai-key`.

```bash
PYTHONPATH=spatial-memory-evaluation:$PYTHONPATH \
python3 -m spatial_memory_evaluation.evaluate_memory \
  spatial-memory-evaluation/results/memory-predictions.json \
  --dataset data/open-eqa-v0.json \
  --output spatial-memory-evaluation/results/memory-metrics.json
```

Optional scorer settings:

```bash
PYTHONPATH=spatial-memory-evaluation:$PYTHONPATH \
python3 -m spatial_memory_evaluation.evaluate_memory \
  spatial-memory-evaluation/results/memory-predictions.json \
  --dataset data/open-eqa-v0.json \
  --openai-model gpt-4-1106-preview \
  --openai-temperature 0.2 \
  --openai-seed 1234
```

## 3. Run Object Predictions

Prepare object queries as JSON:

```json
[
  {
    "query_id": "scene0709_00-chair",
    "episode_history": "scannet-v0/002-scannet-scene0709_00",
    "query": "chair"
  }
]
```

Then run:

```bash
PYTHONPATH=spatial-memory-evaluation:$PYTHONPATH \
python3 -m spatial_memory_evaluation.run_objects \
  --method my_spatial_rag.eval_adapter:create_method \
  --queries spatial-memory-evaluation/examples/object_queries.sample.json \
  --frames-root data/frames \
  --output spatial-memory-evaluation/results/object-predictions.json
```

The output format is:

```json
[
  {
    "query_id": "scene0709_00-chair",
    "episode_history": "scannet-v0/002-scannet-scene0709_00",
    "query": "chair",
    "objects": [
      {
        "label": "chair",
        "score": 0.91,
        "object_id": "12",
        "bbox_3d": [0, 0, 0, 1, 1, 1]
      }
    ]
  }
]
```

## 4. Call Your Existing ScanNet Object Evaluator

If your older ScanNet evaluator is a script, pass it as a command template.
The harness fills `{predictions}`, `{queries}`, and `{output}`.

```bash
PYTHONPATH=spatial-memory-evaluation:$PYTHONPATH \
python3 -m spatial_memory_evaluation.run_objects \
  --method my_spatial_rag.eval_adapter:create_method \
  --queries path/to/scannet_object_queries.json \
  --output spatial-memory-evaluation/results/object-predictions.json \
  --metrics-output spatial-memory-evaluation/results/object-metrics.json \
  --scannet-evaluator-cmd "python3 path/to/scannet_eval.py --predictions {predictions} --queries {queries} --output {output}"
```

Keep any evaluator-specific conversion in either your adapter's `get_object`
return dictionaries or in a small wrapper around the old evaluator.

## Smoke Test

The dummy adapter verifies the harness wiring once you have at least one
matching extracted ScanNet episode under `data/frames`.

```bash
PYTHONPATH=spatial-memory-evaluation:$PYTHONPATH \
python3 -m spatial_memory_evaluation.run_memory \
  --method examples.dummy_method:create_method \
  --episode-prefix scannet-v0 \
  --dry-run
```
