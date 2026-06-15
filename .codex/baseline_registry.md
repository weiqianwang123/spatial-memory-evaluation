# Baseline Registry

Last checked: 2026-06-15

本文件只确认 baseline 方法本身的性质和原生接口。判断依据是
`/home/robin_wang` 下的方法 repo、repo 内的脚本/API、配置和输出 artifact；
evaluation repo 的 `adapters/` 只属于后续接入层，不能作为 baseline 支持某个
接口的证据。

## 判定规则

- `present`：root 路径下有本地代码，并且能定位到方法原生的构建或查询入口。
- `partial`：本地代码存在，但缺少稳定的非交互式评测接口或 memory package。
- `missing`：root 路径下没有对应方法 repo，或没有找到需要的原生能力。
- no-memory controls 单独列出；它们不是 spatial-memory 方法 repo。

## 方法总表

| Method | Root path | Local code | Native build / ingest interface | Native query / read interface | Main memory artifact | Perception / model stack | Status |
|---|---|---:|---|---|---|---|---|
| ClawS SpatialRAG | `/home/robin_wang/ClawS-SpatialRAG` | present | `spatial_rag.pipeline.SpatialPipeline.process_frame(rgb_bgr, depth_m, robot_pose, timestamp)`；ScanNet++/MCAP replay scripts | `SpatialRAGService.tools` exposes `query_spatial_memory`, `get_semantic_anchor`, `remember_spatial_observation`；`get_spatial_objects(limit)`；storage retrieval | SQLite / sqlite-vec spatial DB plus crop images | YOLO11/Ultralytics + ByteTrack, depth/pose projection, optional VLM description/verification, vector embedding, sqlite-vec fusion/search | native memory + query path present |
| DualMap | `/home/robin_wang/DualMap` | present | `applications/runner_dataset.py` with Hydra config；`dualmap.core.Dualmap`；`Detector` + local/global map managers | `applications/offline_local_map_query.py` interactive CLIP query；local map manager object access; no stable QA API found | `map/*.pkl` object map with labels, 3D geometry, CLIP features, relations | YOLO/YOLO-World, SAM, optional FastSAM, OpenCLIP/MobileCLIP, local/global map fusion | object-memory baseline present; QA path partial |
| HOV-SG | `/home/robin_wang/HOV-SG` | present | `application/semantic_segmentation.py`；`Graph.create_feature_map()`；`save_masked_pcds` / `save_full_pcd` / `save_full_pcd_feats`；`application/create_graph.py` | `application/visualize_query_graph.py`；`Graph.query_floor`, `query_room`, `query_object`, `query_hierarchy` | `mask_feats.pt`, object point clouds, full/masked point clouds, optional graph outputs | SAM automatic masks, OpenCLIP features, 3D mask/point-cloud merging, hierarchy graph | open-vocab object/graph baseline present; package path partial |
| ConceptGraphs | `/home/robin_wang/concept-graphs` | present | `scripts/generate_gsa_results.py` or `scripts/streamlined_detections.py`; `conceptgraph/slam/cfslam_pipeline_batch.py`; `conceptgraph/scenegraph/build_scenegraph_cfslam.py` | `scripts/visualize_cfslam_results.py` interactive CLIP text query; `visualize_cfslam_interact_llava.py`; scenegraph JSON/planner utilities; no stable non-interactive evaluator API found | `pcd_saves/*.pkl.gz` object map, CLIP/text features, optional scene graph JSON | SAM segment-all or RAM/Tag2Text + GroundingDINO + SAM; streamlined path uses YOLO-World + MobileSAM; OpenCLIP; optional LLaVA captions | candidate baseline present; integration partial |
| DAAAM | `/home/robin_wang/DAAAM` | present | `scripts/run_pipeline.py`; `daaam.hydra.runner.HydraPipelineRunner`; `HydraIntegration.process_frame` | `scripts/demo_query.py`; `SceneUnderstandingAgent.answer_query`; scene graph tools such as matching subjects, radius lookup, region info, trajectory info | Hydra / Dynamic Scene Graph outputs plus semantic/background-object data | FastSAM/SAM/SAM2 via `UniversalSegmenter`, BotSort tracking, DAM/VLM grounding, CLIP ReID/features, Hydra scene graph | candidate baseline present; heavy integration needed |
| Hydra standalone | `/home/robin_wang/Hydra` | present | `hydra run mp3d <scene_path>` via Python bindings; ROS2/colcon build path; RGB-D + labels + poses dataset layout | `hydra-eval` timing/analysis; DSG artifacts readable through Hydra/Spark-DSG tooling; no NL QA API found | 3D Dynamic Scene Graph / Hydra result directory | real-time spatial perception stack for hierarchical 3D scene graph construction; semantic labels can come from dataset/model outputs | standalone DSG baseline present; evaluator integration needed |
| ReMEmbR | `/home/robin_wang/remembr` | present | `scripts/preprocess_captions.py`; `MemoryItem`; `MilvusMemory.insert`; CODa/NaVQA preprocessing scripts | `ReMEmbRAgent.query`; retrieval tools `retrieve_from_text`, `retrieve_from_position`, `retrieve_from_time`; `scripts/eval.py --model remembr+...` | captions JSON + Milvus collection with caption, pose, time, text embedding | VILA captioning, Milvus vector DB, mixedbread text embeddings, LLM agent/tool loop | caption/spatio-temporal memory baseline present |
| Multi-frame VLM | `/home/robin_wang/remembr` | present as ReMEmbR eval control | `scripts/eval.py --model vlm...` loads `VideoMemory` from sampled CODa frames | `VLMNonAgent.query` sends image frames + pose/time text to a VLM | raw sampled frames in `VideoMemory`; no explicit memory DB | GPT-4o-style VLM no-memory control over sampled frames | code path present; needs smoke/fix before claiming runnable |
| LLM with captions | `/home/robin_wang/remembr` | present as ReMEmbR eval control | `scripts/preprocess_captions.py` creates caption JSON; `scripts/eval.py --model <llm>` loads `TextMemory` | `NonAgent.query` answers from caption context without retrieval tools | caption JSON / `TextMemory` | VILA captions + plain LLM context baseline | no-explicit-memory control present |

## Track-wise Fixed API Query Support

本表只判断 method repo 或导出的 memory package 是否能支持
`capabilities.json` 里的 fixed API 查询，不判断 agent full access。Track key 以
`.codex/memory_package_spec.md` 为准。

状态含义：

- `native`：方法 repo 已有可调用的原生 API/函数，薄封装后可作为 package Python
  entrypoint。
- `export`：原生 artifact 里有足够信息，但需要先写 package exporter 或
  non-interactive reader。
- `candidate`：看起来可支持，但需要 smoke test 或把交互式/agent式调用改成稳定
  Python entrypoint；在完成前，实际 `capabilities.json` 不能写 `supported`。
- `invalid`：第一版 fixed API 应声明 `invalid`。
- `control-only`：no-explicit-memory ablation/control，不作为 memory-package fixed API
  支持。

| Method | Track 1: object inventory API | Track 2: object location query API | Track 3: ScanRefer referring query API | Track 4: OpenEQA QA / retrieval API | First package decision |
|---|---|---|---|---|---|
| ClawS SpatialRAG | `native` via `get_spatial_objects(limit)` / SQLite object rows | `native` via `query_spatial_memory`, semantic anchors, storage retrieval | `invalid`: no native ScanRefer-style referring resolver found | `candidate`: native spatial RAG/retrieval exists, but answer schema needs smoke test | Prefer `supported` for Track 1/2, `invalid` for Track 3, Track 4 after smoke |
| DualMap | `export` from `map/*.pkl` object maps | `candidate`: interactive CLIP/object query exists; needs non-interactive package entrypoint | `invalid`: no referring-expression resolver found | `invalid`: no stable QA/retrieval API found | Track 1 first; Track 2 only after query bridge; Track 3/4 invalid |
| HOV-SG | `export` from graph/object point-cloud artifacts | `native` via `Graph.query_object` / `query_hierarchy` | `invalid`: open-vocab object query is not a ScanRefer resolver yet | `invalid`: no native general QA API found | Track 1/2 likely supportable; Track 3/4 invalid |
| ConceptGraphs | `export` from `pcd_saves/*.pkl.gz` / scene graph JSON | `candidate`: interactive CLIP query exists; needs non-interactive bridge | `invalid`: no stable referring resolver found | `invalid`: no stable QA/retrieval API found | Track 1 first; Track 2 after bridge; Track 3/4 invalid |
| DAAAM | `export` from DSG object / semantic nodes if present | `candidate`: scene graph tools can match subjects and spatial neighborhoods | `invalid`: no ScanRefer-specific resolver found | `candidate`: `SceneUnderstandingAgent.answer_query(...)` exists; needs evaluator-safe call | Track 1/2/4 likely supportable after DSG package; Track 3 invalid |
| Hydra standalone | `candidate`: DSG object nodes may be exportable when labels exist | `invalid`: no natural-language object query API found | `invalid`: no referring resolver found | `invalid`: no natural-language QA API found | Track 1 only if DSG object export is clean; Track 2/3/4 invalid |
| ReMEmbR | `invalid`: caption memory has no object inventory | `invalid`: caption retrieval has pose/time but no object-level location output | `invalid`: no object referring resolver | `native` via `ReMEmbRAgent.query` and retrieval tools | Track 4 supportable; Track 1/2/3 invalid for fixed object-level APIs |
| Multi-frame VLM | `control-only` | `control-only` | `control-only` | `control-only`: query API exists but uses raw frames, not exported memory | Keep as raw-frame ablation, not fixed memory API |
| LLM with captions | `control-only` | `control-only` | `control-only` | `control-only`: `NonAgent.query` exists but is a caption-context control | Keep as caption ablation, not fixed memory API |

## ClawS SpatialRAG

- Root repo: `/home/robin_wang/ClawS-SpatialRAG`
- Native construction:
  - `SpatialPipeline.process_frame(rgb_bgr, depth_m, robot_pose, timestamp)`
    is the main per-frame API.
  - Pipeline stages are YOLO/ByteTrack event detection, depth/pose projection,
    storage/fusion, and optional VLM description.
  - Useful scripts include `scripts/test_scannetpp_pipeline.py`,
    `scripts/build_scannetpp_spatial_rag_memory.py`, and MCAP replay scripts.
- Native query/read:
  - `SpatialRAGService.tools` registers `query_spatial_memory`,
    `get_semantic_anchor`, and `remember_spatial_observation`.
  - `SpatialRAGService.get_spatial_objects(limit=500)` returns dashboard-ready
    objects with id, text, 3D position, timestamp, and crop availability.
  - `SpatialStorage.retrieve_memory(...)` and location retrieval expose lower
    level DB search.
- Memory artifact:
  - SQLite DB with sqlite-vec embeddings, `spatial_memories`, 3D positions,
    timestamps, and crop image fields.
- Current conclusion:
  - This is the only checked method with an explicit native spatial-memory query
    service. It should be first for a full memory package prototype.

## DualMap

- Root repo: `/home/robin_wang/DualMap`
- Native construction:
  - `applications/runner_dataset.py` is the main dataset runner.
  - `dualmap.core.Dualmap` initializes `Detector`, `LocalMapManager`, and
    `GlobalMapManager`.
  - `utils/object_detector.py` loads YOLO/YOLO-World, SAM, optional FastSAM,
    OpenCLIP/MobileCLIP, and class text features.
  - `config/runner_dataset.yaml` enables object detection, optional FastSAM,
    local map generation, local map saving, and parallel processing.
- Native query/read:
  - `applications/offline_local_map_query.py` supports interactive text query
    against map CLIP features.
  - `LocalMapManager` provides object access and candidate search utilities.
  - I did not find a stable non-interactive spatial QA API in the method repo.
- Memory artifact:
  - Saved concrete object maps under `map/*.pkl`, with object labels, 3D
    points/bboxes/centers, CLIP features, observations, and local relations.
- Current conclusion:
  - Strong object-memory baseline. For memory QA, design a method-specific
    package/exporter before treating it as supported.

## HOV-SG

- Root repo: `/home/robin_wang/HOV-SG`
- Native construction:
  - `application/semantic_segmentation.py` creates a `Graph`, calls
    `create_feature_map()`, and saves masked/full point clouds and features.
  - `application/create_graph.py` builds the hierarchical graph from saved
    feature-map outputs.
  - Config defaults use OpenCLIP `ViT-H-14` and SAM `vit_h`; local current-scene
    scripts may override to lighter models.
- Native query/read:
  - `application/visualize_query_graph.py` calls `Graph.query_hierarchy(...)`.
  - `Graph` exposes `query_floor`, `query_room`, `query_object`, and
    `query_hierarchy`.
- Memory artifact:
  - `mask_feats.pt`, `objects/pcd_*.ply`, `full_pcd.ply`, `masked_pcd.ply`,
    full point features, and optional hierarchy graph files.
- Current conclusion:
  - Open-vocabulary 3D object/scene-graph baseline is present. The next missing
    piece is a stable memory package that preserves object features, point-cloud
    evidence, and graph hierarchy.

## ConceptGraphs

- Root repo: `/home/robin_wang/concept-graphs`
- Native construction:
  - `scripts/generate_gsa_results.py` generates 2D detection/segmentation and
    per-region CLIP features.
  - `scripts/streamlined_detections.py` is the newer Hydra-driven detection path.
  - `conceptgraph/slam/cfslam_pipeline_batch.py` fuses detections into a 3D
    object map.
  - `conceptgraph/scenegraph/build_scenegraph_cfslam.py` can add captions and
    scene graph JSON from the object map.
- Native query/read:
  - `scripts/visualize_cfslam_results.py` includes interactive CLIP text query.
  - `scripts/visualize_cfslam_interact_llava.py` supports interactive LLaVA
    querying over visualized objects.
  - I did not find a stable non-interactive evaluator API.
- Memory artifact:
  - `pcd_saves/*.pkl.gz` object maps, serialized `MapObjectList`, CLIP/text
    features, object point clouds/bboxes, optional scene graph JSON.
- Current conclusion:
  - Good candidate after ClawS/DualMap/HOV-SG, but it needs a non-interactive
    exporter/query bridge that respects the native object-map structure.

## DAAAM

- Root repo: `/home/robin_wang/DAAAM`
- Native construction:
  - `scripts/run_pipeline.py` runs the dataset pipeline without ROS.
  - `daaam.hydra.runner.HydraPipelineRunner` orchestrates workers and saving.
  - `daaam.hydra.integration.HydraIntegration.process_frame(...)` is the
    frame-level bridge into Hydra/DSG construction.
- Native query/read:
  - `scripts/demo_query.py` loads a DSG and runs a REPL.
  - `SceneUnderstandingAgent.answer_query(...)` can answer against a loaded scene
    graph using tools such as subject matching, radius search, region
    information, and trajectory information.
- Memory artifact:
  - Dynamic Scene Graph / Hydra outputs, background-object metadata, semantic
    updates, grounding images/annotations if enabled.
- Current conclusion:
  - Method code is present and has a native scene-graph QA agent. Integration is
    heavier than static object-map baselines because we must standardize dataset
    input, DSG output paths, and evaluator-safe query calls.

## Hydra Standalone

- Root repo: `/home/robin_wang/Hydra`
- Native construction:
  - Hydra is a standalone real-time 3D scene graph construction system.
  - Python bindings document `hydra run mp3d /path/to/scene`.
  - Expected image dataset layout includes `camera_info.yaml`, RGB images, depth,
    semantic labels, and `poses.csv`.
  - Installation/running is tied to ROS2/colcon and Hydra Python bindings.
- Native query/read:
  - `hydra-eval --help` / `python -m hydra_eval --help` expose evaluation tools.
  - I found timing/result analysis, but not a natural-language QA interface.
- Memory artifact:
  - Hydra result directory / 3D Dynamic Scene Graph artifacts.
- Current conclusion:
  - This should be treated as a present standalone DSG baseline, separate from
    DAAAM's Hydra integration.

## ReMEmbR And Caption/VLM Controls

- Root repo: `/home/robin_wang/remembr`
- Native construction:
  - `scripts/preprocess_captions.py` captions CODa sequences with VILA and writes
    `data/captions/{seq_id}/captions/captions_<captioner>_<seconds>_secs.json`.
  - `MemoryItem` stores caption, time, position, and theta.
  - `MilvusMemory` stores caption memories with text embeddings, position, and
    time indexes.
- Native query/read:
  - `ReMEmbRAgent.query(...)` uses retrieval tools over caption memory:
    `retrieve_from_text`, `retrieve_from_position`, and `retrieve_from_time`.
  - `scripts/eval.py --model remembr+<llm>` runs the retrieval-agent baseline.
  - `scripts/eval.py --model <llm>` runs a caption-context baseline through
    `NonAgent`.
  - `scripts/eval.py --model vlm...` routes to `VLMNonAgent` and `VideoMemory`.
- Memory artifact:
  - caption JSON files, Milvus collections, optional `TextMemory` or `VideoMemory`
    for controls, and eval outputs under ReMEmbR's `out/`.
- Current conclusion:
  - ReMEmbR is now present and should be included as a caption/spatio-temporal
    memory baseline.
  - `LLM with captions` is not missing; it is the ReMEmbR eval `NonAgent`
    control over caption context.
  - `Multi-frame VLM` is also represented in ReMEmbR's eval path, but the current
    `VLMNonAgent` constructor appears to need a smoke test/fix before we call it
    runnable.
  - DAAAM contains OC-NaVQA data notes that reference ReMEmbR, but I did not find
    the caption/VLM control implementation inside the DAAAM repo itself.

## Missing Or Control Baselines

- No root-level method repo is currently missing from this registry among the
  checked set.
- Multi-frame VLM and LLM-with-captions should remain clearly marked as
  no-explicit-memory controls, even though their code lives under ReMEmbR.

## Next Checks Before Full Evaluate Plan

1. For each `present` method, run or dry-run the native build script on one small
   scene and record exact required inputs/outputs.
2. Define a method-specific memory package for ClawS, DualMap, and HOV-SG before
   adding any full-agent evaluation.
3. For DualMap, HOV-SG, and ConceptGraphs, write down a non-interactive object
   query contract from native artifacts before claiming QA support.
4. For Hydra, decide whether to evaluate raw DSG outputs directly or only through
   a downstream scene-graph QA layer.
5. For ReMEmbR, smoke test `scripts/eval.py` for `remembr+...`, plain caption
   LLM, and VLM-control modes.
6. For DAAAM, decide whether to evaluate it through saved DSG files or through
   the live `SceneUnderstandingAgent` call path.
