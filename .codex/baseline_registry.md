# Baseline Registry

Last checked: 2026-06-17

> 2026-06-17 integration note: Claude tasks 01-05 re-audited the first fixed-API
> baseline capability slice. This file intentionally combines the method-specific
> evidence from all five task branches instead of letting later branches overwrite
> earlier rows in the shared registry.
>
> - Task 01: ClawS SpatialRAG.
> - Task 02: HOV-SG.
> - Task 03: DualMap.
> - Task 04: ConceptGraphs.
> - Task 05: DAAAM, Hydra, ReMEmbR, Multi-frame VLM, and LLM-with-captions controls.
>
> Follow-up finalization tasks build on the Task 05 audit:
>
> - Task 19: finalized the ReMEmbR Track 1/2 fixed-API outcome as `invalid`
>   (object-free caption/pose/time memory) with reproducible root evidence. The
>   matching minimal invalid declaration package is a generated artifact (it lives
>   under the gitignored `memories/` tree, not in git) produced on demand by
>   `scripts/methods/remembr/build_invalid_declaration.py`. Track 4 / agentic
>   relevance is recorded separately and is not promoted by this task.
>
> Evidence paths point only at root method repos or native artifacts. The
> evaluation repo `adapters/` are never cited as proof of method capability.

本文件只确认 baseline 方法本身的性质和原生接口。判断依据是
`/home/robin_wang` 下的方法 repo、repo 内的脚本/API、配置和输出 artifact；
evaluation repo 的 `adapters/` 只属于后续接入层，不能作为 baseline 支持某个
接口的证据。

## 判定规则

- `present`：root 路径下有本地代码，并且能定位到方法原生的构建或查询入口。
- `partial`：本地代码存在，但缺少稳定的非交互式评测接口或 memory package。
- `missing`：root 路径下没有对应方法 repo，或没有找到需要的原生能力。
- no-memory controls 单独列出；它们不是 spatial-memory 方法 repo。

## Formal Track 1/2 Detector Policy

Track 1/2 的正式主评测只使用 shared strongest open-vocabulary detector setup，并继续只报告 detector-coverable split。
所有 detector-backed object-memory 方法都必须使用同一个 shared OV detector、prompt/evaluation label list、checkpoint 和 preprocess。

- 当前 formal target 是 YOLO-World-L (`yolov8l-world.pt`)；它在 DualMap config 和 ConceptGraphs streamlined path 中都有 native 证据。
- 本机 2026-06-17 只找到 `/home/robin_wang/DualMap/yolov8s-world.pt`，因此 YOLO-World-S 只能作为 smoke fallback。
- HOV-SG 这类没有 detector 的方法必须记录其 shared SAM/CLIP open-vocabulary prompt/label route。
- non-shared detector、method-native detector/checkpoint 或不同 prompt list 结果只能标记为 `module_ablation`，不能混入 Track 1/2 主表。
- fixed API support 判断必须来自 root repo 或 native artifact，不能来自 evaluation adapter 或临时 LLM wrapper。

## 方法总表

| Method | Root path | Local code | Native build / ingest interface | Native query / read interface | Main memory artifact | Perception / model stack | Status |
|---|---|---:|---|---|---|---|---|
| ClawS SpatialRAG | `/home/robin_wang/ClawS-SpatialRAG` | present | `SpatialPipeline.process_frame(...)` `spatial_rag/pipeline.py:159`; build script `scripts/build_scannetpp_spatial_rag_memory.py:1`; ScanNet++/MCAP replay scripts | native non-interactive reader `load_memory_records(...)` `spatial_rag/eval/memory_loader.py:32` (labels+3D, GT-free, vec0-aux fallback `:148`); `SpatialRAGService.tools` `spatial_rag/clawspine_adapter.py:147` (`query_spatial_memory`/`get_semantic_anchor`/`remember_spatial_observation`); `get_spatial_objects` `:151`; `SpatialStorage.retrieve_memory`/`retrieve_by_location`/`get_entity_anchor` `spatial_rag/storage.py:702`,`:716`,`:733` | sqlite-vec `vec0` table `spatial_memories` (`spatial_rag/config.py:28`, schema `spatial_rag/storage.py:101`) + `crop_images` `:114`; built artifact `outputs/scannetpp_memory_036bce3393_ollama_vlm.db` (183 records) | YOLO/Ultralytics + ByteTrack `spatial_rag/visual_trigger.py:111`,`:105`; depth/pose projection `spatial_rag/depth_utils.py:139`; optional VLM describe/verify `spatial_rag/pipeline.py:372`; embedding providers `spatial_rag/embedding.py:51`,`:82`,`:176`; sqlite-vec fusion/search `spatial_rag/storage.py:296` | native memory + query + non-interactive read path present |
| DualMap | `/home/robin_wang/DualMap` (commit `157235e`) | present | `applications/runner_dataset.py:17` Hydra runner；`dualmap/core.py:37` `Dualmap` wires `Detector`+`LocalMapManager`+`GlobalMapManager` (`core.py:48-51`)；concrete (local) map saved by default (`config/system_config.yaml:183`,`193`) | `applications/offline_local_map_query.py:171` interactive CLIP top-k over concrete-map feats (open3d keypress + `input()`)；nav-embedded inquiry `utils/global_map_manager.py:656`,`utils/local_map_manager.py:1096`; no non-interactive QA API | concrete map `map/<uid>.pkl` per-object `class_id`+`clip_ft`+`pcd` (`utils/object.py:64-74`); global/abstract map adds `pcd_2d`+`related_objs` anchors (`utils/object.py:739-762`) | YOLO-World + SAM/MobileSAM, optional FastSAM, OpenCLIP/MobileCLIP, low/high-mobility CLIP classifier (`utils/object_detector.py:176-227`) | concrete object-memory baseline present; formal eval needs shared OV route; non-interactive query path partial |
| HOV-SG | `/home/robin_wang/HOV-SG` | present | `application/semantic_segmentation.py:13-25` → `Graph.create_feature_map()` + `save_masked_pcds`/`save_full_pcd`/`save_full_pcd_feats` (`graph.py:141,1225,1252,1327`)；`application/create_graph.py:9-37` adds `build_graph` but **skips it for `scannet`/`replica`** (`create_graph.py:34-37`) | `application/visualize_query_graph.py:31-45` interactive REPL → `Graph.query_hierarchy` (`graph.py:1178`)；`Graph.query_floor`/`query_room`/`query_object` (`graph.py:965,1005,1082`) operate on `self.objects`, populated only by `build_graph` (HM3DSem path) | `mask_feats.pt`, `full_feats.pt`, `objects/pcd_*.ply`, `full_pcd.ply`, `masked_pcd.ply` (`graph.py:1252-1271,1327-1362`); graph dir (`floors/`,`rooms/`,`objects/*.json+.ply`) only on HM3DSem | SAM automatic masks (`vit_h` default, `graph.py:114-127`), OpenCLIP `ViT-H-14`/`laion2b_s32b_b79k` (`graph.py:90-103`, `config/*.yaml`), 3D mask merge + per-mask CLIP feats | open-vocab object/feature-map baseline present; shared OV route via `evaluate_sem_seg.py`; hierarchy graph is HM3DSem-only |
| ConceptGraphs | `/home/robin_wang/concept-graphs` | present | `conceptgraph/scripts/generate_gsa_results.py` or `conceptgraph/scripts/streamlined_detections.py`; `conceptgraph/slam/cfslam_pipeline_batch.py`; `conceptgraph/scenegraph/build_scenegraph_cfslam.py` | `conceptgraph/scripts/visualize_cfslam_results.py:269-310` interactive CLIP recolor (no machine output); `conceptgraph/scripts/visualize_cfslam_interact_llava.py` interactive LLaVA; `conceptgraph/scripts/eval_replica_semseg.py:133-140` non-interactive fixed-list class assignment (Replica-only); no stable non-interactive query API | `pcd_saves/full_pcd_*.pkl.gz` serialized `MapObjectList` with labels + 3D pcd/bbox, CLIP/text features; optional scene graph JSON | SAM `vit_h` segment-all or RAM/Tag2Text + GroundingDINO + SAM; streamlined path uses YOLO-World `yolov8l-world.pt` + MobileSAM; OpenCLIP `ViT-H-14/laion2b_s32b_b79k`; optional LLaVA captions + GPT-4 scene graph | object-memory baseline present; Track 1 export-able, Track 2 candidate; no ScanNet++ loader |
| DAAAM | `/home/robin_wang/DAAAM` | present | `scripts/run_pipeline.py`; `daaam.hydra.runner.HydraPipelineRunner`; `HydraIntegration.process_frame` | `scripts/demo_query.py`; `SceneUnderstandingAgent.answer_query`; scene graph tools such as matching subjects, radius lookup, region info, trajectory info | Hydra / Dynamic Scene Graph outputs plus semantic/background-object data | FastSAM/SAM/SAM2 via `UniversalSegmenter`, BotSort tracking, DAM/VLM grounding, CLIP ReID/features, Hydra scene graph | candidate baseline present; heavy integration needed |
| Hydra standalone | `/home/robin_wang/Hydra` | present | `hydra run mp3d <scene_path>` via Python bindings; ROS2/colcon build path; RGB-D + labels + poses dataset layout | `hydra-eval` timing/analysis; DSG artifacts readable through Hydra/Spark-DSG tooling; no NL QA API found | 3D Dynamic Scene Graph / Hydra result directory | real-time spatial perception stack for hierarchical 3D scene graph construction; semantic labels can come from dataset/model outputs | standalone DSG baseline present; evaluator integration needed |
| ReMEmbR | `/home/robin_wang/remembr` | present | `scripts/preprocess_captions.py`; `MemoryItem`; `MilvusMemory.insert`; CODa/NaVQA preprocessing scripts | `ReMEmbRAgent.query`; retrieval tools `retrieve_from_text`, `retrieve_from_position`, `retrieve_from_time`; `scripts/eval.py --model remembr+...` | captions JSON + Milvus collection with caption, pose, time, text embedding | VILA captioning, Milvus vector DB, mixedbread text embeddings, LLM agent/tool loop | caption/spatio-temporal memory baseline present |
| Multi-frame VLM | `/home/robin_wang/remembr` | present as ReMEmbR eval control | `scripts/eval.py --model vlm...` loads `VideoMemory` from sampled CODa frames | `VLMNonAgent.query` sends image frames + pose/time text to a VLM | raw sampled frames in `VideoMemory`; no explicit memory DB | GPT-4o-style VLM no-memory control over sampled frames | code path present; needs smoke/fix before claiming runnable |
| LLM with captions | `/home/robin_wang/remembr` | present as ReMEmbR eval control | `scripts/preprocess_captions.py` creates caption JSON; `scripts/eval.py --model <llm>` loads `TextMemory` | `NonAgent.query` answers from caption context without retrieval tools | caption JSON / `TextMemory` | VILA captions + plain LLM context baseline | no-explicit-memory control present |

## Track-wise Fixed API Query Support

本表只判断 method repo 或导出的 memory package 是否能支持
`capabilities.json` 里的 fixed API 查询，不判断 agent full access。Track key 以
`.codex/memory_package_spec.md` 为准。Track 1/2 的 `supported` 默认指 formal
shared OV detector route；non-shared detector/method-native detector variants 只作为 module ablation，不提升 fixed API 状态。

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
| ClawS SpatialRAG | `native`: GT-free reader `spatial_rag/eval/memory_loader.py:32` returns label+3D per record (smoke: 183 records, 183 normalized labels); also `get_spatial_objects` `spatial_rag/clawspine_adapter.py:151` | `native`: embedding-free `get_entity_anchor` `spatial_rag/storage.py:733` + `retrieve_by_location` `:716` (name→3D / radius); semantic `query_spatial_memory` `spatial_rag/clawspine_adapter.py:328` needs embedding service | `invalid`: no ScanRefer/referring resolver in root repo (no `scanrefer`/`referring` symbol) | `candidate`: native retrieval `retrieve_memory` `spatial_rag/storage.py:702` exists, but no native answer-synthesis QA API; in-repo eval retrieval/QA path is `not_implemented_first_version` `scripts/evaluate_scannetpp_spatial_rag.py:468` (status `:476`) | Track 1 `supported` (native reader); Track 2 `supported` (anchor/location native, semantic needs embedder); Track 3 `invalid`; Track 4 stays `candidate` pending retrieval→answer bridge |
| DualMap | `export` from concrete map `map/<uid>.pkl` object pickles; formal shared-OV route has native precedent in `evaluation/sem_seg_eval.py:381` (`calc_clip_labels` projects each `clip_ft` onto the fixed ScanNet200 list `utils/eval/scannet200_constants.py` via cosine + argsort, reassigning `obj.class_id`) | `candidate`: only interactive open3d CLIP query (`applications/offline_local_map_query.py:171`) and nav-embedded inquiry (`utils/global_map_manager.py:656`); no native non-interactive bridge, so not `supported` yet | `invalid`: no referring-expression resolver found | `invalid`: no stable QA/retrieval API found | Track 1 shared OV first; Track 2 only after a native non-interactive query bridge; method-native detector/OV override is `module_ablation`; Track 3/4 invalid |
| HOV-SG | `export` (shared OV): per-mask PLY centroid + `mask_feats.pt` → shared OV label via `evaluate_sem_seg.py` argmax over `SCANNET_LABELS_20`; needs thin exporter, no native object-table writer | `candidate` (shared OV): `query_object` is CLIP sim, but global-search path is broken (`graph.py:1125` NameError) and `self.objects` is empty on ScanNet path; shared OV needs a label-constrained reader over `mask_feats.pt`, not the native REPL | `invalid`: open-vocab CLIP object query is not a ScanRefer relation/attribute resolver | `invalid`: no native general QA API; `query_hierarchy` needs `OPENAI_KEY` and is not QA | Track 1 shared-OV export supportable; Track 2 only after a non-interactive shared OV reader (native query path is HM3DSem-only + buggy); method-native detector/OV override is `module_ablation`; Track 3/4 invalid |
| ConceptGraphs | `export` from `pcd_saves/full_pcd_*.pkl.gz` serialized `MapObjectList` (labels + 3D pcd/bbox); formal result requires shared OV detector route | `candidate`: only interactive CLIP recolor + interactive LLaVA exist; `eval_replica_semseg.py:133-140` proves shared OV `clip_ft @ class_feats.T → argmax` primitive but it is Replica-specific, not a query entrypoint; needs non-interactive shared OV bridge | `invalid`: no referring-expression resolver found | `invalid`: no native non-interactive QA/retrieval API (LLaVA path is interactive point-pick only) | Track 1 shared OV first; Track 2 stays candidate until native bridge; method-native detector/OV override is `module_ablation`; Track 3/4 invalid |
| DAAAM | `export`: adapter exports `object_table.jsonl` from DSG OBJECTS + BACKGROUND_OBJECTS via root-repo object readers | `candidate/supported-per-package`: adapter exports a deterministic native semantic index from DAAAM scene-understanding embeddings; package declares Track 2 `supported` only if that index builds, otherwise `invalid` | `invalid`: no ScanRefer-specific resolver found | `candidate`: `SceneUnderstandingAgent.answer_query(...)` is a native LLM QA agent; needs evaluator-safe non-interactive call + judge isolation | Track 1 package exporter implemented in `scripts/methods/daaam/build_memory_smoke.py`; Track 2 fixed API uses only non-LLM native semantic index, never `SceneUnderstandingAgent`; Track 3 invalid; Track 4 deferred |
| Hydra standalone | `candidate`: DSG OBJECTS nodes carry `semantic_label` + centroid and are enumerable (`graph.getLayer(DsgLayers::OBJECTS)`), but labels are external input, not Hydra-generated; needs smoke test | `invalid`: no natural-language object query API found | `invalid`: no referring resolver found | `invalid`: no natural-language QA API found | Track 1 only if DSG object export is clean AND label-space fairness is decided; Track 2/3/4 invalid |
| ReMEmbR | `invalid` (Task 19 finalized): caption memory has no object inventory — `MemoryItem` fields are `caption`/`time`/`position`/`theta` only (`remembr/memory/memory.py:5-9`) and the Milvus collection stores `text_embedding`/`position`(robot)/`theta`/`time`/`caption` with no object/label/bbox field (`remembr/memory/milvus_memory.py:37-45`) | `invalid` (Task 19 finalized): the only read paths are `search_by_text`/`search_by_position`/`search_by_time`, which return `memory_to_string` caption+robot-pose+time strings (`remembr/memory/milvus_memory.py:231-250`); no object-level 3D location output and `position` is the robot pose, not an object position | `invalid`: no object referring resolver | `native` via `ReMEmbRAgent.query` (`remembr/agents/remembr_agent.py:390`) LangGraph agent + retrieval tools `retrieve_from_text`/`retrieve_from_position`/`retrieve_from_time` (`remembr/agents/remembr_agent.py:168,183,199`) | Track 1/2/3 `invalid` for fixed object-level APIs (Task 19 minimal invalid package generated on demand by `scripts/methods/remembr/build_invalid_declaration.py`; the package itself stays under the gitignored `memories/` tree); Track 4 native via the retrieval agent but still `candidate` pending a non-interactive `remembr+<llm>` smoke; primary comparison path is agentic |
| Multi-frame VLM | `control-only` | `control-only` | `control-only` | `control-only`: `VLMNonAgent.query` exists but uses raw frames, not exported memory; constructor bug (`'gpt-4' in 'llm_type'`) blocks instantiation until fixed/smoke-tested | Keep as raw-frame ablation, `explicit_memory=false`; not a fixed memory API |
| LLM with captions | `control-only` | `control-only` | `control-only` | `control-only`: `NonAgent.query` exists but is a no-retrieval caption-context control | Keep as caption ablation, `explicit_memory=false`; not a fixed memory API |

## ClawS SpatialRAG

- Root repo: `/home/robin_wang/ClawS-SpatialRAG` (root-repo evidence only; the
  evaluation-repo `adapters/` are not cited below).

- Native build / ingest:
  - `SpatialPipeline.process_frame(rgb_bgr, depth_m, robot_pose, timestamp)` is
    the main per-frame API — `spatial_rag/pipeline.py:159`.
  - Per-frame stages: LR1 YOLO+ByteTrack detection/tracking, LR2 depth/pose
    projection, LR3 embed + store/fuse, optional LR4 VLM describe —
    `spatial_rag/pipeline.py:180`–`:275`, `:372`.
  - Native ScanNet++ build entrypoint that reuses the production pipeline and
    writes a sqlite memory DB: `scripts/build_scannetpp_spatial_rag_memory.py:1`
    (pipeline wiring at `:311`, frame loop at `:428`). MCAP replay path:
    `scripts/replay_mcap_dashboard.py`; pipeline smoke `scripts/test_scannetpp_pipeline.py`.

- Native memory artifact:
  - sqlite-vec `vec0` virtual table `spatial_memories` with columns
    `embedding float[dim] distance_metric=cosine`, `snapshot_text`,
    `pos_x/pos_y/pos_z`, `timestamp` — schema `spatial_rag/storage.py:101`;
    table/metric/dim defaults `spatial_rag/config.py:28`,`:26`,`:25`.
  - Companion `crop_images` BLOB table keyed by `memory_id` —
    `spatial_rag/storage.py:114`.
  - Built artifact present in repo: `outputs/scannetpp_memory_036bce3393_ollama_vlm.db`.
    Inspected: `spatial_memories_auxiliary`/`spatial_memories_rowids` = 183 rows,
    `crop_images` = 599 rows. (`spatial_memories` MATCH queries need the `vec0`
    module loaded; the native reader has a fallback — see below.)
  - Each record carries a label (parsed from `snapshot_text`, e.g. `**chair**`)
    and an explicit 3D position, satisfying the Track 1 "labels + 3D position"
    requirement directly from native records.

- Native query / read (all inside the ClawS root repo, not an adapter):
  - Non-interactive, GT-free reader `load_memory_records(memory_db, ...)` —
    `spatial_rag/eval/memory_loader.py:32`. Returns `PredMemory` rows
    (`memory_id`, `predicted_label`, `normalized_label`, `snapshot_text`,
    3D `position`, frame/track ids) — dataclass at `:18`, label extraction at
    `:211`. Includes a `vec0` auxiliary-table fallback so it reads the DB even
    when `sqlite-vec` is absent — `:148`–`:171`.
  - `SpatialRAGService.tools` registers three ToolHandler tools —
    `spatial_rag/clawspine_adapter.py:147`: `query_spatial_memory`
    (semantic vector search, `:328`), `get_semantic_anchor` (entity name → most
    recent 3D, `:422`), `remember_spatial_observation` (write/fuse, `:637`).
  - `SpatialRAGService.get_spatial_objects(limit=500)` returns dashboard-ready
    objects with id, `snapshot_text`, 3D position, timestamp —
    `spatial_rag/clawspine_adapter.py:151`.
  - Lower-level storage read APIs: `retrieve_memory` (vector kNN,
    `spatial_rag/storage.py:702`), `retrieve_by_location` (3D radius search,
    `:716`), `get_entity_anchor` (substring name → most recent 3D, `:733`),
    `get_all_objects` (`:745`). `get_entity_anchor`/`retrieve_by_location`/
    `get_all_objects` need no embedding service; only `query_spatial_memory`/
    `retrieve_memory` need an embedder.

- Perception / model stack:
  - Detector/tracker: Ultralytics YOLO + ByteTrack —
    `spatial_rag/visual_trigger.py:111` (`UltralyticsBackend`, loads YOLO at
    `:118`), tracker config `bytetrack.yaml` at `:105`. Repo ships
    `yolo11n.pt` / `yolov8n.pt` weights.
  - 3D projection from depth + pose: `DetectionProjector.project` —
    `spatial_rag/depth_utils.py:139`,`:192`; `ProjectionEngine` /
    `CameraIntrinsics` / `RobotPose` in `spatial_rag/projection.py:80`,`:23`,`:52`.
  - Embedding providers: `Mock`/`Ollama`/`VLLM` —
    `spatial_rag/embedding.py:51`,`:82`,`:176` (optional multimodal image input).
  - Optional VLM description/verification + LLM fusion check —
    `spatial_rag/pipeline.py:372`; `spatial_rag/vlm_describer.py`.
  - Storage/fusion: sqlite-vec with semantic+spatial fusion candidates —
    `spatial_rag/storage.py:296`,`:408`.

- Smoke evidence (this audit, env `spatial-rag`):
  - `load_memory_records('outputs/scannetpp_memory_036bce3393_ollama_vlm.db')`
    returned 183 records, 0 warnings, all 183 with a normalized label and a 3D
    position (e.g. `cup (6.37, 2.61, 0.91)`, `tv→display (6.91, 3.11, 1.61)`).
    This confirms the native read path emits Track 1 object inventory without GT.

- Track 1/2 support decision (root-repo evidence):
  - Track 1 (object inventory): native object/memory records with labels and 3D
    position exist, and `spatial_rag/eval/memory_loader.py:32` is a thin,
    deterministic, GT-free reader over them → `native`/supported.
  - Track 2 (object location): native, embedding-free location/anchor reads
    (`get_entity_anchor` `spatial_rag/storage.py:733`, `retrieve_by_location`
    `:716`) plus the semantic `query_spatial_memory`
    (`spatial_rag/clawspine_adapter.py:328`, needs an embedder) → `native`/supported.
  - Track 3 (ScanRefer referring): no referring-expression resolver in the root
    repo (`grep -i scanrefer|referring` over `spatial_rag/`+`scripts/` is empty)
    → `invalid`.
  - Track 4 (OpenEQA QA): native retrieval exists (`retrieve_memory`,
    `spatial_rag/storage.py:702`) but there is no native answer-synthesis API;
    the in-repo evaluator marks its retrieval/QA sections as
    `not_implemented_first_version`
    (`scripts/evaluate_scannetpp_spatial_rag.py:468`, status `:476`;
    relation placeholder `:491`) → `candidate`.

- Native memory/query vs. future package-export work:
  - The capabilities above (read records, query by name/location/vector) are
    native to the ClawS root repo and need no exporter to be exercised.
  - Writing the spec'd memory package (`manifest.json`/`capabilities.json`/
    `schema.md` + Track entrypoints) is a separate future fixed-API task. The
    native `load_memory_records` reader is the obvious basis for the Track 1/2
    package entrypoints but is not itself a package; do not conflate the two.

- Unresolved questions (Track 4 stays `candidate`):
  1. Track 4: is a thin `retrieve_memory` → answer-formatting wrapper acceptable
     as fixed-API support, or does Track 4 require a method-native QA path? The
     root repo has retrieval but no native answer synthesis, so Track 4 must not
     be promoted to `supported` yet (do not promote a generic
     object-table-to-LLM answerer per `memory_package_spec.md`).
  2. Track 2 semantic variant depends on an embedding service
     (`Ollama`/`VLLM`); the built DB used `ollama_vlm` embeddings. The exact
     embedder model/dim must be pinned before a fair formal Track 2 run; the
     embedding-free anchor/location reads are unaffected.
  3. ClawS uses a COCO-style YOLO vocabulary, not the canonical
     `detector_coverable.txt`; a shared OV label normalization
     (`configs/scannetpp_label_mapping.yaml`, `LabelMapper`) is applied at read
     time, but the formal shared OV route's class-list parity still needs sign-off.

## DualMap

- Root repo: `/home/robin_wang/DualMap` (audited at commit `157235e`).
- Dual-map architecture (concrete map vs global/abstract map):
  - `dualmap/core.py:48-51` builds one `Detector`, one `LocalMapManager`
    (concrete map), and one `GlobalMapManager` (global/abstract map).
  - The **concrete map** is the per-keyframe local object map of
    `LocalObject`s (`utils/local_map_manager.py:36` `self.local_map`). It holds
    full 3D object point clouds, AABBs, CLIP features, per-class Bayesian
    probabilities, mobility flags, and an `on`-relation graph
    (`utils/object.py:202-714`). Config and docs call this the "concrete map"
    (`resources/doc/app_runner_dataset.md:165`,`210`).
  - The **global/abstract map** (`utils/global_map_manager.py:22`) distills
    stabilized low-mobility local objects into `GlobalObject` "anchor" objects
    (`config/system_config.yaml:161` "GLOBAL OBJECTS CONFIG --> ANCHOR
    OBJECTS"). It keeps a 2D-downsampled `pcd_2d`/`bbox_2d` for navigation plus
    `related_objs` CLIP features for nearby anchors (`utils/object.py:717-762`,
    `utils/local_map_manager.py:465-494`). It is built for ROS/nav and is not
    saved during dataset evaluation.
- Concrete map vs global map relevance for evaluation:
  - Default dataset runs set `run_local_mapping_only: true`, `save_local_map:
    true`, `save_global_map: false` (`config/system_config.yaml:183,193,194`;
    `config/runner_dataset.yaml:64,67,70`), so the artifact on disk for
    evaluation is the **concrete map** under `map/<uid>.pkl`. The native
    segmentation eval explicitly evaluates "the saved concrete map"
    (`resources/doc/app_runner_dataset.md:165,173`).
  - The global/abstract map is a downstream 2D navigation/anchor layer
    (`pcd_2d`, walls, paths) and is the right object source only if Track 1/2
    is reframed around abstracted anchors; for object inventory / location it
    is lossy (2D footprints, related-feat-only neighbors) and not exported by
    default. Track 1/2 should consume the concrete map.
- Native construction:
  - `applications/runner_dataset.py:17` is the Hydra dataset runner; it
    instantiates `Dualmap` and calls `sequential_process`/`parallel_process`
    per keyframe, then `end_process` saves the map (`dualmap/core.py:236-654`).
  - `config/runner_dataset.yaml` / `config/system_config.yaml` enable
    detection, optional FastSAM, concrete-map generation and saving.
  - `utils/base_map_manager.py:7-25` gives both managers a shared `Tracker`,
    `is_initialized`, and visualizer.
- Object fields (concrete map `map/<uid>.pkl`):
  - Serialized state is `uid`, `pcd` points+colors, `clip_ft`, `class_id`,
    `nav_goal` (`utils/object.py:64-74`); `bbox` is recomputed from `pcd` only
    when `update_info` runs, so a reader must derive 3D center/bbox from the
    point cloud after load (`utils/object.py:76-95`).
  - Live `LocalObject` additionally carries observations, mobility, major-plane
    z, and class-probability state (`utils/object.py:202-240`); relations live
    in the manager's `networkx` graph (`utils/local_map_manager.py:39,94-119`),
    not inside the per-object pickle.
  - Observation/type fields: `utils/types.py:41-87` (`Observation` has
    `class_id`/`pcd`/`bbox`/`clip_ft`; `LocalObservation` adds
    `mask`/`xyxy`/`conf`/`distance`/`is_low_mobility`; `GlobalObservation` adds
    `uid`/`pcd_2d`/`bbox_2d`/`related_objs`).
  - A Track 1 export is a thin object-table read of this pickle dir (no new
    capability); label resolution still needs the shared OV label-normalization step below.
- Native query/read:
  - `applications/offline_local_map_query.py:171` (`query_callback`) encodes a
    text query with CLIP and returns top-k concrete-map objects by cosine
    similarity, but only inside an open3d `VisualizerWithKeyCallback` keypress
    loop with `input()` (`offline_local_map_query.py:131-133,288-299`) — not a
    callable, non-interactive entrypoint.
  - The nav inquiry path (`utils/global_map_manager.py:656`
    `find_best_candidate_with_inquiry`, `utils/local_map_manager.py:1096`) is
    CLIP-cosine object retrieval too, but it is driven by a watched
    `config/actions.yaml` flag (`dualmap/core.py:347-356,737-755`) and returns a
    single nav goal, not a query API.
  - No stable non-interactive object-location or spatial-QA function was found
    in `applications/`, `dualmap/`, `utils/`, or `evaluation/`.
- Perception / model stack:
  - `utils/object_detector.py:176-201` loads YOLO/YOLO-World
    (`config/system_config.yaml:9-12`, default `yolov8l-world.pt`), SAM
    (default `mobile_sam.pt`, `system_config.yaml:41-44`), and optional FastSAM
    (`system_config.yaml:46-48`) for open-vocabulary mask supplementation
    (FastSAM masks default to the `unknown` class, `object_detector.py:468-473`).
  - Features: OpenCLIP/MobileCLIP (default `MobileCLIP-S2`/`datacompdr`,
    `system_config.yaml:50-57`), used for object CLIP features and a low/high
    mobility prototype classifier (`object_detector.py:203-227,1420-1449`).
  - Detector prompt list defaults to `config/class_list/gpt_indoor_general.txt`
    (open-vocabulary GPT class list, `system_config.yaml:25`); fair-comparison
    runs must set `yolo.given_classes_path` to the shared OV prompt/evaluation list
    from `.codex/modules.md`.
- Formal shared OV detector status — explicit:
  - DualMap is open-vocabulary by default (CLIP-feature objects, YOLO-World +
    FastSAM `unknown` masks), so unrestricted output is `module_ablation` only.
  - A native fixed-list projection precedent exists: `evaluation/sem_seg_eval.py:381`
    (`calc_clip_labels`) text-encodes a **fixed evaluation class list** and reassigns
    each object's `class_id` to the top-1 match of its `clip_ft` against that
    list (`sem_seg_eval.py:387-419`). For ScanNet this list is the canonical
    ScanNet200 vocabulary (`VALID_CLASS_IDS_200`/`CLASS_LABELS_200` from
    `utils/eval/scannet200_constants.py`, loaded at
    `sem_seg_eval.py:19,163-181`; `config/class_list/scannet200_classes.txt` =
    200 labels + `unknown`). `scripts/dualmap_scannet.sh:10,17` runs the dataset
    runner then `evaluation.sem_seg_eval`.
  - This is the required shared OV pattern: keep the native OV model path, use the shared OV prompt/evaluation list, and record raw plus normalized labels. The
    formal Track 1/2 shared OV route should reuse this `clip_ft`→fixed-list
    projection with the shared OV prompt/evaluation list, not the OV GPT list.
- Track 1 / Track 2 fixed API decision:
  - Track 1 (object inventory): `export` — the concrete-map pickle dir already
    contains `class_id`, `clip_ft`, and `pcd` per object; a thin object-table
    reader plus the shared OV label projection above yields canonical labels +
    3D positions. Formal result requires the shared OV route; method-native detector/OV override stays
    `module_ablation`.
  - Track 2 (object location query): `candidate`, not `supported`. CLIP-cosine
    object retrieval exists natively, but only as an interactive open3d loop and
    a nav-flag inquiry; there is no native non-interactive query bridge, so per
    the implementation rules Track 2 must not be marked supported until such a
    bridge is identified or thinly added.
  - Track 3/4: `invalid` — no referring-expression resolver and no native
    QA/retrieval API.
- Current conclusion:
  - Strong concrete object-memory baseline with a native shared OV projection
    precedent for formal shared OV Track 1. Track 2 stays `candidate`
    pending a native non-interactive query bridge; do not export in this audit.

## HOV-SG

- Root repo: `/home/robin_wang/HOV-SG` (commit `d6e65a5`, `setup.py` package
  `hovsg==1.0.0`). Open-vocabulary hierarchical 3D scene graph (RSS 2024).
- Native construction (two distinct entrypoints, both Hydra/`config/`):
  - `application/semantic_segmentation.py:13-25` — builds the feature map only:
    `Graph(params)` → `create_feature_map()` (`graph.py:141`) →
    `save_masked_pcds(state="both")` + `save_full_pcd` + `save_full_pcd_feats`
    (`graph.py:1327,1225,1252`). No floors/rooms/objects graph, no object names.
  - `application/create_graph.py:9-37` — same feature map, then
    `build_graph()` (`graph.py:807`) which runs `segment_floors` →
    `segment_rooms` → `segment_objects` → `create_graph` → `create_nav_graph`
    and `save_graph` (`graph.py:719`). **Critical: `create_graph.py:34-37`
    skips `build_graph()` when `dataset` is `scannet` or `replica`.** The
    hierarchical graph and named `Object` nodes are produced only on the
    `hm3dsem` path.
  - Build inputs: posed RGB-D via dataloaders (`hovsg/dataloader/scannet.py`,
    `replica.py`, `hm3dsem.py`); ScanNet layout = `color/`, `depth/`, `pose/`,
    `intrinsic/intrinsic_{color,depth}.txt`, depth scale 1000 (`scannet.py:36`).
  - Object naming is CLIP-vs-text-label argmax, not a detector:
    `segment_objects` (`graph.py:591-702`) calls `get_label_feats(...)` then
    `identify_object` (`graph.py:578-589`). Default `obj_labels=HM3DSEM_LABELS`
    (1624-class CSV, `hovsg/labels/HM3D_CountsOfObjectTypes.csv`,
    `label_feats.py:53-58`).
- Native query/read:
  - `application/visualize_query_graph.py:31-45` is an interactive Open3D REPL
    (`input()` loop, no non-interactive entrypoint) calling
    `Graph.query_hierarchy` (`graph.py:1178`).
  - `Graph.query_object` (`graph.py:1082`) / `query_room` (`graph.py:1005`) /
    `query_floor` (`graph.py:965`) / `query_hierarchy` (`graph.py:1178`) all
    operate on `self.objects`/`self.rooms`/`self.floors`, which are populated
    only by `build_graph`/`load_graph` — i.e. the HM3DSem path. On the ScanNet
    feature-map path these lists are empty.
  - Two blockers for direct reuse as a fixed API:
    - `query_object` all-rooms default (`room_ids=[]`) reads `objects_list`
      before it is assigned (`graph.py:1125-1140`) → `NameError`; only the
      `room_ids != []` branch assigns `objects_list`. This fires on direct
      all-rooms calls and via `query_hierarchy` whenever the query has no room
      component (`query_hierarchy` passes `room_ids=[]`, `graph.py:1200-1216`).
    - `query_hierarchy` → `parse_hier_query` (`llm_utils.py:169-216`) requires
      `os.environ["OPENAI_KEY"]` and calls GPT-3.5; query parsing is not
      self-contained.
- Shared OV detector (shared OV) formal route — present and is the recommended path:
  - `application/eval/evaluate_sem_seg.py:62-75` loads the native feature map
    (`load_feature_map`, `eval_utils.py:125` → `mask_feats.pt` + `objects/pcd_*.ply`),
    constrains the label space to a fixed list (`SCANNET_LABELS_20`,
    `label_constants.py:3-25`, 20 classes + background; ScanNet path
    `eval_sem_seg.py:61-64`), computes CLIP text-vs-mask similarity
    (`text_prompt`, `eval_utils.py:79`) and assigns each mask its argmax label
    (`sim_2_label`, `eval_utils.py:268`). This is exactly the shared OV prompt/list constraint the
    formal policy requires: native OV CLIP features, but query/label space
    pinned to a shared OV prompt/evaluation class list.
  - Class list is swappable via `obj_labels` (`label_feats.py:26-59`:
    `COCO_STUFF_CLASSES`, `MATTERPORT_LABELS_160/40`, `HM3DSEM_LABELS`, etc.), so
    the evaluation repo's canonical `detector_coverable.txt` can be injected as
    the shared OV label set without editing HOV-SG.
- shared OV detector route feasibility: **feasible (no hard blocker).** A thin reader can
  load `mask_feats.pt` + `objects/pcd_*.ply`, run CLIP argmax over the canonical
  shared OV prompt/evaluation list (reusing `eval_utils.text_prompt`/`sim_2_label` logic), and emit an
  object table with `object_id`, shared OV `label`, centroid `position_3d`, and the
  native PLY as evidence. This is a format/non-interactivity wrapper, not a
  capability change, so it satisfies the fixed-API eligibility gate.
- Modules / checkpoints (`config/semantic_segmentation.yaml`,
  `config/create_graph.yaml`):
  - CLIP: OpenCLIP `ViT-H-14`, `checkpoints/laion2b_s32b_b79k.bin`, dim 1024
    (`graph.py:90-103`, `constants.py CLIP_DIM`). `ViT-L-14` (768) also wired.
  - SAM: `vit_h` automatic-mask generator, `checkpoints/sam_vit_h_4b8939.pth`
    (`graph.py:114-127`). README documents the download; see
    `.codex/modules.md` (shared SAM target still needs centralizing; local smoke
    fallback is `vit_b`).
  - No object detector and no tracker; masks come from SAM-automatic, naming
    from CLIP-vs-label argmax.
- Memory artifact:
  - Always (both entrypoints): `mask_feats.pt` (per-mask CLIP feats),
    `full_feats.pt`, `objects/pcd_*.ply`, `full_pcd.ply`, `masked_pcd.ply`.
  - HM3DSem only: `graph/{floors,rooms,objects}/*.{ply,json}` with named
    `Object` nodes (`object.py:35-66` stores id, vertices, room_id, name,
    embedding) + `graph/nav_graph/`.
- Track decisions (root-repo evidence):
  - Track 1 (object inventory): `export`. Object-level geometry exists as native
    per-mask PLYs; shared OV labels are derivable via the `evaluate_sem_seg.py` argmax
    over a fixed class list. Needs a thin exporter (no native JSONL writer).
  - Track 2 (object location): `candidate`, not native-ready. The native
    `query_object` is OV CLIP similarity, is HM3DSem-graph-only, and the
    global-search branch is broken (`graph.py:1125`). A non-interactive shared OV
    reader over `mask_feats.pt` is required before this can be `supported`.
  - Track 3 (ScanRefer): `invalid`. No referring-expression / relation /
    attribute resolver; CLIP object similarity is not a grounding resolver.
  - Track 4 (OpenEQA): `invalid`. No native QA/retrieval API; `query_hierarchy`
    is an OV object locator that depends on `OPENAI_KEY` for parsing, not a QA
    interface.
- OV vs formal: native default is open-vocabulary (`HM3DSEM_LABELS`, free-text
  CLIP queries). Per the formal policy these method-native detector/OV override results are
  `module_ablation` only. The formal Track 1/2 entry must use the shared OV route
  (shared OV prompt/evaluation list + same CLIP/SAM checkpoints), recorded as
  `vocabulary_mode=open_vocabulary` in manifest/build log.
- Current conclusion:
  - Present open-vocabulary 3D object/feature-map baseline with a clean shared OV route
    through `evaluate_sem_seg.py`'s fixed-class-list CLIP argmax. The remaining
    work is a non-interactive shared OV exporter/reader over the native feature-map
    artifacts (object table for Track 1, label-constrained query for Track 2);
    the native interactive `query_*` REPL is not usable as-is on the ScanNet
    path and is not a fixed API.

## ConceptGraphs

- Root repo: `/home/robin_wang/concept-graphs` (commit `93277a0`)
- Note on paths: runnable scripts live under `conceptgraph/scripts/` and
  `conceptgraph/slam/`, not a top-level `scripts/`. The README invokes them from
  inside the `conceptgraph/` package dir (e.g. `python scripts/...`,
  `python slam/...`). Citations below use repo-relative paths.

- Native build path (two stages):
  - Stage 1 — detections + per-region CLIP features. Either:
    - `conceptgraph/scripts/generate_gsa_results.py` (legacy GSA path):
      GroundingDINO + SAM, or RAM/Tag2Text tagging, or SAM "segment all"
      (`--class_set none`). Per-frame results saved as
      `gsa_detections_<variant>/*.pkl.gz` plus a
      `gsa_classes_<variant>.json` class list
      (`generate_gsa_results.py:419-611`). Detector/segmenter choices at
      `:111-123`.
    - `conceptgraph/scripts/streamlined_detections.py` (Hydra path):
      YOLO-World `yolov8l-world.pt` + UltraLytics MobileSAM, classes read from
      `cfg.classes_file` via `detection_model.set_classes(classes)`
      (`streamlined_detections.py:45-58, 56`).
  - Stage 2 — 3D fusion: `conceptgraph/slam/cfslam_pipeline_batch.py` loads the
    saved detections, unprojects masks with depth + pose into per-object point
    clouds, associates/merges across frames, and writes the object map to
    `pcd_saves/full_pcd_<gsa_variant>_<save_suffix>.pkl.gz` (and a `_post`
    variant after filter/merge) (`cfslam_pipeline_batch.py:119-135, 362-392`).
    Hydra config `conceptgraph/configs/slam_pipeline/base.yaml`.
    `conceptgraph/slam/streamlined_mapping.py` is a parallel variant of the same
    pipeline.
  - Optional Stage 3 — scene graph: `conceptgraph/scenegraph/build_scenegraph_cfslam.py`
    extracts LLaVA node captions, refines them with GPT-4, builds MST-based
    relation edges, and emits `scene_graph.json` /
    `cfslam_object_relations.json` (`build_scenegraph_cfslam.py:51-59, 530-816`).
    Requires `OPENAI_API_KEY` + LLaVA checkpoint.

- Artifact format / object schema:
  - `pcd_saves/full_pcd_*.pkl.gz` is a gzip-pickled dict
    `{objects, bg_objects, cfg, class_names, class_colors}`
    (`cfslam_pipeline_batch.py:365-371`).
  - `objects` is a serialized `MapObjectList`; per-object dict keys after
    `to_serializable()`: `class_id`, `class_name`, `num_detections`, `conf`,
    `image_idx`, `mask_idx`, `color_path`, `xyxy`, `n_points`, `pixel_area`,
    `inst_color`, `is_background`, plus `clip_ft`, `text_ft`, `pcd_np`,
    `bbox_np`, `pcd_color_np` (`slam/slam_classes.py:118-135`;
    field origins `slam/utils.py:544-565`).
  - Each object therefore has a label (`class_id`/`class_name`, with the string
    list in top-level `class_names`) and 3D geometry (point cloud + oriented
    bbox), so a thin deterministic Track 1 exporter is feasible without
    re-running the method.
  - Reload via `MapObjectList.load_serializable(...)`
    (`slam/slam_classes.py:137-156`); both dict and bare-list formats handled in
    `scripts/visualize_cfslam_results.py:60-85` and
    `scenegraph/build_scenegraph_cfslam.py:87-108`.

- Native query / read capability:
  - Only interactive viewers, no machine-readable query return:
    - `conceptgraph/scripts/visualize_cfslam_results.py:269-310` — press `f`,
      type text, recolors point clouds by CLIP cosine similarity. It computes
      `argmax`/softmax internally (`:291-292`) but only updates Open3D colors;
      nothing is returned or written.
    - `conceptgraph/scripts/visualize_cfslam_interact_llava.py` — Open3D
      point-pick + LLaVA chat REPL (`:93-201`); interactive only.
  - The closest native non-interactive primitive is
    `conceptgraph/scripts/eval_replica_semseg.py:133-140`:
    `object_feats @ class_feats.T → argmax` against a shared OV prompted class-text
    feature matrix. This proves the shared OV label-assignment mechanism exists, but it
    is hardwired to Replica GT point clouds / `REPLICA_CLASSES`
    (`:19-26, 255-302`) and is a semseg scorer, not a reusable object-location
    query entrypoint.
  - No `query_object` / `locate_object` / `answer_question` style API found
    anywhere (`grep` for `def .*query|answer|locate|retrieve` returns only the
    LLaVA `__call__` and `eval_replica`).

- Vocabulary (shared OV route vs method-native OV):
  - Default builds are open-vocabulary: segment-all uses class `"item"`
    (`generate_gsa_results.py:397-398`), RAM/Tag2Text generate open tags
    (`:362-396, 441-478`), GroundingDINO grounds arbitrary phrases.
  - A shared OV route exists natively: YOLO-World
    `set_classes(cfg.classes_file)` in the streamlined path constrains detection
    to a fixed class list (`streamlined_detections.py:56-58`), and CLIP→class
    argmax against a canonical class-text matrix is the eval-time shared OV mechanism
    (`eval_replica_semseg.py:133-140`).
  - Formal Track 1/2 must use the shared-OV-detector-coverable list and
    set `vocabulary_mode=open_vocabulary`. Unrestricted segment-all / RAM / Tag2Text /
    open GroundingDINO results are `module_ablation` only.

- Module stack: SAM `vit_h` (`sam_vit_h_4b8939.pth`) or MobileSAM; GroundingDINO
  `groundingdino_swint_ogc.pth`; RAM/Tag2Text checkpoints; YOLO-World
  `yolov8l-world.pt`; OpenCLIP `ViT-H-14 / laion2b_s32b_b79k`
  (`generate_gsa_results.py:69-78, 320-324, 345`;
  `streamlined_detections.py:45-52`); optional LLaVA + GPT-4 for scene graph.
  Aligns with `.codex/modules.md` (shared SAM `vit_h` target, OpenCLIP `ViT-H-14`).

- Track status:
  - Track 1: `export` — serialized object map has labels + 3D geometry; a thin
    deterministic reader can emit the object table. Formal run requires the shared OV
    variant.
  - Track 2: `candidate` — no native non-interactive object-location query path;
    a shared OV query bridge (reusing the CLIP-text-vs-`clip_ft` argmax primitive
    constrained to the shared OV prompt/evaluation list) must be written and smoke-tested
    before this can be `supported`.
  - Track 3: `invalid` — no referring-expression resolver.
  - Track 4: `invalid` — no native non-interactive QA/retrieval API (the only QA
    is the interactive LLaVA REPL).

- Unresolved blockers:
  - No ScanNet++ dataset loader: `get_dataset` only supports
    icl/replica/azure/scannet/ai2thor/record3d/realsense/multiscan/hm3d/
    hm3d-openeqa (`dataset/datasets_common.py:1168-1191`), and there are zero
    `scannetpp` references in the repo. The canonical scene `036bce3393` must be
    preprocessed into the ScanNet layout (or a new loader added in the
    evaluation repo's method adapter, not the external repo) before a build can
    run.
  - No standalone query/exporter contract yet; Track 2 needs a non-interactive
    shared OV bridge and at least one scene smoke test before claiming `supported`.
  - Scene graph stage depends on external OpenAI GPT-4 + LLaVA checkpoint; not
    needed for Track 1/2 but blocks any relation-based capability.

## DAAAM

- Root repo: `/home/robin_wang/DAAAM`
- Native construction:
  - `scripts/run_pipeline.py:24` imports `HydraPipelineRunner`;
    `scripts/run_pipeline.py:523-539` instantiates it with hydra config,
    labelspace, dataset, and output dir (dataset pipeline, no ROS required).
  - `src/daaam/hydra/runner.py:21-50` defines `HydraPipelineRunner`;
    `runner.py:277-436` is the `run()` loop driving per-frame DSG construction.
  - `src/daaam/hydra/integration.py:185-236`
    `HydraIntegration.process_frame(...)` is the frame-level bridge into
    Hydra/DSG construction.
- Native object inventory (Track 1 evidence):
  - `src/daaam/scene_understanding/utils.py:23-43`
    `retrieve_objects_from_scene_graph(...)` enumerates the DSG OBJECTS layer
    (`scene_graph.get_layer(sdsg.DsgLayers.OBJECTS).nodes`, utils.py:33) and
    converts each node to an `ObjectData`.
  - `src/daaam/scene_understanding/models.py:58-85` `ObjectInfo` carries
    `description` (label) and `position` (3D); `from_scene_graph_node()` reads
    both off the node.
  - Background objects: `src/daaam/scene_graph/services.py:339-360` builds
    `KhronosObjectAttributes` with `position = bg_obj.position_world` and
    `semantic_label = bg_obj.semantic_id`; `scene_graph/models.py`
    `BackgroundObjectData` holds `position_world` + `semantic_label`.
  - So native DSG object nodes carry both a label and a 3D position, and there
    is a native, non-interactive way to list them. Track 1 is `export` (needs a
    thin non-interactive reader that dumps `object_table.jsonl`), not `native`.
- Native query/read:
  - `scripts/demo_query.py` loads a DSG and runs a REPL.
  - `SceneUnderstandingAgent.answer_query(...)`
    (`src/daaam/scene_understanding/services.py:64-88`) answers against a loaded
    scene graph. It is an LLM agent: `services.py:38-40` calls
    `detect_provider` / `create_client`; `services.py:111` uses
    `client.responses.create` (OpenAI) and `services.py:203`
    `run_anthropic_tool_loop` (Anthropic). `providers.py:19-22` routes
    `claude*` to Anthropic else OpenAI.
  - The deterministic, non-LLM scene-graph tools live under
    `src/daaam/scene_understanding/tools/`:
    `get_matching_subjects.py`, `get_objects_in_radius.py`,
    `get_region_information.py`, `get_objects_in_region.py`,
    `get_objects_in_view.py`, `get_agent_trajectory_information.py`,
    `get_robot_location.py`. These return objects with positions directly and
    are the honest basis for a Track 2 fixed API, separate from the LLM loop.
- Memory artifact:
  - Dynamic Scene Graph / Hydra outputs, background-object metadata, semantic
    updates, grounding images/annotations if enabled.
- Current conclusion:
  - Method code is present and has a native scene-graph QA agent. Integration is
    heavier than static object-map baselines because we must standardize dataset
    input, DSG output paths, and evaluator-safe query calls.
- Adapter status:
  - `scripts/methods/daaam/build_memory_smoke.py` implements both package-from-output
    and ScanNet++ raw-build preparation routes without modifying the DAAAM repo.
  - Track 1 package fixed API is `tools/list_objects.py:list_objects` over the
    exported DSG/background object table.
  - Track 2 package fixed API is `tools/query_object.py:query_object` only when
    the builder can export a deterministic DAAAM semantic index from native
    scene-understanding embeddings (`GetMatchingSubjects` /
    `precompute_unified_embeddings` route). If embeddings/features are missing,
    the package writes Track 2 as `invalid`.
- Human-review notes (ambiguous, do not auto-promote):
  - Track 1 vocabulary: DAAAM object/background labels come from DAM/VLM
    grounding (open-vocabulary free-text `description`), not a non-shared detector.
    The adapter preserves `raw_label` and writes canonical `label` by exact/alias
    mapping first, then shared-class semantic projection when native DAAAM
    embeddings are available. Method-native/free-text-only labels should still
    be treated as module/vocabulary ablation when comparing formal runs.
  - Track 2 fixed API decision is now explicit: deterministic native semantic
    index only. The shipped LLM `answer_query` entrypoint remains excluded from
    Track 2 fixed API support.
  - Track 4 stays `candidate`: `answer_query` is a method-native QA agent, but it
    needs an evaluator-safe, non-interactive call path and judge/LLM isolation
    before it can be `supported`.

## Hydra Standalone

- Root repo: `/home/robin_wang/Hydra` (separate repo from DAAAM; confirmed no
  DAAAM code/dependency in the Hydra tree).
- Native construction:
  - Hydra is a standalone real-time 3D scene graph construction system.
  - Python bindings document `hydra run mp3d /path/to/scene`
    (`python/README.md:26-31`); the CLI is
    `python/src/hydra_python/commands/run.py:40-102`, which loads an offline
    `FileDataLoader`, steps the pipeline per frame with
    `(timestamp, translation, rotation, depth, labels, color)`
    (`run.py:82-92`), and saves with `pipeline.save(...)` (`run.py:102`).
  - Expected image dataset layout includes `camera_info.yaml`, RGB images, depth,
    semantic labels, and `poses.csv` (`python/README.md:33-71`).
  - Core C++ has no ROS dependency (`package.xml`, `CMakeLists.txt:17-28` list
    only config_utilities/Eigen/GTSAM/kimera_pgmo/spark_dsg/teaserpp/OpenCV/PCL);
    ROS code moved to a separate Hydra-ROS repo (`README.md:74`). Installation
    environment may still assume ROS2, but the offline run path is the Python
    binding.
- Native object inventory (Track 1 evidence):
  - Object nodes are created at `src/frontend/mesh_segmenter.cpp:308-318`:
    `ObjectNodeAttributes` with `semantic_label = label` (mesh_segmenter.cpp:311)
    placed in the OBJECTS layer; 3D position is the cluster centroid via
    `updateObjectGeometry(...)` (`src/utils/mesh_utilities.cpp`).
  - Object nodes are enumerable from a saved DSG via
    `graph.getLayer(DsgLayers::OBJECTS).nodes()` then
    `node->attributes<ObjectNodeAttributes>()` (pattern in
    `eval/tools/compress_graph.cpp:144-157`).
  - Critical caveat: Hydra does NOT produce semantic labels itself. Labels are
    an external input image (`python/bindings/src/python_sensor_input.cpp:76-77`),
    integrated by `src/reconstruction/semantic_integrator.cpp`; Kimera-Semantics
    was dropped (`README.md:66`) in favor of an external segmenter
    (`semantic_inference`). So Hydra object labels come from a dataset GT mask or
    an external segmenter, not from Hydra.
- Native query/read:
  - `hydra-eval` / `eval/` tooling (`eval/README.md:9-29`) provides timing and
    room/place evaluation only; loop-closure `query_*` (`registration.h`) is
    descriptor matching, not semantic query. No natural-language QA interface.
- Memory artifact:
  - Hydra result directory; DSG serialized as `dsg.json` / `dsg_with_mesh.json`
    (`src/frontend/graph_builder.cpp:189-190`), loadable via
    `spark_dsg::DynamicSceneGraph::load(...)` (`eval/tools/merge_graphs.cpp:52`).
- Current conclusion:
  - This should be treated as a present standalone DSG baseline, separate from
    DAAAM's Hydra integration.
- Human-review notes (ambiguous, do not auto-promote):
  - Track 1 stays `candidate`. Object nodes with labels + 3D positions are
    exportable, so a clean DSG object export is plausible, but: (a) labels are
    supplied externally, so the formal shared OV label/module fairness gate depends on
    which segmenter/label-space we feed Hydra, not on Hydra itself; (b) the run
    path needs a smoke test on one scene before Track 1 is `supported`. Human
    owns whether externally-labeled DSG object nodes count as a native object
    inventory under the vocabulary policy.
  - Track 2/3/4 remain `invalid`: no native NL object-location query, no
    referring resolver, no QA API in the Hydra repo.

## ReMEmbR And Caption/VLM Controls

- Root repo: `/home/robin_wang/remembr`
- Native construction:
  - `scripts/preprocess_captions.py` captions CODa sequences with VILA and writes
    `data/captions/{seq_id}/captions/captions_<captioner>_<seconds>_secs.json`.
  - `MemoryItem` stores caption, time, position, and theta.
  - `MilvusMemory` stores caption memories with text embeddings, position, and
    time indexes.
- Native query/read:
  - `ReMEmbRAgent.query(...)` (`remembr/agents/remembr_agent.py:390-410`) runs a
    LangGraph LLM agent loop (`remembr_agent.py:343-387`) over caption memory
    using tools `retrieve_from_text`, `retrieve_from_position`,
    `retrieve_from_time` (`remembr_agent.py:153-206`).
  - The memory schema is caption/pose/time only: `MemoryItem`
    (`remembr/memory/memory.py:5-9`) has fields `caption`, `time`, `position`,
    `theta` and NO object field. Retrieval returns formatted strings via
    `memory_to_string()` (`remembr/memory/milvus_memory.py:231-250`): "At
    time=..., the robot was at position .... The robot saw: <caption>". There is
    no object inventory and no object-level 3D location output.
  - `scripts/eval.py --model remembr+<llm>` runs the retrieval-agent baseline
    (`remembr/scripts/eval.py:232-236`).
  - `scripts/eval.py --model <llm>` runs a caption-context baseline through
    `NonAgent` (`eval.py:245`); `NonAgent.query`
    (`remembr/agents/non_agent.py:40-91`) fills the prompt with the full caption
    context (`non_agent.py:61`) and calls the LLM once, with no retrieval tools.
  - `scripts/eval.py --model vlm...` routes to `VLMNonAgent(llm_type='gpt-4o')`
    (`eval.py:242`) over `VideoMemory` (`eval.py:154`), which stores raw sampled
    frames as `ImageMemoryItem` (`remembr/memory/video_memory.py:22-27`), not an
    exported memory DB.
- Memory artifact:
  - caption JSON files, Milvus collections, optional `TextMemory` or `VideoMemory`
    for controls, and eval outputs under ReMEmbR's `out/`.
- Current conclusion:
  - ReMEmbR is now present and should be included as a caption/spatio-temporal
    memory baseline. Track 4 (QA/retrieval) is its only native fixed-API track;
    Track 1/2/3 are `invalid` because the memory has no object inventory and no
    object-location output.
  - `LLM with captions` is not missing; it is the ReMEmbR eval `NonAgent`
    control over caption context (`eval.py:245`, `non_agent.py:40-91`). It is a
    no-explicit-memory control and must stay `control-only`, never promoted to an
    object-memory fixed API.
  - `Multi-frame VLM` is also represented in ReMEmbR's eval path
    (`eval.py:242`), but `VLMNonAgent.__init__`
    (`remembr/agents/vlm_non_agent.py:73-84`) has a real bug: the guard reads
    `if 'gpt-4' in 'llm_type':` (the string literal `'llm_type'`, not the
    `llm_type` argument), so it always falls through to `raise
    NotImplementedError` and never sets `self.chain`. The control therefore
    cannot be claimed runnable without a fix/smoke test. It is a raw-frame
    no-memory control and must stay `control-only`.
  - DAAAM contains OC-NaVQA data notes that reference ReMEmbR (`DAAAM/data/`),
    but the caption/VLM control implementation lives in the ReMEmbR repo, not in
    DAAAM. The DAAAM repo itself has no `VLMNonAgent`/caption-control code.

## Missing Or Control Baselines

- No root-level method repo is currently missing from this registry among the
  checked set.
- Multi-frame VLM and LLM-with-captions should remain clearly marked as
  no-explicit-memory controls, even though their code lives under ReMEmbR. Their
  packages must set `manifest.explicit_memory=false` and may never be promoted to
  a spatial-memory fixed API on any track.

## Audit Recommendations (Task 05: DSG / Caption / Controls)

This section records the audit decision for DAAAM, Hydra, ReMEmbR, and the two
controls against the Track 1/2 fixed-API gate. DAAAM and Hydra are treated
separately; controls are never object-memory fixed APIs. All four-track statuses
above stay `candidate`/`export`/`invalid`/`control-only` — none are promoted to
`supported`, which requires human sign-off plus a smoke test.

### DAAAM — object-memory fixed-API adapter implemented, smoke pending

- Native DSG object/background nodes carry labels and 3D positions and are
  enumerable (`scene_understanding/utils.py:23-43`,
  `scene_graph/services.py:339-360`), so DAAAM is a genuine object-memory
  candidate for Track 1, and a Track 2/4 candidate via its native scene-graph
  tools and QA agent.
- Recommendation: **fixed-API candidate, not yet supported.** Track 1 = `export`
  (write a non-interactive object-table reader), Track 2 = `candidate` (needs a
  non-LLM entrypoint over the native tools), Track 4 = `candidate` (needs
  evaluator-safe call + LLM-judge isolation), Track 3 = `invalid`. Human owns:
  (a) the shared OV label mapping for DAM/VLM free-text labels, and
  (b) whether Track 2 uses the deterministic tools rather than the LLM loop.

### Hydra standalone — Track 1 DSG candidate, kept separate from DAAAM

- Hydra is its own perception repo; do not conflate with DAAAM's Hydra
  integration. DSG OBJECTS nodes carry `semantic_label` + centroid and are
  enumerable, but Hydra does not generate labels — they are external input.
- Recommendation: **Track 1 object-memory candidate only.** Track 1 =
  `candidate` (clean DSG object export + one-scene smoke test), Track 2/3/4 =
  `invalid` (no NL query, no referring resolver, no QA API). Human owns whether
  externally-labeled DSG object nodes satisfy the shared OV label/module fairness
  gate, and whether Hydra is evaluated on raw DSG vs through a downstream QA
  layer.

### ReMEmbR — agentic / Track 4 only, not an object-memory fixed API

- Caption/pose/time memory only; no object inventory or object-location output
  (`memory/memory.py:5-9`). Its one native fixed-API track is Track 4 via the
  `ReMEmbRAgent.query` retrieval agent.
- Recommendation: **Track 4 fixed-API candidate; Track 1/2/3 `invalid`; primary
  comparison path is agentic.** Do not export an object table for ReMEmbR and do
  not promote it to a Track 1/2 object-memory baseline. Smoke test the
  `remembr+<llm>` path before marking Track 4 `supported`.

- Task 19 finalization (Track 1/2 fixed-API outcome): confirmed and finalized as
  `invalid`. Root evidence (read-only, root repo only):
  - Memory schema is object-free. `MemoryItem`
    (`/home/robin_wang/remembr/remembr/memory/memory.py:5-9`) has exactly
    `caption`, `time`, `position`, `theta` and no object/label/bbox/class field.
    The native store mirrors this: the Milvus `CollectionSchema`
    (`/home/robin_wang/remembr/remembr/memory/milvus_memory.py:37-45`) declares
    `id`, `text_embedding`, `position` (robot pose, dim 3), `theta`, `time`, and
    `caption` — there is no object table to enumerate, so Track 1 cannot honestly
    export label+3D object records.
  - Read API is caption retrieval, not object location. The only readers are
    `search_by_text`/`search_by_position`/`search_by_time`
    (`milvus_memory.py:173-227`); each returns `memory_to_string(...)`
    (`milvus_memory.py:231-250`) which formats `"At time=..., the robot was at an
    average position of ... The robot saw the following: <caption>"`. `position`
    is the robot's averaged pose, not an object's 3D position, and the output is
    free text, so there is no deterministic Track 2 object-location query/read.
  - Per the implementation rules and `.codex/memory_package_spec.md` Track 1/2
    contracts, this is exactly the case to declare Track 1/2 `invalid` rather than
    wrap captions in a generic text→location LLM. No object table was exported.
  - Minimal invalid declaration is a generated artifact, not committed to git:
    run `scripts/methods/remembr/build_invalid_declaration.py` to write it under
    the gitignored `memories/remembr/oc-navqa/sequence_0/<run-id>/` (validates
    clean; `manifest.method.family = caption_memory`, `explicit_memory = true`,
    Track 1/2/3 `invalid`, Track 4 `invalid` for the fixed API with the native
    QA path recorded as agentic in `schema.md`).
- Agentic / future-track note (kept separate from the fixed-API outcome): the
  native `ReMEmbRAgent.query` retrieval agent
  (`/home/robin_wang/remembr/remembr/agents/remembr_agent.py:390`) over the
  `retrieve_from_text`/`retrieve_from_position`/`retrieve_from_time` tools
  (`remembr_agent.py:168,183,199`) makes Track 4 / OC-NaVQA temporal QA the first
  track ReMEmbR should enter, and the package's primary comparison path is
  agentic full-access, not a fixed object API. Track 4 stays `candidate` until a
  non-interactive `remembr+<llm>` smoke is run; this task does not promote it.

### Multi-frame VLM control — control-only

- `VLMNonAgent` over raw sampled frames (`VideoMemory`), no exported memory.
- Recommendation: **control-only on all tracks, `explicit_memory=false`.** Not a
  spatial-memory fixed-API baseline. Also blocked by a real constructor bug
  (`if 'gpt-4' in 'llm_type':` in `agents/vlm_non_agent.py:77` tests a string
  literal, so it always raises `NotImplementedError`); fix + smoke test before
  claiming the control is runnable. Tracking only — do not edit the external repo
  as part of this audit PR.

### LLM-with-captions control — control-only

- `NonAgent` answers from full caption context with no retrieval tools and no
  object memory.
- Recommendation: **control-only on all tracks, `explicit_memory=false`.** Not a
  spatial-memory fixed-API baseline; keep as a no-explicit-memory caption
  ablation.

### Summary classification

- Object-memory fixed-API candidates / adapters: DAAAM Track 1 adapter is
  implemented and Track 2 is supported per package only when the deterministic
  native semantic index builds; Hydra remains Track 1 only.
- Agentic-only / Track-4-native: ReMEmbR.
- Control-only (`explicit_memory=false`, never fixed API): Multi-frame VLM,
  LLM-with-captions.
- Deferred decisions for the human: DAAAM formal shared-OV label projection
  quality after smoke; Hydra external-label fairness + evaluation surface;
  promotion of any remaining `candidate` to `supported` after smoke tests.

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
