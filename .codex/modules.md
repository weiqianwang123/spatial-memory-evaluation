# Shared Modules Registry

Last updated: 2026-06-16

本文件记录各 baseline 方法会用到的 detector、segmenter、vision-language
encoder、captioner、LLM 和 vector store。目标是让不同方法在可比实验中共享相同
模块和 checkpoint；如果多个方法用到同类模块，默认使用能力最强且所有相关方法都
能接入的同一版本。

Runtime code lives in `spatial_memory_evaluation/shared_modules/`. External
method repos must not be edited to load these modules directly. Instead,
method-specific scripts under `scripts/methods/` read the shared registry and
translate it into that method's native CLI/Hydra/config overrides.

## Rules

- 相同功能模块必须统一版本、checkpoint、预处理和 device policy。
- 外部方法需要 detector、SAM、CLIP、class list 等模块时，必须通过
  `spatial_memory_evaluation/shared_modules/` 读取配置。
- `scripts/methods/<method>/` 可以有内部 adapter，把 shared module registry
  转换成外部 repo 的参数；不要修改外部 repo 源码。
- 同一 scene 内用于检测/命名 object 的 label vocabulary 必须统一。默认使用
  `spatial_memory_evaluation/assets/class_lists/detector_coverable.txt`，它由
  `spatial_memory_evaluation/common/labels.py` 的
  `DEFAULT_DETECTOR_COVERABLE_LABELS` 生成并校验。
- Track 1/2 formal eval 必须是 closed-vocabulary setup。DualMap、HOV-SG、
  ConceptGraphs 等 open-vocabulary 方法必须提供 CV eval variant；unrestricted
  OV 结果只能作为 `ov_ablation`。
- checkpoint 统一存到一个共享目录，不放在各方法 repo 内部。建议路径：
  `/data/mondo-training-dataset/semantic_mapping/modules/<module>/<version>/`。
- exporter/package 必须在 `manifest.json` 或 `build_log.json` 记录实际使用的模块
  名称、版本、checkpoint 路径和关键参数。
- exporter 必须记录 detector class list path；如果 class list 与 canonical list
  不一致，默认不允许作为 formal fair-comparison run。
- 如果方法内部使用 open-vocabulary detector/query model，formal run 仍必须把
  query/label space 约束到 canonical closed vocabulary，并在 manifest/build log
  记录 `vocabulary_mode=closed`。
- 如果某方法原生只能用较弱版本，需要在该方法 package 中声明 override reason。
- 如果 checkpoint 不存在，模块状态为 `missing`，不能默默换成别的模型。
- 对 shared module 的升级要整体重跑受影响方法，不能只更新单个方法。

## Priority Policy

同类模块默认选择：

1. 所有相关方法都能调用的最强模型。
2. 如果最强模型无法在某方法/环境运行，则使用所有方法共同可运行的最强模型。
3. 如果没有共同版本，保留 method-native 配置，但必须在结果中标记
   `module_override`，并在报告中分开比较。

## Canonical Storage Layout

```text
/data/mondo-training-dataset/semantic_mapping/modules/
  sam/
    vit_h/sam_vit_h_4b8939.pth
    vit_b/sam_b.pt
  yolo/
    yolo11/<checkpoint>
    yolo_world/<checkpoint>
  openclip/
    ViT-H-14/laion2b_s32b_b79k/
    ViT-B-32/laion2b_s34b_b79k/
  groundingdino/
  vila/
  llava/
  embeddings/
  llm/
```

Current local reality may differ; move or symlink checkpoints into this layout
before formal runs.

## Module Table

| Module | Role | Methods using it | Preferred shared version | Current known checkpoint / source | Status | Notes |
|---|---|---|---|---|---|---|
| SAM | segmentation / mask proposal | HOV-SG, DualMap, ConceptGraphs, DAAAM | `vit_h` if all methods can run it | target: `sam_vit_h_4b8939.pth` | missing | HOV-SG default config points to `checkpoints/sam_vit_h_4b8939.pth`, but it was not found locally/NAS common paths. |
| SAM | segmentation / mask proposal | HOV-SG, DualMap, ConceptGraphs, DAAAM | fallback common version: `vit_b` | `/home/robin_wang/DualMap/sam_b.pt` | present | Historical HOV-SG run used `models.sam.type=vit_b` with this checkpoint. Use this as smoke fallback until `vit_h` is centralized. |
| FastSAM | fast segmentation | DualMap optional, DAAAM optional | TBD | not centralized | unverified | Only use if SAM is too slow or method-native path requires it; record override. |
| SAM2 | video/image segmentation | DAAAM optional | TBD | not centralized | unverified | Do not mix with SAM results unless explicitly running an ablation. |
| Detector class list | detection vocabulary / label naming | HOV-SG, DualMap, future detector-backed methods | canonical detector-coverable list | `spatial_memory_evaluation/assets/class_lists/detector_coverable.txt` | present | Must stay exactly synchronized with `DEFAULT_DETECTOR_COVERABLE_LABELS`; check with `python scripts/package/sync_detector_class_list.py --check`. |
| YOLO / Ultralytics | object detection | ClawS SpatialRAG, DualMap | YOLO11 if both can run | not centralized | unverified | ClawS references YOLO11/Ultralytics; DualMap uses YOLO/YOLO-World variants. Need exact checkpoint discovery. |
| YOLO-World | open-vocabulary detection | DualMap, ConceptGraphs streamlined path | strongest common YOLO-World checkpoint | `/home/robin_wang/DualMap/yolov8s-world.pt` for current smoke | present local, not centralized | If used as detector in multiple methods, share exact checkpoint and the canonical detector class list. |
| GroundingDINO | open-vocabulary grounding | ConceptGraphs legacy path | strongest common checkpoint | not centralized | unverified | Should be fixed before ConceptGraphs exporter is claimed reproducible. |
| OpenCLIP | vision-language feature encoder | HOV-SG, DualMap, ConceptGraphs, DAAAM | `ViT-H-14` if all methods can run | HOV-SG default: `laion2b_s32b_b79k`; smoke fallback: `ViT-B-32/laion2b_s34b_b79k` | partial | For fair CLIP-based object query, model type, pretrained tag, templates, and normalization must match. |
| MobileCLIP | lightweight vision-language encoder | DualMap optional | TBD | not centralized | unverified | Treat as method override unless all methods adopt it. |
| ByteTrack | object tracking | ClawS SpatialRAG | method-native | package dependency | method-specific | Tracking module is not shared yet unless another method uses ByteTrack. |
| BotSort | object tracking | DAAAM | method-native | package dependency | method-specific | Tracking module is method-specific unless shared later. |
| VLM description / verification | object description, QA, captioning | ClawS optional, DAAAM, ConceptGraphs optional, ReMEmbR controls | TBD | model API or local VLM not centralized | unverified | Must separate memory construction VLM from evaluator/LLM judge to avoid leakage/confounds. |
| VILA | captioning | ReMEmbR | ReMEmbR-native first | not centralized | unverified | If used outside ReMEmbR, centralize checkpoint and caption prompt. |
| LLaVA | visual QA/captioning | ConceptGraphs optional | TBD | not centralized | unverified | Only use as method-native component when explicitly documented. |
| Text embeddings | retrieval embeddings | ClawS, ReMEmbR | TBD | ReMEmbR uses mixedbread; ClawS uses vector embedding with sqlite-vec | partial | Need exact model names and embedding dimensions before fair retrieval comparison. |
| Milvus | vector DB | ReMEmbR | method-native infra | local service/config TBD | method-specific | Store config and collection schema in package build log. |
| sqlite-vec | vector DB / retrieval | ClawS SpatialRAG | method-native infra | Python package / SQLite extension | method-specific | Record extension version if possible. |
| Hydra / Spark-DSG | 3D dynamic scene graph | DAAAM, Hydra standalone | shared build if possible | `/home/robin_wang/Hydra` | present | DAAAM integration and standalone Hydra should record exact commit/build. |
| LLM judge | scoring | evaluator only | fixed evaluator model | default TBD | decision needed | Must be separate from method memory construction modules. |
| Agent backend | agentic eval | evaluator only | Claude Code default | external tool/API | decision: default Claude Code | Record model/version and sandbox policy in result metadata. |

## Method To Module Map

| Method | Detector | Segmenter | Feature / Embedding | Caption / VLM | Tracking | Storage / Graph |
|---|---|---|---|---|---|---|
| ClawS SpatialRAG | YOLO11 / Ultralytics | none or method-native masks if enabled | vector embedding + sqlite-vec | optional VLM description / verification | ByteTrack | SQLite + sqlite-vec spatial DB |
| DualMap | YOLO / YOLO-World | SAM, optional FastSAM | OpenCLIP or MobileCLIP | none by default | none found | local/global object maps |
| HOV-SG | none | SAM automatic masks | OpenCLIP | none by default | none | object point clouds + hierarchy graph |
| ConceptGraphs | GroundingDINO/RAM/Tag2Text or YOLO-World streamlined | SAM / MobileSAM | OpenCLIP | optional LLaVA captions | none | object map + optional scene graph JSON |
| DAAAM | DAM / VLM grounding stack | FastSAM/SAM/SAM2 | CLIP ReID/features | VLM grounding / QA agent | BotSort | Hydra / Dynamic Scene Graph |
| Hydra standalone | semantic labels from dataset/model outputs | method-dependent | DSG semantic layer | none found | built-in spatial pipeline | Hydra / Spark-DSG |
| ReMEmbR | none | none | mixedbread text embeddings | VILA captions; LLM/VLM controls | none | Milvus caption memory |
| Multi-frame VLM control | none | none | none | VLM over raw frames | none | raw frame control |
| LLM with captions control | none | none | optional text context only | LLM over captions | none | caption context control |

## Current Decisions

- `spatial_memory_evaluation/shared_modules/registry.py` is the runtime source
  of truth for smoke/formal module profiles.
- `scripts/methods/shared_modules.py` is the current internal adapter layer.
  It injects registry values into HOV-SG and DualMap runners without editing
  those external repos.
- HOV-SG and DualMap smoke now default to the same detector/label vocabulary:
  `spatial_memory_evaluation/assets/class_lists/detector_coverable.txt`.
- Track 1/2 formal runs are closed-vocabulary runs. OV methods such as DualMap,
  HOV-SG, and ConceptGraphs must declare `vocabulary_mode=closed` for main
  results; unrestricted open-vocabulary results are recorded only as
  `ov_ablation`.
- HOV-SG smoke defaults to `SAM vit_b` with
  `/home/robin_wang/DualMap/sam_b.pt`, because this exact config appears in
  prior successful HOV-SG Hydra runs.
- Formal shared SAM target remains `SAM vit_h` with `sam_vit_h_4b8939.pth`,
  but the checkpoint still needs to be centralized.
- Do not compare methods using different SAM/CLIP/YOLO checkpoints as if they
  were pure memory-method differences; mark those as module ablations or rerun
  with shared modules.
- Do not compare methods using different detector class lists as fair object
  memory methods. Different vocabularies are a detector-vocabulary ablation.

## Open Checks

1. Locate or download `sam_vit_h_4b8939.pth` and place it under the canonical
   shared module directory.
2. Centralize `/home/robin_wang/DualMap/yolov8s-world.pt` under the canonical
   modules directory or replace it with a stronger shared YOLO-World checkpoint.
3. Discover exact OpenCLIP checkpoints and prompt templates used by HOV-SG,
   DualMap, ConceptGraphs, and DAAAM.
4. Decide whether smoke runs use weaker shared modules (`SAM vit_b`,
   `OpenCLIP ViT-B-32`) while formal runs use stronger modules.
5. Record module metadata in every memory package manifest/build log.
