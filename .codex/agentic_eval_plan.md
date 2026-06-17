# Agentic Spatial Memory Evaluation 落地计划

本文档把 `agentic_eval.md` 中的研究构想收紧成当前 repo 的工程执行计划。
`agentic_eval.md` 保留研究愿景；本文档只描述接下来按什么顺序实现、验证和拆分任务。

核心原则：

- 所有 evaluation 都消费导出的 memory package，不直接假设 method repo 有统一接口。
- fixed API 只评估方法原生 memory 或 root repo 已有能力能诚实支持的接口。
- 不支持 fixed API 的方法必须输出 `status: invalid`，不能用 LLM wrapper、fallback 或手写转换假装支持。
- Track 1/2 的正式主评测统一使用 shared strongest open-vocabulary detector setup；closed-detector 或 method-native detector 变体只能作为 module ablation。
- agentic evaluation 是不同 memory 形式之间的主要公平比较路径：agent 在 sandbox 中
  full access 读取 memory package、evaluation adapter code、shared module code 和
  方法 root repo 的原始 source code。

## 核心目标

本项目不是提出新的 spatial memory 方法，而是提出一个 benchmark 和 evaluation
protocol，用来评估 spatial memory 是否能作为 embodied agent 可访问、可理解、
可查询、可验证、可组合的外部认知资源。

benchmark 需要同时回答：

- memory 本身是否准确、紧凑、低冗余；
- memory 是否提供稳定、可复现、可比较的 fixed API；
- 在 fixed API 不足时，agent 是否能通过 full access 自主读取和使用 memory；
- 成功或失败来自 memory 缺失、fixed API 缺失、schema 难懂、证据不足，还是 agent 推理错误；
- 不同方法使用的 detector、segmenter、feature encoder、vocabulary 是否足够公平。

## 全局约定

- Memory 输出路径：`memories/<method>/<dataset>/<scene-or-episode>/<run-id>/`。
- Result 输出路径：`results/<method>/<evaluation>/<timestamp>/`。
- 所有方法先导出 minimal memory package，再进入任何 evaluator。
- 每个 package 必须通过 validator，并明确声明 Track 1-4 的 fixed API support 或 invalid reason。
- agentic sandbox 默认 package + source-code full access；raw frames / detector/module overrides
  仍然是单独 ablation。
- spatial-temporal / long-horizon 先不作为当前实现重点。

## Track 定义

### Track 1：Memory Construction / Object Inventory

评估 memory 构建质量。第一版只评估 shared-OV-detector-coverable object inventory：

- object recall；
- false memory ratio；
- duplicate / redundancy ratio；
- localization error；
- memory size、peak size、构建时间和 time per frame。

fixed API requirement：

- `list_objects()` 或等价 exported object table；
- object 至少包含 `object_id`、`label`、`position_3d`、`evidence`；
- label 必须来自 shared OV detector prompt/evaluation list，或提供可映射到该 list 的 normalized label。

如果 package 没有 object-level memory 或 object table，fixed API 结果为 `invalid`。

### Track 2：Basic Object Location Query

评估 memory 是否能回答简单 object lookup/location query，例如 “where is the monitor?”。

fixed API requirement：

- `query_object(query)` / `locate_object(query)` / 等价声明；
- evaluator 传入结构化字段，例如 `target_label` 和自然语言 query；
- package 优先使用 `target_label` 做 exact label / normalized-label 查询；
- 返回 candidate objects、位置、score、evidence。

如果 memory 只有原始图像、caption context 或 DSG，但没有声明可比较的 object
location query API，fixed API 结果为 `invalid`。agentic 版本仍可在 sandbox 中尝试使用 package。

### Track 3：ScanRefer Fine-Grained Referring Query

适配 ScanRefer。评估细粒度 referring expression grounding：

- target object top-1 / top-k；
- 3D IoU；
- center distance；
- attribute / relation evidence。

fixed API requirement：

- `resolve_referring_expression(query)` 或能力等价的 object query API；
- 支持 attribute / relation evidence，或明确声明不支持。

不支持 referring query 的方法 fixed API 结果为 `invalid`。agentic 版本可读取
object table、crops、keyframes、scene graph 和 schema 自主判断。

### Track 4：OpenEQA General Spatial QA

适配 OpenEQA。评估开放空间问答：

- short answer；
- evidence；
- LLM Match / exact category metrics；
- agent trace 和 memory usage。

fixed API requirement：

- 方法原生 memory package 声明 `answer_question(question)`、`get_memory_text`
  或等价 QA/retrieval API；
- 不允许用通用 object-table-to-LLM 答案器把不支持 QA 的 memory 包成支持。

如果方法没有原生 QA/retrieval fixed API，fixed API 结果为 `invalid`。agentic
版本仍可通过 sandbox 使用 memory package。

### Deferred：Spatial-Temporal / Long-Horizon QA

包括 OC-NaVQA、ReMEmbR temporal questions、duration、before/after、last seen
等。先只在 registry 里记录能力，不进入当前实现主线。

## Minimal Memory Package

目标结构：

```text
memories/<method>/<dataset>/<scene-or-episode>/<run-id>/
  manifest.json
  capabilities.json
  schema.md
  memory/
  evidence/
  raw_links/
  tools/
  build_log.json
```

`manifest.json` 必须包含：

- `method`：方法名、repo path、版本或 commit hash，如果可获得；
- `dataset`：dataset、split、scene id 或 episode id；
- `input`：RGB-D、poses、intrinsics、timestamps、frame count；
- `vocabulary`：`vocabulary_mode`、prompt/evaluation list path、detector labels、module ablation 标记；
- `modules`：detector、segmenter、feature encoder、captioner、LLM 等模块版本/checkpoint/preprocess；
- `memory_artifacts`：native map、DB、object table、scene graph、captions、keyframes、crops；
- `schema`：坐标系、object id、relations、confidence、time 字段说明；
- `build`：构建命令、config、环境、runtime、time per frame、memory size、peak size；
- `allowed_access`：确认不包含 GT annotations、benchmark answers、test labels 和为测试问题手写的规则。

`capabilities.json` 必须包含：

```json
{
  "fixed_api": {
    "track1_memory_construction": {
      "status": "supported | invalid",
      "entrypoint": "memory/object_table.jsonl",
      "reason": ""
    },
    "track2_object_location": {
      "status": "supported | invalid",
      "entrypoint": "tools/query_object.py",
      "reason": ""
    },
    "track3_scanrefer": {
      "status": "supported | invalid",
      "entrypoint": "",
      "reason": "no referring-expression resolver"
    },
    "track4_openeqa": {
      "status": "supported | invalid",
      "entrypoint": "",
      "reason": "no native QA API"
    }
  },
  "agent_access": {
    "read_manifest": true,
    "read_schema": true,
    "read_memory_artifacts": true,
    "read_evidence": true,
    "read_raw_links": false,
    "run_package_tools": false
  }
}
```

最低 artifact 要求：

- object-map methods：导出 `memory/object_table.jsonl`，必要时保留 native map。
- scene-graph methods：导出 `memory/graph_nodes.jsonl`、`memory/graph_edges.jsonl`
  或 native DSG/HOV-SG/ConceptGraphs graph，并写清 schema。
- caption-memory methods：导出 `memory/captions.jsonl`，包含 caption、time、
  position、embedding/source。
- no-explicit-memory controls：可以导出 sampled frames 或 captions package，但
  `manifest.explicit_memory` 必须为 `false`。

## Fixed API Eligibility Gate

fixed API 是严格能力判定，不是 evaluator 侧的包装能力。

一个方法只有满足以下条件，才能在对应 track 写 `supported`：

- root repo 或 native memory artifact 已经提供对应信息或稳定查询入口；
- package entrypoint 只是薄封装、格式转换或非交互化，不改变方法能力；
- 不调用 evaluator 私有 GT、不使用 benchmark-specific rules；
- 不把 object table 临时交给通用 LLM 生成 fixed API 答案；
- smoke test 能在至少一个 scene 上复现。

如果不满足，写：

- `status: invalid`
- `reason_code: unsupported_fixed_api`
- `message` 说明缺失的 native 能力

所有 fixed API 支持判断都必须同步到 `.codex/baseline_registry.md`，并给出 root repo 证据路径。

## Module And Vocabulary Fairness

Track 1/2 的主结果只比较 shared strongest open-vocabulary detector setup。

规则：

- repo 内单独维护 `spatial_memory_evaluation/shared_modules/`，它是 detector、
  segmenter、feature encoder、OV prompt/evaluation label list 和 checkpoint 的唯一 registry。
- 外部方法需要 detector/SAM/CLIP 等模块时，必须通过 `scripts/methods/` 内部
  adapter 从 shared modules 读取配置，再翻译成该方法原生 CLI/Hydra override。
- 不修改外部方法 repo 自己的源码；所有接入差异留在 evaluation repo 的 method
  adapter 中。
- 同类模块必须统一版本、checkpoint、preprocess 和 device policy。
- 同一 scene 内用于检测/命名 object 的 OV detector、prompt/evaluation label list、label normalization 必须统一。
- 默认 prompt/evaluation label list 是 `spatial_memory_evaluation/assets/class_lists/detector_coverable.txt`；它用于 Track 1/2 的 detector-coverable split 和 OV detector prompts，不再代表 closed-detector policy。
- detector-backed object memory 使用同一个 strongest shared OV detector。formal target 是 YOLO-World-L (`yolov8l-world.pt`)；当前本地只找到 YOLO-World-S (`/home/robin_wang/DualMap/yolov8s-world.pt`)，只作为 smoke fallback。
- DualMap、ConceptGraphs 等 detector-backed 方法正式评测必须走 shared OV detector route；HOV-SG 这类无 detector 的方法必须记录其 SAM+CLIP open-vocabulary prompt/label route。
- query/label space 不再强行 closed；必须记录 shared OV prompt/evaluation list、raw label、normalized label 和 detector checkpoint。
- closed-detector、method-native detector/checkpoint、不同 prompt list 的结果可以保留为 `module_ablation`，不能混入 Track 1/2 主表。
- 任何 module override 都必须写入 manifest/build log，并在 report 中单独标记。

需要在 `manifest.json` 或 `build_log.json` 记录：

- `vocabulary_mode`: `open_vocabulary | prompted_open_vocabulary | closed_ablation | module_ablation`
- `class_list_path`
- `shared_module_profile`
- `detector_name`
- `detector_checkpoint`
- `segmenter_name`
- `segmenter_checkpoint`
- `feature_encoder`
- `preprocess`
- `module_override_reason`

## Memory Form Fairness

不同方法的 memory 形式不强行统一。object table、native map、scene graph、caption
DB、Milvus collection、SQLite DB 都可以作为 native memory artifact 保留。

公平性来自三层约束：

- 构建阶段共享相同输入、frame sampling、OV detector/prompt list/label normalization 和可比较模块；
- fixed API 只在方法原生支持的 track 上比较 deterministic 能力；
- agentic eval 给不同 memory 形式同样的 sandbox 权限，让 agent 读取 native artifacts。

Track 1/2 fixed API 可能需要导出 object table，但这个 object table 是评测用的最小可读视图，
不是要求所有方法把内部 memory 变成同一种形式。memory size 主指标必须使用原始方法 native
artifact size；package wrapper size 单独报告。

## Evidence Semantics

Evidence 是 provenance、debug 和 agent grounding，不是额外答案来源。

允许的 evidence：

- `frame_id` / timestamp；
- crop 或 keyframe；
- object observation；
- 3D point cloud / bbox / mask reference；
- native artifact path；
- feature id 或 graph node id。

约束：

- evidence 必须在 memory build 阶段自然产生，不能为某个 query 临时生成；
- evidence 不能包含 GT answer、benchmark label、hand-written query rule；
- fixed API 可以返回 evidence；
- agentic eval 可以读取 evidence；
- report 记录 evidence 是否存在、是否被使用，以及 evidence 缺失造成的失败。

## Build Accounting

每个 memory build 必须记录：

- `frame_count`
- `build_runtime_seconds`
- `time_per_frame_seconds`
- `native_memory_size_bytes`
- `memory_artifact_size_bytes`
- `package_size_bytes`
- `peak_ram_bytes`
- `peak_vram_bytes`

规则：

- `native_memory_size_bytes` 是方法原始输出 artifact 的大小，是 memory size 主指标。
- `package_size_bytes` 是 wrapper/package 总大小，单独报告。
- `memory_artifact_size_bytes` 是 package 内 `memory/` 目录大小，用于排查导出膨胀。
- `peak_ram_bytes` 和 `peak_vram_bytes` 尽量测量；无法可靠测量时写 `null` 和 reason。
- `time_per_frame_seconds = build_runtime_seconds / frame_count`，frame_count 为 0 时写 `null`。

## Invalid Result Schema

所有 fixed API evaluator 都要支持 invalid result：

```json
{
  "status": "invalid",
  "reason_code": "unsupported_fixed_api",
  "required_api": "track4_openeqa.answer_question",
  "method": "dualmap",
  "package_path": "memories/dualmap/...",
  "message": "DualMap package does not declare a native OpenEQA fixed API."
}
```

invalid 是正式结果，不是程序错误。程序错误用 `status: error`，并附 traceback 或 log path。

## 执行顺序

### Phase 0：Baseline Capability And Module Audit

目标：先确认每个 baseline 的 native memory、native API、module stack 和 shared OV detector route 可行性。

方法：

- ClawS SpatialRAG
- DualMap
- HOV-SG
- ConceptGraphs
- DAAAM
- Hydra standalone
- ReMEmbR
- Multi-frame VLM control
- LLM-with-captions control

交付物：

- 更新 `.codex/baseline_registry.md` 的 fixed API support matrix。
- 更新 `.codex/modules.md` 的 shared module / checkpoint / shared OV detector route 决策。
- 更新 `spatial_memory_evaluation/shared_modules/` 的 registry 和 method profile。
- 更新 `scripts/methods/` 内部 adapter，把 shared modules 转换为外部方法运行参数。
- 每个方法列出 root repo 证据路径，而不是引用 evaluation adapters。

Definition of Done：

- 每个 method 的 Track 1/2 fixed API 状态为 `supported`、`candidate` 或 `invalid`，且有证据。
- 每个 detector-backed 方法都有 formal shared OV detector route 或明确 invalid reason。
- 共享 detector/class list/checkpoint 策略明确。

### Phase 1：Track 1/2 Fixed API Adaptation

目标：先建立 deterministic fixed API baseline。所有 evaluator 只读 package，不直接进入方法 repo 猜接口。

Track 1 fixed API：

- 输入：memory package path、scene/dataset GT object inventory。
- 读取：`capabilities.fixed_api.track1_memory_construction`。
- 如果 supported，读取 object table / graph node export。
- 如果 invalid，写 invalid result。

Track 2 fixed API：

- 输入：memory package path、object-location query dataset。
- 读取：`capabilities.fixed_api.track2_object_location`。
- 如果 supported，调用 package 内声明的 query entrypoint。
- evaluator 传入 `target_label`，query tool 优先 exact label / normalized-label match。
- 如果 invalid，写 invalid result。

优先方法：

- ClawS：Track 1/2 fixed API first。
- HOV-SG：SAM+CLIP shared open-vocabulary prompt route 后 Track 1/2。
- DualMap：shared OV detector route 后 Track 1；Track 2 只有 native query bridge 成立才支持。
- ConceptGraphs：shared OV detector route 后 Track 1；Track 2 只有 native query bridge 成立才支持。
- DAAAM / Hydra：先探索 DSG object export；不清楚前不写 supported。
- ReMEmbR / controls：Track 1/2 fixed API 默认 invalid/control-only。

Definition of Done：

- 所有 methods 都有 Track 1/2 fixed API result：metrics 或 invalid。
- Track 1 summary 包含 memory size、build runtime、time per frame。
- Track 2 summary 包含 query speed。
- detector/module override 结果不进入主表。

### Phase 2：Track 1/2 Agentic Adaptation

目标：在 fixed API 之外测试 agent 是否能自主理解和使用 memory package。

Agent setup：

- 默认 Claude Code；本地 Bedrock 命令记录在 task/doc 中。
- 每个 run 建 sandbox，复制 memory package。
- 同时复制 evaluation repo 内的 method adapter code，例如 `scripts/methods/<method>/`。
- 同时复制 shared module registry/adapter code，例如 `spatial_memory_evaluation/shared_modules/`
  和 `scripts/methods/shared_modules.py`。
- 同时复制 manifest 中 `method.repo_path` 指向的外部方法 root repo 原始 source code。
- 复制 root repo code 时只排除 `.git`、缓存、数据、checkpoint、结果等重型/泄漏文件；
  方法源码、配置、脚本、schema、README 必须保留。
- GT answers 不进入 sandbox。
- prompt 必须明确告诉 agent：可以自己设计临时接口、查询脚本或 parser 来和 memory
  交互，不局限于 package fixed API。
- raw frames、detector/module overrides、GT、外部路径和外部网络仍然禁止，除非显式 ablation。

Agent output：

- `answer`
- `evidence`
- `trace` 或 compact reasoning summary
- `used_memory`
- `used_raw_input`
- `used_code`
- `failure_mode`
- `cost` / latency
- `temporary_interfaces`

Definition of Done：

- Track 1/2 agentic evaluator 可批量跑所有 package。
- 输出能区分 memory 缺失、schema 不清、agent 未找到 artifact、推理错误。
- 每个 run 生成 summary、details 和 markdown report。
- sandbox 中可以看到 memory package、adapter code、shared_modules code 和方法 root
  repo source code。

### Phase 3：Track 3 ScanRefer

目标：Track 1/2 稳定后适配 ScanRefer，同时支持 fixed API 和 agentic。

Fixed API：

- 读取 `capabilities.fixed_api.track3_scanrefer`。
- supported：调用 package 声明的 referring resolver。
- invalid：写 `status: invalid`。

Agentic：

- agent 读取 memory package、schema、object table、scene graph、crops/keyframes。
- agent 输出 target object candidate、evidence、reasoning summary。

Definition of Done：

- ScanRefer subset 可复现。
- 每个 method 都有 fixed result：metrics 或 invalid。
- 至少一个 method 跑通 agentic ScanRefer。

### Phase 4：Track 4 OpenEQA

目标：把 OpenEQA general spatial QA 接进同一套 memory package pipeline，同时跑 fixed API 和 agentic。

Fixed API：

- 读取 `capabilities.fixed_api.track4_openeqa`。
- supported：调用方法原生 QA/retrieval API。
- invalid：写 `status: invalid`。
- 不引入通用 object-table-to-LLM 答案器作为方法能力。

Agentic：

- agent 从 package 自主读取 memory、schema、evidence、allowed raw links。
- 输出 short answer、evidence、trace。
- scoring 用 LLM Match / exact category / evidence audit。

Definition of Done：

- 不支持 fixed QA API 的方法显式 invalid。
- agentic 输出包含 evidence 和 trace，不只是 answer string。
- LLM judge 与 memory construction LLM 明确隔离。

### Phase 5：Spatial-Temporal Deferred

当前暂时不实现 spatial-temporal track。只保留 design notes：

- OC-NaVQA / ReMEmbR temporal questions；
- duration / before-after / last seen；
- long-horizon captions / trajectories；
- temporal leakage policy；
- temporal evidence scoring。

恢复条件：

- Track 1-4 的 package、fixed API、agentic sandbox 都稳定后；
- ReMEmbR / DAAAM / future temporal methods 的 package schema 明确后；
- 有足够时间单独处理 time coordinate、trajectory evidence 和 temporal metrics。

## Claude PR 任务拆分

Claude agent 任务放在 `.claude/tasks/`，每个 task 一个 md。`task_index.md` 作为总览和分派入口。

任务文件固定包含：

- Goal
- Scope
- Context files
- Implementation rules
- Deliverables
- Acceptance checks
- PR title

Claude 适合做：

- root repo 能力探索；
- 模块/checkpoint 路径审计；
- 单方法 exporter / smoke PR；
- agentic runner 泛化；
- build accounting 和 evidence schema。

需要人工最终确认：

- fixed API support 最终判定；
- strongest shared detector/checkpoint 选择；
- Track 3/4 benchmark 设计；
- memory fairness 和 evidence 意义的论文/报告表述。

## 近期 Checklist

1. 完成 baseline capability audit，并更新 `baseline_registry.md`。
2. 完成 shared module OV detector audit，并更新 `modules.md`。
3. 冻结 Track 1/2 formal shared OV detector route。
4. 完成 ClawS、HOV-SG、DualMap、ConceptGraphs 的 Track 1 fixed API package。
5. 只对 native query bridge 成立的方法完成 Track 2 fixed API。
6. 为所有不支持方法生成 invalid fixed API result。
7. 泛化 Track 1/2 agentic sandbox runner。
8. 给所有 memory build 加 build accounting。
9. 定义 evidence contract 和 validator。
10. 再进入 ScanRefer Track 3。
11. 最后进入 OpenEQA Track 4。

## 当前已定决策

- fixed API 统一成 Python entrypoint；不把 JSON/CLI entrypoint 作为第一版标准。
- Track 1/2 正式主评测只使用 shared OV detector setup，并继续只报告 detector-coverable split。
- DualMap、ConceptGraphs 等 detector-backed 方法必须用 shared OV detector route 进入主表；HOV-SG 必须记录 shared SAM/CLIP prompt route。
- closed detector、method-native detector 或不同 prompt/checkpoint 只作为 `module_ablation`。
- agentic sandbox 默认 package + source-code full access；raw frames / crops /
  detector/module override 作为单独 ablation。
- agent backend 第一版默认 Claude Code。
- memory package 复制进 sandbox，不直接挂载原 package 作为工作目录。
- evidence correctness 第一版使用 LLM judge。
