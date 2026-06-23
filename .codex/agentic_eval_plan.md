# Agentic Spatial Memory Evaluation 落地计划

Last updated: 2026-06-23 (refactor: 3-track + agent-designed baseline)

本文档把 `agentic_eval.md` 的研究构想收紧成当前 repo 的工程执行计划。
`agentic_eval.md` 保留研究愿景；本文档描述按什么顺序实现、验证、拆分任务。

## 核心原则

- 所有 evaluation 都消费导出的 memory package，不直接假设 method repo 有统一接口。
- fixed API 只评估方法 *原生* memory 或 root repo 已有能力能诚实支持的接口；不支持的
  方法输出 `status: invalid`，不能用 LLM wrapper / fallback / 手写转换假装支持。
- Track 1 的正式主评测统一使用 shared strongest open-vocabulary detector setup；
  closed-detector 或 method-native detector 变体只能作为 module ablation。
- agentic 评测是不同 memory 形式之间的主要公平比较路径，但它 **不是 coding-agent
  benchmark**：用 **per-query LLM + method-native tools**。evaluator 每次只给一个问题，
  LLM 通过预声明的原生接口/tool 访问 raw/native memory，再输出答案。不得把全部问题
  一次性丢给 coding agent，不得让 agent 写新接口或读 evaluation adapter 重建能力。
- coding agent 的真正角色在 `agent_designed_baseline.md`：让它 **设计 memory**，而不是
  给现有方法临时补接口。

## 这次重构的关键变更

- **Track 收敛 5 → 3**：
  - 旧 `track1_memory_construction` + 旧 `track2_object_location`
    → 新 `track1_object_location`（object-location query 准确率 + 构建侧 size /
    time-per-frame / peak）。不再单独报告 inventory recall / false-memory /
    redundancy / localization error。
  - 旧 `track3_scanrefer` → 新 `track2_scanrefer`。
  - 旧 `track4_openeqa` → 新 `track3_openeqa`（ScanNet + HM3D）。
- **删除旧 full-access coding-agent runner**：正式 agentic 路线只保留 `tool_llm`。
- **新增 `agent_designed` method family** 与 agent-designed memory harness。
- **OC-NaVQA / SG3D 移出主线**，作为零样本迁移目标延后（Phase 5）。
- **不再强行适配古早方法**：已经是 agent 形态的方法用原生 tools；没有原生能力的方法
  在对应 track 直接 `invalid` / `not_applicable`，只参与 fixed-API deterministic 对照。

## 全局约定

- Memory 输出路径：`memories/<method>/<dataset>/<scene-or-episode>/<run-id>/`。
- Result 输出路径：`results/<method>/<evaluation>/<timestamp>/`。
- Track key（贯穿 capabilities.json、validator、evaluator、report）：
  - `track1_object_location`
  - `track2_scanrefer`
  - `track3_openeqa`
- 每个 package 必须通过 validator，并明确声明三个 track 的 fixed-API support 或 invalid reason。
- tool-LLM sandbox 默认只暴露 raw/native memory + native retrieval/query tools；fixed-API
  转换视图、build code、raw frames / crops 仍是单独 ablation。

## 数据现状（2026-06-23 实测）

| 数据 | 路径 | 状态 |
|---|---|---|
| ScanNet++ | `/data/mondo-training-dataset/semantic_mapping/scannetpp` | 在位（Track 1 当前 scene `036bce3393`） |
| ScanNet scans | `/data/mondo-training-dataset/semantic_mapping/scannet/scans` | 在位（ScanRefer 几何来源候选） |
| OpenEQA ScanNet | `openeqa_scannet_dbs/`, `openeqa_frames/scannet-v0`, `openeqa_scannet_rgbd/` | 部分在位（当前 1 个 episode：`scannet-v0/002-scannet-scene0709_00`） |
| ScanRefer 标注 | — | **缺失**，需获取（Track 2 数据 stub 记录 TODO） |
| HM3D / OpenEQA-HM3D | — | **缺失**，需获取（Track 3 HM3D 部分 stub 记录 TODO） |
| OC-NaVQA / SG3D | — | 缺失；延后到 Phase 5 |

数据获取动作记录在 `path_registry.md` 和各 track `data.py` 的 stub 中，不混进代码逻辑。

## Track 定义

### Track 1：Object-Level Location Query

评估 memory 能否回答基础 object lookup/location query，并记录构建开销。

fixed API requirement：

- `query_object(query)` / `locate_object(query)` / 等价声明（capability key
  `track1_object_location`）；
- evaluator 传入 `target_label` + 自然语言 query；package 优先 exact / normalized-label 匹配；
- 返回 candidate objects、位置、score、evidence。

构建开销（从 `build_log.json` / `manifest.build` 读取，evaluator 汇总进 summary）：

- `native_memory_size_bytes`（主 memory-size 指标）、`package_size_bytes`、
  `memory_artifact_size_bytes`、`frame_count`、`time_per_frame_seconds`、
  `peak_ram_bytes`、`peak_vram_bytes`。

如果 memory 没有 object-level memory 或可比较的 object-location query API，fixed API
结果为 `invalid`。tool-LLM 版本只有方法本身有原生 retrieval/query tools 时才运行，
否则 `not_applicable`。

### Track 2：ScanRefer Instance-Level Referring Query

适配 ScanRefer，评估细粒度 referring expression grounding：target top-1/top-k、3D IoU、
center distance、attribute/relation evidence。

fixed API requirement：

- `resolve_referring_expression(query)` 或能力等价的 object query API（capability key
  `track2_scanrefer`）；
- 支持 attribute / relation evidence，或明确声明不支持。

不支持 referring query 的方法 fixed API 结果为 `invalid`。tool-LLM 版本只暴露 raw/native
memory 和原生 referring/retrieval tools。

### Track 3：OpenEQA General Spatial QA

适配 OpenEQA（ScanNet + HM3D）。评估开放空间问答：short answer、evidence、LLM-Match /
exact category metrics、agent trace 和 memory usage。

fixed API requirement：

- 原生声明 `answer_question(question)` / `get_memory_text` 或等价 QA/retrieval API
  （capability key `track3_openeqa`）；
- 不允许用通用 object-table-to-LLM 答案器把不支持 QA 的 memory 包成支持。

没有原生 QA/retrieval fixed API → `invalid`。tool-LLM 版本只允许原生 QA/retrieval
tools；没有这类 tool 的方法标 `not_applicable`。LLM judge 必须与 memory 构建 / 答题 LLM
隔离。

### Deferred：Long-Horizon / Task Grounding

OC-NaVQA、ReMEmbR temporal questions、SG3D task grounding 留作零样本迁移目标，见 Phase 5。
当前只在 registry / design notes 里记录能力，不进入主实现线。

## Minimal Memory Package

目标结构（详见 `memory_package_spec.md`）：

```text
memories/<method>/<dataset>/<scene-or-episode>/<run-id>/
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

`capabilities.json.fixed_api` 必须包含三个 track key，每个 `supported`（带 Python
entrypoint）或 `invalid`（带 reason）：

```json
{
  "fixed_api": {
    "track1_object_location": { "status": "supported|invalid", "entrypoint": "...", "reason": "" },
    "track2_scanrefer":       { "status": "supported|invalid", "entrypoint": "...", "reason": "" },
    "track3_openeqa":         { "status": "supported|invalid", "entrypoint": "...", "reason": "" }
  },
  "agent_access": { "mode": "tool_llm | not_applicable", "...": "..." }
}
```

`agent_designed` package 用相同契约，额外要求见 `agent_designed_baseline.md`：必须声明
它依赖的 shared modules、构建命令、query 接口，且不得包含 test answers。

## Fixed API Eligibility Gate

fixed API 是严格能力判定，不是 evaluator 侧包装。只有满足以下条件才能写 `supported`：

- root repo 或 native memory artifact 已提供对应信息或稳定查询入口；
- package entrypoint 只是薄封装/格式转换/非交互化，不改变方法能力；
- 不调用 evaluator 私有 GT、不使用 benchmark-specific rules；
- 不把 object table 临时交给通用 LLM 生成 fixed-API 答案；
- smoke test 能在至少一个 scene 上复现。

不满足则写 `status: invalid` + `reason_code: unsupported_fixed_api` + message，并同步到
`.codex/baseline_registry.md`（给 root repo 证据路径）。

## Module And Vocabulary Fairness

Track 1 主结果只比较 shared strongest open-vocabulary detector setup（规则保持不变）：

- `spatial_memory_evaluation/shared_modules/` 是 detector / segmenter / feature encoder /
  OV prompt-eval label list / checkpoint 的唯一 registry。
- 外部方法通过 `scripts/methods/<method>/` adapter 读取 shared modules，再翻译成原生
  CLI/Hydra override；不改外部 repo 源码。
- 同类模块统一版本/checkpoint/preprocess/device；同 scene 内 OV detector / prompt-eval
  list / label normalization 统一。
- 默认 prompt-eval list：`spatial_memory_evaluation/assets/class_lists/detector_coverable.txt`。
- formal OV detector target：YOLO-World-L（`yolov8l-world.pt`）；S 仅 smoke fallback。
- closed-detector / method-native detector / 不同 prompt list 只作为 `module_ablation`。
- agent-designed baseline 调用的 shared modules 也走同一 registry，记录在其 package
  build log，保证与人工方法的模块公平。

## Memory Form Fairness / Evidence / Build Accounting

不强行统一 memory 形式（object table / native map / scene graph / caption DB / vector
DB 都可作为 native memory artifact 保留）。公平性来自三层：构建阶段共享输入与模块；
fixed API 只比 deterministic 原生能力；tool-LLM 给所有形式同样的 per-query tool-loop。
memory size 主指标用 native artifact size；package wrapper size 单独报告。

Evidence 是 provenance/debug/grounding，不是额外答案来源；必须在 build 阶段自然产生，
不含 GT/benchmark label/手写规则。Build accounting 字段（frame_count、build_runtime、
time_per_frame、native/package/memory size、peak RAM/VRAM）在 `manifest.build` 和
`build_log.json` 同步记录，validator 校验。

## Invalid Result Schema

所有 fixed API evaluator 支持 invalid（正式结果，不是程序错误）：

```json
{
  "status": "invalid",
  "reason_code": "unsupported_fixed_api",
  "required_api": "track3_openeqa",
  "method": "hovsg",
  "package_path": "memories/hovsg/...",
  "message": "HOV-SG package does not declare a native OpenEQA fixed API."
}
```

no-explicit-memory control 用 `reason_code: control_no_explicit_memory` + `control: true`。
程序错误用 `status: error` + traceback / log path。

## 执行顺序

### Phase 0：契约冻结（本次重构）

- validator `TRACK_KEYS` 改为 `track1_object_location` / `track2_scanrefer` /
  `track3_openeqa`；新增 `agent_designed` method family。
- 更新 `memory_package_spec.md`、示例 package（minimal / caption_control /
  multiframe_vlm_control + 新 agent_designed 示例）、schemas。
- 更新 `.codex/*` 与 README 的 track 命名与数据现状。

DoD：示例 package 全部过 validator；evaluator import / CLI smoke 通过；无残留
`track4` / `memory_construction` 引用。

### Phase 1：Track 1（object location + accounting）

- fixed_api：读 `capabilities.fixed_api.track1_object_location`，调用 query entrypoint，
  传 `target_label`，汇总 query 指标 + 构建开销；不支持写 invalid。
- tool_llm：per-query LLM + 原生 tools（DAAAM / ReMEmbR / ClawS）。
- 优先方法：ClawS（native）、ConceptGraphs / DualMap / HOV-SG（先 export，再补 query bridge）、
  DAAAM（deterministic native semantic index）。
- DoD：所有方法有 Track 1 result（metrics 或 invalid）；summary 含 size / build runtime /
  time-per-frame / peak。

### Phase 2：Track 2（ScanRefer）

- 先准备 ScanRefer 数据 stub（数据缺失时 evaluator 产出明确的 `data_unavailable` 结果，
  不静默跳过）。
- fixed_api：读 `track2_scanrefer`，supported 调 referring resolver，否则 invalid。
- tool_llm：暴露 raw/native memory + 原生 referring/retrieval tools。
- DoD：ScanRefer subset 可复现；每方法有 fixed result；至少一个方法跑通 tool-LLM。

### Phase 3：Track 3（OpenEQA on ScanNet + HM3D）

- 利用现有 `openeqa_scannet_*`；HM3D 数据获取后接入。
- fixed_api：读 `track3_openeqa`，supported 调原生 QA/retrieval API，否则 invalid，
  不引入通用 object-table-to-LLM 答案器。
- tool_llm：per-query LLM + 原生 QA/retrieval tools；输出 short answer + evidence + trace。
- scoring：LLM-Match / category / evidence audit；LLM judge 与构建/答题 LLM 隔离。
- DoD：不支持的方法显式 invalid；agentic 输出含 evidence + trace；judge 隔离明确。

### Phase 4：Agent-Designed Memory Baseline

实现 `spatial_memory_evaluation/agent_designed/` harness（详见 `agent_designed_baseline.md`）：

- workspace builder：给 coding agent 准备 shared-module 介绍、training scenes/queries
  （无 answers）、示例 package、契约文档、执行环境说明。
- contract：agent 产出 `agent_designed` memory package（schema + build script + query
  接口 + README + evidence format）。
- harness：build → validate → 跑 Track 1/2/3 evaluator → 汇总分数 →（Iterative variant）
  回灌错误反馈再迭代。
- variants：Prompt-Only / Few-Example / Coding-Agent / Iterative。

DoD：harness 能在 held-out split 上端到端跑一个 agent-designed package 并出分；防泄漏
检查（no test answers / no hardcoded test rules）有 validator 支持。

### Phase 5：零样本迁移（含 long-horizon deferred）

- 把 agentic baseline（含 agent-designed）零样本迁移到 SG3D / OC-NaVQA。
- 恢复 temporal / long-horizon 评测（time coordinate、trajectory evidence、temporal
  metrics、leakage policy）。

恢复条件：Track 1-3 package / fixed API / tool-LLM 稳定；agent-designed harness 出分稳定。

## Claude PR 任务拆分

Claude agent 任务放 `.claude/tasks/`，每个 task 一个 md，`task_index.md` 作总览。
任务文件固定包含：Goal / Scope / Context files / Implementation rules / Deliverables /
Acceptance checks / PR title。

Claude 适合：root repo 能力探索、模块/checkpoint 路径审计、单方法 exporter / smoke PR、
agentic runner 泛化、build accounting / evidence schema、agent-designed harness 子模块。

人工最终确认：fixed API support 判定、strongest shared detector/checkpoint 选择、
Track 2/3 benchmark 设计、agent-designed 防泄漏与评价表述、memory fairness 论文表述。

## 近期 Checklist

1. 冻结 3-track + agent_designed 契约（validator / spec / 示例 / schemas）。
2. 重构 track 模块：合并 track1+track2 → track1（object location + accounting）。
3. 加 track2（scanrefer）/ track3（openeqa）skeleton evaluator + data stub。
4. 搭 agent_designed harness 骨架（workspace / contract / harness / variants）。
5. 更新 scripts（evaluate_track1/2/3、build data）、examples、README。
6. 全仓 grep 清理残留 `track4` / `track2_object_location` / `memory_construction` 命名。
7. 之后：ScanRefer / HM3D 数据获取 → Track 2/3 真实评测 → agent-designed 出分 →
   零样本迁移 SG3D / NaVQA。

## 当前已定决策

- Track key：`track1_object_location` / `track2_scanrefer` / `track3_openeqa`。
- fixed API 统一 Python entrypoint。
- Track 1 正式主评测只用 shared OV detector setup，继续报告 detector-coverable split。
- closed detector / method-native detector / 不同 prompt-checkpoint 只作 `module_ablation`。
- tool-LLM sandbox 默认只暴露 prompt / tool specs / raw-native memory /（受控）方法
  原始 source；fixed-API 视图 / adapter / build code / raw frames 都是 ablation。
- agent backend 第一版默认 Claude Code（Bedrock）。
- memory package 复制进 sandbox，不直接挂载原 package 作为工作目录。
- evidence correctness 第一版用 LLM judge，与构建/答题 LLM 隔离。
- agent-designed baseline 是项目重心；旧 full-access coding-agent 一次性答题设定已删除。
