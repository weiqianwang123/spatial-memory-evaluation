# Agentic Spatial Memory Evaluation 执行计划

本文档把 `agentic_eval.md` 中的研究构想整理成当前 repo 的工程 roadmap。
`agentic_eval.md` 是研究愿景；本文档只描述接下来应该按什么顺序实现、验证
和扩展 evaluation。

当前原则：所有 evaluation 都先消费导出的 memory package，而不是直接假设某个
方法 repo 有统一接口。fixed API 只评估 memory package 明确声明支持的能力；
如果某个方法的 memory 本身不支持对应 track，结果必须显式记为 `invalid`，
不能用空结果、fallback 策略或手写转换假装支持。

## 核心目标

这个项目不是提出新的 spatial memory 方法，而是提出一个 benchmark 和
evaluation protocol，用来评估 spatial memory 是否能作为 embodied agent
可访问、可理解、可查询、可验证、可组合的外部认知资源。

benchmark 需要同时回答：

- memory 本身是否准确、紧凑、低冗余；
- memory 是否提供稳定、可复现、可比较的 fixed API；
- 在 fixed API 不足时，agent 是否能通过 full access 自主读取和使用 memory；
- 成功或失败来自 memory 缺失、fixed API 缺失、schema 难懂、证据不足，还是
  agent 推理错误。

## 全局约定

- 所有生成 memory 放在 `memories/<method>/<dataset>/<scene-or-episode>/<run-id>/`。
- 所有结果放在 `results/<method>/<evaluation>/<timestamp>/`。
- 所有方法先导出 minimal memory package，再进入任何 evaluator。
- fixed API evaluation 只调用 package 声明的固定接口；不支持就输出
  `status: invalid`。
- agentic evaluation 在 sandbox 中运行，输入是 memory package、query、允许工具、
  budget 和明确的禁止信息；输出 answer、evidence、trace、cost 和 failure mode。
- spatial-temporal / long-horizon 先不作为当前实现重点。

## Tracks

### Track 1：Memory Construction / Object Inventory

评估 memory 构建质量。第一版聚焦 object inventory：

- object recall / detector-coverable recall；
- false memory ratio；
- duplicate / redundancy ratio；
- localization error；
- memory size 和构建时间。

fixed API requirement：

- `list_objects()` 或等价的 exported object table；
- object 至少包含 `object_id`、`label`、`position_3d`、`evidence`。

如果 package 没有 object-level memory 或 object table，fixed API 结果为
`invalid`。

### Track 2：Basic Object Location Query

评估 memory 是否能回答简单 object lookup/location query，例如 “where is the
chair?”。

fixed API requirement：

- `query_object(query)` / `locate_object(query)` / 等价声明；
- 返回 candidate objects、位置、score、evidence。

如果 memory 只有原始图像、caption context 或 DSG 但没有声明可比较的 object
location query API，fixed API 结果为 `invalid`。agentic 版本仍可在 sandbox 中
尝试使用 package。

### Track 3：ScanRefer Fine-Grained Referring Query

适配 ScanRefer。评估细粒度 referring expression grounding：

- target object top-1 / top-k；
- 3D IoU；
- center distance；
- attribute / relation evidence。

fixed API requirement：

- `resolve_referring_expression(query)` 或能力等价的 object query API；
- 支持 attribute / relation evidence 或能明确声明不支持。

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
  或等价的 QA/retrieval API；
- 不允许用通用 object-table-to-LLM 答案器把不支持 QA 的 memory 包成支持。

如果方法没有原生 QA/retrieval fixed API，fixed API 结果为 `invalid`。agentic
版本仍可通过 sandbox 使用 memory package。

### Deferred：Spatial-Temporal / Long-Horizon QA

包括 OC-NaVQA、ReMEmbR temporal questions、duration、before/after、last seen
等。先只在 registry 里记录能力，不进入当前实现主线。

## Minimal Memory Package

第一步只要求 minimal package，不要求一开始就做到 full package。

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
- `memory_artifacts`：native map、DB、object table、scene graph、captions、
  keyframes、crops；
- `schema`：坐标系、object id、relations、confidence、time 字段说明；
- `build`：构建命令、config、环境、runtime、memory size；
- `allowed_access`：确认不包含 GT annotations、benchmark answers、test labels
  和为测试问题手写的规则。

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

invalid 是正式结果，不是程序错误。程序错误用 `status: error`，并附 traceback 或
log path。

## 执行顺序

### Step 1：导出所有方法的 Minimal Memory

目标：所有 baseline 都先变成可验证、可读取、可比较的 memory package。

第一批需要覆盖：

- ClawS SpatialRAG
- DualMap
- HOV-SG
- ConceptGraphs
- DAAAM
- Hydra standalone
- ReMEmbR
- Multi-frame VLM control
- LLM-with-captions control

Milestones：

- M1.1：冻结 minimal package spec。
  - 实现 `manifest.json`、`capabilities.json`、`schema.md` 的 schema。
  - 写 package validator。
  - validator 只检查 package 是否诚实声明能力，不强迫所有方法支持所有 track。
- M1.2：实现 package export registry。
  - 每个 method 有一个 exporter entry。
  - exporter 只读取方法 repo 的 native outputs 或运行原生构建脚本。
  - exporter 输出到 `memories/<method>/<dataset>/<scene>/<run-id>/`。
- M1.3：导出 object-map family。
  - ClawS SpatialRAG
  - DualMap
  - HOV-SG
  - ConceptGraphs
- M1.4：导出 scene-graph family。
  - DAAAM
  - Hydra standalone
- M1.5：导出 caption / no-explicit-memory family。
  - ReMEmbR caption memory
  - Multi-frame VLM sampled frames
  - LLM-with-captions caption context

交付物：

- `spatial_memory_evaluation/memory_package.py`
- `spatial_memory_evaluation/validate_package.py`
- `scripts/export_memory_package.py`
- `memories/<method>/...` examples，Git ignored
- `.codex/memory_package_spec.md`

Definition of Done：

- 每个 method/control 至少有一个 package 通过 validator。
- 每个 package 都明确声明 Track 1-4 的 fixed API support / invalid reason。
- package 内不包含 GT answer、test labels 或 benchmark-specific hand rules。

### Step 2：用导出 Memory 的 Fixed API 评估 Track 1 和 Track 2

目标：先建立 deterministic fixed API baseline。所有 evaluator 只读 package，不直接
进入方法 repo 猜接口。

Track 1 fixed API：

- 输入：memory package path、scene/dataset GT object inventory。
- 读取：`capabilities.fixed_api.track1_memory_construction`。
- 如果 supported，读取 object table / graph node export。
- 如果 invalid，写 invalid result。

Track 2 fixed API：

- 输入：memory package path、object-location query dataset。
- 读取：`capabilities.fixed_api.track2_object_location`。
- 如果 supported，调用 package 内声明的 query entrypoint。
- 如果 invalid，写 invalid result。

Milestones：

- M2.1：实现 fixed API loader。
  - 读取 `capabilities.json`。
  - resolve entrypoint 到 package 内相对路径。
  - 禁止 entrypoint 访问 package 外写入结果。
- M2.2：Track 1 evaluator。
  - object recall、false memory、redundancy、localization、size、runtime。
- M2.3：Track 2 evaluator。
  - object query top-k、center distance、evidence hit、latency。
- M2.4：统一 invalid/error/report 输出。

交付物：

```text
results/<method>/track1-memory-construction/<timestamp>/
  metrics.json
  predictions.json
  report.md

results/<method>/track2-object-location/<timestamp>/
  metrics.json
  predictions.json
  report.md
```

Definition of Done：

- Track 1 和 Track 2 可以批量扫所有 package。
- unsupported fixed API 会产生 `status: invalid`，不会 crash。
- metrics 可以从 exported packages 重新计算。

### Step 3：引入 Agent Full Access，评估 Track 1 和 Track 2

目标：在 fixed API 之外测试 agent 是否能自主理解和使用 memory package。第一版
agentic evaluation 也只做 Track 1 和 Track 2。

Sandbox 参考：

- 参考 `/home/robin_wang/robocode` 的 sandbox 设计。
- 重点参考：
  - `src/robocode/utils/docker_sandbox.py`
  - `src/robocode/utils/sandbox.py`
  - `src/robocode/utils/backends/claude.py`
  - `integration_tests/red_team_sandbox.py`
- 我没有在 root 下找到 `/home/robin_wang/roboclaude`；如果后续确认有该路径，
  再把参考路径替换进去。

Agent run 结构：

```text
results/<method>/<agentic-evaluation>/<timestamp>/
  queries.jsonl
  predictions.jsonl
  metrics.json
  report.md
  traces/
  sandboxes/<query_id>/
```

每个 query 建一个 sandbox：

- sandbox 只写自己的工作目录和指定 output file。
- memory package 以只读方式挂载或复制进 sandbox。
- query、rubric、output schema 写入 sandbox。
- GT answers 不进入 sandbox。
- raw frames / crops 是否可读由 `agent_access` policy 控制。
- agent 必须输出 `answer.json`，包含 answer、evidence、trace、used_memory、
  used_raw_input、used_code、failure_mode。

Docker sandbox 优先：

- container 只能写 `/sandbox`；
- memory package read-only mount；
- 网络默认关闭或只允许模型 API endpoint；
- stream 写入 `stream.jsonl`；
- run 完成后自动保存 sandbox git diff；
- OS-level sandbox 只允许 dev smoke，因为它可能允许读 host filesystem。

Milestones：

- M3.1：Agent output schema。
- M3.2：Agent sandbox runner。
- M3.3：Track 1 agentic task prompt。
  - 要求 agent 从 package 中导出/判断 object inventory。
  - 评估 agent 是否找到正确 artifact、理解 schema、给出 evidence。
- M3.4：Track 2 agentic task prompt。
  - 要求 agent 回答 object-location query。
  - 评估 answer、evidence、tool usage、latency/cost。
- M3.5：Red-team sandbox smoke。
  - 路径逃逸、绝对路径写入、GT 泄漏、网络访问、输出 schema violation。

交付物：

- `spatial_memory_evaluation/agent_runner.py`
- `spatial_memory_evaluation/agent_sandbox.py`
- `spatial_memory_evaluation/agent_trace.py`
- `spatial_memory_evaluation/evaluate_agentic_track1.py`
- `spatial_memory_evaluation/evaluate_agentic_track2.py`

Definition of Done：

- 一个 scene 上至少跑通一个 method 的 Track 1/2 agentic eval。
- sandbox 输出 answer/evidence/trace/cost。
- failure 能区分 memory 缺失、schema 不清、agent 未找到 artifact、推理错误。

### Step 4：适配 ScanRefer，作为 Track 3

目标：把 ScanRefer 变成第三个 track，并同时跑 fixed API 和 agentic 两种设置。

数据适配：

- 转成统一 query format：
  - `query_id`
  - `dataset`
  - `scene_id`
  - `utterance`
  - `target_object_id`
  - `target_bbox`
  - `allowed_eval_fields`
- 对齐 ScanNet scene ids、object ids、坐标系、bbox 格式。
- 不把 target label/answer 放入 agent sandbox。

Fixed API：

- 读取 `capabilities.fixed_api.track3_scanrefer`。
- supported：调用 package 声明的 referring resolver。
- invalid：写 `status: invalid`。

Agentic：

- agent 可读 memory package、schema、object table、scene graph、crops/keyframes。
- agent 输出 target object candidate、evidence、reasoning summary。
- 评估 top-1/top-k、3D IoU、center distance、evidence correctness。

Milestones：

- M4.1：ScanRefer converter。
- M4.2：Track 3 fixed API evaluator。
- M4.3：Track 3 agentic evaluator。
- M4.4：ScanRefer report template。

交付物：

```text
results/<method>/track3-scanrefer-fixed/<timestamp>/
results/<method>/track3-scanrefer-agentic/<timestamp>/
```

Definition of Done：

- ScanRefer subset 可复现。
- 每个 method 都有 fixed result：supported metrics 或 invalid。
- 至少一个 method 跑通 agentic ScanRefer。

### Step 5：适配 OpenEQA，作为 Track 4

目标：把 OpenEQA general spatial QA 接进同一套 memory package pipeline，同时跑
fixed API 和 agentic。

数据适配：

- 保留 OpenEQA question / answer / episode 信息。
- 将 episode 映射到 exported memory package。
- GT answer 只给 scorer，不进入 fixed API entrypoint 或 agent sandbox。

Fixed API：

- 读取 `capabilities.fixed_api.track4_openeqa`。
- supported：调用方法原生 QA/retrieval API。
- invalid：写 `status: invalid`。
- 不再引入通用 object-table-to-LLM 答案器作为方法能力。

Agentic：

- agent 从 package 自主读取 memory、schema、evidence、allowed raw links。
- 输出 short answer、evidence、trace。
- scoring 用 LLM Match / exact category / evidence audit。

Milestones：

- M5.1：OpenEQA converter。
- M5.2：Track 4 fixed API evaluator。
- M5.3：Track 4 agentic evaluator。
- M5.4：LLM Match scorer 接到统一 result schema。
- M5.5：OpenEQA report template。

交付物：

```text
results/<method>/track4-openeqa-fixed/<timestamp>/
results/<method>/track4-openeqa-agentic/<timestamp>/
```

Definition of Done：

- scene0709 上 fixed / agentic pipeline 都能跑。
- 不支持 fixed QA API 的方法显式 invalid。
- agentic 输出包含 evidence 和 trace，不只是 answer string。

### Step 6：Spatial-Temporal 先 Deferred

当前暂时不实现 spatial-temporal track。只保留 design notes：

- OC-NaVQA / ReMEmbR temporal questions；
- duration / before-after / last seen；
- long-horizon captions / trajectories；
- temporal leakage policy；
- temporal evidence scoring。

何时恢复：

- Track 1-4 的 package、fixed API、agentic sandbox 都稳定后；
- ReMEmbR / DAAAM / future temporal methods 的 package schema 明确后；
- 有足够时间单独处理 time coordinate、trajectory evidence 和 temporal metrics。

## 近期 Checklist

1. 写 `.codex/memory_package_spec.md`，把 minimal package 和
   `capabilities.json` 固定下来。
2. 写 `validate_package.py`，先只做 schema 和 artifact existence 检查。
3. 为 baseline registry 中所有 present methods 创建 exporter stub。
4. 先导出 ClawS、DualMap、HOV-SG 的 current-scene package。
5. 实现 Track 1 fixed evaluator，要求 unsupported package 输出 invalid。
6. 实现 Track 2 fixed evaluator，要求 unsupported package 输出 invalid。
7. 搭建 agent sandbox runner，优先参考 `robocode` Docker sandbox。
8. 在一个 scene、一个 method 上跑通 Track 1/2 agentic。
9. 再进入 ScanRefer Track 3。
10. 最后进入 OpenEQA Track 4。

## 当前未决问题

- fixed API 的最小函数名是否统一成 Python entrypoint，还是允许 JSON/CLI entrypoint？
- Track 1 object inventory 是否必须所有方法导出 object table，还是 DSG/caption
  方法可以只声明 invalid？
- agentic sandbox 是否默认允许读取 raw frames，还是默认 memory-only、raw frames 作为
  单独 ablation？
- agent backend 默认用 Claude Code、OpenCode，还是抽象成 provider config？
- Docker sandbox 需要如何挂载大型 memory package，复制还是 read-only bind mount？
- evidence correctness 第一版由 rule-based、LLM judge、human audit 还是 hybrid？
