# Beyond Fixed Queries: Benchmarking Spatial Memory as an External Resource for Embodied Agents

Last updated: 2026-06-23 (refactor: 3-track agentic benchmark + agent-designed memory baseline)

## 0. 这次重构改了什么

旧版本是 method-centric 的多 track 评测，强调每个方法导出固定 API，然后人工
对齐接口。新版本把重心转向 **agentic 评测** 和 **agent-designed memory**：

- Track 数量从 5 个收敛成 3 个，去冗余：
  - 旧 Track A（memory construction 质量）和旧 Track B（object location query）
    合并成 **新 Track 1：Object-Level Location Query**。Track 1 只评 object
    location 查询准确率，外加构建侧的 memory size / time-per-frame / peak memory。
    不再单独报告 object recall / false-memory ratio / redundancy ratio /
    localization error 这些 inventory 质量指标。
  - 旧 Track C（ScanRefer referring）变成 **新 Track 2：Instance-Level Referring Query**。
  - 旧 Track D（OpenEQA general QA）变成 **新 Track 3：General Spatial QA**，
    同时在 ScanNet 和 HM3D 上做。
- 旧 Track E（OC-NaVQA long-horizon）和 SG3D **从主线移除**，作为
  零样本迁移目标延后处理（见 §6）。
- 不再强行适配古早（非 agent）方法去重建统一 fixed API。已经是 agent 形态的方法
  （DAAAM、ReMEmbR 等）直接开放它们的原生 tools 给 LLM；fixed-API 评测只用于
  deterministic 对照，对没有原生能力的方法直接判 `invalid`。
- 重心从“给现有方法排名”转向 **agent-designed memory baseline**：让 coding agent
  在共享模块、测试用例和评价体系下，自己写 memory 构建和 tool 接口，再在我们的
  benchmark 上评测，最后零样本迁移到 SG3D / NaVQA（见 §7）。

## 1. 核心动机

现有 spatial memory 工作已经能够从 posed RGBD sequence 中构建 object map、semantic
map、scene graph 或 long-horizon memory，并用于导航、问答或规划。但现有评测大多仍是
method-centric 的：每个方法自己定义 memory，自己设计 query interface，自己决定 agent
如何访问 memory，最后只看系统能否回答某些问题。

这种评测方式有一个关键缺陷：它无法判断一个 spatial memory 本身是否真的对 agent 有用。
一个 memory 可能已经包含正确的 object、位置、属性和关系，但因为固定接口无法表达复杂
查询而失败；反过来，一个方法也可能为特定任务设计很强的接口，在 benchmark 上表现很好，
却并不代表它的 memory 具有通用可用性。

因此本项目的核心不是提出一个新的 spatial memory 方法，而是提出一个新的 benchmark：
把 spatial memory 当作 embodied agent 可访问的外部认知资源来评估。我们不再只问
“这个方法能不能通过固定接口回答问题”，而是问：

> 当 agent 拿到一个 memory system 的 map、原生接口和（受控的）原始资源后，它能不能
> 自主理解、检索、验证和组合这些资源，从而完成空间问答和空间推理？

这就是 *Beyond Fixed Queries* 的核心含义。

进一步地，如果 memory 是为 agent 服务的，那么 memory 的形式是否也应该由 agent 根据
任务和评价体系自己设计？这是本项目的第二个核心问题，对应 agent-designed memory
baseline。

## 2. 与已有工作的关系

| 类别 | 代表方法 | 在本 benchmark 中的角色 |
|---|---|---|
| No explicit memory | Multi-frame VLM, LLM-with-captions | 控制组（`explicit_memory=false`）。回答“是否真的需要显式 spatial memory”。 |
| Caption retrieval memory | ReMEmbR | 非结构化 episodic memory；原生 retrieval tools 用于 agentic eval。 |
| Object-centric memory | ConceptGraphs, ClawS/SpatialRAG, DualMap | open-vocabulary object map，object/referring query 主力对照。 |
| Hierarchical scene graph | HOV-SG, Hydra | floor/room/object 层次结构；多数靠预定义接口被使用。 |
| Tool-based scene graph | DAAAM | 4D scene graph + tool-calling agent，highly-engineered tool-based memory 代表。 |

设计目标：用同一套 protocol，公平比较这些差异很大的 memory paradigm，而不是让每个方法
各自定义胜负标准。

与近期 benchmark 的关系：NaVQA（长视频、偏室外大场景）和 SG3D（task-relevant
grounding）是相关工作，但都不完全符合我们的需求。我们聚焦 **室内场景** 的 **分层评测**
（object → instance → general），用更细的 track 定位问题、扩大测试规模，弥补早期
NaVQA 子集太小、层次不清的问题。NaVQA / SG3D 留作零样本迁移目标。

## 3. Benchmark 核心设定

### 3.1 目标

本 benchmark 不提出新的 memory representation，而是提出新的 evaluation protocol，
系统评估：

1. memory 能否从 posed RGBD sequence 在线构建，且紧凑、低开销（size / time-per-frame / peak）；
2. memory 能否支持 object-level location query；
3. memory 能否支持 instance-level fine-grained referring query；
4. agent 能否使用 memory 系统的原生 tools 回答 general spatial QA；
5. 不同 memory paradigm 对 agentic spatial reasoning 的作用差异；
6. agent 能否根据任务和评价体系，自己设计出可用、可泛化的 memory。

### 3.2 输入

每个方法接收相同输入：RGB-D sequence、camera poses、intrinsics、timestamps，可选的
shared detector/segmenter 输出。第一版只用离线给定的 posed RGBD sequence，不做 active
exploration。

### 3.3 评测的两种访问形态

本 benchmark 用统一的两层访问形态，对所有 track 一致：

- **Fixed API（deterministic 对照）**：方法把原生能力暴露成声明式 Python entrypoint
  （见 `memory_package_spec.md`）。evaluator 直接调用，得到 deterministic 分数。
  只在方法 *原生* 支持时才标 `supported`；不支持就标 `invalid`，不允许用 LLM wrapper
  或手写规则假装支持。
- **Tool-LLM（agentic 主路径）**：evaluator 每次只给一个问题，LLM 通过方法 **原生**
  retrieval/query tools 访问 raw/native memory，再输出答案。这是不同 memory 形式之间
  的主要公平比较路径。对没有原生 LLM/tool 接口的方法（如当前 ScanNet++ 上的 HOV-SG），
  不强行纳入 tool-LLM eval，标 `not_applicable`。

> 删除项：旧的“full-access coding-agent 一次性回答所有问题”设定已删除。它会混淆三件
> 事——memory 本身的可用性、原生 tool 的能力、coding agent 临时写 adapter 的能力。
> coding agent 的角色被收敛进 §7 的 agent-designed memory baseline，那里它的目标是
> **设计 memory**，而不是临时给某个方法补接口。

### 3.4 允许 / 不允许提供给评测的资源

允许：构建好的 map/memory、原生 query interface 与 retrieval tools、schema 说明、
build log、（在 tool-LLM 下受控暴露的）方法原始 source code、memory 自带的 evidence。

不允许：GT annotations、benchmark answers、test labels、为测试问题手写的规则、在线
设置中未来时刻的信息、evaluator 侧转换出的 fixed-API 视图作为 agentic 主输入（除非它
本身就是方法原生 memory）。

## 4. Evaluation Tracks

分层 eval：object → instance → general，逐层定位 memory 的能力边界。

### Track 1：Object-Level Location Query（合并自旧 Track A + B）

- **目标**：评估 memory 能否回答基础物体位置问题（“where is the chair?”），同时记录
  memory 构建侧的开销。
- **数据集**：ScanNet++（当前 scene `036bce3393`，可扩展）。
- **查询**：category-level，由场景 GT object inventory 自动派生（每个 canonical label
  一个 query）。GT inventory 只用于把预测对齐到目标 object，不再作为单独报告的 inventory
  质量指标。
- **fixed API**：`query_object(package_dir, query) -> {predictions:[...]}`。evaluator 传入
  `target_label` + 自然语言 query；方法优先做 exact/normalized-label 匹配。
- **指标**：
  - 查询质量：`success@1`、`success@5`、`recall@1`、`recall@5`、`mrr`、
    `mean_first_hit_distance_m`、query latency（mean / median / p95 / QPS）。
  - 构建开销：`native_memory_size_bytes`（主 memory-size 指标）、`package_size_bytes`、
    `memory_artifact_size_bytes`、`frame_count`、`time_per_frame_seconds`、
    `peak_ram_bytes`、`peak_vram_bytes`。
- **invalid 条件**：没有可比较的 object-location query API 或没有 object-level memory。

### Track 2：Instance-Level Referring Query（旧 Track C）

- **目标**：评估 memory 能否回答更具体的 referring expression（“the red table near
  the window”），测试颜色、材质、属性、空间关系、object disambiguation。
- **数据集**：ScanRefer on ScanNet。
- **fixed API**：`resolve_referring_expression(package_dir, query) -> {predictions:[...]}`。
- **指标**：target top-1 / top-k、3D IoU、center distance、attribute/relation evidence
  correctness、query latency。
- **invalid 条件**：没有原生 referring resolver 或能力等价的 object query API。

### Track 3：General Spatial QA（旧 Track D）

- **目标**：评估 agent 能否用 memory 系统回答开放空间问题。
- **数据集**：OpenEQA，在 **ScanNet 和 HM3D** 上都做。
- **问题类型**：object attribute / color / material / state、spatial relation、room-level
  reasoning、functional reasoning、general scene understanding。
- **fixed API**：`answer_question(package_dir, query) -> {answer, evidence}`，只在方法
  原生有 QA/retrieval API 时 supported；不允许用通用 object-table-to-LLM 答案器假装支持。
- **指标**（最终选有代表性的子集）：LLM-Match、category accuracy、supporting-evidence
  correctness、memory usage rate、raw-input fallback rate、tool-call count、token cost、
  end-to-end latency、failure-mode distribution。

### Deferred：Long-Horizon / Task Grounding（旧 Track E + SG3D）

OC-NaVQA static subset、ReMEmbR temporal questions（when / how-long / before-after /
last-seen / duration）、SG3D task grounding 全部从主线移除。它们将作为 **零样本迁移
目标**：当我们的 agentic baseline（尤其 agent-designed baseline）在自有 benchmark 上
效果可接受时，直接零样本迁移到 SG3D / NaVQA 上观察效果（见 §6、§7）。

## 5. 核心指标分组

- **Memory Quality / Cost（Track 1 构建侧）**：memory size、time-per-frame、peak RAM/VRAM。
- **Query Quality（Track 1/2）**：top-1 / top-k、position error / 3D IoU、query latency、
  evidence correctness。
- **Agentic Usability（Track 1/2/3 tool-LLM）**：answer accuracy、supporting-evidence
  correctness、memory usage rate、raw-input fallback rate、tool-call count、token cost、
  end-to-end latency。
  - 最关键：**Memory Usage Rate**（答案多少依赖 memory）、**Raw-Input Fallback Rate**
    （多少要回到 raw frames，越高说明 memory 保存得越不够）、**Evidence Correctness**
    （是否用了正确的 object/relation/region/keyframe，避免靠语言先验猜对）。

## 6. Failure Mode Analysis

每个失败样本归类（用于诊断而非仅排名）：(1) memory 缺目标 object；(2) 位置错误；
(3) 属性错误；(4) 空间关系缺失；(5) 重复 object 致选错；(6) schema 难懂；(7) 没找到
正确 evidence；(8) 找到 evidence 但推理错；(9) 主要靠 raw input 才答对；(10) 答案格式
错误；(11) evaluator 判断歧义。

## 7. 重点：Agent-Designed Memory Baseline

这是本项目重心。详见 `.codex/agent_designed_baseline.md`，这里只给愿景。

### 7.1 动机

当前 spatial memory 方法大多是高度模块化、工程化的系统（segmentation → tracking →
object merging → scene graph → relation extraction → query interface → QA）。但如果
memory 最终是服务 agent 的，一个自然的问题是：**memory 的形式能否由 agent 根据任务和
评价体系自己设计？**

### 7.2 设置

给 coding agent 提供：

1. shared modules 的介绍与可调用接口（detector / segmenter / CLIP / VLM / embeddings）；
2. 部分 training scenes 和 training queries（不含 test answers）；
3. benchmark 的指标定义和 memory package 契约；
4. 可用原始输入（受控）；
5. 若干示例 memory package；
6. 代码执行环境。

agent 需要输出：memory schema、memory construction script、query/tool 接口、README、
memory export format、evidence format。产物必须是一个能过 validator 的
`agent_designed` memory package（+ 构建/查询代码）。

### 7.3 评测闭环

agent 产出的 package 流经 **同一套 Track 1/2/3 evaluator**：先在我们自己的 held-out
scenes / held-out queries 上评测，再 **零样本迁移到 SG3D / NaVQA**。

### 7.4 约束（防泄漏）

agent 不能访问 test answers / GT labels；不能针对 test query 写死规则；所有方法用相同
原始输入和计算预算；生成的 memory 必须可保存、可复现；query 必须输出 evidence。

### 7.5 Variants（递进）

1. **Prompt-Only**：只看 benchmark spec 设计 schema，不看训练样本。
2. **Few-Example**：可看少量训练用例和错误反馈。
3. **Coding Agent**：写完整 construction + query code。
4. **Iterative**：在 training split 上反复提交、看分、改设计。

### 7.6 评价意义

如果 agent-designed memory 表现好，说明人工设计的 scene graph 不是唯一答案；如果不好，
也有价值——说明当前 agent 还无法自动设计稳定、几何一致、可泛化的 spatial memory。

## 8. 实现 Roadmap（概要，细节见 `agentic_eval_plan.md`）

- Phase 0：冻结 3-track + agent-designed 契约（validator / spec / 示例 package）。
- Phase 1：Track 1（object location + accounting），fixed_api + tool_llm。
- Phase 2：Track 2（ScanRefer），先 fixed_api 再 tool_llm。
- Phase 3：Track 3（OpenEQA on ScanNet + HM3D），fixed_api + tool_llm，LLM judge 隔离。
- Phase 4：Agent-Designed Memory Baseline harness（workspace → build → eval → iterate）。
- Phase 5：零样本迁移到 SG3D / NaVQA（含 long-horizon / temporal extension）。

## 9. 预期贡献

1. agent-centric spatial-memory benchmark，分层 object → instance → general。
2. 统一的 Fixed-API + Tool-LLM 评测协议，把方法原生 tools 公平开放给 LLM。
3. 系统评估 memory quality / query efficiency / QA accuracy / evidence correctness /
   agentic usability。
4. 比较多种 memory paradigm（no-memory control、caption memory、object graph、
   hierarchical graph、tool-based scene graph）。
5. **Agent-Designed Memory Baseline**：探索 agent 是否能根据任务与评价体系自行设计
   memory，并零样本迁移到 SG3D / NaVQA。

## 10. 核心 Claim

Spatial memory 的评测不应只依赖固定 query interface。一个真正有用的 spatial memory
应该能作为 agent 的外部资源被检索、验证和组合。更进一步：如果 memory 是为 agent 服务
的，那么 memory 的形式也可以由 agent 根据任务和评价体系自动设计。最终目标不是给现有
方法排名，而是重新定义：什么样的 spatial memory 对 embodied agent 才是真的有用。
