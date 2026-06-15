Beyond Fixed Queries: Benchmarking Spatial Memory as an External Resource for Embodied Agents

1. 核心动机
现有 spatial memory 工作已经能够从 posed RGBD sequence 中构建 object map、semantic map、scene graph 或 long horizon memory，并用于导航、问答或规划。但是，现有评测大多仍然是 method centric 的：每个方法自己定义 memory，自己设计 query interface，自己决定 agent 如何访问 memory，最后只看系统能否回答某些问题或完成某些任务。
这种评测方式有一个关键缺陷：它无法判断一个 spatial memory 本身是否真的对 agent 有用。一个 memory 可能已经包含了正确的 object、位置、属性和关系，但由于固定接口无法表达复杂查询，系统仍然会失败。反过来，一个方法也可能为特定任务设计很强的接口，从而在 benchmark 上表现很好，但这并不代表它的 memory 具有通用可用性。
因此，这个项目的核心不是提出一个新的 spatial memory 方法，而是提出一个新的 benchmark：把 spatial memory 作为 embodied agent 可以访问的外部认知资源来评估。我们不再只问“这个方法能不能通过固定接口回答问题”，而是问：
当 agent 拿到一个 memory system 的 map、代码、原始输入和自带接口后，它能不能自主理解、读取、查询、验证和组合这些资源，从而完成空间问答和空间推理？
这就是 Beyond Fixed Queries 的核心含义。
2. 与已有工作的关系
已有 spatial memory 方法可以分成几类。
2.1 No Explicit Memory
代表 baseline：
1. Multi Frame VLM
2. LLM with Captions
这类方法不构建显式 spatial memory。Multi Frame VLM 直接从输入序列中采样若干帧，让 VLM 根据图像回答问题。LLM with Captions 则先把图像或视频片段转成 caption，再让 LLM 基于 caption 回答。
这类 baseline 很重要，因为它回答一个基础问题：
如果直接让 VLM 看多帧图像就可以回答问题，那么 explicit spatial memory 的必要性就会变弱。
预期它们在颜色、外观、简单属性问题上可能表现不错，但在长序列、多物体一致性、精确定位和复杂空间关系上会受限。
2.2 Caption Retrieval Memory
代表方法：
1. ReMEmbR  https://arxiv.org/abs/2409.13682
这类方法把机器人经历压缩成文本 memory，并结合位置和时间做 retrieval。它适合 long horizon QA，但缺少显式 object graph 和精确 3D grounding。
它适合作为 caption based episodic memory baseline，用来比较“文本化经历”与“显式空间结构”的差异。
2.3 Object Centric Memory
代表方法：
1. ConceptGraphs  https://arxiv.org/abs/2309.16650
这类方法把场景表示成 object instances。每个 object 可以包含类别、位置、视觉特征、语言描述和开放词表语义。
它们适合 object location query 和 referring query，但 object merging、空间关系、room structure 和 agent access 的评估仍然不充分。
2.4 Hierarchical Scene Graph Memory
代表方法：
1. HOV SG  arxiv.org/pdf/2403.17846
2. Hydra  https://www.roboticsproceedings.org/rss18/p050.pdf
这类方法把场景组织成 object、room、region、floor 等层级结构。它们更适合空间推理和导航，但多数方法仍然通过预定义接口被使用和评测。
2.5 Online Semantic Memory
代表方法：
1. DualMap  arxiv.org/pdf/2506.01950
这类方法强调在线构建、轻量化、query efficiency 和自然语言导航。它适合作为在线 spatial memory baseline，尤其适合比较 memory size、time per frame 和 query latency。
2.6 Tool Based Scene Graph Memory
代表方法：
1. DAAAM https://arxiv.org/pdf/2512.00565
DAAAM 构建 4D scene graph，并通过 tool calling agent 进行问答。它很适合作为 highly engineered tool based memory baseline。第一版可以只使用它的 spatial 和 descriptive 能力，暂时不评估 temporal QA。
2.7 Retrieval Based Spatial Memory
代表方法：
1. SpatialRAG
这类方法代表 practical RAG style spatial memory。它适合和 LLM agent 结合，但 spatial structure 可能不如 scene graph 显式，geometric consistency 也不一定稳定。
3. Benchmark 的核心设定
3.1 目标
这个 benchmark 不提出新的 memory representation，而是提出新的 evaluation protocol。
目标是系统评估：
1. memory 是否能从 posed RGBD sequence 中在线构建
2. memory 是否准确、紧凑、低冗余
3. memory 是否能支持 object location query
4. memory 是否能支持 fine grained referring query
5. memory 是否能支持 general spatial QA
6. agent 是否能自主使用 memory system 中的 map、代码、接口和原始输入
7. 不同 memory paradigm 对 agentic spatial reasoning 的作用有什么差异
3.2 输入
每个方法接收相同输入：
1. RGBD sequence
2. camera poses
3. camera intrinsics
4. timestamps
5. 可选 shared detector 或 segmenter 输出
第一版主要使用离线给定的 posed RGBD sequence，不优先做 active exploration。
3.3 输出
每个方法需要导出一个 Full Memory Package。
它可以保留原方法的原生格式，但需要把所有可用资源打包给 agent。
Full Memory Package 包括：
1. 构建好的 map 或 memory
2. object map、scene graph、semantic map、region map、relation graph、database 等
3. 原始 RGBD frames、poses、intrinsics、timestamps
4. keyframes、object crops、visual evidence
5. 方法原始代码
6. 方法自带 query interface
7. visualization tools
8. schema 说明文档
9. data loading scripts
10. helper functions
不允许提供：
1. GT annotations
2. benchmark answers
3. test labels
4. 为测试问题手写的规则
5. 在线设置中未来时刻的信息
3.4 Full Agent Access
本 benchmark 采用统一的 Full Agent Access 设定。
也就是说，不再人为限制 agent 只能调用固定 API，而是把 memory system 能提供的全部非 GT 信息提供给 agent。
agent 可以自己决定：
1. 调用方法自带接口
2. 读取 map 文件
3. 读取 object table 或 scene graph
4. 检查 keyframes 或 object crops
5. 阅读代码和 schema
6. 写 Python 或 SQL 查询
7. 组合 memory、原始输入和代码结果进行推理
8. 输出答案和 supporting evidence
这个设定直接测试：
一个 spatial memory system 是否足够清楚、完整、可读取、可查询、可验证，能够被通用 agent 当作外部认知资源使用。
4. Evaluation Tracks
4.1 Track A：Memory Construction
目标：
评估 memory 是否能从 posed RGBD sequence 中在线增量构建。
推荐数据集：
1. ScanNet++
核心问题：
1. memory 有没有覆盖场景里的 object
2. object location 是否准确
3. memory 是否重复太多
4. memory 是否轻量
5. 每帧更新是否高效
指标：
1. Detector Coverable Object Memory Recall
2. False Memory Ratio
3. Memory Redundancy Ratio
4. Object Localization Error
5. Memory Size
6. Time Per Frame
4.2 Track B：Basic Object Location Query
目标：
评估 memory 是否能回答基础物体位置问题。
示例：
where is the chair
推荐数据集：
1. ScanNet++
输出：
1. object id
2. 3D position
3. supporting evidence
指标：
1. Top 1 Object Accuracy
2. Top K Object Recall
3. Position Error
4. Room Correctness
5. Query Latency
6. Evidence Correctness
4.3 Track C：Fine Grained Referring Query
目标：
评估 memory 是否能回答更具体的 object query。
示例：
where is the red table near the window
推荐数据集：
1. ScanRefer on ScanNet
这一部分测试 memory 是否能处理：
1. 颜色
2. 材质
3. 属性
4. 空间关系
5. object disambiguation
指标：
1. Top 1 Accuracy
2. Top K Recall
3. 3D IoU
4. Center Distance
5. Query Latency
6. Attribute Grounding Accuracy
7. Evidence Correctness
4.4 Track D：General Spatial QA
目标：
评估 agent 是否能使用 Full Memory Package 回答更开放的空间问题。
推荐数据集：
1. OpenEQA
问题类型：
1. object attribute
2. object color
3. object material
4. object state
5. spatial relation
6. room level reasoning
7. functional reasoning
8. general scene understanding
示例：
what color is the chair
what is next to the table
where can I sit
is there anything blocking the door
指标：（最后不一定都用，太多metric会有点混乱，可以选有代表性的）
1. LLM Match
2. Category Accuracy
3. Supporting Evidence Correctness
4. Memory Usage Rate
5. Raw Input Fallback Rate
6. Code Usage Rate
7. Query Cost
8. End to End Latency
9. Failure Mode Distribution
4.5 Track E：Long Horizon Spatial QA
目标：
测试 memory 在 long horizon robot experience 中是否仍然有用。
推荐数据集：
1. OC NaVQA static subset
第一版保留：
1. spatial questions
2. descriptive questions
第一版剔除：
1. when questions
2. how long questions
3. before or after questions
4. last seen questions
5. duration questions
原因：
第一版聚焦 spatial memory，而不是 spatio temporal memory。Temporal QA 可以作为后续 extension。
5. Baseline 选择
第一版 benchmark 应该覆盖以下 memory paradigm。
5.1 Multi Frame VLM
类型：
No explicit memory baseline
作用：
直接让 VLM 看多帧图像回答问题，用来判断是否真的需要 explicit spatial memory。
预期优势：
1. 实现简单
2. 对颜色和外观问题可能有效
3. 不需要复杂建图
预期弱点：
1. 长时序不稳定
2. object identity 难以保持
3. 缺少 3D grounding
4. 精确定位能力弱
5. token 和 latency 成本高
5.2 LLM with Captions
类型：
Weak textual memory baseline
作用：
把图像或视频片段转成 caption，再让 LLM 回答。它可以作为 ReMEmbR 之前的简单 caption baseline。
5.3 ReMEmbR
类型：
Caption retrieval memory
作用：
代表 long horizon caption based episodic memory。它不是 explicit scene graph，但可以作为非结构化 memory baseline。
5.4 ConceptGraphs
类型：
Object centric graph memory
作用：
代表 open vocabulary object graph memory，适合 object query 和 referring query 对比。
5.5 HOV SG
类型：
Hierarchical scene graph memory
作用：
代表 floor、room、object 层次化 open vocabulary scene graph。
5.6 Hydra
类型：
Traditional real time scene graph memory
作用：
代表传统实时层次化 3D scene graph 系统，适合 construction efficiency 和 graph memory 对比。
5.7 DualMap
类型：
Online open vocabulary semantic memory
作用：
代表在线构建、轻量化和自然语言 query 的路线。
5.8 DAAAM
类型：
Tool based 4D scene graph memory
作用：
代表高度工程化的 tool calling scene graph memory。第一版主要用它的 spatial 和 descriptive 能力。
5.9 SpatialRAG
类型：
Retrieval based spatial memory
作用：
代表 practical RAG style spatial memory baseline。
6. 核心指标
6.1 Memory Quality
评估 memory 本身。
指标：
1. Object Recall
2. False Memory Ratio
3. Redundancy Ratio
4. Localization Error
5. Memory Size
6. Time Per Frame
6.2 Query Quality
评估 memory 是否能被有效查询。
指标：
1. Top 1 Accuracy
2. Top K Recall
3. Position Error
4. Query Latency
5. Evidence Correctness
6.3 Agentic Usability
评估 memory 是否真的适合 agent 使用。
指标：（这些指标可以一开始dubug用，最后选有代表性的就可以）
1. Answer Accuracy
2. Supporting Evidence Correctness
3. Memory Usage Rate
4. Raw Input Fallback Rate
5. Code Usage Rate
6. Tool Call Count
7. Token Cost
8. End to End Latency
其中最关键的是：
Memory Usage Rate
衡量 agent 的答案有多少主要依赖 memory 得到。
Raw Input Fallback Rate
衡量 agent 有多少问题需要回到原始 RGBD frames 或 keyframes 才能回答。
如果这个值很高，说明 memory 本身没有充分保存可用信息。
Code Usage Rate
衡量 agent 有多少问题需要阅读代码或写查询脚本。
如果这个值很高，说明 memory 虽然可能信息完整，但使用成本较高。
Evidence Correctness
衡量 agent 是否真的使用了正确 object、relation、region、keyframe 或 crop。
这个指标可以避免 agent 靠语言先验猜对答案。
7. Failure Mode Analysis
每个失败样本需要分类。建议类别：
1. memory 中缺少目标 object
2. object 位置错误
3. object attribute 错误
4. spatial relation 缺失
5. memory 中存在重复 object，导致 agent 选错
6. memory schema 太难理解
7. agent 没有找到正确 evidence
8. agent 找到 evidence 但推理错误
9. agent 主要依赖 raw input 才找到答案
10. final answer 格式错误
11. evaluator 判断存在歧义
这个分析很重要，因为 benchmark 的目标不是只给方法排序，而是诊断 spatial memory 对 agent 到底哪里有用、哪里不够用。
8. Additional Step：Agent Designed Memory Baseline
在完成主要 benchmark 后，可以加入一个更有意思的 baseline：
让 agent 根据部分训练用例和评价体系，自己设计 memory representation。
8.1 动机
当前 spatial memory 方法大多是高度模块化、工程化的系统：
1. segmentation
2. tracking
3. object merging
4. scene graph construction
5. relation extraction
6. query interface
7. QA module
但如果 memory 最终是服务于 agent，那么一个自然问题是：
memory 的形式是否也可以由 agent 根据任务和评价体系自己设计？
这可以作为一个非常有意思的 baseline，而不是传统意义上的 hand designed spatial memory 方法。
8.2 设置
给 agent 提供：
1. 部分 training scenes
2. 部分 training queries
3. benchmark metrics
4. 可用原始输入
5. 可用 detector 或 segmenter
6. 若干示例 memory package
7. 代码执行环境
agent 需要输出：
1. memory schema
2. memory construction script
3. query script
4. README
5. memory export format
6. evidence format
然后在 held out scenes 和 held out queries 上测试。
8.3 约束
为避免泄漏，需要限制：
1. agent 不能访问 test answers
2. agent 不能访问 test GT labels
3. agent 不能针对 test query 写死规则
4. 所有方法使用相同原始输入
5. 所有方法使用相同计算预算
6. 生成的 memory 必须可保存、可复现
7. query 过程必须输出 evidence
8.4 可能的 variants
1. Prompt Only Memory Designer
agent 只根据 benchmark spec 设计 memory schema，不看训练样本。
2. Few Example Memory Designer
agent 可以看少量训练用例和错误反馈。
3. Coding Agent Memory Designer
agent 可以写完整 memory construction 和 query code。
4. Iterative Agent Memory Designer
agent 可以在 training split 上反复提交、看分数、修改 memory 设计。
8.5 评价意义
这个 baseline 可以回答一个更深的问题：
如果 spatial memory 是为 agent 服务的，那么 memory representation 是否应该由 agent 和任务共同决定？
如果 agent designed memory 表现好，说明现有人工设计的 scene graph 不是唯一答案。
如果表现不好，也有价值，说明当前 agent 还不能自动设计稳定、几何一致、可泛化的 spatial memory。
9. 推荐实现计划
Phase 1：Benchmark Specification
先确定：
1. input format
2. Full Memory Package 格式
3. Full Agent Access 规则
4. metrics
5. dataset split
6. evidence 格式
7. failure mode taxonomy
Phase 2：Memory Construction Evaluation
在 ScanNet++ 上实现 Track A。
优先跑：
1. HOV SG
2. DualMap
3. ConceptGraphs
4. SpatialRAG
目标：
先评估 memory 本身，不接 OpenEQA。
Phase 3：Object Query Evaluation
实现 Track B 和 Track C。
使用：
1. ScanNet++ 做 basic object location
2. ScanRefer 做 fine grained referring query
目标：
评估 memory 是否支持 object level query 和 attribute based query。
Phase 4：OpenEQA Agent Evaluation
实现 Track D。
让同一个 agent 使用不同 baseline 的 Full Memory Package 回答 OpenEQA 问题。
目标：
评估 memory 是否真的能被 agent 使用，而不是只看方法自带接口。
Phase 5：OC NaVQA Static Subset
实现 Track E。
只做：
1. spatial questions
2. descriptive questions
目标：
测试 long horizon setting。
Phase 6：Agent Designed Memory Baseline
给 agent 部分训练用例和评价体系，让 agent 自己设计 memory schema、construction code 和 query code。
目标：
测试 agent 是否能提出比人工模块化系统更适合自身使用的 memory representation。
10. 预期贡献
Contribution 1
提出一个 agent centric spatial memory benchmark。
Contribution 2
提出 Full Agent Access 评测设定，把 map、代码、原始输入和方法自带接口全部开放给 agent。
Contribution 3
系统评估 memory quality、query efficiency、QA accuracy、evidence correctness 和 agentic usability。
Contribution 4
比较多种 memory paradigm，包括 Multi Frame VLM、caption memory、object graph、hierarchical graph、online semantic map、tool based scene graph 和 retrieval based spatial memory。
Contribution 5
提出 Agent Designed Memory Baseline，探索 agent 是否可以根据任务和评价体系自行设计 memory representation。
11. 第一版 Scope
第一版包含：
1. online posed RGBD memory construction
2. object location query
3. fine grained referring query
4. general spatial QA
5. long horizon static spatial QA
6. Full Agent Access evaluation
7. memory size
8. time per frame
9. query latency
10. evidence correctness
11. agent designed memory baseline
第一版暂时不做：
1. temporal QA
2. dynamic object tracking
3. downstream navigation
4. downstream manipulation
5. full mobile manipulation planning
12. 核心 Claim
这个 benchmark 想证明：
Spatial memory 的评测不应该只依赖固定 query interface。一个真正有用的 spatial memory 应该能作为 agent 的外部资源被读取、查询、验证和组合。
进一步地，它还探索：
如果 spatial memory 是为 agent 服务的，那么 memory 的形式是否也可以由 agent 根据任务和评价体系自动设计。
最终目标不是简单给现有方法排名，而是重新定义：
什么样的 spatial memory 对 embodied agent 才是真的有用。