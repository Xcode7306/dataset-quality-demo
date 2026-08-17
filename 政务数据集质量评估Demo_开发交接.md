# 政务数据集质量评估 Demo：开发交接

> 当前主题：v1.1 评估前规则生成、默认指标补充规则与规则文件批量导入  
> 最后更新：2026 年 8 月 12 日  
> 当前状态：v1.1 已完成实现、验收、本地提交和 GitHub 独立分支上传，新增评估前指标补充规则、首页多轮大模型规则对话、TXT/Markdown/CSV/Excel/JSON/JSONL/NDJSON 批量导入、逐条前置澄清、全批次 RulePack、统一试运行和批准后最终评估；后续跟进移除了初始页 RAG 的用户文档上传、来源确认和摄取入口，保留项目预置标准检索；326 项自动化测试、规则/RAG Harness 与真实 Streamlit 随机端口健康检查通过。当前分支为 `codex/v1.1`，提交为 `2cda777`，草稿 PR #9 为 `codex/v1.1 → codex/v1.0`；远端 `codex/v1.0`/PR #8 保持不变

## 0. 当前版本交接摘要

v0.6 **直接以 v0.4 为基础开发，不合并或继承 v0.5 的报告历史、跨版本比较、整改计划和治理记录能力**。保留 v0.4 的 13 项指标，并加入 DB31/T 1523-2024 的 30 项独立指标定义；总目录为 43 项。v0.7 在此基础上新增规则编制 Agent，v0.8 新增自然语言自定义规则，v0.9 新增标准依据 RAG，v0.9.1 新增 Agent Harness，v1.0 新增规则编制工作流历史与失败恢复，v1.1 将规则生成和完整性检查前移到最终评估之前并支持文件批量导入，但均不改变既有指标、风险和报告口径。

当前网页流程与用户体验：

- 初始页在主内容区显示统一的 43 项指标卡，默认勾选基础 13 项；卡片不再显示“原有 / 新增 / DB31/T”等来源标识。
- 每张卡片右上角有圆圈“？”；悬停时以 CSS 浮层即时显示指标含义，并同时显示计算方式和“当前可直接计算 / 需补充评价标准”。
- 提供“默认指标”“全部指标”“清空选择”三个统一操作，用户可逐项自由组合。指标目录表展示名称、含义、维度、计算方式和当前能力，但不展示来源、标准代码、层级或内部 ID。
- 点击“运行质量评估”后，指标选择区自动隐藏，只显示评估结果及下载/解读/规则增强内容；报告摘要会列出具体需要补充标准的指标。
- 用户在上传控件中点击文件右侧“×”移除数据集，或替换上传文件时，旧报告、Agent 状态和 RulePack 状态会清除，页面自动恢复到初始指标选择页。
- 每张指标卡片下新增“评价依据”输入栏；已有确定性规则的 15 项指标预填可编辑的“默认：……”评价依据，其余 28 项指标保持空白。用户勾选指标但未填写评价依据时，页面列出缺失指标并禁用“运行质量评估”，补全后才能启动。
- 生成报告后，新增“补充评价标准”页；报告摘要和该页面都会明确列出需要外部标准的指标及其 `required_inputs`，并优先展示这些指标。用户在页面补充标准后，继续执行“AI 解析 → 确定性试运行 → 批准并重新评估”；该页面与指标卡片复用同一份可编辑评价依据。
- 初始页面侧栏提供“大模型 API 配置”：API 地址、API Key 和模型名称。填写后同时用于只读报告诊断 Agent 和补充评价标准 Agent；页面自定义模型使用兼容 Chat Completions 的普通消息模式，不要求工具调用。未填写 API Key 时仅保留本地模板暂行演示。
- v0.8 首批在五类既有规则上新增 `regex_format`、`string_length`、`conditional_required`、`field_comparison`。模型只生成候选草案；审批、试运行和正式执行由本地确定性代码完成。正则限制模式长度、输入长度并拒绝高回溯结构。
- 未配置 API Key 时使用本地模板 Provider；一旦配置外部模型，调用或解析失败直接在页面报错，不再使用模板伪装结果；缺少字段、阈值、条件值或比较运算符时进入澄清。跨表参照、Python / SQL / 任意代码、自动清洗和数据写回明确不支持。
- v0.9 在报告页提供“标准依据 RAG”页签，默认检索项目预置的 Markdown 标准语料；页面不再提供用户上传、来源确认或摄取入口。RAG 仍可继续检索和绑定依据。
- RAG 只负责预置文档分段、确定性检索和来源引用，不会自动把文档变成已批准规则，也不会直接修改指标、风险或报告。检索结果必须由用户选择后，才可绑定到 `RuleEvidence`、`RuleDraft` 或 `RulePack`。
- 评价前已批准的文档和检索绑定会在上传业务数据、首次运行评估时保留；更换已有报告的输入或评价范围时，失效的报告派生状态与绑定会按当前输入重新清理。RAG 数据目前只保存在本地 Streamlit 会话内存，不是跨会话知识库。
- v0.9 仍不强制引入 LangGraph、Dify 或向量数据库；扫描件/图片型 PDF、自动标准符合性判定和无来源的规则声称均不支持。
- v0.9.1 新增独立 Agent Harness，不改变网页业务流程、指标计算、风险口径或规则审批边界。Harness 对规则输出、字段/参数映射、试运行与正式执行一致性、RAG 引用有效性和重复执行语义指纹做离线回归，并保留可比较的 Prompt / Provider / Model 元数据。
- v1.0 使用确定性状态机统一目录指标和自定义规则路径；工作流绑定 `workflow_id`、报告哈希、输入哈希、参考日期、指标集合和请求指纹。模型不能选择状态、审批、正式执行或恢复点。
- Provider、试运行或正式执行失败后，只能针对同一请求指纹恢复到失败前状态一次；正式执行失败保留原已审批 RulePack，页面刷新或重复提交不会生成第二份审批或执行结果。
- 规则工作流历史最多保存 20 条脱敏摘要，只存在当前 Streamlit 会话内存，可查看、下载和清空；不保存自然语言原文、原始上传字节或 API Key，也不提供数据库、跨会话恢复、登录鉴权或跨用户共享。
- v1.1 在初始页提供“评估前 AI 规则生成”：可为已选目录指标填写补充规则，也可在首页与大模型多轮对话创建一条自然语言规则，或导入规则文件；最终报告生成前先临时解析数据结构和脱敏画像，不展示或固化评估结果。
- 规则文件支持 TXT/Markdown 每个非空行一条、CSV/Excel“规则描述”列与可选“指标ID”列、JSON 字符串/对象数组、JSONL/NDJSON；单文件 2 MiB、最多 100 条，重复规则去重，未知指标、非法 JSON、重复键和非标准数值明确拒绝。
- 每条描述先进入 Provider 与确定性完整性保护；缺字段、阈值、允许值、更新频率、正则、条件值或比较运算符时按指标或文件行号立即澄清。任一条未通过时整批不构建 RulePack；全部通过后才合并、统一试运行、人工批准并用于最终评估。模型生成了用户未明确写出的字段、类型或取值时，确定性校验拒绝。

DB31/T 指标采用严格口径：`030300 数据重复率`、`030400 数据唯一性`可在当前单表输入下直接计算；其余指标若缺少数据标准、业务规则、权威参照、跨表关系、授权、访问 SLA 或应用场景等必要依据，必须报告为“无法评估”，不得用启发式代理值冒充标准符合性结论。

主要实现位置：

- `dataset_quality_demo/src/metric_catalog.py`：43 项目录、稳定 ID、计算方式、简明指标含义、默认评价依据和选择规范化；
- `dataset_quality_demo/src/rule_dsl.py`：`RuleDraft`、`RuleSpec`、`RuleEvidence`、Provider 元数据和协议校验；
- `dataset_quality_demo/src/rule_authoring_providers.py`、`src/rule_authoring_service.py`、`src/rule_authoring_tools.py`、`src/rule_authoring_workflow.py`、`src/rule_authoring_coordinator.py`：本地模板/可选兼容模型编译、上下文白名单、规则草案服务、v1.0 显式状态机、有界会话历史和确定性协调层；
- `dataset_quality_demo/src/rule_batch.py`：v1.1 安全规则文件解析、规则输入来源模型、逐条编译结果、全批次通过语义、RuleDraft 合并与 RulePack 构建；
- `dataset_quality_demo/src/rule_authoring_prompts.py`、`src/rule_authoring_harness.py`、`src/rule_authoring_trace.py`、`run_agent_harness.py`：版本化提示词、规则/RAG 金标执行、Provider 对比、语义重放、隐私化 Trace 和命令行验收入口；
- `dataset_quality_demo/src/rag/models.py`、`src/rag/ingestion.py`、`src/rag/retrieval.py`、`src/rag/citations.py`：标准文档模型、Markdown/TXT/文本层 PDF 摄取、分段、版本/命名空间过滤、确定性检索和稳定引用；`src/rule_authoring_service.py` 同时负责将检索证据绑定到规则草案/规则包；
- `dataset_quality_demo/src/model_api.py`、`src/agent_providers.py`：Chat Completions 地址规范化、自定义 API Provider、旧版 DeepSeek 环境变量兼容和不暴露密钥的缓存命名空间；
- `dataset_quality_demo/src/rule_engine.py`、`src/rule_service.py`：审批前 dry-run 摘要入口，复用同一确定性规则计算语义；
- `dataset_quality_demo/app.py`：主内容区指标卡、缺失标准清单、补充评价标准与自定义规则页面、评价依据输入、v0.8 Agent 页面、v0.9 初始页/报告页 RAG 页签、评估前/后页面状态切换及上传文件移除后的自动恢复；
- `dataset_quality_demo/schemas/rag-document.schema.json`、`schemas/rag-search-response.schema.json`、`schemas/rule-evidence.schema.json`、`schemas/rule-draft.schema.json`、`schemas/rule-pack.schema.json`、`schemas/rule-authoring-workflow.schema.json`、`schemas/rule-authoring-history.schema.json`：RAG 文档、搜索结果、规则引用、规则与 v1.0 工作流历史协议；`DB31T_1523-2024_公共数据质量评价指标及计算方式.md` 为默认可检索标准语料；
- `dataset_quality_demo/tests/test_rule_batch_v11.py`、`tests/test_agent_workflow_v10.py`、`tests/test_rag_v09.py`、`tests/test_acceptance_regressions.py`、`tests/test_rule_authoring_v07.py`、`tests/test_rule_authoring_v08.py`、`tests/test_rule_authoring_streamlit.py`：v1.1 文件解析/整批拒绝/猜测拦截/评估前 UI，以及 v1.0 生命周期/Schema/幂等/恢复/历史、RAG 与旧页面闭环回归。
- `dataset_quality_demo/DB31T_1523-2024_指标与v0.6实现口径.md`：DB31/T 指标与计算口径说明。

当前验收命令：

```bash
cd dataset_quality_demo
.venv/bin/python -m unittest discover -s tests -p 'test*.py'
.venv/bin/python run_demo.py --check
```

2026-08-12 v1.1 跟进执行结果：326 项自动化测试通过；规则 Harness 的 Schema 合法率、支持范围判断、字段映射、参数生成、确定性执行和重复执行语义一致率均为 100%，未审批执行数与无依据标准声称数均为 0；RAG Harness 的状态判断与引用有效率均为 100%。新增首页规则对话、无数据时的上传提示和 RAG 用户摄取入口移除回归；`.venv/bin/python run_demo.py --check` 的先前随机本机端口健康检查仍有效。外部模型对比仍须由使用方提供有效 API 配置后单独运行，外部模型不可用不会破坏基础评估。

### 0.0.1 v1.1 评估前规则生成交接记录（2026-08-10）

1. `src/rule_batch.py` 新增 `RuleBatchInput`、`RuleBatchPreflight` 与安全文件摄取；文件只提供自然语言描述，不接受已批准状态、任意代码或可直接执行对象。
2. `app.py` 在报告生成前收集三类来源：目录指标补充规则、首页多轮规则对话和规则文件。存在任一规则时，原“运行质量评估”入口切换为“AI 检查并生成规则”；只有整批通过、试运行和人工审批后才执行最终评估。
3. 前置阶段会调用既有上传服务生成仅供当前会话规则上下文使用的报告对象，但不会写入 `quality_report` 或展示最终结果；澄清路径可由测试直接确认 `quality_report` 尚不存在。
4. `inspect_rule_intent` 对九类白名单规则检查用户是否明确提供关键参数；Provider 仍可补充澄清问题，但不能绕过保护。完整描述下如果模型返回错误字段、错误规则类型或未写明取值，错误归因于模型草案并由确定性验证拒绝，不反向要求用户为模型猜测负责。
5. 全批次 RulePack 复用现有审批哈希、报告/输入/参考日期绑定、规则资源上限和确定性执行服务；重复语义规则合并去重，任何失败条目阻止部分成功规则执行。
6. v1.1 已从远端 `codex/v1.0` 提交 `6599a3a` 建立独立分支 `codex/v1.1`，本地与远端头均为 `2cda777`，草稿 PR #9 以 v1.0 为基线；没有改写或强推 v1.0。后续跟进在工作树上加入首页规则对话、移除初始页 RAG 用户摄取控件，并将回归总数更新为 326；这些跟进改动尚未提交或推送。

### 0.1 当前版本链与提交约定（2026-08-10）

- GitHub 上的 v0.7 已保留在独立提交 `7257e31`（分支 `codex/v0.7-agent-model-fix`）；既有 v0.6 分支和 v0.7 提交均不得覆盖。
- 远端 `codex/v0.8` 已使用 GitHub 连接器提交 `c27318b`（`feat: add v0.8 natural language custom rules`），父提交为 v0.7 的 `7257e31`。它是当前 v0.9 PR 的基准，只包含 v0.8 增量文件。
- 本地旧的 `codex/v0.8` 提交 `a7a09ec` 是从本地 v0.6 工作区一次性提交的累计快照，曾把已扩展的 `rule-evidence.schema.json` 一并带入，因此与远端 `c27318b` 不是同一棵代码树；不能把它作为远端 v0.8 的等价副本。
- 当前本地 `codex/v0.9` 的最后提交为 `6d3da13`（`fix: allow standards RAG before evaluation`）；GitHub PR #7 的远端头为 `64c3c1b`，基于 `codex/v0.8`，状态为 open/draft，目标是评审后合并，不覆盖 v0.8 基线。
- 当前本地工作分支为 `codex/v1.0`，从仅存在于本地的 `6d3da13` 建立；v0.9.1 Harness 与 v1.0 工作流累计增量已提交为 `f21463f`。远端 `codex/v1.0` 从 `codex/v0.9` 的 `64c3c1b` 建立，同一批 22 个文件形成单一提交 `6599a3a`。由于本地旧基线与远端 v0.9 提交树不同，两端提交 SHA 不相同，但增量文件清单、内容与提交说明一致。
- `src/rag/` 内的 `models.py`、`ingestion.py`、`retrieval.py`、`citations.py`，以及 `src/rule_authoring_service.py` 中的 RAG 绑定代码，现已属于 v0.9 实现；对应 Schema、默认标准语料和回归测试一并纳入 PR #7。
- v1.0 已使用 GitHub 连接器完成远端分支、单一提交和草稿 PR，不依赖失效的 GitHub CLI 凭据：远端分支 `codex/v1.0`、提交 `6599a3a`、PR #8（`codex/v1.0 → codex/v0.9`，draft）。随后本地 `codex/v1.0` 已对同一批文件提交为 `f21463f`。
- 远端 `codex/v0.8-rule-authoring` 删除请求已由连接器尝试，但当前连接器不提供删除 Git 引用的接口，GitHub 返回 422；该分支目前仍保留且未包含 v0.8 新提交。
- 本交接文档位于代码仓库外，默认只更新本地文档；若要把它作为仓库文件提交，需另行明确文件路径和提交范围。

### 0.2 v1.0 可演示 Agent 闭环交接记录（2026-08-10）

本轮按已批准 Agent Spec 和后续开发计划，在 v0.9.1 Harness 通过后进入 v1.0。实现范围严格限于本地 Demo 的显式工作流、规则历史、失败恢复和演示文档，没有引入数据库、服务端持久化、登录鉴权、多租户、跨用户共享、LangGraph、Dify 或新的运行时依赖。

核心实现：

1. `src/rule_authoring_workflow.py` 定义与 Spec 一致的确定性状态转换、报告/输入/请求绑定、状态转换审计、一次恢复和审批/执行幂等 ID；非法越级、跨工作流 RuleDraft、未经试运行审批及未经审批执行均被拒绝。
2. `src/rule_authoring_coordinator.py` 统一目录指标和自定义规则的“编译 → 校验 → 试运行 → 等待审批 → 审批 → 正式执行”调用链。上传字节只临时传给既有规则服务，不进入工作流或历史对象。
3. Provider、试运行和正式执行失败进入 `failed`，并记录准确的失败前状态。恢复必须复用同一请求指纹且整个工作流最多一次；正式执行失败继续保留原 `approved_pack` 与 `approval_id`，恢复后无需再次审批。
4. `RuleAuthoringHistory` 在当前 Streamlit 会话内最多保留 20 条去重摘要；摘要包含目标、上下文哈希、状态、转换、草案/试运行哈希、证据 ID、审批 ID 与执行 ID，不含用户自然语言原文、原始上传字节或 API Key。页面支持查看、JSON 下载和显式清空。
5. `schemas/rule-authoring-workflow.schema.json` 与 `schemas/rule-authoring-history.schema.json` 使用 Draft 2020-12 严格约束并拒绝未知字段；历史 Schema 通过相对引用复用工作流 Schema。
6. `app.py` 的目录指标和自定义规则页面保留既有交互标签及兼容状态字段，但状态推进已改由 v1.0 协调层控制；完成执行后不再渲染审批入口，失败时只渲染一次同请求恢复入口。
7. `tests/test_agent_workflow_v10.py` 覆盖完整自然语言到确定性报告闭环、Schema、规则/依据/结果对应、非法状态、验证澄清、同指纹一次恢复、改变请求拒绝、执行失败保留审批、有界脱敏历史和 Streamlit 单次执行；旧规则、RAG、Harness 和页面测试继续作为回归。

必须保持的 v1.0 边界：

- 状态只能由本地代码转换；模型只能返回候选 RuleDraft，不能审批、执行、决定恢复点或伪造工具结果。
- `QualityReport` 仍是基础事实源，规则增强结果保持独立；v1.0 不修改 43 项指标、风险阈值、报告 Schema、资源上限或输入类型。
- 当前“历史”是规则工作流脱敏摘要，不是 v0.5 的 QualityReport 历史、跨版本比较或治理记录，也不是生产审计档案。
- 页面输入、数据、指标、日期、规则或 RAG 片段变化后，旧活动状态不能复用；历史记录只用于回看，不能直接重新执行。
- v0.9.1 与 v1.0 已累计提交到独立 `codex/v1.0`；远端以 `codex/v0.9` 为基线，提交为 `6599a3a`，草稿 PR #8 包含 1 个提交、22 个文件、5570 行新增和 281 行删除。不得直接覆盖既有 v0.9 PR 或 v0.8 基线。

专项验收命令：

```bash
cd dataset_quality_demo
.venv/bin/python -m unittest tests.test_agent_workflow_v10
.venv/bin/python -m unittest discover -s tests -p 'test*.py'
.venv/bin/python run_demo.py --check
```

2026-08-10 最终验收：上述专项测试与全量 320 项测试全部通过，`run_demo.py --check` 在随机 `127.0.0.1` 端口启动 Streamlit、健康端点返回 `ok` 后正常停服。

### 0.3 v0.9.1 Agent Harness 交接记录（2026-08-10）

按《开发计划表》和 Agent Spec 的版本链，v0.9 后的直接开发版本是 **v0.9.1 Agent Harness**，不是 v1.0。v1.0 依赖 Harness 先证明规则生成、检索引用、工具边界和重复执行具备可回归性。

本轮交付如下：

1. `quality-rule-authoring-v0.9.0` 与 `quality-rule-authoring-v0.9.1` 提示词进入版本注册表并拥有稳定指纹；模板和外部 Provider 均记录实际 Prompt 版本，便于同一金标集做版本/模型对比。
2. 外部模型规则输出只接受单个严格 JSON 对象，拒绝 Markdown 包裹、重复键、非有限数、非法 Unicode、未知字段、伪造工具调用、审批或执行字段；解析失败不会自动退回模板冒充外部模型结果。
3. 规则编制工具限制为六个只读/校验型白名单入口：指标定义、画像摘要、字段列表、规则证据检索、草案校验和试运行。审批与正式执行不属于模型工具，只能由本地确定性流程在显式批准后触发。
4. `harness/goldens/rule_authoring_cases.json` 为 9 类受支持规则分别提供正常、边界和错误/澄清路径，并补充 Python/跨表请求拒绝，共 29 例；`harness/goldens/rag_retrieval_cases.json` 覆盖成功、无结果、版本冲突、未审核、过期和错误版本，共 6 例。
5. 每个用例保存 Provider、Model、Prompt、状态迁移、工具调用、检索 chunk、草案语义哈希、校验、试运行、审批、正式执行、重试/失败等 Trace。自由文本参数只留长度和 SHA-256，错误中的密钥/Bearer Token 会脱敏；Trace 受 `schemas/rule-authoring-trace.schema.json` 严格约束。
6. Harness 对每个规则用例重复执行并比较排除时间戳、耗时和临时 ID 后的语义指纹；正式执行前必须先通过 Schema、确定性校验、试运行和本地显式审批，并核对试运行与正式执行结果一致。

验收命令：

```bash
cd dataset_quality_demo
.venv/bin/python run_agent_harness.py
.venv/bin/python -m unittest discover -s tests -p 'test*.py'
.venv/bin/python run_demo.py --check
```

`run_agent_harness.py` 默认运行本地模板 Provider 和 RAG 金标；可重复传入 `--provider` 比较多个 Provider，并通过 `--output` 原子写出完整 JSON Trace 报告。使用 `deepseek` 或 `custom` 时须显式提供对应环境配置；无凭据、超时或模型异常必须报告失败，不能影响无模型的基础质量评估。

本节保留 v0.9.1 当时的交接结论。当前 `codex/v1.0` 已在同一工作树上继续完成上方 0.2 节能力；v0.9.1 与 v1.0 已作为同一累计增量发布到远端提交 `6599a3a` 和草稿 PR #8，评审时必须据此识别差异，不应把它误拆为已有远端 v0.9.1 基线。

### 0.4 v0.9 当前交接记录（2026-08-09）

v0.9 的正确使用顺序是：**先运行评估，再检索项目预置标准依据，最后把检索到的条款绑定到规则证据**。当前页面不提供用户上传、来源确认或摄取控件；RAG 不会自动创建或批准规则。

1. 上传业务 CSV、Excel、JSON、JSONL、GeoJSON 或 ZIP，选择指标并运行质量评估。
2. 在报告页的 RAG 页签中输入问题/条款关键词，可按标准号、版本和命名空间筛选；对结果进行人工确认后，绑定到 `RuleEvidence`、`RuleDraft` 或 `RulePack`。每条引用需要保留文档、章节/条款、chunk、页码（如有）和内容快照信息。
3. 若检索无结果、存在版本冲突或来源不适用，页面必须提示用户补充/筛选；不能把“没有检索到依据”表述成“符合标准”。

当前 v0.9 的边界：检索是本地、确定性、可测试的证据发现能力，不依赖外部模型或向量数据库；文档和绑定只存在当前 Streamlit 会话内存。大模型仍只生成候选规则草案，规则审批、试运行、正式执行、指标计算和风险判定由本地确定性代码完成。

v0.9.1 Agent Harness 已按上述 0.3 节完成本地实现和验收；在 PR #7 与 v0.9.1 变更完成评审/合并前，不把草稿分支描述为已发布主线版本。

### 0.5 历史 v0.5 交接记录（非当前 v1.0 范围）

v0.5 在 v0.3 单报告只读诊断与 v0.4 本地业务规则闭环之外，增加了一套完全本地、确定性的历史比较与整改记录流程。报告只有在用户显式保存当前结果或显式导入固定 `QualityReport` JSON 后才进入当前 Streamlit 会话内存；用户选择两份历史、填写相同 `dataset_series_id` 并明确确认属于同一治理对象后，比较器才生成绑定两份报告哈希的 `ReportComparison`。页面随后可生成本地模板 `RemediationPlan`、人工分派负责人/截止日期/状态，并导出绑定比较、计划和两份报告哈希的 `GovernanceRecord`。

产品 `VERSION` 已更新为 `0.5`，指标引擎 `ENGINE_VERSION` 继续为 `0.4`。本轮没有改变原 13 项指标、默认阈值、风险判定或既有固定 `QualityReport` 哈希；`QualityReport`、`AgentAnalysis`、`RulePack`、`ReportComparison`、`RemediationPlan` 和 `GovernanceRecord` 保持独立。

本轮新增的主要协议和模块：

- `src/history_store.py`：严格 `QualityReport` 导入/校验、会话内存历史、容量/删除策略和不含原始值的版本趋势；
- `src/comparison_models.py` 与 `src/comparison_service.py`：绑定两份固定报告、同一治理对象声明、指标/风险/可评估性/结构变化和兼容性限制的确定性比较；
- `src/remediation.py`：从比较证据生成本地确定性改进摘要、可分派整改任务与治理记录；
- `src/comparison_presentation.py`：比较/行动计划/治理记录的安全序列化，行动计划提供 JSON、Markdown 和 CSV；
- `schemas/report-comparison.schema.json`、`schemas/remediation-plan.schema.json`、`schemas/governance-record.schema.json`：v0.5 三类独立严格协议；
- `app.py`：显式保存/导入、会话历史列表和删除、版本趋势、同一治理对象确认、报告比较、整改任务人工分派及下载。

必须保持的 v0.5 边界：

1. 历史仅存在于当前本地浏览器会话内存，最多 20 份、单份 8 MiB、总计 32 MiB；退出会话即失效，支持逐份删除和全部清空，不提供数据库、登录鉴权、跨用户权限或跨会话恢复。
2. 历史只保存通过严格验证的固定 `QualityReport` 及必要元数据；不另行保存原始上传字节、疑似问题位置 CSV、规则问题位置 CSV、Agent 输出/对话、RulePack 或规则增强对象，当前 Demo 自身生成的报告不导出原始单元格值。
3. 导入只接受 UTF-8 的严格 `QualityReport` JSON，并复核重复键、非标准数值、Unicode、Schema、报告哈希、指标键、风险引用、无法评估项一致性和画像/证据结构；文件名或结构相似不能代替固定报告验证。结构校验无法从语义上证明任意合法文本不含敏感内容，自哈希也不是来源签名或身份认证，因此只应导入当前 Demo 生成或已经人工确认内容与来源可信的报告。
4. 比较必须由用户显式提供相同 `dataset_series_id` 并确认属于同一治理对象；该标识和确认均为本地自声明，不代表认证身份或权威数据目录关系。
5. 改善/恶化只由确定性比较器给出。引擎或报告 Schema 不兼容、指标定义变化、阈值/风险规则变化、关联指标不可评估、业务规则签名不同等情况必须标记为不可比较；不得用风险消失代替“已解除”。
6. 指标方向使用固定白名单；数据集规模、IQR 等中性指标只标记变化。可评估性变化独立记录，不能等同于质量改善；更新滞后出现未来日期时只标记语境变化，不按数值方向自动判断。
7. `RemediationPlan` 由本地模板生成，最多 30 项任务；人工只能更新负责人、截止日期和状态，不能修改比较证据、关联变化 ID 或确定性优先级。操作者标签是自声明，`GovernanceRecord.operator.identity_verified` 固定为 `false`。
8. DeepSeek 仍仅用于 v0.3 既有单报告只读解读且默认关闭；不接收历史、两份报告、比较、整改任务、人工分派或治理记录，也不扩展为 v0.5 闭环的必需组件。

当前最终验收命令：

```bash
python3 run_demo.py --check
```

本轮最终执行结果为：282 项自动化测试全部通过，5 份固定示例报告与当前 v0.4 指标引擎一致，Streamlit `AppTest` 通过，并在随机 `127.0.0.1` 端口真实启动、通过健康检查后正常停服。外部 DeepSeek 真机调用仍需有效 API Key、元数据外发审批和供应商侧用量核对，不影响本地 v0.5 历史、比较与整改闭环。

后续如建设服务端持久化、身份认证与细粒度权限、跨用户共享、批量或周期性治理，必须先定义租户隔离、容量、保留期限、删除、恢复、审计和授权协议；不能把当前会话内存对象直接宣传为生产治理档案。

## 1. 背景与目标

《政务领域公开数据集下载清单》中标注为 JSON / GeoJSON 的 13 个候选项，现有 Demo 只接受顶层扁平对象或扁平对象列表，对可确认真实样本的直接兼容数为 0。本轮目标是在不改变现有指标、风险和报告口径的前提下，将更多“表格型 JSON”可靠地转为统一 DataFrame。

已确认的代表差距：

| 来源 | 真实结构 | 现有结果 | 对应阶段 |
| --- | --- | --- | --- |
| 温州公开平台 | `[[表头…], [数据…], …]`，部分资源为 ZIP 多分片 | 拒绝非对象列表 | A（单文件）/ B（ZIP 分片） |
| 台州开放平台预览 | `{"code":"1","msg":"成功","data":[[表头…], …]}` | 拒绝嵌套值 | A |
| 深圳积涝点实际下载 | `{[{"…":"…"}, …]}` | 不符合标准 JSON | A（仅定向外壳修复） |
| Sen1Floods11 元数据 | GeoJSON FeatureCollection | 支持按 Feature 映射扁平属性和空间摘要；真实样本须满足白名单 | C |

北京相关“JSON API”返回的是下载元数据，实际数据文件为 CSV / XLSX / RDF，不应为追求统计数而当作记录型 JSON 接入。

## 2. 三阶段交付路线

| 阶段 | 交付内容 | 不包含 |
| --- | --- | --- |
| A：常见表格型 JSON | 二维数组、常见接口包装、JSONL / NDJSON、JSON 多编码、深圳 `{[...]}` 定向修复 | ZIP、GeoJSON、通用嵌套展平、JSON5 容错 |
| B：压缩包与分片 | ZIP 安全预检、分片结构一致性、分片合并、大数据流式约束 | 不受限解压、异构分片静默合并 |
| C：GeoJSON 与有限嵌套 | FeatureCollection 映射、可追溯的白名单路径展平 | 坐标全量展开、自动改变记录粒度 |

## 3. 阶段 A 设计决定

### 3.1 支持的结构

1. 保留现有顶层扁平对象或扁平对象列表。
2. 接受“第一行为表头、后续行为数据”的二维数组。短行按缺失补齐，超过表头宽度的行明确失败，重复表头明确失败。
3. 只递归检查 `data`、`rows`、`records`、`items`、`list`、`result` 六类常见包装键。找到唯一记录候选时提取并 warning；同时找到多个候选时拒绝猜测。
4. `.jsonl` / `.ndjson` 的每个非空行必须是单条扁平对象，错误必须包含行号。
5. 仅在去除整个文件最外层的一对花括号后，内容可严格解析为记录列表时，修复深圳样本的 `{[...]}` 外壳。该路径必须 warning，其他语法错误仍拒绝。

### 3.2 编码和严格性

- 编码检测顺序：UTF-8-SIG / UTF-8；有 BOM 的 UTF-16 / UTF-32；GB18030 回退。非 UTF-8 读取必须 warning。
- 所有路径继续拒绝重复键、`NaN` / `Infinity` / `-Infinity`、数值溢出、孤立 Unicode 代理字符和普通嵌套单元格；Unicode 校验覆盖整棵 JSON，包括未映射的包装元数据。
- 表格转换使用对象类型承载 JSON 标量，避免带空值的超大整数被 DataFrame 自动转成浮点数后静默舍入。
- 不引入注释、尾逗号、单引号、未引用键或其他 JSON5 能力。

### 3.3 资源上限

- 单文件 50 MiB；通用 1,000,000 行、10,000 列、20,000,000 单元格；JSON 记录 200,000 条。
- 在 `json.load` 物化前检查嵌套深度、每个数组项数、对象键值对和全文件结构规模；解析为表格后再校验行、列、单元格和单项文本。
- 自动提取、转换、编码回退或定向修复均不能绕过上限。

## 4. 预计影响文件

| 文件 | 变更 |
| --- | --- |
| `dataset_quality_demo/src/parser.py` | 编码检测、结构预检、形态识别、包装提取、JSONL 读取与统一 warning。 |
| `dataset_quality_demo/src/resource_limits.py` | 补充 JSON 数组与数组项总量上限。 |
| `dataset_quality_demo/app.py` | 上传扩展名、支持范围说明。 |
| `dataset_quality_demo/tests/` | 新增真实结构、严格边界和资源上限回归。 |
| `README.md`、`DEMO_GUIDE.md`、输入输出协议 | 同步对外能力与不支持范围。 |

## 5. 阶段 A 验收清单

- [x] 二维数组、接口包装、JSONL / NDJSON、非 UTF-8 和 `{[...]}` 定向外壳都有成功用例。
- [x] 多候选包装、宽行、重复表头、嵌套单元格和普通非标准 JSON 都有可解释失败用例。
- [x] 重复键、非标准数值、Unicode 和资源上限旧防护不回退。
- [x] Streamlit 能上传 `.jsonl` / `.ndjson`，所有自动处理 warning 能在运行信息中看到。
- [x] 输入输出协议、README、演示指南与实现一致。
- [x] 阶段 A 当时的 `python3 run_demo.py --check` 通过：131 项测试全部成功，固定报告一致，真实 `127.0.0.1` Streamlit 健康检查通过。

## 6. 阶段 A 完成结果

- 解析器已将新增形态统一转为现有 `ParsedDataset` / DataFrame，未修改画像、13 项指标、风险规则或报告结构。
- 补充 4 份可直接演示的阶段 A 样例：`json_matrix_dataset.json`、`json_wrapper_dataset.json`、`json_records_dataset.jsonl` 和 `json_targeted_repair_dataset.json`。
- 测试总数由 118 增至 131；新增专项覆盖真实数据结构、页面上传、严格解码和物化前资源拦截。
- 第二轮复审补齐了大整数保真、包装元数据 Unicode 校验、JSONL 严格空白与逐行错误定位、自动处理 warning 保留，以及重复表头、错误值和歧义包装候选的有界展示；包装搜索在确认第二个候选后即停止。
- 阶段 A 当时保留的 ZIP 分片边界已由阶段 B 完成；GeoJSON FeatureCollection 已由阶段 C 完成；当前仍不支持普通嵌套展平、异构 ZIP 和 JSON5 容错。

## 7. 阶段 B 完成结果

- 新增 `.zip` 输入，只允许包内 `.json` / `.jsonl` / `.ndjson` 普通文件与目录；网页上传、上传服务和 CLI 共用同一解析边界。
- ZIP 在读取分片前检查绝对路径、`..` 路径穿越、Windows 路径、重复成员、符号链接、加密成员、异构文件、条目数、单分片/总展开体积与压缩比；压缩方法仅允许 Stored / Deflated，成员始终直接从压缩流读取，不按包内路径解压落地，损坏压缩流统一收口为可解释读取失败。
- 矩阵分片要求表头及字段顺序完全一致；对象 JSON / JSONL 分片允许字段顺序不同，但字段集合必须一致；矩阵与对象结构不能混合。任一分片失败或冲突时整体返回带成员名的可解释失败报告。
- 分片按 ZIP 包内顺序合并；相同的编码、形态转换等 warning 按分片组聚合，避免 160 分片产生 160 条重复提示。
- ZIP 内全体分片共同受 200,000 条 JSON 记录、20,000,000 单元格、1,000,000 键值对和数组项总量等既有上限约束；下一分片在完整物化前即受包级剩余记录、单元格、键值对和数组预算限制。温州形态的 160 分片已建立动态回归；其 800,000 条真实总量会按既定 200,000 条上限明确拒绝，不为 ZIP 放宽安全边界。
- 分片发生字段冲突或资源失败时，当前分片已经产生的编码、提取或修复 warning 会保留在失败报告中。
- 阶段 B 当时的 `python3 run_demo.py --check` 已通过：143 项测试全部成功，固定报告一致，真实 `127.0.0.1` Streamlit 健康检查通过。

## 8. 阶段 C 完成结果

- 新增 `.geojson` 上传类型；`.json` 中顶层 `type: "FeatureCollection"` 也按 GeoJSON 路径处理。ZIP 内仍拒绝 GeoJSON，保持阶段 B 的同构 JSON 分片边界不变。
- 每个 Feature 固定映射为一行，要求显式存在 `properties` 和 `geometry`；仅读取扁平 `properties`、字符串或有限数值类型的可选 `Feature.id`，并写入 `__geojson_` 前缀的几何类型、坐标位置数、坐标维度及二维范围摘要。
- 坐标数组始终只用于计算摘要，不展开、不进入报告；白名单之外的 FeatureCollection、Feature 或 geometry 嵌套成员，嵌套 properties、保留技术字段冲突、非有限坐标、空坐标、非 FeatureCollection 和未支持几何类型均明确失败。`geometry: null` 保留记录并将空间摘要留空，空 `GeometryCollection` 合法。
- LineString / MultiLineString 至少包含两个位置；Polygon / MultiPolygon 的线性环至少包含四个位置且首尾闭合。
- 已覆盖点、线、多边形、GeometryCollection、空几何、严格坐标验证、物化前 Feature 数量限制、ZIP 映射前隔离、上传服务和真实网页上传；补充可演示样例 `sample_data/geojson_feature_collection.geojson`。
- 阶段 C 当时的 `python3 run_demo.py --check` 已通过：151 项测试全部成功，固定报告一致，真实 `127.0.0.1` Streamlit 健康检查通过。

## 9. 后续变更时的定位顺序

1. 先读本交接文档的 0.2–0.4 和《开发计划表》2.4.6；若修改解析/GeoJSON，再回看本文第 3-8 节的历史边界。
2. 查看 `git status --short`，保留用户现有改动。
3. 修改 RAG 时，先确认文档类型、来源审核、标准号/版本筛选、引用快照和 RuleEvidence 绑定仍然独立可验证；不把检索结果自动升级为已批准规则或标准符合性结论。
4. 任何新增 GeoJSON 映射路径、坐标展开、ZIP GeoJSON 或通用嵌套展平都属于阶段 C 范围外变更，需先明确记录粒度、字段命名和资源边界。
5. 不改动已验收的单文件 JSON、ZIP 分片、GeoJSON FeatureCollection 与 RAG 引用口径；先跑对应专项测试，再跑完整 `.venv/bin/python run_demo.py --check`。
6. 若未通过验收，在本文档的“当前状态”中记录已完成、失败项和下一步，不将对应能力标记为完成。

## 10. 2026-07-22 第二轮代码审查与补强

本轮先逐项审查阶段 A / B / C，再对照计划表、输入输出协议和本交接记录完成剩余收口。未扩展既定数据形态，也未接入阶段 5 的可选 Agent。

- 阶段 A：修复带空值大整数的静默精度丢失；让包装元数据也接受整棵 Unicode 严格校验；JSONL 只将 JSON 允许的四类空白视为空行，所有失败包含行号；定向修复、包装提取等 warning 在后续失败时仍保留；重复表头及错误值采用有界展示。
- 阶段 B：仅接受 Stored / Deflated 压缩；损坏 Deflate 流统一转换为读取错误；在下一分片完整物化前传入剩余记录、单元格、键值对和数组预算；字段冲突等失败不再丢失当前分片 warning。
- 阶段 C：拒绝白名单外的嵌套成员；补齐 Feature 必需成员和 `id` 类型；补齐线与多边形拓扑约束；允许合法的空 GeometryCollection；ZIP 中的 FeatureCollection 在进行完整空间映射前即被拒绝。
- 报告交付：CLI 默认原子写入严格 `report.json`，显式 `.md` 时输出 Markdown；网页同时提供 JSON 与 Markdown 两个安全下载，不包含原始字段样例或异常原值。
- 当时验证（2026-07-22）：169 项自动化测试全部通过，固定示例报告一致，Streamlit `AppTest` 通过；该轮真实 `127.0.0.1` 端口复验因执行环境未授予本地绑定权限而待补跑，历史 155 项版本的真实健康检查已通过。当前 v0.4 验收结果以本文第 0 节为准。
