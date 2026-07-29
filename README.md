# 政务数据集质量评估 Demo

版本：v0.4

一个本地运行的 Streamlit Demo，用于对 CSV、Excel（`.xls`、`.xlsx`）、表格型 JSON、JSON Lines、GeoJSON FeatureCollection 和同构 JSON 分片 ZIP 数据集生成可复现的质量指标、风险提示、疑似问题位置 CSV 与无法评估项。v0.4 在只读报告诊断 Agent 之外，新增了待审批 `RulePack`：用户可引导式确认主键、必填、更新时间与更新频率、允许值和数值范围，经本地确定性校验和明确批准后生成独立的规则增强报告。

## 部署与启动

要求 Python 3.10 或更高版本。

```bash
git clone <你的私有仓库地址>
cd government-dataset-quality-demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py
```

Windows PowerShell 请将激活命令替换为：

```powershell
.venv\Scripts\Activate.ps1
```

启动后，在终端显示的本地地址（通常为 `http://127.0.0.1:8501`）打开页面。停止服务时按 `Ctrl+C`。

也可以直接使用 Streamlit：

```bash
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

命令行模式默认生成可供系统对接的严格 `report.json`；若需人可读报告，可显式指定 `.md` 输出：

```bash
.venv/bin/python -m src.cli sample_data/good_dataset.csv
.venv/bin/python -m src.cli sample_data/good_dataset.csv --output reports/report.md
```

## Agent 与规则增强模式

报告诊断 Agent 继续只读消费结构化 `QualityReport`。它不会重新计算 13 项指标、改变风险等级、修改阈值或接触原始字段值。页面中的“概括结果”“优先整改事项”“解释无法评估项”和报告问答都必须由用户显式触发。

“规则增强”页签提供本地引导问题和字段候选，但不会自动启用任何规则。规则草案先绑定当前报告哈希、输入哈希和评估基准日期，再由确定性校验器检查字段、类型、范围和资源边界。只有用户输入本地自声明的审批人、勾选确认并点击批准后，系统才会从当前上传字节重新执行原解析和零配置评估链，再追加业务规则指标与风险。零配置报告始终保留且不被覆盖；草案、文件、工作表、数据集名称或基准日期变化都会使原审批失效。

规则草案、允许值和数值边界均在本地处理，不会发送给 DeepSeek。当前 Demo 没有登录或身份认证能力，审批记录中的审批人仅为本地自声明标识，不代表系统已经验证其身份。

每次规则重评最多检查 2,000,000 个规则字段值并生成 200,000 条规则问题位置；超限时整次增强评估会明确拒绝，不会静默截断位置。数值规则只接受绝对值不超过 `1e308` 的有限数字。

默认使用本地模板模式，不需要 API Key，也不会把报告发送到外部服务。若要启用 DeepSeek API，在启动前显式设置：

```bash
export QUALITY_AGENT_PROVIDER=deepseek
export DEEPSEEK_API_KEY="<你的 API Key>"
export DEEPSEEK_MODEL="deepseek-v4-flash"
python3 run_demo.py
```

`DEEPSEEK_MODEL` 可在 DeepSeek API 当前支持的模型中切换；截至 2026-07-28，官方列出 `deepseek-v4-flash` 和 `deepseek-v4-pro`，本项目默认使用前者。本项目通过 `httpx` 直接调用 DeepSeek Chat Completions，不依赖 OpenAI API 或 SDK。模型超时、限流、工具越权、输出结构不合法、引用不存在或数字无法落到报告证据时，系统会丢弃该模型结果并回退到本地模板。API Key 不会写入报告、Agent 审计信息或缓存键。

启用外部模型前仍应完成所在单位的数据分类分级、服务采购和跨境/外发审批。白名单上下文虽然不包含原始单元格值，但字段名、聚合统计和风险说明仍可能属于内部元数据。

## 使用方式

1. 上传 CSV、`.xls`、`.xlsx`、`.json`、`.jsonl`、`.ndjson`、`.geojson` 或同构 JSON 分片 `.zip` 文件；单文件上限为 50 MiB。
2. 可选填写数据集名称；Excel 可选填写工作表名称。
3. 选择评估基准日期，点击“运行质量评估”。
4. 查看风险提示、质量指标、数据画像与无法评估项，并按需下载结构化 JSON、Markdown 评估报告或疑似问题位置 CSV。
5. 在“Agent 解读”页签中按需生成报告概括、优先整改建议、无法评估项说明，或询问仅基于当前报告的问题。
6. 如需业务规则增强，在“规则增强”页签回答少量字段问题，生成并检查草案；明确批准后查看零配置与增强结果的本次差异，并下载已审批 RulePack、增强结果和规则问题位置 CSV。

疑似问题位置只在独立 CSV 中提供。记录序号从 1 开始，表示解析后的记录顺序，不包含 CSV/Excel 表头；它不是物理文件行号。当前可定位字段缺失、空白记录、类型不一致、格式异常、重复记录、时间信息缺失或不可解析、来源/版本信息缺失和 IQR 统计异常。CSV 写入每项指标定位到的全部位置，不做静默截断，只包含指标名称、字段名称、问题类型、记录序号及重复记录的关联序号，不包含原始单元格值；重复记录行的“备注”会直接说明当前记录与哪条首次出现的记录完全相同或规范化后相同。位置明细不会进入页面表格、JSON、Markdown 或 Agent 上下文。

JSON 支持以下表格型形态：

- 单条扁平对象或扁平对象列表；
- 首行为表头的二维数组；
- 位于 `data`、`rows`、`records`、`items`、`list` 或 `result` 下的唯一表格；
- `.jsonl` / `.ndjson` 中每个非空行一条扁平对象；
- `.zip` 中由上述 `.json` / `.jsonl` / `.ndjson` 组成的同构分片。

对象记录示例：

```json
[
  {"名称": "事项 A", "年份": 2025, "数量": 120},
  {"名称": "事项 B", "年份": 2025, "数量": 80}
]
```

二维数组示例：

```json
[
  ["名称", "年份", "数量"],
  ["事项 A", 2025, 120],
  ["事项 B", 2025, 80]
]
```

只有可唯一判定的接口包装会自动提取，并在运行信息中留下 warning。JSON 读取支持 UTF-8、带 BOM 的 UTF-16 / UTF-32 和 GB18030；非 UTF-8 读取会显式提示。

ZIP 分片直接从压缩流读取，不按成员路径解压落地。矩阵分片必须具有完全一致且同序的表头，对象分片必须具有相同字段集合；两种结构不能混合。ZIP 会预检路径、重复成员、符号链接、加密条目、条目数、单分片与总展开体积及压缩比，且只允许 Stored / Deflated 压缩方法。任一分片失败或结构冲突时整体拒绝；包级剩余资源预算会在下一分片物化前检查。ZIP 内全部分片合计仍受 200,000 条 JSON 记录上限约束。

GeoJSON 仅支持顶层 `FeatureCollection`：一个 `Feature` 固定对应一条记录；每个 Feature 必须显式包含 `properties` 和 `geometry`，可选 `id` 只能是字符串或有限数值。白名单仅包括扁平 `properties`、`Feature.id` 和 `geometry` 的类型、坐标位置数、坐标维度及二维范围摘要。原始 `coordinates` 永不展开或导出；嵌套 `properties`、白名单外嵌套路径、未支持几何类型、非法线/多边形基数、未闭合线性环和非有限坐标会明确失败。GeoJSON 不可作为 ZIP 分片使用。

当前仍不支持普通嵌套对象/数组的自动展平、非 FeatureCollection 的 GeoJSON、注释、尾逗号或其他 JSON5 容错。仅针对已确认的 `{[...]}` 整体外壳做定向修复，且必定记录 warning。

## 功能边界

- 内置确定性数据画像、质量指标与风险规则，无需模型 API；
- `QualityReport` 是指标和风险的唯一事实源；Agent 输出使用独立 `AgentAnalysis`，不会回写报告；
- `RulePack`、审批记录和规则增强结果使用独立协议；未批准、非法、审批后被修改或与当前输入不匹配的规则不会进入引擎；
- 规则增强只追加确定性业务规则结果，不修改原 13 项指标、既有风险或默认阈值，也不实现 v0.5 的跨历史版本“改善/恶化”判断；
- 人工确认绑定当前草案哈希；重新生成或修改草案后必须重新勾选确认，后端执行分支也会再次校验；
- 每条 Agent 事实和整改建议都引用当前报告中的 `metric_key`、`risk_id`、无法评估项或报告摘要；
- Agent 只可调用只读白名单工具，默认不发送文件名、数据集名称、运行错误全文、原始字段样例、异常原值或记录定位列表；
- 上传文件仅用于本次临时评估，完成后自动删除；
- 页面同时提供严格 JSON、UTF-8 Markdown 报告和独立的疑似问题位置 CSV；JSON 包含稳定报告哈希、引擎版本、基准日期、解析路径和风险判定证据，JSON 与 Markdown 不携带位置明细，三类下载均不导出原始字段样例或异常原值；
- 不会修改或覆盖原始数据；
- 仓库包含自动化测试、可演示的合成样例，以及用于一致性检查的固定基准报告；
  用户运行时生成的其他报告默认不纳入版本控制。
