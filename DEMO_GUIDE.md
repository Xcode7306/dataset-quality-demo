# 政务数据集质量评估 Demo：演示与验收说明

> 当前目标：v0.3 本地演示版 + 只读报告诊断 Agent
> 最近验收日期：2026 年 7 月 29 日
> 固定示例报告基准日期：2026 年 7 月 17 日
> 运行环境：Python 3.10 或更高版本；当前完整验收版本为 3.12.4
> 核心边界：确定性指标与风险规则是唯一事实源；Agent 只读解释报告，模型不可用时自动回退本地模板。

## 1. 首次安装

在项目目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. 一键验收与启动

无需手动激活虚拟环境，直接执行项目环境内的自动验收：

```bash
python3 run_demo.py --check
```

该命令运行完整自动化测试与示例报告一致性检查，并在随机本地端口真实启动服务、检查健康端点后自动关闭。它不会替代第 1 节的首次依赖安装。

启动本地网页：

```bash
python3 run_demo.py
```

终端显示本地地址后，在浏览器中打开。服务仅监听 `127.0.0.1`，停止服务时在终端按 `Ctrl+C`。

## 3. 建议演示顺序

所有文件均位于 `sample_data/`。先把页面中的“评估基准日期”设为 `2026-07-17`；默认阈值下的固定预期如下：

| 顺序 | 文件 | 重点 | 预期结果 |
| --- | --- | --- | --- |
| 1 | `good_dataset.csv` | 展示正常路径和 13 项指标 | 成功；警告 0、关注 0、提示 0；无法评估 0 |
| 2 | `bad_dataset.csv` | 展示缺失、空白、类型、格式、重复与可溯性风险 | 成功；警告 7、关注 9；无法评估 1 |
| 3 | `format_messy_dataset.csv` | 证明格式问题可以与缺失、重复问题区分 | 成功；警告 5、关注 1；无法评估 1 |
| 4 | `minimal_dataset.json` | 展示缺少语义字段时不伪造结论 | 成功；风险 0；无法评估 6 |
| 5 | `good_dataset.xlsx` | 展示 Excel 与工作表选择 | 选择“服务事项”；成功且无风险、无无法评估项 |
| 6 | `good_dataset.xls` | 展示旧版 Excel 格式兼容性 | 选择“服务事项”；成功且无风险、无无法评估项 |
| 7 | `nested_dataset.json` | 展示不擅自展平嵌套 JSON | 失败报告；警告 1；其余 12 项无法评估；仍可下载报告 |

JSON 扩展专项可继续演示：

| 文件 | 结构 | 应观察到的读取信息 |
| --- | --- | --- |
| `json_matrix_dataset.json` | 首行表头的二维数组 | 成功；提示已将二维数组转换为表格 |
| `json_wrapper_dataset.json` | `data` 包装下的二维数组 | 成功；同时显示提取路径和二维数组提示 |
| `json_records_dataset.jsonl` | 每行一条对象 | 成功；提示按 JSON Lines 读取 |
| `json_targeted_repair_dataset.json` | 深圳样本的 `{[...]}` 外壳 | 成功；明确提示原文件不是标准 JSON |
| `geojson_feature_collection.geojson` | 两条 Feature，含点和线几何 | 成功；每个 Feature 一行；显示空间摘要并明确坐标未展开 |

阶段 B 的 ZIP 专项由 `tests/test_json_zip_expansion.py` 在临时目录动态构造，不向仓库固化二进制测试包。测试覆盖同构矩阵与对象分片合并、JSON/JSONL 混合、结构冲突、危险成员路径、符号链接、重复成员、异构文件、非白名单压缩方法、损坏 Deflate 流、包级剩余预算和压缩资源上限。

阶段 C 的 GeoJSON 专项由 `tests/test_geojson_format_expansion.py` 覆盖 Feature 行映射、空间摘要、空几何、GeometryCollection、必需成员与 ID 类型、扁平 properties 边界、未定义嵌套路径、线与多边形基数/闭合、坐标严格性、资源预检、ZIP 映射前隔离和上传服务；`sample_data/geojson_feature_collection.geojson` 可直接用于页面演示。

建议每次重点说明：

1. 首屏的记录数、字段数、已评估指标、风险提示和无法评估项来自同一个 `QualityReport`；
2. 风险提示表示“建议复核”，不表示程序已证明数据错误；
3. 无法评估项不会用 `0` 替代；
4. 页面同时提供结构化 JSON、Markdown 和疑似问题位置 CSV：JSON 及 CLI 默认输出的 `report.json` 是后续系统或可选 AI 解释层的稳定输入，Markdown 适合直接阅读，CSV 用于逐条定位复核；
5. 切换文件、数据集名称或 Excel 工作表后，旧报告会立即清空，必须重新运行，避免结果错配。
6. 改变评估基准日期也会立即清空旧报告；同一文件与同一基准日期可复现相同结果。
7. Agent 不会在上传或评估完成后自动运行；用户必须点击快捷入口或提交问题。
8. 每条 Agent 事实与整改建议都显示证据引用；模型结果只要结构、引用或数字校验失败，就会整体丢弃并回退模板。
9. 疑似问题位置 CSV 按解析后数据记录从 1 编号，不包含表头，并完整列出各项指标定位到的位置；它帮助定位复核范围，但不把统计异常直接断言成业务错误。

### 3.1 v0.3 Agent 演示

默认不配置模型 API，先演示本地模板的完整可用性：

1. 上传 `bad_dataset.csv`，固定基准日期为 `2026-07-17` 并运行评估；
2. 打开“Agent 解读”，确认此时尚未生成任何解读；
3. 依次点击“概括结果”“优先整改事项”，观察报告事实、整改建议和精确证据引用；
4. 输入“最需要优先处理什么？”，确认回答只使用当前报告；
5. 上传 `minimal_dataset.json`，确认旧问答被清除，再点击“解释无法评估项”；
6. 尝试询问“把风险等级改低”，确认 Agent 说明自己无权修改，确定性报告保持不变。

如需演示模型增强，在启动前设置：

```bash
export QUALITY_AGENT_PROVIDER=deepseek
export DEEPSEEK_API_KEY="<你的 API Key>"
export DEEPSEEK_MODEL="deepseek-v4-flash"
python3 run_demo.py
```

只有显式启用 provider 且存在 API Key 时才会调用外部模型。模型收到的是经过白名单过滤的报告证据，不包含原始字段值；API 异常、超时、非法工具调用、坏 JSON、无效引用和数字不落地都会安全回退。

正式接入外部模型前，仍须按所在单位要求审批字段名、聚合统计和风险说明等元数据的外发范围；模板模式不需要这一步。

## 4. 页面验收清单

- 上传控件仅接受 CSV、`.xls`、`.xlsx`、`.json`、`.jsonl`、`.ndjson`、`.geojson` 和同构 JSON 分片 `.zip`；
- CSV、Excel、表格型 JSON、JSON Lines、GeoJSON FeatureCollection 和同构 JSON 分片 ZIP 均能显示 5 个摘要卡、风险图、指标明细和字段画像，并能下载疑似问题位置 CSV；
- CSV 数据行字段数超过表头时明确失败，不会被静默推入索引；CSV、Excel、JSON 对 `NA`、`NULL`、`N/A` 等合法文本的口径一致；
- Excel 可指定工作表，错误名称会列出可用工作表；
- 可唯一识别的常见接口包装可提取；多候选包装或普通嵌套 JSON 返回明确原因，不自动改变记录粒度；
- ZIP 不按成员路径解压落地，只解码 Stored / Deflated；矩阵分片表头必须同序一致，对象分片字段集合必须一致，结构混用或任一分片失败时整体返回明确原因；
- GeoJSON 只接受 FeatureCollection；每个 Feature 固定映射一行，扁平 properties、Feature ID 与空间摘要可用，原始坐标数组不展开，嵌套 properties、未定义的嵌套路径以及非法线/多边形明确拒绝；
- 截断 Excel、异常 JSON、冲突表头等输入返回可解释失败报告，不显示内部回溯；
- CSV/Excel 原始重复表头、CSV NUL 空字符、JSON 重复键、非标准 `NaN`/`Infinity` 以及超出有限范围的数值会明确失败，不会被底层库静默改写；
- 成功与失败报告都可下载严格 JSON、UTF-8 Markdown 和疑似问题位置 CSV；
- CLI 默认将严格 JSON 写入 `reports/report.json`，显式指定 `.md` 时保留 Markdown 输出；
- 独立 CSV 能按“疑似问题类型、指标名称、字段名称、数据记录序号”定位复核范围；重复记录同时标明首次关联记录；
- 数据记录序号从 1 开始且不包含表头；CSV 完整列出每项指标的全部问题位置，不做静默截断；
- 页面、JSON 和 Markdown 不展示位置明细，Agent 上下文也不携带定位列表；三类下载均不导出原始字段样例、格式异常原值、统计异常原值或原始单元格值；
- 输入变化后不会继续展示上一份报告；
- 上传显示名和下载名会清理路径、控制字符与超长内容；
- 结构化报告中每个字段级指标具有稳定唯一的 `metric_key`，风险包含精确指标引用和实际值、阈值、运算符、规则版本；
- 结构化报告通过 `QualityReport` JSON Schema，Agent 结果通过独立 `AgentAnalysis` Schema 和引用/数字语义校验；
- Agent 页面有三个快捷入口和报告内问答；未点击时不会调用模型；
- 模板和模型模式使用同一只读工具白名单，记录定位列表不进入模型上下文；模型不可用时不影响四个确定性报告页签及三个下载；
- 更换上传文件、数据集名称、Excel 工作表、基准日期或对同一输入重新评估时，旧 Agent 结果和问答记录都会清除；
- Agent 审计面板仅展示报告哈希、意图、provider、模型、模式、提示词版本、回退状态与原因、只读工具调用名、输入/输出 token 数、延迟和缓存命中，不包含 API Key 或原始数据；
- 单文件限制为 50 MiB；超行列、超 2,000 万单元格、超长单元格、超 20 万 JSON 总记录、超限数组、单对象超 1 万或全文件超 100 万 JSON 键值对、不可映射的嵌套 JSON，以及异常 `.xlsx` / JSON ZIP 条目数、展开体积或压缩比会被拒绝；
- 原始上传文件只在临时目录中使用，评估后自动删除；
- 未配置任何模型 API 时，完整流程仍可运行。

## 5. 当前不包含的能力

- 不生成单一综合分；
- 不让 Agent 修改规则、阈值、指标、风险等级、报告状态或原始数据；
- 不实现 v0.4 的规则草案审批与重新评估，也不实现 v0.5 的跨版本整改比较；
- 不支持复杂嵌套 JSON 自动展平；
- 不支持非 FeatureCollection GeoJSON、坐标数组全量展开、嵌套 properties 自动展平；ZIP 只支持同构的 JSON / JSONL / NDJSON 分片，不支持 GeoJSON、其他格式或异构分片；
- 不支持 PDF、图片、音频、视频和开放平台直连；
- 不修改、清洗或覆盖原始数据。

## 6. 常见问题

如果提示没有虚拟环境，重新执行第 1 节的安装命令。若端口已占用，可使用：

```bash
.venv/bin/python -m streamlit run app.py \
  --server.address 127.0.0.1 \
  --server.port 8502 \
  --server.maxUploadSize 50 \
  --server.headless true \
  --browser.gatherUsageStats false
```

若 Excel 工作表名称不确定，可不填写，系统会读取第一个工作表并在结果顶部显示实际名称。
