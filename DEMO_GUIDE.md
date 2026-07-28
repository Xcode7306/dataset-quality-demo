# 政务数据集质量评估 Demo：演示与验收说明

> 当前目标：v0.2 本地演示版 + JSON 扩展阶段 C
> 最近验收日期：2026 年 7 月 22 日
> 固定示例报告基准日期：2026 年 7 月 17 日
> 运行环境：Python 3.10 或更高版本；当前完整验收版本为 3.12.4
> 核心边界：确定性指标与风险规则可独立运行，Agent 不属于本版硬交付。

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
4. 页面同时提供结构化 JSON 和 Markdown：JSON 及 CLI 默认输出的 `report.json` 是后续系统或可选 AI 解释层的稳定输入，Markdown 适合直接阅读；
5. 切换文件、数据集名称或 Excel 工作表后，旧报告会立即清空，必须重新运行，避免结果错配。
6. 改变评估基准日期也会立即清空旧报告；同一文件与同一基准日期可复现相同结果。

## 4. 页面验收清单

- 上传控件仅接受 CSV、`.xls`、`.xlsx`、`.json`、`.jsonl`、`.ndjson`、`.geojson` 和同构 JSON 分片 `.zip`；
- CSV、Excel、表格型 JSON、JSON Lines、GeoJSON FeatureCollection 和同构 JSON 分片 ZIP 均能显示 5 个摘要卡、风险图、指标明细和字段画像；
- CSV 数据行字段数超过表头时明确失败，不会被静默推入索引；CSV、Excel、JSON 对 `NA`、`NULL`、`N/A` 等合法文本的口径一致；
- Excel 可指定工作表，错误名称会列出可用工作表；
- 可唯一识别的常见接口包装可提取；多候选包装或普通嵌套 JSON 返回明确原因，不自动改变记录粒度；
- ZIP 不按成员路径解压落地，只解码 Stored / Deflated；矩阵分片表头必须同序一致，对象分片字段集合必须一致，结构混用或任一分片失败时整体返回明确原因；
- GeoJSON 只接受 FeatureCollection；每个 Feature 固定映射一行，扁平 properties、Feature ID 与空间摘要可用，原始坐标数组不展开，嵌套 properties、未定义的嵌套路径以及非法线/多边形明确拒绝；
- 截断 Excel、异常 JSON、冲突表头等输入返回可解释失败报告，不显示内部回溯；
- CSV/Excel 原始重复表头、CSV NUL 空字符、JSON 重复键、非标准 `NaN`/`Infinity` 以及超出有限范围的数值会明确失败，不会被底层库静默改写；
- 成功与失败报告都可下载严格 JSON 和 UTF-8 Markdown；
- CLI 默认将严格 JSON 写入 `reports/report.json`，显式指定 `.md` 时保留 Markdown 输出；
- 下载报告中原始字段样例、格式异常原值和统计异常原值均不导出；
- 输入变化后不会继续展示上一份报告；
- 上传显示名和下载名会清理路径、控制字符与超长内容；
- 单文件限制为 50 MiB；超行列、超 2,000 万单元格、超长单元格、超 20 万 JSON 总记录、超限数组、单对象超 1 万或全文件超 100 万 JSON 键值对、不可映射的嵌套 JSON，以及异常 `.xlsx` / JSON ZIP 条目数、展开体积或压缩比会被拒绝；
- 原始上传文件只在临时目录中使用，评估后自动删除；
- 未配置任何模型 API 时，完整流程仍可运行。

## 5. 当前不包含的能力

- 不生成单一综合分；
- 不实现 Agent 或聊天界面；
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
