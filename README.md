# 政务数据集质量评估 Demo

版本：v0.2

一个本地运行的 Streamlit Demo，用于对 CSV、Excel（`.xls`、`.xlsx`）、表格型 JSON、JSON Lines、GeoJSON FeatureCollection 和同构 JSON 分片 ZIP 数据集生成可复现的质量指标、风险提示与无法评估项。

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

## 使用方式

1. 上传 CSV、`.xls`、`.xlsx`、`.json`、`.jsonl`、`.ndjson`、`.geojson` 或同构 JSON 分片 `.zip` 文件；单文件上限为 50 MiB。
2. 可选填写数据集名称；Excel 可选填写工作表名称。
3. 选择评估基准日期，点击“运行质量评估”。
4. 查看数据画像、质量指标、风险提示与无法评估项，并按需下载结构化 JSON 或 Markdown 评估报告。

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
- 上传文件仅用于本次临时评估，完成后自动删除；
- 页面同时提供严格 JSON 与 UTF-8 Markdown 报告，两者均不导出原始字段样例；
- 不会修改或覆盖原始数据；
- 仓库包含自动化测试、可演示的合成样例，以及用于一致性检查的固定基准报告；
  用户运行时生成的其他报告默认不纳入版本控制。
