# 政务数据集质量评估 Demo

版本：v0.1

一个本地运行的 Streamlit Demo，用于对 CSV、Excel（`.xlsx`）和扁平记录型 JSON 数据集生成可复现的质量指标、风险提示与无法评估项。

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

## 使用方式

1. 上传 CSV、`.xlsx` 或扁平记录型 JSON 文件；单文件上限为 50 MiB。
2. 可选填写数据集名称；Excel 可选填写工作表名称。
3. 选择评估基准日期，点击“运行质量评估”。
4. 查看数据画像、质量指标、风险提示与无法评估项，并按需下载 JSON 报告。

JSON 仅支持顶层为对象记录列表或单条对象记录的扁平结构，例如：

```json
[
  {"名称": "事项 A", "年份": 2025, "数量": 120},
  {"名称": "事项 B", "年份": 2025, "数量": 80}
]
```

不支持嵌套 JSON，亦不支持首行表头加数组数据行的二维数组格式。

## 功能边界

- 内置确定性数据画像、质量指标与风险规则，无需模型 API；
- 上传文件仅用于本次临时评估，完成后自动删除；
- 不会修改或覆盖原始数据；
- 本发布仓库不包含开发测试、样例数据或生成报告。
