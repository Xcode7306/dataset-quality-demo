"""由 good_dataset.csv 生成测试用 Excel 文件。"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
source = ROOT / "good_dataset.csv"
target = ROOT / "good_dataset.xlsx"

dataframe = pd.read_csv(source)
dataframe.to_excel(target, index=False, sheet_name="服务事项")
print(target)
