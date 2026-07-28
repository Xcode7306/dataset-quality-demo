"""数据画像模块：统计表格规模、缺失与基础字段类型。"""

import re
from datetime import date, datetime
from numbers import Number
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .field_semantics import identify_semantic_fields


DATE_PATTERN = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T].*)?$")
BOOLEAN_TEXT = {"true", "false", "yes", "no", "是", "否"}


def is_missing_value(value: Any) -> bool:
    """将 NaN、None 和仅含空白的字符串都视为缺失。"""

    if value is None or pd.isna(value):
        return True
    return isinstance(value, str) and not value.strip()


def _non_missing_values(series: pd.Series) -> pd.Series:
    return series[~series.map(is_missing_value)]


def infer_value_type(value: Any) -> str | None:
    """推断单个非空值的基础类型，供类型一致性指标复用。"""

    if is_missing_value(value):
        return None
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return "datetime"
    if isinstance(value, Number):
        return "numeric"

    text = str(value).strip()
    if text.lower() in BOOLEAN_TEXT:
        return "boolean"
    if pd.to_numeric(pd.Series([text]), errors="coerce").notna().all():
        return "numeric"
    if DATE_PATTERN.match(text):
        parsed = pd.to_datetime(pd.Series([text]), errors="coerce", format="mixed")
        if parsed.notna().all():
            return "datetime"
    return "text"


def infer_basic_type(series: pd.Series) -> str:
    """按可观察到的值推断基础类型，不判断字段业务含义。"""

    values = _non_missing_values(series)
    if values.empty:
        return "unknown"
    if is_bool_dtype(series):
        return "boolean"
    if is_numeric_dtype(series):
        return "numeric"
    if is_datetime64_any_dtype(series):
        return "datetime"

    text_values = values.astype(str).str.strip()
    lowered = text_values.str.lower()
    if lowered.isin(BOOLEAN_TEXT).all():
        return "boolean"

    numeric_values = pd.to_numeric(text_values, errors="coerce")
    if numeric_values.notna().all():
        return "numeric"

    if text_values.map(lambda value: bool(DATE_PATTERN.match(value))).all():
        parsed_dates = pd.to_datetime(text_values, errors="coerce", format="mixed")
        if parsed_dates.notna().all():
            return "datetime"
    return "text"


def profile_dataframe(dataframe: pd.DataFrame, sample_size: int = 3) -> dict[str, Any]:
    """生成零配置基础数据画像。

    此处只做客观统计，不计算 13 项质量指标，也不做风险判断。
    """

    # 保留 sample_size 参数以兼容既有调用方，但报告默认不再
    # 携带任何原始字段值，防止姓名、证件号、邮箱等泄露。
    if sample_size < 1:
        raise ValueError("sample_size 必须至少为 1。")

    columns: list[dict[str, Any]] = []
    recognized_fields = identify_semantic_fields(dataframe.columns)
    warnings: list[str] = []
    if dataframe.empty:
        warnings.append("数据集不包含记录，无法对字段值进行进一步画像。")
    if len(dataframe.columns) == 0:
        warnings.append("数据集不包含字段。")

    for column_name in dataframe.columns:
        series = dataframe[column_name]
        missing_mask = series.map(is_missing_value)
        missing_count = int(missing_mask.sum())
        non_missing = series[~missing_mask]
        inferred_type = infer_basic_type(series)
        normalized_name = str(column_name)
        inferred_category = {
            "datetime": "date",
            "numeric": "numeric",
        }.get(inferred_type)
        if (
            inferred_category
            and normalized_name not in recognized_fields[inferred_category]
        ):
            recognized_fields[inferred_category].append(normalized_name)
        columns.append(
            {
                "name": normalized_name,
                "missing_count": missing_count,
                "non_missing_count": int(len(non_missing)),
                "missing_rate": (
                    round(missing_count / len(series), 6) if len(series) else None
                ),
                "inferred_type": inferred_type,
                # 保留原有字段，避免破坏展示层和结构化报告对象；
                # 空列表明确表示不导出原始样例。
                "non_null_samples": [],
            }
        )

    return {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "columns": columns,
        "recognized_fields": recognized_fields,
        "warnings": warnings,
    }
