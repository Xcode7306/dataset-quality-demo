"""13 项零配置自动质量指标的目录与计算入口。"""

from collections import Counter, defaultdict
from datetime import date
import math
import re
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlparse

import pandas as pd

from .field_semantics import (
    DATE_FIELD_PATTERN,
    EMAIL_FIELD_PATTERN,
    NUMERIC_FIELD_PATTERN,
    SOURCE_FIELD_PATTERN,
    SOURCE_IDENTIFIER_FIELD_PATTERN,
    STRUCTURED_TEXT_FIELD_PATTERN,
    UPDATE_FIELD_PATTERN,
    URL_FIELD_PATTERN,
    VERSION_FIELD_PATTERN,
    field_matches,
)
from .models import MetricResult
from .profiler import infer_value_type, is_missing_value


METRIC_CATALOG = [
    {"id": "file_parse_rate", "name": "文件可解析率", "category": "可读取性"},
    {"id": "dataset_scale", "name": "数据规模", "category": "规模"},
    {"id": "field_missing_rate", "name": "字段缺失率", "category": "完整性"},
    {"id": "blank_record_rate", "name": "空白记录率", "category": "完整性"},
    {"id": "field_type_consistency", "name": "字段类型一致率", "category": "类型一致性"},
    {"id": "recognizable_format_anomaly_rate", "name": "可识别格式异常率", "category": "格式规范性"},
    {"id": "exact_duplicate_rate", "name": "完全重复率", "category": "唯一性"},
    {"id": "normalized_duplicate_rate", "name": "规范化重复率", "category": "唯一性"},
    {"id": "time_info_availability", "name": "时间信息可用率", "category": "及时性"},
    {"id": "update_lag_days", "name": "更新滞后天数", "category": "及时性"},
    {"id": "source_info_coverage", "name": "来源信息覆盖率", "category": "可溯性"},
    {"id": "version_info_coverage", "name": "版本信息覆盖率", "category": "可溯性"},
    {"id": "statistical_outlier_rate", "name": "统计异常值比例", "category": "数据异常"},
]


IDENTIFIER_FIELD_PATTERN = re.compile(
    r"^(?:id|uuid|index|record_?id|row_?id|.*_id|序号|索引|.*(?:编号|编码|标识|标识码))$",
    re.IGNORECASE,
)
EMAIL_VALUE_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
VERSION_VALUE_PATTERN = re.compile(
    r"^[vV]?\d+(?:[._-]\d+)+(?:[A-Za-z0-9._-]*)$"
)
CODE_VALUE_PATTERN = re.compile(
    r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)+$"
)
NORMALIZE_TEXT_PATTERN = re.compile(r"[\s\W_]+", re.UNICODE)


def _issue_locations(
    record_numbers: Iterable[int],
    *,
    issue_type: str,
    fields: Iterable[str],
    related_record_numbers: Mapping[int, Iterable[int]] | None = None,
) -> list[dict[str, Any]]:
    """生成完整、无原始值的问题位置列表。

    ``record_number`` 是解析后数据记录的 1 基序号，不是物理文件行号；
    CSV/Excel 的表头不计入，JSON 则对应数组中的记录顺序。
    """

    normalized_fields = list(
        dict.fromkeys(str(field) for field in fields if str(field))
    )
    locations: list[dict[str, Any]] = []
    related = related_record_numbers or {}
    for raw_record_number in record_numbers:
        record_number = int(raw_record_number)
        location: dict[str, Any] = {
            "record_number": record_number,
            "fields": normalized_fields,
            "issue_type": issue_type,
        }
        linked_records = [
            int(value)
            for value in related.get(record_number, ())
            if int(value) > 0
        ]
        if linked_records:
            location["related_record_numbers"] = linked_records
        locations.append(location)
    return locations


def _infer_content_columns(dataframe: pd.DataFrame) -> list[str]:
    """排除明显的技术标识列，将其余字段作为空白记录的内容范围。

    若数据只有标识列，则回退为检查全部字段，避免把每条记录都
    误判为空白。
    """

    columns = [str(column) for column in dataframe.columns]
    content_columns = [
        column for column in columns if not IDENTIFIER_FIELD_PATTERN.match(column.strip())
    ]
    return content_columns or columns


def _matching_columns(dataframe: pd.DataFrame, pattern: re.Pattern[str]) -> list[str]:
    return [
        str(column)
        for column in dataframe.columns
        if field_matches(pattern, column)
    ]


def _matching_temporal_columns(
    dataframe: pd.DataFrame, pattern: re.Pattern[str]
) -> list[str]:
    """识别时间字段，同时排除名称中明确表示 URL 或邮箱的字段。"""

    return [
        field
        for field in _matching_columns(dataframe, pattern)
        if not field_matches(URL_FIELD_PATTERN, field)
        and not field_matches(EMAIL_FIELD_PATTERN, field)
    ]


def _format_value(value: Any) -> str:
    return str(value).strip()


def _parse_datetime(value: Any) -> pd.Timestamp | None:
    if is_missing_value(value):
        return None
    parsed = pd.to_datetime(_format_value(value), errors="coerce", format="mixed")
    if pd.isna(parsed):
        return None
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is not None:
        # 政务表格中的日期通常表达当地业务日期。移除时区但保留当地钟面时间，
        # 既避免带/不带时区的值无法比较，也避免转换 UTC 后日期前移或后移。
        timestamp = timestamp.tz_localize(None)
    return timestamp


def _is_valid_url(value: Any) -> bool:
    if is_missing_value(value):
        return False
    try:
        parsed = urlparse(_format_value(value))
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_valid_email(value: Any) -> bool:
    return not is_missing_value(value) and bool(EMAIL_VALUE_PATTERN.match(_format_value(value)))


def _is_valid_finite_number(value: Any) -> bool:
    if is_missing_value(value):
        return False
    parsed = pd.to_numeric(_format_value(value), errors="coerce")
    return pd.notna(parsed) and math.isfinite(float(parsed))


def _looks_like_structured_text(value: Any) -> bool:
    """识别字段名不明确时仍应保留标点语义的结构化字符串。"""

    if not isinstance(value, str):
        return False
    text = value.strip()
    return bool(
        _is_valid_url(text)
        or _is_valid_email(text)
        or VERSION_VALUE_PATTERN.fullmatch(text)
        or CODE_VALUE_PATTERN.fullmatch(text)
    )


def _exact_value_key(value: Any) -> tuple[str, str, Any]:
    """生成保留原始类型与字符差异的可哈希比较键。"""

    # 完全重复比较不将空字符串或空白字符串当作同一个值；
    # 它们的空白差异仅在规范化重复比较中忽略。
    if value is None or value is pd.NA:
        return ("missing", "missing", None)
    if not isinstance(value, str) and bool(pd.isna(value)):
        return ("missing", "missing", None)
    try:
        hash(value)
        comparable_value = value
    except TypeError:
        comparable_value = repr(value)
    value_type = type(value)
    return (value_type.__module__, value_type.__qualname__, comparable_value)


def _normalize_value(value: Any, field: str) -> tuple[str, str, Any]:
    """只归一化自然文本，保留数值、日期、编号等值的语义。"""

    if is_missing_value(value):
        return ("missing", "missing", None)
    if field_matches(STRUCTURED_TEXT_FIELD_PATTERN, field) or _looks_like_structured_text(value):
        return _exact_value_key(value)
    inferred_type = infer_value_type(value)
    # 字符串缺失值同样可去除空白，使 "" 与 " " 在规范化重复中视为相同。
    if inferred_type not in {None, "text"}:
        return _exact_value_key(value)
    normalized_text = NORMALIZE_TEXT_PATTERN.sub("", _format_value(value).casefold())
    return ("normalized", "text", normalized_text)


def _duplicate_groups(
    dataframe: pd.DataFrame, normalize: bool
) -> list[list[int]]:
    columns = _infer_content_columns(dataframe)
    grouped_indices: dict[tuple[tuple[str, str, Any], ...], list[int]] = defaultdict(list)
    for position, (_, row) in enumerate(dataframe[columns].iterrows()):
        values = row.tolist()
        key = tuple(
            _normalize_value(value, field) if normalize else _exact_value_key(value)
            for field, value in zip(columns, values)
        )
        grouped_indices[key].append(position + 1)
    return [indices for indices in grouped_indices.values() if len(indices) > 1]


def _not_assessable_for_fields(
    metric_id: str, name: str, category: str, reason: str
) -> list[MetricResult]:
    return [_not_assessable(metric_id, name, category, "dataset", reason)]


def _not_assessable(
    metric_id: str,
    name: str,
    category: str,
    scope: str,
    reason: str,
    field: str | None = None,
) -> MetricResult:
    return MetricResult(
        id=metric_id,
        name=name,
        category=category,
        status="not_assessable",
        value=None,
        unit=None,
        scope=scope,  # type: ignore[arg-type]
        field=field,
        reason=reason,
    )


def calculate_file_parse_rate(successful: bool = True) -> MetricResult:
    """单文件模式下记录本次解析是否成功。"""

    return MetricResult(
        id="file_parse_rate",
        name="文件可解析率",
        category="可读取性",
        status="evaluated",
        value=1.0 if successful else 0.0,
        unit="ratio",
        scope="dataset",
        evidence={
            "attempted_file_count": 1,
            "successful_file_count": 1 if successful else 0,
            "failed_file_count": 0 if successful else 1,
        },
    )


def calculate_dataset_scale(dataframe: pd.DataFrame) -> MetricResult:
    """以记录数作为第一版数据规模的主展示值。"""

    return MetricResult(
        id="dataset_scale",
        name="数据规模",
        category="规模",
        status="evaluated",
        value=int(len(dataframe)),
        unit="records",
        scope="dataset",
        evidence={"file_count": 1, "row_count": int(len(dataframe)), "column_count": int(len(dataframe.columns))},
    )


def calculate_field_missing_rates(dataframe: pd.DataFrame) -> list[MetricResult]:
    """逐字段统计缺失值比例；空字符串也视为缺失。"""

    if len(dataframe.columns) == 0:
        return _not_assessable_for_fields(
            "field_missing_rate",
            "字段缺失率",
            "完整性",
            "数据集不包含字段，无法计算字段缺失率。",
        )

    results: list[MetricResult] = []
    for column_name in dataframe.columns:
        series = dataframe[column_name]
        if len(series) == 0:
            results.append(
                _not_assessable(
                    "field_missing_rate",
                    "字段缺失率",
                    "完整性",
                    "field",
                    "数据集不包含记录，无法计算字段缺失率。",
                    str(column_name),
                )
            )
            continue
        missing_mask = series.map(is_missing_value)
        missing_count = int(missing_mask.sum())
        missing_record_numbers = (
            position
            for position, is_missing in enumerate(
                missing_mask.tolist(),
                start=1,
            )
            if is_missing
        )
        results.append(
            MetricResult(
                id="field_missing_rate",
                name="字段缺失率",
                category="完整性",
                status="evaluated",
                value=round(missing_count / len(series), 6),
                unit="ratio",
                scope="field",
                field=str(column_name),
                evidence={
                    "checked_count": int(len(series)),
                    "issue_count": missing_count,
                },
                issue_locations=_issue_locations(
                    missing_record_numbers,
                    issue_type="missing_value",
                    fields=[str(column_name)],
                ),
            )
        )
    return results


def calculate_blank_record_rate(dataframe: pd.DataFrame) -> MetricResult:
    """统计可识别内容字段均为空的记录比例。"""

    if len(dataframe) == 0:
        return _not_assessable(
            "blank_record_rate",
            "空白记录率",
            "完整性",
            "dataset",
            "数据集不包含记录，无法计算空白记录率。",
        )
    if len(dataframe.columns) == 0:
        return _not_assessable(
            "blank_record_rate",
            "空白记录率",
            "完整性",
            "dataset",
            "数据集不包含字段，无法判断记录是否为空白。",
        )

    content_columns = _infer_content_columns(dataframe)
    blank_mask = dataframe[content_columns].apply(
        lambda row: all(is_missing_value(value) for value in row), axis=1
    )
    blank_count = int(blank_mask.sum())
    return MetricResult(
        id="blank_record_rate",
        name="空白记录率",
        category="完整性",
        status="evaluated",
        value=round(blank_count / len(dataframe), 6),
        unit="ratio",
        scope="dataset",
        evidence={
            "checked_count": int(len(dataframe)),
            "issue_count": blank_count,
            "method": "inferred_content_fields_blank",
            "content_fields": content_columns,
        },
        issue_locations=_issue_locations(
            (
                position
                for position, is_blank in enumerate(
                    blank_mask.tolist(),
                    start=1,
                )
                if is_blank
            ),
            issue_type="blank_record",
            fields=content_columns,
        ),
    )


def calculate_field_type_consistency(dataframe: pd.DataFrame) -> list[MetricResult]:
    """逐字段计算占比最高的基础类型在非空值中的比例。"""

    if len(dataframe.columns) == 0:
        return _not_assessable_for_fields(
            "field_type_consistency",
            "字段类型一致率",
            "类型一致性",
            "数据集不包含字段，无法计算字段类型一致率。",
        )

    results: list[MetricResult] = []
    for column_name in dataframe.columns:
        typed_rows = [
            (position, inferred_type)
            for position, value in enumerate(
                dataframe[column_name].tolist(),
                start=1,
            )
            if (inferred_type := infer_value_type(value)) is not None
        ]
        if not typed_rows:
            results.append(
                _not_assessable(
                    "field_type_consistency",
                    "字段类型一致率",
                    "类型一致性",
                    "field",
                    "字段没有非空值，无法推断主要数据类型。",
                    str(column_name),
                )
            )
            continue

        type_counts = Counter(
            inferred_type for _, inferred_type in typed_rows
        )
        dominant_type, dominant_count = sorted(
            type_counts.items(), key=lambda item: (-item[1], item[0])
        )[0]
        inconsistent_record_numbers = [
            position
            for position, inferred_type in typed_rows
            if inferred_type != dominant_type
        ]
        results.append(
            MetricResult(
                id="field_type_consistency",
                name="字段类型一致率",
                category="类型一致性",
                status="evaluated",
                value=round(dominant_count / len(typed_rows), 6),
                unit="ratio",
                scope="field",
                field=str(column_name),
                evidence={
                    "checked_count": len(typed_rows),
                    "issue_count": len(inconsistent_record_numbers),
                    "dominant_type": dominant_type,
                    "type_counts": dict(sorted(type_counts.items())),
                },
                issue_locations=_issue_locations(
                    inconsistent_record_numbers,
                    issue_type="inconsistent_type",
                    fields=[str(column_name)],
                ),
            )
        )
    return results


def calculate_recognizable_format_anomaly_rates(
    dataframe: pd.DataFrame,
) -> list[MetricResult]:
    """对可从字段名识别的日期、数值、URL 和邮箱字段计算格式异常率。"""

    field_checks: dict[str, str] = {}
    for column in dataframe.columns:
        name = str(column)
        if field_matches(URL_FIELD_PATTERN, name):
            field_checks[name] = "url"
        elif field_matches(EMAIL_FIELD_PATTERN, name):
            field_checks[name] = "email"
        elif field_matches(DATE_FIELD_PATTERN, name) or field_matches(
            UPDATE_FIELD_PATTERN, name
        ):
            field_checks[name] = "datetime"
        elif field_matches(NUMERIC_FIELD_PATTERN, name):
            field_checks[name] = "numeric"

    if not field_checks:
        return _not_assessable_for_fields(
            "recognizable_format_anomaly_rate",
            "可识别格式异常率",
            "格式规范性",
            "未识别到可检查的日期、数值、URL 或邮箱字段。",
        )

    validators = {
        "datetime": lambda value: _parse_datetime(value) is not None,
        "numeric": _is_valid_finite_number,
        "url": _is_valid_url,
        "email": _is_valid_email,
    }
    results: list[MetricResult] = []
    for field, expected_format in field_checks.items():
        non_missing_rows = [
            (position, value)
            for position, value in enumerate(
                dataframe[field].tolist(),
                start=1,
            )
            if not is_missing_value(value)
        ]
        if not non_missing_rows:
            results.append(
                _not_assessable(
                    "recognizable_format_anomaly_rate",
                    "可识别格式异常率",
                    "格式规范性",
                    "field",
                    "字段没有非空值，无法检查格式。",
                    field,
                )
            )
            continue
        invalid_record_numbers = [
            position
            for position, value in non_missing_rows
            if not validators[expected_format](value)
        ]
        results.append(
            MetricResult(
                id="recognizable_format_anomaly_rate",
                name="可识别格式异常率",
                category="格式规范性",
                status="evaluated",
                value=round(
                    len(invalid_record_numbers) / len(non_missing_rows),
                    6,
                ),
                unit="ratio",
                scope="field",
                field=field,
                evidence={
                    "expected_format": expected_format,
                    "checked_count": len(non_missing_rows),
                    "issue_count": len(invalid_record_numbers),
                    # 仅输出异常数量，不将可能包含个人信息的
                    # 原始值写入 QualityReport。
                    "invalid_samples": [],
                },
                issue_locations=_issue_locations(
                    invalid_record_numbers,
                    issue_type="invalid_format",
                    fields=[field],
                ),
            )
        )
    return results


def _calculate_duplicate_rate(dataframe: pd.DataFrame, normalize: bool) -> MetricResult:
    metric_id = "normalized_duplicate_rate" if normalize else "exact_duplicate_rate"
    name = "规范化重复率" if normalize else "完全重复率"
    if len(dataframe) == 0:
        return _not_assessable(
            metric_id, name, "唯一性", "dataset", "数据集不包含记录，无法检查重复。"
        )

    groups = _duplicate_groups(dataframe, normalize)
    duplicate_count = sum(len(group) - 1 for group in groups)
    duplicate_record_numbers: list[int] = []
    related_record_numbers: dict[int, list[int]] = {}
    for group in groups:
        original_record = group[0]
        for duplicate_record in group[1:]:
            duplicate_record_numbers.append(duplicate_record)
            related_record_numbers[duplicate_record] = [original_record]
    compared_fields = _infer_content_columns(dataframe)
    return MetricResult(
        id=metric_id,
        name=name,
        category="唯一性",
        status="evaluated",
        value=round(duplicate_count / len(dataframe), 6),
        unit="ratio",
        scope="dataset",
        evidence={
            "checked_count": int(len(dataframe)),
            "issue_count": duplicate_count,
            "duplicate_group_count": len(groups),
            "duplicate_groups": [
                {"row_indices": group, "duplicate_count": len(group) - 1}
                for group in groups[:5]
            ],
            "compared_fields": compared_fields,
            "normalization": normalize,
        },
        issue_locations=_issue_locations(
            duplicate_record_numbers,
            issue_type=(
                "normalized_duplicate_record"
                if normalize
                else "exact_duplicate_record"
            ),
            fields=compared_fields,
            related_record_numbers=related_record_numbers,
        ),
    )


def calculate_exact_duplicate_rate(dataframe: pd.DataFrame) -> MetricResult:
    """统计排除技术标识列后，与首次出现记录完全相同的后续记录占比。"""

    return _calculate_duplicate_rate(dataframe, normalize=False)


def calculate_normalized_duplicate_rate(dataframe: pd.DataFrame) -> MetricResult:
    """统计忽略空白、大小写和常见标点差异后的重复占比。"""

    return _calculate_duplicate_rate(dataframe, normalize=True)


def _collect_parsed_dates(
    dataframe: pd.DataFrame, fields: list[str]
) -> list[tuple[int, str, pd.Timestamp]]:
    parsed_dates: list[tuple[int, str, pd.Timestamp]] = []
    for row_position, (_, row) in enumerate(dataframe[fields].iterrows(), start=1):
        for field in fields:
            parsed = _parse_datetime(row[field])
            if parsed is not None:
                parsed_dates.append((row_position, field, parsed))
    return parsed_dates


def calculate_time_info_availability(dataframe: pd.DataFrame) -> MetricResult:
    """统计每条记录是否至少含有一个可解析的日期/时间值。"""

    fields = sorted(
        set(_matching_temporal_columns(dataframe, DATE_FIELD_PATTERN))
        | set(_matching_temporal_columns(dataframe, UPDATE_FIELD_PATTERN))
    )
    if not fields:
        return _not_assessable(
            "time_info_availability",
            "时间信息可用率",
            "及时性",
            "dataset",
            "未识别到日期或时间字段。",
        )
    if len(dataframe) == 0:
        return _not_assessable(
            "time_info_availability",
            "时间信息可用率",
            "及时性",
            "dataset",
            "数据集不包含记录，无法计算时间信息可用率。",
        )

    parsed_dates = _collect_parsed_dates(dataframe, fields)
    available_rows = {row_position for row_position, _, _ in parsed_dates}
    dates = [parsed for _, _, parsed in parsed_dates]
    unavailable_record_numbers = [
        record_number
        for record_number in range(1, len(dataframe) + 1)
        if record_number not in available_rows
    ]
    return MetricResult(
        id="time_info_availability",
        name="时间信息可用率",
        category="及时性",
        status="evaluated",
        value=round(len(available_rows) / len(dataframe), 6),
        unit="ratio",
        scope="dataset",
        evidence={
            "identified_fields": fields,
            "checked_count": int(len(dataframe)),
            "available_count": len(available_rows),
            "issue_count": len(unavailable_record_numbers),
            "earliest_date": min(dates).date().isoformat() if dates else None,
            "latest_date": max(dates).date().isoformat() if dates else None,
        },
        issue_locations=_issue_locations(
            unavailable_record_numbers,
            issue_type="missing_or_invalid_time",
            fields=fields,
        ),
    )


def calculate_update_lag_days(
    dataframe: pd.DataFrame, reference_date: date | None = None
) -> MetricResult:
    """以可识别的最近更新时间计算更新滞后天数。"""

    fields = _matching_temporal_columns(dataframe, UPDATE_FIELD_PATTERN)
    if not fields:
        return _not_assessable(
            "update_lag_days",
            "更新滞后天数",
            "及时性",
            "dataset",
            "未识别到更新时间字段。",
        )
    parsed_dates = _collect_parsed_dates(dataframe, fields)
    if not parsed_dates:
        return _not_assessable(
            "update_lag_days",
            "更新滞后天数",
            "及时性",
            "dataset",
            "未找到可解析的更新时间值。",
        )

    latest = max(parsed for _, _, parsed in parsed_dates).date()
    today = reference_date or date.today()
    return MetricResult(
        id="update_lag_days",
        name="更新滞后天数",
        category="及时性",
        status="evaluated",
        value=(today - latest).days,
        unit="days",
        scope="dataset",
        evidence={
            "identified_fields": fields,
            "latest_update_date": latest.isoformat(),
            "reference_date": today.isoformat(),
        },
    )


def _calculate_coverage(
    dataframe: pd.DataFrame,
    metric_id: str,
    name: str,
    category: str,
    fields: list[str],
    missing_reason: str,
) -> MetricResult:
    if not fields:
        return _not_assessable(metric_id, name, category, "dataset", missing_reason)
    if len(dataframe) == 0:
        return _not_assessable(
            metric_id, name, category, "dataset", "数据集不包含记录，无法计算覆盖率。"
        )
    covered_count = 0
    issue_count = 0
    missing_row_indices: list[int] = []
    issue_record_numbers: list[int] = []
    for position, (_, row) in enumerate(dataframe[fields].iterrows(), start=1):
        if any(not is_missing_value(value) for value in row.tolist()):
            covered_count += 1
            continue
        issue_count += 1
        if len(missing_row_indices) < 20:
            missing_row_indices.append(position)
        issue_record_numbers.append(position)
    issue_type = (
        "missing_source_info"
        if metric_id == "source_info_coverage"
        else "missing_version_info"
    )
    return MetricResult(
        id=metric_id,
        name=name,
        category=category,
        status="evaluated",
        value=round(covered_count / len(dataframe), 6),
        unit="ratio",
        scope="dataset",
        evidence={
            "identified_fields": fields,
            "checked_count": int(len(dataframe)),
            "covered_count": covered_count,
            "issue_count": issue_count,
            "missing_row_indices": missing_row_indices,
        },
        issue_locations=_issue_locations(
            issue_record_numbers,
            issue_type=issue_type,
            fields=fields,
        ),
    )


def calculate_source_info_coverage(dataframe: pd.DataFrame) -> MetricResult:
    """统计每条记录是否包含可识别的来源字段信息。"""

    fields = sorted(
        set(_matching_columns(dataframe, SOURCE_FIELD_PATTERN))
        | set(_matching_columns(dataframe, URL_FIELD_PATTERN))
        | set(_matching_columns(dataframe, SOURCE_IDENTIFIER_FIELD_PATTERN))
    )
    return _calculate_coverage(
        dataframe,
        "source_info_coverage",
        "来源信息覆盖率",
        "可溯性",
        fields,
        "未识别到来源部门、来源链接或原始标识字段。",
    )


def calculate_version_info_coverage(dataframe: pd.DataFrame) -> MetricResult:
    """统计每条记录是否包含版本、更新或处理记录信息。"""

    fields = sorted(
        set(_matching_columns(dataframe, VERSION_FIELD_PATTERN))
        | set(_matching_temporal_columns(dataframe, UPDATE_FIELD_PATTERN))
    )
    return _calculate_coverage(
        dataframe,
        "version_info_coverage",
        "版本信息覆盖率",
        "可溯性",
        fields,
        "未识别到版本号、更新时间或处理记录字段。",
    )


def calculate_statistical_outlier_rates(dataframe: pd.DataFrame) -> list[MetricResult]:
    """对具有足够数值的非标识字段按 IQR 规则统计异常值。"""

    results: list[MetricResult] = []
    for field in _infer_content_columns(dataframe):
        non_missing_rows = [
            (position, value)
            for position, value in enumerate(
                dataframe[field].tolist(),
                start=1,
            )
            if not is_missing_value(value)
        ]
        if not non_missing_rows:
            continue
        numeric_values = pd.to_numeric(
            pd.Series(
                [_format_value(value) for _, value in non_missing_rows]
            ),
            errors="coerce",
        )
        numeric_count = int(numeric_values.notna().sum())
        finite_mask = numeric_values.map(
            lambda value: pd.notna(value) and math.isfinite(float(value))
        )
        finite_count = int(finite_mask.sum())
        if (
            finite_count < 4
            or numeric_count / len(non_missing_rows) < 0.8
        ):
            continue
        numeric_series = numeric_values[finite_mask]
        first_quartile = float(numeric_series.quantile(0.25))
        third_quartile = float(numeric_series.quantile(0.75))
        if not all(math.isfinite(value) for value in (first_quartile, third_quartile)):
            results.append(
                _not_assessable(
                    "statistical_outlier_rate",
                    "统计异常值比例",
                    "数据异常",
                    "field",
                    "数值范围过大，IQR 四分位数无法表示为有限数值。",
                    field,
                )
            )
            continue
        iqr = third_quartile - first_quartile
        if not math.isfinite(iqr):
            results.append(
                _not_assessable(
                    "statistical_outlier_rate",
                    "统计异常值比例",
                    "数据异常",
                    "field",
                    "数值范围过大，IQR 差值无法表示为有限数值。",
                    field,
                )
            )
            continue
        lower_bound = first_quartile - 1.5 * iqr
        upper_bound = third_quartile + 1.5 * iqr
        if not all(math.isfinite(value) for value in (lower_bound, upper_bound)):
            results.append(
                _not_assessable(
                    "statistical_outlier_rate",
                    "统计异常值比例",
                    "数据异常",
                    "field",
                    "数值范围过大，IQR 异常边界无法表示为有限数值。",
                    field,
                )
            )
            continue
        outlier_mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)
        non_finite_count = numeric_count - finite_count
        issue_count = int(outlier_mask.sum()) + non_finite_count
        issue_value_indices = sorted(
            [
                int(index)
                for index, is_outlier in outlier_mask.items()
                if bool(is_outlier)
            ]
            + [
                int(index)
                for index, value in numeric_values.items()
                if pd.notna(value) and not math.isfinite(float(value))
            ]
        )
        issue_record_numbers = [
            non_missing_rows[index][0]
            for index in issue_value_indices
        ]
        results.append(
            MetricResult(
                id="statistical_outlier_rate",
                name="统计异常值比例",
                category="数据异常",
                status="evaluated",
                value=round(issue_count / numeric_count, 6),
                unit="ratio",
                scope="field",
                field=field,
                evidence={
                    "checked_count": numeric_count,
                    "issue_count": issue_count,
                    "non_finite_count": non_finite_count,
                    "q1": first_quartile,
                    "q3": third_quartile,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    # 边界与数量足以解释计算，不导出原始异常值。
                    "outlier_samples": [],
                    "non_finite_samples": [],
                },
                issue_locations=_issue_locations(
                    issue_record_numbers,
                    issue_type="statistical_outlier",
                    fields=[field],
                ),
            )
        )
    if results:
        return results
    return _not_assessable_for_fields(
        "statistical_outlier_rate",
        "统计异常值比例",
        "数据异常",
        "未识别到至少含 4 个可用数值的非标识字段。",
    )


MetricCalculator = Callable[[pd.DataFrame], list[MetricResult]]


def _file_parse_results(_: pd.DataFrame) -> list[MetricResult]:
    return [calculate_file_parse_rate()]


def _dataset_scale_results(dataframe: pd.DataFrame) -> list[MetricResult]:
    return [calculate_dataset_scale(dataframe)]


def _blank_record_results(dataframe: pd.DataFrame) -> list[MetricResult]:
    return [calculate_blank_record_rate(dataframe)]


def _exact_duplicate_results(dataframe: pd.DataFrame) -> list[MetricResult]:
    return [calculate_exact_duplicate_rate(dataframe)]


def _normalized_duplicate_results(dataframe: pd.DataFrame) -> list[MetricResult]:
    return [calculate_normalized_duplicate_rate(dataframe)]


def _time_availability_results(dataframe: pd.DataFrame) -> list[MetricResult]:
    return [calculate_time_info_availability(dataframe)]


def _update_lag_results(
    dataframe: pd.DataFrame, reference_date: date | None = None
) -> list[MetricResult]:
    return [calculate_update_lag_days(dataframe, reference_date=reference_date)]


def _source_coverage_results(dataframe: pd.DataFrame) -> list[MetricResult]:
    return [calculate_source_info_coverage(dataframe)]


def _version_coverage_results(dataframe: pd.DataFrame) -> list[MetricResult]:
    return [calculate_version_info_coverage(dataframe)]


# 指标执行注册表是增删指标的唯一编排入口。
# 每个计算器始终返回 list[MetricResult]，因此数据集级和字段级指标可统一调度。
METRIC_CALCULATORS: tuple[tuple[str, MetricCalculator], ...] = (
    ("file_parse_rate", _file_parse_results),
    ("dataset_scale", _dataset_scale_results),
    ("field_missing_rate", calculate_field_missing_rates),
    ("blank_record_rate", _blank_record_results),
    ("field_type_consistency", calculate_field_type_consistency),
    ("recognizable_format_anomaly_rate", calculate_recognizable_format_anomaly_rates),
    ("exact_duplicate_rate", _exact_duplicate_results),
    ("normalized_duplicate_rate", _normalized_duplicate_results),
    ("time_info_availability", _time_availability_results),
    ("update_lag_days", _update_lag_results),
    ("source_info_coverage", _source_coverage_results),
    ("version_info_coverage", _version_coverage_results),
    ("statistical_outlier_rate", calculate_statistical_outlier_rates),
)


def calculate_all_metrics(
    dataframe: pd.DataFrame, reference_date: date | None = None
) -> list[MetricResult]:
    """按注册表顺序计算所有已启用的零配置指标。"""

    results: list[MetricResult] = []
    for metric_id, calculator in METRIC_CALCULATORS:
        if metric_id == "update_lag_days":
            results.extend(_update_lag_results(dataframe, reference_date))
        else:
            results.extend(calculator(dataframe))
    return results


def calculate_failed_metrics(reason: str) -> list[MetricResult]:
    """解析失败时根据指标目录动态生成可评估性状态。"""

    unavailable_reason = f"文件未成功解析，无法计算：{reason}"
    unavailable_metrics = [
        _not_assessable(
            item["id"],
            item["name"],
            item["category"],
            "dataset",
            unavailable_reason,
        )
        for item in METRIC_CATALOG
        if item["id"] != "file_parse_rate"
    ]
    return [calculate_file_parse_rate(successful=False), *unavailable_metrics]
