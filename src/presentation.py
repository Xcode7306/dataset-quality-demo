"""将质量报告转换为界面可直接展示的数据，不参与质量计算。"""

import json
from collections import Counter
from typing import Any

from .models import MetricResult, QualityReport


RISK_LEVEL_LABELS = {
    "warning": "警告",
    "attention": "关注",
    "info": "提示",
}
METRIC_STATUS_LABELS = {
    "evaluated": "已评估",
    "not_assessable": "无法评估",
}
TYPE_LABELS = {
    "boolean": "布尔",
    "datetime": "日期时间",
    "numeric": "数值",
    "text": "文本",
    "unknown": "未知",
}


def format_metric_value(metric: MetricResult) -> str:
    """按统一规则格式化指标值。"""

    if metric.status == "not_assessable" or metric.value is None:
        return "—"
    if metric.unit == "ratio":
        return f"{float(metric.value):.2%}"
    if metric.unit == "records":
        return f"{int(metric.value):,} 条"
    if metric.unit == "days":
        return f"{int(metric.value):,} 天"
    return f"{metric.value:,}" if isinstance(metric.value, int) else str(metric.value)


def build_summary(report: QualityReport) -> dict[str, int]:
    """生成页面首屏使用的规模、指标和风险摘要。"""

    evaluated_metric_ids = {
        metric.id for metric in report.metrics if metric.status == "evaluated"
    }
    all_metric_ids = {metric.id for metric in report.metrics}
    risk_counts = Counter(risk.level for risk in report.risks)
    return {
        "row_count": int(report.profile.get("row_count", 0)),
        "column_count": int(report.profile.get("column_count", 0)),
        "evaluated_metric_count": len(evaluated_metric_ids),
        "metric_count": len(all_metric_ids),
        "risk_count": len(report.risks),
        "not_assessable_count": len(report.not_assessable),
        "warning_count": risk_counts["warning"],
        "attention_count": risk_counts["attention"],
        "info_count": risk_counts["info"],
    }


def build_metric_rows(report: QualityReport) -> list[dict[str, Any]]:
    """生成指标明细表；保留字段级结果和无法评估原因。"""

    return [
        {
            "类别": metric.category,
            "指标": metric.name,
            "范围": "字段" if metric.scope == "field" else "数据集",
            "字段": metric.field or "—",
            "状态": METRIC_STATUS_LABELS[metric.status],
            "结果": format_metric_value(metric),
            "原因": metric.reason or "—",
        }
        for metric in report.metrics
    ]


def build_profile_rows(report: QualityReport) -> list[dict[str, Any]]:
    """生成字段画像表。"""

    rows = []
    for column in report.profile.get("columns", []):
        missing_rate = column.get("missing_rate")
        rows.append(
            {
                "字段": column.get("name", ""),
                "推断类型": TYPE_LABELS.get(
                    column.get("inferred_type", "unknown"),
                    str(column.get("inferred_type", "unknown")),
                ),
                "缺失数": int(column.get("missing_count", 0)),
                "非空数": int(column.get("non_missing_count", 0)),
                "缺失率": "—" if missing_rate is None else f"{float(missing_rate):.2%}",
            }
        )
    return rows


def build_risk_chart_rows(report: QualityReport) -> list[dict[str, Any]]:
    """以固定顺序生成三档风险分布，零值也保留。"""

    counts = Counter(risk.level for risk in report.risks)
    return [
        {"级别": RISK_LEVEL_LABELS[level], "数量": counts[level]}
        for level in ("warning", "attention", "info")
    ]


def serialize_report(report: QualityReport) -> bytes:
    """生成与文件报告一致的 UTF-8 JSON 下载内容。"""

    return json.dumps(
        report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8")
