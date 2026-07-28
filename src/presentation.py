"""将质量报告转换为界面、Markdown 或内部结构化内容，不参与质量计算。"""

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


def _markdown_cell(value: object) -> str:
    """将表格内容转为安全、紧凑的 Markdown 单元格。"""

    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    """构建 Markdown 表格；没有内容时由调用方展示说明文字。"""

    return [
        "| " + " | ".join(_markdown_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *[
            "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
            for row in rows
        ],
    ]


def _status_summary(report: QualityReport, summary: dict[str, int]) -> str:
    if report.status == "failed":
        return "文件未能成功解析，以下结果仅说明失败原因和无法评估的项目。"
    if summary["warning_count"]:
        return (
            f"评估完成，发现 {summary['warning_count']} 项警告，"
            "建议优先查看“风险提示”。"
        )
    if summary["attention_count"]:
        return (
            f"评估完成，发现 {summary['attention_count']} 项需要关注的现象，"
            "建议结合业务规则复核。"
        )
    return "评估完成，当前默认规则未发现警告或需要关注的现象。"


def render_markdown_report(report: QualityReport) -> str:
    """生成供人阅读的 UTF-8 Markdown 质量评估报告。"""

    summary = build_summary(report)
    dataset = report.dataset
    lines = [
        f"# 数据集质量评估报告：{dataset.name}",
        "",
        _status_summary(report, summary),
        "",
        "## 数据集概况",
        "",
        *(
            f"- {label}：{value}"
            for label, value in (
                ("文件名", dataset.file_name),
                ("文件类型", dataset.file_type.upper()),
                ("工作表", dataset.sheet_name or "未指定"),
                ("评估状态", "评估完成" if report.status != "failed" else "评估失败"),
                ("记录数", f"{summary['row_count']:,}"),
                ("字段数", f"{summary['column_count']:,}"),
            )
        ),
        "",
        "## 评估摘要",
        "",
        *(
            f"- {label}：{value}"
            for label, value in (
                ("已评估指标", f"{summary['evaluated_metric_count']}/{summary['metric_count']}"),
                ("风险提示", summary["risk_count"]),
                ("警告", summary["warning_count"]),
                ("关注", summary["attention_count"]),
                ("提示", summary["info_count"]),
                ("无法评估项", summary["not_assessable_count"]),
            )
        ),
        "",
        "## 风险提示",
        "",
    ]

    if not report.risks:
        lines.append("当前默认阈值下没有生成风险提示。")
    else:
        for level in ("warning", "attention", "info"):
            risks = [risk for risk in report.risks if risk.level == level]
            if not risks:
                continue
            if lines[-1] != "":
                lines.append("")
            lines.extend([f"### {RISK_LEVEL_LABELS[level]}（{len(risks)}）", ""])
            for risk in risks:
                related = "、".join(risk.related_metrics) or "无"
                lines.extend(
                    [
                        f"- **{risk.title}**：{risk.message}",
                        f"  - 关联指标：{related}",
                    ]
                )

    lines.extend(["", "## 指标明细", ""])
    metric_rows = build_metric_rows(report)
    if metric_rows:
        lines.extend(
            _markdown_table(
                ["类别", "指标", "范围", "字段", "状态", "结果", "原因"],
                [
                    [
                        row["类别"], row["指标"], row["范围"], row["字段"],
                        row["状态"], row["结果"], row["原因"],
                    ]
                    for row in metric_rows
                ],
            )
        )
    else:
        lines.append("当前报告不包含指标明细。")

    lines.extend(["", "## 字段画像", ""])
    profile_rows = build_profile_rows(report)
    if profile_rows:
        lines.extend(
            _markdown_table(
                ["字段", "推断类型", "缺失数", "非空数", "缺失率"],
                [
                    [
                        row["字段"], row["推断类型"], row["缺失数"],
                        row["非空数"], row["缺失率"],
                    ]
                    for row in profile_rows
                ],
            )
        )
    else:
        lines.append("当前文件没有可展示的字段画像。")

    lines.extend(["", "## 无法评估项与运行信息", ""])
    if report.not_assessable:
        lines.extend(
            _markdown_table(
                ["项目", "原因"],
                [[item.name, item.reason] for item in report.not_assessable],
            )
        )
    else:
        lines.append("本次报告没有无法评估项。")

    warnings = report.execution.get("warnings", [])
    errors = report.execution.get("errors", [])
    if warnings or errors:
        lines.extend(["", "### 运行信息", ""])
        lines.extend(f"- 警告：{message}" for message in warnings)
        lines.extend(f"- 错误：{message}" for message in errors)

    lines.extend(
        [
            "",
            "---",
            "",
            "说明：风险提示表示建议复核的现象，不等同于数据错误；"
            "无法评估项不会以 0 替代。报告不包含原始字段样例。",
            "",
        ]
    )
    return "\n".join(lines)


def serialize_markdown_report(report: QualityReport) -> bytes:
    """生成供用户下载的 UTF-8 Markdown 报告。"""

    return render_markdown_report(report).encode("utf-8")


def serialize_report(report: QualityReport) -> bytes:
    """保留结构化报告序列化，供内部兼容性检查或后续系统使用。"""

    return json.dumps(
        report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8")
