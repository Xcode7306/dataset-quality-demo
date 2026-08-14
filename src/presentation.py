"""将质量报告转换为界面、Markdown 或内部结构化内容，不参与质量计算。"""

import csv
import io
import json
from collections import Counter
from typing import Any

from .metric_catalog import (
    DB31_METRIC_IDS,
    ORIGINAL_METRIC_IDS,
    get_metric_definition,
)
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
ISSUE_TYPE_LABELS = {
    "missing_value": "字段值缺失",
    "blank_record": "空白记录",
    "inconsistent_type": "字段类型不一致",
    "invalid_format": "格式异常",
    "exact_duplicate_record": "完全重复记录",
    "normalized_duplicate_record": "规范化后重复记录",
    "missing_or_invalid_time": "时间信息缺失或不可解析",
    "missing_source_info": "来源信息缺失",
    "missing_version_info": "版本信息缺失",
    "statistical_outlier": "统计异常值",
    "rule_primary_key_missing": "主键缺失",
    "rule_primary_key_duplicate": "主键重复",
    "rule_required_missing": "必填字段缺失",
    "rule_update_time_missing_or_invalid": "更新时间缺失或不可解析",
    "rule_allowed_value_violation": "不在允许值范围",
    "rule_numeric_range_violation": "超出数值范围",
}


def spreadsheet_safe_cell(value: Any) -> Any:
    """阻止不可信文本在 Excel/LibreOffice 中被当作公式执行。"""

    if not isinstance(value, str):
        return value
    if value.lstrip().startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return value


def _metric_catalog_details(metric: MetricResult) -> dict[str, str]:
    """返回指标的来源与计算口径；兼容 RulePack 动态追加指标。"""

    definition = get_metric_definition(metric.id)
    if definition is not None:
        return {
            "来源": str(definition["source_label"]),
            "标准代码": str(definition["standard_code"] or "—"),
            "评价维度": str(definition["dimension"]),
            "层级": str(definition["level"]),
            "计算方式": str(definition["formula"]),
        }
    if metric.id.startswith(("business_", "rule_")):
        return {
            "来源": "已审批 RulePack",
            "标准代码": "—",
            "评价维度": metric.category,
            "层级": "业务规则",
            "计算方式": "按已审批 RulePack 中的确定性规则计算",
        }
    return {
        "来源": "扩展指标",
        "标准代码": "—",
        "评价维度": metric.category,
        "层级": "—",
        "计算方式": "当前指标目录未登记计算方式",
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
    """生成面向用户的指标明细表；内部引用键不在默认视图中展示。"""

    rows: list[dict[str, Any]] = []
    for metric in report.metrics:
        rows.append(
            {
                "指标名称": metric.name,
                "字段名称": metric.field or "—",
                "类别": metric.category,
                "范围": "字段" if metric.scope == "field" else "数据集",
                "状态": METRIC_STATUS_LABELS[metric.status],
                "结果": format_metric_value(metric),
                "原因": metric.reason or "—",
                **_metric_catalog_details(metric),
            }
        )
    return rows


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


def build_issue_location_rows(
    report: QualityReport,
    *,
    metric_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """生成供独立 CSV 导出的完整疑似问题位置明细。"""

    rows: list[dict[str, Any]] = []
    for metric in report.metrics:
        if metric_keys is not None and metric.metric_key not in metric_keys:
            continue
        for location in metric.issue_locations:
            if not isinstance(location, dict):
                continue
            record_number = location.get("record_number")
            if (
                isinstance(record_number, bool)
                or not isinstance(record_number, int)
                or record_number < 1
            ):
                continue
            fields = location.get("fields", [])
            field_names = (
                [str(field) for field in fields]
                if isinstance(fields, list)
                else []
            )
            if not field_names and metric.field:
                field_names = [metric.field]
            related = location.get("related_record_numbers", [])
            related_numbers = (
                [
                    str(number)
                    for number in related
                    if isinstance(number, int)
                    and not isinstance(number, bool)
                    and number > 0
                ]
                if isinstance(related, list)
                else []
            )
            issue_type = str(location.get("issue_type") or "unknown")
            related_text = "、".join(related_numbers)
            if issue_type == "exact_duplicate_record" and related_text:
                note = (
                    f"第 {record_number} 条记录与第 {related_text} 条记录"
                    f"内容完全相同；关联记录序号 {related_text} 表示这组"
                    "重复数据中首次出现的记录。"
                )
            elif issue_type == "normalized_duplicate_record" and related_text:
                note = (
                    f"第 {record_number} 条记录与第 {related_text} 条记录"
                    "在忽略自然文本中的大小写、空白和标点差异后相同；"
                    f"关联记录序号 {related_text} 表示这组重复数据中"
                    "首次出现的记录。"
                )
            else:
                note = ""
            rows.append(
                {
                    "疑似问题类型": ISSUE_TYPE_LABELS.get(
                        issue_type,
                        issue_type,
                    ),
                    "指标名称": metric.name,
                    "字段名称": "、".join(field_names) or "整条记录",
                    "数据记录序号": record_number,
                    "关联记录序号": related_text or "—",
                    "备注": note,
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

    return _markdown_text(value).replace("\n", "<br>")


def _markdown_text(value: object) -> str:
    """转义报告数据中的 Markdown、HTML 和外链图片语法。"""

    text = str(value)
    for character in ("\\", "`", "*", "[", "]", "(", ")", "<", ">", "#", "!", "|"):
        text = text.replace(character, f"\\{character}")
    return text


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
    has_rule_metrics = any(
        metric.id.startswith(("rule_", "business_"))
        for metric in report.metrics
    )
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
    if has_rule_metrics:
        return "规则增强评估完成，零配置规则与已审批业务规则均未发现警告或需要关注的现象。"
    return "评估完成，当前默认规则未发现警告或需要关注的现象。"


def _metric_selection_summary(
    report: QualityReport,
    evaluation_context: dict[str, Any],
) -> str:
    """生成写入可读报告的指标来源统计。"""

    selected = evaluation_context.get("selected_metric_ids")
    if isinstance(selected, (list, tuple)):
        selected_ids = tuple(
            metric_id
            for metric_id in selected
            if isinstance(metric_id, str)
        )
    else:
        present = {metric.id for metric in report.metrics}
        selected_ids = tuple(
            metric_id
            for metric_id in (*ORIGINAL_METRIC_IDS, *DB31_METRIC_IDS)
            if metric_id in present
        )
    original_count = sum(
        metric_id in ORIGINAL_METRIC_IDS for metric_id in selected_ids
    )
    db31_count = sum(
        metric_id in DB31_METRIC_IDS for metric_id in selected_ids
    )
    return (
        f"共 {len(selected_ids)} 项（原 v0.4 指标 {original_count} 项，"
        f"DB31/T 1523-2024 指标 {db31_count} 项）"
    )


def render_markdown_report(report: QualityReport) -> str:
    """生成供人阅读的 UTF-8 Markdown 质量评估报告。"""

    summary = build_summary(report)
    dataset = report.dataset
    evaluation_context = report.to_dict().get("evaluation_context", {})
    lines = [
        f"# 数据集质量评估报告：{_markdown_text(dataset.name)}",
        "",
        _status_summary(report, summary),
        "",
        "## 数据集概况",
        "",
        *(
            f"- {label}：{_markdown_text(value)}"
            for label, value in (
                ("文件名", dataset.file_name),
                ("文件类型", dataset.file_type.upper()),
                ("工作表", dataset.sheet_name or "未指定"),
                ("评估状态", "评估完成" if report.status != "failed" else "评估失败"),
                ("记录数", f"{summary['row_count']:,}"),
                ("字段数", f"{summary['column_count']:,}"),
                ("报告哈希", evaluation_context.get("report_sha256", "—")),
                ("引擎版本", evaluation_context.get("engine_version", "—")),
                ("评估基准日期", evaluation_context.get("reference_date", "—")),
                ("阈值配置版本", evaluation_context.get("threshold_config_version", "—")),
                ("解析路径", evaluation_context.get("parser_path", "—")),
                (
                    "指标选择",
                    _metric_selection_summary(report, evaluation_context),
                ),
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
        has_rule_metrics = any(
            metric.id.startswith(("rule_", "business_"))
            for metric in report.metrics
        )
        lines.append(
            "当前默认阈值和已审批业务规则下没有生成风险提示。"
            if has_rule_metrics
            else "当前默认阈值下没有生成风险提示。"
        )
    else:
        for level in ("warning", "attention", "info"):
            risks = [risk for risk in report.risks if risk.level == level]
            if not risks:
                continue
            if lines[-1] != "":
                lines.append("")
            lines.extend([f"### {RISK_LEVEL_LABELS[level]}（{len(risks)}）", ""])
            for risk in risks:
                related = "、".join(
                    risk.related_metric_keys or risk.related_metrics
                ) or "无"
                lines.extend(
                    [
                        f"- **{_markdown_text(risk.title)}**："
                        f"{_markdown_text(risk.message)}",
                        f"  - 关联指标：{_markdown_text(related)}",
                    ]
                )

    lines.extend(["", "## 指标明细", ""])
    metric_rows = build_metric_rows(report)
    if metric_rows:
        lines.extend(
            _markdown_table(
                [
                    "指标名称",
                    "字段名称",
                    "类别",
                    "范围",
                    "状态",
                    "结果",
                    "原因",
                    "来源",
                    "标准代码",
                    "评价维度",
                    "层级",
                    "计算方式",
                ],
                [
                    [
                        row["指标名称"], row["字段名称"],
                        row["类别"], row["范围"],
                        row["状态"], row["结果"], row["原因"],
                        row["来源"], row["标准代码"],
                        row["评价维度"], row["层级"],
                        row["计算方式"],
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
        lines.extend(f"- 警告：{_markdown_text(message)}" for message in warnings)
        lines.extend(f"- 错误：{_markdown_text(message)}" for message in errors)

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


def serialize_issue_locations_csv(report: QualityReport) -> bytes:
    """生成含全部疑似问题位置、且不含原始值的 Excel 兼容 CSV。"""

    fieldnames = [
        "疑似问题类型",
        "指标名称",
        "字段名称",
        "数据记录序号",
        "关联记录序号",
        "备注",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(
        {
            key: spreadsheet_safe_cell(value)
            for key, value in row.items()
        }
        for row in build_issue_location_rows(report)
    )
    return output.getvalue().encode("utf-8-sig")


def serialize_report(report: QualityReport) -> bytes:
    """保留结构化报告序列化，供内部兼容性检查或后续系统使用。"""

    return json.dumps(
        report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8")
