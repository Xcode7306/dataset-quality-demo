"""将确定性指标转换为保守的自动风险提示。"""

from collections import defaultdict
from typing import Iterable, Literal

from .config import DEFAULT_RISK_THRESHOLDS, RiskThresholds
from .models import MetricResult, RiskItem


RiskLevel = Literal["info", "attention", "warning"]


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _field_suffix(metric: MetricResult) -> str:
    return f":{metric.field}" if metric.field else ""


def _field_text(metric: MetricResult) -> str:
    return f"字段“{metric.field}”" if metric.field else "数据集"


def _high_is_risky(
    value: float, attention: float, warning: float
) -> RiskLevel | None:
    if value >= warning:
        return "warning"
    if value > attention:
        return "attention"
    return None


def _low_is_risky(
    value: float, attention: float, warning: float
) -> RiskLevel | None:
    if value <= warning:
        return "warning"
    if value < attention:
        return "attention"
    return None


def _risk(
    metric: MetricResult,
    risk_id: str,
    level: RiskLevel,
    title: str,
    message: str,
    evidence: dict | None = None,
) -> RiskItem:
    return RiskItem(
        id=f"{risk_id}{_field_suffix(metric)}",
        level=level,
        title=title,
        message=message,
        related_metrics=[metric.id],
        evidence={"field": metric.field, **metric.evidence, **(evidence or {})},
    )


def _evaluated_metrics(metrics: Iterable[MetricResult]) -> list[MetricResult]:
    return [
        metric
        for metric in metrics
        if metric.status == "evaluated" and metric.value is not None
    ]


def generate_risks(
    metrics: Iterable[MetricResult],
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
) -> list[RiskItem]:
    """根据指标和集中阈值生成风险提示。

    `not_assessable` 指标不产生风险；本函数不改动任何指标值。
    """

    evaluated = _evaluated_metrics(metrics)
    by_id: dict[str, list[MetricResult]] = defaultdict(list)
    for metric in evaluated:
        by_id[metric.id].append(metric)

    risks: list[RiskItem] = []

    for metric in by_id.get("file_parse_rate", []):
        if float(metric.value) < 1.0:
            risks.append(
                _risk(
                    metric,
                    "file_parse_failed",
                    "warning",
                    "文件未成功解析",
                    "本次文件未成功解析，其余质量指标无法计算。",
                )
            )

    for metric in by_id.get("dataset_scale", []):
        if int(metric.value) == 0:
            risks.append(
                _risk(
                    metric,
                    "empty_dataset",
                    "warning",
                    "数据集不包含记录",
                    "文件可读取，但没有可供评估的数据记录。",
                )
            )

    for metric in by_id.get("field_missing_rate", []):
        value = float(metric.value)
        level = _high_is_risky(
            value,
            thresholds.field_missing_attention,
            thresholds.field_missing_warning,
        )
        if level:
            risks.append(
                _risk(
                    metric,
                    "high_field_missing_rate",
                    level,
                    "字段缺失较多",
                    f"{_field_text(metric)}的缺失率为 {_percent(value)}，建议核对该字段的可用性。",
                )
            )

    for metric in by_id.get("blank_record_rate", []):
        value = float(metric.value)
        level = _high_is_risky(
            value,
            thresholds.blank_record_attention,
            thresholds.blank_record_warning,
        )
        if level:
            risks.append(
                _risk(
                    metric,
                    "blank_records_detected",
                    level,
                    "发现空白记录",
                    f"可识别内容字段均为空的记录占 {_percent(value)}，建议检查对应记录。",
                )
            )

    for metric in by_id.get("field_type_consistency", []):
        value = float(metric.value)
        level = _low_is_risky(
            value,
            thresholds.type_consistency_attention,
            thresholds.type_consistency_warning,
        )
        if level:
            risks.append(
                _risk(
                    metric,
                    "low_type_consistency",
                    level,
                    "字段类型存在混杂",
                    f"{_field_text(metric)}的主要类型一致率为 {_percent(value)}，建议查看混入的值类型。",
                )
            )

    for metric in by_id.get("recognizable_format_anomaly_rate", []):
        value = float(metric.value)
        level = _high_is_risky(
            value,
            thresholds.format_anomaly_attention,
            thresholds.format_anomaly_warning,
        )
        if level:
            risks.append(
                _risk(
                    metric,
                    "format_anomalies_detected",
                    level,
                    "发现可识别格式异常",
                    f"{_field_text(metric)}的可识别格式异常率为 {_percent(value)}，"
                    f"报告统计 {int(metric.evidence.get('issue_count', 0))} 条异常；"
                    "建议回到原始数据按对应格式规则核对。",
                )
            )

    exact_duplicate_value = next(
        (
            float(metric.value)
            for metric in by_id.get("exact_duplicate_rate", [])
        ),
        0.0,
    )
    for metric in by_id.get("exact_duplicate_rate", []):
        value = float(metric.value)
        level = _high_is_risky(
            value,
            thresholds.duplicate_attention,
            thresholds.duplicate_warning,
        )
        if level:
            risks.append(
                _risk(
                    metric,
                    "exact_duplicates_detected",
                    level,
                    "发现完全重复记录",
                    f"排除明显技术标识字段后，完全重复记录占 {_percent(value)}。",
                )
            )

    for metric in by_id.get("normalized_duplicate_rate", []):
        value = float(metric.value)
        additional_rate = max(0.0, value - exact_duplicate_value)
        level = _high_is_risky(
            additional_rate,
            thresholds.duplicate_attention,
            thresholds.duplicate_warning,
        )
        if level:
            risks.append(
                _risk(
                    metric,
                    "normalized_duplicates_detected",
                    level,
                    "发现额外的规范化重复",
                    f"忽略空白、大小写和常见标点后，额外发现 {_percent(additional_rate)} 的重复记录。",
                    {"additional_duplicate_rate": additional_rate},
                )
            )

    for metric in by_id.get("time_info_availability", []):
        value = float(metric.value)
        level = _low_is_risky(
            value,
            thresholds.time_availability_attention,
            thresholds.time_availability_warning,
        )
        if level:
            risks.append(
                _risk(
                    metric,
                    "low_time_availability",
                    level,
                    "时间信息覆盖不足",
                    f"仅有 {_percent(value)} 的记录含可解析时间，时效性结论可能受限。",
                )
            )

    for metric in by_id.get("update_lag_days", []):
        value = int(metric.value)
        if value < 0:
            risks.append(
                _risk(
                    metric,
                    "future_update_date",
                    "attention",
                    "最近更新日期晚于评估日期",
                    f"最近更新日期比评估日期晚 {abs(value)} 天，建议核对日期及时区。",
                )
            )
        elif value >= thresholds.update_lag_warning_days:
            risks.append(
                _risk(
                    metric,
                    "long_update_lag",
                    "warning",
                    "距最近更新时间较长",
                    f"距可识别的最近更新日期已有 {value} 天，建议结合数据更新频率进一步判断。",
                )
            )
        elif value >= thresholds.update_lag_attention_days:
            risks.append(
                _risk(
                    metric,
                    "long_update_lag",
                    "attention",
                    "距最近更新时间较长",
                    f"距可识别的最近更新日期已有 {value} 天，请结合业务更新周期复核。",
                )
            )

    for metric_id, risk_id, title in (
        ("source_info_coverage", "low_source_coverage", "来源信息覆盖不足"),
        ("version_info_coverage", "low_version_coverage", "版本信息覆盖不足"),
    ):
        for metric in by_id.get(metric_id, []):
            value = float(metric.value)
            level = _low_is_risky(
                value,
                thresholds.coverage_attention,
                thresholds.coverage_warning,
            )
            if level:
                risks.append(
                    _risk(
                        metric,
                        risk_id,
                        level,
                        title,
                        f"当前覆盖率为 {_percent(value)}，建议查看报告中未覆盖的记录。",
                    )
                )

    for metric in by_id.get("statistical_outlier_rate", []):
        if int(metric.evidence.get("issue_count", 0)) > 0:
            risks.append(
                _risk(
                    metric,
                    "statistical_outliers_detected",
                    "info",
                    "数值字段存在统计异常值",
                    f"{_field_text(metric)}按 IQR 规则识别到 {metric.evidence['issue_count']} 个统计异常值，该提示不代表数据一定错误。",
                )
            )

    return risks
