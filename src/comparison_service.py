"""v0.5 两份固定 QualityReport 的确定性比较服务。"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .comparison_models import (
    COMPARISON_POLICY_VERSION,
    COMPARISON_SCHEMA_VERSION,
    COMPARATOR_VERSION,
    AssessabilityChange,
    MetricChange,
    ReportComparison,
    ReportReference,
    RiskChange,
    SchemaChange,
)
from .history_store import (
    HistoryValidationError,
    validate_quality_report_payload,
)
from .models import QualityReport


_HIGHER_IS_BETTER = frozenset(
    {
        "file_parse_rate",
        "field_type_consistency",
        "time_info_availability",
        "source_info_coverage",
        "version_info_coverage",
        "business_primary_key_compliance",
        "business_required_compliance",
        "business_update_time_parseability",
        "business_update_frequency_compliance",
        "business_allowed_values_compliance",
        "business_numeric_range_compliance",
    }
)
_LOWER_IS_BETTER = frozenset(
    {
        "field_missing_rate",
        "blank_record_rate",
        "recognizable_format_anomaly_rate",
        "exact_duplicate_rate",
        "normalized_duplicate_rate",
        "update_lag_days",
    }
)
_REFERENCE_DATE_SENSITIVE_METRICS = frozenset(
    {
        "update_lag_days",
        "business_update_frequency_compliance",
    }
)
_CONTROL_OR_SURROGATE_PATTERN = re.compile(
    r"[\x00-\x1f\x7f\ud800-\udfff]"
)
_RISK_LEVEL_ORDER = {"info": 0, "attention": 1, "warning": 2}
_SCHEMA_KIND_ORDER = {
    "field_added": 0,
    "field_removed": 1,
    "field_type_changed": 2,
    "field_order_changed": 3,
    "row_count_changed": 4,
    "column_count_changed": 5,
}
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "report-comparison.schema.json"
)


class ReportComparisonError(ValueError):
    """报告不是固定报告、系列确认无效或比较结果未通过契约。"""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8", errors="strict")


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(
        json.dumps(
            list(parts),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    ).hexdigest()
    return f"{prefix}-{digest[:24]}"


def _canonical_strings(values: list[str]) -> tuple[str, ...]:
    """把无顺序语义的字符串集合固定为唯一、字典序表示。"""

    return tuple(sorted(set(values)))


def _series_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ReportComparisonError("治理对象标识必须是文本。")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 120
        or _CONTROL_OR_SURROGATE_PATTERN.search(normalized)
    ):
        raise ReportComparisonError(
            "治理对象标识必须为 1 到 120 个不含控制字符的 Unicode 字符。"
        )
    return normalized


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _report_reference(payload: Mapping[str, Any]) -> ReportReference:
    context = payload["evaluation_context"]
    profile = payload.get("profile", {})
    dataset = payload["dataset"]
    return ReportReference(
        report_sha256=context["report_sha256"],
        input_sha256=context.get("input_sha256"),
        report_schema_version=str(payload["schema_version"]),
        engine_version=str(context["engine_version"]),
        threshold_config_version=str(context["threshold_config_version"]),
        reference_date=context.get("reference_date"),
        parser_path=context.get("parser_path"),
        dataset_name=str(dataset["name"]),
        file_name=str(dataset["file_name"]),
        status=str(payload["status"]),
        row_count=_non_negative_int(profile.get("row_count")),
        column_count=_non_negative_int(profile.get("column_count")),
    )


def _metric_direction(metric_id: str) -> str:
    if metric_id in _HIGHER_IS_BETTER:
        return "higher_is_better"
    if metric_id in _LOWER_IS_BETTER:
        return "lower_is_better"
    return "neutral"


def _metric_rule_signature(
    metric: Mapping[str, Any],
) -> tuple[str, str] | None:
    evidence = metric.get("evidence")
    if not isinstance(evidence, Mapping):
        return None
    for key in ("rule_definition_sha256", "rule_pack_sha256"):
        value = evidence.get(key)
        if isinstance(value, str) and value:
            return key, value
    return None


def _number_delta(before: Any, after: Any) -> int | float:
    delta = Decimal(str(after)) - Decimal(str(before))
    if delta == delta.to_integral_value():
        return int(delta)
    return float(delta)


def _metric_changes(
    baseline: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    engine_compatible: bool,
    reference_date_compatible: bool,
) -> tuple[MetricChange, ...]:
    baseline_metrics = {
        metric["metric_key"]: metric for metric in baseline["metrics"]
    }
    target_metrics = {
        metric["metric_key"]: metric for metric in target["metrics"]
    }
    changes: list[MetricChange] = []
    for metric_key in sorted(set(baseline_metrics) | set(target_metrics)):
        before = baseline_metrics.get(metric_key)
        after = target_metrics.get(metric_key)
        source = after or before
        assert source is not None
        metric_id = str(source["id"])
        direction = _metric_direction(metric_id)
        reason_codes: list[str] = []
        baseline_status = before.get("status") if before else None
        target_status = after.get("status") if after else None
        baseline_value = before.get("value") if before else None
        target_value = after.get("value") if after else None
        unit = (
            after.get("unit")
            if after is not None
            else before.get("unit") if before is not None else None
        )
        delta: int | float | None = None

        if before is None:
            classification = "added"
        elif after is None:
            classification = "removed"
        elif not engine_compatible:
            classification = "not_comparable"
            reason_codes.append("engine_or_report_schema_changed")
        elif (
            before.get("id") != after.get("id")
            or before.get("scope") != after.get("scope")
            or (
                baseline_status == target_status == "evaluated"
                and before.get("unit") != after.get("unit")
            )
        ):
            classification = "not_comparable"
            reason_codes.append("metric_definition_changed")
        elif metric_id.startswith("business_") and (
            _metric_rule_signature(before) is None
            or _metric_rule_signature(before) != _metric_rule_signature(after)
        ):
            classification = "not_comparable"
            reason_codes.append("business_rule_definition_changed")
        elif (
            not reference_date_compatible
            and metric_id in _REFERENCE_DATE_SENSITIVE_METRICS
        ):
            classification = "not_comparable"
            reason_codes.append("reference_date_changed")
        elif baseline_status == "not_assessable" and target_status == "evaluated":
            classification = "became_assessable"
        elif baseline_status == "evaluated" and target_status == "not_assessable":
            classification = "became_not_assessable"
        elif baseline_status == target_status == "not_assessable":
            classification = "unchanged"
        elif baseline_value == target_value:
            classification = "unchanged"
            if baseline_value is not None and target_value is not None:
                delta = 0
        elif (
            baseline_value is None
            or target_value is None
            or isinstance(baseline_value, bool)
            or isinstance(target_value, bool)
        ):
            classification = "not_comparable"
            reason_codes.append("metric_value_missing")
        else:
            delta = _number_delta(baseline_value, target_value)
            if metric_id == "update_lag_days" and (
                Decimal(str(baseline_value)) < 0
                or Decimal(str(target_value)) < 0
            ):
                # 负滞后表示未来日期，数值越小并不代表更好；该场景由
                # 对应风险的新增/解除表达，本指标只记录变化。
                classification = "changed"
                reason_codes.append("future_date_requires_risk_context")
            elif direction == "neutral":
                classification = "changed"
            elif (
                direction == "higher_is_better" and delta > 0
            ) or (
                direction == "lower_is_better" and delta < 0
            ):
                classification = "improved"
            else:
                classification = "worsened"

        changes.append(
            MetricChange(
                change_id=_stable_id("metric-change", metric_key),
                metric_key=metric_key,
                metric_id=metric_id,
                name=str(source.get("name", metric_id)),
                field=source.get("field"),
                direction=direction,  # type: ignore[arg-type]
                classification=classification,  # type: ignore[arg-type]
                baseline_status=baseline_status,
                target_status=target_status,
                baseline_value=baseline_value,
                target_value=target_value,
                delta=delta,
                unit=unit,
                reason_codes=_canonical_strings(reason_codes),
            )
        )
    return tuple(changes)


def _all_related_metrics_evaluated(
    risk: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
) -> bool:
    related = risk.get("related_metric_keys", [])
    return bool(related) and all(
        key in metrics and metrics[key].get("status") == "evaluated"
        for key in related
    )


def _risk_rule_compatible(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> bool:
    if before is None or after is None:
        return True
    before_decision = before.get("evidence", {}).get("decision", {})
    after_decision = after.get("evidence", {}).get("decision", {})
    before_rule_hash = before.get("evidence", {}).get(
        "rule_definition_sha256"
    )
    after_rule_hash = after.get("evidence", {}).get(
        "rule_definition_sha256"
    )
    if before_rule_hash is not None or after_rule_hash is not None:
        definition_fields = (
            "rule_id",
            "observed_name",
            "operator",
            "threshold",
        )
        definition_hash_compatible = (
            isinstance(before_rule_hash, str)
            and before_rule_hash == after_rule_hash
        )
    else:
        definition_fields = (
            "rule_id",
            "rule_version",
            "threshold_config_version",
            "observed_name",
            "operator",
            "threshold",
        )
        definition_hash_compatible = True
    return (
        set(before.get("related_metric_keys", []))
        == set(after.get("related_metric_keys", []))
        and definition_hash_compatible
        and all(
            before_decision.get(field) == after_decision.get(field)
            for field in definition_fields
        )
    )


def _risk_uses_reference_date(
    risk: Mapping[str, Any],
    baseline_metrics: Mapping[str, Mapping[str, Any]],
    target_metrics: Mapping[str, Mapping[str, Any]],
) -> bool:
    for metric_key in risk.get("related_metric_keys", []):
        metric = target_metrics.get(metric_key) or baseline_metrics.get(
            metric_key
        )
        if (
            metric is not None
            and str(metric.get("id")) in _REFERENCE_DATE_SENSITIVE_METRICS
        ):
            return True
    return False


def _business_risk_metrics_compatible(
    risk: Mapping[str, Any],
    baseline_metrics: Mapping[str, Mapping[str, Any]],
    target_metrics: Mapping[str, Mapping[str, Any]],
) -> bool:
    related = risk.get("related_metric_keys", [])
    for metric_key in related:
        before = baseline_metrics.get(metric_key)
        after = target_metrics.get(metric_key)
        metric_id = str((after or before or {}).get("id", ""))
        if not metric_id.startswith("business_"):
            continue
        if (
            before is None
            or after is None
            or _metric_rule_signature(before) is None
            or _metric_rule_signature(before) != _metric_rule_signature(after)
        ):
            return False
    return True


def _risk_changes(
    baseline: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    risk_context_compatible: bool,
    reference_date_compatible: bool,
) -> tuple[RiskChange, ...]:
    baseline_risks = {risk["id"]: risk for risk in baseline["risks"]}
    target_risks = {risk["id"]: risk for risk in target["risks"]}
    baseline_metrics = {
        metric["metric_key"]: metric for metric in baseline["metrics"]
    }
    target_metrics = {
        metric["metric_key"]: metric for metric in target["metrics"]
    }
    changes: list[RiskChange] = []
    for risk_id in sorted(set(baseline_risks) | set(target_risks)):
        before = baseline_risks.get(risk_id)
        after = target_risks.get(risk_id)
        source = after or before
        assert source is not None
        related = tuple(
            sorted(
                set(
                    (before or {}).get("related_metric_keys", [])
                    + (after or {}).get("related_metric_keys", [])
                )
            )
        )
        reason_codes: list[str] = []
        compatible = True
        if not risk_context_compatible:
            compatible = False
            reason_codes.append("risk_rule_or_threshold_changed")
        if (
            not reference_date_compatible
            and _risk_uses_reference_date(
                source,
                baseline_metrics,
                target_metrics,
            )
        ):
            compatible = False
            reason_codes.append("reference_date_changed")
        if not _risk_rule_compatible(before, after):
            compatible = False
            reason_codes.append("risk_rule_or_threshold_changed")
        if not _business_risk_metrics_compatible(
            source,
            baseline_metrics,
            target_metrics,
        ):
            compatible = False
            reason_codes.append("business_rule_definition_changed")
        if not compatible:
            classification = "not_comparable"
        elif before is None:
            if not _all_related_metrics_evaluated(after, baseline_metrics):
                classification = "not_comparable"
                reason_codes.append("related_metric_added_or_not_assessable")
            else:
                classification = "added"
        elif after is None:
            if not _all_related_metrics_evaluated(before, target_metrics):
                classification = "not_comparable"
                reason_codes.append("related_metric_removed_or_not_assessable")
            else:
                classification = "resolved"
        else:
            before_level = _RISK_LEVEL_ORDER[str(before["level"])]
            after_level = _RISK_LEVEL_ORDER[str(after["level"])]
            if after_level > before_level:
                classification = "severity_increased"
            elif after_level < before_level:
                classification = "severity_decreased"
            else:
                classification = "persistent"

        changes.append(
            RiskChange(
                change_id=_stable_id("risk-change", risk_id),
                risk_id=risk_id,
                title=str(source.get("title", risk_id)),
                classification=classification,  # type: ignore[arg-type]
                baseline_level=before.get("level") if before else None,
                target_level=after.get("level") if after else None,
                related_metric_keys=related,
                reason_codes=_canonical_strings(reason_codes),
            )
        )
    return tuple(changes)


def _assessability_changes(
    baseline: Mapping[str, Any],
    target: Mapping[str, Any],
) -> tuple[AssessabilityChange, ...]:
    baseline_metrics = {
        metric["metric_key"]: metric for metric in baseline["metrics"]
    }
    target_metrics = {
        metric["metric_key"]: metric for metric in target["metrics"]
    }
    baseline_items = {
        item["metric_key"]: item for item in baseline["not_assessable"]
    }
    target_items = {
        item["metric_key"]: item for item in target["not_assessable"]
    }
    changes: list[AssessabilityChange] = []
    for metric_key in sorted(set(baseline_items) | set(target_items)):
        before_item = baseline_items.get(metric_key)
        after_item = target_items.get(metric_key)
        before_metric = baseline_metrics.get(metric_key)
        after_metric = target_metrics.get(metric_key)
        source = after_item or before_item
        assert source is not None
        if before_item is not None and after_item is not None:
            classification = (
                "persistent"
                if before_item.get("reason") == after_item.get("reason")
                else "reason_changed"
            )
        elif before_item is not None:
            classification = (
                "became_assessable"
                if after_metric is not None
                and after_metric.get("status") == "evaluated"
                else "removed_with_metric"
            )
        else:
            classification = (
                "became_not_assessable"
                if before_metric is not None
                and before_metric.get("status") == "evaluated"
                else "added_with_metric"
            )
        changes.append(
            AssessabilityChange(
                change_id=_stable_id("assessability-change", metric_key),
                metric_key=metric_key,
                name=str(source.get("name", metric_key)),
                classification=classification,  # type: ignore[arg-type]
                baseline_reason=(
                    str(before_item["reason"]) if before_item else None
                ),
                target_reason=(
                    str(after_item["reason"]) if after_item else None
                ),
            )
        )
    return tuple(changes)


def _profile_columns(payload: Mapping[str, Any]) -> list[tuple[str, str]]:
    raw_columns = payload.get("profile", {}).get("columns", [])
    columns: list[tuple[str, str]] = []
    for raw in raw_columns if isinstance(raw_columns, list) else []:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("name"), str):
            continue
        columns.append((raw["name"], str(raw.get("inferred_type", "unknown"))))
    names = [name for name, _ in columns]
    if len(names) != len(set(names)):
        raise ReportComparisonError("报告字段画像包含重复字段名，不能比较。")
    return columns


def _schema_changes(
    baseline: Mapping[str, Any],
    target: Mapping[str, Any],
    before_ref: ReportReference,
    after_ref: ReportReference,
) -> tuple[SchemaChange, ...]:
    before_columns = _profile_columns(baseline)
    after_columns = _profile_columns(target)
    before_types = dict(before_columns)
    after_types = dict(after_columns)
    changes: list[SchemaChange] = []
    for field in sorted(
        name for name, _ in after_columns if name not in before_types
    ):
        changes.append(
            SchemaChange(
                change_id=_stable_id("schema-change", "field_added", field),
                kind="field_added",
                field=field,
                baseline_value=None,
                target_value=after_types[field],
            )
        )
    for field in sorted(
        name for name, _ in before_columns if name not in after_types
    ):
        changes.append(
            SchemaChange(
                change_id=_stable_id("schema-change", "field_removed", field),
                kind="field_removed",
                field=field,
                baseline_value=before_types[field],
                target_value=None,
            )
        )
    for field in sorted(set(before_types) & set(after_types)):
        if before_types[field] != after_types[field]:
            changes.append(
                SchemaChange(
                    change_id=_stable_id(
                        "schema-change",
                        "field_type_changed",
                        field,
                    ),
                    kind="field_type_changed",
                    field=field,
                    baseline_value=before_types[field],
                    target_value=after_types[field],
                )
            )
    shared = set(before_types) & set(after_types)
    before_order = [name for name, _ in before_columns if name in shared]
    after_order = [name for name, _ in after_columns if name in shared]
    if before_order != after_order:
        changes.append(
            SchemaChange(
                change_id=_stable_id("schema-change", "field_order_changed"),
                kind="field_order_changed",
                field=None,
                baseline_value=before_order,
                target_value=after_order,
            )
        )
    if before_ref.row_count != after_ref.row_count:
        changes.append(
            SchemaChange(
                change_id=_stable_id("schema-change", "row_count_changed"),
                kind="row_count_changed",
                field=None,
                baseline_value=before_ref.row_count,
                target_value=after_ref.row_count,
            )
        )
    if before_ref.column_count != after_ref.column_count:
        changes.append(
            SchemaChange(
                change_id=_stable_id("schema-change", "column_count_changed"),
                kind="column_count_changed",
                field=None,
                baseline_value=before_ref.column_count,
                target_value=after_ref.column_count,
            )
        )
    return tuple(changes)


def _summary(
    metric_changes: tuple[MetricChange, ...],
    risk_changes: tuple[RiskChange, ...],
    assessability_changes: tuple[AssessabilityChange, ...],
    schema_changes: tuple[SchemaChange, ...],
) -> dict[str, int]:
    metric_classes = [change.classification for change in metric_changes]
    risk_classes = [change.classification for change in risk_changes]
    assessability_classes = [
        change.classification for change in assessability_changes
    ]
    return {
        "metric_change_count": len(metric_changes),
        "improved_metric_count": metric_classes.count("improved"),
        "worsened_metric_count": metric_classes.count("worsened"),
        "changed_metric_count": metric_classes.count("changed"),
        "unchanged_metric_count": metric_classes.count("unchanged"),
        "added_metric_count": metric_classes.count("added"),
        "removed_metric_count": metric_classes.count("removed"),
        "not_comparable_metric_count": metric_classes.count("not_comparable"),
        "became_assessable_metric_count": metric_classes.count(
            "became_assessable"
        ),
        "became_not_assessable_metric_count": metric_classes.count(
            "became_not_assessable"
        ),
        "risk_change_count": len(risk_changes),
        "added_risk_count": risk_classes.count("added"),
        "resolved_risk_count": risk_classes.count("resolved"),
        "persistent_risk_count": risk_classes.count("persistent"),
        "increased_risk_count": risk_classes.count("severity_increased"),
        "decreased_risk_count": risk_classes.count("severity_decreased"),
        "not_comparable_risk_count": risk_classes.count("not_comparable"),
        "assessability_change_count": len(assessability_changes),
        "became_assessable_count": assessability_classes.count(
            "became_assessable"
        ),
        "became_not_assessable_count": assessability_classes.count(
            "became_not_assessable"
        ),
        "schema_change_count": len(schema_changes),
    }


def _comparison_hash_payload(comparison: ReportComparison) -> dict[str, Any]:
    payload = comparison.to_dict()
    payload.pop("comparison_sha256", None)
    return payload


def compare_reports(
    baseline_report: QualityReport | Mapping[str, Any],
    target_report: QualityReport | Mapping[str, Any],
    *,
    dataset_series_id: str,
    same_series_confirmed: bool,
) -> ReportComparison:
    """比较两份固定报告；系列确认必须由调用方显式提供。"""

    if same_series_confirmed is not True:
        raise ReportComparisonError(
            "必须明确确认两份固定报告属于同一治理对象后才能比较。"
        )
    series_id = _series_id(dataset_series_id)
    try:
        baseline = validate_quality_report_payload(baseline_report)
        target = validate_quality_report_payload(target_report)
    except HistoryValidationError as error:
        raise ReportComparisonError(str(error)) from error

    before_ref = _report_reference(baseline)
    after_ref = _report_reference(target)
    engine_compatible = (
        before_ref.report_schema_version == after_ref.report_schema_version
        and before_ref.engine_version == after_ref.engine_version
    )
    risk_context_compatible = (
        engine_compatible
        and before_ref.threshold_config_version
        == after_ref.threshold_config_version
    )
    reference_date_compatible = (
        before_ref.reference_date == after_ref.reference_date
    )

    reason_codes: list[str] = []
    context_change_codes: list[str] = []
    limitations: list[str] = []
    if before_ref.report_schema_version != after_ref.report_schema_version:
        reason_codes.append("report_schema_version_changed")
        limitations.append(
            "两份报告的 QualityReport Schema 版本不同，指标与风险改善结论受限。"
        )
    if before_ref.engine_version != after_ref.engine_version:
        reason_codes.append("engine_version_changed")
        limitations.append(
            "两份报告的质量指标引擎版本不同，仅描述变化，不判断指标改善或恶化。"
        )
    if (
        before_ref.threshold_config_version
        != after_ref.threshold_config_version
    ):
        reason_codes.append("threshold_config_version_changed")
        limitations.append(
            "两份报告的风险阈值配置版本不同，风险新增、解除和等级变化不可比。"
        )
    if before_ref.status == "failed":
        reason_codes.append("baseline_report_failed")
        limitations.append("整改前报告评估失败，比较覆盖范围受限。")
    if after_ref.status == "failed":
        reason_codes.append("target_report_failed")
        limitations.append("整改后报告评估失败，比较覆盖范围受限。")
    if before_ref.dataset_name != after_ref.dataset_name:
        context_change_codes.append("dataset_name_changed")
        limitations.append(
            "两份报告的数据集名称不同；同一治理对象关系来自本次用户显式确认。"
        )
    if not reference_date_compatible:
        context_change_codes.append("reference_date_changed")
        reason_codes.append("reference_date_changed_for_time_metrics")
        limitations.append(
            "两份报告的评估基准日期不同，更新滞后和更新频率相关指标与风险不可比。"
        )
    if before_ref.parser_path != after_ref.parser_path:
        context_change_codes.append("parser_path_changed")

    metric_changes = _metric_changes(
        baseline,
        target,
        engine_compatible=engine_compatible,
        reference_date_compatible=reference_date_compatible,
    )
    if any(
        "business_rule_definition_changed" in change.reason_codes
        for change in metric_changes
    ):
        reason_codes.append("business_rule_definition_changed")
        limitations.append(
            "至少一项业务规则指标未证明使用相同规则定义，不能判断其改善或恶化。"
        )
    if any(
        "metric_definition_changed" in change.reason_codes
        for change in metric_changes
    ):
        reason_codes.append("metric_definition_changed")
        limitations.append(
            "至少一项指标定义发生变化，不能判断其改善或恶化。"
        )
    risk_changes = _risk_changes(
        baseline,
        target,
        risk_context_compatible=risk_context_compatible,
        reference_date_compatible=reference_date_compatible,
    )
    risk_reason_codes = {
        reason
        for change in risk_changes
        for reason in change.reason_codes
    }
    if "risk_rule_or_threshold_changed" in risk_reason_codes:
        reason_codes.append("risk_rule_definition_changed")
        limitations.append(
            "至少一项风险的规则定义、阈值或关联指标发生变化，不能判断其新增、解除或等级变化。"
        )
    if "business_rule_definition_changed" in risk_reason_codes:
        reason_codes.append("business_rule_definition_changed")
        limitations.append(
            "至少一项业务规则风险未证明使用相同规则定义，不能判断其状态变化。"
        )
    assessability_changes = _assessability_changes(baseline, target)
    schema_changes = _schema_changes(
        baseline,
        target,
        before_ref,
        after_ref,
    )

    ordered_hashes = [
        before_ref.report_sha256,
        after_ref.report_sha256,
    ]
    confirmation_hash = hashlib.sha256(
        _canonical_bytes(
            {
                "dataset_series_id": series_id,
                "ordered_report_sha256": ordered_hashes,
                "same_series_confirmed": True,
            }
        )
    ).hexdigest()
    comparison_id = _stable_id(
        "comparison",
        COMPARATOR_VERSION,
        COMPARISON_POLICY_VERSION,
        series_id,
        *ordered_hashes,
    )
    compatibility_status = "limited" if reason_codes else "full"
    provisional = ReportComparison(
        comparison_id=comparison_id,
        comparison_sha256="0" * 64,
        baseline=before_ref,
        target=after_ref,
        lineage={
            "dataset_series_id": series_id,
            "same_series_confirmed": True,
            "ordered_report_sha256": ordered_hashes,
            "confirmation_sha256": confirmation_hash,
        },
        compatibility={
            "status": compatibility_status,
            "reason_codes": list(_canonical_strings(reason_codes)),
            "context_change_codes": list(
                _canonical_strings(context_change_codes)
            ),
        },
        summary=_summary(
            metric_changes,
            risk_changes,
            assessability_changes,
            schema_changes,
        ),
        metric_changes=metric_changes,
        risk_changes=risk_changes,
        assessability_changes=assessability_changes,
        schema_changes=schema_changes,
        limitations=_canonical_strings(limitations),
    )
    comparison_sha256 = hashlib.sha256(
        _canonical_bytes(_comparison_hash_payload(provisional))
    ).hexdigest()
    comparison = ReportComparison(
        **{
            **provisional.__dict__,
            "comparison_sha256": comparison_sha256,
        }
    )
    validate_report_comparison(comparison)
    return comparison


@lru_cache(maxsize=1)
def _comparison_validator() -> Draft202012Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_report_comparison(
    comparison: ReportComparison | Mapping[str, Any],
) -> dict[str, Any]:
    """复核比较 Schema 与自身哈希，供展示和整改计划复用。"""

    payload = (
        comparison.to_dict()
        if isinstance(comparison, ReportComparison)
        else deepcopy(dict(comparison))
    )
    errors = sorted(
        _comparison_validator().iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        first = errors[0]
        path = ".".join(str(part) for part in first.absolute_path) or "root"
        raise ReportComparisonError(
            f"ReportComparison Schema 校验失败（{path}）：{first.message}"
        )
    reported_hash = payload["comparison_sha256"]
    hash_payload = deepcopy(payload)
    hash_payload.pop("comparison_sha256")
    actual_hash = hashlib.sha256(_canonical_bytes(hash_payload)).hexdigest()
    if reported_hash != actual_hash:
        raise ReportComparisonError("ReportComparison 自身哈希校验失败。")

    lineage = payload["lineage"]
    series_id = _series_id(lineage["dataset_series_id"])
    ordered_hashes = [
        payload["baseline"]["report_sha256"],
        payload["target"]["report_sha256"],
    ]
    if lineage["ordered_report_sha256"] != ordered_hashes:
        raise ReportComparisonError(
            "ReportComparison 的报告顺序与前后报告哈希不一致。"
        )
    expected_confirmation_hash = hashlib.sha256(
        _canonical_bytes(
            {
                "dataset_series_id": series_id,
                "ordered_report_sha256": ordered_hashes,
                "same_series_confirmed": True,
            }
        )
    ).hexdigest()
    if lineage["confirmation_sha256"] != expected_confirmation_hash:
        raise ReportComparisonError(
            "ReportComparison 的同一治理对象确认绑定无效。"
        )
    expected_comparison_id = _stable_id(
        "comparison",
        COMPARATOR_VERSION,
        COMPARISON_POLICY_VERSION,
        series_id,
        *ordered_hashes,
    )
    if payload["comparison_id"] != expected_comparison_id:
        raise ReportComparisonError(
            "ReportComparison ID 与治理对象及两份报告不一致。"
        )

    def require_canonical_strings(
        values: list[str],
        label: str,
    ) -> None:
        if values != sorted(set(values)):
            raise ReportComparisonError(
                f"ReportComparison 的{label}未按唯一字典序保存。"
            )

    metric_changes = payload["metric_changes"]
    metric_keys = [item["metric_key"] for item in metric_changes]
    if len(metric_keys) != len(set(metric_keys)):
        raise ReportComparisonError("ReportComparison 包含重复指标变化。")
    if metric_keys != sorted(metric_keys):
        raise ReportComparisonError(
            "ReportComparison 指标变化未按固定键排序。"
        )
    for item in metric_changes:
        require_canonical_strings(
            item["reason_codes"],
            "指标原因代码",
        )
        if item["change_id"] != _stable_id(
            "metric-change",
            item["metric_key"],
        ):
            raise ReportComparisonError(
                "ReportComparison 包含无效指标变化 ID。"
            )
        if item["direction"] != _metric_direction(item["metric_id"]):
            raise ReportComparisonError(
                "ReportComparison 指标方向与登记口径不一致。"
            )
        classification = item["classification"]
        before_status = item["baseline_status"]
        after_status = item["target_status"]
        before_value = item["baseline_value"]
        after_value = item["target_value"]
        delta = item["delta"]
        if (
            (before_status != "evaluated" and before_value is not None)
            or (after_status != "evaluated" and after_value is not None)
        ):
            raise ReportComparisonError(
                "ReportComparison 指标状态与数值不一致。"
            )
        if classification == "added":
            valid_shape = before_status is None and after_status is not None
        elif classification == "removed":
            valid_shape = before_status is not None and after_status is None
        elif classification == "became_assessable":
            valid_shape = (
                before_status == "not_assessable"
                and after_status == "evaluated"
            )
        elif classification == "became_not_assessable":
            valid_shape = (
                before_status == "evaluated"
                and after_status == "not_assessable"
            )
        elif classification == "unchanged":
            valid_shape = (
                before_status == after_status == "not_assessable"
                and before_value is None
                and after_value is None
                and delta is None
            ) or (
                before_status == after_status == "evaluated"
                and before_value == after_value
                and delta == 0
            )
        elif classification in {"improved", "worsened", "changed"}:
            valid_shape = (
                before_status == after_status == "evaluated"
                and before_value is not None
                and after_value is not None
                and before_value != after_value
                and delta == _number_delta(before_value, after_value)
            )
            if classification in {"improved", "worsened"}:
                if item["direction"] == "neutral":
                    valid_shape = False
                else:
                    is_improvement = (
                        item["direction"] == "higher_is_better"
                        and delta is not None
                        and delta > 0
                    ) or (
                        item["direction"] == "lower_is_better"
                        and delta is not None
                        and delta < 0
                    )
                    valid_shape = valid_shape and (
                        is_improvement
                        if classification == "improved"
                        else not is_improvement
                    )
            elif (
                item["direction"] != "neutral"
                and "future_date_requires_risk_context"
                not in item["reason_codes"]
            ):
                valid_shape = False
        else:
            valid_shape = (
                classification == "not_comparable"
                and before_status is not None
                and after_status is not None
                and delta is None
                and bool(item["reason_codes"])
            )
        if not valid_shape:
            raise ReportComparisonError(
                "ReportComparison 指标分类、方向、状态或差值不一致。"
            )
        if classification in {
            "added",
            "removed",
            "became_assessable",
            "became_not_assessable",
        } and delta is not None:
            raise ReportComparisonError(
                "ReportComparison 指标状态变化不得携带数值差。"
            )

    risk_changes = payload["risk_changes"]
    risk_ids = [item["risk_id"] for item in risk_changes]
    if len(risk_ids) != len(set(risk_ids)):
        raise ReportComparisonError("ReportComparison 包含重复风险变化。")
    if risk_ids != sorted(risk_ids):
        raise ReportComparisonError(
            "ReportComparison 风险变化未按固定键排序。"
        )
    metric_key_set = set(metric_keys)
    for item in risk_changes:
        require_canonical_strings(
            item["related_metric_keys"],
            "风险关联指标键",
        )
        require_canonical_strings(
            item["reason_codes"],
            "风险原因代码",
        )
        if item["change_id"] != _stable_id(
            "risk-change",
            item["risk_id"],
        ):
            raise ReportComparisonError(
                "ReportComparison 包含无效风险变化 ID。"
            )
        if any(
            key not in metric_key_set
            for key in item["related_metric_keys"]
        ):
            raise ReportComparisonError(
                "ReportComparison 风险变化引用了不存在的指标变化。"
            )
        classification = item["classification"]
        before_level = item["baseline_level"]
        after_level = item["target_level"]
        if classification == "added":
            valid_shape = before_level is None and after_level is not None
        elif classification == "resolved":
            valid_shape = before_level is not None and after_level is None
        elif classification == "persistent":
            valid_shape = (
                before_level is not None
                and before_level == after_level
            )
        elif classification == "severity_increased":
            valid_shape = (
                before_level is not None
                and after_level is not None
                and _RISK_LEVEL_ORDER[after_level]
                > _RISK_LEVEL_ORDER[before_level]
            )
        elif classification == "severity_decreased":
            valid_shape = (
                before_level is not None
                and after_level is not None
                and _RISK_LEVEL_ORDER[after_level]
                < _RISK_LEVEL_ORDER[before_level]
            )
        else:
            valid_shape = (
                classification == "not_comparable"
                and bool(item["reason_codes"])
            )
        if not valid_shape:
            raise ReportComparisonError(
                "ReportComparison 风险分类与前后等级不一致。"
            )

    assessability_changes = payload["assessability_changes"]
    assessability_keys = [
        item["metric_key"] for item in assessability_changes
    ]
    if len(assessability_keys) != len(set(assessability_keys)):
        raise ReportComparisonError(
            "ReportComparison 包含重复可评估性变化。"
        )
    if assessability_keys != sorted(assessability_keys):
        raise ReportComparisonError(
            "ReportComparison 可评估性变化未按固定键排序。"
        )
    for item in assessability_changes:
        if (
            item["metric_key"] not in metric_key_set
            or item["change_id"]
            != _stable_id(
                "assessability-change",
                item["metric_key"],
            )
        ):
            raise ReportComparisonError(
                "ReportComparison 包含无效可评估性变化引用。"
            )
        metric_change = next(
            change
            for change in metric_changes
            if change["metric_key"] == item["metric_key"]
        )
        before_reason = item["baseline_reason"]
        after_reason = item["target_reason"]
        classification = item["classification"]
        if classification == "became_assessable":
            valid_shape = (
                isinstance(before_reason, str)
                and after_reason is None
                and metric_change["classification"] == "became_assessable"
            )
        elif classification == "became_not_assessable":
            valid_shape = (
                before_reason is None
                and isinstance(after_reason, str)
                and metric_change["classification"]
                == "became_not_assessable"
            )
        elif classification == "persistent":
            valid_shape = (
                isinstance(before_reason, str)
                and before_reason == after_reason
            )
        elif classification == "reason_changed":
            valid_shape = (
                isinstance(before_reason, str)
                and isinstance(after_reason, str)
                and before_reason != after_reason
            )
        elif classification == "added_with_metric":
            valid_shape = (
                before_reason is None
                and isinstance(after_reason, str)
                and metric_change["classification"] == "added"
            )
        else:
            valid_shape = (
                classification == "removed_with_metric"
                and isinstance(before_reason, str)
                and after_reason is None
                and metric_change["classification"] == "removed"
            )
        if not valid_shape:
            raise ReportComparisonError(
                "ReportComparison 可评估性分类与指标变化不一致。"
            )

    schema_changes = payload["schema_changes"]
    schema_change_ids: list[str] = []
    field_kinds = {
        "field_added",
        "field_removed",
        "field_type_changed",
    }
    for item in schema_changes:
        kind = item["kind"]
        field = item["field"]
        if (kind in field_kinds) != (field is not None):
            raise ReportComparisonError(
                "ReportComparison 字段结构变化缺少或错误携带字段名。"
            )
        expected_id = (
            _stable_id("schema-change", kind, field)
            if field is not None
            else _stable_id("schema-change", kind)
        )
        if item["change_id"] != expected_id:
            raise ReportComparisonError(
                "ReportComparison 包含无效结构变化 ID。"
            )
        before_value = item["baseline_value"]
        after_value = item["target_value"]
        if kind == "field_added":
            valid_shape = before_value is None and isinstance(
                after_value,
                str,
            )
        elif kind == "field_removed":
            valid_shape = after_value is None and isinstance(
                before_value,
                str,
            )
        elif kind == "field_type_changed":
            valid_shape = (
                isinstance(before_value, str)
                and isinstance(after_value, str)
                and before_value != after_value
            )
        elif kind == "field_order_changed":
            valid_shape = (
                isinstance(before_value, list)
                and isinstance(after_value, list)
                and before_value != after_value
                and all(
                    isinstance(field_name, str)
                    for field_name in before_value + after_value
                )
            )
        else:
            valid_shape = (
                isinstance(before_value, int)
                and not isinstance(before_value, bool)
                and before_value >= 0
                and isinstance(after_value, int)
                and not isinstance(after_value, bool)
                and after_value >= 0
                and before_value != after_value
            )
        if not valid_shape:
            raise ReportComparisonError(
                "ReportComparison 结构变化类型与前后值不一致。"
            )
        schema_change_ids.append(item["change_id"])
    if len(schema_change_ids) != len(set(schema_change_ids)):
        raise ReportComparisonError("ReportComparison 包含重复结构变化。")
    expected_schema_order = sorted(
        schema_changes,
        key=lambda item: (
            _SCHEMA_KIND_ORDER[item["kind"]],
            item["field"] or "",
        ),
    )
    if schema_changes != expected_schema_order:
        raise ReportComparisonError(
            "ReportComparison 结构变化未按固定键排序。"
        )

    metric_classes = [
        item["classification"] for item in metric_changes
    ]
    risk_classes = [item["classification"] for item in risk_changes]
    assessability_classes = [
        item["classification"] for item in assessability_changes
    ]
    expected_summary = {
        "metric_change_count": len(metric_changes),
        "improved_metric_count": metric_classes.count("improved"),
        "worsened_metric_count": metric_classes.count("worsened"),
        "changed_metric_count": metric_classes.count("changed"),
        "unchanged_metric_count": metric_classes.count("unchanged"),
        "added_metric_count": metric_classes.count("added"),
        "removed_metric_count": metric_classes.count("removed"),
        "not_comparable_metric_count": metric_classes.count(
            "not_comparable"
        ),
        "became_assessable_metric_count": metric_classes.count(
            "became_assessable"
        ),
        "became_not_assessable_metric_count": metric_classes.count(
            "became_not_assessable"
        ),
        "risk_change_count": len(risk_changes),
        "added_risk_count": risk_classes.count("added"),
        "resolved_risk_count": risk_classes.count("resolved"),
        "persistent_risk_count": risk_classes.count("persistent"),
        "increased_risk_count": risk_classes.count(
            "severity_increased"
        ),
        "decreased_risk_count": risk_classes.count(
            "severity_decreased"
        ),
        "not_comparable_risk_count": risk_classes.count(
            "not_comparable"
        ),
        "assessability_change_count": len(assessability_changes),
        "became_assessable_count": assessability_classes.count(
            "became_assessable"
        ),
        "became_not_assessable_count": assessability_classes.count(
            "became_not_assessable"
        ),
        "schema_change_count": len(schema_changes),
    }
    if payload["summary"] != expected_summary:
        raise ReportComparisonError(
            "ReportComparison 摘要与逐项变化不一致。"
        )
    compatibility = payload["compatibility"]
    reason_codes = compatibility["reason_codes"]
    require_canonical_strings(reason_codes, "兼容性原因代码")
    require_canonical_strings(
        compatibility["context_change_codes"],
        "上下文变化代码",
    )
    require_canonical_strings(payload["limitations"], "局限说明")
    expected_status = "limited" if reason_codes else "full"
    if payload["compatibility"]["status"] != expected_status:
        raise ReportComparisonError(
            "ReportComparison 兼容状态与限制原因不一致。"
        )
    return payload


def serialize_report_comparison(comparison: ReportComparison) -> bytes:
    payload = validate_report_comparison(comparison)
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
