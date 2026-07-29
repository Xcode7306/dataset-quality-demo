"""v0.4 已审批业务规则的确定性计算引擎。

本模块只在现有 ``QualityReport`` 之上追加业务规则指标、风险和无法评估项。
零配置指标、零配置风险、默认阈值和输入 DataFrame 均不会被改写。
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import json
import math
from numbers import Integral, Number
from typing import Any, Iterable, Mapping

import pandas as pd

from .models import MetricResult, NotAssessableItem, QualityReport, RiskItem
from .profiler import is_missing_value
from .rule_pack import (
    Rule,
    RulePack,
    draft_sha256,
    is_rule_pack_executable,
    validate_rule_pack,
)


RULE_EVALUATION_SCHEMA_VERSION = "0.4"
MAX_RULE_INSPECTION_CELLS = 2_000_000
MAX_RULE_ISSUE_LOCATIONS = 200_000


class RulePackExecutionError(ValueError):
    """RulePack 未通过当前报告上的执行前复核。"""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(str(error) for error in errors)
        detail = "；".join(self.errors) or "RulePack 当前不可执行。"
        super().__init__(detail)


@dataclass(frozen=True)
class RuleEvaluationDiff:
    """只描述本次 v0.4 规则增强新增的结果，不做跨历史趋势判断。"""

    added_metric_keys: tuple[str, ...]
    added_risk_ids: tuple[str, ...]
    added_not_assessable_metric_keys: tuple[str, ...]
    counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "added_metric_keys": list(self.added_metric_keys),
            "added_risk_ids": list(self.added_risk_ids),
            "added_not_assessable_metric_keys": list(
                self.added_not_assessable_metric_keys
            ),
            "counts": {
                "added_metrics": int(self.counts["added_metrics"]),
                "added_evaluated_metrics": int(
                    self.counts["added_evaluated_metrics"]
                ),
                "added_risks": int(self.counts["added_risks"]),
                "added_not_assessable": int(
                    self.counts["added_not_assessable"]
                ),
                "added_issue_locations": int(
                    self.counts["added_issue_locations"]
                ),
            },
        }


@dataclass(frozen=True)
class RuleEvaluationResult:
    """与基础报告分离、可持久化的 v0.4 规则增强结果。"""

    baseline_report: QualityReport
    enhanced_report: QualityReport
    approved_rule_pack: RulePack
    diff: RuleEvaluationDiff
    schema_version: str = RULE_EVALUATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "baseline_report": self.baseline_report.to_dict(),
            "enhanced_report": self.enhanced_report.to_dict(),
            "approved_rule_pack": self.approved_rule_pack.to_dict(),
            "diff": self.diff.to_dict(),
        }
        # 在协议边界立即拒绝 NaN、Infinity、非字符串键或其他非 JSON 值。
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )


def _record_location(
    record_number: int,
    fields: Iterable[str],
    issue_type: str,
    *,
    related_record_numbers: Iterable[int] = (),
) -> dict[str, Any]:
    location: dict[str, Any] = {
        "record_number": int(record_number),
        "fields": list(dict.fromkeys(str(field) for field in fields)),
        "issue_type": issue_type,
    }
    related = [
        int(number)
        for number in related_record_numbers
        if int(number) > 0 and int(number) != int(record_number)
    ]
    if related:
        location["related_record_numbers"] = list(dict.fromkeys(related))
    return location


def _ensure_issue_location_budget(
    issue_count: int,
    remaining_count: int,
) -> None:
    """在物化位置字典前拒绝超出 v0.4 独立位置预算的规则结果。"""

    if issue_count > remaining_count:
        raise RulePackExecutionError(
            [
                f"业务规则预计生成至少 {issue_count} 条疑似问题位置，"
                f"超过本次剩余 {remaining_count} 条、总计 "
                f"{MAX_RULE_ISSUE_LOCATIONS} 条的安全上限。"
            ]
        )


def _not_assessable(
    metric_id: str,
    name: str,
    *,
    scope: str,
    reason: str,
    field: str | None = None,
) -> MetricResult:
    return MetricResult(
        id=metric_id,
        name=name,
        category="业务规则",
        status="not_assessable",
        value=None,
        unit=None,
        scope=scope,  # type: ignore[arg-type]
        field=field,
        reason=reason,
    )


def _report_snapshot_without_locations(
    report: QualityReport,
) -> QualityReport:
    """复制可持久化报告，但不复制仅供原始基线 CSV 使用的位置明细。"""

    return QualityReport(
        dataset=deepcopy(report.dataset),
        status=report.status,
        schema_version=report.schema_version,
        profile=deepcopy(report.profile),
        metrics=[
            MetricResult(
                id=metric.id,
                name=metric.name,
                category=metric.category,
                status=metric.status,
                value=metric.value,
                unit=metric.unit,
                scope=metric.scope,
                field=metric.field,
                evidence=deepcopy(metric.evidence),
                reason=metric.reason,
            )
            for metric in report.metrics
        ],
        risks=deepcopy(report.risks),
        not_assessable=deepcopy(report.not_assessable),
        evaluation_context=deepcopy(report.evaluation_context),
        execution=deepcopy(report.execution),
    )


def _rule_evidence(
    pack: RulePack,
    rule: Rule,
    **evidence: Any,
) -> dict[str, Any]:
    return {
        "rule_pack_id": pack.rule_pack_id,
        "rule_pack_version": pack.version,
        "rule_pack_sha256": draft_sha256(pack),
        "rule_id": rule.rule_id,
        "rule_type": rule.type,
        "threshold_source": "approved_rule_pack",
        **evidence,
    }


def _decimal_number(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, Integral):
        return Decimal(int(value))
    if not isinstance(value, Number):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal(str(number))


def _parse_numeric_value(value: Any) -> Decimal | None:
    """按表格常见表示解析有限数值；布尔值不视为数值。"""

    number = _decimal_number(value)
    if number is not None:
        return number
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = Decimal(text)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def _comparison_key(value: Any) -> tuple[str, Any]:
    """生成不泄露原值、同时适用于主键和允许值比较的稳定键。"""

    if isinstance(value, bool):
        return ("boolean", value)
    number = _decimal_number(value)
    if number is not None:
        return ("number", number.normalize())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)
        return ("datetime", timestamp.isoformat())
    return ("text", str(value))


def _matches_allowed_value(value: Any, allowed_keys: set[tuple[str, Any]]) -> bool:
    if is_missing_value(value):
        return False
    return _comparison_key(value) in allowed_keys


def _parse_datetime(value: Any) -> pd.Timestamp | None:
    if is_missing_value(value):
        return None
    if isinstance(value, bool) or (
        isinstance(value, Number)
        and not isinstance(value, (pd.Timestamp, datetime, date))
    ):
        # 数值字段不能借 pandas 的纳秒时间解释而被误判为业务日期。
        return None
    try:
        parsed = pd.to_datetime(
            str(value).strip(),
            errors="coerce",
            format="mixed",
        )
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(parsed):
        return None
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp


def _primary_key_metric(
    dataframe: pd.DataFrame,
    pack: RulePack,
    rule: Rule,
    *,
    remaining_issue_locations: int,
) -> MetricResult:
    fields = list(rule.fields)
    row_count = int(len(dataframe))
    if row_count == 0:
        return _not_assessable(
            "business_primary_key_compliance",
            "主键完整且唯一符合率",
            scope="dataset",
            reason="数据集不包含记录，无法评估主键完整性与唯一性。",
        )

    missing_by_record: dict[int, list[str]] = {}
    complete_keys: dict[tuple[tuple[str, Any], ...], list[int]] = defaultdict(list)
    for record_number, values in enumerate(
        dataframe[fields].itertuples(index=False, name=None),
        start=1,
    ):
        missing_fields = [
            field
            for field, value in zip(fields, values)
            if is_missing_value(value)
        ]
        if missing_fields:
            missing_by_record[record_number] = missing_fields
            continue
        complete_keys[
            tuple(_comparison_key(value) for value in values)
        ].append(record_number)

    duplicate_groups = [
        record_numbers
        for record_numbers in complete_keys.values()
        if len(record_numbers) > 1
    ]
    duplicate_records = {
        record_number
        for group in duplicate_groups
        for record_number in group
    }
    _ensure_issue_location_budget(
        len(missing_by_record) + len(duplicate_records),
        remaining_issue_locations,
    )
    locations = [
        _record_location(
            record_number,
            missing_fields,
            "rule_primary_key_missing",
        )
        for record_number, missing_fields in missing_by_record.items()
    ]
    for group in duplicate_groups:
        first_record = group[0]
        for index, record_number in enumerate(group):
            related = (
                [group[1]]
                if index == 0 and len(group) > 1
                else [first_record]
            )
            locations.append(
                _record_location(
                    record_number,
                    fields,
                    "rule_primary_key_duplicate",
                    related_record_numbers=related,
                )
            )
    locations.sort(key=lambda item: (item["record_number"], item["issue_type"]))

    issue_records = set(missing_by_record) | duplicate_records
    compliant_count = row_count - len(issue_records)
    return MetricResult(
        id="business_primary_key_compliance",
        name="主键完整且唯一符合率",
        category="业务规则",
        status="evaluated",
        value=round(compliant_count / row_count, 6),
        unit="ratio",
        scope="dataset",
        evidence=_rule_evidence(
            pack,
            rule,
            fields=fields,
            checked_count=row_count,
            compliant_count=compliant_count,
            issue_count=len(issue_records),
            missing_record_count=len(missing_by_record),
            duplicate_record_count=len(duplicate_records),
            duplicate_group_count=len(duplicate_groups),
        ),
        issue_locations=locations,
    )


def _required_metric(
    dataframe: pd.DataFrame,
    pack: RulePack,
    rule: Rule,
    *,
    remaining_issue_locations: int,
) -> MetricResult:
    field = rule.fields[0]
    row_count = int(len(dataframe))
    if row_count == 0:
        return _not_assessable(
            "business_required_compliance",
            "必填字段完整率",
            scope="field",
            field=field,
            reason="数据集不包含记录，无法评估必填字段完整率。",
        )

    missing_records = [
        record_number
        for record_number, value in enumerate(dataframe[field].tolist(), start=1)
        if is_missing_value(value)
    ]
    _ensure_issue_location_budget(
        len(missing_records),
        remaining_issue_locations,
    )
    compliant_count = row_count - len(missing_records)
    return MetricResult(
        id="business_required_compliance",
        name="必填字段完整率",
        category="业务规则",
        status="evaluated",
        value=round(compliant_count / row_count, 6),
        unit="ratio",
        scope="field",
        field=field,
        evidence=_rule_evidence(
            pack,
            rule,
            checked_count=row_count,
            compliant_count=compliant_count,
            issue_count=len(missing_records),
        ),
        issue_locations=[
            _record_location(
                record_number,
                [field],
                "rule_required_missing",
            )
            for record_number in missing_records
        ],
    )


def _update_metrics(
    dataframe: pd.DataFrame,
    pack: RulePack,
    rule: Rule,
    reference_date: date,
    *,
    remaining_issue_locations: int,
) -> list[MetricResult]:
    field = rule.fields[0]
    row_count = int(len(dataframe))
    if row_count == 0:
        reason = "数据集不包含记录，无法评估更新时间规则。"
        return [
            _not_assessable(
                "business_update_time_parseability",
                "指定更新时间可解析率",
                scope="field",
                field=field,
                reason=reason,
            ),
            _not_assessable(
                "business_update_frequency_compliance",
                "更新频率符合性",
                scope="field",
                field=field,
                reason=reason,
            ),
        ]

    parsed_rows: list[tuple[int, pd.Timestamp]] = []
    invalid_records: list[int] = []
    for record_number, value in enumerate(dataframe[field].tolist(), start=1):
        parsed = _parse_datetime(value)
        if parsed is None:
            invalid_records.append(record_number)
        else:
            parsed_rows.append((record_number, parsed))

    parseable_count = len(parsed_rows)
    _ensure_issue_location_budget(
        len(invalid_records),
        remaining_issue_locations,
    )
    parseability = MetricResult(
        id="business_update_time_parseability",
        name="指定更新时间可解析率",
        category="业务规则",
        status="evaluated",
        value=round(parseable_count / row_count, 6),
        unit="ratio",
        scope="field",
        field=field,
        evidence=_rule_evidence(
            pack,
            rule,
            checked_count=row_count,
            parseable_count=parseable_count,
            issue_count=len(invalid_records),
        ),
        issue_locations=[
            _record_location(
                record_number,
                [field],
                "rule_update_time_missing_or_invalid",
            )
            for record_number in invalid_records
        ],
    )

    if not parsed_rows:
        freshness = _not_assessable(
            "business_update_frequency_compliance",
            "更新频率符合性",
            scope="field",
            field=field,
            reason="指定更新时间字段没有可解析值，无法计算更新滞后与频率符合性。",
        )
        return [parseability, freshness]

    latest = max(timestamp for _, timestamp in parsed_rows).date()
    lag_days = int((reference_date - latest).days)
    max_age_days = int(rule.max_age_days)  # 已由 RulePack 校验器保证存在且合法。
    compliant = 0 <= lag_days <= max_age_days
    freshness = MetricResult(
        id="business_update_frequency_compliance",
        name="更新频率符合性",
        category="业务规则",
        status="evaluated",
        value=1.0 if compliant else 0.0,
        unit="ratio",
        scope="field",
        field=field,
        evidence=_rule_evidence(
            pack,
            rule,
            checked_count=parseable_count,
            compliant_count=1 if compliant else 0,
            issue_count=0 if compliant else 1,
            frequency=rule.frequency,
            max_age_days=max_age_days,
            latest_update_date=latest.isoformat(),
            reference_date=reference_date.isoformat(),
            update_lag_days=lag_days,
            future_date=lag_days < 0,
        ),
    )
    return [parseability, freshness]


def _allowed_values_metric(
    dataframe: pd.DataFrame,
    pack: RulePack,
    rule: Rule,
    *,
    remaining_issue_locations: int,
) -> MetricResult:
    field = rule.fields[0]
    row_count = int(len(dataframe))
    if row_count == 0:
        return _not_assessable(
            "business_allowed_values_compliance",
            "允许值符合率",
            scope="field",
            field=field,
            reason="数据集不包含记录，无法评估允许值符合率。",
        )

    values = dataframe[field].tolist()
    non_missing_rows = [
        (record_number, value)
        for record_number, value in enumerate(values, start=1)
        if not is_missing_value(value)
    ]
    if not non_missing_rows:
        return _not_assessable(
            "business_allowed_values_compliance",
            "允许值符合率",
            scope="field",
            field=field,
            reason="字段没有非缺失值，无法评估允许值符合率。",
        )

    allowed_keys = {_comparison_key(value) for value in rule.allowed_values}
    violation_records = [
        record_number
        for record_number, value in non_missing_rows
        if not _matches_allowed_value(value, allowed_keys)
    ]
    _ensure_issue_location_budget(
        len(violation_records),
        remaining_issue_locations,
    )
    checked_count = len(non_missing_rows)
    compliant_count = checked_count - len(violation_records)
    return MetricResult(
        id="business_allowed_values_compliance",
        name="允许值符合率",
        category="业务规则",
        status="evaluated",
        value=round(compliant_count / checked_count, 6),
        unit="ratio",
        scope="field",
        field=field,
        evidence=_rule_evidence(
            pack,
            rule,
            total_count=row_count,
            checked_count=checked_count,
            excluded_missing_count=row_count - checked_count,
            compliant_count=compliant_count,
            issue_count=len(violation_records),
            allowed_value_count=len(rule.allowed_values),
        ),
        issue_locations=[
            _record_location(
                record_number,
                [field],
                "rule_allowed_value_violation",
            )
            for record_number in violation_records
        ],
    )


def _numeric_range_metric(
    dataframe: pd.DataFrame,
    pack: RulePack,
    rule: Rule,
    *,
    remaining_issue_locations: int,
) -> MetricResult:
    field = rule.fields[0]
    row_count = int(len(dataframe))
    if row_count == 0:
        return _not_assessable(
            "business_numeric_range_compliance",
            "闭区间数值范围符合率",
            scope="field",
            field=field,
            reason="数据集不包含记录，无法评估数值范围符合率。",
        )

    values = dataframe[field].tolist()
    non_missing_rows = [
        (record_number, value)
        for record_number, value in enumerate(values, start=1)
        if not is_missing_value(value)
    ]
    if not non_missing_rows:
        return _not_assessable(
            "business_numeric_range_compliance",
            "闭区间数值范围符合率",
            scope="field",
            field=field,
            reason="字段没有非缺失值，无法评估数值范围符合率。",
        )

    minimum = (
        Decimal(str(rule.minimum))
        if rule.minimum is not None
        else None
    )
    maximum = (
        Decimal(str(rule.maximum))
        if rule.maximum is not None
        else None
    )
    violation_records: list[int] = []
    numeric_count = 0
    for record_number, value in non_missing_rows:
        number = _parse_numeric_value(value)
        if number is None:
            violation_records.append(record_number)
            continue
        numeric_count += 1
        if (
            (minimum is not None and number < minimum)
            or (maximum is not None and number > maximum)
        ):
            violation_records.append(record_number)

    _ensure_issue_location_budget(
        len(violation_records),
        remaining_issue_locations,
    )
    checked_count = len(non_missing_rows)
    compliant_count = checked_count - len(violation_records)
    return MetricResult(
        id="business_numeric_range_compliance",
        name="闭区间数值范围符合率",
        category="业务规则",
        status="evaluated",
        value=round(compliant_count / checked_count, 6),
        unit="ratio",
        scope="field",
        field=field,
        evidence=_rule_evidence(
            pack,
            rule,
            total_count=row_count,
            checked_count=checked_count,
            excluded_missing_count=row_count - checked_count,
            numeric_count=numeric_count,
            compliant_count=compliant_count,
            issue_count=len(violation_records),
            inclusive=True,
            minimum=rule.minimum,
            maximum=rule.maximum,
        ),
        issue_locations=[
            _record_location(
                record_number,
                [field],
                "rule_numeric_range_violation",
            )
            for record_number in violation_records
        ],
    )


def _calculate_business_metrics(
    dataframe: pd.DataFrame,
    baseline_report: QualityReport,
    pack: RulePack,
) -> list[MetricResult]:
    inspection_cells = int(len(dataframe)) * sum(
        len(rule.fields) for rule in pack.rules
    )
    if inspection_cells > MAX_RULE_INSPECTION_CELLS:
        raise RulePackExecutionError(
            [
                f"业务规则预计检查 {inspection_cells} 个字段值，超过 "
                f"{MAX_RULE_INSPECTION_CELLS} 个的独立安全上限。"
            ]
        )

    reference_date_text = baseline_report.evaluation_context.get("reference_date")
    try:
        reference_date = date.fromisoformat(str(reference_date_text))
    except (TypeError, ValueError) as error:
        raise RulePackExecutionError(
            ["基线报告缺少合法的 reference_date，无法执行更新时间规则。"]
        ) from error

    metrics: list[MetricResult] = []
    issue_location_count = 0
    for rule in pack.rules:
        remaining_issue_locations = (
            MAX_RULE_ISSUE_LOCATIONS - issue_location_count
        )
        if rule.type == "primary_key":
            rule_metrics = [
                _primary_key_metric(
                    dataframe,
                    pack,
                    rule,
                    remaining_issue_locations=remaining_issue_locations,
                )
            ]
        elif rule.type == "required":
            rule_metrics = [
                _required_metric(
                    dataframe,
                    pack,
                    rule,
                    remaining_issue_locations=remaining_issue_locations,
                )
            ]
        elif rule.type == "update_freshness":
            rule_metrics = (
                _update_metrics(
                    dataframe,
                    pack,
                    rule,
                    reference_date,
                    remaining_issue_locations=remaining_issue_locations,
                )
            )
        elif rule.type == "allowed_values":
            rule_metrics = [
                _allowed_values_metric(
                    dataframe,
                    pack,
                    rule,
                    remaining_issue_locations=remaining_issue_locations,
                )
            ]
        elif rule.type == "numeric_range":
            rule_metrics = [
                _numeric_range_metric(
                    dataframe,
                    pack,
                    rule,
                    remaining_issue_locations=remaining_issue_locations,
                )
            ]
        else:  # pragma: no cover - RulePack 校验器负责拒绝未知类型。
            raise RulePackExecutionError([f"不支持规则类型：{rule.type}。"])
        new_location_count = sum(
            len(metric.issue_locations)
            for metric in rule_metrics
        )
        _ensure_issue_location_budget(
            new_location_count,
            remaining_issue_locations,
        )
        issue_location_count += new_location_count
        metrics.extend(rule_metrics)

    metric_keys = [metric.metric_key for metric in metrics]
    if len(metric_keys) != len(set(metric_keys)):
        raise RulePackExecutionError(
            ["RulePack 会生成重复业务指标键，已拒绝执行。"]
        )
    return metrics


def _business_risk(
    metric: MetricResult,
    pack: RulePack,
    rule: Rule,
) -> RiskItem | None:
    if metric.status != "evaluated" or metric.value is None:
        return None
    if float(metric.value) >= 1.0:
        return None

    if metric.id == "business_primary_key_compliance":
        title = "主键不完整或不唯一"
        message = "存在不满足已审批主键完整性或唯一性规则的记录，建议复核问题位置。"
    elif metric.id == "business_required_compliance":
        title = "必填字段存在缺失"
        message = f"字段“{metric.field}”存在不满足已审批必填规则的记录。"
    elif metric.id == "business_update_time_parseability":
        title = "指定更新时间存在缺失或不可解析值"
        message = f"字段“{metric.field}”未全部满足已审批的时间可解析规则。"
    elif metric.id == "business_update_frequency_compliance":
        lag_days = int(metric.evidence["update_lag_days"])
        max_age_days = int(metric.evidence["max_age_days"])
        title = "更新时间不符合已审批频率"
        if lag_days < 0:
            message = (
                f"字段“{metric.field}”的最近日期晚于评估基准日期，"
                "应复核未来日期。"
            )
        else:
            message = (
                f"字段“{metric.field}”相对评估基准日期滞后 {lag_days} 天，"
                f"超过已审批的 {max_age_days} 天上限。"
            )
    elif metric.id == "business_allowed_values_compliance":
        title = "字段存在允许值之外的记录"
        message = f"字段“{metric.field}”未全部满足已审批的允许值规则。"
    elif metric.id == "business_numeric_range_compliance":
        title = "字段存在数值范围之外的记录"
        message = f"字段“{metric.field}”未全部满足已审批的闭区间数值规则。"
    else:  # pragma: no cover - 仅由本模块固定指标调用。
        return None

    return RiskItem(
        id=f"business_rule_violation:{rule.rule_id}:{metric.id}",
        level="attention",
        title=title,
        message=message,
        related_metrics=[metric.id],
        related_metric_keys=[metric.metric_key],
        evidence={
            **metric.evidence,
            "decision": {
                "rule_id": rule.rule_id,
                "rule_version": pack.version,
                "threshold_config_version": (
                    f"rule-pack:{pack.rule_pack_id}:{pack.version}"
                ),
                "observed_name": metric.id,
                "observed_value": metric.value,
                "operator": "<",
                "threshold": 1.0,
            },
        },
    )


def _generate_business_risks(
    metrics: list[MetricResult],
    pack: RulePack,
) -> list[RiskItem]:
    rules_by_id = {rule.rule_id: rule for rule in pack.rules}
    risks: list[RiskItem] = []
    for metric in metrics:
        rule_id = str(metric.evidence.get("rule_id", ""))
        rule = rules_by_id.get(rule_id)
        if rule is None:
            continue
        risk = _business_risk(metric, pack, rule)
        if risk is not None:
            risks.append(risk)
    return risks


def _business_not_assessable(
    metrics: Iterable[MetricResult],
) -> list[NotAssessableItem]:
    return [
        NotAssessableItem(
            id=(
                f"{metric.id}:{metric.field}"
                if metric.field
                else metric.id
            ),
            name=(
                f"{metric.name}（{metric.field}）"
                if metric.field
                else metric.name
            ),
            reason=metric.reason or "当前业务规则无法评估。",
            metric_key=metric.metric_key,
        )
        for metric in metrics
        if metric.status == "not_assessable"
    ]


def _validate_for_execution(
    pack: RulePack,
    baseline_report: QualityReport,
) -> None:
    validation = validate_rule_pack(
        pack,
        baseline_report,
        require_approved=True,
    )
    if not validation.valid:
        raise RulePackExecutionError(validation.errors)
    if not is_rule_pack_executable(pack, baseline_report):
        raise RulePackExecutionError(
            ["RulePack 未通过当前报告上的最终可执行性复核。"]
        )


def _validate_dataframe_matches_baseline(
    dataframe: Any,
    baseline_report: QualityReport,
) -> pd.DataFrame:
    """拒绝与已绑定基线的表结构或规模不一致的直接引擎调用。"""

    if not isinstance(dataframe, pd.DataFrame):
        raise RulePackExecutionError(["规则引擎输入必须是 pandas DataFrame。"])
    expected_columns = [
        str(column.get("name"))
        for column in baseline_report.profile.get("columns", [])
        if isinstance(column, Mapping) and isinstance(column.get("name"), str)
    ]
    actual_columns = [str(column) for column in dataframe.columns]
    expected_row_count = baseline_report.profile.get("row_count")
    expected_column_count = baseline_report.profile.get("column_count")
    if actual_columns != expected_columns:
        raise RulePackExecutionError(
            ["规则引擎表格字段与当前绑定基线报告不一致。"]
        )
    if (
        isinstance(expected_row_count, bool)
        or not isinstance(expected_row_count, int)
        or isinstance(expected_column_count, bool)
        or not isinstance(expected_column_count, int)
        or len(dataframe) != expected_row_count
        or len(dataframe.columns) != expected_column_count
    ):
        raise RulePackExecutionError(
            ["规则引擎表格规模与当前绑定基线报告不一致。"]
        )
    return dataframe


def _evaluate_rule_pack_on_verified_dataframe(
    dataframe: pd.DataFrame,
    baseline_report: QualityReport,
    rule_pack: RulePack,
) -> RuleEvaluationResult:
    """在服务层已复核上传字节后，对其重解析表格执行 RulePack。

    这是模块内部入口。公开调用方必须使用 ``rule_service`` 的上传重评
    服务；仅凭同结构 DataFrame 无法证明其内容与 RulePack 的输入哈希一致。
    """

    baseline_snapshot = baseline_report.to_dict()
    _validate_for_execution(rule_pack, baseline_report)
    dataframe = _validate_dataframe_matches_baseline(
        dataframe,
        baseline_report,
    )

    business_metrics = _calculate_business_metrics(
        dataframe,
        baseline_report,
        rule_pack,
    )
    business_risks = _generate_business_risks(
        business_metrics,
        rule_pack,
    )
    business_not_assessable = _business_not_assessable(business_metrics)

    result_baseline_report = _report_snapshot_without_locations(
        baseline_report
    )
    enhanced_report = _report_snapshot_without_locations(baseline_report)
    enhanced_report.metrics.extend(business_metrics)
    enhanced_report.risks.extend(business_risks)
    enhanced_report.not_assessable.extend(business_not_assessable)

    # 防止后续重构意外将增强结果写回基线。
    if baseline_report.to_dict() != baseline_snapshot:
        raise RuntimeError("规则增强过程意外修改了基线 QualityReport。")
    baseline_metric_count = len(baseline_report.metrics)
    baseline_risk_count = len(baseline_report.risks)
    if (
        enhanced_report.metrics[:baseline_metric_count]
        != baseline_report.metrics
        or enhanced_report.risks[:baseline_risk_count]
        != baseline_report.risks
    ):
        raise RuntimeError("规则增强过程改变了既有指标或风险。")

    diff = RuleEvaluationDiff(
        added_metric_keys=tuple(metric.metric_key for metric in business_metrics),
        added_risk_ids=tuple(risk.id for risk in business_risks),
        added_not_assessable_metric_keys=tuple(
            item.metric_key for item in business_not_assessable
        ),
        counts={
            "added_metrics": len(business_metrics),
            "added_evaluated_metrics": sum(
                metric.status == "evaluated"
                for metric in business_metrics
            ),
            "added_risks": len(business_risks),
            "added_not_assessable": len(business_not_assessable),
            "added_issue_locations": sum(
                len(metric.issue_locations)
                for metric in business_metrics
            ),
        },
    )
    result = RuleEvaluationResult(
        baseline_report=result_baseline_report,
        enhanced_report=enhanced_report,
        approved_rule_pack=rule_pack,
        diff=diff,
    )
    # 在返回前执行一次严格序列化检查，避免将不可持久化结果交给展示层。
    result.to_dict()
    return result
