"""面向质量报告的只读 Agent 快照与白名单工具。

本模块采用正向白名单构造上下文：原始文件名、数据集名、执行错误、样本值、
行号列表以及未知证据字段不会进入 Agent 工具结果。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import quote

from .agent_models import AgentCitation, CitationSourceType


TOOL_NAMES: tuple[str, ...] = (
    "get_report_summary",
    "list_priority_risks",
    "get_risk_evidence",
    "get_metric_evidence",
    "list_not_assessable",
)

_SAFE_EVIDENCE_KEYS: frozenset[str] = frozenset(
    {
        "attempted_file_count",
        "successful_file_count",
        "failed_file_count",
        "available_count",
        "file_count",
        "row_count",
        "column_count",
        "checked_count",
        "issue_count",
        "non_finite_count",
        "covered_count",
        "duplicate_group_count",
        "dominant_type",
        "type_counts",
        "expected_format",
        "method",
        "normalization",
        "q1",
        "q3",
        "iqr",
        "lower_bound",
        "upper_bound",
        "earliest_date",
        "latest_date",
        "latest_update_date",
        "reference_date",
        "rule_id",
        "rule_version",
        "observed_value",
        "threshold",
        "operator",
        "threshold_source",
        "additional_duplicate_rate",
        "absolute_lag_days",
        "decision",
    }
)
_BANNED_KEY_PARTS: tuple[str, ...] = (
    "sample",
    "row_index",
    "row_indices",
    "raw_value",
    "record_value",
)
_SAFE_DECISION_KEYS: frozenset[str] = frozenset(
    {
        "observed_value",
        "observed_name",
        "threshold",
        "operator",
        "rule_id",
        "rule_version",
        "threshold_config_version",
        "threshold_source",
    }
)
_SAFE_TYPE_COUNT_KEYS: frozenset[str] = frozenset(
    {"boolean", "datetime", "numeric", "text", "unknown"}
)
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
)
_RISK_LEVEL_ORDER = {"warning": 0, "attention": 1, "info": 2}
_SUMMARY_NUMERIC_LABELS: dict[str, tuple[str, ...]] = {
    "row_count": ("记录数", "数据行数"),
    "column_count": ("字段数", "列数"),
    "metric_definition_count": ("类指标", "指标种类", "指标定义"),
    "evaluated_metric_definition_count": ("已评估指标种类",),
    "metric_result_count": ("指标结果", "指标结果数"),
    "evaluated_metric_result_count": ("已评估指标结果",),
    "risk_count": ("风险数", "风险提示", "风险"),
    "warning_count": ("警告数", "警告"),
    "attention_count": ("关注数", "关注"),
    "info_count": ("提示数", "提示"),
    "not_assessable_count": ("无法评估数", "无法评估"),
}
_METRIC_NUMERIC_LABELS: dict[str, tuple[str, ...]] = {
    "file_parse_rate": ("文件可解析率", "解析率"),
    "dataset_scale": ("数据规模", "记录数"),
    "field_missing_rate": ("字段缺失率", "缺失率"),
    "blank_record_rate": ("空白记录率", "空白记录", "为空"),
    "field_type_consistency": ("字段类型一致率", "类型一致率", "主要类型一致率"),
    "recognizable_format_anomaly_rate": (
        "可识别格式异常率",
        "格式异常率",
    ),
    "exact_duplicate_rate": ("完全重复率", "完全重复", "重复率"),
    "normalized_duplicate_rate": ("规范化重复率", "规范化重复", "重复率"),
    "time_info_availability": ("时间信息可用率", "可解析时间", "时间信息"),
    "update_lag_days": ("更新滞后天数", "更新滞后", "最近更新"),
    "source_info_coverage": ("来源信息覆盖率", "来源覆盖率", "来源信息"),
    "version_info_coverage": ("版本信息覆盖率", "版本覆盖率", "版本信息"),
    "statistical_outlier_rate": ("统计异常值比例", "统计异常率", "统计异常值"),
}
_EVIDENCE_NUMERIC_LABELS: dict[str, tuple[str, ...]] = {
    "attempted_file_count": ("尝试文件数",),
    "successful_file_count": ("成功文件数",),
    "failed_file_count": ("失败文件数",),
    "available_count": ("可用数", "可用数量"),
    "file_count": ("文件数",),
    "row_count": ("记录数", "数据行数"),
    "column_count": ("字段数", "列数"),
    "checked_count": ("检查数", "已检查", "检查记录数"),
    "issue_count": ("问题数", "异常数", "报告统计", "统计异常"),
    "non_finite_count": ("非有限数值数", "非有限数值"),
    "covered_count": ("覆盖数", "已覆盖"),
    "duplicate_group_count": ("重复组数", "重复组"),
    "q1": ("Q1", "下四分位数"),
    "q3": ("Q3", "上四分位数"),
    "iqr": ("IQR", "四分位距"),
    "lower_bound": ("下界",),
    "upper_bound": ("上界",),
    "additional_duplicate_rate": ("额外重复率", "额外", "规范化重复"),
    "absolute_lag_days": ("晚于", "晚", "相差天数"),
}


class AgentToolError(ValueError):
    """工具名、参数或引用目标无效。"""


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


def _report_hash(payload: Mapping[str, Any]) -> str:
    context = payload.get("evaluation_context")
    reported_hash = context.get("report_sha256") if isinstance(context, Mapping) else None
    if isinstance(reported_hash, str) and _HASH_PATTERN.fullmatch(reported_hash):
        return reported_hash

    hash_payload = deepcopy(dict(payload))
    hash_context = hash_payload.get("evaluation_context")
    if isinstance(hash_context, dict):
        hash_context.pop("report_sha256", None)
    canonical = json.dumps(
        _canonical_value(hash_payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)


def _forbidden_strings(payload: Mapping[str, Any]) -> tuple[str, ...]:
    forbidden: list[str] = []
    dataset = payload.get("dataset")
    if isinstance(dataset, Mapping):
        for key in ("name", "file_name"):
            value = dataset.get(key)
            if isinstance(value, str) and len(value.strip()) >= 2:
                forbidden.append(value.strip())
    execution = payload.get("execution")
    if isinstance(execution, Mapping):
        for key in ("errors", "warnings"):
            values = execution.get(key)
            if isinstance(values, list):
                forbidden.extend(
                    value.strip()
                    for value in values
                    if isinstance(value, str) and len(value.strip()) >= 2
                )
    return tuple(sorted(set(forbidden), key=len, reverse=True))


def _redact_text(value: Any, forbidden: tuple[str, ...], maximum: int = 500) -> str:
    text = str(value)
    for sensitive in forbidden:
        text = text.replace(sensitive, "[已隐藏]")
    return text[:maximum]


def _safe_scalar(value: Any, forbidden: tuple[str, ...]) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _redact_text(value, forbidden, maximum=300)
    return None


def _safe_evidence(
    evidence: Any,
    forbidden: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        return {}
    sanitized: dict[str, Any] = {}
    for raw_key, raw_value in evidence.items():
        key = str(raw_key)
        lowered = key.casefold()
        if (
            key not in _SAFE_EVIDENCE_KEYS
            or any(part in lowered for part in _BANNED_KEY_PARTS)
        ):
            continue
        if isinstance(raw_value, Mapping):
            nested: dict[str, Any] = {}
            for nested_key, nested_value in list(raw_value.items())[:30]:
                normalized_nested_key = str(nested_key)
                allowed_nested_keys = (
                    _SAFE_DECISION_KEYS
                    if key == "decision"
                    else _SAFE_TYPE_COUNT_KEYS
                    if key == "type_counts"
                    else frozenset()
                )
                if (
                    normalized_nested_key not in allowed_nested_keys
                    or any(
                        part in normalized_nested_key.casefold()
                        for part in _BANNED_KEY_PARTS
                    )
                ):
                    continue
                safe_value = _safe_scalar(nested_value, forbidden)
                if safe_value is not None:
                    nested[
                        _redact_text(normalized_nested_key, forbidden, 100)
                    ] = safe_value
            if nested:
                sanitized[key] = nested
            continue
        safe_value = _safe_scalar(raw_value, forbidden)
        if safe_value is not None:
            sanitized[key] = safe_value
    return sanitized


def _safe_reason(value: Any, forbidden: tuple[str, ...]) -> str:
    reason = _redact_text(value or "", forbidden, maximum=500)
    if not reason:
        return "当前报告未提供可评估依据。"
    if "[已隐藏]" in reason:
        if "解析" in reason:
            return "文件未成功解析，相关指标无法计算。"
        return "相关依据含受保护的执行信息，未向 Agent 提供。"
    safe_patterns = (
        "数据集不包含字段",
        "数据集不包含记录",
        "字段没有非空值",
        "未识别到可检查",
        "未识别到时间",
        "未识别到来源",
        "未识别到版本",
        "没有足够",
        "当前无法计算",
    )
    if any(pattern in reason for pattern in safe_patterns):
        return reason
    return "当前报告未提供可安全披露的可评估依据。"


def _fallback_metric_key(metric: Mapping[str, Any]) -> str:
    metric_id = str(metric.get("id") or "unknown_metric")
    field = metric.get("field")
    if not field:
        return metric_id
    return f"{metric_id}::{quote(str(field), safe='')}"


def _numeric_values(*values: Any) -> tuple[float, ...]:
    """只收集结构化数值，不从自由文本、日期或版本号中猜测数字。"""

    numbers: set[float] = set()

    def visit(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                numbers.add(number)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return
    for value in values:
        visit(value)
    return tuple(sorted(numbers))


def extract_number_mentions(text: str) -> tuple[str, ...]:
    """提取需要证据支持的数字表达。"""

    return tuple(_NUMBER_PATTERN.findall(text))


def _mention_value_and_tolerance(token: str) -> tuple[float, float]:
    normalized = token.replace(",", "")
    is_percent = normalized.endswith("%")
    if is_percent:
        normalized = normalized[:-1]
    value = float(normalized)
    decimal_places = (
        len(normalized.rsplit(".", 1)[1]) if "." in normalized else 0
    )
    scale = 100.0 if is_percent else 1.0
    value /= scale
    rounding_tolerance = (0.5 * (10 ** -decimal_places)) / scale
    return value, max(1e-12, rounding_tolerance + 1e-12)


def numbers_are_supported(text: str, evidence_values: tuple[float, ...]) -> bool:
    """确认文本中的每个数字都能由所引证据直接或按显示精度支持。"""

    for token in extract_number_mentions(text):
        value, tolerance = _mention_value_and_tolerance(token)
        if not any(abs(value - evidence) <= tolerance for evidence in evidence_values):
            return False
    return True


@dataclass(frozen=True)
class _NumericClaim:
    value: float
    labels: tuple[str, ...]


def _numeric_claim(value: Any, labels: tuple[str, ...]) -> _NumericClaim | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or not labels:
        return None
    return _NumericClaim(value=normalized, labels=labels)


def _summary_numeric_claims(summary: Mapping[str, Any]) -> tuple[_NumericClaim, ...]:
    claims = []
    for key, labels in _SUMMARY_NUMERIC_LABELS.items():
        claim = _numeric_claim(summary.get(key), labels)
        if claim is not None:
            claims.append(claim)
    return tuple(claims)


def _metric_numeric_claims(
    metric: Mapping[str, Any],
) -> tuple[_NumericClaim, ...]:
    metric_id = str(metric.get("metric_id") or "")
    labels = (
        str(metric.get("name") or ""),
        *_METRIC_NUMERIC_LABELS.get(metric_id, ()),
    )
    cleaned_labels = tuple(label for label in dict.fromkeys(labels) if label)
    claims: list[_NumericClaim] = []
    value_claim = _numeric_claim(metric.get("value"), cleaned_labels)
    if value_claim is not None:
        claims.append(value_claim)
    evidence = metric.get("evidence")
    if isinstance(evidence, Mapping):
        claims.extend(_evidence_numeric_claims(evidence))
    return tuple(claims)


def _evidence_numeric_claims(
    evidence: Mapping[str, Any],
    *,
    observed_fallback_labels: tuple[str, ...] = (),
) -> tuple[_NumericClaim, ...]:
    claims: list[_NumericClaim] = []
    for raw_key, value in evidence.items():
        key = str(raw_key)
        if key == "decision" and isinstance(value, Mapping):
            observed_name = str(value.get("observed_name") or "")
            observed_labels = _METRIC_NUMERIC_LABELS.get(
                observed_name,
                _EVIDENCE_NUMERIC_LABELS.get(
                    observed_name,
                    observed_fallback_labels,
                ),
            )
            observed_claim = _numeric_claim(
                value.get("observed_value"),
                observed_labels,
            )
            if observed_claim is not None:
                claims.append(observed_claim)
            threshold_claim = _numeric_claim(
                value.get("threshold"),
                ("阈值", "门槛"),
            )
            if threshold_claim is not None:
                claims.append(threshold_claim)
            continue
        if key == "type_counts" and isinstance(value, Mapping):
            for type_name, count in value.items():
                claim = _numeric_claim(
                    count,
                    (f"{type_name} 类型数", f"{type_name} 数量"),
                )
                if claim is not None:
                    claims.append(claim)
            continue
        labels = _EVIDENCE_NUMERIC_LABELS.get(key, ())
        claim = _numeric_claim(value, labels)
        if claim is not None:
            claims.append(claim)
    return tuple(claims)


def _risk_numeric_claims(
    risk: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
) -> tuple[_NumericClaim, ...]:
    evidence = risk.get("evidence")
    if not isinstance(evidence, Mapping):
        return ()
    fallback_labels: list[str] = []
    for metric_key in risk.get("related_metric_keys", ()):
        metric = metrics.get(str(metric_key))
        if not isinstance(metric, Mapping):
            continue
        metric_name = str(metric.get("name") or "")
        if metric_name:
            fallback_labels.append(metric_name)
        fallback_labels.extend(
            _METRIC_NUMERIC_LABELS.get(
                str(metric.get("metric_id") or ""),
                (),
            )
        )
    return _evidence_numeric_claims(
        evidence,
        observed_fallback_labels=tuple(
            dict.fromkeys(fallback_labels)
        ),
    )


def _label_distance(
    text: str,
    *,
    mention_start: int,
    mention_end: int,
    label: str,
) -> int | None:
    normalized_text = text.casefold()
    normalized_label = label.casefold()
    if not normalized_label:
        return None
    best: int | None = None
    offset = 0
    while True:
        position = normalized_text.find(normalized_label, offset)
        if position < 0:
            break
        label_end = position + len(normalized_label)
        if label_end <= mention_start:
            distance = mention_start - label_end
        elif position >= mention_end:
            distance = position - mention_end
        else:
            distance = 0
        best = distance if best is None else min(best, distance)
        offset = position + 1
    return best


def _numbers_match_claims(
    text: str,
    claims: tuple[_NumericClaim, ...],
) -> bool:
    maximum_label_distance = 12
    for mention in _NUMBER_PATTERN.finditer(text):
        value, tolerance = _mention_value_and_tolerance(mention.group(0))
        candidates: list[tuple[int, _NumericClaim]] = []
        for claim in claims:
            distances = [
                distance
                for label in claim.labels
                if (
                    distance := _label_distance(
                        text,
                        mention_start=mention.start(),
                        mention_end=mention.end(),
                        label=label,
                    )
                )
                is not None
            ]
            if distances:
                candidates.append((min(distances), claim))
        if not candidates:
            return False
        nearest_distance = min(distance for distance, _ in candidates)
        if nearest_distance > maximum_label_distance:
            return False
        if not any(
            distance == nearest_distance
            and abs(value - claim.value) <= tolerance
            for distance, claim in candidates
        ):
            return False
    return True


@dataclass(frozen=True)
class _CitationRecord:
    citation: AgentCitation
    numeric_values: tuple[float, ...]
    numeric_claims: tuple[_NumericClaim, ...]


class ReportSnapshot:
    """由报告深拷贝生成的不可变、安全投影。

    快照不会保留原始文件内容；所有公开工具每次均返回新的普通对象，调用方
    无法通过修改返回值影响快照或 ``QualityReport``。
    """

    __slots__ = (
        "report_sha256",
        "_summary",
        "_metrics",
        "_risks",
        "_not_assessable",
        "_citations",
    )

    def __init__(
        self,
        *,
        report_sha256: str,
        summary: Mapping[str, Any],
        metrics: Mapping[str, Mapping[str, Any]],
        risks: Mapping[str, Mapping[str, Any]],
        not_assessable: Mapping[str, Mapping[str, Any]],
        citations: Mapping[str, _CitationRecord],
    ) -> None:
        object.__setattr__(self, "report_sha256", report_sha256)
        object.__setattr__(self, "_summary", _freeze(summary))
        object.__setattr__(self, "_metrics", _freeze(metrics))
        object.__setattr__(self, "_risks", _freeze(risks))
        object.__setattr__(self, "_not_assessable", _freeze(not_assessable))
        object.__setattr__(
            self, "_citations", MappingProxyType(dict(citations))
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ReportSnapshot 是只读对象。")

    @classmethod
    def from_report(cls, report: Any) -> "ReportSnapshot":
        to_dict = getattr(report, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("report 必须提供 to_dict()。")
        raw_payload = to_dict()
        if not isinstance(raw_payload, Mapping):
            raise TypeError("report.to_dict() 必须返回对象。")
        payload = deepcopy(dict(raw_payload))
        forbidden = _forbidden_strings(payload)
        report_sha256 = _report_hash(payload)

        raw_metrics = payload.get("metrics")
        raw_metrics = raw_metrics if isinstance(raw_metrics, list) else []
        metrics: dict[str, dict[str, Any]] = {}
        for raw_metric in raw_metrics:
            if not isinstance(raw_metric, Mapping):
                continue
            metric_key = str(
                raw_metric.get("metric_key") or _fallback_metric_key(raw_metric)
            )
            if metric_key in metrics:
                continue
            status = (
                str(raw_metric.get("status"))
                if raw_metric.get("status") in {"evaluated", "not_assessable"}
                else "not_assessable"
            )
            metric = {
                "citation_id": f"metric:{metric_key}",
                "metric_key": metric_key,
                "metric_id": _redact_text(
                    raw_metric.get("id") or "unknown_metric", forbidden, 150
                ),
                "name": _redact_text(
                    raw_metric.get("name") or "未命名指标", forbidden, 200
                ),
                "category": _redact_text(
                    raw_metric.get("category") or "未分类", forbidden, 100
                ),
                "status": status,
                "scope": (
                    str(raw_metric.get("scope"))
                    if raw_metric.get("scope") in {"dataset", "field"}
                    else "dataset"
                ),
                "field": (
                    _redact_text(raw_metric.get("field"), forbidden, 200)
                    if raw_metric.get("field") is not None
                    else None
                ),
                "value": _safe_scalar(raw_metric.get("value"), forbidden),
                "unit": (
                    _redact_text(raw_metric.get("unit"), forbidden, 50)
                    if raw_metric.get("unit") is not None
                    else None
                ),
                "reason": (
                    _safe_reason(raw_metric.get("reason"), forbidden)
                    if status == "not_assessable"
                    else None
                ),
                "evidence": _safe_evidence(
                    raw_metric.get("evidence"), forbidden
                ),
            }
            metrics[metric_key] = metric

        raw_risks = payload.get("risks")
        raw_risks = raw_risks if isinstance(raw_risks, list) else []
        risks: dict[str, dict[str, Any]] = {}
        for raw_risk in raw_risks:
            if not isinstance(raw_risk, Mapping):
                continue
            risk_id = _redact_text(
                raw_risk.get("id") or f"risk-{len(risks) + 1}",
                forbidden,
                200,
            )
            if risk_id in risks:
                continue
            related_keys_raw = raw_risk.get("related_metric_keys")
            related_keys = (
                [str(item) for item in related_keys_raw]
                if isinstance(related_keys_raw, list)
                else []
            )
            if not related_keys:
                legacy_ids = raw_risk.get("related_metrics")
                if isinstance(legacy_ids, list):
                    for legacy_id in legacy_ids:
                        related_keys.extend(
                            key
                            for key, metric in metrics.items()
                            if metric["metric_id"] == str(legacy_id)
                        )
            related_keys = list(
                dict.fromkeys(key for key in related_keys if key in metrics)
            )
            risks[risk_id] = {
                "citation_id": f"risk:{risk_id}",
                "risk_id": risk_id,
                "level": (
                    str(raw_risk.get("level"))
                    if raw_risk.get("level")
                    in {"warning", "attention", "info"}
                    else "info"
                ),
                "title": _redact_text(
                    raw_risk.get("title") or "未命名风险", forbidden, 300
                ),
                "message": _redact_text(
                    raw_risk.get("message") or "报告未提供风险说明。",
                    forbidden,
                    700,
                ),
                "related_metric_keys": related_keys,
                "evidence": _safe_evidence(
                    raw_risk.get("evidence"), forbidden
                ),
            }

        raw_not_assessable = payload.get("not_assessable")
        raw_not_assessable = (
            raw_not_assessable if isinstance(raw_not_assessable, list) else []
        )
        not_assessable: dict[str, dict[str, Any]] = {}
        for raw_item in raw_not_assessable:
            if not isinstance(raw_item, Mapping):
                continue
            metric_key = str(
                raw_item.get("metric_key")
                or raw_item.get("id")
                or f"not_assessable_{len(not_assessable) + 1}"
            )
            if metric_key in not_assessable:
                continue
            linked_metric = metrics.get(metric_key)
            if linked_metric is None:
                item_id = str(raw_item.get("id") or "")
                linked_metric = next(
                    (
                        metric
                        for metric in metrics.values()
                        if metric["metric_id"] == item_id
                    ),
                    None,
                )
                if linked_metric is not None:
                    metric_key = str(linked_metric["metric_key"])
            not_assessable[metric_key] = {
                "citation_id": f"not_assessable:{metric_key}",
                "metric_key": metric_key,
                "name": _redact_text(
                    raw_item.get("name")
                    or (linked_metric or {}).get("name")
                    or "未命名指标",
                    forbidden,
                    250,
                ),
                "reason": _safe_reason(raw_item.get("reason"), forbidden),
            }

        risk_counts = {"warning": 0, "attention": 0, "info": 0}
        for risk in risks.values():
            risk_counts[str(risk["level"])] += 1
        profile = payload.get("profile")
        profile = profile if isinstance(profile, Mapping) else {}
        metric_definition_ids = {
            str(metric["metric_id"]) for metric in metrics.values()
        }
        evaluated_definition_ids = {
            str(metric["metric_id"])
            for metric in metrics.values()
            if metric["status"] == "evaluated"
        }
        evaluation_context = payload.get("evaluation_context")
        evaluation_context = (
            evaluation_context
            if isinstance(evaluation_context, Mapping)
            else {}
        )
        parser_path_raw = evaluation_context.get("parser_path")
        parser_path = (
            _redact_text(parser_path_raw, forbidden, 100)
            if isinstance(parser_path_raw, str)
            and "/" not in parser_path_raw
            and "\\" not in parser_path_raw
            else None
        )
        summary = {
            "citation_id": "report:summary",
            "report_sha256": report_sha256,
            "status": (
                str(payload.get("status"))
                if payload.get("status") in {"success", "partial_success", "failed"}
                else "failed"
            ),
            "row_count": int(profile.get("row_count", 0) or 0),
            "column_count": int(profile.get("column_count", 0) or 0),
            "metric_definition_count": len(metric_definition_ids),
            "evaluated_metric_definition_count": len(
                evaluated_definition_ids
            ),
            "metric_result_count": len(metrics),
            "evaluated_metric_result_count": sum(
                metric["status"] == "evaluated" for metric in metrics.values()
            ),
            "risk_count": len(risks),
            "warning_count": risk_counts["warning"],
            "attention_count": risk_counts["attention"],
            "info_count": risk_counts["info"],
            "not_assessable_count": len(not_assessable),
            "engine_version": _safe_scalar(
                evaluation_context.get("engine_version"), forbidden
            ),
            "reference_date": _safe_scalar(
                evaluation_context.get("reference_date"), forbidden
            ),
            "threshold_config_version": _safe_scalar(
                evaluation_context.get("threshold_config_version"), forbidden
            ),
            "parser_path": parser_path,
        }

        citations: dict[str, _CitationRecord] = {
            "report:summary": _CitationRecord(
                citation=AgentCitation(
                    id="report:summary",
                    source_type="summary",
                    source_id=report_sha256,
                    label="报告摘要",
                    excerpt=(
                        f"记录数 {summary['row_count']}，"
                        f"字段数 {summary['column_count']}，"
                        f"指标种类 {summary['metric_definition_count']}，"
                        f"指标结果 {summary['metric_result_count']}，"
                        f"风险 {summary['risk_count']}，"
                        f"警告 {summary['warning_count']}，"
                        f"关注 {summary['attention_count']}，"
                        f"提示 {summary['info_count']}，"
                        f"无法评估 {summary['not_assessable_count']} 项。"
                    ),
                ),
                numeric_values=_numeric_values(
                    {
                        key: summary[key]
                        for key in (
                            "row_count",
                            "column_count",
                            "metric_definition_count",
                            "evaluated_metric_definition_count",
                            "metric_result_count",
                            "evaluated_metric_result_count",
                            "risk_count",
                            "warning_count",
                            "attention_count",
                            "info_count",
                            "not_assessable_count",
                        )
                    }
                ),
                numeric_claims=_summary_numeric_claims(summary),
            )
        }
        for metric_key, metric in metrics.items():
            citation_id = str(metric["citation_id"])
            value_text = (
                "无法评估"
                if metric["value"] is None
                else f"{metric['value']} {metric['unit'] or ''}".strip()
            )
            citations[citation_id] = _CitationRecord(
                citation=AgentCitation(
                    id=citation_id,
                    source_type="metric",
                    source_id=metric_key,
                    label=str(metric["name"]),
                    excerpt=f"{metric['name']}：{value_text}",
                ),
                numeric_values=_numeric_values(
                    metric["value"],
                    metric["evidence"],
                ),
                numeric_claims=_metric_numeric_claims(metric),
            )
        for risk_id, risk in risks.items():
            citation_id = str(risk["citation_id"])
            linked_values = [
                {
                    "value": metrics[key]["value"],
                    "evidence": metrics[key]["evidence"],
                }
                for key in risk["related_metric_keys"]
                if key in metrics
            ]
            citations[citation_id] = _CitationRecord(
                citation=AgentCitation(
                    id=citation_id,
                    source_type="risk",
                    source_id=risk_id,
                    label=str(risk["title"]),
                    excerpt=str(risk["message"]),
                ),
                numeric_values=_numeric_values(
                    risk["evidence"],
                    linked_values,
                ),
                numeric_claims=_risk_numeric_claims(risk, metrics),
            )
        for metric_key, item in not_assessable.items():
            citation_id = str(item["citation_id"])
            citations[citation_id] = _CitationRecord(
                citation=AgentCitation(
                    id=citation_id,
                    source_type="not_assessable",
                    source_id=metric_key,
                    label=str(item["name"]),
                    excerpt=str(item["reason"]),
                ),
                numeric_values=(),
                numeric_claims=(),
            )

        return cls(
            report_sha256=report_sha256,
            summary=summary,
            metrics=metrics,
            risks=risks,
            not_assessable=not_assessable,
            citations=citations,
        )

    @property
    def citation_ids(self) -> frozenset[str]:
        return frozenset(self._citations)

    def get_portable_context(self) -> dict[str, Any]:
        """返回不依赖工具调用的兼容模型上下文。

        一些模型接口只支持普通 messages，不支持 tools 或 tool_choice。
        该上下文仍只包含本模块已经生成的聚合证据，不包含原始单元格、样例值、
        行号列表、文件名或执行错误。
        """

        return {
            "report_summary": self.get_report_summary(),
            "metrics": [
                self.get_metric_evidence(metric_key=metric_key)
                for metric_key in sorted(self._metrics)
            ],
            "risks": [
                self.get_risk_evidence(risk_id=risk_id)
                for risk_id in sorted(self._risks)
            ],
            "not_assessable": self.list_not_assessable(limit=20),
            "citations": [
                self.citation(citation_id).to_dict()
                for citation_id in sorted(self._citations)
            ],
        }

    def citation(self, citation_id: str) -> AgentCitation:
        try:
            return self._citations[citation_id].citation
        except KeyError as error:
            raise AgentToolError("引用不属于当前报告。") from error

    def numeric_values_for(
        self, citation_ids: tuple[str, ...] | list[str]
    ) -> tuple[float, ...]:
        values: set[float] = set()
        for citation_id in citation_ids:
            try:
                values.update(self._citations[citation_id].numeric_values)
            except KeyError as error:
                raise AgentToolError("引用不属于当前报告。") from error
        return tuple(sorted(values))

    def statement_numbers_are_supported(
        self,
        text: str,
        citation_ids: tuple[str, ...] | list[str],
    ) -> bool:
        """按数值、最近语义标签和引用共同验证一句模型文本。"""

        claims: list[_NumericClaim] = []
        for citation_id in citation_ids:
            try:
                claims.extend(
                    self._citations[citation_id].numeric_claims
                )
            except KeyError as error:
                raise AgentToolError("引用不属于当前报告。") from error
        return _numbers_match_claims(text, tuple(claims))

    def get_report_summary(self) -> dict[str, Any]:
        return _thaw(self._summary)

    def list_priority_risks(self, *, limit: int = 5) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
            raise AgentToolError("limit 必须是 1 到 10 的整数。")
        risks = sorted(
            (dict(_thaw(item)) for item in self._risks.values()),
            key=lambda item: (
                _RISK_LEVEL_ORDER.get(str(item["level"]), 99),
                str(item["risk_id"]),
            ),
        )
        return risks[:limit]

    def get_risk_evidence(self, *, risk_id: str) -> dict[str, Any]:
        if not isinstance(risk_id, str) or not risk_id:
            raise AgentToolError("risk_id 必须是非空字符串。")
        try:
            return _thaw(self._risks[risk_id])
        except KeyError as error:
            raise AgentToolError("未找到对应风险。") from error

    def get_metric_evidence(self, *, metric_key: str) -> dict[str, Any]:
        if not isinstance(metric_key, str) or not metric_key:
            raise AgentToolError("metric_key 必须是非空字符串。")
        try:
            return _thaw(self._metrics[metric_key])
        except KeyError as error:
            raise AgentToolError("未找到对应指标。") from error

    def list_not_assessable(self, *, limit: int = 10) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise AgentToolError("limit 必须是 1 到 20 的整数。")
        return [
            _thaw(item)
            for _, item in sorted(self._not_assessable.items())
        ][:limit]

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Any:
        if name not in TOOL_NAMES:
            raise AgentToolError("工具不在只读白名单中。")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            raise AgentToolError("工具参数必须是对象。")
        args = dict(arguments)
        if name == "get_report_summary":
            if args:
                raise AgentToolError("get_report_summary 不接受参数。")
            return self.get_report_summary()
        if name == "list_priority_risks":
            if set(args) - {"limit"}:
                raise AgentToolError("list_priority_risks 收到未知参数。")
            return self.list_priority_risks(limit=args.get("limit", 5))
        if name == "get_risk_evidence":
            if set(args) != {"risk_id"}:
                raise AgentToolError("get_risk_evidence 需要 risk_id。")
            return self.get_risk_evidence(risk_id=args["risk_id"])
        if name == "get_metric_evidence":
            if set(args) != {"metric_key"}:
                raise AgentToolError("get_metric_evidence 需要 metric_key。")
            return self.get_metric_evidence(metric_key=args["metric_key"])
        if set(args) - {"limit"}:
            raise AgentToolError("list_not_assessable 收到未知参数。")
        return self.list_not_assessable(limit=args.get("limit", 10))


def deepseek_tool_definitions() -> list[dict[str, Any]]:
    """返回 DeepSeek Chat Completions 使用的只读函数工具定义。

    生产端点不启用 DeepSeek beta ``strict`` 模式；参数仍会在本地白名单
    调度层逐项校验。
    """

    functions = [
        {
            "name": "get_report_summary",
            "description": "读取不含文件名、数据集名和执行错误的报告汇总。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "name": "list_priority_risks",
            "description": "按警告、关注、提示的顺序列出风险及其引用编号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    }
                },
                "required": ["limit"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_risk_evidence",
            "description": "按 risk_id 读取某项风险的聚合证据。",
            "parameters": {
                "type": "object",
                "properties": {"risk_id": {"type": "string"}},
                "required": ["risk_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_metric_evidence",
            "description": "按 metric_key 读取某项指标的值和聚合证据。",
            "parameters": {
                "type": "object",
                "properties": {"metric_key": {"type": "string"}},
                "required": ["metric_key"],
                "additionalProperties": False,
            },
        },
        {
            "name": "list_not_assessable",
            "description": "列出无法评估项目及经过脱敏归类的原因。",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                    }
                },
                "required": ["limit"],
                "additionalProperties": False,
            },
        },
    ]
    return [
        {"type": "function", "function": function}
        for function in functions
    ]
