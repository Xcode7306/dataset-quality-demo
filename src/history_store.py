"""v0.5 会话级报告历史、严格导入校验与版本趋势。

历史功能只保存已经固化的 ``QualityReport`` JSON。默认实现驻留在当前
Streamlit 会话内存中，不保存上传字节、问题位置 CSV、Agent 输出或 RulePack。
未来若接入持久化仓库，应继续复用本模块的严格报告校验和容量边界。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

from jsonschema import Draft202012Validator, FormatChecker

from .models import QualityReport, build_metric_key


MAX_HISTORY_REPORT_BYTES = 8 * 1024 * 1024
MAX_HISTORY_TOTAL_BYTES = 32 * 1024 * 1024
MAX_HISTORY_REPORTS = 20
MAX_HISTORY_JSON_DEPTH = 64
HISTORY_POLICY_VERSION = "0.5"

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_OR_SURROGATE_PATTERN = re.compile(
    r"[\x00-\x1f\x7f\ud800-\udfff]"
)
_PROFILE_KEYS = frozenset(
    {
        "row_count",
        "column_count",
        "columns",
        "recognized_fields",
        "warnings",
    }
)
_PROFILE_COLUMN_KEYS = frozenset(
    {
        "name",
        "missing_count",
        "non_missing_count",
        "missing_rate",
        "inferred_type",
        "non_null_samples",
    }
)
_RECOGNIZED_FIELD_KEYS = frozenset(
    {"date", "numeric", "url", "source", "version"}
)
_EMPTY_SAMPLE_EVIDENCE_KEYS = frozenset(
    {
        "invalid_samples",
        "non_finite_samples",
        "outlier_samples",
    }
)
_FIELD_LIST_EVIDENCE_KEYS = frozenset(
    {
        "content_fields",
        "compared_fields",
        "identified_fields",
        "fields",
    }
)
_ROW_INDEX_EVIDENCE_KEYS = frozenset({"missing_row_indices"})
_ALLOWED_EVIDENCE_KEYS = frozenset(
    {
        "absolute_lag_days",
        "additional_duplicate_rate",
        "allowed_value_count",
        "attempted_file_count",
        "available_count",
        "checked_count",
        "column_count",
        "compliant_count",
        "compared_fields",
        "content_fields",
        "covered_count",
        "decision",
        "dominant_type",
        "duplicate_group_count",
        "duplicate_groups",
        "duplicate_record_count",
        "earliest_date",
        "excluded_missing_count",
        "expected_format",
        "failed_file_count",
        "field",
        "fields",
        "file_count",
        "frequency",
        "future_date",
        "identified_fields",
        "inclusive",
        "invalid_samples",
        "iqr",
        "issue_count",
        "latest_date",
        "latest_update_date",
        "lower_bound",
        "max_age_days",
        "maximum",
        "method",
        "minimum",
        "missing_record_count",
        "missing_row_indices",
        "non_finite_count",
        "non_finite_samples",
        "normalization",
        "numeric_count",
        "outlier_samples",
        "parseable_count",
        "q1",
        "q3",
        "reference_date",
        "row_count",
        "rule_id",
        "rule_definition_sha256",
        "rule_pack_id",
        "rule_pack_sha256",
        "rule_pack_version",
        "rule_type",
        "successful_file_count",
        "threshold_source",
        "total_count",
        "type_counts",
        "update_lag_days",
        "upper_bound",
    }
)
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "quality-report.schema.json"
)


class HistoryValidationError(ValueError):
    """历史报告、版本标签或会话容量未通过安全校验。"""


class ReportHistoryStore(Protocol):
    """为未来持久化适配器预留的最小历史仓库协议。"""

    def list_entries(self) -> tuple["HistoryEntry", ...]:
        ...

    def add_report(
        self,
        report: QualityReport | Mapping[str, Any],
        *,
        version_label: str,
        dataset_series_id: str,
        saved_at: datetime | str | None = None,
    ) -> "HistoryEntry":
        ...

    def delete(self, entry_id: str) -> bool:
        ...

    def clear(self) -> int:
        ...


@dataclass(frozen=True)
class HistoryPolicy:
    """当前本地 Demo 的历史保存、访问和删除策略快照。"""

    policy_version: str = HISTORY_POLICY_VERSION
    storage_mode: str = "session_memory"
    retention: str = "until_explicit_deletion_or_session_termination"
    access_scope: str = "current_local_browser_session"
    identity_authentication: bool = False
    manual_deletion_supported: bool = True
    raw_upload_bytes_stored: bool = False
    issue_location_csv_stored: bool = False
    max_reports: int = MAX_HISTORY_REPORTS
    max_report_bytes: int = MAX_HISTORY_REPORT_BYTES
    max_total_bytes: int = MAX_HISTORY_TOTAL_BYTES

    def __post_init__(self) -> None:
        limits = (
            self.max_reports,
            self.max_report_bytes,
            self.max_total_bytes,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in limits
        ) or self.max_report_bytes > self.max_total_bytes:
            raise HistoryValidationError("历史容量策略配置无效。")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "storage_mode": self.storage_mode,
            "retention": self.retention,
            "access_scope": self.access_scope,
            "identity_authentication": self.identity_authentication,
            "manual_deletion_supported": self.manual_deletion_supported,
            "raw_upload_bytes_stored": self.raw_upload_bytes_stored,
            "issue_location_csv_stored": self.issue_location_csv_stored,
            "max_reports": self.max_reports,
            "max_report_bytes": self.max_report_bytes,
            "max_total_bytes": self.max_total_bytes,
        }


DEFAULT_HISTORY_POLICY = HistoryPolicy()


@dataclass(frozen=True)
class HistoryEntry:
    """一份绑定数据系列、版本标签和完整报告哈希的会话历史。"""

    entry_id: str
    version_label: str
    dataset_series_id: str
    saved_at: str
    report_sha256: str
    size_bytes: int
    _report_payload: dict[str, Any] = field(repr=False, compare=False)

    @property
    def report_payload(self) -> dict[str, Any]:
        """返回深拷贝，避免页面代码修改已固化历史。"""

        return deepcopy(self._report_payload)

    def to_summary_dict(self) -> dict[str, Any]:
        payload = self._report_payload
        context = payload["evaluation_context"]
        profile = payload.get("profile", {})
        risks = payload.get("risks", [])
        return {
            "entry_id": self.entry_id,
            "version_label": self.version_label,
            "dataset_series_id": self.dataset_series_id,
            "saved_at": self.saved_at,
            "report_sha256": self.report_sha256,
            "input_sha256": context.get("input_sha256"),
            "dataset_name": payload["dataset"]["name"],
            "status": payload["status"],
            "reference_date": context.get("reference_date"),
            "engine_version": context.get("engine_version"),
            "row_count": _non_negative_int(profile.get("row_count")),
            "column_count": _non_negative_int(profile.get("column_count")),
            "evaluated_metric_count": sum(
                metric.get("status") == "evaluated"
                for metric in payload.get("metrics", [])
                if isinstance(metric, Mapping)
            ),
            "risk_count": len(risks),
            "warning_count": sum(
                risk.get("level") == "warning"
                for risk in risks
                if isinstance(risk, Mapping)
            ),
            "attention_count": sum(
                risk.get("level") == "attention"
                for risk in risks
                if isinstance(risk, Mapping)
            ),
            "info_count": sum(
                risk.get("level") == "info"
                for risk in risks
                if isinstance(risk, Mapping)
            ),
            "not_assessable_count": len(payload.get("not_assessable", [])),
            "size_bytes": self.size_bytes,
        }


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _strict_text(value: Any, label: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise HistoryValidationError(f"{label}必须是文本。")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > maximum
        or _CONTROL_OR_SURROGATE_PATTERN.search(normalized)
    ):
        raise HistoryValidationError(
            f"{label}必须为 1 到 {maximum} 个不含控制字符的 Unicode 字符。"
        )
    return normalized


def _utc_text(value: datetime | str | None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as error:
            raise HistoryValidationError("历史保存时间必须是 ISO 8601 时间。") from error
    else:
        raise HistoryValidationError("历史保存时间类型无效。")
    if parsed.tzinfo is None:
        raise HistoryValidationError("历史保存时间必须包含时区。")
    return (
        parsed.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _scan_json_depth(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAX_HISTORY_JSON_DEPTH:
                raise HistoryValidationError(
                    f"历史报告 JSON 嵌套深度超过 {MAX_HISTORY_JSON_DEPTH} 层。"
                )
        elif character in "]}":
            depth = max(0, depth - 1)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HistoryValidationError(f"历史报告 JSON 包含重复键：{key}。")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise HistoryValidationError(f"历史报告 JSON 包含非标准数值：{value}。")


def _validate_unicode(value: Any, location: str = "root") -> None:
    if isinstance(value, str):
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise HistoryValidationError(
                f"历史报告 {location} 包含孤立 Unicode 代理字符。"
            ) from error
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_unicode(key, f"{location}.<key>")
            _validate_unicode(item, f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_unicode(item, f"{location}[{index}]")


@lru_cache(maxsize=1)
def _report_validator() -> Draft202012Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _canonical_report_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8", errors="strict")


def _recompute_report_sha256(payload: Mapping[str, Any]) -> str:
    hash_payload = deepcopy(dict(payload))
    context = hash_payload.get("evaluation_context")
    if not isinstance(context, dict):
        raise HistoryValidationError("历史报告缺少有效 evaluation_context。")
    context.pop("report_sha256", None)
    return hashlib.sha256(_canonical_report_bytes(hash_payload)).hexdigest()


def _validate_report_invariants(payload: Mapping[str, Any]) -> None:
    metrics = payload.get("metrics")
    risks = payload.get("risks")
    not_assessable = payload.get("not_assessable")
    if not isinstance(metrics, list) or not isinstance(risks, list):
        raise HistoryValidationError("历史报告指标或风险结构无效。")
    if not isinstance(not_assessable, list):
        raise HistoryValidationError("历史报告无法评估项结构无效。")
    _validate_privacy_shape(payload)

    metric_keys: list[str] = []
    not_assessable_metric_keys: set[str] = set()
    metrics_by_key: dict[str, Mapping[str, Any]] = {}
    for metric in metrics:
        if not isinstance(metric, Mapping):
            raise HistoryValidationError("历史报告包含非对象指标。")
        expected_key = build_metric_key(
            str(metric.get("id", "")),
            metric.get("scope"),  # type: ignore[arg-type]
            metric.get("field"),
        )
        metric_key = metric.get("metric_key")
        if metric_key != expected_key:
            raise HistoryValidationError(
                f"历史报告指标键与指标定义不匹配：{metric_key}。"
            )
        metric_keys.append(str(metric_key))
        metrics_by_key[str(metric_key)] = metric
        status = metric.get("status")
        value = metric.get("value")
        reason = metric.get("reason")
        if status == "not_assessable":
            if value is not None or not isinstance(reason, str) or not reason:
                raise HistoryValidationError(
                    "历史报告无法评估指标的值与原因不一致。"
                )
            not_assessable_metric_keys.add(str(metric_key))
        elif status == "evaluated":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or reason is not None
            ):
                raise HistoryValidationError(
                    "历史报告已评估指标的值或原因无效。"
                )
            try:
                decimal_value = Decimal(str(value))
            except (InvalidOperation, ValueError) as error:
                raise HistoryValidationError(
                    "历史报告指标数值超出安全范围。"
                ) from error
            if (
                not decimal_value.is_finite()
                or abs(decimal_value) > Decimal("1e308")
            ):
                raise HistoryValidationError(
                    "历史报告指标数值超出安全范围。"
                )
            unit = metric.get("unit")
            if (
                unit not in {None, "ratio", "records", "days"}
                or (
                    unit == "ratio"
                    and not Decimal(0) <= decimal_value <= Decimal(1)
                )
                or (
                    unit == "records"
                    and (
                        not isinstance(value, int)
                        or value < 0
                    )
                )
                or (
                    unit == "days"
                    and not isinstance(value, int)
                )
            ):
                raise HistoryValidationError(
                    "历史报告指标数值与单位不符合固定指标契约。"
                )
    if len(metric_keys) != len(set(metric_keys)):
        raise HistoryValidationError("历史报告包含重复 metric_key。")

    risk_ids = [
        str(risk.get("id"))
        for risk in risks
        if isinstance(risk, Mapping)
    ]
    if len(risk_ids) != len(risks) or len(risk_ids) != len(set(risk_ids)):
        raise HistoryValidationError("历史报告包含重复或无效 risk_id。")
    metric_key_set = set(metric_keys)
    threshold_config_version = payload.get(
        "evaluation_context",
        {},
    ).get("threshold_config_version")
    for risk in risks:
        related = risk.get("related_metric_keys", [])
        if (
            not related
            or any(key not in metric_key_set for key in related)
        ):
            raise HistoryValidationError("历史报告风险引用了不存在的指标键。")
        expected_related_ids = {
            str(metrics_by_key[key].get("id"))
            for key in related
        }
        if set(risk.get("related_metrics", [])) != expected_related_ids:
            raise HistoryValidationError(
                "历史报告风险的指标 ID 与指标键引用不一致。"
            )
        decision = risk.get("evidence", {}).get("decision", {})
        is_business_risk = bool(expected_related_ids) and all(
            metric_id.startswith("business_")
            for metric_id in expected_related_ids
        )
        if is_business_risk:
            evidence = risk.get("evidence", {})
            rule_pack_id = evidence.get("rule_pack_id")
            rule_pack_version = evidence.get("rule_pack_version")
            expected_threshold_version = (
                f"rule-pack:{rule_pack_id}:{rule_pack_version}"
                if isinstance(rule_pack_id, str)
                and rule_pack_id
                and isinstance(rule_pack_version, str)
                and rule_pack_version
                else None
            )
            for key in related:
                metric_evidence = metrics_by_key[key].get("evidence", {})
                if any(
                    metric_evidence.get(field) != evidence.get(field)
                    for field in (
                        "rule_pack_id",
                        "rule_pack_version",
                        "rule_pack_sha256",
                        "rule_definition_sha256",
                        "rule_id",
                    )
                ):
                    raise HistoryValidationError(
                        "历史报告业务风险与关联规则指标证据不一致。"
                    )
        else:
            expected_threshold_version = threshold_config_version
        if (
            decision.get("threshold_config_version")
            != expected_threshold_version
        ):
            raise HistoryValidationError(
                "历史报告风险判定的阈值版本与报告上下文不一致。"
            )

    item_keys: list[str] = []
    for item in not_assessable:
        if not isinstance(item, Mapping):
            continue
        item_key = str(item.get("metric_key"))
        item_keys.append(item_key)
        metric = metrics_by_key.get(item_key)
        if metric is None:
            continue
        field = metric.get("field")
        allowed_item_ids = {str(metric.get("id"))}
        allowed_item_names = {str(metric.get("name"))}
        if field is not None:
            allowed_item_ids.add(f"{metric.get('id')}:{field}")
            allowed_item_names.add(f"{metric.get('name')}（{field}）")
        if (
            item.get("id") not in allowed_item_ids
            or item.get("name") not in allowed_item_names
            or item.get("reason") != metric.get("reason")
        ):
            raise HistoryValidationError(
                "历史报告无法评估项与对应指标定义不一致。"
            )
    if (
        len(item_keys) != len(not_assessable)
        or len(item_keys) != len(set(item_keys))
        or set(item_keys) != not_assessable_metric_keys
    ):
        raise HistoryValidationError(
            "历史报告的无法评估项与指标状态不一致。"
        )


def _validate_privacy_shape(payload: Mapping[str, Any]) -> None:
    """拒绝引擎契约外的原始行、样例值或任意嵌套证据。"""

    profile = payload.get("profile", {})
    if not isinstance(profile, Mapping):
        raise HistoryValidationError("历史报告字段画像结构无效。")
    if payload.get("status") != "failed" and not profile:
        raise HistoryValidationError(
            "成功或部分成功的历史报告必须包含完整字段画像。"
        )
    if profile and set(profile) != _PROFILE_KEYS:
        raise HistoryValidationError(
            "历史报告字段画像包含契约外内容，可能携带原始记录。"
        )
    columns = profile.get("columns", [])
    if not isinstance(columns, list):
        raise HistoryValidationError("历史报告字段画像 columns 结构无效。")
    column_names: list[str] = []
    for column in columns:
        if (
            not isinstance(column, Mapping)
            or set(column) != _PROFILE_COLUMN_KEYS
            or not isinstance(column.get("name"), str)
            or isinstance(column.get("missing_count"), bool)
            or not isinstance(column.get("missing_count"), int)
            or column["missing_count"] < 0
            or isinstance(column.get("non_missing_count"), bool)
            or not isinstance(column.get("non_missing_count"), int)
            or column["non_missing_count"] < 0
            or (
                column.get("missing_rate") is not None
                and (
                    isinstance(column["missing_rate"], bool)
                    or not isinstance(column["missing_rate"], (int, float))
                    or not 0 <= column["missing_rate"] <= 1
                )
            )
            or column.get("inferred_type")
            not in {"boolean", "datetime", "numeric", "text", "unknown"}
            or column.get("non_null_samples") != []
        ):
            raise HistoryValidationError(
                "历史报告字段画像包含原始样例或非标准字段。"
            )
        column_names.append(column["name"])
    if len(column_names) != len(set(column_names)):
        raise HistoryValidationError("历史报告字段画像包含重复字段名。")
    if "column_count" in profile and profile["column_count"] != len(columns):
        raise HistoryValidationError(
            "历史报告字段画像的字段数量与 columns 不一致。"
        )
    row_count = profile.get("row_count")
    column_count = profile.get("column_count")
    if profile and (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 0
        or isinstance(column_count, bool)
        or not isinstance(column_count, int)
        or column_count < 0
        or any(
            column["missing_count"] + column["non_missing_count"]
            != row_count
            for column in columns
        )
    ):
        raise HistoryValidationError(
            "历史报告字段画像的行列计数不一致。"
        )
    if "recognized_fields" in profile:
        recognized = profile["recognized_fields"]
        if (
            not isinstance(recognized, Mapping)
            or set(recognized) != _RECOGNIZED_FIELD_KEYS
            or any(
                not isinstance(fields, list)
                or any(
                    not isinstance(field, str)
                    or field not in set(column_names)
                    for field in fields
                )
                for fields in recognized.values()
            )
        ):
            raise HistoryValidationError(
                "历史报告字段语义画像不符合固定字段引用契约。"
            )
    if "warnings" in profile and (
        not isinstance(profile["warnings"], list)
        or any(
            not isinstance(message, str)
            for message in profile["warnings"]
        )
    ):
        raise HistoryValidationError("历史报告字段画像 warning 结构无效。")

    evidence_items = [
        item.get("evidence", {})
        for collection in (payload.get("metrics", []), payload.get("risks", []))
        for item in collection
        if isinstance(item, Mapping)
    ]
    for evidence in evidence_items:
        if not isinstance(evidence, Mapping):
            raise HistoryValidationError("历史报告证据必须是对象。")
        unknown_keys = set(evidence) - _ALLOWED_EVIDENCE_KEYS
        if unknown_keys:
            raise HistoryValidationError(
                "历史报告证据包含契约外内容，可能携带原始记录。"
            )
        for key, value in evidence.items():
            if isinstance(value, Mapping) and key not in {
                "decision",
                "type_counts",
            }:
                raise HistoryValidationError(
                    "历史报告证据包含契约外嵌套对象。"
                )
            if isinstance(value, list) and key not in (
                _EMPTY_SAMPLE_EVIDENCE_KEYS
                | _FIELD_LIST_EVIDENCE_KEYS
                | _ROW_INDEX_EVIDENCE_KEYS
                | {"duplicate_groups"}
            ):
                raise HistoryValidationError(
                    "历史报告证据包含契约外数组。"
                )
        if any(
            key in evidence and evidence[key] != []
            for key in _EMPTY_SAMPLE_EVIDENCE_KEYS
        ):
            raise HistoryValidationError(
                "历史报告证据不得保存原始值或异常样例。"
            )
        for key in _FIELD_LIST_EVIDENCE_KEYS:
            if key in evidence and (
                not isinstance(evidence[key], list)
                or any(
                    not isinstance(field, str)
                    or field not in set(column_names)
                    for field in evidence[key]
                )
            ):
                raise HistoryValidationError(
                    "历史报告证据包含无效字段引用。"
                )
        if (
            "field" in evidence
            and evidence["field"] is not None
            and (
                not isinstance(evidence["field"], str)
                or evidence["field"] not in set(column_names)
            )
        ):
            raise HistoryValidationError(
                "历史报告证据包含无效字段引用。"
            )
        for key in _ROW_INDEX_EVIDENCE_KEYS:
            if key in evidence and (
                not isinstance(evidence[key], list)
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 1
                    for index in evidence[key]
                )
            ):
                raise HistoryValidationError(
                    "历史报告证据包含无效记录位置。"
                )
        groups = evidence.get("duplicate_groups")
        if groups is not None and (
            not isinstance(groups, list)
            or any(
                not isinstance(group, Mapping)
                or set(group) != {"row_indices", "duplicate_count"}
                or not isinstance(group["row_indices"], list)
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 1
                    for index in group["row_indices"]
                )
                or isinstance(group["duplicate_count"], bool)
                or not isinstance(group["duplicate_count"], int)
                or group["duplicate_count"] < 1
                for group in groups
            )
        ):
            raise HistoryValidationError(
                "历史报告重复组证据结构无效。"
            )
        type_counts = evidence.get("type_counts")
        if type_counts is not None and (
            not isinstance(type_counts, Mapping)
            or any(
                key not in {
                    "boolean",
                    "datetime",
                    "numeric",
                    "text",
                    "unknown",
                }
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                for key, count in type_counts.items()
            )
        ):
            raise HistoryValidationError(
                "历史报告类型统计证据结构无效。"
            )


def validate_quality_report_payload(
    report: QualityReport | Mapping[str, Any],
) -> dict[str, Any]:
    """返回经过 Schema、哈希和交叉引用复核的报告深拷贝。"""

    if isinstance(report, QualityReport):
        payload: Any = report.to_dict()
    elif isinstance(report, Mapping):
        payload = deepcopy(dict(report))
    else:
        to_dict = getattr(report, "to_dict", None)
        if not callable(to_dict):
            raise HistoryValidationError("历史对象不是 QualityReport。")
        payload = to_dict()
    if not isinstance(payload, dict):
        raise HistoryValidationError("历史报告根节点必须是对象。")
    _validate_unicode(payload)
    try:
        payload_bytes = _canonical_report_bytes(payload)
    except (TypeError, ValueError) as error:
        raise HistoryValidationError("历史报告不能严格序列化为 JSON。") from error
    if len(payload_bytes) > MAX_HISTORY_REPORT_BYTES:
        raise HistoryValidationError(
            f"单份历史报告不能超过 {MAX_HISTORY_REPORT_BYTES // (1024 * 1024)} MiB。"
        )

    errors = sorted(
        _report_validator().iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        messages: list[str] = []
        for error in errors[:8]:
            path = ".".join(str(part) for part in error.absolute_path) or "root"
            message = " ".join(error.message.split())
            messages.append(f"{path}: {message[:300]}")
        suffix = "；其余错误已省略" if len(errors) > 8 else ""
        raise HistoryValidationError(
            "历史报告不符合 QualityReport Schema："
            + "；".join(messages)
            + suffix
        )

    context = payload["evaluation_context"]
    reported_hash = context.get("report_sha256")
    recomputed_hash = _recompute_report_sha256(payload)
    if (
        not isinstance(reported_hash, str)
        or not _HASH_PATTERN.fullmatch(reported_hash)
        or reported_hash != recomputed_hash
    ):
        raise HistoryValidationError(
            "历史报告哈希校验失败，报告可能被修改或未完整导出。"
        )
    _validate_report_invariants(payload)
    return deepcopy(payload)


def parse_quality_report_json(content: bytes) -> dict[str, Any]:
    """严格读取用户导入的 UTF-8 QualityReport JSON。"""

    if not isinstance(content, bytes):
        raise HistoryValidationError("历史报告导入内容必须是字节。")
    if len(content) > MAX_HISTORY_REPORT_BYTES:
        raise HistoryValidationError(
            f"导入的历史报告不能超过 {MAX_HISTORY_REPORT_BYTES // (1024 * 1024)} MiB。"
        )
    try:
        text = content.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise HistoryValidationError("历史报告必须使用 UTF-8 编码。") from error
    _scan_json_depth(text)
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except HistoryValidationError:
        raise
    except (RecursionError, TypeError, ValueError) as error:
        raise HistoryValidationError("历史报告不是合法的严格 JSON。") from error
    if not isinstance(payload, Mapping):
        raise HistoryValidationError("历史报告 JSON 根节点必须是对象。")
    return validate_quality_report_payload(payload)


def _history_entry(
    report: QualityReport | Mapping[str, Any],
    *,
    version_label: str,
    dataset_series_id: str,
    saved_at: datetime | str | None,
) -> HistoryEntry:
    payload = validate_quality_report_payload(report)
    label = _strict_text(version_label, "版本标签", maximum=80)
    series_id = _strict_text(dataset_series_id, "治理对象标识", maximum=120)
    saved_at_text = _utc_text(saved_at)
    report_sha256 = payload["evaluation_context"]["report_sha256"]
    entry_digest = hashlib.sha256(
        json.dumps(
            [series_id, report_sha256],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    ).hexdigest()
    size_bytes = len(_canonical_report_bytes(payload))
    return HistoryEntry(
        entry_id=f"history-{entry_digest}",
        version_label=label,
        dataset_series_id=series_id,
        saved_at=saved_at_text,
        report_sha256=report_sha256,
        size_bytes=size_bytes,
        _report_payload=payload,
    )


class InMemoryReportHistoryStore:
    """带容量边界的当前会话内存历史仓库。"""

    def __init__(
        self,
        entries: Sequence[HistoryEntry] = (),
        *,
        policy: HistoryPolicy = DEFAULT_HISTORY_POLICY,
    ) -> None:
        self.policy = policy
        self._entries: list[HistoryEntry] = list(entries)
        self._validate_capacity()

    def _validate_capacity(self) -> None:
        if len(self._entries) > self.policy.max_reports:
            raise HistoryValidationError(
                f"历史报告最多保存 {self.policy.max_reports} 份。"
            )
        total_size = sum(entry.size_bytes for entry in self._entries)
        if any(
            entry.size_bytes > self.policy.max_report_bytes
            for entry in self._entries
        ):
            raise HistoryValidationError(
                "单份历史报告超过当前会话策略上限。"
            )
        if total_size > self.policy.max_total_bytes:
            raise HistoryValidationError(
                "历史报告总容量超过当前会话的安全上限。"
            )
        entry_ids = [entry.entry_id for entry in self._entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise HistoryValidationError("会话历史包含重复报告。")

    def list_entries(self) -> tuple[HistoryEntry, ...]:
        return tuple(self._entries)

    def get(self, entry_id: str) -> HistoryEntry | None:
        return next(
            (entry for entry in self._entries if entry.entry_id == entry_id),
            None,
        )

    def add_report(
        self,
        report: QualityReport | Mapping[str, Any],
        *,
        version_label: str,
        dataset_series_id: str,
        saved_at: datetime | str | None = None,
    ) -> HistoryEntry:
        entry = _history_entry(
            report,
            version_label=version_label,
            dataset_series_id=dataset_series_id,
            saved_at=saved_at,
        )
        if self.get(entry.entry_id) is not None:
            raise HistoryValidationError(
                "当前治理对象下已保存相同报告哈希的历史版本。"
            )
        self._entries.append(entry)
        try:
            self._validate_capacity()
        except Exception:
            self._entries.pop()
            raise
        return entry

    def import_json(
        self,
        content: bytes,
        *,
        version_label: str,
        dataset_series_id: str,
        saved_at: datetime | str | None = None,
    ) -> HistoryEntry:
        payload = parse_quality_report_json(content)
        return self.add_report(
            payload,
            version_label=version_label,
            dataset_series_id=dataset_series_id,
            saved_at=saved_at,
        )

    def delete(self, entry_id: str) -> bool:
        for index, entry in enumerate(self._entries):
            if entry.entry_id == entry_id:
                del self._entries[index]
                return True
        return False

    def clear(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count


def build_version_trend(
    entries: Sequence[HistoryEntry],
    *,
    dataset_series_id: str,
) -> list[dict[str, Any]]:
    """按保存顺序生成不含原始值的版本趋势表。"""

    series_id = _strict_text(
        dataset_series_id,
        "治理对象标识",
        maximum=120,
    )
    return [
        entry.to_summary_dict()
        for entry in entries
        if entry.dataset_series_id == series_id
    ]
