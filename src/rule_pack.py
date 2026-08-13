"""v0.9 RulePack 协议、引导建议、确定性校验与本地审批。

RulePack 与 ``QualityReport``、``AgentAnalysis`` 相互独立。模型或页面可以
提出草案，但只有本模块生成的审批记录、且仍绑定当前报告与输入的规则包才可
交给规则引擎执行。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
from typing import Any, Literal, Mapping, Sequence

from .metric_catalog import get_metric_definition


RULE_PACK_SCHEMA_VERSION = "0.1"
RULE_PACK_GENERATOR = "quality-rule-agent-v0.4"
MAX_RULES = 100
MAX_PRIMARY_KEY_FIELDS = 5
MAX_ALLOWED_VALUES = 100
MAX_RULE_NUMBER_ABS = 10**308
MAX_REGEX_PATTERN_LENGTH = 200
MAX_STRING_LENGTH = 10_000
MAX_CONDITION_VALUES = 100

RuleType = Literal[
    "primary_key",
    "required",
    "update_freshness",
    "allowed_values",
    "numeric_range",
    "regex_format",
    "string_length",
    "conditional_required",
    "field_comparison",
]
RulePackSourceType = Literal[
    "local_guided",
    "user_natural_language",
    "standard_retrieval",
]
RulePackStatus = Literal["draft", "approved"]
JsonScalar = str | int | float | bool

SUPPORTED_RULE_TYPES: frozenset[str] = frozenset(
    {
        "primary_key",
        "required",
        "update_freshness",
        "allowed_values",
        "numeric_range",
        "regex_format",
        "string_length",
        "conditional_required",
        "field_comparison",
    }
)
SUPPORTED_FREQUENCIES: frozenset[str] = frozenset(
    {"daily", "weekly", "monthly", "quarterly", "yearly", "custom"}
)
SUPPORTED_COMPARISON_OPERATORS: frozenset[str] = frozenset(
    {"lt", "lte", "gt", "gte", "eq", "neq"}
)
SUPPORTED_COMPARISON_TYPES: frozenset[str] = frozenset(
    {"auto", "numeric", "datetime", "text"}
)

_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
_RULE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_IDENTIFIER_FIELD_PATTERN = re.compile(
    r"^(?:id|uuid|index|record_?id|row_?id|.*_id|序号|索引|"
    r".*(?:编号|编码|标识|标识码))$",
    re.IGNORECASE,
)


class RulePackValidationError(ValueError):
    """RulePack 草案未通过确定性校验。"""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(str(error) for error in errors)
        super().__init__("；".join(self.errors) or "RulePack 校验失败。")


@dataclass(frozen=True)
class Rule:
    """白名单业务规则；未使用的参数必须保持为空。"""

    type: RuleType
    rule_id: str
    fields: tuple[str, ...]
    frequency: str | None = None
    max_age_days: int | None = None
    allowed_values: tuple[JsonScalar, ...] = ()
    minimum: int | float | None = None
    maximum: int | float | None = None
    regex_pattern: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    condition_field: str | None = None
    condition_values: tuple[JsonScalar, ...] = ()
    comparison_operator: str | None = None
    comparison_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "rule_id": self.rule_id,
            "fields": list(self.fields),
            "frequency": self.frequency,
            "max_age_days": self.max_age_days,
            "allowed_values": list(self.allowed_values),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "regex_pattern": self.regex_pattern,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "condition_field": self.condition_field,
            "condition_values": list(self.condition_values),
            "comparison_operator": self.comparison_operator,
            "comparison_type": self.comparison_type,
        }


@dataclass(frozen=True)
class FieldSemanticMapping:
    """由规则确定性派生的字段语义，不接受独立漂移。"""

    primary_key_fields: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = ()
    update_time_field: str | None = None
    categorical_fields: tuple[str, ...] = ()
    numeric_fields: tuple[str, ...] = ()
    formatted_fields: tuple[str, ...] = ()
    length_checked_fields: tuple[str, ...] = ()
    conditional_required_fields: tuple[str, ...] = ()
    compared_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_key_fields": list(self.primary_key_fields),
            "required_fields": list(self.required_fields),
            "update_time_field": self.update_time_field,
            "categorical_fields": list(self.categorical_fields),
            "numeric_fields": list(self.numeric_fields),
            "formatted_fields": list(self.formatted_fields),
            "length_checked_fields": list(self.length_checked_fields),
            "conditional_required_fields": list(self.conditional_required_fields),
            "compared_fields": list(self.compared_fields),
        }


@dataclass(frozen=True)
class RulePackSource:
    """规则草案来源；保留本地引导并支持 v0.7/v0.8 自然语言编制。"""

    type: RulePackSourceType
    generator: str
    generated_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "type": self.type,
            "generator": self.generator,
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True)
class RuleMetricTarget:
    """把一条已审批规则绑定到一个需补充依据的目录指标。"""

    rule_id: str
    target_metric_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "target_metric_id": self.target_metric_id,
        }


@dataclass(frozen=True)
class ApprovalRecord:
    """本地生成的审批记录；审批人身份仅为自声明。"""

    approval_id: str
    approver: str
    approved_at: str
    draft_sha256: str
    base_report_sha256: str
    base_input_sha256: str
    reference_date: str
    statement: str
    identity_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "approver": self.approver,
            "approved_at": self.approved_at,
            "draft_sha256": self.draft_sha256,
            "base_report_sha256": self.base_report_sha256,
            "base_input_sha256": self.base_input_sha256,
            "reference_date": self.reference_date,
            "statement": self.statement,
            "identity_verified": self.identity_verified,
        }


@dataclass(frozen=True)
class RulePack:
    """版本化、报告绑定且可审批的业务规则包。"""

    rule_pack_id: str
    name: str
    version: str
    status: RulePackStatus
    base_report_sha256: str
    base_input_sha256: str
    reference_date: str
    source: RulePackSource
    field_semantics: FieldSemanticMapping
    rules: tuple[Rule, ...]
    metric_targets: tuple[RuleMetricTarget, ...] = ()
    approval: ApprovalRecord | None = None
    schema_version: str = RULE_PACK_SCHEMA_VERSION
    evidence: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "rule_pack_id": self.rule_pack_id,
            "name": self.name,
            "version": self.version,
            "status": self.status,
            "base_report_sha256": self.base_report_sha256,
            "base_input_sha256": self.base_input_sha256,
            "reference_date": self.reference_date,
            "source": self.source.to_dict(),
            "field_semantics": self.field_semantics.to_dict(),
            "rules": [rule.to_dict() for rule in self.rules],
            "metric_targets": [item.to_dict() for item in self.metric_targets],
            "evidence": [dict(item) for item in self.evidence],
            "approval": (
                self.approval.to_dict()
                if self.approval is not None
                else None
            ),
        }
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return payload


@dataclass(frozen=True)
class RulePackValidationResult:
    """可安全展示或作为纯校验工具返回的结果。"""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    draft_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "draft_sha256": self.draft_sha256,
        }


@dataclass(frozen=True)
class RuleGuidance:
    """只使用报告画像生成的本地引导，不包含字段原值。"""

    report_sha256: str
    primary_key_candidates: tuple[str, ...]
    required_field_candidates: tuple[str, ...]
    update_time_candidates: tuple[str, ...]
    numeric_field_candidates: tuple[str, ...]
    questions: tuple[str, ...]
    source: str = RULE_PACK_GENERATOR

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_sha256": self.report_sha256,
            "primary_key_candidates": list(self.primary_key_candidates),
            "required_field_candidates": list(
                self.required_field_candidates
            ),
            "update_time_candidates": list(self.update_time_candidates),
            "numeric_field_candidates": list(self.numeric_field_candidates),
            "questions": list(self.questions),
            "source": self.source,
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8", errors="strict")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _draft_payload(pack: RulePack) -> dict[str, Any]:
    """审批哈希排除审批状态与记录，只覆盖实际草案内容和绑定。"""

    return {
        "schema_version": pack.schema_version,
        "name": pack.name,
        "version": pack.version,
        "base_report_sha256": pack.base_report_sha256,
        "base_input_sha256": pack.base_input_sha256,
        "reference_date": pack.reference_date,
        "source": pack.source.to_dict(),
        "field_semantics": pack.field_semantics.to_dict(),
        "rules": [rule.to_dict() for rule in pack.rules],
        "metric_targets": [
            item.to_dict() if isinstance(item, RuleMetricTarget) else item
            for item in pack.metric_targets
        ],
        "evidence": [dict(item) for item in pack.evidence],
    }


def draft_sha256(pack: RulePack) -> str:
    """返回规范化草案哈希；不可 JSON 序列化的草案会明确失败。"""

    if not isinstance(pack, RulePack):
        raise TypeError("pack 必须是 RulePack。")
    return _sha256(_draft_payload(pack))


def _rule_pack_id(pack: RulePack) -> str:
    return f"rulepack-{draft_sha256(pack)[:20]}"


def _format_utc(value: datetime | str | None) -> str:
    if value is None:
        timestamp = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            timestamp = datetime.fromisoformat(text)
        except ValueError as error:
            raise ValueError("时间必须是合法 ISO 8601 日期时间。") from error
    else:
        raise TypeError("时间必须是 datetime、ISO 8601 字符串或 None。")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.isoformat(timespec="seconds").replace("+00:00", "Z")


def _is_valid_iso_utc(value: Any) -> bool:
    return _parse_iso_utc(value) is not None


def _parse_iso_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        return datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError:
        return None


def _valid_text(value: Any, *, maximum: int, allow_blank: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    if not allow_blank and not value.strip():
        return False
    if len(value) > maximum:
        return False
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return False
    return True


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) <= MAX_RULE_NUMBER_ABS
    if not isinstance(value, float):
        return False
    return math.isfinite(value) and abs(value) <= MAX_RULE_NUMBER_ABS


def _number_decimal(value: int | float) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("数值边界无法表示为有限十进制数。") from error


def _regex_errors(pattern: Any) -> list[str]:
    """校验受限正则，避免把 DSL 变成无界表达式执行入口。"""

    errors: list[str] = []
    if not isinstance(pattern, str) or not pattern:
        return ["正则规则必须包含非空 pattern。"]
    if len(pattern) > MAX_REGEX_PATTERN_LENGTH:
        errors.append(
            f"正则 pattern 不能超过 {MAX_REGEX_PATTERN_LENGTH} 个字符。"
        )
    try:
        pattern.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        errors.append("正则 pattern 包含无法编码的字符。")
    # v0.8 只开放常见字符类、分组和量词；拒绝会扩大执行语义或增加回溯
    # 风险的 lookaround、条件分支、反向引用和命名组。
    if re.search(r"\(\?|\\(?:[1-9][0-9]*|g<|k<)", pattern):
        errors.append("正则 pattern 不支持 lookaround、条件分支、命名组或反向引用。")
    if re.search(r"\([^)]*[+*][^)]*\)[+*]", pattern):
        errors.append("正则 pattern 不支持可能造成大量回溯的嵌套量词。")
    try:
        re.compile(pattern)
    except re.error as error:
        errors.append(f"正则 pattern 无法编译：{error}。")
    return errors


def _scalar_key(value: Any) -> tuple[str, Any]:
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, str):
        return ("string", value)
    if _finite_number(value):
        return ("number", _number_decimal(value).normalize())
    raise ValueError("允许值只能是有限 JSON 字符串、数字或布尔值。")


def _mapping_from_rules(rules: Sequence[Rule]) -> FieldSemanticMapping:
    primary_key_fields: tuple[str, ...] = ()
    required_fields: list[str] = []
    update_time_field: str | None = None
    categorical_fields: list[str] = []
    numeric_fields: list[str] = []
    formatted_fields: list[str] = []
    length_checked_fields: list[str] = []
    conditional_required_fields: list[str] = []
    compared_fields: list[str] = []
    for rule in rules:
        if rule.type == "primary_key":
            primary_key_fields = tuple(rule.fields)
        elif rule.type == "required" and rule.fields:
            required_fields.append(rule.fields[0])
        elif rule.type == "update_freshness" and rule.fields:
            update_time_field = rule.fields[0]
        elif rule.type == "allowed_values" and rule.fields:
            categorical_fields.append(rule.fields[0])
        elif rule.type == "numeric_range" and rule.fields:
            numeric_fields.append(rule.fields[0])
        elif rule.type == "regex_format" and rule.fields:
            formatted_fields.append(rule.fields[0])
        elif rule.type == "string_length" and rule.fields:
            length_checked_fields.append(rule.fields[0])
        elif rule.type == "conditional_required" and len(rule.fields) == 2:
            conditional_required_fields.append(rule.fields[1])
        elif rule.type == "field_comparison" and rule.fields:
            compared_fields.extend(rule.fields)
    return FieldSemanticMapping(
        primary_key_fields=primary_key_fields,
        required_fields=tuple(dict.fromkeys(required_fields)),
        update_time_field=update_time_field,
        categorical_fields=tuple(dict.fromkeys(categorical_fields)),
        numeric_fields=tuple(dict.fromkeys(numeric_fields)),
        formatted_fields=tuple(dict.fromkeys(formatted_fields)),
        length_checked_fields=tuple(dict.fromkeys(length_checked_fields)),
        conditional_required_fields=tuple(
            dict.fromkeys(conditional_required_fields)
        ),
        compared_fields=tuple(dict.fromkeys(compared_fields)),
    )


def _report_payload(report: Any) -> Mapping[str, Any] | None:
    to_dict = getattr(report, "to_dict", None)
    if not callable(to_dict):
        return None
    try:
        payload = to_dict()
    except Exception:
        return None
    return payload if isinstance(payload, Mapping) else None


def _profile_columns(report_payload: Mapping[str, Any]) -> tuple[
    tuple[str, ...],
    dict[str, str],
    Mapping[str, Any],
]:
    profile = report_payload.get("profile")
    if not isinstance(profile, Mapping):
        return (), {}, {}
    raw_columns = profile.get("columns")
    if not isinstance(raw_columns, list):
        return (), {}, profile
    fields: list[str] = []
    inferred_types: dict[str, str] = {}
    for column in raw_columns:
        if not isinstance(column, Mapping):
            continue
        field = column.get("name")
        if not isinstance(field, str):
            continue
        fields.append(field)
        inferred_type = column.get("inferred_type")
        if isinstance(inferred_type, str):
            inferred_types[field] = inferred_type
    return tuple(fields), inferred_types, profile


def _rule_errors(
    rule: Any,
    *,
    field_set: frozenset[str],
    inferred_types: Mapping[str, str],
    recognized_date_fields: frozenset[str],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(rule, Rule):
        return ["规则列表只能包含 Rule 对象。"]
    if rule.type not in SUPPORTED_RULE_TYPES:
        errors.append(f"规则 {rule.rule_id!r} 的类型不在白名单中。")
    if not _valid_text(rule.rule_id, maximum=80) or not _RULE_ID_PATTERN.fullmatch(
        rule.rule_id
    ):
        errors.append("rule_id 必须为 1 到 80 个安全字符。")
    if not isinstance(rule.fields, tuple):
        errors.append(f"规则 {rule.rule_id!r} 的 fields 必须是元组。")
        return errors
    if not rule.fields:
        errors.append(f"规则 {rule.rule_id!r} 必须指定字段。")
    if len(set(rule.fields)) != len(rule.fields):
        errors.append(f"规则 {rule.rule_id!r} 不能重复引用同一字段。")
    for field in rule.fields:
        if not _valid_text(field, maximum=1000):
            errors.append(f"规则 {rule.rule_id!r} 包含非法或过长字段名。")
        elif field not in field_set:
            errors.append(f"规则 {rule.rule_id!r} 引用了不存在的字段“{field}”。")

    unused_parameters = (
        rule.frequency is not None
        or rule.max_age_days is not None
        or bool(rule.allowed_values)
        or rule.minimum is not None
        or rule.maximum is not None
        or rule.regex_pattern is not None
        or rule.min_length is not None
        or rule.max_length is not None
        or rule.condition_field is not None
        or bool(rule.condition_values)
        or rule.comparison_operator is not None
        or rule.comparison_type is not None
    )
    if rule.type == "primary_key":
        if not 1 <= len(rule.fields) <= MAX_PRIMARY_KEY_FIELDS:
            errors.append(
                f"主键规则必须包含 1 到 {MAX_PRIMARY_KEY_FIELDS} 个字段。"
            )
        if unused_parameters:
            errors.append("主键规则不能包含频率、允许值或数值范围参数。")
    elif rule.type == "required":
        if len(rule.fields) != 1:
            errors.append("每条必填规则必须且只能包含一个字段。")
        if unused_parameters:
            errors.append("必填规则不能包含频率、允许值或数值范围参数。")
    elif rule.type == "update_freshness":
        if len(rule.fields) != 1:
            errors.append("更新时间规则必须且只能包含一个字段。")
        if rule.frequency not in SUPPORTED_FREQUENCIES:
            errors.append("更新时间规则的 frequency 不在白名单中。")
        if (
            isinstance(rule.max_age_days, bool)
            or not isinstance(rule.max_age_days, int)
            or not 1 <= rule.max_age_days <= 3660
        ):
            errors.append("更新时间规则的 max_age_days 必须为 1 到 3660。")
        if rule.allowed_values or rule.minimum is not None or rule.maximum is not None:
            errors.append("更新时间规则不能包含允许值或数值范围参数。")
        if rule.fields:
            field = rule.fields[0]
            if (
                field in field_set
                and inferred_types.get(field) != "datetime"
                and field not in recognized_date_fields
            ):
                errors.append(
                    f"字段“{field}”未被画像识别为日期时间字段。"
                )
    elif rule.type == "allowed_values":
        if len(rule.fields) != 1:
            errors.append("每条允许值规则必须且只能包含一个字段。")
        if not 1 <= len(rule.allowed_values) <= MAX_ALLOWED_VALUES:
            errors.append(
                f"允许值规则必须包含 1 到 {MAX_ALLOWED_VALUES} 个值。"
            )
        seen_values: set[tuple[str, Any]] = set()
        for value in rule.allowed_values:
            try:
                key = _scalar_key(value)
            except ValueError as error:
                errors.append(str(error))
                continue
            if isinstance(value, str) and not _valid_text(
                value,
                maximum=500,
                allow_blank=False,
            ):
                errors.append("允许值字符串必须为 1 到 500 个有效 Unicode 字符。")
            if key in seen_values:
                errors.append("允许值列表包含类型语义相同的重复值。")
            seen_values.add(key)
        if (
            rule.frequency is not None
            or rule.max_age_days is not None
            or rule.minimum is not None
            or rule.maximum is not None
        ):
            errors.append("允许值规则不能包含频率或数值范围参数。")
    elif rule.type == "numeric_range":
        if len(rule.fields) != 1:
            errors.append("每条数值范围规则必须且只能包含一个字段。")
        if rule.minimum is None and rule.maximum is None:
            errors.append("数值范围规则至少需要一个有限边界。")
        for label, value in (
            ("minimum", rule.minimum),
            ("maximum", rule.maximum),
        ):
            if value is not None and not _finite_number(value):
                errors.append(f"数值范围规则的 {label} 必须为有限数。")
        if _finite_number(rule.minimum) and _finite_number(rule.maximum):
            if _number_decimal(rule.minimum) > _number_decimal(rule.maximum):
                errors.append("数值范围规则的 minimum 不能大于 maximum。")
        if rule.frequency is not None or rule.max_age_days is not None or rule.allowed_values:
            errors.append("数值范围规则不能包含频率或允许值参数。")
        if rule.fields:
            field = rule.fields[0]
            if field in field_set and inferred_types.get(field) != "numeric":
                errors.append(f"字段“{field}”未被画像识别为数值字段。")
    elif rule.type == "regex_format":
        if len(rule.fields) != 1:
            errors.append("每条格式规则必须且只能包含一个字段。")
        errors.extend(_regex_errors(rule.regex_pattern))
        if any(
            value is not None
            for value in (
                rule.frequency,
                rule.max_age_days,
                rule.minimum,
                rule.maximum,
                rule.min_length,
                rule.max_length,
                rule.condition_field,
                rule.comparison_operator,
                rule.comparison_type,
            )
        ) or rule.allowed_values or rule.condition_values:
            errors.append("格式规则不能包含其他规则参数。")
    elif rule.type == "string_length":
        if len(rule.fields) != 1:
            errors.append("每条字符长度规则必须且只能包含一个字段。")
        if rule.min_length is None and rule.max_length is None:
            errors.append("字符长度规则至少需要一个长度边界。")
        for label, value in (
            ("min_length", rule.min_length),
            ("max_length", rule.max_length),
        ):
            if (
                value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value <= MAX_STRING_LENGTH
                )
            ):
                errors.append(
                    f"字符长度规则的 {label} 必须为 0 到 {MAX_STRING_LENGTH} 的整数。"
                )
        if (
            isinstance(rule.min_length, int)
            and not isinstance(rule.min_length, bool)
            and isinstance(rule.max_length, int)
            and not isinstance(rule.max_length, bool)
            and rule.min_length > rule.max_length
        ):
            errors.append("字符长度规则的 min_length 不能大于 max_length。")
        if any(
            value is not None
            for value in (
                rule.frequency,
                rule.max_age_days,
                rule.minimum,
                rule.maximum,
                rule.regex_pattern,
                rule.condition_field,
                rule.comparison_operator,
                rule.comparison_type,
            )
        ) or rule.allowed_values or rule.condition_values:
            errors.append("字符长度规则不能包含其他规则参数。")
    elif rule.type == "conditional_required":
        if len(rule.fields) != 2:
            errors.append("条件必填规则必须包含条件字段和被要求字段。")
        if rule.condition_field != (rule.fields[0] if rule.fields else None):
            errors.append("条件必填规则的 condition_field 必须等于第一个字段。")
        if not 1 <= len(rule.condition_values) <= MAX_CONDITION_VALUES:
            errors.append(
                f"条件必填规则必须包含 1 到 {MAX_CONDITION_VALUES} 个条件值。"
            )
        seen_values: set[tuple[str, Any]] = set()
        for value in rule.condition_values:
            try:
                key = _scalar_key(value)
            except ValueError as error:
                errors.append(str(error))
                continue
            if key in seen_values:
                errors.append("条件值列表包含类型语义相同的重复值。")
            seen_values.add(key)
        if any(
            value is not None
            for value in (
                rule.frequency,
                rule.max_age_days,
                rule.minimum,
                rule.maximum,
                rule.regex_pattern,
                rule.min_length,
                rule.max_length,
                rule.comparison_operator,
                rule.comparison_type,
            )
        ) or rule.allowed_values:
            errors.append("条件必填规则不能包含其他规则参数。")
    elif rule.type == "field_comparison":
        if len(rule.fields) != 2:
            errors.append("跨字段比较规则必须包含左侧和右侧两个字段。")
        if rule.comparison_operator not in SUPPORTED_COMPARISON_OPERATORS:
            errors.append("跨字段比较规则的 comparison_operator 不在白名单中。")
        if rule.comparison_type not in SUPPORTED_COMPARISON_TYPES:
            errors.append("跨字段比较规则的 comparison_type 不在白名单中。")
        if any(
            value is not None
            for value in (
                rule.frequency,
                rule.max_age_days,
                rule.minimum,
                rule.maximum,
                rule.regex_pattern,
                rule.min_length,
                rule.max_length,
                rule.condition_field,
            )
        ) or rule.allowed_values or rule.condition_values:
            errors.append("跨字段比较规则不能包含其他规则参数。")
        if len(rule.fields) == 2 and rule.comparison_type in {"numeric", "datetime"}:
            expected_type = rule.comparison_type
            for field in rule.fields:
                if field in field_set and inferred_types.get(field) != expected_type:
                    errors.append(
                        f"字段“{field}”未被画像识别为 {expected_type} 字段。"
                    )
    return errors


def validate_rule_pack(
    pack: Any,
    report: Any,
    *,
    require_approved: bool = False,
) -> RulePackValidationResult:
    """以当前固定报告确定性校验 RulePack，不启用或执行任何规则。"""

    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(pack, RulePack):
        return RulePackValidationResult(
            valid=False,
            errors=("待校验对象不是 RulePack。",),
            warnings=(),
            draft_sha256=None,
        )

    payload = _report_payload(report)
    if payload is None:
        errors.append("当前报告不提供合法的 to_dict()。")
        report_context: Mapping[str, Any] = {}
        report_status = None
    else:
        context = payload.get("evaluation_context")
        report_context = context if isinstance(context, Mapping) else {}
        report_status = payload.get("status")
    if report_status != "success":
        errors.append("只有成功的零配置报告可以生成或执行 RulePack。")

    if not isinstance(pack.evidence, tuple):
        errors.append("RulePack evidence 必须是元组。")
        evidence_items: tuple[Any, ...] = ()
    else:
        evidence_items = pack.evidence
    if len(evidence_items) > 20:
        errors.append("RulePack evidence 最多包含 20 条依据。")
    for index, item in enumerate(evidence_items):
        if not isinstance(item, Mapping):
            errors.append(f"RulePack evidence[{index}] 必须是对象。")
            continue
        evidence_type = item.get("type")
        if evidence_type not in {
            "user_statement",
            "metric_definition",
            "standard_clause",
            "data_dictionary",
            "system_inference",
        }:
            errors.append(f"RulePack evidence[{index}] 类型不受支持。")
        try:
            json.dumps(dict(item), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            errors.append(f"RulePack evidence[{index}] 不能安全序列化。")

    try:
        current_draft_hash = draft_sha256(pack)
    except (TypeError, ValueError, UnicodeError):
        current_draft_hash = None
        errors.append("RulePack 包含不可安全序列化的值。")

    if pack.schema_version != RULE_PACK_SCHEMA_VERSION:
        errors.append("RulePack schema_version 不受当前版本支持。")
    if not _valid_text(pack.name, maximum=120):
        errors.append("RulePack 名称必须为 1 到 120 个有效 Unicode 字符。")
    if not _valid_text(pack.version, maximum=32) or not _VERSION_PATTERN.fullmatch(
        pack.version
    ):
        errors.append("RulePack 版本必须为 1 到 32 个安全字符。")
    if pack.status not in {"draft", "approved"}:
        errors.append("RulePack 状态只能是 draft 或 approved。")
    if current_draft_hash is not None:
        expected_rule_pack_id = f"rulepack-{current_draft_hash[:20]}"
        if pack.rule_pack_id != expected_rule_pack_id:
            errors.append("RulePack ID 与当前草案内容不匹配。")

    current_report_hash = report_context.get("report_sha256")
    current_input_hash = report_context.get("input_sha256")
    current_reference_date = report_context.get("reference_date")
    if (
        not isinstance(current_report_hash, str)
        or len(current_report_hash) != 64
        or pack.base_report_sha256 != current_report_hash
    ):
        errors.append("RulePack 与当前零配置报告哈希不匹配。")
    if (
        not isinstance(current_input_hash, str)
        or len(current_input_hash) != 64
        or pack.base_input_sha256 != current_input_hash
    ):
        errors.append("RulePack 与当前输入哈希不匹配。")
    if (
        not isinstance(current_reference_date, str)
        or pack.reference_date != current_reference_date
    ):
        errors.append("RulePack 与当前评估基准日期不匹配。")

    if not isinstance(pack.source, RulePackSource):
        errors.append("RulePack source 结构无效。")
    else:
        expected_generators = {
            "local_guided": {RULE_PACK_GENERATOR},
            # v0.7 草案必须继续可读取；v0.8 新建草案使用 v0.8 生成器。
            "user_natural_language": {
                "quality-rule-agent-v0.7",
                "quality-rule-agent-v0.8",
            },
            "standard_retrieval": {"quality-rule-agent-v0.9"},
        }
        if pack.source.type not in expected_generators:
            errors.append("RulePack source.type 不在当前白名单中。")
        elif pack.source.generator not in expected_generators[pack.source.type]:
            errors.append("RulePack source.generator 与来源类型不匹配。")
        if not _is_valid_iso_utc(pack.source.generated_at):
            errors.append("RulePack source.generated_at 必须是 UTC 时间。")
        if pack.source.type == "standard_retrieval" and not any(
            isinstance(item, Mapping)
            and (
                item.get("type") in {"standard_clause", "data_dictionary"}
                or (
                    item.get("type") == "user_statement"
                    and item.get("chunk_id")
                    and item.get("document_name")
                    and item.get("document_version")
                )
            )
            for item in evidence_items
        ):
            errors.append("standard_retrieval RulePack 必须绑定至少一条可定位来源。")

    fields, inferred_types, profile = _profile_columns(payload or {})
    field_set = frozenset(fields)
    if len(field_set) != len(fields):
        errors.append("当前报告画像包含重复字段，不能创建 RulePack。")
    recognized_fields = profile.get("recognized_fields")
    recognized_date_fields: frozenset[str] = frozenset()
    if isinstance(recognized_fields, Mapping):
        raw_date_fields = recognized_fields.get("date")
        if isinstance(raw_date_fields, list):
            recognized_date_fields = frozenset(
                field for field in raw_date_fields if isinstance(field, str)
            )

    if not isinstance(pack.rules, tuple):
        errors.append("RulePack rules 必须是元组。")
        rules: tuple[Any, ...] = ()
    else:
        rules = pack.rules
    if not 1 <= len(rules) <= MAX_RULES:
        errors.append(f"RulePack 必须包含 1 到 {MAX_RULES} 条规则。")
    for rule in rules:
        errors.extend(
            _rule_errors(
                rule,
                field_set=field_set,
                inferred_types=inferred_types,
                recognized_date_fields=recognized_date_fields,
            )
        )

    typed_rules = tuple(rule for rule in rules if isinstance(rule, Rule))
    rule_ids = [rule.rule_id for rule in typed_rules]
    if len(rule_ids) != len(set(rule_ids)):
        errors.append("RulePack 中的 rule_id 必须唯一。")
    if sum(rule.type == "primary_key" for rule in typed_rules) > 1:
        errors.append("RulePack 最多只能包含一条主键规则。")
    if sum(rule.type == "update_freshness" for rule in typed_rules) > 1:
        errors.append("RulePack 最多只能包含一条更新时间规则。")
    signatures = [
        (rule.type, tuple(rule.fields))
        for rule in typed_rules
    ]
    if len(signatures) != len(set(signatures)):
        errors.append("RulePack 不能包含同类型、同字段的重复规则。")

    if not isinstance(pack.metric_targets, tuple):
        errors.append("RulePack metric_targets 必须是元组。")
        metric_targets: tuple[Any, ...] = ()
    else:
        metric_targets = pack.metric_targets
    if len(metric_targets) > len(rules):
        errors.append("RulePack 指标目标数量不能超过规则数量。")
    target_rule_ids: list[str] = []
    target_metric_ids: list[str] = []
    baseline_metrics = payload.get("metrics") if isinstance(payload, Mapping) else None
    baseline_by_id: dict[str, list[Mapping[str, Any]]] = {}
    if isinstance(baseline_metrics, list):
        for metric in baseline_metrics:
            if isinstance(metric, Mapping) and isinstance(metric.get("id"), str):
                baseline_by_id.setdefault(str(metric["id"]), []).append(metric)
    selected_metric_ids = report_context.get("selected_metric_ids")
    selected_metric_set = (
        {str(item) for item in selected_metric_ids}
        if isinstance(selected_metric_ids, list)
        else set()
    )
    for index, target in enumerate(metric_targets):
        if not isinstance(target, RuleMetricTarget):
            errors.append(f"RulePack metric_targets[{index}] 结构无效。")
            continue
        target_rule_ids.append(target.rule_id)
        target_metric_ids.append(target.target_metric_id)
        if target.rule_id not in rule_ids:
            errors.append(
                f"指标目标引用了不存在的规则：{target.rule_id}。"
            )
        definition = get_metric_definition(target.target_metric_id)
        if definition is None:
            errors.append(
                f"指标目标引用了未知目录指标：{target.target_metric_id}。"
            )
            continue
        if bool(definition.get("auto_assessable")):
            errors.append(
                f"自动可评估指标不能由补充规则覆盖：{target.target_metric_id}。"
            )
        if target.target_metric_id not in selected_metric_set:
            errors.append(
                f"指标目标未包含在当前评估选择中：{target.target_metric_id}。"
            )
        matching_metrics = baseline_by_id.get(target.target_metric_id, [])
        if len(matching_metrics) != 1:
            errors.append(
                f"当前报告中未唯一找到指标目标：{target.target_metric_id}。"
            )
        elif matching_metrics[0].get("status") != "not_assessable":
            errors.append(
                f"指标目标当前并非需补充依据状态：{target.target_metric_id}。"
            )
    if len(target_rule_ids) != len(set(target_rule_ids)):
        errors.append("同一规则不能绑定多个目录指标目标。")
    if len(target_metric_ids) != len(set(target_metric_ids)):
        errors.append("同一目录指标不能绑定多条规则。")

    expected_mapping = _mapping_from_rules(typed_rules)
    if pack.field_semantics != expected_mapping:
        errors.append("字段语义映射与当前规则不一致。")

    if pack.status == "draft":
        if pack.approval is not None:
            errors.append("draft RulePack 不能携带审批记录。")
    elif pack.approval is None:
        errors.append("approved RulePack 必须携带本地审批记录。")
    else:
        approval = pack.approval
        if not isinstance(approval, ApprovalRecord):
            errors.append("审批记录结构无效。")
        else:
            if approval.identity_verified is not False:
                errors.append("本地 Demo 不能把审批人标记为已验证身份。")
            if not _valid_text(approval.approver, maximum=100):
                errors.append("审批人标识必须为 1 到 100 个有效 Unicode 字符。")
            if not _is_valid_iso_utc(approval.approved_at):
                errors.append("审批时间必须是 UTC 时间。")
            source_timestamp = (
                _parse_iso_utc(pack.source.generated_at)
                if isinstance(pack.source, RulePackSource)
                else None
            )
            approval_timestamp = _parse_iso_utc(approval.approved_at)
            if (
                source_timestamp is not None
                and approval_timestamp is not None
                and approval_timestamp < source_timestamp
            ):
                errors.append("审批时间不能早于规则草案生成时间。")
            if current_draft_hash != approval.draft_sha256:
                errors.append("审批记录绑定的草案哈希已失效。")
            if approval.base_report_sha256 != pack.base_report_sha256:
                errors.append("审批记录绑定的报告哈希不匹配。")
            if approval.base_input_sha256 != pack.base_input_sha256:
                errors.append("审批记录绑定的输入哈希不匹配。")
            if approval.reference_date != pack.reference_date:
                errors.append("审批记录绑定的基准日期不匹配。")
            expected_statement = (
                f"批准启用 {pack.rule_pack_id} v{pack.version}，"
                "仅由确定性 Python 规则重评当前绑定输入。"
            )
            if approval.statement != expected_statement:
                errors.append("审批声明与当前 RulePack 不匹配。")
            approval_payload = {
                "approver": approval.approver,
                "approved_at": approval.approved_at,
                "draft_sha256": approval.draft_sha256,
                "base_report_sha256": approval.base_report_sha256,
                "base_input_sha256": approval.base_input_sha256,
                "reference_date": approval.reference_date,
                "statement": approval.statement,
                "identity_verified": approval.identity_verified,
            }
            expected_approval_id = f"approval-{_sha256(approval_payload)[:20]}"
            if approval.approval_id != expected_approval_id:
                errors.append("审批记录 ID 与审批内容不匹配。")

    if require_approved and pack.status != "approved":
        errors.append("RulePack 尚未获得用户明确批准。")
    if pack.status == "approved":
        warnings.append("审批人身份为本地自声明，系统未进行身份认证。")

    return RulePackValidationResult(
        valid=not errors,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
        draft_sha256=current_draft_hash,
    )


def build_rule_pack(
    report: Any,
    *,
    name: str,
    version: str,
    rules: Sequence[Rule],
    generated_at: datetime | str | None = None,
    source_type: RulePackSourceType = "local_guided",
    generator: str | None = None,
    evidence: Sequence[Mapping[str, Any]] = (),
    metric_targets: Sequence[RuleMetricTarget] = (),
) -> RulePack:
    """基于当前零配置报告创建仍未生效的 RulePack 草案。"""

    payload = _report_payload(report)
    if payload is None:
        raise RulePackValidationError(("当前报告不提供合法的 to_dict()。",))
    context = payload.get("evaluation_context")
    if not isinstance(context, Mapping):
        raise RulePackValidationError(("当前报告缺少 evaluation_context。",))
    allowed_generators = {
        "local_guided": RULE_PACK_GENERATOR,
        "user_natural_language": "quality-rule-agent-v0.7",
        "standard_retrieval": "quality-rule-agent-v0.9",
    }
    if source_type not in allowed_generators:
        raise RulePackValidationError(("RulePack 来源类型不在当前白名单中。",))
    source = RulePackSource(
        type=source_type,
        generator=generator or allowed_generators[source_type],
        generated_at=_format_utc(generated_at),
    )
    typed_rules = tuple(rules)
    pack = RulePack(
        rule_pack_id="",
        name=str(name),
        version=str(version),
        status="draft",
        base_report_sha256=str(context.get("report_sha256") or ""),
        base_input_sha256=str(context.get("input_sha256") or ""),
        reference_date=str(context.get("reference_date") or ""),
        source=source,
        field_semantics=_mapping_from_rules(typed_rules),
        rules=typed_rules,
        metric_targets=tuple(metric_targets),
        approval=None,
        evidence=tuple(dict(item) for item in evidence),
    )
    try:
        pack = replace(pack, rule_pack_id=_rule_pack_id(pack))
    except (TypeError, ValueError, UnicodeError) as error:
        validation = validate_rule_pack(pack, report)
        raise RulePackValidationError(
            validation.errors
            or ("RulePack 包含不可安全序列化的值。",)
        ) from error
    validation = validate_rule_pack(pack, report)
    if not validation.valid:
        raise RulePackValidationError(validation.errors)
    return pack


def approve_rule_pack(
    pack: RulePack,
    report: Any,
    *,
    approver: str,
    approved_at: datetime | str | None = None,
) -> RulePack:
    """本地记录一次明确审批；模型不能构造或跳过本函数。"""

    if not isinstance(pack, RulePack):
        raise TypeError("pack 必须是 RulePack。")
    if pack.status != "draft" or pack.approval is not None:
        raise RulePackValidationError(("只有未审批草案可以进入审批流程。",))
    draft_validation = validate_rule_pack(pack, report)
    if not draft_validation.valid:
        raise RulePackValidationError(draft_validation.errors)
    normalized_approver = str(approver).strip()
    if not _valid_text(normalized_approver, maximum=100):
        raise RulePackValidationError(
            ("审批人标识必须为 1 到 100 个有效 Unicode 字符。",)
        )
    approved_at_text = _format_utc(approved_at)
    current_draft_hash = draft_sha256(pack)
    statement = (
        f"批准启用 {pack.rule_pack_id} v{pack.version}，"
        "仅由确定性 Python 规则重评当前绑定输入。"
    )
    approval_payload = {
        "approver": normalized_approver,
        "approved_at": approved_at_text,
        "draft_sha256": current_draft_hash,
        "base_report_sha256": pack.base_report_sha256,
        "base_input_sha256": pack.base_input_sha256,
        "reference_date": pack.reference_date,
        "statement": statement,
        "identity_verified": False,
    }
    approval = ApprovalRecord(
        approval_id=f"approval-{_sha256(approval_payload)[:20]}",
        **approval_payload,
    )
    approved = replace(pack, status="approved", approval=approval)
    validation = validate_rule_pack(
        approved,
        report,
        require_approved=True,
    )
    if not validation.valid:
        raise RulePackValidationError(validation.errors)
    return approved


def is_rule_pack_executable(pack: RulePack, report: Any) -> bool:
    """只在审批、草案哈希和当前报告三重绑定均有效时返回 True。"""

    return validate_rule_pack(
        pack,
        report,
        require_approved=True,
    ).valid


def validate_rule_pack_tool(pack: RulePack, report: Any) -> dict[str, Any]:
    """无副作用的受控校验工具；不审批，也不执行规则。"""

    return validate_rule_pack(pack, report).to_dict()


def build_rule_guidance(report: Any) -> RuleGuidance:
    """从脱敏字段画像生成候选和五个确认问题，不读取任何记录值。"""

    payload = _report_payload(report)
    if payload is None:
        raise TypeError("report 必须提供 to_dict()。")
    context = payload.get("evaluation_context")
    report_sha256 = (
        context.get("report_sha256")
        if isinstance(context, Mapping)
        else None
    )
    fields, inferred_types, profile = _profile_columns(payload)
    if not isinstance(report_sha256, str) or len(report_sha256) != 64:
        raise ValueError("当前报告缺少稳定报告哈希。")

    primary_key_candidates = tuple(
        field
        for field in fields
        if _IDENTIFIER_FIELD_PATTERN.fullmatch(field.strip())
    )
    recognized_fields = profile.get("recognized_fields")
    date_candidates: list[str] = []
    if isinstance(recognized_fields, Mapping):
        raw_dates = recognized_fields.get("date")
        if isinstance(raw_dates, list):
            date_candidates.extend(
                field
                for field in raw_dates
                if isinstance(field, str) and field in fields
            )
    date_candidates.extend(
        field
        for field in fields
        if inferred_types.get(field) == "datetime"
    )
    update_time_candidates = tuple(dict.fromkeys(date_candidates))
    numeric_field_candidates = tuple(
        field
        for field in fields
        if inferred_types.get(field) == "numeric"
    )
    return RuleGuidance(
        report_sha256=report_sha256,
        primary_key_candidates=primary_key_candidates,
        required_field_candidates=fields,
        update_time_candidates=update_time_candidates,
        numeric_field_candidates=numeric_field_candidates,
        questions=(
            "哪些字段共同组成主键？",
            "哪些字段在业务上必须填写？",
            "哪个字段代表更新时间，允许的最长更新间隔是多少天？",
            "哪些字段只能使用明确的允许值？",
            "哪些数值字段需要闭区间范围约束？",
        ),
    )
