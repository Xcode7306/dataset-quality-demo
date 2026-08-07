"""v0.9 规则编制领域协议。

本模块只保存“用户依据 → 规则草案”的结构化结果，不负责审批或正式执行。
可执行规则复用 ``src.rule_pack.Rule`` 和现有确定性引擎。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any, Literal, Mapping, Sequence

from .rule_pack import Rule, SUPPORTED_RULE_TYPES


RULE_DRAFT_SCHEMA_VERSION = "0.1"
RULE_DRAFT_GENERATOR = "quality-rule-agent-v0.7"
RULE_DRAFT_GENERATOR_V08 = "quality-rule-agent-v0.8"
RULE_DRAFT_GENERATOR_V09 = "quality-rule-agent-v0.9"
MAX_USER_INTENT_LENGTH = 4000
MAX_CLARIFICATION_QUESTIONS = 5
MAX_EVIDENCE_ITEMS = 20

RuleDraftStatus = Literal[
    "collecting",
    "retrieving",
    "needs_clarification",
    "compiling",
    "draft",
    "validated",
    "dry_run_complete",
    "awaiting_approval",
    "approved",
    "executed",
    "rejected",
    "failed",
]
RuleTargetType = Literal["catalog_metric", "custom_rule"]
RuleEvidenceType = Literal[
    "user_statement",
    "metric_definition",
    "standard_clause",
    "data_dictionary",
    "system_inference",
]
RuleAuthoringOutcome = Literal["draft", "clarification", "unsupported"]

SUPPORTED_DRAFT_STATUSES: frozenset[str] = frozenset(
    {
        "collecting",
        "retrieving",
        "needs_clarification",
        "compiling",
        "draft",
        "validated",
        "dry_run_complete",
        "awaiting_approval",
        "approved",
        "executed",
        "rejected",
        "failed",
    }
)
SUPPORTED_EVIDENCE_TYPES: frozenset[str] = frozenset(
    {
        "user_statement",
        "metric_definition",
        "standard_clause",
        "data_dictionary",
        "system_inference",
    }
)
_RULE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{2,79}$")
_DRAFT_ID_PATTERN = re.compile(r"^draft-[a-f0-9]{20}$")
_WORKFLOW_ID_PATTERN = re.compile(r"^workflow-[a-z0-9][a-z0-9._-]{5,79}$")


class RuleDraftValidationError(ValueError):
    """RuleDraft 或模型规则结果不符合 v0.7 协议。"""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(str(error) for error in errors)
        super().__init__("；".join(self.errors) or "RuleDraft 校验失败。")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8", errors="strict")


def _content_id(prefix: str, value: Any, length: int = 20) -> str:
    return f"{prefix}-{hashlib.sha256(_canonical_json(value)).hexdigest()[:length]}"


def make_rule_id(
    rule_type: str,
    fields: Sequence[str],
    parameters: Mapping[str, Any] | None = None,
) -> str:
    """由本地代码依据规则内容生成稳定 ID；模型不能决定最终 ID。"""

    return _content_id(
        rule_type,
        {
            "rule_type": rule_type,
            "fields": list(fields),
            "parameters": dict(parameters or {}),
        },
        length=12,
    )


def make_workflow_id(value: Any) -> str:
    return _content_id("workflow", value)


def make_draft_id(
    workflow_id: str,
    target_type: str,
    target_metric_id: str | None,
    user_intent: str,
    created_at: str,
) -> str:
    return _content_id(
        "draft",
        [workflow_id, target_type, target_metric_id, user_intent, created_at],
    )


def _utc_timestamp(value: datetime | str | None = None) -> str:
    if value is None:
        timestamp = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        timestamp = datetime.fromisoformat(text)
    else:
        raise TypeError("时间必须是 datetime、ISO 8601 字符串或 None。")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True)
class RuleEvidence:
    """规则草案中的一条依据；系统推断不等同于外部标准。"""

    id: str
    type: RuleEvidenceType
    text: str
    source_id: str | None = None
    source_label: str | None = None
    location: str | None = None
    authoritative: bool = False
    document_id: str | None = None
    document_name: str | None = None
    document_version: str | None = None
    section: str | None = None
    clause: str | None = None
    chunk_id: str | None = None
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "source_id": self.source_id,
            "source_label": self.source_label,
            "location": self.location,
            "authoritative": self.authoritative,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_version": self.document_version,
            "section": self.section,
            "clause": self.clause,
            "chunk_id": self.chunk_id,
            "page": self.page,
        }


@dataclass(frozen=True)
class RuleSpec:
    """与 v0.8 白名单 RulePack 规则对应的稳定 DSL。"""

    rule_type: str
    rule_id: str
    name: str
    description: str
    fields: tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    severity: str = "attention"
    denominator_policy: str = "all_records"
    missing_value_policy: str = "missing_is_violation"
    evidence_ids: tuple[str, ...] = ()
    resource_policy: Mapping[str, Any] = field(
        default_factory=lambda: {"max_inspection_cells": 2_000_000}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_type": self.rule_type,
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "fields": list(self.fields),
            "parameters": dict(self.parameters),
            "severity": self.severity,
            "denominator_policy": self.denominator_policy,
            "missing_value_policy": self.missing_value_policy,
            "evidence_ids": list(self.evidence_ids),
            "resource_policy": dict(self.resource_policy),
        }

    def to_rule(self) -> Rule:
        """转换为现有确定性 Rule；参数白名单由校验器负责。"""

        parameters = dict(self.parameters)
        common = {
            "type": self.rule_type,
            "rule_id": self.rule_id,
            "fields": tuple(self.fields),
        }
        if self.rule_type == "update_freshness":
            return Rule(
                **common,
                frequency=parameters.get("frequency"),
                max_age_days=parameters.get("max_age_days"),
            )
        if self.rule_type == "allowed_values":
            return Rule(
                **common,
                allowed_values=tuple(parameters.get("allowed_values", ())),
            )
        if self.rule_type == "numeric_range":
            return Rule(
                **common,
                minimum=parameters.get("minimum"),
                maximum=parameters.get("maximum"),
            )
        if self.rule_type == "regex_format":
            return Rule(
                **common,
                regex_pattern=parameters.get("pattern"),
            )
        if self.rule_type == "string_length":
            return Rule(
                **common,
                min_length=parameters.get("minimum"),
                max_length=parameters.get("maximum"),
            )
        if self.rule_type == "conditional_required":
            return Rule(
                **common,
                condition_field=self.fields[0],
                condition_values=tuple(parameters.get("condition_values", ())),
            )
        if self.rule_type == "field_comparison":
            return Rule(
                **common,
                comparison_operator=parameters.get("operator"),
                comparison_type=parameters.get("comparison_type", "auto"),
            )
        return Rule(**common)


@dataclass(frozen=True)
class ProviderMetadata:
    provider: str
    model: str | None
    mode: Literal["template", "model"]
    prompt_version: str
    fallback_used: bool = False
    fallback_reason: str | None = None
    request_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "prompt_version": self.prompt_version,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "request_id": self.request_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass(frozen=True)
class RuleDraft:
    """一次规则编制结果；只描述草案，不包含审批或执行结果。"""

    schema_version: str
    draft_id: str
    workflow_id: str
    target_type: RuleTargetType
    target_metric_id: str | None
    user_intent: str
    status: RuleDraftStatus
    rule_spec: RuleSpec | None
    evidence: tuple[RuleEvidence, ...]
    assumptions: tuple[str, ...]
    clarification_questions: tuple[str, ...]
    unsupported_reason: str | None
    provider: ProviderMetadata
    context: Mapping[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "draft_id": self.draft_id,
            "workflow_id": self.workflow_id,
            "target_type": self.target_type,
            "target_metric_id": self.target_metric_id,
            "user_intent": self.user_intent,
            "status": self.status,
            "rule_spec": self.rule_spec.to_dict() if self.rule_spec else None,
            "evidence": [item.to_dict() for item in self.evidence],
            "assumptions": list(self.assumptions),
            "clarification_questions": list(self.clarification_questions),
            "unsupported_reason": self.unsupported_reason,
            "provider": self.provider.to_dict(),
            "context": dict(self.context),
            "created_at": self.created_at,
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
class RuleDraftValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _validate_text(value: Any, label: str, maximum: int) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [f"{label}必须是非空字符串。"]
    if len(value) > maximum:
        return [f"{label}不能超过 {maximum} 个字符。"]
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return [f"{label}包含无法编码的字符。"]
    return []


def validate_rule_spec(
    rule_spec: RuleSpec | None,
    *,
    available_fields: Sequence[str] = (),
    evidence_ids: Sequence[str] = (),
) -> RuleDraftValidationResult:
    """对 DSL 做不依赖模型的基础校验。RulePack 校验负责数据类型适配。"""

    errors: list[str] = []
    if not isinstance(rule_spec, RuleSpec):
        return RuleDraftValidationResult(False, ("rule_spec 不是 RuleSpec。",))
    errors.extend(_validate_text(rule_spec.rule_type, "rule_type", 80))
    if rule_spec.rule_type not in SUPPORTED_RULE_TYPES:
        errors.append(f"规则类型“{rule_spec.rule_type}”不在当前 v0.8 白名单中。")
    if not _RULE_ID_PATTERN.fullmatch(rule_spec.rule_id):
        errors.append("rule_id 必须由本地代码生成，并符合小写 ID 格式。")
    for label, value, maximum in (
        ("name", rule_spec.name, 120),
        ("description", rule_spec.description, 2000),
    ):
        errors.extend(_validate_text(value, label, maximum))
    max_fields = 2 if rule_spec.rule_type in {
        "conditional_required",
        "field_comparison",
    } else 5
    if not 1 <= len(rule_spec.fields) <= max_fields:
        errors.append(f"fields 必须包含 1 到 {max_fields} 个字段。")
    if len(set(rule_spec.fields)) != len(rule_spec.fields):
        errors.append("fields 不能包含重复字段。")
    available = set(str(field) for field in available_fields)
    if available:
        missing = [field for field in rule_spec.fields if field not in available]
        if missing:
            errors.append(f"规则引用了不存在的字段：{'、'.join(missing)}。")
    if not isinstance(rule_spec.parameters, Mapping):
        errors.append("parameters 必须是对象。")
    else:
        parameters = dict(rule_spec.parameters)
        allowed_parameters = {
            "primary_key": set(),
            "required": set(),
            "update_freshness": {"frequency", "max_age_days"},
            "allowed_values": {"allowed_values"},
            "numeric_range": {"minimum", "maximum"},
            "regex_format": {"pattern"},
            "string_length": {"minimum", "maximum"},
            "conditional_required": {"condition_values"},
            "field_comparison": {"operator", "comparison_type"},
        }.get(rule_spec.rule_type, set())
        unknown_parameters = sorted(set(parameters) - allowed_parameters)
        if unknown_parameters:
            errors.append(
                f"规则类型“{rule_spec.rule_type}”包含未定义参数：{unknown_parameters}。"
            )
        if rule_spec.rule_type in {"update_freshness", "allowed_values"}:
            if set(parameters) != allowed_parameters:
                errors.append(
                    f"规则类型“{rule_spec.rule_type}”的 parameters 必须完整匹配白名单。"
                )
        if rule_spec.rule_type == "numeric_range" and not (
            set(parameters) & allowed_parameters
        ):
            errors.append("numeric_range 至少需要 minimum 或 maximum 参数。")
        if rule_spec.rule_type == "regex_format" and set(parameters) != {"pattern"}:
            errors.append("regex_format 的 parameters 必须只包含 pattern。")
        if rule_spec.rule_type == "string_length":
            if not set(parameters) & {"minimum", "maximum"}:
                errors.append("string_length 至少需要 minimum 或 maximum 参数。")
            if not set(parameters) <= {"minimum", "maximum"}:
                errors.append("string_length 只能包含 minimum 或 maximum 参数。")
        if rule_spec.rule_type == "conditional_required" and set(parameters) != {
            "condition_values"
        }:
            errors.append("conditional_required 必须只包含 condition_values 参数。")
        if rule_spec.rule_type == "field_comparison":
            if "operator" not in parameters:
                errors.append("field_comparison 必须包含 operator 参数。")
            if set(parameters) - {"operator", "comparison_type"}:
                errors.append("field_comparison 只能包含 operator 和 comparison_type 参数。")
    if rule_spec.severity not in {"info", "attention", "warning"}:
        errors.append("severity 不在允许范围内。")
    if rule_spec.denominator_policy not in {"all_records", "non_missing"}:
        errors.append("denominator_policy 不在允许范围内。")
    if rule_spec.missing_value_policy not in {
        "missing_is_violation",
        "missing_excluded",
    }:
        errors.append("missing_value_policy 不在允许范围内。")
    evidence_set = set(evidence_ids)
    missing_evidence = [
        item for item in rule_spec.evidence_ids if item not in evidence_set
    ]
    if missing_evidence:
        errors.append(f"规则引用了不存在的依据：{'、'.join(missing_evidence)}。")
    try:
        json.dumps(rule_spec.to_dict(), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        errors.append("RuleSpec 包含不可序列化或非有限数值。")
    return RuleDraftValidationResult(not errors, tuple(dict.fromkeys(errors)))


def validate_rule_draft_shape(draft: RuleDraft) -> RuleDraftValidationResult:
    """校验不需要数据集的协议形状。"""

    errors: list[str] = []
    if not isinstance(draft, RuleDraft):
        return RuleDraftValidationResult(False, ("对象不是 RuleDraft。",))
    if draft.schema_version != RULE_DRAFT_SCHEMA_VERSION:
        errors.append("RuleDraft schema_version 不受当前版本支持。")
    if not _DRAFT_ID_PATTERN.fullmatch(draft.draft_id):
        errors.append("draft_id 格式无效。")
    if not _WORKFLOW_ID_PATTERN.fullmatch(draft.workflow_id):
        errors.append("workflow_id 格式无效。")
    if draft.target_type not in {"catalog_metric", "custom_rule"}:
        errors.append("target_type 只能是 catalog_metric 或 custom_rule。")
    if draft.target_type == "catalog_metric" and not draft.target_metric_id:
        errors.append("目录指标 RuleDraft 必须包含 target_metric_id。")
    if draft.target_type == "custom_rule" and draft.target_metric_id is not None:
        errors.append("自定义规则 RuleDraft 的 target_metric_id 必须为空。")
    errors.extend(_validate_text(draft.user_intent, "user_intent", MAX_USER_INTENT_LENGTH))
    if draft.status not in SUPPORTED_DRAFT_STATUSES:
        errors.append("RuleDraft status 不在允许范围内。")
    if not 0 <= len(draft.evidence) <= MAX_EVIDENCE_ITEMS:
        errors.append("evidence 数量超出限制。")
    evidence_ids = [item.id for item in draft.evidence]
    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append("evidence.id 必须唯一。")
    for item in draft.evidence:
        if item.type not in SUPPORTED_EVIDENCE_TYPES:
            errors.append(f"依据类型“{item.type}”不受支持。")
        errors.extend(_validate_text(item.id, "evidence.id", 120))
        errors.extend(_validate_text(item.text, "evidence.text", 4000))
        if item.type in {"standard_clause", "data_dictionary"}:
            if not item.source_id:
                errors.append(f"{item.type} 必须包含 source_id。")
            if not item.document_name:
                errors.append(f"{item.type} 必须包含 document_name。")
            if not item.document_version:
                errors.append(f"{item.type} 必须包含 document_version。")
            if not item.section and not item.clause:
                errors.append(f"{item.type} 必须包含 section 或 clause 定位。")
            if not item.chunk_id:
                errors.append(f"{item.type} 必须包含 chunk_id。")
            if item.source_id and item.chunk_id and item.source_id != item.chunk_id:
                errors.append(f"{item.type} 的 source_id 必须等于 chunk_id。")
        if item.type == "system_inference" and item.authoritative:
            errors.append("system_inference 不能标记为 authoritative。")
        if item.page is not None and (
            isinstance(item.page, bool) or not isinstance(item.page, int) or item.page < 1
        ):
            errors.append("依据 page 必须是正整数或 null。")
    if len(draft.clarification_questions) > MAX_CLARIFICATION_QUESTIONS:
        errors.append("clarification_questions 最多包含 5 个问题。")
    for item in (*draft.assumptions, *draft.clarification_questions):
        errors.extend(_validate_text(item, "RuleDraft 文本", 2000))
    if draft.rule_spec is not None:
        errors.extend(
            validate_rule_spec(
                draft.rule_spec,
                evidence_ids=evidence_ids,
            ).errors
        )
    if draft.status in {"needs_clarification", "rejected"} and draft.rule_spec is not None:
        errors.append("需要澄清或已拒绝的草案不能包含可执行 rule_spec。")
    if draft.status not in {"needs_clarification", "rejected"} and draft.unsupported_reason:
        errors.append("非拒绝状态不能携带 unsupported_reason。")
    try:
        datetime.fromisoformat(draft.created_at.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        errors.append("created_at 必须是合法 ISO 8601 时间。")
    try:
        json.dumps(draft.to_dict(), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        errors.append("RuleDraft 包含不可序列化或非有限数值。")
    return RuleDraftValidationResult(not errors, tuple(dict.fromkeys(errors)))


def new_evidence(
    evidence_type: RuleEvidenceType,
    text: str,
    *,
    source_id: str | None = None,
    source_label: str | None = None,
    location: str | None = None,
    authoritative: bool = False,
    document_id: str | None = None,
    document_name: str | None = None,
    document_version: str | None = None,
    section: str | None = None,
    clause: str | None = None,
    chunk_id: str | None = None,
    page: int | None = None,
) -> RuleEvidence:
    evidence_id = _content_id(
        "evidence",
        [
            evidence_type,
            text,
            source_id,
            source_label,
            location,
            document_id,
            document_name,
            document_version,
            section,
            clause,
            chunk_id,
            page,
        ],
    )
    return RuleEvidence(
        id=evidence_id,
        type=evidence_type,
        text=text.strip(),
        source_id=source_id,
        source_label=source_label,
        location=location,
        authoritative=authoritative,
        document_id=document_id,
        document_name=document_name,
        document_version=document_version,
        section=section,
        clause=clause,
        chunk_id=chunk_id,
        page=page,
    )


def rule_spec_from_dict(payload: Mapping[str, Any]) -> RuleSpec:
    expected = {
        "rule_type",
        "rule_id",
        "name",
        "description",
        "fields",
        "parameters",
        "severity",
        "denominator_policy",
        "missing_value_policy",
        "evidence_ids",
        "resource_policy",
    }
    actual = set(payload)
    if actual != expected:
        raise RuleDraftValidationError(
            [f"rule_spec 字段不匹配；缺少 {sorted(expected - actual)}，多出 {sorted(actual - expected)}。"]
        )
    fields = payload["fields"]
    evidence_ids = payload["evidence_ids"]
    if not isinstance(fields, list) or not all(isinstance(item, str) for item in fields):
        raise RuleDraftValidationError(["rule_spec.fields 必须是字符串数组。"])
    if not isinstance(evidence_ids, list) or not all(
        isinstance(item, str) for item in evidence_ids
    ):
        raise RuleDraftValidationError(["rule_spec.evidence_ids 必须是字符串数组。"])
    if not isinstance(payload["parameters"], Mapping) or not isinstance(
        payload["resource_policy"], Mapping
    ):
        raise RuleDraftValidationError(["rule_spec.parameters/resource_policy 必须是对象。"])
    return RuleSpec(
        rule_type=str(payload["rule_type"]),
        rule_id=str(payload["rule_id"]),
        name=str(payload["name"]),
        description=str(payload["description"]),
        fields=tuple(fields),
        parameters=dict(payload["parameters"]),
        severity=str(payload["severity"]),
        denominator_policy=str(payload["denominator_policy"]),
        missing_value_policy=str(payload["missing_value_policy"]),
        evidence_ids=tuple(evidence_ids),
        resource_policy=dict(payload["resource_policy"]),
    )


def normalized_rule_spec(
    rule_spec: RuleSpec,
    *,
    evidence_ids: Sequence[str],
) -> RuleSpec:
    """用本地生成的 rule_id 和依据引用规范化模型候选。"""

    return RuleSpec(
        rule_type=rule_spec.rule_type,
        rule_id=make_rule_id(
            rule_spec.rule_type,
            rule_spec.fields,
            rule_spec.parameters,
        ),
        name=rule_spec.name,
        description=rule_spec.description,
        fields=tuple(rule_spec.fields),
        parameters=dict(rule_spec.parameters),
        severity=rule_spec.severity,
        denominator_policy=rule_spec.denominator_policy,
        missing_value_policy=rule_spec.missing_value_policy,
        evidence_ids=tuple(dict.fromkeys(evidence_ids)),
        resource_policy=dict(rule_spec.resource_policy),
    )


__all__ = [
    "MAX_CLARIFICATION_QUESTIONS",
    "MAX_EVIDENCE_ITEMS",
    "ProviderMetadata",
    "RULE_DRAFT_GENERATOR",
    "RULE_DRAFT_GENERATOR_V08",
    "RULE_DRAFT_GENERATOR_V09",
    "RULE_DRAFT_SCHEMA_VERSION",
    "RuleDraft",
    "RuleDraftStatus",
    "RuleDraftValidationError",
    "RuleDraftValidationResult",
    "RuleEvidence",
    "RuleEvidenceType",
    "RuleSpec",
    "RuleTargetType",
    "make_draft_id",
    "make_rule_id",
    "make_workflow_id",
    "new_evidence",
    "normalized_rule_spec",
    "rule_spec_from_dict",
    "validate_rule_draft_shape",
    "validate_rule_spec",
]
