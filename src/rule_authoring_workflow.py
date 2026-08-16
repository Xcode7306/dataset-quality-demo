"""Deterministic v1.0 rule-authoring lifecycle and bounded session history.

The model may produce a candidate ``RuleDraft`` but cannot choose workflow
states, approve a rule, execute it, or recover a failed run.  This module is a
pure domain layer: it has no Streamlit, HTTP, filesystem, or engine dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Literal, Mapping, Sequence

from .rule_dsl import (
    RuleDraft,
    RuleDraftValidationResult,
    make_workflow_id,
)
from .text_utils import contains_unsafe_unicode_controls


WORKFLOW_SCHEMA_VERSION = "1.0"
WORKFLOW_HISTORY_LIMIT = 20
MAX_WORKFLOW_TRANSITIONS = 64

WorkflowState = Literal[
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


class RuleAuthoringWorkflowError(ValueError):
    """The requested lifecycle operation violates a deterministic invariant."""


_WORKFLOW_ID_PATTERN = re.compile(r"^workflow-[a-z0-9][a-z0-9._-]{5,79}$")
_HASH_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_APPROVAL_ID_PATTERN = re.compile(r"^approval-[a-f0-9]{20}$")
_EXECUTION_ID_PATTERN = re.compile(r"^execution-[a-f0-9]{20}$")

_ALLOWED_TRANSITIONS: Mapping[str, frozenset[str]] = {
    "collecting": frozenset(
        {"retrieving", "compiling", "needs_clarification", "rejected"}
    ),
    "retrieving": frozenset({"compiling", "needs_clarification", "rejected"}),
    "needs_clarification": frozenset({"collecting", "rejected"}),
    "compiling": frozenset({"draft", "needs_clarification", "failed"}),
    "draft": frozenset({"validated", "needs_clarification", "rejected"}),
    "validated": frozenset({"dry_run_complete", "failed"}),
    "dry_run_complete": frozenset(
        {"awaiting_approval", "needs_clarification", "failed"}
    ),
    "awaiting_approval": frozenset({"approved", "rejected"}),
    "approved": frozenset({"executed", "failed"}),
    "executed": frozenset(),
    "rejected": frozenset(),
    # ``retry`` is the only operation allowed to choose one of these targets;
    # it always restores the exact state captured immediately before failure.
    "failed": frozenset(
        {"compiling", "validated", "dry_run_complete", "approved"}
    ),
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


def _utc(value: datetime | str | None = None) -> str:
    if value is None:
        timestamp = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        text = value.strip()
        timestamp = datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
    else:
        raise TypeError("工作流时间必须是 datetime、ISO 8601 字符串或 None。")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _safe_text(value: Any, *, maximum: int) -> str:
    text = " ".join(str(value or "").split())[:maximum]
    text.encode("utf-8", errors="strict")
    return text


def _safe_error(value: Any) -> str:
    text = _safe_text(value, maximum=2000)
    text = re.sub(
        r"(?i)\b(api[_ -]?key|authorization|access[_ -]?token)\b\s*[:=]\s*\S+",
        r"\1=[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", text)
    return text[:500] or "工作流步骤失败。"


def _artifact_id(prefix: str, value: str | None, fallback: Any) -> str:
    if value is None:
        return f"{prefix}-{_sha256(fallback)[:20]}"
    normalized = str(value).strip()
    pattern = _APPROVAL_ID_PATTERN if prefix == "approval" else _EXECUTION_ID_PATTERN
    if not pattern.fullmatch(normalized):
        raise RuleAuthoringWorkflowError(f"{prefix}_id 格式无效。")
    return normalized


def make_rule_authoring_request_fingerprint(
    *,
    target_type: str,
    target_metric_id: str | None,
    report_sha256: str | None,
    input_sha256: str | None,
    reference_date: str | None,
    selected_metric_ids: Sequence[str],
    user_intent: str,
    selected_chunk_ids: Sequence[str] = (),
) -> str:
    """Hash all state that must remain unchanged across a recovery retry."""

    intent = str(user_intent or "").strip()
    if not intent or len(intent) > 4000:
        raise RuleAuthoringWorkflowError("规则编制请求必须包含 1 到 4000 个字符。")
    intent.encode("utf-8", errors="strict")
    if contains_unsafe_unicode_controls(intent):
        raise RuleAuthoringWorkflowError("规则编制请求不能包含 Unicode 控制字符。")
    if target_type not in {"catalog_metric", "custom_rule"}:
        raise RuleAuthoringWorkflowError("工作流 target_type 无效。")
    if target_type == "catalog_metric" and not target_metric_id:
        raise RuleAuthoringWorkflowError("目录指标工作流必须绑定 target_metric_id。")
    if target_type == "custom_rule" and target_metric_id is not None:
        raise RuleAuthoringWorkflowError("自定义规则工作流不能绑定目录指标。")
    selected = tuple(dict.fromkeys(str(item) for item in selected_metric_ids))
    chunks = tuple(dict.fromkeys(str(item) for item in selected_chunk_ids))
    if len(selected) > 100 or len(chunks) > 20:
        raise RuleAuthoringWorkflowError("工作流上下文数量超出限制。")
    return _sha256(
        {
            "target_type": target_type,
            "target_metric_id": target_metric_id,
            "report_sha256": report_sha256,
            "input_sha256": input_sha256,
            "reference_date": reference_date,
            "selected_metric_ids": list(selected),
            "user_intent_sha256": hashlib.sha256(
                intent.encode("utf-8", errors="strict")
            ).hexdigest(),
            "selected_chunk_ids": list(chunks),
        }
    )


@dataclass(frozen=True)
class WorkflowTransition:
    sequence: int
    from_state: WorkflowState
    to_state: WorkflowState
    at: str
    reason_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "at": self.at,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class RuleAuthoringWorkflow:
    """One report-bound workflow with recoverable, auditable state."""

    workflow_id: str
    target_metric_id: str | None
    state: WorkflowState = "collecting"
    draft: RuleDraft | None = None
    validation: RuleDraftValidationResult | None = None
    dry_run: Mapping[str, Any] | None = None
    error: str | None = None
    target_type: str | None = None
    report_sha256: str | None = None
    input_sha256: str | None = None
    reference_date: str | None = None
    selected_metric_ids: tuple[str, ...] = ()
    request_fingerprint: str | None = None
    revision: int = 1
    retry_count: int = 0
    recoverable_state: WorkflowState | None = None
    approval_id: str | None = None
    execution_result_id: str | None = None
    failure_stage: str | None = None
    failure_code: str | None = None
    transitions: tuple[WorkflowTransition, ...] = ()
    created_at: str = field(default_factory=_utc)
    updated_at: str | None = None
    schema_version: str = WORKFLOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _WORKFLOW_ID_PATTERN.fullmatch(str(self.workflow_id)):
            raise RuleAuthoringWorkflowError("workflow_id 格式无效。")
        if self.state not in _ALLOWED_TRANSITIONS:
            raise RuleAuthoringWorkflowError("工作流 state 无效。")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or not 1 <= self.revision <= 1_000_000
        ):
            raise RuleAuthoringWorkflowError("工作流 revision 无效。")
        if (
            isinstance(self.retry_count, bool)
            or not isinstance(self.retry_count, int)
            or not 0 <= self.retry_count <= 1
        ):
            raise RuleAuthoringWorkflowError("工作流 retry_count 无效。")
        target_type = self.target_type or (
            "catalog_metric" if self.target_metric_id else "custom_rule"
        )
        if target_type not in {"catalog_metric", "custom_rule"}:
            raise RuleAuthoringWorkflowError("工作流 target_type 无效。")
        if target_type == "catalog_metric" and not self.target_metric_id:
            raise RuleAuthoringWorkflowError("目录指标工作流必须绑定 target_metric_id。")
        if target_type == "custom_rule" and self.target_metric_id is not None:
            raise RuleAuthoringWorkflowError("自定义规则工作流不能绑定 target_metric_id。")
        selected = tuple(dict.fromkeys(str(item) for item in self.selected_metric_ids))
        if len(selected) > 100:
            raise RuleAuthoringWorkflowError("selected_metric_ids 数量超出限制。")
        created = _utc(self.created_at)
        updated = _utc(self.updated_at or created)
        fingerprint = self.request_fingerprint or _sha256(
            {"workflow_id": self.workflow_id}
        )
        if not _HASH_PATTERN.fullmatch(fingerprint):
            raise RuleAuthoringWorkflowError("request_fingerprint 格式无效。")
        object.__setattr__(self, "target_type", target_type)
        object.__setattr__(self, "selected_metric_ids", selected)
        object.__setattr__(self, "request_fingerprint", fingerprint)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)

    def _transition(
        self,
        state: WorkflowState,
        *,
        at: datetime | str | None = None,
        reason_code: str | None = None,
        **updates: Any,
    ) -> "RuleAuthoringWorkflow":
        allowed = _ALLOWED_TRANSITIONS.get(self.state, frozenset())
        if state not in allowed:
            raise RuleAuthoringWorkflowError(
                f"工作流不能从 {self.state} 转换到 {state}。"
            )
        if len(self.transitions) >= MAX_WORKFLOW_TRANSITIONS:
            raise RuleAuthoringWorkflowError("工作流状态转换数量超出限制。")
        timestamp = _utc(at)
        transition = WorkflowTransition(
            sequence=len(self.transitions) + 1,
            from_state=self.state,
            to_state=state,
            at=timestamp,
            reason_code=(
                _safe_text(reason_code, maximum=120) if reason_code else None
            ),
        )
        cleared = {
            "recoverable_state": None,
            "failure_stage": None,
            "failure_code": None,
        }
        cleared.update(updates)
        return replace(
            self,
            state=state,
            transitions=(*self.transitions, transition),
            updated_at=timestamp,
            **cleared,
        )

    def start_retrieving(
        self, *, at: datetime | str | None = None
    ) -> "RuleAuthoringWorkflow":
        return self._transition("retrieving", at=at, error=None)

    def start_compiling(
        self, *, at: datetime | str | None = None
    ) -> "RuleAuthoringWorkflow":
        return self._transition("compiling", at=at, error=None)

    def accept_draft(
        self,
        draft: RuleDraft,
        *,
        at: datetime | str | None = None,
    ) -> "RuleAuthoringWorkflow":
        if self.state != "compiling":
            raise RuleAuthoringWorkflowError("只有 compiling 状态可以接收 Provider 草案。")
        if not isinstance(draft, RuleDraft):
            raise RuleAuthoringWorkflowError("工作流只能接收 RuleDraft。")
        if draft.workflow_id != self.workflow_id:
            raise RuleAuthoringWorkflowError("RuleDraft 与当前 workflow_id 不一致。")
        if draft.target_type != self.target_type:
            raise RuleAuthoringWorkflowError("RuleDraft target_type 与工作流不一致。")
        if draft.target_metric_id != self.target_metric_id:
            raise RuleAuthoringWorkflowError("RuleDraft 指标绑定与工作流不一致。")
        if draft.status == "needs_clarification":
            return self._transition(
                "needs_clarification",
                at=at,
                draft=draft,
                error=_safe_error(
                    "；".join(draft.clarification_questions)
                    or "当前规则需要补充关键信息。"
                ),
            )
        if draft.status == "rejected":
            return self._transition(
                "rejected",
                at=at,
                draft=draft,
                error=_safe_error(draft.unsupported_reason or "当前规则不受支持。"),
            )
        if draft.status == "failed":
            return self.fail(
                stage="compiling",
                code="provider_failed_draft",
                message=draft.unsupported_reason or "Provider 未生成规则草案。",
                at=at,
            )
        if draft.status != "draft" or draft.rule_spec is None:
            raise RuleAuthoringWorkflowError("Provider 草案状态与内容不一致。")
        return self._transition("draft", at=at, draft=draft, error=None)

    def mark_validated(
        self,
        validation: RuleDraftValidationResult,
        *,
        at: datetime | str | None = None,
    ) -> "RuleAuthoringWorkflow":
        if self.state != "draft":
            raise RuleAuthoringWorkflowError("只有 draft 状态可以执行规则校验。")
        if not isinstance(validation, RuleDraftValidationResult):
            raise RuleAuthoringWorkflowError("validation 必须是确定性校验结果。")
        if validation.valid:
            return self._transition(
                "validated",
                at=at,
                validation=validation,
                error=None,
            )
        return self._transition(
            "needs_clarification",
            at=at,
            reason_code="deterministic_validation_failed",
            validation=validation,
            error=_safe_error("；".join(validation.errors)),
        )

    def mark_dry_run_complete(
        self,
        preview: Mapping[str, Any],
        *,
        at: datetime | str | None = None,
    ) -> "RuleAuthoringWorkflow":
        if self.state != "validated":
            raise RuleAuthoringWorkflowError("只有 validated 状态可以完成试运行。")
        payload = dict(preview)
        _canonical_json(payload)
        return self._transition(
            "dry_run_complete",
            at=at,
            dry_run=payload,
            error=None,
        )

    def await_approval(
        self, *, at: datetime | str | None = None
    ) -> "RuleAuthoringWorkflow":
        if self.validation is None or not self.validation.valid or self.dry_run is None:
            raise RuleAuthoringWorkflowError("校验和试运行完成前不能进入审批。")
        return self._transition("awaiting_approval", at=at, error=None)

    def approve(
        self,
        approval_id: str | None = None,
        *,
        at: datetime | str | None = None,
    ) -> "RuleAuthoringWorkflow":
        normalized = _artifact_id(
            "approval",
            approval_id,
            [self.workflow_id, self.request_fingerprint, "local-approval"],
        )
        if self.state in {"approved", "executed"}:
            if self.approval_id == normalized:
                return self
            raise RuleAuthoringWorkflowError("当前工作流已经绑定另一审批记录。")
        return self._transition(
            "approved",
            at=at,
            approval_id=normalized,
            error=None,
        )

    def execute(
        self,
        execution_result_id: str | None = None,
        *,
        at: datetime | str | None = None,
    ) -> "RuleAuthoringWorkflow":
        normalized = _artifact_id(
            "execution",
            execution_result_id,
            [self.workflow_id, self.approval_id, "deterministic-result"],
        )
        if self.state == "executed":
            if self.execution_result_id == normalized:
                return self
            raise RuleAuthoringWorkflowError("当前工作流已经绑定另一执行结果。")
        if not self.approval_id:
            raise RuleAuthoringWorkflowError("没有审批 ID 的工作流不能执行。")
        return self._transition(
            "executed",
            at=at,
            execution_result_id=normalized,
            error=None,
        )

    def reject(
        self,
        reason: str = "用户拒绝当前规则草案。",
        *,
        at: datetime | str | None = None,
    ) -> "RuleAuthoringWorkflow":
        return self._transition(
            "rejected",
            at=at,
            reason_code="user_rejected",
            error=_safe_error(reason),
        )

    def fail(
        self,
        *,
        stage: str,
        code: str,
        message: str,
        at: datetime | str | None = None,
    ) -> "RuleAuthoringWorkflow":
        if "failed" not in _ALLOWED_TRANSITIONS.get(self.state, frozenset()):
            raise RuleAuthoringWorkflowError(
                f"工作流不能从 {self.state} 进入失败恢复流程。"
            )
        return self._transition(
            "failed",
            at=at,
            reason_code=code,
            recoverable_state=self.state,
            failure_stage=_safe_text(stage, maximum=80),
            failure_code=_safe_text(code, maximum=120),
            error=_safe_error(message),
        )

    @property
    def can_retry(self) -> bool:
        return (
            self.state == "failed"
            and self.recoverable_state is not None
            and self.retry_count < 1
        )

    def retry(
        self,
        *,
        request_fingerprint: str | None = None,
        at: datetime | str | None = None,
    ) -> "RuleAuthoringWorkflow":
        if self.state != "failed" or self.recoverable_state is None:
            raise RuleAuthoringWorkflowError("只有可恢复的 failed 状态可以重试。")
        if self.retry_count >= 1:
            raise RuleAuthoringWorkflowError("同一工作流最多只允许重试一次。")
        if (
            request_fingerprint is not None
            and request_fingerprint != self.request_fingerprint
        ):
            raise RuleAuthoringWorkflowError("请求指纹已变化，不能复用旧工作流重试。")
        target = self.recoverable_state
        return self._transition(
            target,
            at=at,
            reason_code="retry_same_request",
            retry_count=self.retry_count + 1,
            error=None,
        )

    def to_dict(self) -> dict[str, Any]:
        draft_summary = None
        if self.draft is not None:
            rule_spec = self.draft.rule_spec
            draft_summary = {
                "draft_id": self.draft.draft_id,
                "draft_sha256": _sha256(self.draft.to_dict()),
                "status": self.draft.status,
                "rule_id": rule_spec.rule_id if rule_spec else None,
                "rule_type": rule_spec.rule_type if rule_spec else None,
                "evidence_ids": [item.id for item in self.draft.evidence],
            }
        validation_summary = None
        if self.validation is not None:
            validation_summary = {
                "valid": self.validation.valid,
                "error_count": len(self.validation.errors),
                "warning_count": len(self.validation.warnings),
            }
        dry_run_summary = None
        if self.dry_run is not None:
            counts = self.dry_run.get("counts", {})
            dry_run_summary = {
                "sha256": _sha256(dict(self.dry_run)),
                "rule_pack_id": self.dry_run.get("rule_pack_id"),
                "rule_pack_version": self.dry_run.get("rule_pack_version"),
                "counts": dict(counts) if isinstance(counts, Mapping) else {},
            }
        safe_message = _safe_error(self.error) if self.error else None
        failure = None
        if self.state == "failed":
            failure = {
                "stage": self.failure_stage,
                "code": self.failure_code,
                "message": safe_message,
                "recoverable_state": self.recoverable_state,
            }
        payload = {
            "schema_version": self.schema_version,
            "workflow_id": self.workflow_id,
            "target": {
                "type": self.target_type,
                "metric_id": self.target_metric_id,
            },
            "context": {
                "report_sha256": self.report_sha256,
                "input_sha256": self.input_sha256,
                "reference_date": self.reference_date,
                "selected_metric_ids": list(self.selected_metric_ids),
                "request_fingerprint": self.request_fingerprint,
            },
            "state": self.state,
            "message": safe_message,
            "revision": self.revision,
            "retry_count": self.retry_count,
            "draft": draft_summary,
            "validation": validation_summary,
            "dry_run": dry_run_summary,
            "approval_id": self.approval_id,
            "execution_result_id": self.execution_result_id,
            "failure": failure,
            "transitions": [item.to_dict() for item in self.transitions],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        _canonical_json(payload)
        return payload


def new_rule_authoring_workflow(
    *,
    target_type: str,
    target_metric_id: str | None,
    report_sha256: str | None,
    input_sha256: str | None,
    reference_date: str | None,
    selected_metric_ids: Sequence[str],
    user_intent: str,
    selected_chunk_ids: Sequence[str] = (),
    revision: int = 1,
    created_at: datetime | str | None = None,
) -> RuleAuthoringWorkflow:
    fingerprint = make_rule_authoring_request_fingerprint(
        target_type=target_type,
        target_metric_id=target_metric_id,
        report_sha256=report_sha256,
        input_sha256=input_sha256,
        reference_date=reference_date,
        selected_metric_ids=selected_metric_ids,
        user_intent=user_intent,
        selected_chunk_ids=selected_chunk_ids,
    )
    timestamp = _utc(created_at)
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or not 1 <= revision <= 1_000_000
    ):
        raise RuleAuthoringWorkflowError("工作流 revision 无效。")
    return RuleAuthoringWorkflow(
        workflow_id=make_workflow_id(["v1.0", fingerprint, revision]),
        target_metric_id=target_metric_id,
        target_type=target_type,
        report_sha256=report_sha256,
        input_sha256=input_sha256,
        reference_date=reference_date,
        selected_metric_ids=tuple(selected_metric_ids),
        request_fingerprint=fingerprint,
        revision=revision,
        created_at=timestamp,
        updated_at=timestamp,
    )


def validate_rule_authoring_workflow(
    workflow: RuleAuthoringWorkflow,
) -> tuple[str, ...]:
    """Validate a lifecycle snapshot independently of UI or execution code."""

    errors: list[str] = []
    if not isinstance(workflow, RuleAuthoringWorkflow):
        return ("对象不是 RuleAuthoringWorkflow。",)
    if workflow.schema_version != WORKFLOW_SCHEMA_VERSION:
        errors.append("工作流 schema_version 不受支持。")
    if workflow.state not in _ALLOWED_TRANSITIONS:
        errors.append("工作流 state 无效。")
    if not 1 <= workflow.revision <= 1_000_000:
        errors.append("工作流 revision 无效。")
    if not 0 <= workflow.retry_count <= 1:
        errors.append("工作流 retry_count 无效。")
    for label, value in (
        ("report_sha256", workflow.report_sha256),
        ("input_sha256", workflow.input_sha256),
        ("request_fingerprint", workflow.request_fingerprint),
    ):
        if value is not None and not _HASH_PATTERN.fullmatch(value):
            errors.append(f"{label} 格式无效。")
    if workflow.draft is not None:
        if workflow.draft.workflow_id != workflow.workflow_id:
            errors.append("RuleDraft workflow_id 与工作流不一致。")
        if workflow.draft.target_type != workflow.target_type:
            errors.append("RuleDraft target_type 与工作流不一致。")
        if workflow.draft.target_metric_id != workflow.target_metric_id:
            errors.append("RuleDraft 指标绑定与工作流不一致。")
    states_requiring_draft = {
        "draft",
        "validated",
        "dry_run_complete",
        "awaiting_approval",
        "approved",
        "executed",
    }
    if workflow.state in states_requiring_draft and workflow.draft is None:
        errors.append(f"{workflow.state} 状态必须包含 RuleDraft。")
    states_requiring_validation = {
        "validated",
        "dry_run_complete",
        "awaiting_approval",
        "approved",
        "executed",
    }
    if workflow.state in states_requiring_validation and (
        workflow.validation is None or not workflow.validation.valid
    ):
        errors.append(f"{workflow.state} 状态必须绑定已通过的确定性校验。")
    states_requiring_dry_run = {
        "dry_run_complete",
        "awaiting_approval",
        "approved",
        "executed",
    }
    if workflow.state in states_requiring_dry_run and workflow.dry_run is None:
        errors.append(f"{workflow.state} 状态必须绑定试运行摘要。")
    if workflow.state in {"approved", "executed"} and not workflow.approval_id:
        errors.append(f"{workflow.state} 状态必须绑定 approval_id。")
    if workflow.approval_id is not None and not _APPROVAL_ID_PATTERN.fullmatch(
        workflow.approval_id
    ):
        errors.append("approval_id 格式无效。")
    if workflow.state == "executed" and not workflow.execution_result_id:
        errors.append("executed 状态必须绑定 execution_result_id。")
    if workflow.execution_result_id is not None and not _EXECUTION_ID_PATTERN.fullmatch(
        workflow.execution_result_id
    ):
        errors.append("execution_result_id 格式无效。")
    if workflow.state == "failed":
        if not workflow.recoverable_state:
            errors.append("failed 状态必须记录 recoverable_state。")
        elif workflow.recoverable_state not in _ALLOWED_TRANSITIONS["failed"]:
            errors.append("failed 状态的 recoverable_state 无效。")
        if not workflow.failure_stage or not workflow.failure_code or not workflow.error:
            errors.append("failed 状态必须记录失败阶段、代码和安全消息。")
    elif any(
        value is not None
        for value in (
            workflow.recoverable_state,
            workflow.failure_stage,
            workflow.failure_code,
        )
    ):
        errors.append("非 failed 状态不能保留失败恢复字段。")
    if not workflow.transitions and workflow.state != "collecting":
        errors.append("非 collecting 工作流必须包含状态转换记录。")
    if workflow.transitions and workflow.transitions[0].from_state != "collecting":
        errors.append("工作流状态转换链必须从 collecting 开始。")
    previous_to: str | None = None
    previous_at: str | None = None
    retry_transition_count = 0
    for index, transition in enumerate(workflow.transitions, start=1):
        if transition.sequence != index:
            errors.append("工作流状态转换序号不连续。")
        if previous_to is not None and transition.from_state != previous_to:
            errors.append("工作流状态转换链不连续。")
        if transition.to_state not in _ALLOWED_TRANSITIONS.get(
            transition.from_state, frozenset()
        ):
            errors.append(
                f"存在非法状态转换：{transition.from_state} → {transition.to_state}。"
            )
        if previous_at is not None and transition.at < previous_at:
            errors.append("工作流状态转换时间不是非递减顺序。")
        if transition.reason_code == "retry_same_request":
            retry_transition_count += 1
        previous_to = transition.to_state
        previous_at = transition.at
    if workflow.transitions and workflow.transitions[-1].to_state != workflow.state:
        errors.append("最后一次状态转换与当前 state 不一致。")
    if retry_transition_count != workflow.retry_count:
        errors.append("工作流 retry_count 与恢复转换记录不一致。")
    if workflow.state == "failed" and workflow.transitions:
        last = workflow.transitions[-1]
        if last.from_state != workflow.recoverable_state:
            errors.append("failed 状态的恢复点与失败前状态不一致。")
    try:
        workflow.to_dict()
    except (TypeError, ValueError, UnicodeError):
        errors.append("工作流包含不可安全序列化的值。")
    return tuple(dict.fromkeys(errors))


@dataclass(frozen=True)
class RuleAuthoringHistory:
    """Bounded current-session history; never implies server persistence."""

    records: tuple[RuleAuthoringWorkflow, ...] = ()
    limit: int = WORKFLOW_HISTORY_LIMIT
    schema_version: str = WORKFLOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKFLOW_SCHEMA_VERSION:
            raise RuleAuthoringWorkflowError("工作流历史 schema_version 不受支持。")
        if not 1 <= self.limit <= WORKFLOW_HISTORY_LIMIT:
            raise RuleAuthoringWorkflowError("工作流历史容量无效。")
        if len(self.records) > self.limit:
            raise RuleAuthoringWorkflowError("工作流历史记录数量超出容量。")
        workflow_ids = [item.workflow_id for item in self.records]
        if len(workflow_ids) != len(set(workflow_ids)):
            raise RuleAuthoringWorkflowError("工作流历史不能包含重复 workflow_id。")
        for workflow in self.records:
            errors = validate_rule_authoring_workflow(workflow)
            if errors:
                raise RuleAuthoringWorkflowError("；".join(errors))

    def upsert(self, workflow: RuleAuthoringWorkflow) -> "RuleAuthoringHistory":
        errors = validate_rule_authoring_workflow(workflow)
        if errors:
            raise RuleAuthoringWorkflowError("；".join(errors))
        retained = tuple(
            item for item in self.records if item.workflow_id != workflow.workflow_id
        )
        return replace(self, records=(*retained, workflow)[-self.limit :])

    def get(self, workflow_id: str) -> RuleAuthoringWorkflow | None:
        return next(
            (item for item in reversed(self.records) if item.workflow_id == workflow_id),
            None,
        )

    def clear(self) -> "RuleAuthoringHistory":
        return replace(self, records=())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "storage": "current_session_memory",
            "cross_session_persistence": False,
            "limit": self.limit,
            "records": [item.to_dict() for item in self.records],
        }
        _canonical_json(payload)
        return payload


__all__ = [
    "MAX_WORKFLOW_TRANSITIONS",
    "RuleAuthoringHistory",
    "RuleAuthoringWorkflow",
    "RuleAuthoringWorkflowError",
    "WORKFLOW_HISTORY_LIMIT",
    "WORKFLOW_SCHEMA_VERSION",
    "WorkflowState",
    "WorkflowTransition",
    "make_rule_authoring_request_fingerprint",
    "new_rule_authoring_workflow",
    "validate_rule_authoring_workflow",
]
