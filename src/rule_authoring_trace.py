"""Strict, privacy-bounded execution traces for the v0.9.1 Agent Harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


TRACE_SCHEMA_VERSION = "0.1"
MAX_TRACE_TRANSITIONS = 50
MAX_TRACE_TOOL_CALLS = 50


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
    else:
        text = str(value).strip()
        timestamp = datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _safe_text(value: Any, *, maximum: int = 500) -> str:
    text = " ".join(str(value or "").split())[:maximum]
    text.encode("utf-8", errors="strict")
    return text


def _safe_failure_message(value: Any) -> str:
    text = _safe_text(value, maximum=2000)
    text = re.sub(
        r"(?i)\b(api[_ -]?key|authorization|access[_ -]?token)\b\s*[:=]\s*\S+",
        r"\1=[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", text)
    return text[:500]


def summarize_tool_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize tool arguments without recording free-form user or data text."""

    safe_identifiers = {
        "metric_id",
        "standard_number",
        "version",
        "source_namespace",
        "draft_id",
        "workflow_id",
        "limit",
    }
    summary: dict[str, Any] = {}
    for key in sorted(arguments):
        value = arguments[key]
        if key in safe_identifiers and (
            value is None or isinstance(value, (str, int, float, bool))
        ):
            summary[str(key)] = value
        elif isinstance(value, str):
            summary[str(key)] = {
                "type": "text",
                "length": len(value),
                "sha256": hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest(),
            }
        elif isinstance(value, Mapping):
            summary[str(key)] = {
                "type": "object",
                "keys": sorted(str(item) for item in value)[:50],
            }
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            summary[str(key)] = {"type": "array", "length": len(value)}
        elif isinstance(value, (bytes, bytearray)):
            summary[str(key)] = {
                "type": "bytes",
                "length": len(value),
                "sha256": hashlib.sha256(bytes(value)).hexdigest(),
            }
        elif value is None or isinstance(value, (int, float, bool)):
            summary[str(key)] = value
        else:
            summary[str(key)] = {"type": type(value).__name__}
    _canonical_json(summary)
    return summary


@dataclass(frozen=True)
class TraceTransition:
    sequence: int
    from_state: str
    to_state: str
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
class ToolCallTrace:
    sequence: int
    call_id: str
    tool_name: str
    arguments_sha256: str
    argument_summary: Mapping[str, Any]
    result_status: str
    result_sha256: str | None
    started_at: str
    duration_ms: int
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments_sha256": self.arguments_sha256,
            "argument_summary": dict(self.argument_summary),
            "result_status": self.result_status,
            "result_sha256": self.result_sha256,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "error_code": self.error_code,
        }


@dataclass(frozen=True)
class RuleAuthoringTrace:
    trace_id: str
    workflow_id: str
    target: Mapping[str, Any]
    context: Mapping[str, Any]
    provider: Mapping[str, Any]
    started_at: str
    completed_at: str
    outcome: str
    transitions: tuple[TraceTransition, ...]
    tool_calls: tuple[ToolCallTrace, ...]
    retrieval: Mapping[str, Any] | None
    draft: Mapping[str, Any] | None
    validation_status: str
    dry_run_status: str
    approval_id: str | None
    execution_result_id: str | None
    retry_count: int
    fallback_used: bool
    fallback_reason: str | None
    failure: Mapping[str, Any] | None
    semantic_fingerprint: str
    schema_version: str = TRACE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "workflow_id": self.workflow_id,
            "target": dict(self.target),
            "context": dict(self.context),
            "provider": dict(self.provider),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "outcome": self.outcome,
            "transitions": [item.to_dict() for item in self.transitions],
            "tool_calls": [item.to_dict() for item in self.tool_calls],
            "retrieval": dict(self.retrieval) if self.retrieval is not None else None,
            "draft": dict(self.draft) if self.draft is not None else None,
            "validation_status": self.validation_status,
            "dry_run_status": self.dry_run_status,
            "approval_id": self.approval_id,
            "execution_result_id": self.execution_result_id,
            "retry_count": self.retry_count,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "failure": dict(self.failure) if self.failure is not None else None,
            "semantic_fingerprint": self.semantic_fingerprint,
        }
        _canonical_json(payload)
        return payload


@dataclass
class RuleAuthoringTraceBuilder:
    """Mutable builder kept out of domain objects and Streamlit state."""

    workflow_id: str
    target_type: str
    target_metric_id: str | None
    report_sha256: str | None
    input_sha256: str | None
    reference_date: str | None
    report_status: str | None = None
    row_count: int | None = None
    column_count: int | None = None
    selected_metric_ids: Sequence[str] = ()
    case_id: str | None = None
    started_at: str = field(default_factory=_utc)
    _transitions: list[TraceTransition] = field(default_factory=list, init=False)
    _tools: list[ToolCallTrace] = field(default_factory=list, init=False)
    _retrieval: dict[str, Any] | None = field(default=None, init=False)
    _draft: dict[str, Any] | None = field(default=None, init=False)
    _provider: dict[str, Any] = field(default_factory=dict, init=False)
    _validation_status: str = field(default="not_run", init=False)
    _dry_run_status: str = field(default="not_run", init=False)
    _approval_id: str | None = field(default=None, init=False)
    _execution_result_id: str | None = field(default=None, init=False)
    _retry_count: int = field(default=0, init=False)
    _fallback_used: bool = field(default=False, init=False)
    _fallback_reason: str | None = field(default=None, init=False)
    _failure: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.started_at = _utc(self.started_at)
        _canonical_json(self._base_identity())

    def _base_identity(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "target_type": self.target_type,
            "target_metric_id": self.target_metric_id,
            "report_sha256": self.report_sha256,
            "input_sha256": self.input_sha256,
            "reference_date": self.reference_date,
            "case_id": self.case_id,
        }

    def transition(
        self,
        from_state: str,
        to_state: str,
        *,
        at: datetime | str | None = None,
        reason_code: str | None = None,
    ) -> None:
        if len(self._transitions) >= MAX_TRACE_TRANSITIONS:
            raise ValueError("运行轨迹状态转换数超出上限。")
        self._transitions.append(
            TraceTransition(
                sequence=len(self._transitions) + 1,
                from_state=_safe_text(from_state, maximum=60),
                to_state=_safe_text(to_state, maximum=60),
                at=_utc(at or self.started_at),
                reason_code=(
                    _safe_text(reason_code, maximum=120) if reason_code else None
                ),
            )
        )

    def tool_call(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        result: Any | None = None,
        result_status: str = "ok",
        started_at: datetime | str | None = None,
        duration_ms: int = 0,
        error_code: str | None = None,
    ) -> None:
        if len(self._tools) >= MAX_TRACE_TOOL_CALLS:
            raise ValueError("运行轨迹工具调用数超出上限。")
        if result_status not in {"ok", "error", "rejected"}:
            raise ValueError("工具调用结果状态无效。")
        arguments_hash = _sha256(dict(arguments))
        sequence = len(self._tools) + 1
        call_id = "tool-" + _sha256(
            [self.workflow_id, sequence, tool_name, arguments_hash]
        )[:20]
        self._tools.append(
            ToolCallTrace(
                sequence=sequence,
                call_id=call_id,
                tool_name=_safe_text(tool_name, maximum=80),
                arguments_sha256=arguments_hash,
                argument_summary=summarize_tool_arguments(arguments),
                result_status=result_status,
                result_sha256=_sha256(result) if result is not None else None,
                started_at=_utc(started_at or self.started_at),
                duration_ms=max(0, min(int(duration_ms), 3_600_000)),
                error_code=_safe_text(error_code, maximum=120) if error_code else None,
            )
        )

    def bind_provider(self, metadata: Any, *, prompt_sha256: str | None = None) -> None:
        to_dict = getattr(metadata, "to_dict", None)
        raw = to_dict() if callable(to_dict) else dict(metadata or {})
        self._provider = {
            "provider": str(raw.get("provider") or "unknown")[:120],
            "model": str(raw["model"])[:200] if raw.get("model") else None,
            "mode": raw.get("mode") if raw.get("mode") in {"template", "model"} else "template",
            "prompt_version": str(raw.get("prompt_version") or "unknown")[:200],
            "prompt_sha256": prompt_sha256,
            "request_id": str(raw["request_id"])[:200] if raw.get("request_id") else None,
            "input_tokens": max(0, int(raw.get("input_tokens") or 0)),
            "output_tokens": max(0, int(raw.get("output_tokens") or 0)),
        }
        self._fallback_used = bool(raw.get("fallback_used", False))
        reason = raw.get("fallback_reason")
        self._fallback_reason = _safe_text(reason, maximum=200) if reason else None

    def bind_retrieval(self, response: Any, *, query: str, filters: Mapping[str, Any]) -> None:
        payload = response.to_dict(include_text=False)
        self._retrieval = {
            "status": payload.get("status"),
            "query_sha256": hashlib.sha256(
                query.encode("utf-8", errors="strict")
            ).hexdigest(),
            "filters": dict(filters),
            "chunk_ids": [
                item.get("chunk_id")
                for item in payload.get("results", [])
                if isinstance(item, Mapping) and item.get("chunk_id")
            ],
            "conflict": payload.get("conflict") is not None,
        }

    def bind_draft(self, draft: Any) -> None:
        payload = draft.to_dict()
        rule_spec = payload.get("rule_spec") or {}
        semantic = {
            "target_type": payload.get("target_type"),
            "target_metric_id": payload.get("target_metric_id"),
            "status": payload.get("status"),
            "rule_type": rule_spec.get("rule_type"),
            "fields": rule_spec.get("fields"),
            "parameters": rule_spec.get("parameters"),
            "evidence_sources": sorted(
                str(item.get("source_id"))
                for item in payload.get("evidence", [])
                if isinstance(item, Mapping) and item.get("source_id")
            ),
        }
        self._draft = {
            "draft_id": payload.get("draft_id"),
            "draft_schema_version": payload.get("schema_version"),
            "status": payload.get("status"),
            "rule_id": rule_spec.get("rule_id"),
            "semantic_sha256": _sha256(semantic),
        }

    def record_validation(self, passed: bool) -> None:
        self._validation_status = "passed" if passed else "failed"

    def record_dry_run(self, passed: bool) -> None:
        self._dry_run_status = "passed" if passed else "failed"

    def record_approval(self, approval_id: str) -> None:
        self._approval_id = _safe_text(approval_id, maximum=120)

    def record_execution(self, result: Any) -> None:
        payload = result.to_dict() if callable(getattr(result, "to_dict", None)) else result
        self._execution_result_id = "execution-" + _sha256(payload)[:20]

    def record_failure(self, *, stage: str, code: str, message: str) -> None:
        self._failure = {
            "stage": _safe_text(stage, maximum=80),
            "code": _safe_text(code, maximum=120),
            "message": _safe_failure_message(message),
        }

    def set_retry_count(self, value: int) -> None:
        self._retry_count = max(0, min(int(value), 1))

    def finish(
        self,
        outcome: str,
        *,
        completed_at: datetime | str | None = None,
    ) -> RuleAuthoringTrace:
        if outcome not in {"draft", "clarification", "unsupported", "failed"}:
            raise ValueError("运行轨迹 outcome 无效。")
        completed = _utc(completed_at or self.started_at)
        target = {
            "type": self.target_type,
            "metric_id": self.target_metric_id,
            "case_id": self.case_id,
        }
        context = {
            "report_sha256": self.report_sha256,
            "input_sha256": self.input_sha256,
            "reference_date": self.reference_date,
            "report_status": self.report_status,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "selected_metric_ids": list(self.selected_metric_ids),
        }
        provider = self._provider or {
            "provider": "unknown",
            "model": None,
            "mode": "template",
            "prompt_version": "unknown",
            "prompt_sha256": None,
            "request_id": None,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        semantic_payload = {
            "workflow_id": self.workflow_id,
            "target": target,
            "context": context,
            "provider": {
                key: provider.get(key)
                for key in ("provider", "model", "mode", "prompt_version", "prompt_sha256")
            },
            "outcome": outcome,
            "transitions": [
                {
                    "from_state": item.from_state,
                    "to_state": item.to_state,
                    "reason_code": item.reason_code,
                }
                for item in self._transitions
            ],
            "tools": [
                {
                    "tool_name": item.tool_name,
                    "arguments_sha256": item.arguments_sha256,
                    "result_status": item.result_status,
                    "result_sha256": item.result_sha256,
                    "error_code": item.error_code,
                }
                for item in self._tools
            ],
            "retrieval": self._retrieval,
            "draft": self._draft,
            "validation_status": self._validation_status,
            "dry_run_status": self._dry_run_status,
            "fallback_used": self._fallback_used,
            "fallback_reason": self._fallback_reason,
            "failure": self._failure,
        }
        fingerprint = _sha256(semantic_payload)
        trace_id = "trace-" + _sha256(self._base_identity())[:20]
        trace = RuleAuthoringTrace(
            trace_id=trace_id,
            workflow_id=self.workflow_id,
            target=target,
            context=context,
            provider=provider,
            started_at=self.started_at,
            completed_at=completed,
            outcome=outcome,
            transitions=tuple(self._transitions),
            tool_calls=tuple(self._tools),
            retrieval=self._retrieval,
            draft=self._draft,
            validation_status=self._validation_status,
            dry_run_status=self._dry_run_status,
            approval_id=self._approval_id,
            execution_result_id=self._execution_result_id,
            retry_count=self._retry_count,
            fallback_used=self._fallback_used,
            fallback_reason=self._fallback_reason,
            failure=self._failure,
            semantic_fingerprint=fingerprint,
        )
        trace.to_dict()
        return trace


__all__ = [
    "MAX_TRACE_TOOL_CALLS",
    "MAX_TRACE_TRANSITIONS",
    "RuleAuthoringTrace",
    "RuleAuthoringTraceBuilder",
    "TRACE_SCHEMA_VERSION",
    "ToolCallTrace",
    "TraceTransition",
    "summarize_tool_arguments",
]
