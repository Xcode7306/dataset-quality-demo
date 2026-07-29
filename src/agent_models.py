"""Agent 输出协议与严格的模型草稿校验。

质量报告和 Agent 解读是两个独立的协议。模型只能生成受限草稿，最终
``AgentAnalysis`` 由本地服务在完成引用、数字与报告绑定校验后组装。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal, Mapping, Sequence


AgentIntent = Literal["summary", "priority", "not_assessable", "question"]
ActionPriority = Literal["high", "medium", "low"]
CitationSourceType = Literal["summary", "risk", "metric", "not_assessable"]

SUPPORTED_INTENTS: frozenset[str] = frozenset(
    {"summary", "priority", "not_assessable", "question"}
)


class AgentOutputValidationError(ValueError):
    """模型输出不符合 Agent 草稿协议或语义约束。"""


@dataclass(frozen=True)
class AgentFact:
    id: str
    text: str
    citation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "citation_ids": list(self.citation_ids),
        }


@dataclass(frozen=True)
class AgentAction:
    id: str
    priority: ActionPriority
    title: str
    detail: str
    citation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "priority": self.priority,
            "title": self.title,
            "detail": self.detail,
            "citation_ids": list(self.citation_ids),
        }


@dataclass(frozen=True)
class AgentLimitation:
    id: str
    text: str
    citation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "citation_ids": list(self.citation_ids),
        }


@dataclass(frozen=True)
class AgentAnswer:
    text: str
    citation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citation_ids": list(self.citation_ids),
        }


@dataclass(frozen=True)
class AgentCitation:
    id: str
    source_type: CitationSourceType
    source_id: str
    label: str
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "label": self.label,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True)
class AgentAudit:
    provider: str
    model: str | None
    mode: Literal["template", "model"]
    prompt_version: str
    fallback_used: bool
    fallback_reason: str | None
    tool_calls: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "prompt_version": self.prompt_version,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "tool_calls": list(self.tool_calls),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
        }


@dataclass(frozen=True)
class AgentAnalysis:
    """与 ``QualityReport`` 分离的只读解读结果。"""

    schema_version: str
    report_sha256: str
    intent: AgentIntent
    facts: tuple[AgentFact, ...]
    actions: tuple[AgentAction, ...]
    limitations: tuple[AgentLimitation, ...]
    answer: AgentAnswer
    citations: tuple[AgentCitation, ...]
    audit: AgentAudit

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_sha256": self.report_sha256,
            "intent": self.intent,
            "facts": [item.to_dict() for item in self.facts],
            "actions": [item.to_dict() for item in self.actions],
            "limitations": [item.to_dict() for item in self.limitations],
            "answer": self.answer.to_dict(),
            "citations": [item.to_dict() for item in self.citations],
            "audit": self.audit.to_dict(),
        }


PROVIDER_DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["facts", "actions", "limitations", "answer"],
    "properties": {
        "facts": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "citation_ids"],
                "properties": {
                    "text": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "citation_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {"type": "string", "minLength": 1, "maxLength": 300},
                    },
                },
            },
        },
        "actions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["priority", "title", "detail", "citation_ids"],
                "properties": {
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "title": {"type": "string", "minLength": 1, "maxLength": 300},
                    "detail": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "citation_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {"type": "string", "minLength": 1, "maxLength": 300},
                    },
                },
            },
        },
        "limitations": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "citation_ids"],
                "properties": {
                    "text": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "citation_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {"type": "string", "minLength": 1, "maxLength": 300},
                    },
                },
            },
        },
        "answer": {
            "type": "object",
            "additionalProperties": False,
            "required": ["text", "citation_ids"],
            "properties": {
                "text": {"type": "string", "minLength": 1, "maxLength": 3000},
                "citation_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {"type": "string", "minLength": 1, "maxLength": 300},
                },
            },
        },
    },
}


_DRAFT_KEYS = frozenset({"facts", "actions", "limitations", "answer"})
_STATEMENT_KEYS = frozenset({"text", "citation_ids"})
_ACTION_KEYS = frozenset(
    {"priority", "title", "detail", "citation_ids"}
)
_ANSWER_KEYS = frozenset({"text", "citation_ids"})


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    location: str,
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise AgentOutputValidationError(
            f"{location} 字段不匹配；缺少 {missing}，多出 {unknown}。"
        )


def _require_text(value: Any, location: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise AgentOutputValidationError(f"{location} 必须是字符串。")
    normalized = value.strip()
    if not normalized or len(normalized) > maximum:
        raise AgentOutputValidationError(
            f"{location} 必须为 1 到 {maximum} 个字符。"
        )
    return normalized


def _require_citations(value: Any, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise AgentOutputValidationError(
            f"{location} 必须包含 1 到 8 个引用。"
        )
    citations = tuple(
        _require_text(item, f"{location}[{index}]", 300)
        for index, item in enumerate(value)
    )
    if len(set(citations)) != len(citations):
        raise AgentOutputValidationError(f"{location} 不得包含重复引用。")
    return citations


def _require_object(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AgentOutputValidationError(f"{location} 必须是对象。")
    if not all(isinstance(key, str) for key in value):
        raise AgentOutputValidationError(f"{location} 的字段名必须是字符串。")
    return value


def _require_array(
    value: Any,
    location: str,
    *,
    minimum: int = 1,
    maximum: int = 8,
) -> list[Any]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise AgentOutputValidationError(
            f"{location} 必须包含 {minimum} 到 {maximum} 项。"
        )
    return value


def parse_provider_draft(payload: Any) -> dict[str, Any]:
    """将模型返回值解析成严格、规范化的 Agent 草稿。

    该函数故意不进行宽松修复。非法 JSON、额外字段、缺失引用或超长内容
    都由上层触发模板回退，避免把未验证内容带入最终分析。
    """

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError) as error:
            raise AgentOutputValidationError("模型未返回合法 JSON。") from error
    root = _require_object(payload, "root")
    _require_exact_keys(root, _DRAFT_KEYS, "root")

    facts: list[dict[str, Any]] = []
    for index, raw_fact in enumerate(_require_array(root["facts"], "facts")):
        fact = _require_object(raw_fact, f"facts[{index}]")
        _require_exact_keys(fact, _STATEMENT_KEYS, f"facts[{index}]")
        facts.append(
            {
                "text": _require_text(fact["text"], f"facts[{index}].text", 2000),
                "citation_ids": _require_citations(
                    fact["citation_ids"], f"facts[{index}].citation_ids"
                ),
            }
        )

    actions: list[dict[str, Any]] = []
    for index, raw_action in enumerate(_require_array(root["actions"], "actions")):
        action = _require_object(raw_action, f"actions[{index}]")
        _require_exact_keys(action, _ACTION_KEYS, f"actions[{index}]")
        priority = action["priority"]
        if priority not in {"high", "medium", "low"}:
            raise AgentOutputValidationError(
                f"actions[{index}].priority 不在允许范围内。"
            )
        actions.append(
            {
                "priority": priority,
                "title": _require_text(
                    action["title"], f"actions[{index}].title", 300
                ),
                "detail": _require_text(
                    action["detail"], f"actions[{index}].detail", 2000
                ),
                "citation_ids": _require_citations(
                    action["citation_ids"], f"actions[{index}].citation_ids"
                ),
            }
        )

    limitations: list[dict[str, Any]] = []
    for index, raw_limitation in enumerate(
        _require_array(root["limitations"], "limitations")
    ):
        limitation = _require_object(raw_limitation, f"limitations[{index}]")
        _require_exact_keys(
            limitation, _STATEMENT_KEYS, f"limitations[{index}]"
        )
        limitations.append(
            {
                "text": _require_text(
                    limitation["text"], f"limitations[{index}].text", 2000
                ),
                "citation_ids": _require_citations(
                    limitation["citation_ids"],
                    f"limitations[{index}].citation_ids",
                ),
            }
        )

    raw_answer = _require_object(root["answer"], "answer")
    _require_exact_keys(raw_answer, _ANSWER_KEYS, "answer")
    answer = {
        "text": _require_text(raw_answer["text"], "answer.text", 3000),
        "citation_ids": _require_citations(
            raw_answer["citation_ids"], "answer.citation_ids"
        ),
    }
    return {
        "facts": facts,
        "actions": actions,
        "limitations": limitations,
        "answer": answer,
    }


def collect_draft_citation_ids(draft: Mapping[str, Any]) -> tuple[str, ...]:
    """按首次出现顺序收集草稿引用。"""

    citation_ids: list[str] = []
    for collection_name in ("facts", "actions", "limitations"):
        for item in draft[collection_name]:
            citation_ids.extend(item["citation_ids"])
    citation_ids.extend(draft["answer"]["citation_ids"])
    return tuple(dict.fromkeys(citation_ids))


def make_analysis(
    *,
    report_sha256: str,
    intent: AgentIntent,
    draft: Mapping[str, Any],
    citations: Sequence[AgentCitation],
    audit: AgentAudit,
) -> AgentAnalysis:
    """从已经通过本地语义验证的草稿组装最终协议对象。"""

    return AgentAnalysis(
        schema_version="0.1",
        report_sha256=report_sha256,
        intent=intent,
        facts=tuple(
            AgentFact(
                id=f"fact-{index}",
                text=item["text"],
                citation_ids=tuple(item["citation_ids"]),
            )
            for index, item in enumerate(draft["facts"], start=1)
        ),
        actions=tuple(
            AgentAction(
                id=f"action-{index}",
                priority=item["priority"],
                title=item["title"],
                detail=item["detail"],
                citation_ids=tuple(item["citation_ids"]),
            )
            for index, item in enumerate(draft["actions"], start=1)
        ),
        limitations=tuple(
            AgentLimitation(
                id=f"limitation-{index}",
                text=item["text"],
                citation_ids=tuple(item["citation_ids"]),
            )
            for index, item in enumerate(draft["limitations"], start=1)
        ),
        answer=AgentAnswer(
            text=draft["answer"]["text"],
            citation_ids=tuple(draft["answer"]["citation_ids"]),
        ),
        citations=tuple(citations),
        audit=audit,
    )
