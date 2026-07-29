"""只读质量诊断 Agent 的统一门面。"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Any, Mapping

from .agent_models import (
    AgentAnalysis,
    AgentAudit,
    AgentIntent,
    AgentOutputValidationError,
    SUPPORTED_INTENTS,
    collect_draft_citation_ids,
    make_analysis,
    parse_provider_draft,
)
from .agent_providers import (
    AgentProvider,
    DeepSeekChatProvider,
    PROMPT_VERSION,
    ProviderExecutionError,
    ProviderResult,
    ProviderUnavailableError,
    TemplateAgentProvider,
)
from .agent_tools import (
    AgentToolError,
    ReportSnapshot,
    TOOL_NAMES,
)


_CACHE_MAX_ENTRIES = 64
_ANALYSIS_CACHE: "OrderedDict[tuple[str, str, str, str], AgentAnalysis]" = (
    OrderedDict()
)
_CACHE_LOCK = RLock()
_ANALYSIS_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "agent-analysis.schema.json"
)


@lru_cache(maxsize=1)
def _analysis_schema_validator() -> Any:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as error:
        raise RuntimeError(
            "运行时缺少 AgentAnalysis JSON Schema 校验器。"
        ) from error
    schema = json.loads(_ANALYSIS_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_agent_analysis(analysis: AgentAnalysis) -> dict[str, Any]:
    """使用发布的 JSON Schema 校验最终展示对象并返回普通字典。"""

    if not isinstance(analysis, AgentAnalysis):
        raise TypeError("analysis 必须是 AgentAnalysis。")
    payload = analysis.to_dict()
    errors = tuple(_analysis_schema_validator().iter_errors(payload))
    if errors:
        raise AgentOutputValidationError(
            "最终 AgentAnalysis 未通过已发布 JSON Schema。"
        )
    return payload


def clear_agent_cache() -> None:
    """清空进程内 Agent 缓存；主要供测试和长驻进程管理使用。"""

    with _CACHE_LOCK:
        _ANALYSIS_CACHE.clear()


def agent_cache_size() -> int:
    with _CACHE_LOCK:
        return len(_ANALYSIS_CACHE)


def _cache_get(
    key: tuple[str, str, str, str],
) -> AgentAnalysis | None:
    with _CACHE_LOCK:
        analysis = _ANALYSIS_CACHE.get(key)
        if analysis is None:
            return None
        _ANALYSIS_CACHE.move_to_end(key)
        cached = replace(
            analysis,
            audit=replace(analysis.audit, cache_hit=True),
        )
        validate_agent_analysis(cached)
        return cached


def _cache_put(
    key: tuple[str, str, str, str],
    analysis: AgentAnalysis,
) -> None:
    with _CACHE_LOCK:
        _ANALYSIS_CACHE[key] = replace(
            analysis,
            audit=replace(analysis.audit, cache_hit=False),
        )
        _ANALYSIS_CACHE.move_to_end(key)
        while len(_ANALYSIS_CACHE) > _CACHE_MAX_ENTRIES:
            _ANALYSIS_CACHE.popitem(last=False)


def _provider_namespace(provider: Any) -> str:
    namespace = getattr(provider, "cache_namespace", None)
    if isinstance(namespace, str) and namespace:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if api_key:
            namespace = namespace.replace(api_key, "[redacted]")
        return namespace[:300]
    provider_type = type(provider)
    return (
        f"custom:{provider_type.__module__}.{provider_type.__qualname__}:"
        f"{id(provider)}"
    )


def _default_provider() -> AgentProvider:
    selection = os.environ.get("QUALITY_AGENT_PROVIDER", "").strip().casefold()
    if selection == "deepseek":
        return DeepSeekChatProvider()
    return TemplateAgentProvider()


def _normalize_question(
    intent: AgentIntent,
    question: str | None,
) -> str | None:
    if intent != "question":
        return None
    if question is None:
        return None
    if not isinstance(question, str):
        raise TypeError("question 必须是字符串或 None。")
    normalized = question.strip()
    if len(normalized) > 2000:
        raise ValueError("question 最多包含 2000 个字符。")
    return normalized or None


def _safe_provider_label(provider: Any) -> str:
    if isinstance(provider, TemplateAgentProvider):
        return "template"
    if isinstance(provider, DeepSeekChatProvider):
        return "deepseek"
    label = getattr(provider, "name", None)
    if not isinstance(label, str) or not label.strip():
        label = type(provider).__name__
    return _safe_provider_value(label)


def _safe_provider_value(label: Any) -> str:
    if not isinstance(label, str) or not label.strip():
        return "custom"
    normalized = label.strip()[:100]
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if (
        "api_key" in normalized.casefold()
        or normalized.startswith("sk-")
        or bool(api_key and api_key in normalized)
    ):
        return "custom"
    return normalized


def _safe_model_value(model: Any) -> str | None:
    if not isinstance(model, str) or not model.strip():
        return None
    normalized = model.strip()[:150]
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if api_key and api_key in normalized:
        return None
    return normalized


def _safe_model_label(provider: Any) -> str | None:
    return _safe_model_value(getattr(provider, "model", None))


def _safe_prompt_version(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return PROMPT_VERSION
    normalized = value.strip()[:100]
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if api_key and api_key in normalized:
        return PROMPT_VERSION
    return normalized


def _non_negative_integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _normalize_provider_result(
    generated: Any,
    provider: AgentProvider,
    elapsed_ms: int,
) -> ProviderResult:
    if isinstance(generated, ProviderResult):
        return ProviderResult(
            payload=generated.payload,
            provider=_safe_provider_value(generated.provider),
            model=_safe_model_value(generated.model),
            mode=(
                generated.mode
                if generated.mode in {"template", "model"}
                else "model"
            ),
            prompt_version=_safe_prompt_version(generated.prompt_version),
            tool_calls=tuple(
                str(name)[:100]
                for name in generated.tool_calls
                if isinstance(name, str)
            ),
            input_tokens=_non_negative_integer(generated.input_tokens),
            output_tokens=_non_negative_integer(generated.output_tokens),
            latency_ms=_non_negative_integer(generated.latency_ms),
            available_citation_ids=(
                None
                if generated.available_citation_ids is None
                else tuple(
                    str(citation_id)[:300]
                    for citation_id in generated.available_citation_ids
                    if isinstance(citation_id, str) and citation_id
                )
            ),
        )
    return ProviderResult(
        payload=generated,
        provider=_safe_provider_label(provider),
        model=_safe_model_label(provider),
        mode=(
            "template"
            if isinstance(provider, TemplateAgentProvider)
            else "model"
        ),
        prompt_version=_safe_prompt_version(
            getattr(provider, "prompt_version", PROMPT_VERSION)
        ),
        tool_calls=tuple(
            str(name)[:100]
            for name in getattr(provider, "tool_calls", ())
            if isinstance(name, str)
        ),
        latency_ms=max(0, elapsed_ms),
    )


def _validate_citations_and_numbers(
    draft: Mapping[str, Any],
    snapshot: ReportSnapshot,
    available_citation_ids: tuple[str, ...] | None,
    *,
    validate_numbers: bool = True,
) -> None:
    cited = collect_draft_citation_ids(draft)
    for citation_id in cited:
        if citation_id not in snapshot.citation_ids:
            raise AgentOutputValidationError("模型引用不属于当前报告。")
    if (
        available_citation_ids is not None
        and not set(cited).issubset(available_citation_ids)
    ):
        raise AgentOutputValidationError(
            "模型引用了本次工具调用未返回的证据。"
        )

    if not validate_numbers:
        return

    statements: list[tuple[str, tuple[str, ...]]] = []
    for fact in draft["facts"]:
        statements.append((fact["text"], tuple(fact["citation_ids"])))
    for action in draft["actions"]:
        action_text = f"{action['title']}。{action['detail']}"
        statements.append((action_text, tuple(action["citation_ids"])))
    for limitation in draft["limitations"]:
        statements.append(
            (limitation["text"], tuple(limitation["citation_ids"]))
        )
    answer = draft["answer"]
    statements.append((answer["text"], tuple(answer["citation_ids"])))

    for text, citation_ids in statements:
        if not snapshot.statement_numbers_are_supported(
            text,
            citation_ids,
        ):
            raise AgentOutputValidationError(
                "模型文本包含无法由引用证据及其语义标签支持的数字。"
            )


def _analysis_from_result(
    result: ProviderResult,
    *,
    snapshot: ReportSnapshot,
    intent: AgentIntent,
    fallback_used: bool,
    fallback_reason: str | None,
    provider_override: str | None = None,
    model_override: str | None = None,
    latency_override_ms: int | None = None,
    input_tokens_override: int | None = None,
    output_tokens_override: int | None = None,
    prompt_version_override: str | None = None,
    tool_calls_override: tuple[str, ...] | None = None,
    validate_numbers: bool = True,
) -> AgentAnalysis:
    if any(tool_name not in TOOL_NAMES for tool_name in result.tool_calls):
        raise AgentOutputValidationError("提供方声明了非白名单工具调用。")
    draft = parse_provider_draft(result.payload)
    _validate_citations_and_numbers(
        draft,
        snapshot,
        result.available_citation_ids,
        validate_numbers=validate_numbers,
    )
    citation_ids = collect_draft_citation_ids(draft)
    citations = [snapshot.citation(citation_id) for citation_id in citation_ids]
    audit = AgentAudit(
        provider=_safe_provider_value(provider_override or result.provider),
        model=_safe_model_value(
            model_override
            if model_override is not None
            else result.model
        ),
        mode=result.mode,
        prompt_version=_safe_prompt_version(
            prompt_version_override or result.prompt_version
        ),
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        tool_calls=tool_calls_override or result.tool_calls,
        input_tokens=(
            result.input_tokens
            if input_tokens_override is None
            else input_tokens_override
        ),
        output_tokens=(
            result.output_tokens
            if output_tokens_override is None
            else output_tokens_override
        ),
        latency_ms=(
            result.latency_ms
            if latency_override_ms is None
            else latency_override_ms
        ),
    )
    analysis = make_analysis(
        report_sha256=snapshot.report_sha256,
        intent=intent,
        draft=draft,
        citations=citations,
        audit=audit,
    )
    validate_agent_analysis(analysis)
    return analysis


def _fallback_reason(error: Exception) -> str:
    if isinstance(error, ProviderUnavailableError):
        return "provider_unavailable"
    if isinstance(error, AgentOutputValidationError):
        return "invalid_model_output"
    if isinstance(error, AgentToolError):
        return "invalid_tool_or_citation"
    if isinstance(error, ProviderExecutionError):
        reason_code = getattr(error, "reason_code", "provider_error")
        if reason_code in {
            "provider_error",
            "invalid_tool_or_citation",
        }:
            return reason_code
        return "provider_error"
    return "provider_error"


def _fallback_analysis(
    *,
    snapshot: ReportSnapshot,
    intent: AgentIntent,
    question: str | None,
    attempted_provider: AgentProvider,
    error: Exception,
    attempted_result: ProviderResult | None,
    elapsed_ms: int,
) -> AgentAnalysis:
    template_result = TemplateAgentProvider().generate(
        snapshot,
        intent=intent,
        question=question,
    )
    attempted_name = (
        attempted_result.provider
        if attempted_result is not None
        else _safe_provider_label(attempted_provider)
    )
    attempted_model = (
        attempted_result.model
        if attempted_result is not None
        else _safe_model_label(attempted_provider)
    )
    attempted_prompt = (
        attempted_result.prompt_version
        if attempted_result is not None
        else str(
            getattr(attempted_provider, "prompt_version", PROMPT_VERSION)
        )[:100]
    )
    attempted_tools = (
        attempted_result.tool_calls if attempted_result is not None else ()
    )
    safe_attempted_tools = tuple(
        name for name in attempted_tools if name in TOOL_NAMES
    )
    tool_calls = (
        *safe_attempted_tools,
        *template_result.tool_calls,
    )[:6]
    return _analysis_from_result(
        template_result,
        snapshot=snapshot,
        intent=intent,
        fallback_used=True,
        fallback_reason=_fallback_reason(error),
        provider_override=attempted_name,
        model_override=attempted_model,
        latency_override_ms=max(
            elapsed_ms,
            attempted_result.latency_ms if attempted_result else 0,
        ),
        input_tokens_override=(
            attempted_result.input_tokens if attempted_result else 0
        ),
        output_tokens_override=(
            attempted_result.output_tokens if attempted_result else 0
        ),
        prompt_version_override=attempted_prompt,
        tool_calls_override=tool_calls,
        # 该结果由本地确定性模板生成；自由文本中的数字可能只是字段标识符
        # （例如字段名“2024”），不应套用针对外部模型幻觉的启发式校验。
        validate_numbers=False,
    )


def run_agent(
    report: Any,
    *,
    intent: AgentIntent = "summary",
    question: str | None = None,
    provider: AgentProvider | None = None,
    use_cache: bool = True,
) -> AgentAnalysis:
    """只读解读质量报告，并在任何提供方失败时返回本地模板结果。

    外部模型默认关闭。只有设置 ``QUALITY_AGENT_PROVIDER=deepseek`` 时才会
    选择 DeepSeek；没有 key、HTTP 客户端、网络或合法输出时均自动回退，
    不暴露异常详情。
    """

    if intent not in SUPPORTED_INTENTS:
        raise ValueError(f"不支持的 Agent intent：{intent!r}")
    normalized_question = _normalize_question(intent, question)

    to_dict = getattr(report, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("report 必须提供 to_dict()。")
    before = deepcopy(to_dict())
    snapshot = ReportSnapshot.from_report(report)
    selected_provider = provider or _default_provider()
    trusted_local_template = (
        provider is None
        and type(selected_provider) is TemplateAgentProvider
    )
    question_digest = sha256(
        (normalized_question or "").encode("utf-8")
    ).hexdigest()
    cache_key = (
        snapshot.report_sha256,
        str(intent),
        question_digest,
        _provider_namespace(selected_provider),
    )
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            after = deepcopy(to_dict())
            if after != before:
                raise RuntimeError("Agent 调用期间质量报告发生变化。")
            return cached

    started = monotonic()
    attempted_result: ProviderResult | None = None
    try:
        generated = selected_provider.generate(
            snapshot,
            intent=intent,
            question=normalized_question,
        )
        elapsed_ms = max(0, round((monotonic() - started) * 1000))
        attempted_result = _normalize_provider_result(
            generated, selected_provider, elapsed_ms
        )
        analysis = _analysis_from_result(
            attempted_result,
            snapshot=snapshot,
            intent=intent,
            fallback_used=False,
            fallback_reason=None,
            validate_numbers=not trusted_local_template,
        )
    except Exception as error:
        elapsed_ms = max(0, round((monotonic() - started) * 1000))
        partial_result = getattr(error, "partial_result", None)
        if isinstance(partial_result, ProviderResult):
            attempted_result = _normalize_provider_result(
                partial_result,
                selected_provider,
                elapsed_ms,
            )
        analysis = _fallback_analysis(
            snapshot=snapshot,
            intent=intent,
            question=normalized_question,
            attempted_provider=selected_provider,
            error=error,
            attempted_result=attempted_result,
            elapsed_ms=elapsed_ms,
        )

    after = deepcopy(to_dict())
    if after != before:
        raise RuntimeError("Agent 调用期间质量报告发生变化。")
    if use_cache and not analysis.audit.fallback_used:
        _cache_put(cache_key, analysis)
    return analysis


def _markdown_text(value: str) -> str:
    text = " ".join(value.split())
    for character in ("\\", "`", "*", "_", "[", "]", "<", ">", "#", "|"):
        text = text.replace(character, f"\\{character}")
    return text


def export_action_plan_markdown(analysis: AgentAnalysis) -> str:
    """将已经验证的行动项导出为不含原始数据的 Markdown。"""

    citations = {citation.id: citation for citation in analysis.citations}
    priority_labels = {"high": "高", "medium": "中", "low": "低"}
    lines = [
        "# 数据质量改进行动计划",
        "",
        f"- 报告标识：`{analysis.report_sha256[:12]}`",
        f"- 解读模式：{_markdown_text(analysis.audit.mode)}",
        "",
        "## 行动项",
        "",
    ]
    for action in analysis.actions:
        evidence_labels = [
            citations[citation_id].label
            for citation_id in action.citation_ids
            if citation_id in citations
        ]
        lines.extend(
            [
                (
                    f"- [ ] **{priority_labels[action.priority]}优先级 · "
                    f"{_markdown_text(action.title)}**"
                ),
                f"  - 建议：{_markdown_text(action.detail)}",
                "  - 依据：" + "、".join(
                    _markdown_text(label) for label in evidence_labels
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            *[
                f"- {_markdown_text(limitation.text)}"
                for limitation in analysis.limitations
            ],
            "",
        ]
    )
    return "\n".join(lines)
