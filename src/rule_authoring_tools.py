"""v0.9 规则编制工具。

这些工具只从指标目录和脱敏画像构建上下文，不读取原始单元格值，也不执行
审批、文件写入或正式规则重评。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .metric_catalog import get_metric_definition
from .text_utils import contains_unsafe_unicode_controls


RULE_AUTHORING_TOOL_POLICIES: Mapping[str, Mapping[str, Any]] = {
    "get_metric_definition": {
        "required": frozenset({"metric_id"}),
        "optional": frozenset(),
    },
    "get_profile_summary": {"required": frozenset(), "optional": frozenset()},
    "list_available_fields": {"required": frozenset(), "optional": frozenset()},
    "retrieve_rule_evidence": {
        "required": frozenset({"query"}),
        "optional": frozenset(
            {
                "metric_id",
                "standard_number",
                "version",
                "source_namespace",
                "limit",
            }
        ),
    },
    "validate_rule_draft": {
        "required": frozenset({"draft_id"}),
        "optional": frozenset(),
    },
    "dry_run_rule": {
        "required": frozenset({"draft_id"}),
        "optional": frozenset(),
    },
}
RULE_AUTHORING_TOOL_NAMES = frozenset(RULE_AUTHORING_TOOL_POLICIES)


class RuleAuthoringToolRequestError(ValueError):
    """A model-requested tool or argument is outside the no-side-effect allowlist."""


def _tool_text(value: Any, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuleAuthoringToolRequestError(f"{label}必须是非空字符串。")
    text = value.strip()
    if len(text) > maximum:
        raise RuleAuthoringToolRequestError(f"{label}超过 {maximum} 个字符。")
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise RuleAuthoringToolRequestError(f"{label}包含非法 Unicode。") from error
    if contains_unsafe_unicode_controls(text):
        raise RuleAuthoringToolRequestError(f"{label}不能包含 Unicode 控制字符。")
    return text


def validate_rule_authoring_tool_request(
    tool_name: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a proposed tool call without executing it or touching UI state."""

    name = str(tool_name or "").strip()
    policy = RULE_AUTHORING_TOOL_POLICIES.get(name)
    if policy is None:
        raise RuleAuthoringToolRequestError(f"工具“{name or '空'}”不在规则编制白名单中。")
    if not isinstance(arguments, Mapping):
        raise RuleAuthoringToolRequestError("工具参数必须是 JSON 对象。")
    payload = dict(arguments)
    required = set(policy["required"])
    allowed = required | set(policy["optional"])
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - allowed)
    if missing:
        raise RuleAuthoringToolRequestError(f"工具 {name} 缺少参数：{missing}。")
    if unknown:
        raise RuleAuthoringToolRequestError(f"工具 {name} 包含未允许参数：{unknown}。")

    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "query":
            normalized[key] = _tool_text(value, "query", maximum=2000)
        elif key in {"metric_id", "draft_id"}:
            normalized[key] = _tool_text(value, key, maximum=120)
        elif key in {"standard_number", "version", "source_namespace"}:
            normalized[key] = (
                None if value is None else _tool_text(value, key, maximum=200)
            )
        elif key == "limit":
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 20:
                raise RuleAuthoringToolRequestError("limit 必须是 1 到 20 的整数。")
            normalized[key] = value
    return normalized


def _report_payload(report: Any) -> Mapping[str, Any]:
    to_dict = getattr(report, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("report 必须提供 to_dict()。")
    payload = to_dict()
    if not isinstance(payload, Mapping):
        raise TypeError("report.to_dict() 必须返回对象。")
    return payload


def get_metric_definition_tool(metric_id: str) -> dict[str, Any]:
    definition = get_metric_definition(metric_id)
    if definition is None:
        return {"metric_id": metric_id, "found": False}
    keys = (
        "id",
        "name",
        "category",
        "dimension",
        "description",
        "formula",
        "direction",
        "auto_assessable",
        "reason_code",
        "required_inputs",
        "available_proxy_metric_ids",
    )
    return {
        "metric_id": metric_id,
        "found": True,
        **{
            key: list(definition[key])
            if key in {"required_inputs", "available_proxy_metric_ids"}
            else definition.get(key)
            for key in keys
        },
    }


def list_available_fields_tool(report: Any) -> dict[str, Any]:
    payload = _report_payload(report)
    profile = payload.get("profile")
    columns = profile.get("columns", []) if isinstance(profile, Mapping) else []
    fields = []
    if isinstance(columns, list):
        for column in columns:
            if not isinstance(column, Mapping) or not isinstance(column.get("name"), str):
                continue
            fields.append(
                {
                    "name": column["name"],
                    "inferred_type": str(column.get("inferred_type", "unknown")),
                    "missing_rate": column.get("missing_rate"),
                    "non_missing_count": column.get("non_missing_count"),
                }
            )
    return {"fields": fields}


def get_profile_summary_tool(report: Any) -> dict[str, Any]:
    payload = _report_payload(report)
    profile = payload.get("profile")
    if not isinstance(profile, Mapping):
        return {"row_count": 0, "column_count": 0, "warnings": []}
    return {
        "row_count": profile.get("row_count", 0),
        "column_count": profile.get("column_count", 0),
        "warnings": list(profile.get("warnings", []))
        if isinstance(profile.get("warnings", []), list)
        else [],
        "recognized_fields": {
            str(key): list(value)
            for key, value in profile.get("recognized_fields", {}).items()
            if isinstance(value, list)
        }
        if isinstance(profile.get("recognized_fields"), Mapping)
        else {},
    }


def retrieve_rule_evidence_tool(
    knowledge_base: Any,
    query: str,
    *,
    metric_id: str | None = None,
    standard_number: str | None = None,
    version: str | None = None,
    source_namespace: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """只读检索已批准标准依据，并返回可绑定的片段快照。

    该工具不接收报告或原始数据，Provider 也只能引用返回结果中的
    ``chunk_id``；空结果和来源冲突会原样暴露给上层工作流。
    """

    search = getattr(knowledge_base, "search", None)
    if not callable(search):
        raise TypeError("knowledge_base 必须提供 search()。")
    response = search(
        query,
        metric_id=metric_id,
        standard_number=standard_number,
        version=version,
        source_namespace=source_namespace,
        limit=limit,
    )
    to_dict = getattr(response, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("knowledge_base.search() 必须返回可序列化的 RAG 响应。")
    payload = to_dict(include_text=True)
    if not isinstance(payload, Mapping):
        raise TypeError("RAG 响应必须是对象。")
    return dict(payload)


def _rag_context(
    response: Any | None,
    *,
    selected_chunk_ids: Iterable[str] = (),
) -> dict[str, Any] | None:
    if response is None:
        return None
    to_dict = getattr(response, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("rag response 必须提供 to_dict()。")
    payload = to_dict(include_text=True)
    if not isinstance(payload, Mapping):
        raise TypeError("rag response.to_dict() 必须返回对象。")
    results = payload.get("results", [])
    selected = list(dict.fromkeys(str(item) for item in selected_chunk_ids))
    result_ids = {
        str(item.get("chunk_id"))
        for item in results
        if isinstance(item, Mapping) and item.get("chunk_id")
    } if isinstance(results, list) else set()
    if selected and not set(selected).issubset(result_ids):
        raise ValueError("RAG 绑定片段必须来自本次检索结果。")
    return {
        "status": payload.get("status"),
        "query": payload.get("query"),
        "namespace": payload.get("namespace"),
        "metric_id": payload.get("metric_id"),
        "standard_number": payload.get("standard_number"),
        "version": payload.get("version"),
        "chunk_ids": selected,
        "document_versions": sorted(
            {
                str(item.get("document_name"))
                + "@"
                + str(item.get("document_version") or "未标注版本")
                for item in results
                if isinstance(item, Mapping) and item.get("document_name")
            }
        ) if isinstance(results, list) else [],
        "results": [
            {
                key: item.get(key)
                for key in (
                    "chunk_id",
                    "document_id",
                    "document_name",
                    "document_version",
                    "standard_number",
                    "section",
                    "clause",
                    "page",
                    "line_start",
                    "line_end",
                    "text",
                )
                if key in item
            }
            for item in results
            if isinstance(item, Mapping)
        ][:20],
        "conflict": payload.get("conflict"),
    }


def build_rule_authoring_context(
    report: Any,
    metric_id: str,
    *,
    rag_response: Any | None = None,
    selected_chunk_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """返回给 Provider 的最小上下文白名单。"""

    payload = _report_payload(report)
    context = payload.get("evaluation_context")
    context = context if isinstance(context, Mapping) else {}
    metric = get_metric_definition_tool(metric_id)
    fields = list_available_fields_tool(report)["fields"]
    result = {
        "report_sha256": context.get("report_sha256"),
        "input_sha256": context.get("input_sha256"),
        "reference_date": context.get("reference_date"),
        "metric_catalog_version": context.get("metric_catalog_version"),
        "selected_metric_ids": list(context.get("selected_metric_ids", []))
        if isinstance(context.get("selected_metric_ids", []), list)
        else [],
        "metric": metric,
        "fields": fields,
        "profile_summary": get_profile_summary_tool(report),
    }
    rag = _rag_context(
        rag_response,
        selected_chunk_ids=selected_chunk_ids,
    )
    if rag is not None:
        result["rag"] = rag
    return result


def build_custom_rule_authoring_context(
    report: Any,
    *,
    rag_response: Any | None = None,
    selected_chunk_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """返回自定义规则编译所需的脱敏上下文，不绑定目录指标。"""

    context = build_rule_authoring_context(
        report,
        "",
        rag_response=rag_response,
        selected_chunk_ids=selected_chunk_ids,
    )
    context["target_type"] = "custom_rule"
    return context


__all__ = [
    "RULE_AUTHORING_TOOL_NAMES",
    "RULE_AUTHORING_TOOL_POLICIES",
    "RuleAuthoringToolRequestError",
    "build_rule_authoring_context",
    "build_custom_rule_authoring_context",
    "get_metric_definition_tool",
    "get_profile_summary_tool",
    "list_available_fields_tool",
    "retrieve_rule_evidence_tool",
    "validate_rule_authoring_tool_request",
]
