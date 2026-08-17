"""v0.9 检索片段到 RuleEvidence 的确定性引用绑定。"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from ..rule_dsl import RuleEvidence, new_evidence
from .models import RagSearchResponse, RagSearchResult


class RagCitationError(ValueError):
    """引用不存在、来源不一致或缺少定位信息。"""


def evidence_from_result(
    result: RagSearchResult,
    *,
    evidence_type: str | None = None,
) -> RuleEvidence:
    """把已检索结果转换为带文档、版本、条款和 chunk 定位的依据。"""

    source_type = evidence_type or (
        "standard_clause"
        if result.document.source_namespace == "standards"
        else (
            "data_dictionary"
            if result.document.source_namespace == "data_dictionary"
            else "user_statement"
        )
    )
    if source_type not in {
        "standard_clause",
        "data_dictionary",
        "user_statement",
    }:
        raise RagCitationError("RAG 结果来源类型不受支持。")
    citation = result.citation
    section = citation.section or ("正文" if citation.clause is None else None)
    location_parts = [
        item
        for item in (
            section,
            citation.clause,
            f"第{citation.page}页" if citation.page is not None else None,
            (
                f"行 {citation.line_start}-{citation.line_end}"
                if citation.line_start is not None and citation.line_end is not None
                else None
            ),
            f"chunk:{citation.chunk_id}",
        )
        if item
    ]
    return new_evidence(
        source_type,  # type: ignore[arg-type]
        result.chunk.text,
        source_id=citation.chunk_id,
        source_label=citation.document_name,
        location=" / ".join(location_parts),
        authoritative=(
            bool(result.document.approved)
            and result.document.effective_status == "active"
            and result.document.source_namespace != "user_specification"
        ),
        document_id=citation.document_id,
        document_name=citation.document_name,
        document_version=citation.document_version,
        section=section,
        clause=citation.clause,
        chunk_id=citation.chunk_id,
        page=citation.page,
    )


def evidence_from_response(
    response: RagSearchResponse,
    *,
    selected_chunk_ids: Sequence[str] | None = None,
) -> tuple[RuleEvidence, ...]:
    """只将本次检索返回的片段绑定为依据；不存在的 ID 直接失败。"""

    selected = (
        tuple(dict.fromkeys(selected_chunk_ids))
        if selected_chunk_ids is not None
        else tuple(item.chunk.chunk_id for item in response.results)
    )
    result_map = {item.chunk.chunk_id: item for item in response.results}
    missing = [chunk_id for chunk_id in selected if chunk_id not in result_map]
    if missing:
        raise RagCitationError(f"检索结果中不存在所选片段：{'、'.join(missing)}。")
    if response.status == "no_results" and selected:
        raise RagCitationError("无检索结果时不能绑定标准依据。")
    if response.status == "conflict":
        raise RagCitationError(
            response.conflict.reason if response.conflict else "检索来源存在冲突。"
        )
    if not selected:
        return ()
    return tuple(evidence_from_result(result_map[chunk_id]) for chunk_id in selected)


def validate_evidence_against_response(
    evidence: Iterable[RuleEvidence],
    response: RagSearchResponse | None,
) -> tuple[str, ...]:
    """验证 RuleDraft 中的 RAG 引用仍来自本次检索结果。"""

    items = tuple(evidence)
    rag_items = tuple(
        item
        for item in items
        if item.type in {"standard_clause", "data_dictionary"}
    )
    if not rag_items:
        return ()
    if response is None:
        return ("规则包含标准依据，但缺少本次检索结果绑定。",)
    if response.status == "conflict":
        return (response.conflict.reason if response.conflict else "检索来源存在冲突。",)
    allowed = {item.chunk.chunk_id: item for item in response.results}
    errors: list[str] = []
    seen: set[str] = set()
    for item in rag_items:
        chunk_id = item.chunk_id or item.source_id
        if not chunk_id or chunk_id not in allowed:
            errors.append(f"依据 {item.id} 没有对应的本次检索片段。")
            continue
        if chunk_id in seen:
            errors.append(f"依据重复引用检索片段 {chunk_id}。")
        seen.add(chunk_id)
        result = allowed[chunk_id]
        citation = result.citation
        if item.document_id != citation.document_id:
            errors.append(f"依据 {item.id} 的文档 ID 与检索结果不一致。")
        if item.document_name != citation.document_name:
            errors.append(f"依据 {item.id} 的文档名称与检索结果不一致。")
        if item.document_version != citation.document_version:
            errors.append(f"依据 {item.id} 的文档版本与检索结果不一致。")
        if item.chunk_id != citation.chunk_id:
            errors.append(f"依据 {item.id} 的 chunk_id 与检索结果不一致。")
    return tuple(dict.fromkeys(errors))


def response_source_summary(response: RagSearchResponse) -> list[dict[str, object]]:
    """为页面提供不含分数细节的来源展示。"""

    return [
        {
            "chunk_id": item.chunk.chunk_id,
            "document_name": item.document.title,
            "version": item.document.version,
            "standard_number": item.document.standard_number,
            "section": item.chunk.section,
            "clause": item.chunk.clause,
            "page": item.chunk.page,
            "location": item.to_dict(include_text=False)["citation"],
        }
        for item in response.results
    ]


__all__ = [
    "RagCitationError",
    "evidence_from_response",
    "evidence_from_result",
    "response_source_summary",
    "validate_evidence_against_response",
]
