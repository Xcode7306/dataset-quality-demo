"""v0.9 本地标准依据 RAG 的纯领域模型。

这些对象只描述已批准文档、可定位片段、检索结果和冲突状态；不包含向量
数据库、模型客户端或 Streamlit 状态。文档内容来自标准、数据字典和用户
明确批准的规范文件，不包含上传业务数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


RAG_SCHEMA_VERSION = "0.9"
RAG_PARSER_VERSION = "quality-rag-parser-v0.9"
RAG_NAMESPACE_STANDARDS = "standards"
RAG_NAMESPACE_DATA_DICTIONARY = "data_dictionary"
RAG_NAMESPACE_USER_SPEC = "user_specification"
RAG_NAMESPACES = frozenset(
    {
        RAG_NAMESPACE_STANDARDS,
        RAG_NAMESPACE_DATA_DICTIONARY,
        RAG_NAMESPACE_USER_SPEC,
    }
)
RAG_DOCUMENT_STATUSES = frozenset(
    {"active", "draft", "superseded", "expired", "unknown"}
)
RAG_RETRIEVAL_STATUSES = frozenset({"ok", "no_results", "conflict"})

RagDocumentStatus = Literal[
    "active", "draft", "superseded", "expired", "unknown"
]
RagRetrievalStatus = Literal["ok", "no_results", "conflict"]


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


@dataclass(frozen=True)
class RagDocument:
    """已摄取文档的可审计元数据。"""

    document_id: str
    title: str
    standard_number: str | None
    version: str | None
    published_at: str | None
    effective_status: RagDocumentStatus
    parser_version: str
    source_namespace: str
    source_name: str
    source_path: str | None
    content_sha256: str
    ingested_at: str
    approved: bool
    chunk_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = RAG_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "title": self.title,
            "standard_number": self.standard_number,
            "version": self.version,
            "published_at": self.published_at,
            "effective_status": self.effective_status,
            "parser_version": self.parser_version,
            "source_namespace": self.source_namespace,
            "source_name": self.source_name,
            "source_path": self.source_path,
            "content_sha256": self.content_sha256,
            "ingested_at": self.ingested_at,
            "approved": self.approved,
            "chunk_ids": list(self.chunk_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RagChunk:
    """文档中的稳定、可展示检索片段。"""

    chunk_id: str
    document_id: str
    ordinal: int
    text: str
    section: str | None = None
    clause: str | None = None
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    table_row: bool = False
    metric_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "ordinal": self.ordinal,
            "text": self.text,
            "section": self.section,
            "clause": self.clause,
            "page": self.page,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "table_row": self.table_row,
            "metric_ids": list(self.metric_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RagCitation:
    """面向页面和 RuleEvidence 的来源定位摘要。"""

    chunk_id: str
    document_id: str
    document_name: str
    document_version: str | None
    standard_number: str | None
    section: str | None
    clause: str | None
    page: int | None
    line_start: int | None
    line_end: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_version": self.document_version,
            "standard_number": self.standard_number,
            "section": self.section,
            "clause": self.clause,
            "page": self.page,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


@dataclass(frozen=True)
class RagSearchResult:
    """单个检索结果；score 只用于排序，不作为引用依据。"""

    chunk: RagChunk
    document: RagDocument
    score: float
    matched_terms: tuple[str, ...] = ()

    @property
    def citation(self) -> RagCitation:
        return RagCitation(
            chunk_id=self.chunk.chunk_id,
            document_id=self.document.document_id,
            document_name=self.document.title,
            document_version=self.document.version,
            standard_number=self.document.standard_number,
            section=self.chunk.section,
            clause=self.chunk.clause,
            page=self.chunk.page,
            line_start=self.chunk.line_start,
            line_end=self.chunk.line_end,
        )

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        payload = {
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.document.document_id,
            "document_name": self.document.title,
            "document_version": self.document.version,
            "standard_number": self.document.standard_number,
            "source_namespace": self.document.source_namespace,
            "section": self.chunk.section,
            "clause": self.chunk.clause,
            "page": self.chunk.page,
            "line_start": self.chunk.line_start,
            "line_end": self.chunk.line_end,
            "score": round(float(self.score), 6),
            "matched_terms": list(self.matched_terms),
            "citation": self.citation.to_dict(),
        }
        if include_text:
            payload["text"] = self.chunk.text
        return payload


@dataclass(frozen=True)
class RagConflict:
    """同一检索意图命中多个版本或冲突来源时的显式状态。"""

    reason: str
    document_ids: tuple[str, ...]
    document_labels: tuple[str, ...]
    versions: tuple[str, ...]
    standard_numbers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "document_ids": list(self.document_ids),
            "document_labels": list(self.document_labels),
            "versions": list(self.versions),
            "standard_numbers": list(self.standard_numbers),
        }


@dataclass(frozen=True)
class RagSearchResponse:
    """检索返回的完整、可绑定到工作流的结果。"""

    query: str
    status: RagRetrievalStatus
    results: tuple[RagSearchResult, ...] = ()
    conflict: RagConflict | None = None
    filtered_document_count: int = 0
    total_candidate_count: int = 0
    namespace: str | None = None
    metric_id: str | None = None
    standard_number: str | None = None
    version: str | None = None

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        return {
            "query": self.query,
            "status": self.status,
            "results": [
                item.to_dict(include_text=include_text) for item in self.results
            ],
            "conflict": self.conflict.to_dict() if self.conflict else None,
            "filtered_document_count": self.filtered_document_count,
            "total_candidate_count": self.total_candidate_count,
            "namespace": self.namespace,
            "metric_id": self.metric_id,
            "standard_number": self.standard_number,
            "version": self.version,
        }


@dataclass(frozen=True)
class RagDocumentBundle:
    """一次摄取产生的文档和片段。"""

    document: RagDocument
    chunks: tuple[RagChunk, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }


__all__ = [
    "RAG_DOCUMENT_STATUSES",
    "RAG_NAMESPACE_DATA_DICTIONARY",
    "RAG_NAMESPACE_STANDARDS",
    "RAG_NAMESPACE_USER_SPEC",
    "RAG_NAMESPACES",
    "RAG_PARSER_VERSION",
    "RAG_RETRIEVAL_STATUSES",
    "RAG_SCHEMA_VERSION",
    "RagChunk",
    "RagCitation",
    "RagConflict",
    "RagDocument",
    "RagDocumentBundle",
    "RagDocumentStatus",
    "RagRetrievalStatus",
    "RagSearchResponse",
    "RagSearchResult",
]
