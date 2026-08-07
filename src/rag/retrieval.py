"""v0.9 本地混合全文检索。

首版采用确定性的关键词、短语、标题/条款权重和中文字符 n-gram 组合评分，
不要求向量数据库或外部服务。过滤先于排序，检索结果只来自已批准文档。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence

from .ingestion import ingest_document_bytes, ingest_document_path
from .models import (
    RAG_NAMESPACE_STANDARDS,
    RAG_NAMESPACE_USER_SPEC,
    RAG_NAMESPACES,
    RagDocument,
    RagDocumentBundle,
    RagSearchResponse,
    RagSearchResult,
    RagConflict,
)


MAX_QUERY_LENGTH = 2_000
MAX_RESULTS = 20
MAX_RESULT_TEXT_CHARS = 2_400
MAX_TOTAL_RESULT_CHARS = 12_000
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:-]+|[\u3400-\u9fff]")
_CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")


class RagRetrievalError(ValueError):
    """检索参数或知识库状态不符合边界。"""


def _tokens(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text.casefold()):
        token = match.group(0)
        if _CJK_PATTERN.fullmatch(token):
            values.append(token)
        elif len(token) >= 2:
            values.append(token)
    # 中文双字片段帮助没有空格的标准名称检索，同时保留原词和编号。
    cjk_text = "".join(_CJK_PATTERN.findall(text.casefold()))
    values.extend(cjk_text[index : index + 2] for index in range(len(cjk_text) - 1))
    return tuple(dict.fromkeys(item for item in values if item))


def _contains_phrase(text: str, query: str) -> bool:
    return query.casefold().strip() in text.casefold()


def _normalize_filter(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"\s+", "", str(value)).casefold()


def _score(query: str, result: RagSearchResult) -> tuple[float, tuple[str, ...]]:
    query_terms = set(_tokens(query))
    if not query_terms:
        return 0.0, ()
    chunk_text = result.chunk.text.casefold()
    title_text = result.document.title.casefold()
    section_text = (result.chunk.section or "").casefold()
    clause_text = (result.chunk.clause or "").casefold()
    matched: list[str] = []
    score = 0.0
    for term in query_terms:
        in_chunk = term in chunk_text
        in_title = term in title_text
        in_section = term in section_text or term in clause_text
        if in_chunk or in_title or in_section:
            matched.append(term)
        if in_chunk:
            score += 1.0
        if in_title:
            score += 2.0
        if in_section:
            score += 1.5
    if _contains_phrase(chunk_text, query):
        score += 4.0
    if matched and result.chunk.table_row:
        score += 0.15
    return score, tuple(sorted(set(matched)))


def _conflict_for(results: Sequence[RagSearchResult]) -> RagConflict | None:
    if not results:
        return None
    top_score = max(item.score for item in results)
    relevant = [item for item in results if item.score >= top_score * 0.55]
    documents = {item.document.document_id: item.document for item in relevant}
    if len(documents) < 2:
        return None
    standard_groups: dict[str, set[str]] = {}
    for document in documents.values():
        key = document.standard_number or document.document_id
        standard_groups.setdefault(key, set()).add(document.version or "未标注版本")
    conflicting_groups = {
        key: versions for key, versions in standard_groups.items() if len(versions) > 1
    }
    if not conflicting_groups:
        # 同一版本下的不同用户规范也不能被静默拼接成一个权威结论。
        namespaces = {item.document.source_namespace for item in relevant}
        if len(namespaces) < 2:
            return None
        reason = "检索结果来自多个已批准来源命名空间，请确认适用来源。"
    else:
        reason = "同一标准或适用范围命中了多个版本，请先确认适用版本。"
    ordered = sorted(documents.values(), key=lambda item: item.document_id)
    return RagConflict(
        reason=reason,
        document_ids=tuple(item.document_id for item in ordered),
        document_labels=tuple(item.title for item in ordered),
        versions=tuple(item.version or "未标注版本" for item in ordered),
        standard_numbers=tuple(
            item.standard_number or "未标注标准号" for item in ordered
        ),
    )


@dataclass
class RagKnowledgeBase:
    """当前会话内存知识库；不同 namespace 的文档不会互相混用。"""

    _documents: dict[str, RagDocument] | None = None
    _chunks: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self._documents is None:
            self._documents = {}
        if self._chunks is None:
            self._chunks = {}

    @property
    def documents(self) -> tuple[RagDocument, ...]:
        assert self._documents is not None
        return tuple(
            sorted(self._documents.values(), key=lambda item: (item.title, item.document_id))
        )

    def add_bundle(self, bundle: RagDocumentBundle) -> RagDocument:
        if not isinstance(bundle, RagDocumentBundle):
            raise RagRetrievalError("知识库只能接收 RagDocumentBundle。")
        assert self._documents is not None and self._chunks is not None
        previous = self._documents.get(bundle.document.document_id)
        if previous is not None:
            for chunk_id in previous.chunk_ids:
                self._chunks.pop(chunk_id, None)
        self._documents[bundle.document.document_id] = bundle.document
        for chunk in bundle.chunks:
            self._chunks[chunk.chunk_id] = chunk
        return bundle.document

    def ingest_bytes(self, raw: bytes, source_name: str, **kwargs: Any) -> RagDocument:
        bundle = ingest_document_bytes(raw, source_name, **kwargs)
        return self.add_bundle(bundle)

    def ingest_path(self, path: str, **kwargs: Any) -> RagDocument:
        bundle = ingest_document_path(path, **kwargs)
        return self.add_bundle(bundle)

    def remove_document(self, document_id: str) -> bool:
        assert self._documents is not None and self._chunks is not None
        document = self._documents.pop(document_id, None)
        if document is None:
            return False
        for chunk_id in document.chunk_ids:
            self._chunks.pop(chunk_id, None)
        return True

    def get_chunk(self, chunk_id: str):
        assert self._chunks is not None
        return self._chunks.get(chunk_id)

    def search(
        self,
        query: str,
        *,
        metric_id: str | None = None,
        standard_number: str | None = None,
        version: str | None = None,
        source_namespace: str | None = None,
        limit: int = 5,
    ) -> RagSearchResponse:
        text = str(query or "").strip()
        if not text:
            raise RagRetrievalError("检索查询不能为空。")
        if len(text) > MAX_QUERY_LENGTH:
            raise RagRetrievalError(f"检索查询不能超过 {MAX_QUERY_LENGTH} 个字符。")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_RESULTS:
            raise RagRetrievalError(f"检索结果数量必须在 1 到 {MAX_RESULTS} 之间。")
        if source_namespace is not None and source_namespace not in RAG_NAMESPACES:
            raise RagRetrievalError("检索命名空间不受支持。")

        metric_filter = _normalize_filter(metric_id)
        standard_filter = _normalize_filter(standard_number)
        version_filter = _normalize_filter(version)
        assert self._documents is not None and self._chunks is not None
        documents = [
            document
            for document in self._documents.values()
            if document.approved
            and document.effective_status == "active"
            and (source_namespace is None or document.source_namespace == source_namespace)
            and (
                standard_filter is None
                or _normalize_filter(document.standard_number) == standard_filter
            )
            and (
                version_filter is None
                or _normalize_filter(document.version) == version_filter
            )
        ]
        document_map = {document.document_id: document for document in documents}
        candidates = []
        for chunk in self._chunks.values():
            document = document_map.get(chunk.document_id)
            if document is None:
                continue
            if metric_filter is not None and metric_filter not in {
                item.casefold() for item in chunk.metric_ids
            }:
                continue
            candidates.append(RagSearchResult(chunk=chunk, document=document, score=0.0))

        scored: list[RagSearchResult] = []
        for candidate in candidates:
            score, matched_terms = _score(text, candidate)
            if score > 0:
                scored.append(
                    RagSearchResult(
                        chunk=candidate.chunk,
                        document=candidate.document,
                        score=score,
                        matched_terms=matched_terms,
                    )
                )
        scored.sort(
            key=lambda item: (
                -item.score,
                item.document.document_id,
                item.chunk.ordinal,
            )
        )
        selected: list[RagSearchResult] = []
        total_chars = 0
        for item in scored:
            if len(selected) >= limit:
                break
            remaining = MAX_TOTAL_RESULT_CHARS - total_chars
            if remaining <= 0:
                break
            text_excerpt = item.chunk.text[: min(MAX_RESULT_TEXT_CHARS, remaining)]
            if text_excerpt != item.chunk.text:
                chunk = type(item.chunk)(
                    **{
                        **item.chunk.to_dict(),
                        "text": text_excerpt,
                        "metric_ids": tuple(item.chunk.metric_ids),
                    }
                )
                item = RagSearchResult(
                    chunk=chunk,
                    document=item.document,
                    score=item.score,
                    matched_terms=item.matched_terms,
                )
            selected.append(item)
            total_chars += len(item.chunk.text)
        conflict = _conflict_for(selected)
        status = "conflict" if conflict else ("ok" if selected else "no_results")
        return RagSearchResponse(
            query=text,
            status=status,  # type: ignore[arg-type]
            results=tuple(selected),
            conflict=conflict,
            filtered_document_count=len(documents),
            total_candidate_count=len(candidates),
            namespace=source_namespace,
            metric_id=metric_id,
            standard_number=standard_number,
            version=version,
        )

    def summary(self) -> dict[str, Any]:
        documents = self.documents
        return {
            "document_count": len(documents),
            "approved_document_count": sum(item.approved for item in documents),
            "chunk_count": sum(len(item.chunk_ids) for item in documents),
            "namespaces": sorted({item.source_namespace for item in documents}),
            "documents": [
                {
                    "document_id": item.document_id,
                    "title": item.title,
                    "standard_number": item.standard_number,
                    "version": item.version,
                    "ingested_at": item.ingested_at,
                    "effective_status": item.effective_status,
                    "approved": item.approved,
                    "source_namespace": item.source_namespace,
                    "chunk_count": len(item.chunk_ids),
                }
                for item in documents
            ],
        }


def build_default_knowledge_base(project_dir: str | None = None) -> RagKnowledgeBase:
    """加载项目标准文档和 v0.6 实现口径作为首批已批准语料。"""

    from pathlib import Path

    root = Path(project_dir) if project_dir else Path(__file__).resolve().parents[2]
    project_root = root if (root / "DB31T_1523-2024_指标与v0.6实现口径.md").is_file() else root / "dataset_quality_demo"
    workspace_root = project_root.parent
    knowledge_base = RagKnowledgeBase()
    standard_path = next(
        (
            candidate
            for candidate in (
                root / "DB31T_1523-2024_公共数据质量评价指标及计算方式.md",
                workspace_root / "DB31T_1523-2024_公共数据质量评价指标及计算方式.md",
            )
            if candidate.is_file()
        ),
        None,
    )
    if standard_path is not None:
        knowledge_base.ingest_path(
            str(standard_path),
            title="DB31/T 1523-2024 公共数据质量评价指标及计算方式",
            standard_number="DB31/T 1523-2024",
            version="2024",
            source_namespace=RAG_NAMESPACE_STANDARDS,
            approved=True,
            metadata={"origin": "project_preloaded", "approved_by": "project"},
        )
    implementation_path = next(
        (
            candidate
            for candidate in (
                project_root / "DB31T_1523-2024_指标与v0.6实现口径.md",
                root / "DB31T_1523-2024_指标与v0.6实现口径.md",
            )
            if candidate.is_file()
        ),
        None,
    )
    if implementation_path is not None:
        knowledge_base.ingest_path(
            str(implementation_path),
            title="DB31/T 1523-2024 指标与 v0.6 实现口径",
            standard_number="DB31/T 1523-2024",
            version="v0.6",
            source_namespace=RAG_NAMESPACE_USER_SPEC,
            approved=True,
            metadata={"origin": "project_preloaded", "approved_by": "project"},
        )
    return knowledge_base


__all__ = [
    "MAX_QUERY_LENGTH",
    "MAX_RESULTS",
    "RagKnowledgeBase",
    "RagRetrievalError",
    "build_default_knowledge_base",
]
