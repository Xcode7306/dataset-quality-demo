"""v0.9 本地标准依据 RAG 公共接口。"""

from .citations import (
    RagCitationError,
    evidence_from_response,
    evidence_from_result,
    response_source_summary,
    validate_evidence_against_response,
)
from .ingestion import (
    MAX_CHUNK_CHARS,
    MAX_DOCUMENT_BYTES,
    MAX_DOCUMENT_CHARS,
    MAX_DOCUMENT_CHUNKS,
    RagIngestionError,
    ingest_document_bytes,
    ingest_document_path,
)
from .models import (
    RAG_NAMESPACE_DATA_DICTIONARY,
    RAG_NAMESPACE_STANDARDS,
    RAG_NAMESPACE_USER_SPEC,
    RagChunk,
    RagCitation,
    RagConflict,
    RagDocument,
    RagDocumentBundle,
    RagSearchResponse,
    RagSearchResult,
)
from .retrieval import (
    MAX_QUERY_LENGTH,
    MAX_RESULTS,
    RagKnowledgeBase,
    RagRetrievalError,
    build_default_knowledge_base,
)


__all__ = [
    "MAX_CHUNK_CHARS",
    "MAX_DOCUMENT_BYTES",
    "MAX_DOCUMENT_CHARS",
    "MAX_DOCUMENT_CHUNKS",
    "MAX_QUERY_LENGTH",
    "MAX_RESULTS",
    "RAG_NAMESPACE_DATA_DICTIONARY",
    "RAG_NAMESPACE_STANDARDS",
    "RAG_NAMESPACE_USER_SPEC",
    "RagChunk",
    "RagCitation",
    "RagCitationError",
    "RagConflict",
    "RagDocument",
    "RagDocumentBundle",
    "RagIngestionError",
    "RagKnowledgeBase",
    "RagRetrievalError",
    "RagSearchResponse",
    "RagSearchResult",
    "build_default_knowledge_base",
    "evidence_from_response",
    "evidence_from_result",
    "ingest_document_bytes",
    "ingest_document_path",
    "response_source_summary",
    "validate_evidence_against_response",
]
