"""v0.9 标准依据文档摄取与条款感知分段。

首版只把用户明确提供、项目预置或已明确批准的 Markdown、TXT 和可抽取
PDF 转成内存中的 ``RagDocumentBundle``。摄取模块不创建向量数据库、不写
入业务数据，也不保存原始上传文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    RAG_NAMESPACE_STANDARDS,
    RAG_NAMESPACES,
    RAG_PARSER_VERSION,
    RagChunk,
    RagDocument,
    RagDocumentBundle,
)


MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
MAX_DOCUMENT_CHARS = 1_000_000
MAX_DOCUMENT_CHUNKS = 5_000
MAX_CHUNK_CHARS = 3_500
MAX_METADATA_TEXT = 300

_STANDARD_NUMBER_PATTERN = re.compile(
    r"\b(?:DB\s*\d{1,4}\s*/\s*T|GB\s*/\s*T|GB|ISO|HJ|CJJ|T/)[\s-]*"
    r"[A-Z0-9][A-Z0-9./_-]{1,40}\b",
    re.IGNORECASE,
)
_CLAUSE_PATTERN = re.compile(
    r"^\s*(?P<clause>第\s*[0-9一二三四五六七八九十百]+\s*[章节条款项]?|"
    r"[0-9]+(?:\.[0-9]+){0,6}|表\s*[0-9]+|附录\s*[A-Z])"
    r"(?:[、.．：:\s]|$)(?P<rest>.*)$",
    re.IGNORECASE,
)
_HEADING_PATTERN = re.compile(r"^\s*(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
_METRIC_CODE_PATTERN = re.compile(r"(?<!\d)(\d{6})(?!\d)")
_DATE_PATTERN = re.compile(
    r"(?:发布日期|发布(?:日期|时间)|生效日期|发布时间)\s*[:：]?\s*"
    r"(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})"
)
_VERSION_PATTERN = re.compile(
    r"(?:版本|修订版|版次)\s*[:：]?\s*([A-Za-z0-9][A-Za-z0-9._-]{0,31})",
    re.IGNORECASE,
)


class RagIngestionError(ValueError):
    """文档不满足 v0.9 摄取边界。"""


@dataclass(frozen=True)
class _SourceLine:
    text: str
    line_number: int
    page: int | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8", errors="strict")


def _clean_text(value: Any, *, label: str, maximum: int = MAX_METADATA_TEXT) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > maximum:
        raise RagIngestionError(f"{label}不能超过 {maximum} 个字符。")
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise RagIngestionError(f"{label}包含无法编码的字符。") from error
    return text


def _validate_namespace(namespace: str) -> str:
    value = str(namespace or RAG_NAMESPACE_STANDARDS).strip()
    if value not in RAG_NAMESPACES:
        raise RagIngestionError(
            f"文档命名空间“{value}”不受支持，只能使用 {sorted(RAG_NAMESPACES)}。"
        )
    return value


def _decode_text(raw: bytes, *, source_name: str) -> str:
    if not raw:
        raise RagIngestionError(f"文档“{source_name}”为空。")
    for encoding in ("utf-8-sig", "utf-16", "utf-32"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" not in text:
            return text
    raise RagIngestionError(
        f"文档“{source_name}”不是受支持的 UTF-8/UTF-16/UTF-32 文本。"
    )


def _extract_pdf_pages(raw: bytes, *, source_name: str) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RagIngestionError(
            "PDF 摄取需要安装 pypdf；Markdown/TXT 摄取不依赖额外运行时。"
        ) from error
    try:
        reader = PdfReader(BytesIO(raw), strict=True)
        pages: list[tuple[int, str]] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append((index, text))
    except Exception as error:
        raise RagIngestionError(f"PDF 文档“{source_name}”无法抽取文本。") from error
    if not any(text.strip() for _, text in pages):
        raise RagIngestionError(f"PDF 文档“{source_name}”没有可抽取的文本层。")
    return pages


def _metadata_from_text(text: str) -> dict[str, str | None]:
    first_heading = next(
        (
            match.group("title").strip()
            for line in text.splitlines()
            if (match := _HEADING_PATTERN.match(line))
        ),
        None,
    )
    standard_match = _STANDARD_NUMBER_PATTERN.search(text)
    version_match = _VERSION_PATTERN.search(text)
    date_match = _DATE_PATTERN.search(text)
    return {
        "title": first_heading,
        "standard_number": standard_match.group(0) if standard_match else None,
        "version": version_match.group(1) if version_match else None,
        "published_at": date_match.group(1).replace("/", "-") if date_match else None,
    }


def _normalized_standard(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"\s+", "", value).replace("－", "-").upper()


def _metric_ids(text: str) -> tuple[str, ...]:
    ids = {
        f"db31_{match.group(1)}" for match in _METRIC_CODE_PATTERN.finditer(text)
    }
    return tuple(sorted(ids))


def _line_blocks(lines: Sequence[_SourceLine]) -> list[tuple[list[_SourceLine], str | None, str | None]]:
    """按标题、条款、段落和表格行形成上下文完整的块。"""

    blocks: list[tuple[list[_SourceLine], str | None, str | None]] = []
    current: list[_SourceLine] = []
    section: str | None = None
    clause: str | None = None

    def flush() -> None:
        nonlocal current
        if any(item.text.strip() for item in current):
            blocks.append((current, section, clause))
        current = []

    for item in lines:
        text = item.text.rstrip()
        heading = _HEADING_PATTERN.match(text)
        clause_match = _CLAUSE_PATTERN.match(text)
        if heading:
            flush()
            section = heading.group("title").strip()
            clause = None
            current = [item]
            continue
        if clause_match:
            flush()
            clause = clause_match.group("clause").strip()
            current = [item]
            continue
        if not text.strip():
            flush()
            continue
        if current and current[-1].text.lstrip().startswith("|") != text.lstrip().startswith("|"):
            flush()
        current.append(_SourceLine(text=text, line_number=item.line_number, page=item.page))
        if text.lstrip().startswith("|") and len(current) >= 8:
            flush()
    flush()
    return blocks


def _split_oversized(
    source_lines: Sequence[_SourceLine],
    *,
    section: str | None,
    clause: str | None,
) -> Iterable[tuple[list[_SourceLine], str | None, str | None]]:
    current: list[_SourceLine] = []
    length = 0
    for item in source_lines:
        extra = len(item.text) + (1 if current else 0)
        if current and length + extra > MAX_CHUNK_CHARS:
            yield current, section, clause
            overlap = current[-1:] if not current[-1].text.lstrip().startswith("|") else []
            current = list(overlap)
            length = sum(len(line.text) + 1 for line in current)
        current.append(item)
        length += extra
    if current:
        yield current, section, clause


def _build_chunks(
    lines: Sequence[_SourceLine],
    *,
    document_id: str,
) -> tuple[RagChunk, ...]:
    chunks: list[RagChunk] = []
    for block, section, clause in _line_blocks(lines):
        for split_block, split_section, split_clause in _split_oversized(
            block,
            section=section,
            clause=clause,
        ):
            text = "\n".join(item.text.strip() for item in split_block).strip()
            if not text:
                continue
            ordinal = len(chunks)
            payload = [document_id, ordinal, text, split_section, split_clause]
            chunk_id = f"chunk-{hashlib.sha256(_canonical(payload)).hexdigest()[:20]}"
            pages = {item.page for item in split_block if item.page is not None}
            page = min(pages) if len(pages) == 1 else None
            table_row = any(item.text.lstrip().startswith("|") for item in split_block)
            chunks.append(
                RagChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    ordinal=ordinal,
                    text=text,
                    section=split_section,
                    clause=split_clause,
                    page=page,
                    line_start=split_block[0].line_number,
                    line_end=split_block[-1].line_number,
                    table_row=table_row,
                    metric_ids=_metric_ids(text),
                )
            )
            if len(chunks) > MAX_DOCUMENT_CHUNKS:
                raise RagIngestionError(
                    f"文档分段数量超过 {MAX_DOCUMENT_CHUNKS} 的上限。"
                )
    if not chunks:
        raise RagIngestionError("文档没有可检索的正文片段。")
    return tuple(chunks)


def _pages_to_lines(pages: Sequence[tuple[int | None, str]]) -> list[_SourceLine]:
    lines: list[_SourceLine] = []
    line_number = 0
    for page, text in pages:
        for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line_number += 1
            lines.append(_SourceLine(raw_line, line_number, page))
    return lines


def ingest_document_bytes(
    raw: bytes,
    source_name: str,
    *,
    title: str | None = None,
    standard_number: str | None = None,
    version: str | None = None,
    published_at: str | None = None,
    effective_status: str = "active",
    source_namespace: str = RAG_NAMESPACE_STANDARDS,
    source_path: str | None = None,
    approved: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> RagDocumentBundle:
    """将一份已批准或待批准文档摄取为稳定文档和条款感知片段。"""

    if not isinstance(raw, (bytes, bytearray)):
        raise RagIngestionError("文档内容必须是 bytes。")
    raw = bytes(raw)
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise RagIngestionError(
            f"文档大小不能超过 {MAX_DOCUMENT_BYTES // (1024 * 1024)} MiB。"
        )
    source_name_text = _clean_text(source_name, label="source_name", maximum=300)
    if source_name_text is None:
        raise RagIngestionError("source_name 不能为空。")
    namespace = _validate_namespace(source_namespace)
    status = str(effective_status or "unknown").strip().casefold()
    if status not in {"active", "draft", "superseded", "expired", "unknown"}:
        raise RagIngestionError("effective_status 不在允许范围内。")

    suffix = Path(source_name_text).suffix.casefold()
    if suffix == ".pdf":
        page_text = _extract_pdf_pages(raw, source_name=source_name_text)
        pages: list[tuple[int | None, str]] = page_text
    elif suffix in {".md", ".markdown", ".txt"}:
        pages = [(None, _decode_text(raw, source_name=source_name_text))]
    else:
        raise RagIngestionError("v0.9 仅支持 Markdown、TXT 和可抽取 PDF。")

    full_text = "\n".join(text for _, text in pages)
    if len(full_text) > MAX_DOCUMENT_CHARS:
        raise RagIngestionError(
            f"文档文本不能超过 {MAX_DOCUMENT_CHARS} 个字符。"
        )
    inferred = _metadata_from_text(full_text)
    final_title = _clean_text(title, label="title") or inferred["title"] or Path(
        source_name_text
    ).stem
    final_standard = _clean_text(
        standard_number or inferred["standard_number"],
        label="standard_number",
    )
    final_version = _clean_text(version or inferred["version"], label="version")
    final_date = _clean_text(
        published_at or inferred["published_at"],
        label="published_at",
    )
    source_path_text = _clean_text(source_path, label="source_path", maximum=500)
    extra_metadata = dict(metadata or {})
    try:
        json.dumps(extra_metadata, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise RagIngestionError("metadata 必须是可安全序列化的 JSON 对象。") from error

    content_sha256 = hashlib.sha256(raw).hexdigest()
    document_id = f"doc-{hashlib.sha256(_canonical([source_name_text, final_title, final_standard, final_version, content_sha256])).hexdigest()[:20]}"
    lines = _pages_to_lines(pages)
    chunks = _build_chunks(lines, document_id=document_id)
    document = RagDocument(
        document_id=document_id,
        title=final_title,
        standard_number=_normalized_standard(final_standard),
        version=final_version,
        published_at=final_date,
        effective_status=status,  # type: ignore[arg-type]
        parser_version=RAG_PARSER_VERSION,
        source_namespace=namespace,
        source_name=source_name_text,
        source_path=source_path_text,
        content_sha256=content_sha256,
        ingested_at=_utc_now(),
        approved=bool(approved),
        chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
        metadata=extra_metadata,
    )
    return RagDocumentBundle(document=document, chunks=chunks)


def ingest_document_path(
    path: str | Path,
    **kwargs: Any,
) -> RagDocumentBundle:
    """从本地路径摄取文档；只读取文件，不将路径写入检索文本。"""

    candidate = Path(path)
    try:
        if not candidate.is_file():
            raise RagIngestionError("指定文档不存在或不是普通文件。")
        raw = candidate.read_bytes()
    except RagIngestionError:
        raise
    except OSError as error:
        raise RagIngestionError("读取文档失败。") from error
    return ingest_document_bytes(
        raw,
        candidate.name,
        source_path=candidate.name,
        **kwargs,
    )


__all__ = [
    "MAX_CHUNK_CHARS",
    "MAX_DOCUMENT_BYTES",
    "MAX_DOCUMENT_CHARS",
    "MAX_DOCUMENT_CHUNKS",
    "RagIngestionError",
    "ingest_document_bytes",
    "ingest_document_path",
]
