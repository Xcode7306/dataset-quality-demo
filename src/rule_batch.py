"""Pre-evaluation natural-language rule batches and safe rule-file imports.

The model still produces one bounded :class:`RuleDraft` at a time.  This module
only coordinates those drafts, refuses partial batches, and combines validated
drafts into one deterministic ``RulePack`` for dry-run and human approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Literal, Mapping, Sequence

import pandas as pd

from .metric_catalog import ALL_METRIC_IDS, get_metric_definition
from .parser import DatasetReadError, parse_dataset
from .rule_authoring_service import (
    compile_custom_rule_draft,
    compile_rule_draft,
    validate_rule_draft,
)
from .rule_dsl import RuleDraft, RuleDraftValidationResult
from .rule_pack import MAX_RULES, RulePack, RulePackValidationError, build_rule_pack


MAX_RULE_IMPORT_BYTES = 2 * 1024 * 1024
MAX_RULE_TEXT_LENGTH = 4000
MAX_RULE_FILE_NAME_LENGTH = 255
SUPPORTED_RULE_IMPORT_EXTENSIONS = frozenset(
    {".txt", ".md", ".csv", ".xls", ".xlsx", ".json", ".jsonl", ".ndjson"}
)

RuleInputOrigin = Literal["metric_supplement", "dialog", "file_import"]

_RULE_COLUMN_ALIASES = (
    "规则描述",
    "规则",
    "评判规则",
    "评价规则",
    "要求",
    "description",
    "rule",
    "user_intent",
)
_METRIC_COLUMN_ALIASES = (
    "指标ID",
    "指标id",
    "指标名称",
    "metric_id",
    "metric",
)
_MARKDOWN_BULLET = re.compile(r"^\s*(?:[-*+]\s+|\d{1,3}[.)、]\s*)")
_RULE_LABEL_PREFIX = re.compile(r"^\s*(?:规则|要求)\s*\d*\s*[:：]\s*")


class RuleImportError(ValueError):
    """The uploaded rule description file cannot be safely interpreted."""


@dataclass(frozen=True)
class RuleBatchInput:
    """One user-authored description that must compile into one RuleDraft."""

    item_id: str
    origin: RuleInputOrigin
    user_intent: str
    label: str
    target_metric_id: str | None = None
    source_name: str | None = None
    source_location: str | None = None

    @classmethod
    def create(
        cls,
        *,
        origin: RuleInputOrigin,
        user_intent: str,
        label: str,
        target_metric_id: str | None = None,
        source_name: str | None = None,
        source_location: str | None = None,
    ) -> "RuleBatchInput":
        intent = str(user_intent or "").strip()
        if not intent:
            raise RuleImportError("规则描述不能为空。")
        if len(intent) > MAX_RULE_TEXT_LENGTH:
            raise RuleImportError(
                f"单条规则描述不能超过 {MAX_RULE_TEXT_LENGTH} 个字符。"
            )
        try:
            intent.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise RuleImportError("规则描述包含非法 Unicode 字符。") from error
        normalized_label = str(label or "").strip() or "未命名规则"
        fingerprint = json.dumps(
            [
                origin,
                intent,
                target_metric_id,
                source_name,
                source_location,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
        item_id = f"rule-input-{hashlib.sha256(fingerprint).hexdigest()[:20]}"
        return cls(
            item_id=item_id,
            origin=origin,
            user_intent=intent,
            label=normalized_label[:300],
            target_metric_id=(
                str(target_metric_id).strip() if target_metric_id else None
            ),
            source_name=str(source_name).strip()[:MAX_RULE_FILE_NAME_LENGTH]
            if source_name
            else None,
            source_location=str(source_location).strip()[:300]
            if source_location
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "origin": self.origin,
            "user_intent": self.user_intent,
            "label": self.label,
            "target_metric_id": self.target_metric_id,
            "source_name": self.source_name,
            "source_location": self.source_location,
        }


@dataclass(frozen=True)
class RuleImportResult:
    items: tuple[RuleBatchInput, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuleBatchItemResult:
    request: RuleBatchInput
    draft: RuleDraft | None = None
    validation: RuleDraftValidationResult | None = None
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error:
            return "failed"
        if self.draft is None:
            return "failed"
        if self.draft.status == "needs_clarification":
            return "needs_clarification"
        if self.draft.status == "rejected":
            return "unsupported"
        if self.validation is not None and self.validation.valid:
            return "ready"
        return "invalid"

    @property
    def messages(self) -> tuple[str, ...]:
        if self.error:
            return (self.error,)
        if self.draft is None:
            return ("没有生成规则草案。",)
        if self.draft.status == "needs_clarification":
            return tuple(self.draft.clarification_questions) or (
                "规则描述还缺少可执行信息。",
            )
        if self.draft.status == "rejected":
            return (self.draft.unsupported_reason or "当前规则超出支持范围。",)
        if self.validation is not None and not self.validation.valid:
            return tuple(self.validation.errors)
        return ()


@dataclass(frozen=True)
class RuleBatchPreflight:
    """All item outcomes plus the combined draft pack when every item is ready."""

    items: tuple[RuleBatchItemResult, ...]
    draft_pack: RulePack | None = None
    warnings: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return bool(self.items) and all(item.status == "ready" for item in self.items) and (
            self.draft_pack is not None
        )

    @property
    def blocking_count(self) -> int:
        return sum(item.status != "ready" for item in self.items)


def _decode_text(content: bytes) -> tuple[str, str]:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding, errors="strict"), encoding
        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error.reason}")
    raise RuleImportError(
        "规则文件无法按 UTF-8 或 GB18030 解码：" + "；".join(errors)
    )


def _strict_json_loads(text: str, *, location: str) -> Any:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuleImportError(f"{location} 包含重复 JSON 键：{key}。")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RuleImportError(f"{location} 包含非标准数值：{value}。")
            ),
        )
    except RuleImportError:
        raise
    except (TypeError, ValueError) as error:
        raise RuleImportError(f"{location} 不是严格 JSON：{error}。") from error


def _clean_rule_line(value: Any) -> str:
    text = str(value or "").strip()
    text = _MARKDOWN_BULLET.sub("", text)
    text = _RULE_LABEL_PREFIX.sub("", text)
    return text.strip()


def _inputs_from_rows(
    rows: Iterable[tuple[Any, Any, str]],
    *,
    source_name: str,
) -> RuleImportResult:
    items: list[RuleBatchInput] = []
    warnings: list[str] = []
    seen: set[tuple[str, str | None]] = set()
    for raw_description, raw_metric, location in rows:
        description = _clean_rule_line(raw_description)
        metric = str(raw_metric).strip() if raw_metric is not None else ""
        if not description:
            continue
        signature = (description, metric or None)
        if signature in seen:
            warnings.append(f"{location} 与前文规则重复，已忽略。")
            continue
        seen.add(signature)
        items.append(
            RuleBatchInput.create(
                origin="file_import",
                user_intent=description,
                label=f"{source_name} · {location}",
                target_metric_id=metric or None,
                source_name=source_name,
                source_location=location,
            )
        )
        if len(items) > MAX_RULES:
            raise RuleImportError(
                f"规则文件最多包含 {MAX_RULES} 条非重复规则。"
            )
    if not items:
        raise RuleImportError("规则文件中没有找到非空规则描述。")
    return RuleImportResult(tuple(items), tuple(dict.fromkeys(warnings)))


def _parse_text_rule_file(text: str, *, source_name: str) -> RuleImportResult:
    rows: list[tuple[str, None, str]] = []
    in_code_fence = False
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence or not stripped or stripped.startswith("#"):
            continue
        rows.append((stripped, None, f"第 {line_number} 行"))
    return _inputs_from_rows(rows, source_name=source_name)


def _column_name(columns: Sequence[Any], aliases: Sequence[str]) -> Any | None:
    normalized = {str(column).strip().casefold(): column for column in columns}
    for alias in aliases:
        if alias.casefold() in normalized:
            return normalized[alias.casefold()]
    return None


def _parse_tabular_rule_file(
    content: bytes,
    *,
    suffix: str,
    source_name: str,
) -> RuleImportResult:
    try:
        with TemporaryDirectory(prefix="rule-import-") as temporary_directory:
            path = Path(temporary_directory) / f"rules{suffix}"
            path.write_bytes(content)
            parsed = parse_dataset(path)
            dataframe = parsed.dataframe
    except DatasetReadError as error:
        raise RuleImportError(f"规则表格无法读取：{error}") from error
    rule_column = _column_name(tuple(dataframe.columns), _RULE_COLUMN_ALIASES)
    if rule_column is None:
        raise RuleImportError(
            "规则表格缺少规则描述列；请使用“规则描述”或 rule/description 列。"
        )
    metric_column = _column_name(tuple(dataframe.columns), _METRIC_COLUMN_ALIASES)
    rows: list[tuple[Any, Any, str]] = []
    for row_number, (_, row) in enumerate(dataframe.iterrows(), start=2):
        description = row[rule_column]
        metric = row[metric_column] if metric_column is not None else None
        if pd.isna(description):
            description = ""
        if metric is not None and pd.isna(metric):
            metric = None
        rows.append((description, metric, f"第 {row_number} 行"))
    return _inputs_from_rows(rows, source_name=source_name)


def _json_entry(entry: Any, *, location: str) -> tuple[Any, Any, str]:
    if isinstance(entry, str):
        return entry, None, location
    if not isinstance(entry, Mapping):
        raise RuleImportError(f"{location} 必须是字符串或规则对象。")
    rule_key = _column_name(tuple(entry), _RULE_COLUMN_ALIASES)
    if rule_key is None:
        raise RuleImportError(
            f"{location} 缺少规则描述字段；请使用 description、rule 或“规则描述”。"
        )
    metric_key = _column_name(tuple(entry), _METRIC_COLUMN_ALIASES)
    return entry[rule_key], entry.get(metric_key) if metric_key is not None else None, location


def _parse_json_rule_file(text: str, *, source_name: str) -> RuleImportResult:
    payload = _strict_json_loads(text, location="规则 JSON")
    if isinstance(payload, Mapping):
        unknown = set(payload) - {"rules", "规则"}
        key = "rules" if "rules" in payload else "规则" if "规则" in payload else None
        if key is None or unknown:
            raise RuleImportError("规则 JSON 顶层必须是数组，或只包含 rules/规则 数组。")
        payload = payload[key]
    if not isinstance(payload, list):
        raise RuleImportError("规则 JSON 顶层必须是规则数组。")
    rows = [
        _json_entry(entry, location=f"第 {index} 项")
        for index, entry in enumerate(payload, start=1)
    ]
    return _inputs_from_rows(rows, source_name=source_name)


def _parse_jsonl_rule_file(text: str, *, source_name: str) -> RuleImportResult:
    rows: list[tuple[Any, Any, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        entry = _strict_json_loads(line, location=f"第 {line_number} 行")
        rows.append(_json_entry(entry, location=f"第 {line_number} 行"))
    return _inputs_from_rows(rows, source_name=source_name)


def parse_rule_import(content: bytes, file_name: str) -> RuleImportResult:
    """Parse a bounded rule-description file without accepting executable DSL/code."""

    if not isinstance(content, (bytes, bytearray)):
        raise RuleImportError("规则文件内容必须是字节。")
    raw = bytes(content)
    if not raw:
        raise RuleImportError("规则文件为空。")
    if len(raw) > MAX_RULE_IMPORT_BYTES:
        raise RuleImportError(
            f"规则文件不能超过 {MAX_RULE_IMPORT_BYTES // (1024 * 1024)} MiB。"
        )
    safe_name = Path(str(file_name or "")).name
    if not safe_name or len(safe_name) > MAX_RULE_FILE_NAME_LENGTH:
        raise RuleImportError("规则文件名为空或过长。")
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_RULE_IMPORT_EXTENSIONS:
        supported = "、".join(sorted(SUPPORTED_RULE_IMPORT_EXTENSIONS))
        raise RuleImportError(f"不支持该规则文件类型；当前支持：{supported}。")
    if suffix in {".csv", ".xls", ".xlsx"}:
        return _parse_tabular_rule_file(
            raw,
            suffix=suffix,
            source_name=safe_name,
        )
    text, encoding = _decode_text(raw)
    if suffix in {".txt", ".md"}:
        result = _parse_text_rule_file(text, source_name=safe_name)
    elif suffix == ".json":
        result = _parse_json_rule_file(text, source_name=safe_name)
    else:
        result = _parse_jsonl_rule_file(text, source_name=safe_name)
    warnings = list(result.warnings)
    if encoding not in {"utf-8", "utf-8-sig"}:
        warnings.append(f"规则文件使用 {encoding.upper()} 编码读取。")
    return RuleImportResult(result.items, tuple(dict.fromkeys(warnings)))


def rule_inputs_from_dialog(text: str) -> tuple[RuleBatchInput, ...]:
    """Create one custom rule input per non-empty line in the dialog."""

    items: list[RuleBatchInput] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(str(text or "").splitlines(), start=1):
        intent = _clean_rule_line(raw_line)
        if not intent or intent in seen:
            continue
        seen.add(intent)
        items.append(
            RuleBatchInput.create(
                origin="dialog",
                user_intent=intent,
                label=f"对话框第 {line_number} 行",
                source_location=f"第 {line_number} 行",
            )
        )
    if len(items) > MAX_RULES:
        raise RuleImportError(f"一次最多生成 {MAX_RULES} 条规则。")
    return tuple(items)


def _resolve_metric_id(value: str | None) -> str | None:
    if not value:
        return None
    requested = str(value).strip()
    if requested in ALL_METRIC_IDS:
        return requested
    matches = [
        metric_id
        for metric_id in ALL_METRIC_IDS
        if str((get_metric_definition(metric_id) or {}).get("name", "")).strip()
        == requested
    ]
    if len(matches) == 1:
        return matches[0]
    raise RuleImportError(f"未知或不唯一的指标 ID/名称：{requested}。")


def _safe_error(error: Exception) -> str:
    text = str(error).strip() or type(error).__name__
    text = re.sub(
        r"(?i)\b(api[_ -]?key|token|secret|password)\b\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", text)
    return text[:1000]


def build_rule_pack_from_drafts(
    drafts: Sequence[RuleDraft],
    report: Any,
    *,
    name: str = "AI 生成规则批次",
    version: str = "1.1.0",
    generated_at: datetime | str | None = None,
) -> tuple[RulePack, tuple[str, ...]]:
    """Combine only fully validated drafts; identical generated rules are deduped."""

    if not drafts:
        raise RulePackValidationError(("规则批次不能为空。",))
    rules = []
    evidence: dict[str, Mapping[str, Any]] = {}
    warnings: list[str] = []
    seen_rule_ids: set[str] = set()
    has_standard_evidence = False
    for draft in drafts:
        validation = validate_rule_draft(draft, report)
        if not validation.valid or draft.rule_spec is None:
            raise RulePackValidationError(
                tuple(validation.errors) or (f"草案 {draft.draft_id} 尚不可执行。",)
            )
        rule = draft.rule_spec.to_rule()
        if rule.rule_id in seen_rule_ids:
            warnings.append(f"规则 {rule.rule_id} 与前一条语义相同，合并时已去重。")
            continue
        seen_rule_ids.add(rule.rule_id)
        rules.append(rule)
        for item in draft.evidence:
            evidence[item.id] = item.to_dict()
            if item.type in {"standard_clause", "data_dictionary"}:
                has_standard_evidence = True
    if not rules:
        raise RulePackValidationError(("规则批次去重后没有可执行规则。",))
    if len(rules) > MAX_RULES:
        raise RulePackValidationError((f"规则批次不能超过 {MAX_RULES} 条。",))
    source_type = "standard_retrieval" if has_standard_evidence else "user_natural_language"
    generator = "quality-rule-agent-v0.9" if has_standard_evidence else "quality-rule-agent-v0.8"
    pack = build_rule_pack(
        report,
        name=name,
        version=version,
        rules=tuple(rules),
        generated_at=generated_at,
        source_type=source_type,
        generator=generator,
        evidence=tuple(evidence.values()),
    )
    return pack, tuple(dict.fromkeys(warnings))


def compile_rule_batch(
    report: Any,
    requests: Sequence[RuleBatchInput],
    *,
    provider: Any | None = None,
    allow_template_fallback: bool = True,
    rag_response: Any | None = None,
    selected_chunk_ids: Iterable[str] = (),
    created_at: datetime | str | None = None,
) -> RuleBatchPreflight:
    """Compile and validate every request; never build a partial RulePack."""

    if not requests:
        return RuleBatchPreflight(())
    if len(requests) > MAX_RULES:
        raise RuleImportError(f"一次最多处理 {MAX_RULES} 条规则。")
    chunks = tuple(dict.fromkeys(str(item) for item in selected_chunk_ids))
    results: list[RuleBatchItemResult] = []
    ready_drafts: list[RuleDraft] = []
    for request in requests:
        try:
            target_metric_id = _resolve_metric_id(request.target_metric_id)
            if target_metric_id:
                draft = compile_rule_draft(
                    report,
                    target_metric_id=target_metric_id,
                    user_intent=request.user_intent,
                    provider=provider,
                    created_at=created_at,
                    allow_template_fallback=allow_template_fallback,
                    rag_response=rag_response,
                    selected_chunk_ids=chunks,
                )
            else:
                draft = compile_custom_rule_draft(
                    report,
                    user_intent=request.user_intent,
                    provider=provider,
                    created_at=created_at,
                    allow_template_fallback=allow_template_fallback,
                    rag_response=rag_response,
                    selected_chunk_ids=chunks,
                )
            validation = (
                validate_rule_draft(draft, report)
                if draft.status == "draft"
                else None
            )
            item_result = RuleBatchItemResult(
                request=request,
                draft=draft,
                validation=validation,
            )
            if item_result.status == "ready":
                ready_drafts.append(draft)
        except Exception as error:
            item_result = RuleBatchItemResult(
                request=request,
                error=_safe_error(error),
            )
        results.append(item_result)

    if any(item.status != "ready" for item in results):
        return RuleBatchPreflight(tuple(results))
    try:
        pack, warnings = build_rule_pack_from_drafts(
            ready_drafts,
            report,
            generated_at=created_at,
        )
    except Exception as error:
        return RuleBatchPreflight(
            tuple(results),
            warnings=(f"批量 RulePack 构建失败：{_safe_error(error)}",),
        )
    return RuleBatchPreflight(tuple(results), pack, warnings)


__all__ = [
    "MAX_RULE_IMPORT_BYTES",
    "RuleBatchInput",
    "RuleBatchItemResult",
    "RuleBatchPreflight",
    "RuleImportError",
    "RuleImportResult",
    "SUPPORTED_RULE_IMPORT_EXTENSIONS",
    "build_rule_pack_from_drafts",
    "compile_rule_batch",
    "parse_rule_import",
    "rule_inputs_from_dialog",
]
