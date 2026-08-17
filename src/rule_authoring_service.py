"""v0.9 规则编制服务。

服务负责组装上下文、调用 Provider、生成 RuleDraft 和执行确定性校验；它不
审批、不直接正式执行规则。正式 RulePack 仍由现有本地审批服务接管。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from .metric_catalog import get_metric_definition, metric_rule_type_error
from .rule_authoring_providers import (
    RuleAuthoringProvider,
    RuleAuthoringProviderError,
    RuleAuthoringProviderUnavailable,
    RuleAuthoringProviderResult,
    TemplateRuleAuthoringProvider,
    default_rule_authoring_provider,
    inspect_rule_intent,
)
from .rule_authoring_tools import (
    build_custom_rule_authoring_context,
    build_rule_authoring_context,
)
from .rag.citations import (
    RagCitationError,
    evidence_from_response,
)
from .rule_dsl import (
    RULE_DRAFT_GENERATOR,
    RULE_DRAFT_GENERATOR_V08,
    RULE_DRAFT_GENERATOR_V09,
    RULE_DRAFT_SCHEMA_VERSION,
    ProviderMetadata,
    RuleDraft,
    RuleDraftValidationError,
    RuleDraftValidationResult,
    RuleEvidence,
    make_draft_id,
    make_workflow_id,
    new_evidence,
    normalized_rule_spec,
    validate_rule_draft_shape,
    validate_rule_spec,
)
from .rule_pack import (
    RuleMetricTarget,
    RulePack,
    RulePackValidationError,
    build_rule_pack,
    validate_rule_pack,
)


def _metric_targets_for_draft(
    draft: RuleDraft,
    report: Any,
) -> tuple[RuleMetricTarget, ...]:
    """只让目录指标草案覆盖其明确绑定的、需补充依据的目标。"""

    if (
        draft.target_type != "catalog_metric"
        or not draft.target_metric_id
        or draft.rule_spec is None
    ):
        return ()
    definition = get_metric_definition(draft.target_metric_id)
    if definition is None or bool(definition.get("auto_assessable")):
        return ()
    try:
        payload = report.to_dict()
    except Exception:
        return ()
    metrics = payload.get("metrics") if isinstance(payload, Mapping) else None
    matching = [
        metric
        for metric in metrics or ()
        if isinstance(metric, Mapping)
        and metric.get("id") == draft.target_metric_id
        and metric.get("status") == "not_assessable"
    ]
    if len(matching) != 1:
        return ()
    return (
        RuleMetricTarget(
            rule_id=draft.rule_spec.rule_id,
            target_metric_id=draft.target_metric_id,
        ),
    )


def _created_at(value: datetime | str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
    if isinstance(value, datetime):
        timestamp = value
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
    text = str(value).strip()
    if text.endswith("Z"):
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    else:
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _with_fallback_metadata(
    metadata: ProviderMetadata,
    *,
    reason: str,
) -> ProviderMetadata:
    return replace(
        metadata,
        provider="template",
        model=None,
        mode="template",
        fallback_used=True,
        fallback_reason=reason,
    )


def _fallback_result(
    context: dict[str, Any],
    *,
    user_intent: str,
    provider: RuleAuthoringProvider,
    reason: str,
) -> RuleAuthoringProviderResult:
    fallback = TemplateRuleAuthoringProvider().generate(
        context,
        user_intent=user_intent,
    )
    return replace(
        fallback,
        metadata=_with_fallback_metadata(fallback.metadata, reason=reason),
    )


def _enforce_explicit_rule_inputs(
    context: Mapping[str, Any],
    *,
    user_intent: str,
    result: RuleAuthoringProviderResult,
) -> RuleAuthoringProviderResult:
    """Prevent either a template or an external model from guessing key inputs."""

    inspection = inspect_rule_intent(context, user_intent=user_intent)
    if inspection.unsupported_reason:
        return replace(
            result,
            outcome="unsupported",
            rule_spec=None,
            clarification_questions=(),
            unsupported_reason=inspection.unsupported_reason,
        )

    questions = list(inspection.clarification_questions)
    if result.outcome == "draft" and result.rule_spec is not None:
        intent_text = str(user_intent).casefold()
        if result.rule_spec.fields and all(
            field.casefold() in intent_text for field in result.rule_spec.fields
        ):
            questions = [
                question
                for question in questions
                if not (
                    "字段名称" in question
                    or "字段和" in question
                    or "全部字段" in question
                )
            ]
    if questions:
        return replace(
            result,
            outcome="clarification",
            rule_spec=None,
            clarification_questions=tuple(
                dict.fromkeys((*result.clarification_questions, *questions))
            )[:5],
            unsupported_reason=None,
        )
    return result


def _base_evidence(
    context: dict[str, Any],
    metric_id: str | None,
    user_intent: str,
) -> tuple[RuleEvidence, ...]:
    evidence_target = metric_id or "custom_rule"
    evidence = [
        new_evidence(
            "user_statement",
            user_intent,
            source_id=f"user-input:{evidence_target}",
            source_label="用户评价依据",
            location=(
                f"metric:{metric_id}:evidence"
                if metric_id
                else "custom-rule:evidence"
            ),
        )
    ]
    metric = context.get("metric", {})
    if metric_id and isinstance(metric, dict) and metric.get("found"):
        evidence.append(
            new_evidence(
                "metric_definition",
                f"{metric.get('name', metric_id)}：{metric.get('description', '')}；计算方式：{metric.get('formula', '')}",
                source_id=metric_id,
                source_label="指标目录",
                location=f"metric-catalog:{metric_id}",
            )
        )
    return tuple(evidence)


def _provider_evidence(
    result: RuleAuthoringProviderResult,
    *,
    allowed_source_ids: set[str],
    retrieved_evidence: Sequence[RuleEvidence] = (),
) -> tuple[RuleEvidence, ...]:
    """只接受与本次检索快照逐字段一致的模型引用。"""

    retrieved_by_source = {
        item.source_id: item
        for item in retrieved_evidence
        if item.source_id
    }
    accepted: list[RuleEvidence] = []
    for item in result.evidence:
        if not isinstance(item, RuleEvidence):
            continue
        if item.type in {"standard_clause", "data_dictionary"}:
            if item.source_id not in allowed_source_ids:
                continue
            retrieved = retrieved_by_source.get(item.source_id)
            if retrieved is None:
                continue
            if any(
                getattr(item, field) != getattr(retrieved, field)
                for field in (
                    "text",
                    "document_id",
                    "document_name",
                    "document_version",
                    "section",
                    "clause",
                    "chunk_id",
                    "page",
                )
            ):
                continue
        accepted.append(item)
    return tuple(accepted)


def _rag_state_from_context(context: Mapping[str, Any]) -> Mapping[str, Any] | None:
    rag = context.get("rag")
    return rag if isinstance(rag, Mapping) else None


def _validate_draft_rag_binding(draft: RuleDraft) -> tuple[str, ...]:
    """不重新检索时也验证草案引用的片段仍来自其绑定快照。"""

    rag = _rag_state_from_context(draft.context)
    if rag is None:
        standard_evidence = tuple(
            item
            for item in draft.evidence
            if item.type in {"standard_clause", "data_dictionary"}
        )
        return (
            ("规则包含 RAG 依据，但草案没有保存检索快照。",)
            if standard_evidence
            else ()
        )
    allowed = {
        str(item)
        for item in rag.get("chunk_ids", [])
        if isinstance(item, str)
    }
    rag_evidence = tuple(
        item
        for item in draft.evidence
        if item.type in {"standard_clause", "data_dictionary"}
        or item.chunk_id in allowed
        or item.source_id in allowed
    )
    if not rag_evidence:
        return ()
    if rag.get("status") == "conflict":
        return ("规则绑定的标准依据存在版本或来源冲突，请重新选择适用来源。",)
    if not allowed:
        return ("规则包含 RAG 依据，但没有绑定检索片段。",)
    rag_results = {
        str(item.get("chunk_id")): item
        for item in rag.get("results", [])
        if isinstance(item, Mapping) and item.get("chunk_id")
    }
    errors = []
    for item in rag_evidence:
        source_id = item.chunk_id or item.source_id
        if source_id not in allowed:
            errors.append(f"依据 {item.id} 不属于草案绑定的 RAG 检索结果。")
            continue
        retrieved = rag_results.get(source_id)
        if retrieved is None:
            errors.append(f"依据 {item.id} 缺少检索片段快照。")
            continue
        for field in (
            "text",
            "document_id",
            "document_name",
            "document_version",
            "section",
            "clause",
            "page",
        ):
            if getattr(item, field) != retrieved.get(field):
                errors.append(f"依据 {item.id} 的 {field} 与检索快照不一致。")
        if item.chunk_id != retrieved.get("chunk_id"):
            errors.append(f"依据 {item.id} 的 chunk_id 与检索快照不一致。")
    return tuple(dict.fromkeys(errors))


def _pack_source_for_draft(draft: RuleDraft) -> tuple[str, str, str]:
    rag = _rag_state_from_context(draft.context)
    allowed = {
        str(item)
        for item in (rag or {}).get("chunk_ids", [])
        if isinstance(item, str)
    }
    has_rag = any(
        item.type in {"standard_clause", "data_dictionary"}
        or item.chunk_id in allowed
        or item.source_id in allowed
        for item in draft.evidence
    )
    if has_rag:
        return "standard_retrieval", RULE_DRAFT_GENERATOR_V09, "0.9.0"
    if draft.target_type == "catalog_metric":
        return "user_natural_language", RULE_DRAFT_GENERATOR, "0.7.0"
    return "user_natural_language", RULE_DRAFT_GENERATOR_V08, "0.8.0"


def compile_rule_draft(
    report: Any,
    *,
    target_metric_id: str,
    user_intent: str,
    workflow_id: str | None = None,
    provider: RuleAuthoringProvider | None = None,
    created_at: datetime | str | None = None,
    allow_template_fallback: bool = True,
    rag_response: Any | None = None,
    selected_chunk_ids: Iterable[str] = (),
) -> RuleDraft:
    """将某个目录指标下的用户评价依据编译成 RuleDraft。

    外部模型正式模式应关闭 ``allow_template_fallback``，这样 API 或模型
    输出错误会返回给页面，而不会被本地模板伪装成模型结果。
    """

    if get_metric_definition(target_metric_id) is None:
        raise RuleDraftValidationError([f"未知目录指标：{target_metric_id}。"])
    normalized_intent = str(user_intent or "").strip()
    if not normalized_intent:
        raise RuleDraftValidationError(["评价依据不能为空。"])
    selected_chunk_ids = tuple(dict.fromkeys(str(item) for item in selected_chunk_ids))
    try:
        rag_evidence = evidence_from_response(
            rag_response,
            selected_chunk_ids=selected_chunk_ids,
        ) if rag_response is not None and selected_chunk_ids else ()
    except RagCitationError as error:
        raise RuleDraftValidationError([str(error)]) from error
    context = build_rule_authoring_context(
        report,
        target_metric_id,
        rag_response=rag_response,
        selected_chunk_ids=selected_chunk_ids,
    )
    effective_workflow_id = workflow_id or make_workflow_id(
        [context.get("report_sha256"), target_metric_id, normalized_intent]
    )
    timestamp = _created_at(created_at)
    selected_provider = provider or default_rule_authoring_provider()
    try:
        provider_result = selected_provider.generate(
            context,
            user_intent=normalized_intent,
        )
    except (RuleAuthoringProviderUnavailable, RuleAuthoringProviderError) as error:
        if not allow_template_fallback:
            raise
        provider_result = _fallback_result(
            context,
            user_intent=normalized_intent,
            provider=selected_provider,
            reason=(
                "provider_unavailable"
                if isinstance(error, RuleAuthoringProviderUnavailable)
                else "provider_error"
            ),
        )
    except Exception:
        # 无 API 的暂行模式可以继续使用模板；正式外部模式由上面的开关
        # 直接抛出错误，页面必须明确告诉用户模型调用没有完成。
        if not allow_template_fallback:
            raise
        provider_result = _fallback_result(
            context,
            user_intent=normalized_intent,
            provider=selected_provider,
            reason="provider_error",
        )

    provider_result = _enforce_explicit_rule_inputs(
        context,
        user_intent=normalized_intent,
        result=provider_result,
    )
    evidence = list(_base_evidence(context, target_metric_id, normalized_intent))
    evidence.extend(rag_evidence)
    evidence.extend(
        _provider_evidence(
            provider_result,
            allowed_source_ids={
                target_metric_id,
                *[item.source_id for item in rag_evidence if item.source_id],
            },
            retrieved_evidence=rag_evidence,
        )
    )
    evidence = list({item.id: item for item in evidence}.values())
    evidence_ids = [item.id for item in evidence]

    rule_spec = provider_result.rule_spec
    status = "draft"
    unsupported_reason = None
    if provider_result.outcome == "clarification":
        status = "needs_clarification"
    elif provider_result.outcome == "unsupported":
        status = "rejected"
        unsupported_reason = provider_result.unsupported_reason or "当前需求超出支持范围。"
    elif rule_spec is None:
        status = "failed"
        unsupported_reason = "Provider 未返回规则草案。"
    else:
        rule_spec = normalized_rule_spec(rule_spec, evidence_ids=evidence_ids)

    draft = RuleDraft(
        schema_version=RULE_DRAFT_SCHEMA_VERSION,
        draft_id=make_draft_id(
            effective_workflow_id,
            "catalog_metric",
            target_metric_id,
            normalized_intent,
            timestamp,
        ),
        workflow_id=effective_workflow_id,
        target_type="catalog_metric",
        target_metric_id=target_metric_id,
        user_intent=normalized_intent,
        status=status,
        rule_spec=rule_spec,
        evidence=tuple(evidence),
        assumptions=tuple(provider_result.assumptions),
        clarification_questions=tuple(provider_result.clarification_questions),
        unsupported_reason=unsupported_reason,
        provider=provider_result.metadata,
        context=context,
        created_at=timestamp,
    )
    shape_validation = validate_rule_draft_shape(draft)
    if not shape_validation.valid:
        raise RuleDraftValidationError(shape_validation.errors)
    return draft


def compile_custom_rule_draft(
    report: Any,
    *,
    user_intent: str,
    workflow_id: str | None = None,
    provider: RuleAuthoringProvider | None = None,
    created_at: datetime | str | None = None,
    allow_template_fallback: bool = True,
    rag_response: Any | None = None,
    selected_chunk_ids: Iterable[str] = (),
) -> RuleDraft:
    """将自然语言自定义需求编译为 v0.9 可绑定来源的 RuleDraft。"""

    normalized_intent = str(user_intent or "").strip()
    if not normalized_intent:
        raise RuleDraftValidationError(["自定义规则描述不能为空。"])
    selected_chunk_ids = tuple(dict.fromkeys(str(item) for item in selected_chunk_ids))
    try:
        rag_evidence = evidence_from_response(
            rag_response,
            selected_chunk_ids=selected_chunk_ids,
        ) if rag_response is not None and selected_chunk_ids else ()
    except RagCitationError as error:
        raise RuleDraftValidationError([str(error)]) from error
    context = build_custom_rule_authoring_context(
        report,
        rag_response=rag_response,
        selected_chunk_ids=selected_chunk_ids,
    )
    effective_workflow_id = workflow_id or make_workflow_id(
        [context.get("report_sha256"), "custom_rule", normalized_intent]
    )
    timestamp = _created_at(created_at)
    selected_provider = provider or default_rule_authoring_provider()
    try:
        provider_result = selected_provider.generate(
            context,
            user_intent=normalized_intent,
        )
    except (RuleAuthoringProviderUnavailable, RuleAuthoringProviderError) as error:
        if not allow_template_fallback:
            raise
        provider_result = _fallback_result(
            context,
            user_intent=normalized_intent,
            provider=selected_provider,
            reason=(
                "provider_unavailable"
                if isinstance(error, RuleAuthoringProviderUnavailable)
                else "provider_error"
            ),
        )
    except Exception:
        if not allow_template_fallback:
            raise
        provider_result = _fallback_result(
            context,
            user_intent=normalized_intent,
            provider=selected_provider,
            reason="provider_error",
        )

    provider_result = _enforce_explicit_rule_inputs(
        context,
        user_intent=normalized_intent,
        result=provider_result,
    )
    evidence = list(_base_evidence(context, None, normalized_intent))
    evidence.extend(rag_evidence)
    evidence.extend(
        _provider_evidence(
            provider_result,
            allowed_source_ids={
                item.source_id for item in rag_evidence if item.source_id
            },
            retrieved_evidence=rag_evidence,
        )
    )
    evidence = list({item.id: item for item in evidence}.values())
    rule_spec = provider_result.rule_spec
    status = "draft"
    unsupported_reason = None
    if provider_result.outcome == "clarification":
        status = "needs_clarification"
    elif provider_result.outcome == "unsupported":
        status = "rejected"
        unsupported_reason = provider_result.unsupported_reason or "当前需求超出支持范围。"
    elif rule_spec is None:
        status = "failed"
        unsupported_reason = "Provider 未返回规则草案。"
    else:
        rule_spec = normalized_rule_spec(rule_spec, evidence_ids=[item.id for item in evidence])

    draft = RuleDraft(
        schema_version=RULE_DRAFT_SCHEMA_VERSION,
        draft_id=make_draft_id(
            effective_workflow_id,
            "custom_rule",
            None,
            normalized_intent,
            timestamp,
        ),
        workflow_id=effective_workflow_id,
        target_type="custom_rule",
        target_metric_id=None,
        user_intent=normalized_intent,
        status=status,
        rule_spec=rule_spec,
        evidence=tuple(evidence),
        assumptions=tuple(provider_result.assumptions),
        clarification_questions=tuple(provider_result.clarification_questions),
        unsupported_reason=unsupported_reason,
        provider=provider_result.metadata,
        context=context,
        created_at=timestamp,
    )
    shape_validation = validate_rule_draft_shape(draft)
    if not shape_validation.valid:
        raise RuleDraftValidationError(shape_validation.errors)
    return draft


def validate_rule_draft(
    draft: RuleDraft,
    report: Any,
) -> RuleDraftValidationResult:
    """根据当前报告画像验证字段、Rule DSL 和报告绑定。"""

    shape = validate_rule_draft_shape(draft)
    errors = list(shape.errors)
    warnings = list(shape.warnings)
    if errors:
        return RuleDraftValidationResult(False, tuple(dict.fromkeys(errors)), tuple(warnings))
    if draft.status in {"needs_clarification", "rejected", "failed"}:
        return RuleDraftValidationResult(
            False,
            (draft.unsupported_reason or "当前 RuleDraft 还不能进入校验或试运行。",),
            tuple(warnings),
        )
    if draft.target_type == "catalog_metric" and draft.target_metric_id is None:
        return RuleDraftValidationResult(False, ("目录指标 RuleDraft 缺少 target_metric_id。",))
    current_context = (
        build_rule_authoring_context(report, draft.target_metric_id)
        if draft.target_type == "catalog_metric" and draft.target_metric_id
        else build_custom_rule_authoring_context(report)
    )
    if draft.context.get("report_sha256") != current_context.get("report_sha256"):
        errors.append("RuleDraft 与当前零配置报告不匹配，请重新解析评价依据。")
    if draft.context.get("input_sha256") != current_context.get("input_sha256"):
        errors.append("RuleDraft 与当前输入文件不匹配，请重新解析评价依据。")
    fields = [item["name"] for item in current_context.get("fields", [])]
    rule_validation = validate_rule_spec(
        draft.rule_spec,
        available_fields=fields,
        evidence_ids=[item.id for item in draft.evidence],
    )
    errors.extend(rule_validation.errors)
    if draft.rule_spec is not None:
        if draft.target_type == "catalog_metric" and draft.target_metric_id:
            compatibility_error = metric_rule_type_error(
                draft.target_metric_id,
                draft.rule_spec.rule_type,
            )
            if compatibility_error:
                errors.append(compatibility_error)
        inspection = inspect_rule_intent(
            draft.context,
            user_intent=draft.user_intent,
        )
        if (
            inspection.recognized_rule_type is not None
            and draft.rule_spec.rule_type != inspection.recognized_rule_type
        ):
            errors.append("模型生成的规则类型与用户明确描述的规则类型不一致。")
        intent_text = draft.user_intent.casefold()
        guessed_fields = [
            field
            for field in draft.rule_spec.fields
            if field.casefold() not in intent_text
        ]
        if guessed_fields:
            errors.append(
                "模型生成了用户未明确写出的字段："
                + "、".join(guessed_fields)
                + "。"
            )
        if draft.rule_spec.rule_type in {"allowed_values", "conditional_required"}:
            parameter_name = (
                "allowed_values"
                if draft.rule_spec.rule_type == "allowed_values"
                else "condition_values"
            )
            invented_values = [
                str(value)
                for value in draft.rule_spec.parameters.get(parameter_name, ())
                if str(value).casefold() not in intent_text
            ]
            if invented_values:
                errors.append(
                    "模型生成了用户未明确列出的取值："
                    + "、".join(invented_values[:10])
                    + "。"
                )
    errors.extend(_validate_draft_rag_binding(draft))
    if draft.rule_spec is not None and not errors:
        source_type, generator, pack_version = _pack_source_for_draft(draft)
        try:
            pack = build_rule_pack(
                report,
                name=draft.rule_spec.name,
                version=pack_version,
                rules=(draft.rule_spec.to_rule(),),
                source_type=source_type,
                generator=generator,
                generated_at=draft.created_at,
                evidence=tuple(item.to_dict() for item in draft.evidence),
            )
        except (RulePackValidationError, TypeError, ValueError, IndexError) as error:
            errors.append(f"RulePack 确定性校验失败：{error}")
        else:
            pack_validation = validate_rule_pack(pack, report)
            errors.extend(pack_validation.errors)
    if errors:
        return RuleDraftValidationResult(False, tuple(dict.fromkeys(errors)), tuple(warnings))
    return RuleDraftValidationResult(True, (), tuple(warnings))


def build_rule_pack_from_draft(
    draft: RuleDraft,
    report: Any,
    *,
    version: str | None = None,
) -> RulePack:
    """把已经通过确定性校验的 RuleDraft 转换为未审批 RulePack。"""

    validation = validate_rule_draft(draft, report)
    if not validation.valid:
        raise RuleDraftValidationError(validation.errors)
    if draft.rule_spec is None:
        raise RuleDraftValidationError(["RuleDraft 没有可转换的 rule_spec。"])
    source_type, generator, default_version = _pack_source_for_draft(draft)
    return build_rule_pack(
        report,
        name=draft.rule_spec.name,
        version=version or default_version,
        rules=(draft.rule_spec.to_rule(),),
        source_type=source_type,
        generator=generator,
        generated_at=draft.created_at,
        evidence=tuple(item.to_dict() for item in draft.evidence),
        metric_targets=_metric_targets_for_draft(draft, report),
    )


__all__ = [
    "build_rule_pack_from_draft",
    "compile_custom_rule_draft",
    "compile_rule_draft",
    "validate_rule_draft",
]
