"""v0.8 规则编制服务。

服务负责组装上下文、调用 Provider、生成 RuleDraft 和执行确定性校验；它不
审批、不直接正式执行规则。正式 RulePack 仍由现有本地审批服务接管。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from .metric_catalog import get_metric_definition
from .rule_authoring_providers import (
    RuleAuthoringProvider,
    RuleAuthoringProviderError,
    RuleAuthoringProviderUnavailable,
    RuleAuthoringProviderResult,
    TemplateRuleAuthoringProvider,
    default_rule_authoring_provider,
)
from .rule_authoring_tools import (
    build_custom_rule_authoring_context,
    build_rule_authoring_context,
)
from .rule_dsl import (
    RULE_DRAFT_GENERATOR,
    RULE_DRAFT_GENERATOR_V08,
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
from .rule_pack import RulePack, RulePackValidationError, build_rule_pack, validate_rule_pack


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
) -> tuple[RuleEvidence, ...]:
    accepted: list[RuleEvidence] = []
    for item in result.evidence:
        if not isinstance(item, RuleEvidence):
            continue
        if item.type in {"standard_clause", "data_dictionary"} and item.source_id not in allowed_source_ids:
            continue
        accepted.append(item)
    return tuple(accepted)


def compile_rule_draft(
    report: Any,
    *,
    target_metric_id: str,
    user_intent: str,
    workflow_id: str | None = None,
    provider: RuleAuthoringProvider | None = None,
    created_at: datetime | str | None = None,
    allow_template_fallback: bool = True,
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
    context = build_rule_authoring_context(report, target_metric_id)
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

    evidence = list(_base_evidence(context, target_metric_id, normalized_intent))
    evidence.extend(
        _provider_evidence(
            provider_result,
            allowed_source_ids={target_metric_id},
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
) -> RuleDraft:
    """将自然语言自定义需求编译为不绑定目录指标的 v0.8 RuleDraft。"""

    normalized_intent = str(user_intent or "").strip()
    if not normalized_intent:
        raise RuleDraftValidationError(["自定义规则描述不能为空。"])
    context = build_custom_rule_authoring_context(report)
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

    evidence = list(_base_evidence(context, None, normalized_intent))
    evidence.extend(_provider_evidence(provider_result, allowed_source_ids=set()))
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
    if draft.rule_spec is not None and not rule_validation.errors:
        try:
            pack = build_rule_pack(
                report,
                name=draft.rule_spec.name,
                version=(
                    "0.7.0"
                    if draft.target_type == "catalog_metric"
                    else "0.8.0"
                ),
                rules=(draft.rule_spec.to_rule(),),
                source_type="user_natural_language",
                generator=(
                    RULE_DRAFT_GENERATOR
                    if draft.target_type == "catalog_metric"
                    else RULE_DRAFT_GENERATOR_V08
                ),
                generated_at=draft.created_at,
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
    return build_rule_pack(
        report,
        name=draft.rule_spec.name,
        version=version
        or ("0.7.0" if draft.target_type == "catalog_metric" else "0.8.0"),
        rules=(draft.rule_spec.to_rule(),),
        source_type="user_natural_language",
        generator=(
            RULE_DRAFT_GENERATOR
            if draft.target_type == "catalog_metric"
            else RULE_DRAFT_GENERATOR_V08
        ),
        generated_at=draft.created_at,
    )


__all__ = [
    "build_rule_pack_from_draft",
    "compile_custom_rule_draft",
    "compile_rule_draft",
    "validate_rule_draft",
]
