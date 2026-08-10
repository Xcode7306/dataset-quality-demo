"""v1.0 deterministic coordinator for the recoverable rule-authoring flow.

The coordinator owns state transitions and calls the existing bounded rule
compiler, validator, dry-run, approval, and execution services.  Uploaded bytes
are passed through to the existing temporary-file services and are never stored
in the workflow or its history summary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from .rule_authoring_service import (
    build_rule_pack_from_draft,
    compile_custom_rule_draft,
    compile_rule_draft,
    validate_rule_draft,
)
from .rule_authoring_workflow import (
    RuleAuthoringWorkflow,
    RuleAuthoringWorkflowError,
    make_rule_authoring_request_fingerprint,
    new_rule_authoring_workflow,
    validate_rule_authoring_workflow,
)
from .rule_engine import RuleDryRunResult, RuleEvaluationResult
from .rule_pack import RulePack, approve_rule_pack
from .rule_service import (
    dry_run_uploaded_dataset_with_rule_pack,
    evaluate_uploaded_dataset_with_rule_pack,
)


class RuleAuthoringCoordinatorError(ValueError):
    """The requested operation cannot safely continue in the current run."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8", errors="strict")


def _report_context(report: Any) -> Mapping[str, Any]:
    if report is None or not callable(getattr(report, "to_dict", None)):
        raise RuleAuthoringCoordinatorError("当前规则编制缺少可绑定的质量报告。")
    payload = report.to_dict()
    context = payload.get("evaluation_context") if isinstance(payload, Mapping) else None
    if not isinstance(context, Mapping):
        raise RuleAuthoringCoordinatorError("当前质量报告缺少 evaluation_context。")
    return context


def _selected_metrics(
    context: Mapping[str, Any],
    selected_metric_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    values = (
        selected_metric_ids
        if selected_metric_ids is not None
        else context.get("selected_metric_ids", ())
    )
    if not isinstance(values, (list, tuple)):
        raise RuleAuthoringCoordinatorError("当前质量报告的指标绑定无效。")
    return tuple(dict.fromkeys(str(item) for item in values))


def _assert_same_request(
    workflow: RuleAuthoringWorkflow,
    report: Any,
    *,
    user_intent: str,
    selected_chunk_ids: Sequence[str],
) -> None:
    context = _report_context(report)
    current_report_sha256 = context.get("report_sha256")
    current_input_sha256 = context.get("input_sha256")
    current_reference_date = context.get("reference_date")
    if current_report_sha256 != workflow.report_sha256:
        raise RuleAuthoringCoordinatorError("质量报告已变化，请创建新的规则工作流。")
    if current_input_sha256 != workflow.input_sha256:
        raise RuleAuthoringCoordinatorError("输入文件已变化，请创建新的规则工作流。")
    if current_reference_date != workflow.reference_date:
        raise RuleAuthoringCoordinatorError("参考日期已变化，请创建新的规则工作流。")
    fingerprint = make_rule_authoring_request_fingerprint(
        target_type=str(workflow.target_type),
        target_metric_id=workflow.target_metric_id,
        report_sha256=workflow.report_sha256,
        input_sha256=workflow.input_sha256,
        reference_date=workflow.reference_date,
        selected_metric_ids=workflow.selected_metric_ids,
        user_intent=user_intent,
        selected_chunk_ids=selected_chunk_ids,
    )
    if fingerprint != workflow.request_fingerprint:
        raise RuleAuthoringCoordinatorError("请求内容或依据绑定已变化，请创建新的规则工作流。")


def _execution_result_id(result: RuleEvaluationResult) -> str:
    digest = hashlib.sha256(_canonical_json(result.to_dict())).hexdigest()
    return f"execution-{digest[:20]}"


@dataclass(frozen=True)
class RuleAuthoringRun:
    """In-session artifacts for one workflow; no uploaded bytes are retained."""

    workflow: RuleAuthoringWorkflow
    draft_pack: RulePack | None = None
    preview: RuleDryRunResult | None = None
    approved_pack: RulePack | None = None
    result: RuleEvaluationResult | None = None

    def __post_init__(self) -> None:
        errors = validate_rule_authoring_workflow(self.workflow)
        if errors:
            raise RuleAuthoringCoordinatorError("；".join(errors))
        if self.preview is not None and self.draft_pack is None:
            raise RuleAuthoringCoordinatorError("试运行摘要必须绑定未审批 RulePack。")
        if self.approved_pack is not None:
            approval = self.approved_pack.approval
            if approval is None or self.approved_pack.status != "approved":
                raise RuleAuthoringCoordinatorError("approved_pack 缺少有效审批记录。")
            if (
                self.workflow.approval_id is not None
                and approval.approval_id != self.workflow.approval_id
            ):
                raise RuleAuthoringCoordinatorError("工作流与 RulePack 审批 ID 不一致。")
        if self.workflow.state in {"approved", "executed"} and self.approved_pack is None:
            raise RuleAuthoringCoordinatorError("已审批工作流必须绑定 approved_pack。")
        if self.result is not None and self.workflow.state != "executed":
            raise RuleAuthoringCoordinatorError("执行结果只能绑定 executed 工作流。")
        if self.workflow.state == "executed" and self.result is None:
            raise RuleAuthoringCoordinatorError("executed 工作流必须绑定执行结果。")


def begin_rule_authoring_run(
    report: Any,
    *,
    target_metric_id: str | None,
    user_intent: str,
    selected_metric_ids: Sequence[str] | None = None,
    selected_chunk_ids: Sequence[str] = (),
    revision: int = 1,
    created_at: datetime | str | None = None,
) -> RuleAuthoringRun:
    """Create a report-bound run and enter the deterministic compiling state."""

    context = _report_context(report)
    target_type = "catalog_metric" if target_metric_id else "custom_rule"
    metrics = _selected_metrics(context, selected_metric_ids)
    chunks = tuple(dict.fromkeys(str(item) for item in selected_chunk_ids))
    workflow = new_rule_authoring_workflow(
        target_type=target_type,
        target_metric_id=target_metric_id,
        report_sha256=context.get("report_sha256"),
        input_sha256=context.get("input_sha256"),
        reference_date=context.get("reference_date"),
        selected_metric_ids=metrics,
        user_intent=user_intent,
        selected_chunk_ids=chunks,
        revision=revision,
        created_at=created_at,
    )
    if chunks:
        workflow = workflow.start_retrieving(at=created_at)
    workflow = workflow.start_compiling(at=created_at)
    return RuleAuthoringRun(workflow=workflow)


def compile_rule_authoring_run(
    run: RuleAuthoringRun,
    report: Any,
    *,
    user_intent: str,
    provider: Any | None = None,
    allow_template_fallback: bool = True,
    rag_response: Any | None = None,
    selected_chunk_ids: Iterable[str] = (),
    created_at: datetime | str | None = None,
) -> RuleAuthoringRun:
    """Compile a candidate draft; provider failures become a recoverable state."""

    workflow = run.workflow
    chunks = tuple(dict.fromkeys(str(item) for item in selected_chunk_ids))
    _assert_same_request(
        workflow,
        report,
        user_intent=user_intent,
        selected_chunk_ids=chunks,
    )
    if workflow.state != "compiling":
        if workflow.draft is not None and workflow.state in {
            "draft",
            "validated",
            "dry_run_complete",
            "awaiting_approval",
            "approved",
            "executed",
            "needs_clarification",
            "rejected",
        }:
            return run
        raise RuleAuthoringCoordinatorError("当前工作流不在 compiling 状态。")
    kwargs = {
        "user_intent": user_intent,
        "workflow_id": workflow.workflow_id,
        "provider": provider,
        "created_at": created_at,
        "allow_template_fallback": allow_template_fallback,
        "rag_response": rag_response,
        "selected_chunk_ids": chunks,
    }
    try:
        if workflow.target_type == "catalog_metric":
            draft = compile_rule_draft(
                report,
                target_metric_id=str(workflow.target_metric_id),
                **kwargs,
            )
        else:
            draft = compile_custom_rule_draft(report, **kwargs)
        workflow = workflow.accept_draft(draft, at=created_at)
    except Exception as error:
        workflow = workflow.fail(
            stage="compiling",
            code="rule_compilation_failed",
            message=str(error),
            at=created_at,
        )
    return replace(
        run,
        workflow=workflow,
        draft_pack=None,
        preview=None,
        approved_pack=None,
        result=None,
    )


def validate_rule_authoring_run(
    run: RuleAuthoringRun,
    report: Any,
    *,
    at: datetime | str | None = None,
) -> RuleAuthoringRun:
    """Apply deterministic draft validation and move to its explicit state."""

    workflow = run.workflow
    if workflow.state in {
        "validated",
        "dry_run_complete",
        "awaiting_approval",
        "approved",
        "executed",
    }:
        return run
    if workflow.state != "draft" or workflow.draft is None:
        raise RuleAuthoringCoordinatorError("当前工作流没有可校验的 RuleDraft。")
    try:
        validation = validate_rule_draft(workflow.draft, report)
    except Exception as error:
        from .rule_dsl import RuleDraftValidationResult

        validation = RuleDraftValidationResult(
            False,
            (f"确定性校验未完成：{type(error).__name__}。",),
            (),
        )
    return replace(run, workflow=workflow.mark_validated(validation, at=at))


def dry_run_rule_authoring_run(
    run: RuleAuthoringRun,
    report: Any,
    *,
    content: bytes,
    file_name: str,
    dataset_name: str | None = None,
    sheet_name: str | None = None,
    reference_date: date | None = None,
    selected_metric_ids: Sequence[str] | None = None,
    at: datetime | str | None = None,
) -> RuleAuthoringRun:
    """Build the draft pack, dry-run it, and stop at explicit approval."""

    workflow = run.workflow
    if workflow.state == "awaiting_approval" and run.preview is not None:
        return run
    if workflow.state != "validated" or workflow.draft is None:
        raise RuleAuthoringCoordinatorError("只有 validated 工作流可以试运行。")
    try:
        pack = build_rule_pack_from_draft(workflow.draft, report)
        preview = dry_run_uploaded_dataset_with_rule_pack(
            content,
            file_name,
            pack,
            dataset_name=dataset_name,
            sheet_name=sheet_name,
            reference_date=reference_date,
            selected_metric_ids=selected_metric_ids,
        )
    except Exception as error:
        failed = workflow.fail(
            stage="dry_run",
            code="deterministic_dry_run_failed",
            message=str(error),
            at=at,
        )
        return replace(
            run,
            workflow=failed,
            draft_pack=None,
            preview=None,
            approved_pack=None,
            result=None,
        )
    workflow = workflow.mark_dry_run_complete(preview.to_dict(), at=at)
    workflow = workflow.await_approval(at=at)
    return replace(
        run,
        workflow=workflow,
        draft_pack=pack,
        preview=preview,
        approved_pack=None,
        result=None,
    )


def approve_rule_authoring_run(
    run: RuleAuthoringRun,
    report: Any,
    *,
    approver: str,
    approved_at: datetime | str | None = None,
) -> RuleAuthoringRun:
    """Record the local human approval exactly once."""

    workflow = run.workflow
    if workflow.state in {"approved", "executed"}:
        if run.approved_pack is None:
            raise RuleAuthoringCoordinatorError("已审批工作流缺少已审批 RulePack。")
        return run
    if workflow.state != "awaiting_approval" or run.draft_pack is None:
        raise RuleAuthoringCoordinatorError("校验和试运行完成前不能审批。")
    try:
        approved_pack = approve_rule_pack(
            run.draft_pack,
            report,
            approver=approver,
            approved_at=approved_at,
        )
    except Exception as error:
        raise RuleAuthoringCoordinatorError(str(error)) from error
    approval = approved_pack.approval
    if approval is None:
        raise RuleAuthoringCoordinatorError("审批服务没有返回 approval_id。")
    workflow = workflow.approve(approval.approval_id, at=approved_at)
    return replace(run, workflow=workflow, approved_pack=approved_pack, result=None)


def execute_rule_authoring_run(
    run: RuleAuthoringRun,
    *,
    content: bytes,
    file_name: str,
    dataset_name: str | None = None,
    sheet_name: str | None = None,
    reference_date: date | None = None,
    selected_metric_ids: Sequence[str] | None = None,
    at: datetime | str | None = None,
) -> RuleAuthoringRun:
    """Execute an approved pack once; retain it if execution needs one retry."""

    workflow = run.workflow
    if workflow.state == "executed":
        if run.result is None:
            raise RuleAuthoringCoordinatorError("executed 工作流缺少执行结果。")
        return run
    if workflow.state != "approved" or run.approved_pack is None:
        raise RuleAuthoringCoordinatorError("只有已审批工作流可以正式执行。")
    try:
        result = evaluate_uploaded_dataset_with_rule_pack(
            content,
            file_name,
            run.approved_pack,
            dataset_name=dataset_name,
            sheet_name=sheet_name,
            reference_date=reference_date,
            selected_metric_ids=selected_metric_ids,
        )
    except Exception as error:
        failed = workflow.fail(
            stage="execution",
            code="deterministic_execution_failed",
            message=str(error),
            at=at,
        )
        return replace(run, workflow=failed, result=None)
    workflow = workflow.execute(_execution_result_id(result), at=at)
    return replace(run, workflow=workflow, result=result)


def retry_rule_authoring_run(
    run: RuleAuthoringRun,
    *,
    user_intent: str,
    selected_chunk_ids: Sequence[str] = (),
    at: datetime | str | None = None,
) -> RuleAuthoringRun:
    """Restore the exact pre-failure state once for the same request."""

    workflow = run.workflow
    fingerprint = make_rule_authoring_request_fingerprint(
        target_type=str(workflow.target_type),
        target_metric_id=workflow.target_metric_id,
        report_sha256=workflow.report_sha256,
        input_sha256=workflow.input_sha256,
        reference_date=workflow.reference_date,
        selected_metric_ids=workflow.selected_metric_ids,
        user_intent=user_intent,
        selected_chunk_ids=selected_chunk_ids,
    )
    try:
        workflow = workflow.retry(request_fingerprint=fingerprint, at=at)
    except RuleAuthoringWorkflowError as error:
        raise RuleAuthoringCoordinatorError(str(error)) from error
    if workflow.state == "compiling":
        return RuleAuthoringRun(workflow=workflow)
    if workflow.state == "validated":
        return replace(
            run,
            workflow=workflow,
            draft_pack=None,
            preview=None,
            approved_pack=None,
            result=None,
        )
    if workflow.state == "approved":
        if run.approved_pack is None:
            raise RuleAuthoringCoordinatorError("执行恢复缺少原已审批 RulePack。")
        return replace(run, workflow=workflow, result=None)
    return replace(run, workflow=workflow, result=None)


__all__ = [
    "RuleAuthoringCoordinatorError",
    "RuleAuthoringRun",
    "approve_rule_authoring_run",
    "begin_rule_authoring_run",
    "compile_rule_authoring_run",
    "dry_run_rule_authoring_run",
    "execute_rule_authoring_run",
    "retry_rule_authoring_run",
    "validate_rule_authoring_run",
]
