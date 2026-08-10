"""Offline, reproducible v0.9.1 harness for rule authoring and RAG retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .rag.citations import RagCitationError, evidence_from_response
from .rag.models import (
    RAG_DOCUMENT_STATUSES,
    RAG_NAMESPACE_STANDARDS,
    RAG_RETRIEVAL_STATUSES,
)
from .rag.retrieval import RagKnowledgeBase
from .rule_authoring_prompts import get_rule_authoring_prompt
from .rule_authoring_providers import (
    RuleAuthoringProvider,
    TemplateRuleAuthoringProvider,
)
from .rule_authoring_service import (
    build_rule_pack_from_draft,
    compile_custom_rule_draft,
    compile_rule_draft,
    validate_rule_draft,
)
from .rule_authoring_tools import (
    get_metric_definition_tool,
    get_profile_summary_tool,
    list_available_fields_tool,
    retrieve_rule_evidence_tool,
    validate_rule_authoring_tool_request,
)
from .rule_authoring_trace import RuleAuthoringTrace, RuleAuthoringTraceBuilder
from .rule_dsl import RuleDraft, make_workflow_id
from .rule_pack import approve_rule_pack
from .rule_service import (
    dry_run_uploaded_dataset_with_rule_pack,
    evaluate_uploaded_dataset_with_rule_pack,
)
from .workflow import build_profile_report


HARNESS_SCHEMA_VERSION = "0.1"
DEFAULT_RULE_GOLDENS = Path("harness/goldens/rule_authoring_cases.json")
DEFAULT_RAG_GOLDENS = Path("harness/goldens/rag_retrieval_cases.json")


class AgentHarnessError(ValueError):
    """A golden suite or harness run is invalid."""


def _strict_json(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as error:
        raise AgentHarnessError(f"无法读取 Harness 金标集：{path}。") from error

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AgentHarnessError(f"金标集 JSON 包含重复键：{key}。")
            result[key] = value
        return result

    try:
        payload = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AgentHarnessError(f"金标集包含非标准数值：{value}。")
            ),
        )
    except AgentHarnessError:
        raise
    except (TypeError, ValueError) as error:
        raise AgentHarnessError(f"金标集不是严格 JSON：{path}。") from error
    if not isinstance(payload, Mapping):
        raise AgentHarnessError("金标集顶层必须是 JSON 对象。")
    return payload


def _exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise AgentHarnessError(
            f"{label}字段不匹配；缺少 {sorted(expected - actual)}，"
            f"多出 {sorted(actual - expected)}。"
        )


@dataclass(frozen=True)
class RuleGoldenCase:
    case_id: str
    target_type: str
    target_metric_id: str | None
    user_intent: str
    expected_outcome: str
    expected_rule_type: str | None
    expected_fields: tuple[str, ...]
    expected_parameters: Mapping[str, Any]


@dataclass(frozen=True)
class RuleGoldenSuite:
    dataset_path: Path
    reference_date: date
    created_at: str
    cases: tuple[RuleGoldenCase, ...]


def load_rule_golden_suite(
    project_root: Path,
    path: Path | None = None,
) -> RuleGoldenSuite:
    source = path or project_root / DEFAULT_RULE_GOLDENS
    payload = _strict_json(source)
    _exact_keys(
        payload,
        {"schema_version", "dataset", "reference_date", "created_at", "cases"},
        "规则金标集",
    )
    if payload["schema_version"] != HARNESS_SCHEMA_VERSION:
        raise AgentHarnessError("规则金标集版本不受支持。")
    dataset = project_root / str(payload["dataset"])
    if not dataset.is_file() or dataset.resolve().parent != (project_root / "harness/data").resolve():
        raise AgentHarnessError("金标数据集必须位于 harness/data 且文件存在。")
    try:
        reference_date = date.fromisoformat(str(payload["reference_date"]))
    except ValueError as error:
        raise AgentHarnessError("金标集 reference_date 无效。") from error
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list) or not raw_cases or len(raw_cases) > 200:
        raise AgentHarnessError("规则金标集必须包含 1 到 200 个案例。")
    cases: list[RuleGoldenCase] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_cases):
        if not isinstance(item, Mapping):
            raise AgentHarnessError(f"规则金标案例 {index} 必须是对象。")
        _exact_keys(
            item,
            {"case_id", "target_type", "target_metric_id", "user_intent", "expected"},
            f"规则金标案例 {index}",
        )
        expected = item["expected"]
        if not isinstance(expected, Mapping):
            raise AgentHarnessError(f"案例 {index} expected 必须是对象。")
        _exact_keys(
            expected,
            {"outcome", "rule_type", "fields", "parameters"},
            f"规则金标案例 {index}.expected",
        )
        case_id = str(item["case_id"])
        if not case_id or case_id in seen:
            raise AgentHarnessError("规则金标 case_id 必须非空且唯一。")
        seen.add(case_id)
        target_type = str(item["target_type"])
        if target_type not in {"catalog_metric", "custom_rule"}:
            raise AgentHarnessError(f"案例 {case_id} target_type 无效。")
        target_metric_id = item["target_metric_id"]
        if target_type == "catalog_metric" and not isinstance(target_metric_id, str):
            raise AgentHarnessError(f"案例 {case_id} 缺少目录指标 ID。")
        if target_type == "custom_rule" and target_metric_id is not None:
            raise AgentHarnessError(f"案例 {case_id} 自定义规则指标 ID 必须为 null。")
        outcome = str(expected["outcome"])
        if outcome not in {"draft", "clarification", "unsupported"}:
            raise AgentHarnessError(f"案例 {case_id} expected.outcome 无效。")
        fields = expected["fields"]
        parameters = expected["parameters"]
        if not isinstance(fields, list) or not all(isinstance(value, str) for value in fields):
            raise AgentHarnessError(f"案例 {case_id} expected.fields 无效。")
        if not isinstance(parameters, Mapping):
            raise AgentHarnessError(f"案例 {case_id} expected.parameters 无效。")
        intent = str(item["user_intent"])
        if not intent or len(intent) > 4000:
            raise AgentHarnessError(f"案例 {case_id} user_intent 无效。")
        intent.encode("utf-8", errors="strict")
        cases.append(
            RuleGoldenCase(
                case_id=case_id,
                target_type=target_type,
                target_metric_id=target_metric_id,
                user_intent=intent,
                expected_outcome=outcome,
                expected_rule_type=(
                    str(expected["rule_type"])
                    if expected["rule_type"] is not None
                    else None
                ),
                expected_fields=tuple(fields),
                expected_parameters=dict(parameters),
            )
        )
    return RuleGoldenSuite(
        dataset_path=dataset,
        reference_date=reference_date,
        created_at=str(payload["created_at"]),
        cases=tuple(cases),
    )


@dataclass(frozen=True)
class HarnessCaseResult:
    case_id: str
    passed: bool
    expected_outcome: str
    actual_outcome: str
    schema_valid: bool
    outcome_correct: bool
    field_mapping_correct: bool | None
    parameters_correct: bool | None
    deterministic_validation_passed: bool | None
    dry_run_execution_consistent: bool | None
    replay_consistent: bool | None
    ungrounded_standard_claim_count: int
    unapproved_execution_count: int
    errors: tuple[str, ...]
    trace: RuleAuthoringTrace

    def to_dict(self, *, include_trace: bool = True) -> dict[str, Any]:
        payload = {
            "case_id": self.case_id,
            "passed": self.passed,
            "expected_outcome": self.expected_outcome,
            "actual_outcome": self.actual_outcome,
            "schema_valid": self.schema_valid,
            "outcome_correct": self.outcome_correct,
            "field_mapping_correct": self.field_mapping_correct,
            "parameters_correct": self.parameters_correct,
            "deterministic_validation_passed": self.deterministic_validation_passed,
            "dry_run_execution_consistent": self.dry_run_execution_consistent,
            "replay_consistent": self.replay_consistent,
            "ungrounded_standard_claim_count": self.ungrounded_standard_claim_count,
            "unapproved_execution_count": self.unapproved_execution_count,
            "errors": list(self.errors),
        }
        if include_trace:
            payload["trace"] = self.trace.to_dict()
        return payload


def _draft_outcome(draft: RuleDraft) -> str:
    return {
        "draft": "draft",
        "needs_clarification": "clarification",
        "rejected": "unsupported",
    }.get(draft.status, "failed")


def _schema_errors(validator: Draft202012Validator, payload: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        f"{'/'.join(str(item) for item in error.absolute_path)}: {error.message}"
        for error in sorted(
            validator.iter_errors(payload),
            key=lambda item: list(item.absolute_path),
        )
    )


def _dry_run_matches_execution(preview: Any, result: Any) -> bool:
    enhanced = {metric.metric_key: metric for metric in result.enhanced_report.metrics}
    for row in preview.metrics:
        metric = enhanced.get(row.get("metric_key"))
        if metric is None:
            return False
        evidence = metric.evidence if isinstance(metric.evidence, Mapping) else {}
        if any(
            (
                row.get("status") != metric.status,
                row.get("value") != metric.value,
                row.get("field") != metric.field,
                row.get("checked_count") != evidence.get("checked_count"),
                row.get("compliant_count") != evidence.get("compliant_count"),
                row.get("issue_count") != evidence.get("issue_count"),
            )
        ):
            return False
    return True


def _build_trace(
    case: RuleGoldenCase,
    report: Any,
    *,
    created_at: str,
) -> RuleAuthoringTraceBuilder:
    context = report.evaluation_context
    report_payload = report.to_dict()
    profile = report_payload.get("profile", {})
    if not isinstance(profile, Mapping):
        profile = {}
    workflow_id = make_workflow_id(
        [context.get("report_sha256"), case.case_id, case.user_intent]
    )
    return RuleAuthoringTraceBuilder(
        workflow_id=workflow_id,
        target_type=case.target_type,
        target_metric_id=case.target_metric_id,
        report_sha256=context.get("report_sha256"),
        input_sha256=context.get("input_sha256"),
        reference_date=context.get("reference_date"),
        report_status=str(report_payload.get("status") or "unknown"),
        row_count=(
            int(profile["row_count"])
            if isinstance(profile.get("row_count"), int)
            and not isinstance(profile.get("row_count"), bool)
            else None
        ),
        column_count=(
            int(profile["column_count"])
            if isinstance(profile.get("column_count"), int)
            and not isinstance(profile.get("column_count"), bool)
            else None
        ),
        selected_metric_ids=tuple(context.get("selected_metric_ids", ())),
        case_id=case.case_id,
        started_at=created_at,
    )


def _run_rule_case_once(
    case: RuleGoldenCase,
    *,
    report: Any,
    dataset_bytes: bytes,
    dataset_name: str,
    reference_date: date,
    created_at: str,
    provider: RuleAuthoringProvider,
    draft_validator: Draft202012Validator,
    trace_validator: Draft202012Validator,
) -> HarnessCaseResult:
    errors: list[str] = []
    trace = _build_trace(case, report, created_at=created_at)
    trace.transition("collecting", "compiling")

    fields = list_available_fields_tool(report)
    validate_rule_authoring_tool_request("list_available_fields", {})
    trace.tool_call("list_available_fields", {}, result=fields)
    profile = get_profile_summary_tool(report)
    validate_rule_authoring_tool_request("get_profile_summary", {})
    trace.tool_call("get_profile_summary", {}, result=profile)
    if case.target_metric_id:
        metric_args = validate_rule_authoring_tool_request(
            "get_metric_definition", {"metric_id": case.target_metric_id}
        )
        trace.tool_call(
            "get_metric_definition",
            metric_args,
            result=get_metric_definition_tool(case.target_metric_id),
        )

    try:
        if case.target_type == "catalog_metric":
            draft = compile_rule_draft(
                report,
                target_metric_id=str(case.target_metric_id),
                user_intent=case.user_intent,
                workflow_id=trace.workflow_id,
                provider=provider,
                created_at=created_at,
                allow_template_fallback=False,
            )
        else:
            draft = compile_custom_rule_draft(
                report,
                user_intent=case.user_intent,
                workflow_id=trace.workflow_id,
                provider=provider,
                created_at=created_at,
                allow_template_fallback=False,
            )
    except Exception as error:
        trace.transition("compiling", "failed", reason_code="provider_or_compile_error")
        trace.record_failure(
            stage="compiling",
            code=type(error).__name__,
            message=str(error),
        )
        finished = trace.finish("failed", completed_at=created_at)
        trace_errors = _schema_errors(trace_validator, finished.to_dict())
        errors.extend(trace_errors)
        errors.append(str(error))
        return HarnessCaseResult(
            case_id=case.case_id,
            passed=False,
            expected_outcome=case.expected_outcome,
            actual_outcome="failed",
            schema_valid=False,
            outcome_correct=False,
            field_mapping_correct=None,
            parameters_correct=None,
            deterministic_validation_passed=None,
            dry_run_execution_consistent=None,
            replay_consistent=False,
            ungrounded_standard_claim_count=0,
            unapproved_execution_count=0,
            errors=tuple(errors),
            trace=finished,
        )

    try:
        prompt_sha256 = get_rule_authoring_prompt(draft.provider.prompt_version).sha256
    except ValueError:
        prompt_sha256 = None
    trace.bind_provider(draft.provider, prompt_sha256=prompt_sha256)
    trace.bind_draft(draft)
    rag_context = draft.context.get("rag")
    allowed_chunk_ids = {
        str(item)
        for item in (
            rag_context.get("chunk_ids", ())
            if isinstance(rag_context, Mapping)
            else ()
        )
        if isinstance(item, str)
    }
    ungrounded_standard_claim_count = sum(
        1
        for item in draft.evidence
        if item.type in {"standard_clause", "data_dictionary"}
        and (item.chunk_id or item.source_id) not in allowed_chunk_ids
    )
    if ungrounded_standard_claim_count:
        errors.append("规则草案包含未绑定当前检索快照的标准依据。")
    actual_outcome = _draft_outcome(draft)
    target_state = {
        "draft": "draft",
        "clarification": "needs_clarification",
        "unsupported": "rejected",
        "failed": "failed",
    }[actual_outcome]
    trace.transition("compiling", target_state)

    draft_schema_errors = _schema_errors(draft_validator, draft.to_dict())
    schema_valid = not draft_schema_errors
    errors.extend(draft_schema_errors)
    outcome_correct = actual_outcome == case.expected_outcome
    if not outcome_correct:
        errors.append(
            f"outcome 期望 {case.expected_outcome}，实际 {actual_outcome}。"
        )

    field_correct: bool | None = None
    parameter_correct: bool | None = None
    validation_passed: bool | None = None
    dry_execution_consistent: bool | None = None
    terminal_failure = False
    if case.expected_outcome == "draft":
        if draft.rule_spec is None:
            field_correct = False
            parameter_correct = False
            errors.append("应生成规则的案例没有 rule_spec。")
        else:
            field_correct = (
                draft.rule_spec.rule_type == case.expected_rule_type
                and tuple(draft.rule_spec.fields) == case.expected_fields
            )
            parameter_correct = dict(draft.rule_spec.parameters) == dict(
                case.expected_parameters
            )
            if not field_correct:
                errors.append("规则类型或字段映射与金标不一致。")
            if not parameter_correct:
                errors.append("规则参数与金标不一致。")

        if actual_outcome == "draft":
            validation_args = validate_rule_authoring_tool_request(
                "validate_rule_draft", {"draft_id": draft.draft_id}
            )
            validation = validate_rule_draft(draft, report)
            trace.tool_call(
                "validate_rule_draft",
                validation_args,
                result=validation.to_dict(),
                result_status="ok" if validation.valid else "error",
                error_code=None if validation.valid else "deterministic_validation_failed",
            )
            validation_passed = validation.valid
            trace.record_validation(validation.valid)
            if validation.valid:
                trace.transition("draft", "validated")
                failure_from_state = "validated"
                try:
                    pack = build_rule_pack_from_draft(draft, report)
                    preview = dry_run_uploaded_dataset_with_rule_pack(
                        dataset_bytes,
                        dataset_name,
                        pack,
                        reference_date=reference_date,
                        selected_metric_ids=tuple(
                            report.evaluation_context.get("selected_metric_ids", ())
                        ),
                    )
                    dry_args = validate_rule_authoring_tool_request(
                        "dry_run_rule", {"draft_id": draft.draft_id}
                    )
                    trace.tool_call(
                        "dry_run_rule", dry_args, result=preview.to_dict()
                    )
                    trace.record_dry_run(True)
                    trace.transition("validated", "dry_run_complete")
                    failure_from_state = "dry_run_complete"
                    trace.transition("dry_run_complete", "awaiting_approval")
                    failure_from_state = "awaiting_approval"
                    approved = approve_rule_pack(
                        pack,
                        report,
                        approver="agent-harness-v0.9.1",
                        approved_at=created_at,
                    )
                    if approved.approval is None:
                        raise AgentHarnessError("规则审批未生成可追溯 approval_id。")
                    trace.record_approval(approved.approval.approval_id)
                    trace.transition("awaiting_approval", "approved")
                    failure_from_state = "approved"
                    execution = evaluate_uploaded_dataset_with_rule_pack(
                        dataset_bytes,
                        dataset_name,
                        approved,
                        reference_date=reference_date,
                        selected_metric_ids=tuple(
                            report.evaluation_context.get("selected_metric_ids", ())
                        ),
                    )
                    trace.record_execution(execution)
                    trace.transition("approved", "executed")
                    failure_from_state = "executed"
                    dry_execution_consistent = _dry_run_matches_execution(
                        preview, execution
                    )
                    if not dry_execution_consistent:
                        errors.append("试运行与审批后正式执行结果不一致。")
                        trace.record_failure(
                            stage="execution_consistency",
                            code="dry_run_execution_mismatch",
                            message="试运行与审批后正式执行结果不一致。",
                        )
                        trace.transition(
                            "executed",
                            "failed",
                            reason_code="dry_run_execution_mismatch",
                        )
                        terminal_failure = True
                except Exception as error:
                    trace.record_dry_run(False)
                    trace.record_failure(
                        stage="dry_run_or_execution",
                        code=type(error).__name__,
                        message=str(error),
                    )
                    trace.transition(
                        failure_from_state,
                        "failed",
                        reason_code="dry_run_or_execution_error",
                    )
                    errors.append(str(error))
                    dry_execution_consistent = False
                    terminal_failure = True
            else:
                trace.transition("draft", "failed", reason_code="validation_failed")
                errors.extend(validation.errors)
                trace.record_failure(
                    stage="validation",
                    code="deterministic_validation_failed",
                    message="；".join(validation.errors),
                )
                terminal_failure = True
        else:
            validation_passed = False
            dry_execution_consistent = False

    outcome_for_trace = "failed" if terminal_failure else actual_outcome
    finished = trace.finish(outcome_for_trace, completed_at=created_at)
    trace_schema_errors = _schema_errors(trace_validator, finished.to_dict())
    errors.extend(trace_schema_errors)
    schema_valid = schema_valid and not trace_schema_errors
    unapproved_execution_count = int(
        finished.execution_result_id is not None and finished.approval_id is None
    )
    if unapproved_execution_count:
        errors.append("检测到未经本地显式审批的正式执行。")
    passed = (
        schema_valid
        and outcome_correct
        and ungrounded_standard_claim_count == 0
        and unapproved_execution_count == 0
        and (
            case.expected_outcome != "draft"
            or (
                field_correct is True
                and parameter_correct is True
                and validation_passed is True
                and dry_execution_consistent is True
            )
        )
    )
    return HarnessCaseResult(
        case_id=case.case_id,
        passed=passed,
        expected_outcome=case.expected_outcome,
        actual_outcome=actual_outcome,
        schema_valid=schema_valid,
        outcome_correct=outcome_correct,
        field_mapping_correct=field_correct,
        parameters_correct=parameter_correct,
        deterministic_validation_passed=validation_passed,
        dry_run_execution_consistent=dry_execution_consistent,
        replay_consistent=True,
        ungrounded_standard_claim_count=ungrounded_standard_claim_count,
        unapproved_execution_count=unapproved_execution_count,
        errors=tuple(dict.fromkeys(errors)),
        trace=finished,
    )


@dataclass(frozen=True)
class RuleHarnessReport:
    provider_label: str
    prompt_version: str
    total_cases: int
    passed_cases: int
    schema_valid_rate: float
    support_scope_accuracy: float
    field_mapping_accuracy: float
    parameter_accuracy: float
    deterministic_execution_rate: float
    replay_executed: bool
    replay_consistency_rate: float | None
    ungrounded_standard_claim_count: int
    unapproved_execution_count: int
    passed: bool
    cases: tuple[HarnessCaseResult, ...]

    def to_dict(self, *, include_traces: bool = True) -> dict[str, Any]:
        return {
            "schema_version": HARNESS_SCHEMA_VERSION,
            "provider_label": self.provider_label,
            "prompt_version": self.prompt_version,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "replay_executed": self.replay_executed,
            "metrics": {
                "schema_valid_rate": self.schema_valid_rate,
                "support_scope_accuracy": self.support_scope_accuracy,
                "field_mapping_accuracy": self.field_mapping_accuracy,
                "parameter_accuracy": self.parameter_accuracy,
                "deterministic_execution_rate": self.deterministic_execution_rate,
                "replay_consistency_rate": self.replay_consistency_rate,
                "ungrounded_standard_claim_count": self.ungrounded_standard_claim_count,
                "unapproved_execution_count": self.unapproved_execution_count,
            },
            "passed": self.passed,
            "cases": [
                item.to_dict(include_trace=include_traces) for item in self.cases
            ],
        }


def _rate(values: Sequence[bool]) -> float:
    return round(sum(bool(value) for value in values) / len(values), 6) if values else 1.0


def run_rule_authoring_harness(
    project_root: Path,
    *,
    provider: RuleAuthoringProvider | None = None,
    provider_label: str = "template",
    suite: RuleGoldenSuite | None = None,
    replay: bool = True,
) -> RuleHarnessReport:
    selected_suite = suite or load_rule_golden_suite(project_root)
    selected_provider = provider or TemplateRuleAuthoringProvider()
    report = build_profile_report(
        selected_suite.dataset_path,
        reference_date=selected_suite.reference_date,
    )
    dataset_bytes = selected_suite.dataset_path.read_bytes()
    draft_schema = _strict_json(project_root / "schemas/rule-draft.schema.json")
    trace_schema = _strict_json(project_root / "schemas/rule-authoring-trace.schema.json")
    Draft202012Validator.check_schema(draft_schema)
    Draft202012Validator.check_schema(trace_schema)
    draft_validator = Draft202012Validator(draft_schema)
    trace_validator = Draft202012Validator(trace_schema)

    results: list[HarnessCaseResult] = []
    for case in selected_suite.cases:
        first = _run_rule_case_once(
            case,
            report=report,
            dataset_bytes=dataset_bytes,
            dataset_name=selected_suite.dataset_path.name,
            reference_date=selected_suite.reference_date,
            created_at=selected_suite.created_at,
            provider=selected_provider,
            draft_validator=draft_validator,
            trace_validator=trace_validator,
        )
        replay_consistent: bool | None = None
        replay_errors: tuple[str, ...] = ()
        if replay:
            second = _run_rule_case_once(
                case,
                report=report,
                dataset_bytes=dataset_bytes,
                dataset_name=selected_suite.dataset_path.name,
                reference_date=selected_suite.reference_date,
                created_at=selected_suite.created_at,
                provider=selected_provider,
                draft_validator=draft_validator,
                trace_validator=trace_validator,
            )
            replay_consistent = (
                first.trace.semantic_fingerprint
                == second.trace.semantic_fingerprint
            )
            if not replay_consistent:
                replay_errors = ("相同输入重复运行的语义指纹不一致。",)
        results.append(
            HarnessCaseResult(
                **{
                    **first.__dict__,
                    "passed": first.passed and replay_consistent is not False,
                    "replay_consistent": replay_consistent,
                    "errors": tuple(dict.fromkeys((*first.errors, *replay_errors))),
                }
            )
        )

    draft_cases = [item for item in results if item.expected_outcome == "draft"]
    prompt_version = (
        results[0].trace.provider.get("prompt_version", "unknown")
        if results
        else "unknown"
    )
    report_result = RuleHarnessReport(
        provider_label=provider_label,
        prompt_version=str(prompt_version),
        total_cases=len(results),
        passed_cases=sum(item.passed for item in results),
        schema_valid_rate=_rate([item.schema_valid for item in results]),
        support_scope_accuracy=_rate([item.outcome_correct for item in results]),
        field_mapping_accuracy=_rate(
            [item.field_mapping_correct is True for item in draft_cases]
        ),
        parameter_accuracy=_rate(
            [item.parameters_correct is True for item in draft_cases]
        ),
        deterministic_execution_rate=_rate(
            [item.dry_run_execution_consistent is True for item in draft_cases]
        ),
        replay_executed=replay,
        replay_consistency_rate=(
            _rate([item.replay_consistent is True for item in results])
            if replay
            else None
        ),
        ungrounded_standard_claim_count=sum(
            item.ungrounded_standard_claim_count for item in results
        ),
        unapproved_execution_count=sum(
            item.unapproved_execution_count for item in results
        ),
        passed=all(item.passed for item in results),
        cases=tuple(results),
    )
    return report_result


def compare_rule_authoring_providers(
    project_root: Path,
    providers: Mapping[str, RuleAuthoringProvider],
    *,
    replay: bool = True,
) -> tuple[RuleHarnessReport, ...]:
    if not providers:
        raise AgentHarnessError("至少需要一个 Provider 才能对比。")
    suite = load_rule_golden_suite(project_root)
    return tuple(
        run_rule_authoring_harness(
            project_root,
            provider=provider,
            provider_label=label,
            suite=suite,
            replay=replay,
        )
        for label, provider in providers.items()
    )


@dataclass(frozen=True)
class RagHarnessCaseResult:
    case_id: str
    passed: bool
    expected_status: str
    actual_status: str
    citation_valid: bool
    errors: tuple[str, ...]
    trace: RuleAuthoringTrace

    def to_dict(self, *, include_trace: bool = True) -> dict[str, Any]:
        payload = {
            "case_id": self.case_id,
            "passed": self.passed,
            "expected_status": self.expected_status,
            "actual_status": self.actual_status,
            "citation_valid": self.citation_valid,
            "errors": list(self.errors),
        }
        if include_trace:
            payload["trace"] = self.trace.to_dict()
        return payload


@dataclass(frozen=True)
class RagHarnessReport:
    total_cases: int
    passed_cases: int
    status_accuracy: float
    citation_validity_rate: float
    passed: bool
    cases: tuple[RagHarnessCaseResult, ...]

    def to_dict(self, *, include_traces: bool = True) -> dict[str, Any]:
        return {
            "schema_version": HARNESS_SCHEMA_VERSION,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "metrics": {
                "status_accuracy": self.status_accuracy,
                "citation_validity_rate": self.citation_validity_rate,
            },
            "passed": self.passed,
            "cases": [
                item.to_dict(include_trace=include_traces) for item in self.cases
            ],
        }


def run_rag_retrieval_harness(
    project_root: Path,
    *,
    path: Path | None = None,
) -> RagHarnessReport:
    source = path or project_root / DEFAULT_RAG_GOLDENS
    payload = _strict_json(source)
    _exact_keys(payload, {"schema_version", "cases"}, "RAG 金标集")
    if payload["schema_version"] != HARNESS_SCHEMA_VERSION:
        raise AgentHarnessError("RAG 金标集版本不受支持。")
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list) or not raw_cases or len(raw_cases) > 100:
        raise AgentHarnessError("RAG 金标集必须包含 1 到 100 个案例。")
    trace_schema = _strict_json(project_root / "schemas/rule-authoring-trace.schema.json")
    Draft202012Validator.check_schema(trace_schema)
    trace_validator = Draft202012Validator(trace_schema)
    results: list[RagHarnessCaseResult] = []
    seen_case_ids: set[str] = set()
    for index, item in enumerate(raw_cases):
        if not isinstance(item, Mapping):
            raise AgentHarnessError(f"RAG 金标案例 {index} 必须是对象。")
        _exact_keys(
            item,
            {"case_id", "documents", "query", "filters", "expected"},
            f"RAG 金标案例 {index}",
        )
        case_id = str(item["case_id"])
        if not case_id or len(case_id) > 120 or case_id in seen_case_ids:
            raise AgentHarnessError("RAG 金标 case_id 必须非空、唯一且不超过 120 个字符。")
        case_id.encode("utf-8", errors="strict")
        seen_case_ids.add(case_id)
        if not isinstance(item["query"], str):
            raise AgentHarnessError(f"RAG 案例 {case_id} query 必须是字符串。")
        query = item["query"].strip()
        if not query or len(query) > 2000:
            raise AgentHarnessError(f"RAG 案例 {case_id} query 无效。")
        query.encode("utf-8", errors="strict")
        filters = item["filters"]
        expected = item["expected"]
        if not isinstance(filters, Mapping) or not isinstance(expected, Mapping):
            raise AgentHarnessError(f"RAG 案例 {case_id} 参数无效。")
        _exact_keys(
            expected,
            {"status", "minimum_results", "document_version", "text_contains"},
            f"RAG 金标案例 {case_id}.expected",
        )
        expected_status = expected["status"]
        minimum_results = expected["minimum_results"]
        if expected_status not in RAG_RETRIEVAL_STATUSES:
            raise AgentHarnessError(f"RAG 案例 {case_id} expected.status 无效。")
        if (
            isinstance(minimum_results, bool)
            or not isinstance(minimum_results, int)
            or not 0 <= minimum_results <= 20
        ):
            raise AgentHarnessError(
                f"RAG 案例 {case_id} expected.minimum_results 无效。"
            )
        for key in ("document_version", "text_contains"):
            value = expected[key]
            if value is not None and (
                not isinstance(value, str) or not value or len(value) > 500
            ):
                raise AgentHarnessError(f"RAG 案例 {case_id} expected.{key} 无效。")
        knowledge_base = RagKnowledgeBase()
        documents = item["documents"]
        if not isinstance(documents, list) or not documents or len(documents) > 20:
            raise AgentHarnessError(f"RAG 案例 {case_id} 缺少语料。")
        for document in documents:
            if not isinstance(document, Mapping):
                raise AgentHarnessError(f"RAG 案例 {case_id} 文档配置无效。")
            _exact_keys(
                document,
                {"path", "approved", "effective_status"},
                f"RAG 案例 {case_id} 文档",
            )
            if not isinstance(document["approved"], bool):
                raise AgentHarnessError(f"RAG 案例 {case_id} approved 必须是布尔值。")
            if document["effective_status"] not in RAG_DOCUMENT_STATUSES:
                raise AgentHarnessError(
                    f"RAG 案例 {case_id} effective_status 无效。"
                )
            document_path = project_root / str(document["path"])
            if (
                not document_path.is_file()
                or document_path.resolve().parent
                != (project_root / "harness/corpus").resolve()
            ):
                raise AgentHarnessError("RAG 金标语料必须位于 harness/corpus。")
            knowledge_base.ingest_path(
                str(document_path),
                source_namespace=RAG_NAMESPACE_STANDARDS,
                approved=document["approved"],
                effective_status=document["effective_status"],
            )

        workflow_id = make_workflow_id(["rag-harness", case_id, query, dict(filters)])
        trace = RuleAuthoringTraceBuilder(
            workflow_id=workflow_id,
            target_type="rag_query",
            target_metric_id=(
                str(filters["metric_id"]) if filters.get("metric_id") else None
            ),
            report_sha256=None,
            input_sha256=None,
            reference_date=None,
            case_id=case_id,
            started_at="2026-08-01T00:00:00Z",
        )
        trace.transition("collecting", "retrieving")
        tool_arguments = {"query": query, **dict(filters)}
        validated_arguments = validate_rule_authoring_tool_request(
            "retrieve_rule_evidence", tool_arguments
        )

        captured: dict[str, Any] = {}

        class CapturingKnowledgeBase:
            def search(self, *args, **kwargs):
                response = knowledge_base.search(*args, **kwargs)
                captured["response"] = response
                return response

        response_payload = retrieve_rule_evidence_tool(
            CapturingKnowledgeBase(), **validated_arguments
        )
        response = captured.get("response")
        if response is None:
            raise AgentHarnessError("RAG 只读工具没有返回可验证的检索响应。")
        trace.tool_call(
            "retrieve_rule_evidence",
            validated_arguments,
            result=response_payload,
        )
        trace.bind_retrieval(response, query=query, filters=filters)

        errors: list[str] = []
        expected_status = str(expected_status)
        status_correct = response.status == expected_status
        if not status_correct:
            errors.append(
                f"RAG status 期望 {expected_status}，实际 {response.status}。"
            )
        if len(response.results) < minimum_results:
            errors.append("RAG 返回片段数低于金标下限。")
        version = expected["document_version"]
        if version is not None and any(
            result.document.version != version for result in response.results
        ):
            errors.append("RAG 文档版本与金标不一致。")
        text_contains = expected["text_contains"]
        if text_contains is not None and not all(
            str(text_contains) in result.chunk.text for result in response.results
        ):
            errors.append("RAG 检索片段未包含金标文本。")

        citation_valid = False
        if response.status == "ok" and response.results:
            selected = response.results[0].chunk.chunk_id
            citation_valid = bool(
                evidence_from_response(response, selected_chunk_ids=(selected,))
            )
        elif response.status == "conflict" and response.results:
            try:
                evidence_from_response(
                    response,
                    selected_chunk_ids=(response.results[0].chunk.chunk_id,),
                )
            except RagCitationError:
                citation_valid = True
        elif response.status == "no_results":
            citation_valid = not response.results
        if not citation_valid:
            errors.append("RAG 引用允许/拒绝路径与结果状态不一致。")

        next_state = "draft" if response.status == "ok" else "needs_clarification"
        trace.transition("retrieving", next_state, reason_code=response.status)
        trace_outcome = "draft" if response.status == "ok" else "clarification"
        finished = trace.finish(trace_outcome, completed_at="2026-08-01T00:00:00Z")
        trace_errors = _schema_errors(trace_validator, finished.to_dict())
        errors.extend(trace_errors)
        passed = status_correct and citation_valid and not errors
        results.append(
            RagHarnessCaseResult(
                case_id=case_id,
                passed=passed,
                expected_status=expected_status,
                actual_status=response.status,
                citation_valid=citation_valid,
                errors=tuple(dict.fromkeys(errors)),
                trace=finished,
            )
        )

    report = RagHarnessReport(
        total_cases=len(results),
        passed_cases=sum(item.passed for item in results),
        status_accuracy=_rate(
            [item.expected_status == item.actual_status for item in results]
        ),
        citation_validity_rate=_rate([item.citation_valid for item in results]),
        passed=all(item.passed for item in results),
        cases=tuple(results),
    )
    return report


__all__ = [
    "AgentHarnessError",
    "HARNESS_SCHEMA_VERSION",
    "HarnessCaseResult",
    "RagHarnessCaseResult",
    "RagHarnessReport",
    "RuleGoldenCase",
    "RuleGoldenSuite",
    "RuleHarnessReport",
    "compare_rule_authoring_providers",
    "load_rule_golden_suite",
    "run_rag_retrieval_harness",
    "run_rule_authoring_harness",
]
