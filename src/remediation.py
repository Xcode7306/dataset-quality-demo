"""v0.5 本地整改闭环 Agent、任务分派与治理记录。

本模块只消费已经验证的 ``ReportComparison``，不读取两份原始报告，
也不重新计算指标或风险。默认生成器是完全本地的确定性模板。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker

from .comparison_models import ReportComparison
from .comparison_service import (
    ReportComparisonError,
    validate_report_comparison,
)
from .history_store import DEFAULT_HISTORY_POLICY


REMEDIATION_SCHEMA_VERSION = "0.1"
GOVERNANCE_SCHEMA_VERSION = "0.1"
REMEDIATION_GENERATOR_VERSION = "0.5"
MAX_REMEDIATION_TASKS = 30

TaskPriority = Literal["high", "medium", "low"]
TaskStatus = Literal["open", "in_progress", "done", "accepted_risk"]
TaskCategory = Literal["risk", "metric", "assessability", "schema"]

_CONTROL_OR_SURROGATE_PATTERN = re.compile(
    r"[\x00-\x1f\x7f\ud800-\udfff]"
)
_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_CATEGORY_ORDER = {
    "risk": 0,
    "metric": 1,
    "assessability": 2,
    "schema": 3,
}
_RISK_TASK_ORDER = {
    "added": 0,
    "severity_increased": 1,
    "persistent": 2,
    "severity_decreased": 3,
}
_PLAN_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "remediation-plan.schema.json"
)
_GOVERNANCE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "governance-record.schema.json"
)


class RemediationValidationError(ValueError):
    """整改计划、分派或治理记录未通过确定性校验。"""


@dataclass(frozen=True)
class ImprovementSummary:
    headline: str
    improved_change_ids: tuple[str, ...]
    regressed_change_ids: tuple[str, ...]
    resolved_risk_change_ids: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "improved_change_ids": list(self.improved_change_ids),
            "regressed_change_ids": list(self.regressed_change_ids),
            "resolved_risk_change_ids": list(
                self.resolved_risk_change_ids
            ),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class RemediationTask:
    task_id: str
    category: TaskCategory
    priority: TaskPriority
    status: TaskStatus
    title: str
    detail: str
    acceptance_criteria: tuple[str, ...]
    suggested_owner_role: str
    assignee: str | None
    due_date: str | None
    change_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "priority": self.priority,
            "status": self.status,
            "title": self.title,
            "detail": self.detail,
            "acceptance_criteria": list(self.acceptance_criteria),
            "suggested_owner_role": self.suggested_owner_role,
            "assignee": self.assignee,
            "due_date": self.due_date,
            "change_ids": list(self.change_ids),
        }


@dataclass(frozen=True)
class RemediationPlan:
    plan_id: str
    plan_sha256: str
    comparison_sha256: str
    improvement_summary: ImprovementSummary
    tasks: tuple[RemediationTask, ...]
    next_round_suggestions: tuple[str, ...]
    schema_version: str = REMEDIATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "plan_sha256": self.plan_sha256,
            "comparison_sha256": self.comparison_sha256,
            "generator": {
                "name": "quality-remediation-agent",
                "version": REMEDIATION_GENERATOR_VERSION,
                "mode": "local_template",
            },
            "improvement_summary": self.improvement_summary.to_dict(),
            "tasks": [task.to_dict() for task in self.tasks],
            "next_round_suggestions": list(self.next_round_suggestions),
        }


@dataclass(frozen=True)
class GovernanceRecord:
    record_id: str
    record_sha256: str
    comparison_sha256: str
    plan_sha256: str
    baseline_report_sha256: str
    target_report_sha256: str
    dataset_series_id: str
    operator: str
    recorded_at: str
    outcomes: dict[str, Any]
    history_policy: dict[str, Any]
    schema_version: str = GOVERNANCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "record_sha256": self.record_sha256,
            "comparison_sha256": self.comparison_sha256,
            "plan_sha256": self.plan_sha256,
            "baseline_report_sha256": self.baseline_report_sha256,
            "target_report_sha256": self.target_report_sha256,
            "dataset_series_id": self.dataset_series_id,
            "operator": {
                "label": self.operator,
                "identity_verified": False,
            },
            "recorded_at": self.recorded_at,
            "outcomes": deepcopy(self.outcomes),
            "history_policy": deepcopy(self.history_policy),
        }


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8", errors="strict")


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(
        json.dumps(
            list(parts),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    ).hexdigest()
    return f"{prefix}-{digest[:24]}"


def _strict_optional_text(
    value: Any,
    label: str,
    *,
    maximum: int,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RemediationValidationError(f"{label}必须是文本。")
    normalized = value.strip()
    if not normalized:
        return None
    if (
        len(normalized) > maximum
        or _CONTROL_OR_SURROGATE_PATTERN.search(normalized)
    ):
        raise RemediationValidationError(
            f"{label}不能超过 {maximum} 个字符，且不能包含控制字符。"
        )
    return normalized


def _required_text(value: Any, label: str, *, maximum: int) -> str:
    normalized = _strict_optional_text(value, label, maximum=maximum)
    if normalized is None:
        raise RemediationValidationError(f"{label}不能为空。")
    return normalized


def _safe_generated_text(
    value: object,
    *,
    maximum: int,
    fallback: str,
) -> str:
    """约束来自固定报告元数据的展示文本，避免控制字符破坏导出。"""

    normalized = _CONTROL_OR_SURROGATE_PATTERN.sub(" ", str(value)).strip()
    if not normalized:
        normalized = fallback
    if len(normalized) > maximum:
        normalized = normalized[: maximum - 1].rstrip() + "…"
    return normalized


def _date_text(value: date | str | None) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = date.fromisoformat(value.strip())
        except ValueError as error:
            raise RemediationValidationError(
                "任务截止日期必须是 YYYY-MM-DD。"
            ) from error
    else:
        raise RemediationValidationError("任务截止日期类型无效。")
    return parsed.isoformat()


def _utc_text(value: datetime | str | None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(
                value.strip().replace("Z", "+00:00")
            )
        except ValueError as error:
            raise RemediationValidationError(
                "治理记录时间必须是 ISO 8601 时间。"
            ) from error
    else:
        raise RemediationValidationError("治理记录时间类型无效。")
    if parsed.tzinfo is None:
        raise RemediationValidationError("治理记录时间必须包含时区。")
    return (
        parsed.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _task(
    comparison_sha256: str,
    *,
    category: TaskCategory,
    priority: TaskPriority,
    title: str,
    detail: str,
    criteria: Sequence[str],
    owner_role: str,
    change_ids: Sequence[str],
) -> RemediationTask:
    ordered_change_ids = tuple(dict.fromkeys(change_ids))
    return RemediationTask(
        task_id=_stable_id(
            "task",
            comparison_sha256,
            category,
            *ordered_change_ids,
        ),
        category=category,
        priority=priority,
        status="open",
        title=_safe_generated_text(
            title,
            maximum=500,
            fallback="复核确定性比较变化",
        ),
        detail=_safe_generated_text(
            detail,
            maximum=2000,
            fallback="请复核关联变化及其确定性证据。",
        ),
        acceptance_criteria=tuple(
            _safe_generated_text(
                item,
                maximum=1000,
                fallback="完成复核并记录结果。",
            )
            for item in criteria
        ),
        suggested_owner_role=_safe_generated_text(
            owner_role,
            maximum=200,
            fallback="数据质量负责人",
        ),
        assignee=None,
        due_date=None,
        change_ids=ordered_change_ids,
    )


def _risk_priority(level: str | None) -> TaskPriority:
    if level == "warning":
        return "high"
    if level == "attention":
        return "medium"
    return "low"


def _format_value(value: Any, unit: str | None) -> str:
    if value is None:
        return "无法评估"
    if unit == "ratio":
        return f"{float(value):.2%}"
    if unit == "days":
        return f"{value} 天"
    return str(value)


def _plan_without_hash(plan: RemediationPlan) -> dict[str, Any]:
    payload = plan.to_dict()
    payload.pop("plan_sha256", None)
    return payload


def _rehash_plan(plan: RemediationPlan) -> RemediationPlan:
    plan_sha256 = hashlib.sha256(
        _canonical_bytes(_plan_without_hash(plan))
    ).hexdigest()
    return replace(plan, plan_sha256=plan_sha256)


def build_action_plan(
    comparison: ReportComparison | Mapping[str, Any],
) -> RemediationPlan:
    """从确定性变化生成本地模板整改任务和改进摘要。"""

    try:
        payload = validate_report_comparison(comparison)
    except ReportComparisonError as error:
        raise RemediationValidationError(str(error)) from error
    comparison_sha256 = payload["comparison_sha256"]
    metric_changes = payload["metric_changes"]
    risk_changes = payload["risk_changes"]
    assessability_changes = payload["assessability_changes"]
    schema_changes = payload["schema_changes"]

    improved_change_ids = tuple(
        change["change_id"]
        for change in metric_changes
        if change["classification"] == "improved"
    )
    resolved_risk_change_ids = tuple(
        change["change_id"]
        for change in risk_changes
        if change["classification"] in {
            "resolved",
            "severity_decreased",
        }
    )
    regressed_change_ids = tuple(
        [
            change["change_id"]
            for change in metric_changes
            if change["classification"] in {
                "worsened",
                "became_not_assessable",
            }
        ]
        + [
            change["change_id"]
            for change in risk_changes
            if change["classification"] in {
                "added",
                "severity_increased",
            }
        ]
        + [
            change["change_id"]
            for change in assessability_changes
            if change["classification"] == "became_not_assessable"
        ]
    )
    if regressed_change_ids and (
        improved_change_ids or resolved_risk_change_ids
    ):
        headline = "本轮同时存在改善与退化，建议先处理新增或升级风险。"
    elif regressed_change_ids:
        headline = "本轮存在退化或新增风险，需要进入下一轮整改。"
    elif improved_change_ids or resolved_risk_change_ids:
        headline = "本轮已出现可追溯改善，仍需复核未解除风险与结构变化。"
    else:
        headline = "本轮未形成可确定的改善或退化结论。"

    tasks: list[RemediationTask] = []
    task_urgency: dict[str, int] = {}
    risk_covered_metric_keys: set[str] = set()
    for change in risk_changes:
        classification = change["classification"]
        if classification not in {
            "added",
            "severity_increased",
            "persistent",
            "severity_decreased",
        }:
            continue
        level = change["target_level"]
        risk_covered_metric_keys.update(change["related_metric_keys"])
        if classification == "added":
            prefix = "处理新增风险"
        elif classification == "severity_increased":
            prefix = "处理升级风险"
        elif classification == "severity_decreased":
            prefix = "继续跟踪已降级风险"
        else:
            prefix = "继续处理未解除风险"
        task = _task(
            comparison_sha256,
            category="risk",
            priority=_risk_priority(level),
            title=f"{prefix}：{change['title']}",
            detail=(
                "依据确定性风险变化复核关联指标和问题位置；"
                "不得直接把风险提示当作已经证明的数据错误。"
            ),
            criteria=(
                "完成问题原因确认、修复或经授权的接受风险记录。",
                "重新生成固定 QualityReport，并在下一轮比较中核对风险状态。",
            ),
            owner_role="数据责任部门",
            change_ids=(change["change_id"],),
        )
        tasks.append(task)
        task_urgency[task.task_id] = _RISK_TASK_ORDER[classification]

    for change in metric_changes:
        if (
            change["classification"] != "worsened"
            or change["metric_key"] in risk_covered_metric_keys
        ):
            continue
        tasks.append(
            _task(
                comparison_sha256,
                category="metric",
                priority="medium",
                title=f"复核恶化指标：{change['name']}",
                detail=(
                    f"该指标从 {_format_value(change['baseline_value'], change['unit'])} "
                    f"变化为 {_format_value(change['target_value'], change['unit'])}；"
                    "请结合业务规则确认原因和影响范围。"
                ),
                criteria=(
                    "记录指标变化原因和整改措施。",
                    "下一轮报告使用相同口径复算，并引用同一 metric_key。",
                ),
                owner_role="数据质量负责人",
                change_ids=(change["change_id"],),
            )
        )

    for change in assessability_changes:
        if change["classification"] != "became_not_assessable":
            continue
        tasks.append(
            _task(
                comparison_sha256,
                category="assessability",
                priority="medium",
                title=f"恢复可评估能力：{change['name']}",
                detail=(
                    "整改后报告失去了该指标的计算依据；请补齐必要字段或"
                    "明确记录业务规则仍不足的原因。"
                ),
                criteria=(
                    "补齐计算所需信息，或形成经确认的无法评估说明。",
                    "下一轮比较中该项恢复可评估，或保留可追溯限制依据。",
                ),
                owner_role="数据提供部门",
                change_ids=(change["change_id"],),
            )
        )

    for change in schema_changes:
        kind = change["kind"]
        if kind not in {
            "field_added",
            "field_removed",
            "field_type_changed",
            "field_order_changed",
        }:
            continue
        field = change["field"] or "字段顺序"
        priority: TaskPriority = (
            "medium"
            if kind in {"field_removed", "field_type_changed"}
            else "low"
        )
        labels = {
            "field_added": "确认新增字段",
            "field_removed": "确认删除字段",
            "field_type_changed": "确认字段类型变化",
            "field_order_changed": "确认字段顺序变化",
        }
        tasks.append(
            _task(
                comparison_sha256,
                category="schema",
                priority=priority,
                title=f"{labels[kind]}：{field}",
                detail=(
                    "字段结构变化本身不自动代表质量改善或恶化；"
                    "请确认它符合发布协议和下游使用方预期。"
                ),
                criteria=(
                    "记录结构变更原因、影响范围和兼容方案。",
                    "确认数据字典与下游接口已同步。",
                ),
                owner_role="数据结构维护人员",
                change_ids=(change["change_id"],),
            )
        )

    candidate_task_count = len(tasks)
    tasks = sorted(
        tasks,
        key=lambda task: (
            _PRIORITY_ORDER[task.priority],
            _CATEGORY_ORDER[task.category],
            task_urgency.get(task.task_id, 0),
            task.task_id,
        ),
    )[:MAX_REMEDIATION_TASKS]
    truncated_task_count = candidate_task_count - len(tasks)
    suggestions: list[str] = []
    if payload["compatibility"]["status"] != "full":
        suggestions.append(
            "先统一指标引擎、阈值或业务规则口径，再判断受限项目的改善或恶化。"
        )
    if tasks:
        suggestions.append(
            "完成或确认任务后，使用相同治理对象标识生成下一份固定报告并再次比较。"
        )
    else:
        suggestions.append(
            "保持当前口径，下一轮仍使用固定报告哈希验证趋势是否持续。"
        )
    suggestions.append(
        "保留 ReportComparison、行动计划和治理记录；不要覆盖两份原始报告。"
    )
    plan_limitations = list(payload["limitations"])
    if truncated_task_count:
        plan_limitations.append(
            f"候选任务共 {candidate_task_count} 项；按优先级仅生成前 "
            f"{MAX_REMEDIATION_TASKS} 项，另有 {truncated_task_count} 项未进入计划。"
        )
        suggestions.insert(
            0,
            "当前计划已达到任务上限；处理后应重新比较并生成下一批任务。",
        )

    provisional = RemediationPlan(
        plan_id=_stable_id(
            "plan",
            comparison_sha256,
            REMEDIATION_GENERATOR_VERSION,
        ),
        plan_sha256="0" * 64,
        comparison_sha256=comparison_sha256,
        improvement_summary=ImprovementSummary(
            headline=headline,
            improved_change_ids=improved_change_ids,
            regressed_change_ids=regressed_change_ids,
            resolved_risk_change_ids=resolved_risk_change_ids,
            limitations=tuple(plan_limitations),
        ),
        tasks=tuple(tasks),
        next_round_suggestions=tuple(suggestions),
    )
    plan = _rehash_plan(provisional)
    validate_remediation_plan(plan)
    return plan


def assign_task(
    plan: RemediationPlan,
    task_id: str,
    *,
    assignee: str | None = None,
    due_date: date | str | None = None,
    status: TaskStatus = "open",
) -> RemediationPlan:
    """更新一项人工分派信息；不改变任务证据、内容或最低优先级。"""

    validate_remediation_plan(plan)
    if status not in {"open", "in_progress", "done", "accepted_risk"}:
        raise RemediationValidationError("任务状态不在允许范围内。")
    normalized_assignee = _strict_optional_text(
        assignee,
        "负责人",
        maximum=100,
    )
    normalized_due_date = _date_text(due_date)
    found = False
    tasks: list[RemediationTask] = []
    for task in plan.tasks:
        if task.task_id != task_id:
            tasks.append(task)
            continue
        found = True
        tasks.append(
            replace(
                task,
                assignee=normalized_assignee,
                due_date=normalized_due_date,
                status=status,
            )
        )
    if not found:
        raise RemediationValidationError("待分派任务不存在于当前行动计划。")
    updated = _rehash_plan(replace(plan, tasks=tuple(tasks)))
    validate_remediation_plan(updated)
    return updated


@lru_cache(maxsize=1)
def _plan_validator() -> Draft202012Validator:
    schema = json.loads(_PLAN_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_remediation_plan(
    plan: RemediationPlan | Mapping[str, Any],
) -> dict[str, Any]:
    payload = (
        plan.to_dict()
        if isinstance(plan, RemediationPlan)
        else deepcopy(dict(plan))
    )
    errors = sorted(
        _plan_validator().iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        first = errors[0]
        path = ".".join(str(part) for part in first.absolute_path) or "root"
        raise RemediationValidationError(
            f"RemediationPlan Schema 校验失败（{path}）：{first.message}"
        )
    hash_payload = deepcopy(payload)
    reported_hash = hash_payload.pop("plan_sha256")
    actual_hash = hashlib.sha256(_canonical_bytes(hash_payload)).hexdigest()
    if reported_hash != actual_hash:
        raise RemediationValidationError("RemediationPlan 自身哈希校验失败。")
    expected_plan_id = _stable_id(
        "plan",
        payload["comparison_sha256"],
        REMEDIATION_GENERATOR_VERSION,
    )
    if payload["plan_id"] != expected_plan_id:
        raise RemediationValidationError(
            "RemediationPlan ID 与比较哈希不一致。"
        )
    task_ids = [task["task_id"] for task in payload["tasks"]]
    if len(task_ids) != len(set(task_ids)):
        raise RemediationValidationError("RemediationPlan 包含重复任务 ID。")
    for task in payload["tasks"]:
        expected_task_id = _stable_id(
            "task",
            payload["comparison_sha256"],
            task["category"],
            *task["change_ids"],
        )
        if task["task_id"] != expected_task_id:
            raise RemediationValidationError(
                "RemediationPlan 包含与比较变化不一致的任务 ID。"
            )
    priority_order = [
        _PRIORITY_ORDER[task["priority"]]
        for task in payload["tasks"]
    ]
    if priority_order != sorted(priority_order):
        raise RemediationValidationError(
            "RemediationPlan 任务顺序与确定性优先级不一致。"
        )
    return payload


def _record_without_hash(record: GovernanceRecord) -> dict[str, Any]:
    payload = record.to_dict()
    payload.pop("record_sha256", None)
    return payload


def build_governance_record(
    comparison: ReportComparison | Mapping[str, Any],
    plan: RemediationPlan | Mapping[str, Any],
    *,
    operator: str,
    recorded_at: datetime | str | None = None,
) -> GovernanceRecord:
    """生成绑定比较、计划、两份报告和当时历史策略的治理留痕。"""

    try:
        comparison_payload = validate_report_comparison(comparison)
    except ReportComparisonError as error:
        raise RemediationValidationError(str(error)) from error
    plan_payload = validate_remediation_plan(plan)
    if (
        plan_payload["comparison_sha256"]
        != comparison_payload["comparison_sha256"]
    ):
        raise RemediationValidationError(
            "行动计划与当前 ReportComparison 不匹配。"
        )
    canonical_plan = build_action_plan(comparison_payload).to_dict()
    for field in (
        "schema_version",
        "plan_id",
        "comparison_sha256",
        "generator",
        "improvement_summary",
        "next_round_suggestions",
    ):
        if plan_payload[field] != canonical_plan[field]:
            raise RemediationValidationError(
                "行动计划不是当前 ReportComparison 的确定性派生结果。"
            )
    immutable_task_fields = (
        "task_id",
        "category",
        "priority",
        "title",
        "detail",
        "acceptance_criteria",
        "suggested_owner_role",
        "change_ids",
    )
    if len(plan_payload["tasks"]) != len(canonical_plan["tasks"]):
        raise RemediationValidationError(
            "行动计划任务集合与当前 ReportComparison 不匹配。"
        )
    for provided, canonical in zip(
        plan_payload["tasks"],
        canonical_plan["tasks"],
    ):
        if any(
            provided[field] != canonical[field]
            for field in immutable_task_fields
        ):
            raise RemediationValidationError(
                "行动计划任务证据或优先级已偏离确定性比较。"
            )
    operator_text = _required_text(operator, "记录人标识", maximum=100)
    recorded_at_text = _utc_text(recorded_at)
    task_status_counts = {
        status: sum(task["status"] == status for task in plan_payload["tasks"])
        for status in ("open", "in_progress", "done", "accepted_risk")
    }
    open_task_ids = [
        task["task_id"]
        for task in plan_payload["tasks"]
        if task["status"] in {"open", "in_progress"}
    ]
    outcomes = {
        "improved_change_ids": plan_payload["improvement_summary"][
            "improved_change_ids"
        ],
        "regressed_change_ids": plan_payload["improvement_summary"][
            "regressed_change_ids"
        ],
        "resolved_risk_change_ids": plan_payload["improvement_summary"][
            "resolved_risk_change_ids"
        ],
        "task_status_counts": task_status_counts,
        "open_task_ids": open_task_ids,
    }
    provisional = GovernanceRecord(
        record_id=_stable_id(
            "governance",
            comparison_payload["comparison_sha256"],
            plan_payload["plan_sha256"],
            operator_text,
            recorded_at_text,
        ),
        record_sha256="0" * 64,
        comparison_sha256=comparison_payload["comparison_sha256"],
        plan_sha256=plan_payload["plan_sha256"],
        baseline_report_sha256=comparison_payload["baseline"][
            "report_sha256"
        ],
        target_report_sha256=comparison_payload["target"]["report_sha256"],
        dataset_series_id=comparison_payload["lineage"][
            "dataset_series_id"
        ],
        operator=operator_text,
        recorded_at=recorded_at_text,
        outcomes=outcomes,
        history_policy=DEFAULT_HISTORY_POLICY.to_dict(),
    )
    record_sha256 = hashlib.sha256(
        _canonical_bytes(_record_without_hash(provisional))
    ).hexdigest()
    record = replace(provisional, record_sha256=record_sha256)
    validate_governance_record(record)
    return record


@lru_cache(maxsize=1)
def _governance_validator() -> Draft202012Validator:
    schema = json.loads(_GOVERNANCE_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_governance_record(
    record: GovernanceRecord | Mapping[str, Any],
) -> dict[str, Any]:
    payload = (
        record.to_dict()
        if isinstance(record, GovernanceRecord)
        else deepcopy(dict(record))
    )
    errors = sorted(
        _governance_validator().iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        first = errors[0]
        path = ".".join(str(part) for part in first.absolute_path) or "root"
        raise RemediationValidationError(
            f"GovernanceRecord Schema 校验失败（{path}）：{first.message}"
        )
    hash_payload = deepcopy(payload)
    reported_hash = hash_payload.pop("record_sha256")
    actual_hash = hashlib.sha256(_canonical_bytes(hash_payload)).hexdigest()
    if reported_hash != actual_hash:
        raise RemediationValidationError("GovernanceRecord 自身哈希校验失败。")
    expected_record_id = _stable_id(
        "governance",
        payload["comparison_sha256"],
        payload["plan_sha256"],
        payload["operator"]["label"],
        payload["recorded_at"],
    )
    if payload["record_id"] != expected_record_id:
        raise RemediationValidationError(
            "GovernanceRecord ID 与绑定对象不一致。"
        )
    status_counts = payload["outcomes"]["task_status_counts"]
    if len(payload["outcomes"]["open_task_ids"]) != (
        status_counts["open"] + status_counts["in_progress"]
    ) or sum(status_counts.values()) > MAX_REMEDIATION_TASKS:
        raise RemediationValidationError(
            "GovernanceRecord 未关闭任务与状态计数不一致。"
        )
    return payload
