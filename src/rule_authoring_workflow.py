"""v0.9 规则编制状态机。

状态由本地代码控制，Provider 只能提供候选 RuleDraft，不能指定审批或执行
状态。这个轻量状态机先独立于 LangGraph，后续可作为图节点的领域内核。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping

from .rule_dsl import RuleDraft, RuleDraftValidationResult


WorkflowState = Literal[
    "collecting",
    "retrieving",
    "needs_clarification",
    "compiling",
    "draft",
    "validated",
    "dry_run_complete",
    "awaiting_approval",
    "approved",
    "executed",
    "rejected",
    "failed",
]


class RuleAuthoringWorkflowError(ValueError):
    """非法状态转换。"""


_ALLOWED_TRANSITIONS: Mapping[str, frozenset[str]] = {
    "collecting": frozenset({"retrieving", "compiling", "needs_clarification", "rejected"}),
    "retrieving": frozenset({"compiling", "needs_clarification", "rejected"}),
    "needs_clarification": frozenset({"collecting", "rejected"}),
    "compiling": frozenset({"draft", "needs_clarification", "failed"}),
    "draft": frozenset({"validated", "needs_clarification", "rejected", "failed"}),
    "validated": frozenset({"dry_run_complete", "failed"}),
    "dry_run_complete": frozenset({"awaiting_approval", "needs_clarification", "failed"}),
    "awaiting_approval": frozenset({"approved", "rejected"}),
    "approved": frozenset({"executed", "failed"}),
    "executed": frozenset(),
    "rejected": frozenset(),
    "failed": frozenset({"collecting", "compiling"}),
}


@dataclass(frozen=True)
class RuleAuthoringWorkflow:
    workflow_id: str
    target_metric_id: str | None
    state: WorkflowState = "collecting"
    draft: RuleDraft | None = None
    validation: RuleDraftValidationResult | None = None
    dry_run: Mapping[str, Any] | None = None
    error: str | None = None

    def _transition(self, state: WorkflowState, **updates: Any) -> "RuleAuthoringWorkflow":
        allowed = _ALLOWED_TRANSITIONS.get(self.state, frozenset())
        if state not in allowed:
            raise RuleAuthoringWorkflowError(
                f"工作流不能从 {self.state} 转换到 {state}。"
            )
        return replace(self, state=state, **updates)

    def start_retrieving(self) -> "RuleAuthoringWorkflow":
        """进入标准依据检索阶段；检索本身仍由只读本地工具完成。"""

        return self._transition("retrieving", error=None)

    def start_compiling(self) -> "RuleAuthoringWorkflow":
        return self._transition("compiling", error=None)

    def accept_draft(self, draft: RuleDraft) -> "RuleAuthoringWorkflow":
        if self.state != "compiling":
            raise RuleAuthoringWorkflowError("只有 compiling 状态可以接收 Provider 草案。")
        if draft.status == "needs_clarification":
            return self._transition(
                "needs_clarification",
                draft=draft,
                error=None,
            )
        if draft.status == "rejected":
            return self._transition(
                "rejected",
                draft=draft,
                error=draft.unsupported_reason,
            )
        return self._transition("draft", draft=draft, error=None)

    def mark_validated(
        self,
        validation: RuleDraftValidationResult,
    ) -> "RuleAuthoringWorkflow":
        if self.state != "draft":
            raise RuleAuthoringWorkflowError("只有 draft 状态可以执行规则校验。")
        if validation.valid:
            return self._transition("validated", validation=validation, error=None)
        return replace(self, validation=validation, error="；".join(validation.errors))

    def mark_dry_run_complete(self, preview: Mapping[str, Any]) -> "RuleAuthoringWorkflow":
        if self.state != "validated":
            raise RuleAuthoringWorkflowError("只有 validated 状态可以完成试运行。")
        return self._transition(
            "dry_run_complete",
            dry_run=dict(preview),
            error=None,
        )

    def await_approval(self) -> "RuleAuthoringWorkflow":
        return self._transition("awaiting_approval")

    def approve(self) -> "RuleAuthoringWorkflow":
        return self._transition("approved")

    def execute(self) -> "RuleAuthoringWorkflow":
        return self._transition("executed")

    def reject(self, reason: str = "用户拒绝当前规则草案。") -> "RuleAuthoringWorkflow":
        if "rejected" not in _ALLOWED_TRANSITIONS.get(self.state, frozenset()):
            raise RuleAuthoringWorkflowError(
                f"工作流不能从 {self.state} 直接拒绝。"
            )
        return replace(self, state="rejected", error=str(reason))

    def retry(self) -> "RuleAuthoringWorkflow":
        if self.state != "failed":
            raise RuleAuthoringWorkflowError("只有 failed 状态可以重试。")
        return replace(
            self,
            state="collecting",
            validation=None,
            dry_run=None,
            error=None,
        )


__all__ = [
    "RuleAuthoringWorkflow",
    "RuleAuthoringWorkflowError",
    "WorkflowState",
]
