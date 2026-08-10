"""Versioned prompt registry for the rule-authoring provider.

Prompt text is kept outside the HTTP adapter so harness runs can identify the
exact instructions used by a provider without importing networking code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Mapping


RULE_AUTHORING_PROMPT_V090 = "quality-rule-authoring-v0.9.0"
RULE_AUTHORING_PROMPT_V091 = "quality-rule-authoring-v0.9.1"
DEFAULT_RULE_AUTHORING_PROMPT_VERSION = RULE_AUTHORING_PROMPT_V091


_BASE_SYSTEM_PROMPT = (
    "你是政务数据质量规则编译器。只将用户依据编译为当前允许的 Rule DSL："
    "primary_key、required、update_freshness、allowed_values、numeric_range、"
    "regex_format、string_length、conditional_required、field_comparison。"
    "不得输出 Python、SQL、脚本、外键、跨表查询或任意函数。"
    "关键字段、阈值、频率、正则、条件值或比较运算符缺失时输出 clarification；"
    "跨表参照、自动清洗、数据写回和删除需求输出 unsupported。"
    "若输入包含 rag_evidence，只能引用其中真实存在的 chunk_id，"
    "不能创建文档、版本或条款；没有检索依据时不得声称符合标准。"
    "只返回指定 JSON，不要 Markdown。"
)


@dataclass(frozen=True)
class RuleAuthoringPrompt:
    """Immutable prompt specification recorded in every harness trace."""

    version: str
    system_prompt: str
    change_note: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, str]:
        return {
            "version": self.version,
            "sha256": self.sha256,
            "change_note": self.change_note,
        }


_PROMPTS: Mapping[str, RuleAuthoringPrompt] = MappingProxyType(
    {
        RULE_AUTHORING_PROMPT_V090: RuleAuthoringPrompt(
            version=RULE_AUTHORING_PROMPT_V090,
            system_prompt=_BASE_SYSTEM_PROMPT,
            change_note="v0.9 RAG citation boundary baseline.",
        ),
        RULE_AUTHORING_PROMPT_V091: RuleAuthoringPrompt(
            version=RULE_AUTHORING_PROMPT_V091,
            system_prompt=(
                _BASE_SYSTEM_PROMPT
                + "输出对象只能包含 outcome、rule_spec、evidence、assumptions、"
                "clarification_questions 和 unsupported_reason。"
                "不得输出 tool_calls、approval、approved、execution_result 或工作流终态。"
            ),
            change_note="v0.9.1 rejects tool, approval, and execution fields in model output.",
        ),
    }
)


def get_rule_authoring_prompt(
    version: str = DEFAULT_RULE_AUTHORING_PROMPT_VERSION,
) -> RuleAuthoringPrompt:
    try:
        return _PROMPTS[str(version)]
    except KeyError as error:
        raise ValueError(f"未知规则编制提示词版本：{version}。") from error


def list_rule_authoring_prompts() -> tuple[dict[str, str], ...]:
    return tuple(prompt.to_dict() for prompt in _PROMPTS.values())


__all__ = [
    "DEFAULT_RULE_AUTHORING_PROMPT_VERSION",
    "RULE_AUTHORING_PROMPT_V090",
    "RULE_AUTHORING_PROMPT_V091",
    "RuleAuthoringPrompt",
    "get_rule_authoring_prompt",
    "list_rule_authoring_prompts",
]
