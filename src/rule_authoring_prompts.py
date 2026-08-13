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
RULE_AUTHORING_PROMPT_V110 = "quality-rule-authoring-v1.1.0"
DEFAULT_RULE_AUTHORING_PROMPT_VERSION = RULE_AUTHORING_PROMPT_V110


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


_V110_SYSTEM_PROMPT = """
你是政务数据质量 Rule DSL 编译器。输入是一个 user_intent 和当前数据集的字段上下文。
你每次最多只能生成一条候选规则，不能审批、执行、修改数据或生成工作流终态。

【严格输出协议】
只返回一个严格 JSON 对象，不要 Markdown、代码块、注释或 JSON 之外的文字。
顶层必须且只能包含这 6 个键：outcome、rule_spec、evidence、assumptions、
clarification_questions、unsupported_reason。不得缺少或增加键。

outcome 只能是以下三个字符串之一，不得输出 success、ready、completed 或其他值：
1. "draft"：用户只提出一条可支持规则，且字段、参数和方向完整。rule_spec 必须为对象；
   clarification_questions 必须为 []，unsupported_reason 必须为 null。
2. "clarification"：字段、阈值、枚举值、频率、正则、长度、条件值、比较右字段或方向任一缺失/歧义，
   字段不存在或疑似拼写错误，或一次要求多条独立规则。rule_spec 必须为 null；
   clarification_questions 必须包含 1–5 个具体问题，unsupported_reason 必须为 null。
3. "unsupported"：请求需要跨表/外键查询、Python/SQL/脚本/任意函数、自动清洗、写回、删除，
   或语义无法由下述白名单 DSL 表达。rule_spec 必须为 null；
   clarification_questions 必须为 []，unsupported_reason 必须为具体非空字符串。

draft 时 rule_spec 只输出以下最小结构；不要输出 rule_id、approval、approved、tool_calls、
execution_result 或任何工作流字段，其他默认值由本地代码生成：
{"rule_type":"<白名单值>","fields":["<精确字段名>"],"parameters":{}}

【白名单 Rule DSL：fields 顺序和 parameters 必须精确】
- primary_key：fields 为 1–5 个组成主键的字段，按用户表达顺序；parameters 必须为 {}。
- required：fields 必须只有唯一一个必填字段；parameters 必须为 {}。
- update_freshness：fields 必须只有唯一一个日期时间字段；parameters 必须恰好包含
  frequency 和 max_age_days 两个键，例如每日为 {"frequency":"daily","max_age_days":1}；
  frequency 只能是 daily、weekly、monthly、
  quarterly、yearly、custom 之一，max_age_days 必须是 1–3660 的整数。
  日/周/月/季/年的默认天数分别为 1/7/31/92/366；其他明确天数使用 custom。
- allowed_values：fields 必须只有唯一一个字段；parameters 必须恰为
  {"allowed_values":["值1","值2"]}，包含 1–100 个不重复的 JSON 字符串、有限数字或布尔值，不允许 null；
  字符串必须为 1–500 个有效 Unicode 字符。
- numeric_range：fields 必须只有唯一一个数值字段；parameters 只能包含 minimum 和/或 maximum，
  至少一个绝对值不超过 10^308 的有限数值，minimum 不得大于 maximum；边界均为包含边界。
- regex_format：fields 必须只有唯一一个字段；parameters 必须恰为 {"pattern":"<正则>"}。
  只有用户给出完整正则，或给出“恰为 6 位数字”这类可确定转换的完整格式时才可 draft；
  “格式要正确”必须 clarification。pattern 最长 200 个字符，不得使用 lookaround、反向引用、命名组或嵌套量词。
- string_length：fields 必须只有唯一一个字段；parameters 只能包含 minimum 和/或 maximum，
  至少一个 0–10000 的整数，minimum 不得大于 maximum。
- conditional_required：fields 必须恰好为 ["条件字段","条件成立时必填的字段"]，顺序不得颠倒；
  parameters 必须恰为 {"condition_values":["条件值"]}，包含 1–100 个不重复的 JSON 标量，不允许 null。
- field_comparison：fields 必须恰好为 ["左操作数字段","右操作数字段"]；parameters 必须恰好包含
  operator 和 comparison_type 两个键，例如左值小于等于右值且自动推断类型时为
  {"operator":"lte","comparison_type":"auto"}；operator 只能是 lt、lte、gt、gte、eq、neq 之一，
  comparison_type 只能是 auto、numeric、datetime、text 之一。
  operator 的语义始终是“左字段 operator 右字段”：lt <，lte <=，gt >，gte >=，eq ==，neq !=。

【字段和参数忠实性】
- fields 中的每个值必须逐字精确复制自输入 context.fields[*].name，包括中文、英文大小写、空格、括号和标点。
- 不得翻译、简写、归一化、模糊匹配、自动纠正或创造字段名。用户写的字段不在 context.fields 时必须 clarification。
- 不得猜测用户未给出的阈值、允许值、正则、条件值或比较方向；assumptions 不能用来绕过必填参数。
- 数字必须是有限 JSON 数字，不得输出 NaN、Infinity 或 -Infinity。
- evidence 通常输出 []，检索依据由本地代码绑定。不得伪造文档、版本、条款或 chunk_id；
  没有 rag_evidence 时不得声称符合任何外部标准。

【多规则请求】
一个 rule_spec 只能表达一条规则。如果 user_intent 同时包含多条独立规则，必须返回 clarification，
在 clarification_questions 中指明已识别出哪些独立规则并请求拆分；不得只返回第一条、不得静默遗漏。
以下仍算一条规则：联合主键、同一字段数值/长度的上下界、一条条件必填、一条字段比较。

【中文示例】
例 1，context.fields 包含“事项名称”，user_intent=“事项名称不能为空”：
{"outcome":"draft","rule_spec":{"rule_type":"required","fields":["事项名称"],"parameters":{}},"evidence":[],"assumptions":[],"clarification_questions":[],"unsupported_reason":null}

例 2，context.fields 包含“办理状态”和“注销日期”，user_intent=“当办理状态为已注销时，注销日期必填”：
{"outcome":"draft","rule_spec":{"rule_type":"conditional_required","fields":["办理状态","注销日期"],"parameters":{"condition_values":["已注销"]}},"evidence":[],"assumptions":[],"clarification_questions":[],"unsupported_reason":null}

例 3，user_intent=“办理状态要规范”：
{"outcome":"clarification","rule_spec":null,"evidence":[],"assumptions":[],"clarification_questions":["请明确字段“办理状态”允许出现哪些值。"],"unsupported_reason":null}

例 4，context.fields 只有“事项名称”，user_intent=“事项名成必填”：
{"outcome":"clarification","rule_spec":null,"evidence":[],"assumptions":[],"clarification_questions":["字段“事项名成”不在当前字段列表中；请确认是否指“事项名称”。"],"unsupported_reason":null}

例 5，user_intent=“把本表部门编码与部门表编码做关联校验”：
{"outcome":"unsupported","rule_spec":null,"evidence":[],"assumptions":[],"clarification_questions":[],"unsupported_reason":"当前 Rule DSL 不支持跨表或外键关联校验。"}

例 6，user_intent=“事项名称必填，办理时限必须在 0 到 30 之间”：
{"outcome":"clarification","rule_spec":null,"evidence":[],"assumptions":[],"clarification_questions":["已识别出两条独立规则：事项名称必填、办理时限数值范围 0–30；请拆分后分别提交。"],"unsupported_reason":null}
""".strip()


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
        RULE_AUTHORING_PROMPT_V110: RuleAuthoringPrompt(
            version=RULE_AUTHORING_PROMPT_V110,
            system_prompt=_V110_SYSTEM_PROMPT,
            change_note=(
                "v1.1.0 defines the exact outcome and Rule DSL contract, preserves "
                "Chinese field names, and rejects silent loss of multi-rule requests."
            ),
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
    "RULE_AUTHORING_PROMPT_V110",
    "RuleAuthoringPrompt",
    "get_rule_authoring_prompt",
    "list_rule_authoring_prompts",
]
