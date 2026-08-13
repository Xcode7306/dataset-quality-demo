"""v0.9 规则编制 Provider：本地模板 + 可选 Chat Completions 适配。

Provider 只负责把用户自然语言解析成候选结构，不拥有审批或执行权限。
当外部模型不可用时，模板 Provider 保证本地功能仍可使用。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
import re
from typing import Any, Callable, Mapping, Protocol

from .rule_dsl import (
    ProviderMetadata,
    RuleEvidence,
    RuleSpec,
    make_rule_id,
    new_evidence,
    rule_spec_from_dict,
)
from .model_api import (
    extract_message_content,
    normalize_chat_completions_url,
    response_error_detail,
)
from .rule_authoring_prompts import (
    DEFAULT_RULE_AUTHORING_PROMPT_VERSION,
    get_rule_authoring_prompt,
)


RULE_AUTHORING_PROMPT_VERSION = DEFAULT_RULE_AUTHORING_PROMPT_VERSION
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 20.0


class RuleAuthoringProviderUnavailable(RuntimeError):
    """规则编制 Provider 未配置或不可用。"""


class RuleAuthoringProviderError(RuntimeError):
    """规则编制 Provider 调用或返回结果失败。"""


@dataclass(frozen=True)
class RuleAuthoringProviderResult:
    outcome: str
    rule_spec: RuleSpec | None
    evidence: tuple[RuleEvidence, ...] = ()
    assumptions: tuple[str, ...] = ()
    clarification_questions: tuple[str, ...] = ()
    unsupported_reason: str | None = None
    metadata: ProviderMetadata = ProviderMetadata(
        provider="template",
        model=None,
        mode="template",
        prompt_version=RULE_AUTHORING_PROMPT_VERSION,
    )


@dataclass(frozen=True)
class RuleIntentInspection:
    """Deterministic completeness guard that a model result cannot bypass."""

    recognized_rule_type: str | None
    clarification_questions: tuple[str, ...] = ()
    unsupported_reason: str | None = None

    @property
    def complete(self) -> bool:
        return not self.clarification_questions and self.unsupported_reason is None


class RuleAuthoringProvider(Protocol):
    def generate(
        self,
        context: Mapping[str, Any],
        *,
        user_intent: str,
    ) -> RuleAuthoringProviderResult:
        ...


def _field_items(context: Mapping[str, Any]) -> list[dict[str, str]]:
    raw_fields = context.get("fields", [])
    if not isinstance(raw_fields, list):
        return []
    return [
        {
            "name": str(item.get("name")),
            "inferred_type": str(item.get("inferred_type", "unknown")),
        }
        for item in raw_fields
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    ]


def _find_fields(text: str, fields: list[dict[str, str]]) -> list[str]:
    lowered = text.casefold()
    return [
        item["name"]
        for item in sorted(fields, key=lambda item: len(item["name"]), reverse=True)
        if item["name"].casefold() in lowered
    ]


def _first_field(text: str, fields: list[dict[str, str]]) -> str | None:
    matches = _find_fields(text, fields)
    return matches[0] if matches else None


def _metadata(
    *,
    provider: str = "template",
    model: str | None = None,
    mode: str = "template",
    fallback_used: bool = False,
    fallback_reason: str | None = None,
    request_id: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    prompt_version: str = RULE_AUTHORING_PROMPT_VERSION,
) -> ProviderMetadata:
    return ProviderMetadata(
        provider=provider,
        model=model,
        mode="model" if mode == "model" else "template",
        prompt_version=prompt_version,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        request_id=request_id,
        input_tokens=max(0, int(input_tokens)),
        output_tokens=max(0, int(output_tokens)),
    )


def _clarification(
    questions: list[str],
    *,
    assumptions: list[str] | None = None,
) -> RuleAuthoringProviderResult:
    return RuleAuthoringProviderResult(
        outcome="clarification",
        rule_spec=None,
        assumptions=tuple(assumptions or ()),
        clarification_questions=tuple(dict.fromkeys(questions[:5])),
        metadata=_metadata(),
    )


def _unsupported(reason: str) -> RuleAuthoringProviderResult:
    return RuleAuthoringProviderResult(
        outcome="unsupported",
        rule_spec=None,
        unsupported_reason=reason,
        metadata=_metadata(),
    )


def _candidate(
    *,
    rule_type: str,
    field_names: list[str],
    name: str,
    description: str,
    parameters: Mapping[str, Any] | None = None,
) -> RuleAuthoringProviderResult:
    parameters = dict(parameters or {})
    return RuleAuthoringProviderResult(
        outcome="draft",
        rule_spec=RuleSpec(
            rule_type=rule_type,
            rule_id=make_rule_id(rule_type, field_names, parameters),
            name=name,
            description=description,
            fields=tuple(field_names),
            parameters=parameters,
            evidence_ids=(),
        ),
        metadata=_metadata(),
    )


class TemplateRuleAuthoringProvider:
    """无需网络的规则编制模板，覆盖 v0.8 白名单 Rule DSL。"""

    cache_namespace = f"template:{RULE_AUTHORING_PROMPT_VERSION}"

    def __init__(self, *, prompt_version: str = RULE_AUTHORING_PROMPT_VERSION) -> None:
        self.prompt_version = get_rule_authoring_prompt(prompt_version).version
        self.cache_namespace = f"template:{self.prompt_version}"

    def generate(
        self,
        context: Mapping[str, Any],
        *,
        user_intent: str,
    ) -> RuleAuthoringProviderResult:
        result = self._generate(context, user_intent=user_intent)
        return replace(
            result,
            metadata=replace(
                result.metadata,
                prompt_version=self.prompt_version,
            ),
        )

    def _generate(
        self,
        context: Mapping[str, Any],
        *,
        user_intent: str,
    ) -> RuleAuthoringProviderResult:
        text = str(user_intent or "").strip()
        if not text:
            return _clarification(["请先说明需要评价哪个字段，以及应满足什么条件。"])
        if len(text) > 4000:
            return _clarification(["评价依据不能超过 4000 个字符，请压缩为规则条件。"])

        if re.search(
            r"\bpython\b|\bjavascript\b|\bshell\b|\bsql\b|脚本|代码执行|运行代码|"
            r"任意代码|eval\s*\(|exec\s*\(|动态执行|调用函数",
            text,
            flags=re.IGNORECASE,
        ):
            return _unsupported(
                "v0.8 只支持白名单数据质量规则，不执行 Python、SQL、脚本或任意函数。"
            )

        fields = _field_items(context)
        date_fields = [
            item["name"]
            for item in fields
            if item["inferred_type"] == "datetime"
        ]
        numeric_fields = [
            item["name"]
            for item in fields
            if item["inferred_type"] == "numeric"
        ]

        if re.search(r"存在于|参照表|权威表|外键|跨表|自动修复|自动清洗|写回|删除", text):
            return _unsupported(
                "v0.8 暂不支持跨表参照、外键、自动清洗、数据写回或删除。"
            )

        if re.search(r"条件必填|条件下.*必填|时.*必须填写|时.*必填", text):
            matched_fields = sorted(
                self._find_fields_in_text(text, fields),
                key=lambda item: text.casefold().find(item.casefold()),
            )
            if len(matched_fields) < 2:
                return _clarification(["请同时指出触发条件字段和条件满足时必须填写的字段。"])
            condition_field, required_field = matched_fields[:2]
            condition_values = self._condition_values(
                text,
                condition_field,
                required_field,
            )
            if not condition_values:
                return _clarification([f"请说明字段“{condition_field}”触发条件的完整取值。"])
            return _candidate(
                rule_type="conditional_required",
                field_names=[condition_field, required_field],
                name=f"{condition_field}条件下{required_field}必填",
                description=(
                    f"当字段“{condition_field}”取指定值时，"
                    f"字段“{required_field}”必须填写。"
                ),
                parameters={"condition_values": condition_values},
            )

        if re.search(r"不得晚于|不晚于|不能晚于|不得早于|不早于|不能早于|"
                     r"小于等于|大于等于|不大于|不小于|不超过|不低于|≤|≥", text):
            matched_fields = sorted(
                self._find_fields_in_text(text, fields),
                key=lambda item: text.casefold().find(item.casefold()),
            )
            if len(matched_fields) < 2:
                return _clarification(["请指出需要比较的左侧字段和右侧字段。"])
            operator = self._comparison_operator(text)
            if operator is None:
                return _clarification(["请说明两个字段之间是小于、等于还是大于关系。"])
            comparison_type = "auto"
            first_type = next(
                (item["inferred_type"] for item in fields if item["name"] == matched_fields[0]),
                "unknown",
            )
            second_type = next(
                (item["inferred_type"] for item in fields if item["name"] == matched_fields[1]),
                "unknown",
            )
            if first_type == second_type and first_type in {"numeric", "datetime", "text"}:
                comparison_type = first_type
            return _candidate(
                rule_type="field_comparison",
                field_names=matched_fields[:2],
                name=f"{matched_fields[0]}与{matched_fields[1]}跨字段比较",
                description=(
                    f"字段“{matched_fields[0]}”与“{matched_fields[1]}”"
                    "应满足指定的顺序关系。"
                ),
                parameters={
                    "operator": operator,
                    "comparison_type": comparison_type,
                },
            )

        if re.search(r"正则|格式|位数字|位字符|数字代码|匹配", text):
            field = _first_field(text, fields)
            if field is None:
                return _clarification(["请指出需要校验格式的字段名称。"])
            pattern = self._regex_pattern(text)
            if pattern is None:
                return _clarification(["请提供完整正则表达式，或明确说明固定长度和字符类型。"])
            return _candidate(
                rule_type="regex_format",
                field_names=[field],
                name=f"{field}格式",
                description=f"字段“{field}”必须满足指定格式。",
                parameters={"pattern": pattern},
            )

        if re.search(r"长度|字符数|位数|几个字符", text):
            field = _first_field(text, fields)
            if field is None:
                return _clarification(["请指出需要限制字符长度的字段名称。"])
            length_parameters = self._length_parameters(text)
            if not length_parameters:
                return _clarification(["请提供字符长度的最小值、最大值或固定长度。"])
            return _candidate(
                rule_type="string_length",
                field_names=[field],
                name=f"{field}字符长度",
                description=f"字段“{field}”的字符长度必须满足指定范围。",
                parameters=length_parameters,
            )

        if re.search(r"主键|唯一标识|唯一编码|唯一编号", text):
            primary_fields = _find_fields(text, fields)
            if not primary_fields:
                return _clarification(["请指出组成主键或唯一标识的字段名称。"])
            return _candidate(
                rule_type="primary_key",
                field_names=primary_fields[:5],
                name="主键完整性与唯一性",
                description="指定字段共同组成主键，每条记录应完整且唯一。",
            )

        if re.search(r"必填|必须填写|不得为空|不能为空|不可为空|必需有值", text):
            field = _first_field(text, fields)
            if field is None:
                return _clarification(["请指出需要设为必填的字段名称。"])
            return _candidate(
                rule_type="required",
                field_names=[field],
                name=f"{field}必填",
                description=f"字段“{field}”不得为空。",
            )

        if re.search(r"只能|允许值|枚举|取值范围|状态值", text):
            field = _first_field(text, fields)
            if field is None:
                return _clarification(["请指出允许值约束对应的字段名称。"])
            values_match = re.search(
                r"(?:只能(?:为|是)|允许值(?:为|是|包括)?|取值(?:为|是|包括)?|枚举(?:为|是|包括)?)\s*(.+)$",
                text,
            )
            values_text = values_match.group(1) if values_match else ""
            values_text = re.split(r"[。；;！!]", values_text)[0]
            values_text = re.sub(r"[等等等]+$", "", values_text).strip()
            values = [
                item.strip().strip("\"'“”‘’")
                for item in re.split(r"[、,，/／|或]+", values_text)
                if item.strip().strip("\"'“”‘’")
            ]
            values = [item for item in values if item not in {"和", "以及"}]
            if not values or len(values) > 100:
                return _clarification([f"请列出字段“{field}”允许的完整取值列表。"])
            return _candidate(
                rule_type="allowed_values",
                field_names=[field],
                name=f"{field}允许值",
                description=f"字段“{field}”只能取列出的允许值。",
                parameters={"allowed_values": values},
            )

        if re.search(r"更新时间|更新日期|及时更新|时效|新鲜度|及时", text):
            field = _first_field(text, fields)
            if field is None and len(date_fields) == 1:
                field = date_fields[0]
            if field is None:
                return _clarification(["请指出更新时间字段名称。"])
            period_days: int | None = None
            frequency: str | None = None
            for pattern, candidate_frequency, candidate_days in (
                (r"每日|每天|日更新", "daily", 1),
                (r"每周|每星期|周更新", "weekly", 7),
                (r"每月|月更新", "monthly", 31),
                (r"每季度|季度更新", "quarterly", 92),
                (r"每年|年度更新|年更新", "yearly", 366),
            ):
                if re.search(pattern, text):
                    frequency, period_days = candidate_frequency, candidate_days
                    break
            custom_match = re.search(r"(?:不超过|最长|间隔|滞后)\s*(\d+)\s*天", text)
            if custom_match:
                frequency, period_days = "custom", int(custom_match.group(1))
            if frequency is None or period_days is None:
                return _clarification(["请说明更新频率或允许的最长间隔天数。"])
            return _candidate(
                rule_type="update_freshness",
                field_names=[field],
                name=f"{field}更新及时性",
                description=f"字段“{field}”相对评估基准日期应在允许间隔内更新。",
                parameters={
                    "frequency": frequency,
                    "max_age_days": period_days,
                },
            )

        if re.search(r"范围|区间|不低于|不高于|至少|至多|大于|小于|不得低于|不得超过|≥|≤", text):
            field = _first_field(text, fields)
            if field is None:
                return _clarification(["请指出需要进行数值范围约束的字段名称。"])
            range_match = re.search(
                r"(?:在|介于)?\s*(-?\d+(?:\.\d+)?)\s*(?:到|至|~|～|-)\s*(-?\d+(?:\.\d+)?)",
                text,
            )
            parameters: dict[str, Any] = {}
            if range_match:
                parameters = {
                    "minimum": float(range_match.group(1))
                    if "." in range_match.group(1)
                    else int(range_match.group(1)),
                    "maximum": float(range_match.group(2))
                    if "." in range_match.group(2)
                    else int(range_match.group(2)),
                }
            else:
                minimum_match = re.search(
                    r"(?:不低于|至少|大于等于|不得低于|≥)\s*(-?\d+(?:\.\d+)?)",
                    text,
                )
                maximum_match = re.search(
                    r"(?:不高于|至多|小于等于|不得超过|≤)\s*(-?\d+(?:\.\d+)?)",
                    text,
                )
                if minimum_match:
                    raw = minimum_match.group(1)
                    parameters["minimum"] = float(raw) if "." in raw else int(raw)
                if maximum_match:
                    raw = maximum_match.group(1)
                    parameters["maximum"] = float(raw) if "." in raw else int(raw)
            if not parameters:
                return _clarification(["请提供数值下限、上限，或完整的闭区间范围。"])
            return _candidate(
                rule_type="numeric_range",
                field_names=[field],
                name=f"{field}数值范围",
                description=f"字段“{field}”应处于给定的闭区间范围内。",
                parameters=parameters,
            )

        return _clarification(
            [
                "当前依据还不能映射为白名单规则；请补充字段名称和必填、允许值、更新时间、主键、数值范围、格式、长度、条件必填或跨字段比较条件。"
            ]
        )

    @staticmethod
    def _find_fields_in_text(text: str, fields: list[dict[str, str]]) -> list[str]:
        return [
            item["name"]
            for item in fields
            if item["name"].casefold() in text.casefold()
        ]

    @staticmethod
    def _condition_values(
        text: str,
        condition_field: str,
        required_field: str,
    ) -> list[str]:
        start = text.casefold().find(condition_field.casefold())
        end = text.casefold().find(required_field.casefold(), start + len(condition_field))
        fragment = text[start + len(condition_field): end if end >= 0 else None]
        match = re.search(r"(?:为|是|等于|=)\s*([^，,；;。]+?)(?:时|则|，|,|$)", fragment)
        if not match:
            return []
        values_text = match.group(1).strip().strip("：:")
        return [
            item.strip().strip("\"'“”‘’")
            for item in re.split(r"[、,，/／|或]+", values_text)
            if item.strip().strip("\"'“”‘’")
        ][:100]

    @staticmethod
    def _comparison_operator(text: str) -> str | None:
        if re.search(r"不得晚于|不晚于|不能晚于|小于等于|不大于|不超过|≤", text):
            return "lte"
        if re.search(r"不得早于|不早于|不能早于|大于等于|不小于|不低于|≥", text):
            return "gte"
        if re.search(r"严格小于|应小于", text):
            return "lt"
        if re.search(r"严格大于|应大于", text):
            return "gt"
        if re.search(r"等于|相等", text):
            return "eq"
        return None

    @staticmethod
    def _regex_pattern(text: str) -> str | None:
        anchored = re.search(r"(\^.*?\$)", text)
        if anchored:
            return anchored.group(1).strip("\"'“”‘’")
        explicit = re.search(
            r"(?:正则(?:表达式)?|匹配)\s*[:：]?\s*([^\s。；;]+)",
            text,
        )
        if explicit:
            candidate = explicit.group(1).strip("\"'“”‘’")
            if candidate not in {"正则", "表达式", "格式"}:
                return candidate
        fixed = re.search(r"(\d+|[一二三四五六七八九十]+)\s*位\s*(?:数字|字符)", text)
        if fixed:
            number = fixed.group(1)
            mapping = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
            length = int(number) if number.isdigit() else mapping.get(number)
            if length is not None:
                token = r"\d" if "数字" in fixed.group(0) else r"."
                return rf"^{token}{{{length}}}$"
        return None

    @staticmethod
    def _length_parameters(text: str) -> dict[str, int]:
        fixed = re.search(r"长度\s*(?:为|是|等于)\s*(\d+)\s*(?:位|个字符|字符)?", text)
        if fixed:
            value = int(fixed.group(1))
            return {"minimum": value, "maximum": value}
        minimum = re.search(r"(?:至少|不少于|不低于)\s*(\d+)\s*(?:位|个字符|字符)?", text)
        maximum = re.search(r"(?:至多|不超过|不多于|不高于)\s*(\d+)\s*(?:位|个字符|字符)?", text)
        result: dict[str, int] = {}
        if minimum:
            result["minimum"] = int(minimum.group(1))
        if maximum:
            result["maximum"] = int(maximum.group(1))
        return result


def inspect_rule_intent(
    context: Mapping[str, Any],
    *,
    user_intent: str,
) -> RuleIntentInspection:
    """Identify missing critical inputs before a draft may enter validation.

    This guard intentionally checks what the user explicitly stated.  A model
    may add clearer questions, but it cannot silently invent fields, thresholds,
    trigger values, frequencies, regexes, or comparison operators.
    """

    text = str(user_intent or "").strip()
    if not text:
        return RuleIntentInspection(
            None,
            ("请说明需要评价哪个字段，以及该字段必须满足什么条件。",),
        )
    if len(text) > 4000:
        return RuleIntentInspection(
            None,
            ("规则描述不能超过 4000 个字符，请压缩为一条明确规则。",),
        )
    if re.search(
        r"\bpython\b|\bjavascript\b|\bshell\b|\bsql\b|脚本|代码执行|运行代码|"
        r"任意代码|eval\s*\(|exec\s*\(|动态执行|调用函数",
        text,
        flags=re.IGNORECASE,
    ):
        return RuleIntentInspection(
            None,
            unsupported_reason=(
                "当前只支持白名单数据质量规则，不执行 Python、SQL、脚本或任意函数。"
            ),
        )
    if re.search(r"存在于|参照表|权威表|外键|跨表|自动修复|自动清洗|写回|删除", text):
        return RuleIntentInspection(
            None,
            unsupported_reason=(
                "当前暂不支持跨表参照、外键、自动清洗、数据写回或删除。"
            ),
        )

    fields = _field_items(context)
    matched_fields = _find_fields(text, fields)
    questions: list[str] = []

    rule_signal = re.compile(
        r"主键|唯一|必填|必须填写|不得为空|允许值|枚举|只能|数值范围|范围|区间|不低于|不超过|不高于|正则|格式|位数字|位字符|数字代码|长度|字符|更新|时效|新鲜度|不得晚于|小于等于|大于等于|"
        r"条件必填|字段比较|跨字段",
    )

    def independent_rule_segments(separator_pattern: str) -> list[str]:
        return [
            segment.strip(" \t\r\n，,。；;，")
            for segment in re.split(separator_pattern, text)
            if segment.strip(" \t\r\n，,。；;")
            and rule_signal.search(segment)
        ]

    strong_segments = independent_rule_segments(r"[；;。\n]+")
    comma_segments = independent_rule_segments(r"[，,]+")
    conjunction_segments = independent_rule_segments(r"(?:并且|同时|以及|且)")
    multi_segments = strong_segments
    if len(multi_segments) < 2:
        distinct_comma_fields = {
            field
            for segment in comma_segments
            for field in _find_fields(segment, fields)
        }
        if len(comma_segments) >= 2 and len(distinct_comma_fields) >= 2:
            multi_segments = comma_segments
    if len(multi_segments) < 2:
        distinct_conjunction_fields = {
            field
            for segment in conjunction_segments
            for field in _find_fields(segment, fields)
        }
        if len(conjunction_segments) >= 2 and len(distinct_conjunction_fields) >= 2:
            multi_segments = conjunction_segments
    if len(multi_segments) >= 2:
        fields_in_request = tuple(dict.fromkeys(
            field
            for segment in multi_segments[:5]
            for field in _find_fields(segment, fields)
        ))
        field_hint = "、".join(fields_in_request)
        detail = f"（涉及字段：{field_hint}）" if field_hint else ""
        return RuleIntentInspection(
            None,
            (
                f"已识别出多条独立规则{detail}，当前一次只能编制一条；请拆分后分别提交。",
            ),
        )

    def require_fields(count: int, message: str) -> None:
        if len(matched_fields) < count:
            questions.append(message)

    if re.search(r"条件必填|条件下.*必填|时.*必须填写|时.*必填", text):
        require_fields(2, "请同时写明触发条件字段和条件成立后必须填写的字段。")
        if not re.search(r"(?:为|是|等于|=)\s*[^，,；;。]+?(?:时|则|，|,)", text):
            questions.append("请写明触发条件字段的完整取值，例如“状态为注销时”。")
        return RuleIntentInspection(
            "conditional_required", tuple(dict.fromkeys(questions))[:5]
        )

    if re.search(
        r"不得晚于|不晚于|不能晚于|不得早于|不早于|不能早于|"
        r"小于等于|大于等于|不大于|不小于|严格小于|严格大于|"
        r"应小于|应大于|等于|相等|跨字段比较|字段比较|≤|≥",
        text,
    ):
        require_fields(2, "请依次写明需要比较的左侧字段和右侧字段。")
        if TemplateRuleAuthoringProvider._comparison_operator(text) is None:
            questions.append("请明确两个字段之间应满足小于、等于还是大于关系。")
        return RuleIntentInspection(
            "field_comparison", tuple(dict.fromkeys(questions))[:5]
        )

    if re.search(r"正则|格式|位数字|位字符|数字代码|匹配", text):
        require_fields(1, "请写明需要校验格式的字段名称。")
        if TemplateRuleAuthoringProvider._regex_pattern(text) is None:
            questions.append("请提供完整正则表达式，或明确固定长度和字符类型。")
        return RuleIntentInspection(
            "regex_format", tuple(dict.fromkeys(questions))[:5]
        )

    if re.search(r"长度|字符数|位数|几个字符", text):
        require_fields(1, "请写明需要限制字符长度的字段名称。")
        if not TemplateRuleAuthoringProvider._length_parameters(text):
            questions.append("请提供字符长度的最小值、最大值或固定长度。")
        return RuleIntentInspection(
            "string_length", tuple(dict.fromkeys(questions))[:5]
        )

    if re.search(r"主键|唯一标识|唯一编码|唯一编号", text):
        require_fields(1, "请写明组成主键或唯一标识的全部字段名称。")
        return RuleIntentInspection(
            "primary_key", tuple(dict.fromkeys(questions))[:5]
        )

    if re.search(r"只能|允许值|枚举|取值范围|状态值", text):
        require_fields(1, "请写明允许值约束对应的字段名称。")
        values_match = re.search(
            r"(?:只能(?:为|是)|允许值(?:为|是|包括)?|取值(?:为|是|包括)?|"
            r"枚举(?:为|是|包括)?)\s*(.+)$",
            text,
        )
        values_text = values_match.group(1).strip() if values_match else ""
        values_text = re.split(r"[。；;！!]", values_text)[0].strip()
        if not values_text:
            questions.append("请列出字段允许的完整取值列表。")
        return RuleIntentInspection(
            "allowed_values", tuple(dict.fromkeys(questions))[:5]
        )

    if re.search(r"更新时间|更新日期|及时更新|时效|新鲜度|及时", text):
        require_fields(1, "请明确写出更新时间字段名称。")
        if not re.search(
            r"每日|每天|日更新|每周|每星期|周更新|每月|月更新|每季度|季度更新|"
            r"每年|年度更新|年更新|(?:不超过|最长|间隔|滞后)\s*\d+\s*天",
            text,
        ):
            questions.append("请说明更新频率或允许的最长间隔天数。")
        return RuleIntentInspection(
            "update_freshness", tuple(dict.fromkeys(questions))[:5]
        )

    if re.search(
        r"范围|区间|不低于|不高于|至少|至多|大于|小于|不得低于|不得超过|≥|≤",
        text,
    ):
        require_fields(1, "请写明需要进行数值范围约束的字段名称。")
        if not re.search(r"-?\d+(?:\.\d+)?", text):
            questions.append("请提供数值下限、上限，或完整的闭区间范围。")
        return RuleIntentInspection(
            "numeric_range", tuple(dict.fromkeys(questions))[:5]
        )

    if re.search(r"必填|必须填写|不得为空|不能为空|不可为空|必需有值", text):
        require_fields(1, "请写明需要设为必填的字段名称。")
        return RuleIntentInspection(
            "required", tuple(dict.fromkeys(questions))[:5]
        )

    metric = context.get("metric")
    required_inputs = (
        metric.get("required_inputs", []) if isinstance(metric, Mapping) else []
    )
    requirement = "、".join(str(item) for item in required_inputs if str(item).strip())
    suffix = f"；该指标还需要：{requirement}" if requirement else ""
    return RuleIntentInspection(
        None,
        (
            "请明确规则类型、字段名称和完整条件；当前支持必填、唯一、允许值、"
            f"更新时间、数值范围、格式、长度、条件必填或跨字段比较{suffix}。",
        ),
    )


def build_rule_input_guidance(
    context: Mapping[str, Any],
    *,
    user_intent: str,
) -> tuple[str, ...]:
    """根据用户输入给出可执行规则所缺少的具体信息。"""

    text = str(user_intent or "").strip()
    lowered = text.casefold()
    fields = _field_items(context)
    questions: list[str] = []
    matched_fields = _find_fields(text, fields)

    if re.search(r"主键|唯一标识|唯一编码|唯一编号", text):
        if not matched_fields:
            questions.append("缺少组成主键或唯一标识的字段名称。")
    if re.search(r"必填|必须填写|不得为空|不能为空|不可为空|必需有值", text):
        if not matched_fields:
            questions.append("缺少需要设为必填的字段名称。")
    if re.search(r"只能|允许值|枚举|取值范围|状态值", text):
        if not matched_fields:
            questions.append("缺少允许值约束对应的字段名称。")
        if not re.search(r"(?:只能|允许值|取值|枚举).{0,12}(?:为|是|包括)", text):
            questions.append("缺少字段允许的完整取值列表。")
    if re.search(r"更新时间|更新日期|及时更新|时效|新鲜度|及时", text):
        if not matched_fields and len(
            [item for item in fields if item["inferred_type"] == "datetime"]
        ) != 1:
            questions.append("缺少更新时间字段名称。")
        if not re.search(r"每日|每天|每周|每星期|每月|每季度|每年|不超过\s*\d+\s*天|最长\s*\d+\s*天", text):
            questions.append("缺少更新频率或允许的最长间隔天数。")
    if re.search(r"范围|区间|不低于|不高于|至少|至多|大于|小于|不得低于|不得超过|≥|≤", text):
        if not matched_fields:
            questions.append("缺少需要进行数值范围约束的字段名称。")
        if not re.search(r"-?\d+(?:\.\d+)?", text):
            questions.append("缺少数值下限、上限或完整的闭区间范围。")
    if re.search(r"跨表|外键|参照|自动清洗|自动修复|写回|删除", lowered):
        questions.append(
            "v0.8 暂不支持跨表参照、外键、自动清洗、数据写回或删除。"
        )
    if re.search(r"类型|格式|长度|正则|条件|比较|跨字段|数据模型|元数据|权威参考|业务规则", lowered):
        questions.append(
            "v0.8 支持格式/正则、字符长度、条件必填和跨字段比较；"
            "请补充字段名称以及完整正则、长度边界、触发条件或比较运算符。"
        )
    if not questions:
        questions.append(
            "缺少可执行规则类型；请补充字段名称，以及必填、唯一、允许值、更新时间、数值范围、格式、长度、条件必填或跨字段比较条件。"
        )
    return tuple(dict.fromkeys(questions))[:5]


def _move_provider_alias(
    payload: dict[str, Any],
    canonical: str,
    aliases: tuple[str, ...],
) -> None:
    """归一化已知的模型字段别名，不解决有冲突的输出。"""

    present_aliases = [alias for alias in aliases if alias in payload]
    if not present_aliases:
        return
    if canonical in payload or len(present_aliases) > 1:
        names = [canonical] if canonical in payload else []
        names.extend(present_aliases)
        raise RuleAuthoringProviderError(
            f"模型 JSON 同时包含语义重复字段：{names}。"
        )
    payload[canonical] = payload.pop(present_aliases[0])


def _normalize_provider_outcome(payload: dict[str, Any]) -> None:
    """将常见结果标签保守地映射到 Demo 的三态协议。"""

    outcome = payload.get("outcome")
    if not isinstance(outcome, str):
        raise RuleAuthoringProviderError("模型 outcome 必须是字符串。")
    normalized = re.sub(r"[\s\-]+", "_", outcome.strip().casefold())
    aliases = {
        "draft": {
            "draft",
            "success",
            "succeeded",
            "ready",
            "valid",
            "supported",
            "candidate",
            "complete",
            "completed",
            "ok",
            "草案",
            "成功",
            "已生成",
        },
        "clarification": {
            "clarification",
            "clarify",
            "need_clarification",
            "needs_clarification",
            "ask_user",
            "澄清",
            "需要澄清",
            "需补充信息",
        },
        "unsupported": {
            "unsupported",
            "not_supported",
            "out_of_scope",
            "unsupported_request",
            "不支持",
            "超出范围",
        },
    }
    canonical = next(
        (name for name, values in aliases.items() if normalized in values),
        None,
    )
    if canonical is None:
        raise RuleAuthoringProviderError("模型 outcome 不在允许范围内。")

    rule_spec = payload.get("rule_spec")
    if canonical == "draft" and not isinstance(rule_spec, Mapping):
        raise RuleAuthoringProviderError(
            "模型将结果标记为可用草案，但未返回 rule_spec 对象。"
        )
    if canonical != "draft" and rule_spec is not None:
        raise RuleAuthoringProviderError(
            "模型非草案结果不得携带 rule_spec。"
        )
    payload["outcome"] = canonical


def parse_provider_payload(payload: Any) -> RuleAuthoringProviderResult:
    """严格解析外部模型 JSON，仅归一化无歧义的已知别名。"""

    if isinstance(payload, str):
        try:
            payload_size = len(payload.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as error:
            raise RuleAuthoringProviderError("模型规则结果包含非法 Unicode。") from error
        if payload_size > 64 * 1024:
            raise RuleAuthoringProviderError("模型规则结果超过 64 KiB 上限。")

        def reject_duplicate_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise RuleAuthoringProviderError(f"模型 JSON 包含重复键：{key}。")
                result[key] = value
            return result

        try:
            payload = json.loads(
                payload,
                object_pairs_hook=reject_duplicate_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    RuleAuthoringProviderError(
                        f"模型 JSON 包含非标准数值：{value}。"
                    )
                ),
            )
        except RuleAuthoringProviderError:
            raise
        except (TypeError, ValueError, UnicodeError) as error:
            raise RuleAuthoringProviderError("模型未返回严格 JSON 规则。") from error
    if not isinstance(payload, Mapping):
        raise RuleAuthoringProviderError("模型规则结果必须是 JSON 对象。")
    payload = dict(payload)
    if not all(isinstance(key, str) for key in payload):
        raise RuleAuthoringProviderError("模型规则结果的 JSON 键必须是字符串。")
    try:
        json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(
            "utf-8", errors="strict"
        )
    except (TypeError, ValueError, UnicodeError) as error:
        raise RuleAuthoringProviderError(
            "模型规则结果包含不可序列化、非有限数值或非法 Unicode。"
        ) from error

    wrapper_fields = {
        "outcome",
        "rule_spec",
        "evidence",
        "assumptions",
        "clarification_questions",
        "unsupported_reason",
    }
    rule_spec_fields = {
        "rule_type",
        "rule_id",
        "name",
        "description",
        "fields",
        "parameters",
        "severity",
        "denominator_policy",
        "missing_value_policy",
        "evidence_ids",
        "resource_policy",
    }
    _move_provider_alias(payload, "outcome", ("status", "result_type"))
    _move_provider_alias(payload, "rule_spec", ("rule", "ruleSpec", "rule_draft"))
    _move_provider_alias(
        payload,
        "clarification_questions",
        ("questions", "clarifications"),
    )
    _move_provider_alias(payload, "unsupported_reason", ("reason",))
    if not {"outcome", "rule_spec"}.issubset(payload):
        if "rule_type" in payload and "fields" in payload:
            unknown = sorted(set(payload) - rule_spec_fields)
            if unknown:
                raise RuleAuthoringProviderError(
                    f"模型 rule_spec 包含未允许字段：{unknown}。"
                )
            payload = {
                "outcome": "draft",
                "rule_spec": payload,
                "assumptions": [],
                "clarification_questions": [],
                "unsupported_reason": None,
            }
        elif "clarification_questions" in payload:
            unknown = sorted(
                set(payload)
                - {"outcome", "assumptions", "clarification_questions"}
            )
            if unknown:
                raise RuleAuthoringProviderError(
                    f"模型澄清结果包含未允许字段：{unknown}。"
                )
            payload = {
                "outcome": payload.get("outcome", "clarification"),
                "rule_spec": None,
                "assumptions": payload.get("assumptions", []),
                "clarification_questions": payload.get(
                    "clarification_questions", []
                ),
                "unsupported_reason": None,
            }
        elif "outcome" in payload and "unsupported_reason" in payload:
            unknown = sorted(
                set(payload) - {"outcome", "assumptions", "unsupported_reason"}
            )
            if unknown:
                raise RuleAuthoringProviderError(
                    f"模型不支持结果包含未允许字段：{unknown}。"
                )
            payload = {
                "outcome": payload["outcome"],
                "rule_spec": None,
                "assumptions": payload.get("assumptions", []),
                "clarification_questions": [],
                "unsupported_reason": payload["unsupported_reason"],
            }
        else:
            raise RuleAuthoringProviderError(
                "模型规则结果缺少 outcome、rule_spec 或 clarification_questions。"
            )
    unknown = sorted(set(payload) - wrapper_fields)
    if unknown:
        raise RuleAuthoringProviderError(
            f"模型规则结果包含未允许字段：{unknown}。"
        )
    payload.setdefault("assumptions", [])
    payload.setdefault("clarification_questions", [])
    payload.setdefault("unsupported_reason", None)
    _normalize_provider_outcome(payload)
    outcome = payload["outcome"]
    assumptions = payload["assumptions"]
    questions = payload["clarification_questions"]
    if not isinstance(assumptions, list) or not all(isinstance(item, str) for item in assumptions):
        raise RuleAuthoringProviderError("assumptions 必须是字符串数组。")
    if not isinstance(questions, list) or not all(isinstance(item, str) for item in questions):
        raise RuleAuthoringProviderError("clarification_questions 必须是字符串数组。")
    if len(assumptions) > 20 or len(questions) > 5:
        raise RuleAuthoringProviderError("模型返回的假设或澄清问题数量超出限制。")
    if any(len(item) > 2000 or not item.strip() for item in (*assumptions, *questions)):
        raise RuleAuthoringProviderError("模型返回的假设或澄清问题包含空值或超长文本。")
    raw_evidence = payload.get("evidence", [])
    if not isinstance(raw_evidence, list) or len(raw_evidence) > 20:
        raise RuleAuthoringProviderError("模型返回的 evidence 必须是不超过 20 项的数组。")
    evidence: list[RuleEvidence] = []
    evidence_fields = {
        "type",
        "text",
        "source_id",
        "source_label",
        "location",
        "authoritative",
        "document_id",
        "document_name",
        "document_version",
        "section",
        "clause",
        "chunk_id",
        "page",
    }
    for index, item in enumerate(raw_evidence):
        if not isinstance(item, Mapping):
            raise RuleAuthoringProviderError(f"evidence[{index}] 必须是对象。")
        if not all(isinstance(key, str) for key in item):
            raise RuleAuthoringProviderError(f"evidence[{index}] 的 JSON 键必须是字符串。")
        unknown = sorted(set(item) - evidence_fields)
        if unknown:
            raise RuleAuthoringProviderError(
                f"evidence[{index}] 包含未允许字段：{unknown}。"
            )
        evidence_type = item.get("type")
        if evidence_type not in {"standard_clause", "data_dictionary"}:
            raise RuleAuthoringProviderError(
                "模型只能引用 standard_clause 或 data_dictionary 依据。"
            )
        text = item.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise RuleAuthoringProviderError(
                f"evidence[{index}].text 必须是 1 到 4000 个字符。"
            )
        page = item.get("page")
        if page is not None and (
            isinstance(page, bool) or not isinstance(page, int) or page < 1
        ):
            raise RuleAuthoringProviderError(f"evidence[{index}].page 无效。")
        authoritative = item.get("authoritative", False)
        if not isinstance(authoritative, bool):
            raise RuleAuthoringProviderError(
                f"evidence[{index}].authoritative 必须是布尔值。"
            )
        for key, maximum in (
            ("source_id", 300),
            ("source_label", 300),
            ("location", 500),
            ("document_id", 120),
            ("document_name", 300),
            ("document_version", 100),
            ("section", 300),
            ("clause", 120),
            ("chunk_id", 120),
        ):
            value = item.get(key)
            if value is not None and (
                not isinstance(value, str) or len(value) > maximum
            ):
                raise RuleAuthoringProviderError(
                    f"evidence[{index}].{key} 必须是不超过 {maximum} 个字符的字符串或 null。"
                )
        evidence.append(
            new_evidence(
                evidence_type,  # type: ignore[arg-type]
                text,
                source_id=item.get("source_id"),
                source_label=item.get("source_label"),
                location=item.get("location"),
                authoritative=authoritative,
                document_id=item.get("document_id"),
                document_name=item.get("document_name"),
                document_version=item.get("document_version"),
                section=item.get("section"),
                clause=item.get("clause"),
                chunk_id=item.get("chunk_id"),
                page=page,
            )
        )
    rule_spec_payload = payload["rule_spec"]
    rule_spec = None
    if rule_spec_payload is not None:
        if not isinstance(rule_spec_payload, Mapping):
            raise RuleAuthoringProviderError("rule_spec 必须是对象或 null。")
        if not all(isinstance(key, str) for key in rule_spec_payload):
            raise RuleAuthoringProviderError("rule_spec 的 JSON 键必须是字符串。")
        unknown = sorted(set(rule_spec_payload) - rule_spec_fields)
        if unknown:
            raise RuleAuthoringProviderError(
                f"模型 rule_spec 包含未允许字段：{unknown}。"
            )
        normalized_rule_spec = {
            "rule_type": rule_spec_payload.get("rule_type"),
            "rule_id": rule_spec_payload.get("rule_id", "model-candidate"),
            "name": rule_spec_payload.get("name", "模型生成的质量规则"),
            "description": rule_spec_payload.get(
                "description", "根据用户补充的评价依据生成。"
            ),
            "fields": rule_spec_payload.get("fields", []),
            "parameters": rule_spec_payload.get("parameters", {}),
            "severity": rule_spec_payload.get("severity", "attention"),
            "denominator_policy": rule_spec_payload.get(
                "denominator_policy", "all_records"
            ),
            "missing_value_policy": rule_spec_payload.get(
                "missing_value_policy", "missing_is_violation"
            ),
            "evidence_ids": rule_spec_payload.get("evidence_ids", []),
            "resource_policy": rule_spec_payload.get(
                "resource_policy", {"max_inspection_cells": 2_000_000}
            ),
        }
        try:
            rule_spec = rule_spec_from_dict(normalized_rule_spec)
        except Exception as error:
            raise RuleAuthoringProviderError(
                f"模型规则字段无法转换：{error}"
            ) from error
        if len(set(rule_spec.fields)) != len(rule_spec.fields):
            raise RuleAuthoringProviderError("模型 rule_spec.fields 不能包含重复字段。")
        try:
            json.dumps(rule_spec.to_dict(), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise RuleAuthoringProviderError(
                "模型 rule_spec 包含不可序列化或非有限数值。"
            ) from error
    unsupported_reason = payload["unsupported_reason"]
    if unsupported_reason is not None and not isinstance(unsupported_reason, str):
        raise RuleAuthoringProviderError("unsupported_reason 必须是字符串或 null。")
    if isinstance(unsupported_reason, str) and (
        not unsupported_reason.strip() or len(unsupported_reason) > 2000
    ):
        raise RuleAuthoringProviderError(
            "unsupported_reason 必须是 1 到 2000 个字符。"
        )
    if outcome == "draft" and rule_spec is None:
        raise RuleAuthoringProviderError("draft outcome 必须携带 rule_spec。")
    if outcome != "draft" and rule_spec is not None:
        raise RuleAuthoringProviderError("非 draft outcome 不能携带 rule_spec。")
    if outcome == "draft":
        if questions:
            raise RuleAuthoringProviderError(
                "draft outcome 不能同时携带 clarification_questions。"
            )
        if unsupported_reason is not None:
            raise RuleAuthoringProviderError(
                "draft outcome 不能同时携带 unsupported_reason。"
            )
    elif outcome == "clarification":
        if not questions:
            raise RuleAuthoringProviderError(
                "clarification outcome 必须至少提出一个问题。"
            )
        if unsupported_reason is not None:
            raise RuleAuthoringProviderError(
                "clarification outcome 不能携带 unsupported_reason。"
            )
    elif questions:
        raise RuleAuthoringProviderError(
            "unsupported outcome 不能携带 clarification_questions。"
        )
    if outcome == "unsupported" and not str(unsupported_reason or "").strip():
        raise RuleAuthoringProviderError("unsupported outcome 必须说明原因。")
    return RuleAuthoringProviderResult(
        outcome=outcome,
        rule_spec=rule_spec,
        evidence=tuple(evidence),
        assumptions=tuple(assumptions[:20]),
        clarification_questions=tuple(questions[:5]),
        unsupported_reason=unsupported_reason,
        metadata=_metadata(provider="deepseek", mode="model"),
    )


class DeepSeekRuleAuthoringProvider:
    """可选 Chat Completions 结构化输出适配。"""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
        provider_name: str = "deepseek",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client_factory: Callable[..., Any] | None = None,
        prompt_version: str = RULE_AUTHORING_PROMPT_VERSION,
    ) -> None:
        self.model = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
        self._api_key = api_key
        self._api_url = api_url
        self.provider_name = provider_name.strip() or "deepseek"
        self.timeout_seconds = float(timeout_seconds)
        self._client_factory = client_factory
        self.prompt = get_rule_authoring_prompt(prompt_version)
        self.prompt_version = self.prompt.version

    def _client(self) -> tuple[Any, dict[str, str]]:
        api_key = (
            self._api_key.strip()
            if self._api_key is not None
            else os.environ.get("DEEPSEEK_API_KEY", "").strip()
        )
        if not api_key:
            raise RuleAuthoringProviderUnavailable("未配置模型 API key。")
        factory = self._client_factory
        if factory is None:
            try:
                import httpx
            except ImportError as error:
                raise RuleAuthoringProviderUnavailable("未安装 httpx。") from error
            factory = httpx.Client
        return factory(timeout=self.timeout_seconds), {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _api_endpoint(self) -> str:
        value = self._api_url
        if value is None:
            value = os.environ.get(
                "DEEPSEEK_API_URL",
                DEEPSEEK_CHAT_COMPLETIONS_URL,
            )
        return normalize_chat_completions_url(value)

    def _post_request(
        self,
        client: Any,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """兼容不支持 response_format、temperature 或 max_tokens 的端点。"""

        endpoint = self._api_endpoint()
        variants: list[dict[str, Any]] = [dict(payload)]
        without_json_mode = dict(payload)
        without_json_mode.pop("response_format", None)
        variants.append(without_json_mode)
        without_optional = dict(without_json_mode)
        without_optional.pop("temperature", None)
        without_optional.pop("max_tokens", None)
        variants.append(without_optional)
        last_error: Exception | None = None
        for index, request_payload in enumerate(variants):
            try:
                response = client.post(
                    endpoint,
                    headers=dict(headers),
                    json=request_payload,
                )
                status_code = getattr(response, "status_code", None)
                if not isinstance(status_code, int):
                    raise RuleAuthoringProviderError(
                        "模型 API 响应缺少 HTTP 状态码。"
                    )
                if status_code >= 400:
                    if index < len(variants) - 1 and status_code in {400, 404, 405, 415, 422}:
                        last_error = RuleAuthoringProviderError(
                            f"模型 API 拒绝当前请求参数：{response_error_detail(response)}"
                        )
                        continue
                    raise RuleAuthoringProviderError(
                        f"模型 API 规则请求失败：{response_error_detail(response)}"
                    )
                body = response.json()
                if not isinstance(body, Mapping):
                    raise RuleAuthoringProviderError("模型 API 响应不是 JSON 对象。")
                return body
            except RuleAuthoringProviderError:
                raise
            except Exception as error:
                last_error = error
                if index == len(variants) - 1:
                    break
        raise RuleAuthoringProviderError(
            f"模型 API 规则请求失败：{str(last_error)[:500] if last_error else '未知错误'}"
        ) from last_error

    def generate(
        self,
        context: Mapping[str, Any],
        *,
        user_intent: str,
    ) -> RuleAuthoringProviderResult:
        client, headers = self._client()
        request_id = make_rule_id(
            "request",
            [
                str(context.get("report_sha256", "")),
                user_intent,
                str(context.get("rag", {}).get("chunk_ids", []))
                if isinstance(context.get("rag"), Mapping)
                else "",
            ],
        )
        prompt = {
            "user_intent": user_intent,
            "metric": context.get("metric"),
            "fields": context.get("fields", []),
            "profile_summary": context.get("profile_summary", {}),
            "rag_evidence": (
                context.get("rag", {}).get("results", [])
                if isinstance(context.get("rag"), Mapping)
                else []
            ),
            "allowed_rule_types": [
                "primary_key",
                "required",
                "update_freshness",
                "allowed_values",
                "numeric_range",
                "regex_format",
                "string_length",
                "conditional_required",
                "field_comparison",
            ],
        }
        try:
            response = self._post_request(
                client,
                headers=headers,
                payload={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.prompt.system_prompt},
                        {
                            "role": "user",
                    "content": json.dumps(
                                prompt, ensure_ascii=False, separators=(",", ":")
                            ),
                        },
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0,
                    "max_tokens": 2500,
                    "stream": False,
                },
            )
            choices = response.get("choices")
            message = choices[0].get("message") if choices else None
            if isinstance(message, Mapping):
                content = extract_message_content(message.get("content"))
            else:
                content = extract_message_content(
                    choices[0].get("text") if choices else None
                )
            result = parse_provider_payload(content)
            usage = response.get("usage", {}) if isinstance(response, Mapping) else {}
            return RuleAuthoringProviderResult(
                outcome=result.outcome,
                rule_spec=result.rule_spec,
                evidence=result.evidence,
                assumptions=result.assumptions,
                clarification_questions=result.clarification_questions,
                unsupported_reason=result.unsupported_reason,
                metadata=_metadata(
                    provider=self.provider_name,
                    model=self.model,
                    mode="model",
                    request_id=request_id,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    prompt_version=self.prompt_version,
                ),
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()


class OpenAICompatibleRuleAuthoringProvider(DeepSeekRuleAuthoringProvider):
    """页面自定义的兼容 Chat Completions 规则编制提供方。"""

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        model: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client_factory: Callable[..., Any] | None = None,
        prompt_version: str = RULE_AUTHORING_PROMPT_VERSION,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            api_url=api_url,
            provider_name="custom",
            timeout_seconds=timeout_seconds,
            client_factory=client_factory,
            prompt_version=prompt_version,
        )


def default_rule_authoring_provider() -> RuleAuthoringProvider:
    if os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return DeepSeekRuleAuthoringProvider()
    return TemplateRuleAuthoringProvider()


__all__ = [
    "DEFAULT_DEEPSEEK_MODEL",
    "DEEPSEEK_CHAT_COMPLETIONS_URL",
    "DeepSeekRuleAuthoringProvider",
    "OpenAICompatibleRuleAuthoringProvider",
    "RULE_AUTHORING_PROMPT_VERSION",
    "RuleAuthoringProvider",
    "RuleAuthoringProviderError",
    "RuleAuthoringProviderResult",
    "RuleAuthoringProviderUnavailable",
    "RuleIntentInspection",
    "TemplateRuleAuthoringProvider",
    "build_rule_input_guidance",
    "default_rule_authoring_provider",
    "inspect_rule_intent",
    "parse_provider_payload",
]
