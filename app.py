"""政务数据集质量评估的 Streamlit 网页入口。"""

import hashlib
import json
import math
import os
from datetime import date
from html import escape as escape_html
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent_models import AgentAnalysis
from src.agent_service import run_agent
from src.metric_catalog import (
    ALL_METRIC_IDS,
    DEFAULT_SELECTED_METRIC_IDS,
    build_metric_catalog_rows,
    get_metric_definition,
    metric_description,
    normalize_selected_metric_ids,
)
from src.models import QualityReport
from src.parser import DatasetReadError, UnsupportedFileTypeError
from src.presentation import (
    RISK_LEVEL_LABELS,
    build_metric_rows,
    build_profile_rows,
    build_risk_chart_rows,
    build_summary,
    serialize_issue_locations_csv,
    serialize_markdown_report,
    serialize_report,
)
from src.resource_limits import MAX_INPUT_FILE_MIB
from src.rule_engine import RulePackExecutionError
from src.rule_pack import (
    Rule,
    MAX_RULE_NUMBER_ABS,
    RulePackValidationError,
    approve_rule_pack,
    build_rule_guidance,
    build_rule_pack,
    draft_sha256,
    validate_rule_pack,
)
from src.rule_service import (
    evaluate_uploaded_dataset_with_rule_pack,
    serialize_rule_evaluation_result,
    serialize_rule_issue_locations_csv,
)
from src.upload_service import evaluate_uploaded_dataset, sanitize_file_name


AGENT_STATE_KEY = "agent_ui_state"
RULE_STATE_KEY = "rule_ui_state"
METRIC_SELECTION_KEY = "selected_metric_ids"
METRIC_SELECTION_WIDGET_PREFIX = "metric_selection_checkbox_"
AGENT_HISTORY_LIMIT = 8
AGENT_PRIORITY_LABELS = {
    "high": "高",
    "medium": "中",
    "low": "低",
}
RULE_WIDGET_KEYS = (
    "rule_pack_name",
    "rule_pack_version",
    "rule_primary_key_fields",
    "rule_required_fields",
    "rule_update_time_field",
    "rule_update_frequency",
    "rule_update_custom_days",
    "rule_allowed_values_field",
    "rule_allowed_values_json",
    "rule_numeric_range_field",
    "rule_numeric_minimum",
    "rule_numeric_maximum",
    "rule_approver",
    "rule_approval_confirmed",
)
RULE_FREQUENCIES = {
    "每日（1 天）": ("daily", 1),
    "每周（7 天）": ("weekly", 7),
    "每月（31 天）": ("monthly", 31),
    "每季度（92 天）": ("quarterly", 92),
    "每年（366 天）": ("yearly", 366),
    "自定义天数": ("custom", None),
}

METRIC_PRESET_BUTTON_KEYS = {
    "default": "metric_preset_default",
    "all": "metric_preset_all",
    "clear": "metric_preset_clear",
}


st.set_page_config(
    page_title="政务数据集质量评估",
    page_icon="📊",
    layout="wide",
)


def _clear_agent_state() -> None:
    """清除只属于当前确定性报告的 Agent 结果与问答记录。"""

    st.session_state.pop(AGENT_STATE_KEY, None)


def _clear_rule_state() -> None:
    """清除当前报告的规则草案、审批、增强结果及表单缓存。"""

    st.session_state.pop(RULE_STATE_KEY, None)
    for key in RULE_WIDGET_KEYS:
        st.session_state.pop(key, None)


def _metric_checkbox_key(metric_id: str) -> str:
    """为每一张指标卡生成稳定且独立的复选框状态键。"""

    return f"{METRIC_SELECTION_WIDGET_PREFIX}{metric_id}"


def _set_metric_selection(metric_ids: tuple[str, ...]) -> None:
    """由快捷预设在指标卡控件创建前同步更新所有选择状态。"""

    normalized = normalize_selected_metric_ids(metric_ids)
    st.session_state[METRIC_SELECTION_KEY] = list(normalized)
    selected = set(normalized)
    for metric_id in ALL_METRIC_IDS:
        st.session_state[_metric_checkbox_key(metric_id)] = (
            metric_id in selected
        )


def _clear_metric_selection() -> None:
    """清空所有指标卡的选择状态。"""

    st.session_state[METRIC_SELECTION_KEY] = []
    for metric_id in ALL_METRIC_IDS:
        st.session_state[_metric_checkbox_key(metric_id)] = False


def _initialize_metric_selection_state() -> None:
    """初始化指标卡状态，并兼容旧版多选框保存的选择结果。"""

    widget_keys = {
        metric_id: _metric_checkbox_key(metric_id)
        for metric_id in ALL_METRIC_IDS
    }
    if any(key in st.session_state for key in widget_keys.values()):
        requested = {
            metric_id
            for metric_id, key in widget_keys.items()
            if bool(st.session_state.get(key, False))
        }
    else:
        current = st.session_state.get(METRIC_SELECTION_KEY)
        if not isinstance(current, (list, tuple, set)):
            current = DEFAULT_SELECTED_METRIC_IDS
        requested = {
            metric_id for metric_id in current if isinstance(metric_id, str)
        }

    normalized = [
        metric_id for metric_id in ALL_METRIC_IDS if metric_id in requested
    ]
    st.session_state[METRIC_SELECTION_KEY] = normalized
    for metric_id, key in widget_keys.items():
        if key not in st.session_state:
            st.session_state[key] = metric_id in requested


def _selected_metric_ids_from_cards() -> tuple[str, ...]:
    """从指标卡读取选择，并按照固定目录顺序固化到会话状态。"""

    selected = tuple(
        metric_id
        for metric_id in ALL_METRIC_IDS
        if bool(st.session_state.get(_metric_checkbox_key(metric_id), False))
    )
    st.session_state[METRIC_SELECTION_KEY] = list(selected)
    return selected


def _render_metric_card(metric_id: str) -> None:
    """在主页面渲染一张带悬停释义的可选指标卡。"""

    definition = get_metric_definition(metric_id)
    if definition is None:
        return
    capability = (
        "当前可直接计算"
        if definition["auto_assessable"]
        else "需补充评价依据"
    )

    with st.container(border=True):
        title_column, help_column = st.columns((12, 1))
        with title_column:
            st.checkbox(
                str(definition["name"]),
                key=_metric_checkbox_key(metric_id),
            )
        with help_column:
            description = escape_html(metric_description(metric_id), quote=True)
            st.markdown(
                (
                    '<span class="metric-help-icon" '
                    f'data-tooltip="{description}" '
                    f'aria-label="{description}" role="img" tabindex="0">?</span>'
                ),
                unsafe_allow_html=True,
            )
        st.caption(f"{definition['dimension']} · {capability}")
        st.caption(f"计算方式：{definition['formula']}")


def _render_metric_cards(metric_ids: tuple[str, ...]) -> None:
    """按三列布局绘制统一的指标卡列表。"""

    for start_index in range(0, len(metric_ids), 3):
        for column, metric_id in zip(
            st.columns(3), metric_ids[start_index : start_index + 3]
        ):
            with column:
                _render_metric_card(metric_id)


def _render_metric_selection_panel() -> tuple[str, ...]:
    """在主内容区展示统一的指标卡、预设和计算方式目录。"""

    st.subheader("选择评价指标")
    st.caption(
        "默认已选中一组基础指标。点击卡片即可自由组合；将鼠标悬停在每张卡片"
        "右上角的“？”上可查看指标含义。缺少必要评价依据的指标会在报告中如实"
        "标记为无法评估。"
    )
    preset_columns = st.columns(3)
    preset_columns[0].button(
        "默认指标",
        key=METRIC_PRESET_BUTTON_KEYS["default"],
        width="stretch",
        on_click=_set_metric_selection,
        args=(DEFAULT_SELECTED_METRIC_IDS,),
    )
    preset_columns[1].button(
        "全部指标",
        key=METRIC_PRESET_BUTTON_KEYS["all"],
        width="stretch",
        on_click=_set_metric_selection,
        args=(ALL_METRIC_IDS,),
    )
    preset_columns[2].button(
        "清空选择",
        key=METRIC_PRESET_BUTTON_KEYS["clear"],
        width="stretch",
        on_click=_clear_metric_selection,
    )

    st.markdown(
        """
        <style>
        .metric-help-icon {
            align-items: center;
            border: 1px solid #64748b;
            border-radius: 50%;
            color: #334155;
            cursor: help;
            display: inline-flex;
            font-size: 0.85rem;
            font-weight: 700;
            height: 1.45rem;
            justify-content: center;
            line-height: 1;
            margin-top: 0.12rem;
            position: relative;
            width: 1.45rem;
        }
        .metric-help-icon::after {
            background: #0f172a;
            border-radius: 0.35rem;
            color: #ffffff;
            content: attr(data-tooltip);
            font-size: 0.8rem;
            font-weight: 400;
            line-height: 1.4;
            max-width: 18rem;
            opacity: 0;
            padding: 0.5rem 0.65rem;
            pointer-events: none;
            position: absolute;
            right: 0;
            text-align: left;
            top: calc(100% + 0.4rem);
            transform: translateY(-0.2rem);
            transition: opacity 80ms ease, transform 80ms ease;
            visibility: hidden;
            width: max-content;
            z-index: 1000;
        }
        .metric-help-icon:hover {
            background: #e2e8f0;
            border-color: #0f172a;
            color: #0f172a;
        }
        .metric-help-icon:hover::after,
        .metric-help-icon:focus::after {
            opacity: 1;
            transform: translateY(0);
            visibility: visible;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"评价指标（{len(ALL_METRIC_IDS)} 项）", expanded=True):
        _render_metric_cards(ALL_METRIC_IDS)

    selected_metric_ids = _selected_metric_ids_from_cards()
    st.caption(f"已选 {len(selected_metric_ids)} 项。")
    if not selected_metric_ids:
        st.warning("请至少选择一个评价指标后再运行。")

    with st.expander("查看指标目录与计算方式"):
        catalog = pd.DataFrame(build_metric_catalog_rows()).drop(
            columns=["来源", "标准代码", "层级", "指标 ID"],
            errors="ignore",
        )
        st.dataframe(
            catalog,
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "“当前能力”说明本地单表是否具备直接评价依据。"
        )

    return selected_metric_ids


def _escape_markdown(value: object) -> str:
    """转义会触发 Markdown 链接、图片或 HTML 的不可信展示文本。"""

    text = str(value)
    for character in ("\\", "`", "*", "_", "[", "]", "(", ")", "<", ">", "#", "!", "|"):
        text = text.replace(character, f"\\{character}")
    return text


def _report_sha256(report: QualityReport) -> str:
    """读取由结构化报告规范化计算出的稳定哈希。"""

    return str(
        report.to_dict()
        .get("evaluation_context", {})
        .get("report_sha256", "")
    )


def _selected_metric_ids_for_report(
    report: QualityReport,
) -> tuple[str, ...]:
    """优先读取报告固化的选择；兼容尚未携带该字段的旧报告。"""

    context = report.to_dict().get("evaluation_context", {})
    selected = (
        context.get("selected_metric_ids")
        if isinstance(context, dict)
        else None
    )
    if isinstance(selected, list):
        requested = {
            metric_id
            for metric_id in selected
            if isinstance(metric_id, str)
        }
    else:
        requested = {
            metric.id
            for metric in report.metrics
            if get_metric_definition(metric.id) is not None
        }
    return tuple(
        metric_id for metric_id in ALL_METRIC_IDS if metric_id in requested
    )


def _render_metric_selection_summary(report: QualityReport) -> None:
    """展示绑定当前报告的统一指标选择与可评估能力统计。"""

    selected = _selected_metric_ids_for_report(report)
    auto_assessable_count = sum(
        bool(definition and definition.get("auto_assessable"))
        for metric_id in selected
        for definition in (get_metric_definition(metric_id),)
    )
    st.caption(
        "本次指标选择："
        f"共 {len(selected)} 项 · 当前可直接计算 {auto_assessable_count} 项 · "
        f"需补充评价依据 {len(selected) - auto_assessable_count} 项"
    )


def _agent_state_for(report: QualityReport) -> dict:
    """返回绑定当前报告的 Agent 页面状态，拒绝跨报告复用。"""

    report_sha256 = _report_sha256(report)
    state = st.session_state.get(AGENT_STATE_KEY)
    if not isinstance(state, dict) or state.get("report_sha256") != report_sha256:
        state = {
            "report_sha256": report_sha256,
            "latest_analysis": None,
            "history": [],
        }
        st.session_state[AGENT_STATE_KEY] = state
    return state


def _rule_state_for(report: QualityReport) -> dict:
    """返回绑定当前零配置报告的 RulePack 页面状态。"""

    report_sha256 = _report_sha256(report)
    state = st.session_state.get(RULE_STATE_KEY)
    if not isinstance(state, dict) or state.get("report_sha256") != report_sha256:
        state = {
            "report_sha256": report_sha256,
            "guidance_started": False,
            "draft": None,
            "draft_signature": None,
            "confirmed_draft_sha256": None,
            "approved_pack": None,
            "result": None,
            "execution_error": None,
        }
        st.session_state[RULE_STATE_KEY] = state
    return state


def _clear_rule_approval_widgets() -> None:
    """在新草案出现前清除只适用于上一草案的人工确认。"""

    st.session_state.pop("rule_approver", None)
    st.session_state.pop("rule_approval_confirmed", None)


def _bind_rule_confirmation() -> None:
    """把一次真实 checkbox 变更绑定到当前草案哈希。"""

    state = st.session_state.get(RULE_STATE_KEY)
    if not isinstance(state, dict):
        return
    draft = state.get("draft")
    confirmed = bool(st.session_state.get("rule_approval_confirmed"))
    confirmed_hash = (
        draft_sha256(draft)
        if confirmed and draft is not None
        else None
    )
    st.session_state[RULE_STATE_KEY] = {
        **state,
        "confirmed_draft_sha256": confirmed_hash,
    }


def _citation_caption(
    analysis: AgentAnalysis,
    citation_ids: list[str],
) -> str:
    citations = {
        citation.id: citation
        for citation in analysis.citations
    }
    labels = []
    for citation_id in citation_ids:
        citation = citations.get(citation_id)
        if citation is None:
            continue
        labels.append(
            f"{citation.label}（{citation.source_id}）：{citation.excerpt}"
        )
    return "证据：" + "；".join(labels) if labels else "证据：当前报告未提供足够依据"


def _render_agent_analysis(analysis: AgentAnalysis) -> None:
    """只按已校验结构展示 AgentAnalysis，不渲染任意 HTML。"""

    if analysis.audit.mode == "model" and not analysis.audit.fallback_used:
        st.success(f"模型增强模式 · {analysis.audit.model}")
    elif analysis.audit.fallback_used:
        if analysis.audit.fallback_reason == "provider_unavailable":
            st.info("DeepSeek 未配置或暂不可用，已使用本地模板生成可追溯解读。")
        elif analysis.audit.fallback_reason == "provider_error":
            st.info("外部模型调用失败，已使用本地模板生成可追溯解读。")
        else:
            st.info("模型结果未通过安全校验，已使用本地模板生成可追溯解读。")
    else:
        st.info("当前使用本地模板模式；无需 API，也不会向外部发送报告。")

    st.markdown("#### 回答")
    st.text(analysis.answer.text)
    st.text(_citation_caption(analysis, analysis.answer.citation_ids))

    if analysis.facts:
        st.markdown("#### 报告事实")
        for fact in analysis.facts:
            st.text(fact.text)
            st.text(_citation_caption(analysis, fact.citation_ids))

    if analysis.actions:
        st.markdown("#### 优先整改建议")
        for action in analysis.actions:
            priority = AGENT_PRIORITY_LABELS.get(
                action.priority,
                action.priority,
            )
            st.text(f"{priority}优先级 · {action.title}")
            st.text(action.detail)
            st.text(_citation_caption(analysis, action.citation_ids))

    if analysis.limitations:
        st.markdown("#### 缺少依据、无法判断")
        for limitation in analysis.limitations:
            st.text(limitation.text)
            st.text(_citation_caption(analysis, limitation.citation_ids))

    with st.expander("查看 Agent 审计信息"):
        st.json(
            {
                "report_sha256": analysis.report_sha256,
                "intent": analysis.intent,
                "provider": analysis.audit.provider,
                "model": analysis.audit.model,
                "mode": analysis.audit.mode,
                "prompt_version": analysis.audit.prompt_version,
                "fallback_used": analysis.audit.fallback_used,
                "fallback_reason": analysis.audit.fallback_reason,
                "tool_calls": analysis.audit.tool_calls,
                "input_tokens": analysis.audit.input_tokens,
                "output_tokens": analysis.audit.output_tokens,
                "latency_ms": analysis.audit.latency_ms,
                "cache_hit": analysis.audit.cache_hit,
            }
        )


def _run_agent_request(
    report: QualityReport,
    *,
    intent: str,
    prompt_label: str,
    question: str | None = None,
) -> None:
    """执行一次只读诊断，并把结果绑定到当前报告。"""

    state = _agent_state_for(report)
    # 新请求开始后不继续展示旧结论；失败时保留历史，但 latest 保持为空。
    st.session_state[AGENT_STATE_KEY] = {
        "report_sha256": state["report_sha256"],
        "latest_analysis": None,
        "history": list(state.get("history", [])),
    }
    with st.spinner("Agent 正在核对报告证据……"):
        analysis = run_agent(
            report,
            intent=intent,
            question=question,
        )
    history = list(state.get("history", []))
    if question is not None:
        history = [
            *history,
            {
                "prompt": prompt_label,
                "is_question": True,
                "analysis": analysis,
            },
        ][-AGENT_HISTORY_LIMIT:]
    st.session_state[AGENT_STATE_KEY] = {
        "report_sha256": analysis.report_sha256,
        "latest_analysis": analysis,
        "history": history,
    }


def _render_agent(report: QualityReport) -> None:
    """渲染由用户显式触发的只读报告诊断 Agent。"""

    st.subheader("只读报告诊断 Agent")
    st.caption(
        "Agent 只读取当前 QualityReport，不重新计算指标、不改变风险等级，"
        "也不会修改或清洗原始数据。"
    )
    if (
        os.environ.get("QUALITY_AGENT_PROVIDER", "")
        .strip()
        .casefold()
        == "deepseek"
    ):
        if os.environ.get("DEEPSEEK_API_KEY", "").strip():
            st.warning(
                "当前部署已配置 DeepSeek 外部模式。点击快捷入口或提交问题时，"
                "会发送经过白名单过滤的报告投影；不发送原始单元格值。"
            )
        else:
            st.warning(
                "当前部署已选择 DeepSeek 外部模式，但尚未配置 "
                "DEEPSEEK_API_KEY；请求会回退到本地模板，不会向外发送报告。"
            )
    else:
        st.caption("当前部署使用本地模板；点击后也不会向外部发送报告。")
    action_columns = st.columns(3)
    summarize = action_columns[0].button(
        "概括结果",
        key="agent_summary",
        width="stretch",
    )
    prioritize = action_columns[1].button(
        "优先整改事项",
        key="agent_priority",
        width="stretch",
    )
    explain_missing = action_columns[2].button(
        "解释无法评估项",
        key="agent_not_assessable",
        width="stretch",
    )
    question = st.chat_input(
        "询问当前报告，例如：最需要优先处理什么？",
        key="agent_report_question",
        max_chars=500,
    )

    try:
        if summarize:
            _run_agent_request(
                report,
                intent="summary",
                prompt_label="概括结果",
            )
        elif prioritize:
            _run_agent_request(
                report,
                intent="priority",
                prompt_label="优先整改事项",
            )
        elif explain_missing:
            _run_agent_request(
                report,
                intent="not_assessable",
                prompt_label="解释无法评估项",
            )
        elif question:
            _run_agent_request(
                report,
                intent="question",
                prompt_label=question,
                question=question,
            )
    except Exception:
        st.error("Agent 解读暂时不可用；确定性评估报告不受影响，请稍后重试。")

    state = _agent_state_for(report)
    history = state.get("history", [])
    question_history = [entry for entry in history if entry.get("is_question")]
    if question_history:
        st.markdown("#### 当前报告问答记录")
        for entry in question_history:
            with st.chat_message("user"):
                st.text(entry["prompt"])
            with st.chat_message("assistant"):
                answer = entry["analysis"].answer
                st.text(answer.text)
                st.text(
                    _citation_caption(entry["analysis"], answer.citation_ids)
                )

    latest_analysis = state.get("latest_analysis")
    if latest_analysis is None:
        st.info("选择一个快捷入口或提交问题后，才会开始 Agent 解读。")
        return
    _render_agent_analysis(latest_analysis)


def _rule_form_signature(payload: dict) -> str:
    """稳定标识当前规则问答；任一答案变化都会使旧审批失效。"""

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8", errors="strict")
    return hashlib.sha256(canonical).hexdigest()


def _rule_id(rule_type: str, fields: list[str]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            [rule_type, *fields],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    ).hexdigest()[:12]
    return f"{rule_type}-{digest}"


def _parse_allowed_values(value: str) -> tuple[object, ...]:
    """严格解析本地 JSON 数组，不接受 NaN/Infinity 等扩展常量。"""

    try:
        parsed = json.loads(
            value,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"不允许 {constant}")
            ),
        )
    except (TypeError, ValueError) as error:
        raise RulePackValidationError(
            ("允许值必须是合法 JSON 数组。",)
        ) from error
    if not isinstance(parsed, list):
        raise RulePackValidationError(("允许值必须使用 JSON 数组。",))
    return tuple(parsed)


def _parse_optional_number(value: str, label: str) -> int | float | None:
    text = value.strip()
    if not text:
        return None
    try:
        parsed = json.loads(
            text,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"不允许 {constant}")
            ),
        )
    except (TypeError, ValueError) as error:
        raise RulePackValidationError(
            (f"{label}必须是合法 JSON 数字。",)
        ) from error
    if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
        raise RulePackValidationError((f"{label}必须是有限数字。",))
    if isinstance(parsed, int):
        if abs(parsed) > MAX_RULE_NUMBER_ABS:
            raise RulePackValidationError(
                (f"{label}绝对值不能超过 1e308。",)
            )
    elif (
        not math.isfinite(parsed)
        or abs(parsed) > MAX_RULE_NUMBER_ABS
    ):
        raise RulePackValidationError(
            (f"{label}必须是绝对值不超过 1e308 的有限数字。",)
        )
    return parsed


def _rule_form_rules(
    *,
    primary_key_fields: list[str],
    required_fields: list[str],
    update_time_field: str | None,
    frequency_label: str,
    custom_days: int,
    allowed_values_field: str | None,
    allowed_values_json: str,
    numeric_range_field: str | None,
    numeric_minimum_text: str,
    numeric_maximum_text: str,
) -> tuple[Rule, ...]:
    """把已经由用户回答的本地表单转换为严格白名单规则。"""

    rules: list[Rule] = []
    if primary_key_fields:
        rules.append(
            Rule(
                type="primary_key",
                rule_id=_rule_id("primary_key", primary_key_fields),
                fields=tuple(primary_key_fields),
            )
        )
    for field in required_fields:
        rules.append(
            Rule(
                type="required",
                rule_id=_rule_id("required", [field]),
                fields=(field,),
            )
        )
    if update_time_field is not None:
        frequency, default_days = RULE_FREQUENCIES[frequency_label]
        rules.append(
            Rule(
                type="update_freshness",
                rule_id=_rule_id(
                    "update_freshness",
                    [update_time_field],
                ),
                fields=(update_time_field,),
                frequency=frequency,
                max_age_days=(
                    int(custom_days)
                    if default_days is None
                    else int(default_days)
                ),
            )
        )
    if allowed_values_field is not None:
        rules.append(
            Rule(
                type="allowed_values",
                rule_id=_rule_id(
                    "allowed_values",
                    [allowed_values_field],
                ),
                fields=(allowed_values_field,),
                allowed_values=_parse_allowed_values(
                    allowed_values_json
                ),
            )
        )
    if numeric_range_field is not None:
        rules.append(
            Rule(
                type="numeric_range",
                rule_id=_rule_id(
                    "numeric_range",
                    [numeric_range_field],
                ),
                fields=(numeric_range_field,),
                minimum=_parse_optional_number(
                    numeric_minimum_text,
                    "数值下限",
                ),
                maximum=_parse_optional_number(
                    numeric_maximum_text,
                    "数值上限",
                ),
            )
        )
    if not rules:
        raise RulePackValidationError(
            ("请至少确认一条主键、必填、更新时间、允许值或数值范围规则。",)
        )
    return tuple(rules)


def _render_rule_result(result) -> None:
    """展示本次零配置与规则增强差异，不推断跨历史改善或恶化。"""

    baseline_summary = build_summary(result.baseline_report)
    enhanced_summary = build_summary(result.enhanced_report)
    st.success("已审批 RulePack 已由确定性 Python 引擎重新评估。")
    st.caption(
        "下表只说明本次新增业务规则结果；不表示跨版本的改善或恶化。"
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "结果层": "零配置基线",
                    "指标结果数": len(result.baseline_report.metrics),
                    "风险提示数": baseline_summary["risk_count"],
                    "无法评估项": baseline_summary["not_assessable_count"],
                },
                {
                    "结果层": "规则增强",
                    "指标结果数": len(result.enhanced_report.metrics),
                    "风险提示数": enhanced_summary["risk_count"],
                    "无法评估项": enhanced_summary["not_assessable_count"],
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    added_keys = set(result.diff.added_metric_keys)
    added_metrics = [
        metric
        for metric in result.enhanced_report.metrics
        if metric.metric_key in added_keys
    ]
    st.markdown("#### 新增业务规则指标")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "指标名称": metric.name,
                    "字段名称": metric.field or "—",
                    "状态": (
                        "已评估"
                        if metric.status == "evaluated"
                        else "无法评估"
                    ),
                    "结果": (
                        f"{float(metric.value):.2%}"
                        if metric.value is not None
                        and metric.unit == "ratio"
                        else "—"
                    ),
                    "原因": metric.reason or "—",
                }
                for metric in added_metrics
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    added_risk_ids = set(result.diff.added_risk_ids)
    added_risks = [
        risk
        for risk in result.enhanced_report.risks
        if risk.id in added_risk_ids
    ]
    st.markdown("#### 新增业务规则风险")
    if not added_risks:
        st.info("已审批业务规则没有新增风险提示。")
    for risk in added_risks:
        st.text(f"{RISK_LEVEL_LABELS[risk.level]} · {risk.title}")
        st.text(risk.message)

    approval = result.approved_rule_pack.approval
    with st.expander("查看规则来源与审批记录"):
        st.json(
            {
                "rule_pack_id": result.approved_rule_pack.rule_pack_id,
                "rule_pack_version": result.approved_rule_pack.version,
                "base_report_sha256": (
                    result.approved_rule_pack.base_report_sha256
                ),
                "base_input_sha256": (
                    result.approved_rule_pack.base_input_sha256
                ),
                "reference_date": result.approved_rule_pack.reference_date,
                "approval": (
                    approval.to_dict() if approval is not None else None
                ),
            }
        )
        st.caption("审批人是本地自声明标识，系统未验证其身份。")


def _render_rule_enhancement(
    report: QualityReport,
    *,
    uploaded_file,
    dataset_name: str,
    sheet_name: str,
    reference_date: date,
    selected_metric_ids: tuple[str, ...],
) -> None:
    """渲染 v0.4 引导、草案校验、明确审批和确定性重评闭环。"""

    st.subheader("引导式规则增强")
    st.caption(
        "候选字段只来自脱敏字段画像，不读取原始值。所有规则默认处于草案状态，"
        "只有明确批准后才会重新解析当前上传文件并执行。规则元数据不会发送给 DeepSeek。"
    )
    if report.status != "success":
        st.info("零配置评估成功后才能创建 RulePack。")
        return
    if uploaded_file is None:
        st.info("当前上传内容已不可用，请重新选择文件并运行零配置评估。")
        return

    guidance = build_rule_guidance(report)
    fields = list(guidance.required_field_candidates)
    state = _rule_state_for(report)
    if not state.get("guidance_started"):
        start_guidance = st.button(
            "开始配置业务规则",
            key="start_rule_guidance",
            width="stretch",
        )
        if not start_guidance:
            st.info("点击后才会显示规则问题；此操作不会启用或执行任何规则。")
            return
        state = {
            **state,
            "guidance_started": True,
        }
        st.session_state[RULE_STATE_KEY] = state
    with st.expander("查看本地 Agent 的确认问题与候选", expanded=True):
        for question in guidance.questions:
            st.text(question)
        st.caption(
            "主键候选："
            + ("、".join(guidance.primary_key_candidates) or "未自动识别")
        )
        st.caption(
            "更新时间候选："
            + ("、".join(guidance.update_time_candidates) or "未自动识别")
        )

    rule_pack_name = st.text_input(
        "规则包名称",
        value=f"{report.dataset.name}业务规则",
        max_chars=120,
        key="rule_pack_name",
    )
    rule_pack_version = st.text_input(
        "规则包版本",
        value="1.0.0",
        max_chars=32,
        key="rule_pack_version",
    )
    primary_key_fields = st.multiselect(
        "主键字段（可组合，最多 5 个）",
        options=fields,
        default=[],
        max_selections=5,
        key="rule_primary_key_fields",
    )
    required_fields = st.multiselect(
        "必填字段",
        options=fields,
        default=[],
        key="rule_required_fields",
    )
    update_time_field = st.selectbox(
        "更新时间字段",
        options=[None, *guidance.update_time_candidates],
        format_func=lambda value: "不启用" if value is None else str(value),
        key="rule_update_time_field",
    )
    frequency_label = st.selectbox(
        "更新频率",
        options=list(RULE_FREQUENCIES),
        index=2,
        disabled=update_time_field is None,
        key="rule_update_frequency",
    )
    custom_days = st.number_input(
        "自定义最长更新间隔（天）",
        min_value=1,
        max_value=3660,
        value=30,
        step=1,
        disabled=(
            update_time_field is None
            or frequency_label != "自定义天数"
        ),
        key="rule_update_custom_days",
    )
    allowed_values_field = st.selectbox(
        "允许值字段（可选）",
        options=[None, *fields],
        format_func=lambda value: "不启用" if value is None else str(value),
        key="rule_allowed_values_field",
    )
    allowed_values_json = st.text_area(
        "允许值 JSON 数组",
        value="[]",
        max_chars=5000,
        disabled=allowed_values_field is None,
        help='示例：["启用", "停用"]；缺失值请另设必填规则。',
        key="rule_allowed_values_json",
    )
    numeric_range_field = st.selectbox(
        "数值范围字段（可选）",
        options=[None, *guidance.numeric_field_candidates],
        format_func=lambda value: "不启用" if value is None else str(value),
        key="rule_numeric_range_field",
    )
    range_columns = st.columns(2)
    numeric_minimum_text = range_columns[0].text_input(
        "数值下限（闭区间，可留空）",
        disabled=numeric_range_field is None,
        key="rule_numeric_minimum",
    )
    numeric_maximum_text = range_columns[1].text_input(
        "数值上限（闭区间，可留空）",
        disabled=numeric_range_field is None,
        key="rule_numeric_maximum",
    )

    form_payload = {
        "name": rule_pack_name,
        "version": rule_pack_version,
        "primary_key_fields": primary_key_fields,
        "required_fields": required_fields,
        "update_time_field": update_time_field,
        "frequency_label": frequency_label,
        "custom_days": int(custom_days),
        "allowed_values_field": allowed_values_field,
        "allowed_values_json": allowed_values_json,
        "numeric_range_field": numeric_range_field,
        "numeric_minimum_text": numeric_minimum_text,
        "numeric_maximum_text": numeric_maximum_text,
    }
    current_signature = _rule_form_signature(form_payload)
    if (
        state.get("draft") is not None
        and state.get("draft_signature") != current_signature
    ):
        state = {
            "report_sha256": state["report_sha256"],
            "guidance_started": True,
            "draft": None,
            "draft_signature": None,
            "confirmed_draft_sha256": None,
            "approved_pack": None,
            "result": None,
            "execution_error": None,
        }
        _clear_rule_approval_widgets()
        st.session_state[RULE_STATE_KEY] = state
        st.info("规则答案已变化，旧草案、审批和增强结果已失效。")

    generate_draft = st.button(
        "生成并校验规则草案",
        key="generate_rule_pack_draft",
        width="stretch",
    )
    if generate_draft:
        try:
            rules = _rule_form_rules(
                primary_key_fields=primary_key_fields,
                required_fields=required_fields,
                update_time_field=update_time_field,
                frequency_label=frequency_label,
                custom_days=int(custom_days),
                allowed_values_field=allowed_values_field,
                allowed_values_json=allowed_values_json,
                numeric_range_field=numeric_range_field,
                numeric_minimum_text=numeric_minimum_text,
                numeric_maximum_text=numeric_maximum_text,
            )
            draft = build_rule_pack(
                report,
                name=rule_pack_name,
                version=rule_pack_version,
                rules=rules,
            )
            state = {
                "report_sha256": _report_sha256(report),
                "guidance_started": True,
                "draft": draft,
                "draft_signature": current_signature,
                "confirmed_draft_sha256": None,
                "approved_pack": None,
                "result": None,
                "execution_error": None,
            }
            _clear_rule_approval_widgets()
            st.session_state[RULE_STATE_KEY] = state
        except RulePackValidationError as error:
            st.error(_escape_markdown(str(error)))

    state = _rule_state_for(report)
    draft = state.get("draft")
    if draft is None:
        st.info("回答规则问题并生成草案后，才会出现审批入口。")
        return
    validation = validate_rule_pack(draft, report)
    if not validation.valid:
        st.error("规则草案未通过确定性校验，不能进入审批。")
        for error in validation.errors:
            st.text(error)
        return

    st.success(f"规则草案校验通过：{draft.rule_pack_id} v{draft.version}")
    with st.expander("查看待审批 RulePack"):
        st.json(draft.to_dict())
    draft_download_name = sanitize_file_name(
        f"{report.dataset.name}_rule_pack_draft.json",
        default_name="rule_pack_draft.json",
        safe_extension=".json",
    )
    st.download_button(
        "下载规则草案（JSON）",
        data=json.dumps(
            draft.to_dict(),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ).encode("utf-8"),
        file_name=draft_download_name,
        mime="application/json",
    )
    approver = st.text_input(
        "审批人标识（本地自声明）",
        max_chars=100,
        key="rule_approver",
    )
    confirmed = st.checkbox(
        "我已核对当前 RulePack，并明确批准其仅用于当前绑定输入的确定性重评。",
        key="rule_approval_confirmed",
        on_change=_bind_rule_confirmation,
    )
    state = _rule_state_for(report)
    current_draft_sha256 = validation.draft_sha256 or draft_sha256(draft)
    confirmation_matches = (
        confirmed
        and state.get("confirmed_draft_sha256")
        == current_draft_sha256
    )
    approve_and_run = st.button(
        "批准并重新评估",
        type="primary",
        width="stretch",
        disabled=not confirmation_matches or not approver.strip(),
        key="approve_and_run_rule_pack",
    )
    if approve_and_run:
        if (
            not confirmed
            or state.get("confirmed_draft_sha256")
            != current_draft_sha256
            or not approver.strip()
        ):
            st.error("当前草案尚未获得与其哈希绑定的明确确认，不能批准或执行。")
        else:
            try:
                approved_pack = approve_rule_pack(
                    draft,
                    report,
                    approver=approver,
                )
            except RulePackValidationError as error:
                st.error(_escape_markdown(f"规则审批失败：{error}"))
            else:
                st.session_state[RULE_STATE_KEY] = {
                    **state,
                    "approved_pack": approved_pack,
                    "result": None,
                    "execution_error": None,
                }
                try:
                    with st.spinner("正在重新解析当前输入并执行已审批业务规则……"):
                        result = evaluate_uploaded_dataset_with_rule_pack(
                            uploaded_file.getvalue(),
                            uploaded_file.name,
                            approved_pack,
                            dataset_name=dataset_name.strip() or None,
                            sheet_name=sheet_name.strip() or None,
                            reference_date=reference_date,
                            selected_metric_ids=selected_metric_ids,
                        )
                except RulePackExecutionError as error:
                    st.session_state[RULE_STATE_KEY] = {
                        **state,
                        "approved_pack": approved_pack,
                        "result": None,
                        "execution_error": str(error),
                    }
                    st.error(_escape_markdown(f"规则增强未执行：{error}"))
                except Exception:
                    failure_message = (
                        "规则增强未能完成；零配置报告不受影响，"
                        "请核对输入后重试。"
                    )
                    st.session_state[RULE_STATE_KEY] = {
                        **state,
                        "approved_pack": approved_pack,
                        "result": None,
                        "execution_error": failure_message,
                    }
                    st.error(failure_message)
                else:
                    st.session_state[RULE_STATE_KEY] = {
                        **state,
                        "approved_pack": approved_pack,
                        "result": result,
                        "execution_error": None,
                    }

    final_state = _rule_state_for(report)
    result = final_state.get("result")
    if result is None:
        approved_pack = final_state.get("approved_pack")
        if approved_pack is None:
            st.caption("草案尚未批准；零配置报告未发生变化。")
            return
        st.warning("RulePack 已批准，但确定性重评尚未成功；零配置报告不受影响。")
        execution_error = final_state.get("execution_error")
        if execution_error:
            st.text(str(execution_error))
        failed_approved_download_name = sanitize_file_name(
            f"{report.dataset.name}_approved_rule_pack.json",
            default_name="approved_rule_pack.json",
            safe_extension=".json",
        )
        st.download_button(
            "下载已审批 RulePack（JSON）",
            data=json.dumps(
                approved_pack.to_dict(),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ).encode("utf-8"),
            file_name=failed_approved_download_name,
            mime="application/json",
        )
        return
    _render_rule_result(result)

    approved_download_name = sanitize_file_name(
        f"{report.dataset.name}_approved_rule_pack.json",
        default_name="approved_rule_pack.json",
        safe_extension=".json",
    )
    result_download_name = sanitize_file_name(
        f"{report.dataset.name}_rule_evaluation.json",
        default_name="rule_evaluation.json",
        safe_extension=".json",
    )
    enhanced_markdown_name = sanitize_file_name(
        f"{report.dataset.name}_rule_enhanced_report.md",
        default_name="rule_enhanced_report.md",
        safe_extension=".md",
    )
    rule_locations_name = sanitize_file_name(
        f"{report.dataset.name}_rule_issue_locations.csv",
        default_name="rule_issue_locations.csv",
        safe_extension=".csv",
    )
    st.download_button(
        "下载已审批 RulePack（JSON）",
        data=json.dumps(
            result.approved_rule_pack.to_dict(),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ).encode("utf-8"),
        file_name=approved_download_name,
        mime="application/json",
    )
    st.download_button(
        "下载规则增强结果（JSON）",
        data=serialize_rule_evaluation_result(result),
        file_name=result_download_name,
        mime="application/json",
    )
    st.download_button(
        "下载规则增强报告（Markdown）",
        data=serialize_markdown_report(result.enhanced_report),
        file_name=enhanced_markdown_name,
        mime="text/markdown",
    )
    st.download_button(
        "下载规则问题位置（CSV）",
        data=serialize_rule_issue_locations_csv(result),
        file_name=rule_locations_name,
        mime="text/csv",
    )


def _render_status(report: QualityReport, summary: dict[str, int]) -> None:
    if report.status == "failed":
        st.error("文件未能成功解析。请查看下方运行信息中的具体原因。")
    elif summary["warning_count"]:
        st.warning(
            f"评估完成，发现 {summary['warning_count']} 项警告。"
            "风险提示不等同于数据错误，请结合业务规则复核。"
        )
    elif summary["attention_count"]:
        st.warning(
            f"评估完成，发现 {summary['attention_count']} 项需要关注的现象。"
        )
    else:
        st.success("评估完成，当前默认规则未发现警告或关注项。")


def _render_summary(report: QualityReport) -> dict[str, int]:
    summary = build_summary(report)
    _render_status(report, summary)
    columns = st.columns(5)
    columns[0].metric("记录数", f"{summary['row_count']:,}")
    columns[1].metric("字段数", f"{summary['column_count']:,}")
    columns[2].metric(
        "已评估指标",
        f"{summary['evaluated_metric_count']}/{summary['metric_count']}",
    )
    columns[3].metric("风险提示", f"{summary['risk_count']:,}")
    columns[4].metric("无法评估项", f"{summary['not_assessable_count']:,}")
    return summary


def _render_risks(report: QualityReport) -> None:
    st.subheader("风险分布")
    risk_chart = pd.DataFrame(build_risk_chart_rows(report)).set_index("级别")
    st.bar_chart(risk_chart, horizontal=True, height=220)

    if not report.risks:
        st.info("当前默认阈值下没有生成风险提示。")
        return

    for level in ("warning", "attention", "info"):
        risks = [risk for risk in report.risks if risk.level == level]
        if not risks:
            continue
        st.markdown(f"#### {RISK_LEVEL_LABELS[level]}（{len(risks)}）")
        for risk in risks:
            with st.expander(
                _escape_markdown(risk.title),
                expanded=level == "warning",
            ):
                st.text(risk.message)
                related = "、".join(
                    risk.related_metric_keys or risk.related_metrics
                ) or "无"
                st.caption(f"关联指标：{related}")


def _render_metrics(report: QualityReport) -> None:
    rows = build_metric_rows(report)
    if not rows:
        st.info("当前报告不包含指标明细。")
        return
    visible_rows = pd.DataFrame(rows).drop(
        columns=["来源", "标准代码", "层级"],
        errors="ignore",
    )
    st.dataframe(visible_rows, width="stretch", hide_index=True)
    st.caption("指标值由确定性 Python 规则计算；“无法评估”不会被替换为 0。")
    with st.expander("查看指标引用键（技术信息）"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "指标名称": metric.name,
                        "字段名称": metric.field or "—",
                        "引用键": metric.metric_key,
                    }
                    for metric in report.metrics
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def _render_profile(report: QualityReport) -> None:
    rows = build_profile_rows(report)
    if not rows:
        st.info("当前文件没有可展示的字段画像。")
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_execution(report: QualityReport) -> None:
    if report.not_assessable:
        st.markdown("#### 无法评估项")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "项目": item.name,
                        "原因": item.reason,
                    }
                    for item in report.not_assessable
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("本次报告没有无法评估项。")

    warnings = report.execution.get("warnings", [])
    errors = report.execution.get("errors", [])
    st.markdown("#### 运行信息")
    if not warnings and not errors:
        st.caption("本次运行没有解析警告或错误。")
    for message in warnings:
        st.warning(_escape_markdown(message))
    for message in errors:
        st.error(_escape_markdown(message))


def _evaluation_request_signature(
    uploaded_file,
    dataset_name: str,
    sheet_name: str,
    reference_date: date,
    selected_metric_ids: tuple[str, ...],
) -> tuple[str, str, str, str, str, tuple[str, ...]] | None:
    """标识当前评估请求，防止输入变化后继续展示旧报告。"""

    if uploaded_file is None:
        return None
    digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    return (
        uploaded_file.name,
        digest,
        dataset_name.strip(),
        sheet_name.strip(),
        reference_date.isoformat(),
        selected_metric_ids,
    )


st.title("政务数据集质量评估")
st.caption(
    "v0.6 · 上传结构化文件，选择评价指标并生成可复现报告。"
)

_initialize_metric_selection_state()
report_is_displayed = st.session_state.get("quality_report") is not None
if not report_is_displayed:
    selected_metric_ids = _render_metric_selection_panel()
else:
    selected_metric_ids = tuple(
        metric_id
        for metric_id in ALL_METRIC_IDS
        if metric_id in st.session_state.get(METRIC_SELECTION_KEY, [])
    )

with st.sidebar:
    st.header("开始评估")
    uploaded_file = st.file_uploader(
        "选择数据文件",
        type=[
            "csv",
            "xls",
            "xlsx",
            "json",
            "jsonl",
            "ndjson",
            "geojson",
            "zip",
        ],
        help=(
            "支持 CSV、Excel（.xls、.xlsx）、表格型 JSON "
            "、JSONL / NDJSON、GeoJSON FeatureCollection 以及同构 JSON 分片 ZIP；"
            f"单文件上限 {MAX_INPUT_FILE_MIB} MiB。"
        ),
    )
    dataset_name = st.text_input(
        "数据集名称（可选）",
        placeholder="默认使用文件名",
    )
    reference_date = st.date_input(
        "评估基准日期",
        value=date.today(),
        help="更新滞后天数以此日期为基准；固定该日期可复现同一份报告。",
    )
    sheet_name = ""
    if (
        uploaded_file
        and Path(uploaded_file.name).suffix.lower() in {".xls", ".xlsx"}
    ):
        sheet_name = st.text_input(
            "工作表名称（可选）",
            placeholder="默认读取第一个工作表",
        )
    request_signature = _evaluation_request_signature(
        uploaded_file,
        dataset_name,
        sheet_name,
        reference_date,
        selected_metric_ids,
    )
    if st.session_state.get("evaluation_request_signature") != request_signature:
        had_report = st.session_state.get("quality_report") is not None
        st.session_state["evaluation_request_signature"] = request_signature
        st.session_state.pop("quality_report", None)
        _clear_agent_state()
        _clear_rule_state()
        if had_report:
            st.rerun()
    if not report_is_displayed:
        run_evaluation = st.button(
            "运行质量评估",
            type="primary",
            width="stretch",
            disabled=uploaded_file is None or not selected_metric_ids,
        )
    else:
        run_evaluation = False
    st.caption("原始文件仅写入临时目录用于本次计算，评估结束后自动删除。")

if run_evaluation and uploaded_file is not None and selected_metric_ids:
    _clear_agent_state()
    _clear_rule_state()
    with st.spinner("正在解析文件并计算质量指标……"):
        try:
            st.session_state["quality_report"] = evaluate_uploaded_dataset(
                uploaded_file.getvalue(),
                uploaded_file.name,
                dataset_name=dataset_name.strip() or None,
                sheet_name=sheet_name.strip() or None,
                reference_date=reference_date,
                selected_metric_ids=selected_metric_ids,
            )
            st.rerun()
        except (DatasetReadError, UnsupportedFileTypeError) as error:
            st.session_state.pop("quality_report", None)
            _clear_agent_state()
            _clear_rule_state()
            st.error(_escape_markdown(f"评估未能启动：{error}"))
        except Exception:  # 防止界面中断，且不暴露本地路径等环境细节
            st.session_state.pop("quality_report", None)
            _clear_agent_state()
            _clear_rule_state()
            st.error("评估未能启动：运行环境或临时文件不可用，请重试。")

report = st.session_state.get("quality_report")
if report is None:
    st.info(
        "请从左侧上传 CSV、Excel、JSON、JSONL、GeoJSON 或 ZIP 文件。"
    )
else:
    _agent_state_for(report)
    _rule_state_for(report)
    dataset = report.dataset
    title = dataset.name
    details = f"文件：{dataset.file_name} · 类型：{dataset.file_type.upper()}"
    if dataset.sheet_name:
        details += f" · 工作表：{dataset.sheet_name}"
    st.subheader(_escape_markdown(title))
    st.caption(_escape_markdown(details))

    _render_summary(report)
    _render_metric_selection_summary(report)
    (
        risk_tab,
        metric_tab,
        profile_tab,
        execution_tab,
        agent_tab,
        rule_tab,
    ) = st.tabs(
        [
            "风险提示",
            "指标明细",
            "字段画像",
            "无法评估与运行信息",
            "Agent 解读",
            "规则增强",
        ]
    )
    with risk_tab:
        _render_risks(report)
    with metric_tab:
        _render_metrics(report)
    with profile_tab:
        _render_profile(report)
    with execution_tab:
        _render_execution(report)
    with agent_tab:
        _render_agent(report)
    with rule_tab:
        _render_rule_enhancement(
            report,
            uploaded_file=uploaded_file,
            dataset_name=dataset_name,
            sheet_name=sheet_name,
            reference_date=reference_date,
            selected_metric_ids=selected_metric_ids,
        )

    json_download_file_name = sanitize_file_name(
        f"{dataset.name}_quality_report.json",
        default_name="quality_report.json",
        safe_extension=".json",
    )
    markdown_download_file_name = sanitize_file_name(
        f"{dataset.name}_quality_report.md",
        default_name="quality_report.md",
        safe_extension=".md",
    )
    issue_locations_download_file_name = sanitize_file_name(
        f"{dataset.name}_quality_issue_locations.csv",
        default_name="quality_issue_locations.csv",
        safe_extension=".csv",
    )
    st.download_button(
        "下载结构化报告（JSON）",
        data=serialize_report(report),
        file_name=json_download_file_name,
        mime="application/json",
        type="primary",
    )
    st.download_button(
        "下载评估报告（Markdown）",
        data=serialize_markdown_report(report),
        file_name=markdown_download_file_name,
        mime="text/markdown",
    )
    st.download_button(
        "下载疑似问题位置（CSV）",
        data=serialize_issue_locations_csv(report),
        file_name=issue_locations_download_file_name,
        mime="text/csv",
    )
    st.caption(
        "JSON 报告用于系统对接，Markdown 报告适合直接阅读；"
        "疑似问题位置 CSV 包含各项指标定位到的全部记录序号，"
        "序号从 1 开始且不包含表头。"
        "三种下载均不包含原始字段样例。"
    )
