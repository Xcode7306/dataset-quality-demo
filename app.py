"""政务数据集质量评估的 Streamlit 网页入口。"""

import hashlib
import os
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent_models import AgentAnalysis
from src.agent_service import run_agent
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
from src.upload_service import evaluate_uploaded_dataset, sanitize_file_name


AGENT_STATE_KEY = "agent_ui_state"
AGENT_HISTORY_LIMIT = 8
AGENT_PRIORITY_LABELS = {
    "high": "高",
    "medium": "中",
    "low": "低",
}


st.set_page_config(
    page_title="政务数据集质量评估",
    page_icon="📊",
    layout="wide",
)


def _clear_agent_state() -> None:
    """清除只属于当前确定性报告的 Agent 结果与问答记录。"""

    st.session_state.pop(AGENT_STATE_KEY, None)


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
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
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
) -> tuple[str, str, str, str, str] | None:
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
    )


st.title("政务数据集质量评估")
st.caption("上传结构化文件，生成可复现的质量指标、风险提示和无法评估项。")

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
    )
    if st.session_state.get("evaluation_request_signature") != request_signature:
        st.session_state["evaluation_request_signature"] = request_signature
        st.session_state.pop("quality_report", None)
        _clear_agent_state()
    run_evaluation = st.button(
        "运行质量评估",
        type="primary",
        width="stretch",
        disabled=uploaded_file is None,
    )
    st.caption("原始文件仅写入临时目录用于本次计算，评估结束后自动删除。")

if run_evaluation and uploaded_file is not None:
    _clear_agent_state()
    with st.spinner("正在解析文件并计算质量指标……"):
        try:
            st.session_state["quality_report"] = evaluate_uploaded_dataset(
                uploaded_file.getvalue(),
                uploaded_file.name,
                dataset_name=dataset_name.strip() or None,
                sheet_name=sheet_name.strip() or None,
                reference_date=reference_date,
            )
        except (DatasetReadError, UnsupportedFileTypeError) as error:
            st.session_state.pop("quality_report", None)
            _clear_agent_state()
            st.error(_escape_markdown(f"评估未能启动：{error}"))
        except Exception:  # 防止界面中断，且不暴露本地路径等环境细节
            st.session_state.pop("quality_report", None)
            _clear_agent_state()
            st.error("评估未能启动：运行环境或临时文件不可用，请重试。")

report = st.session_state.get("quality_report")
if report is None:
    st.info(
        "请从左侧上传 CSV、Excel、JSON、JSONL、GeoJSON 或 ZIP 文件。"
    )
else:
    _agent_state_for(report)
    dataset = report.dataset
    title = dataset.name
    details = f"文件：{dataset.file_name} · 类型：{dataset.file_type.upper()}"
    if dataset.sheet_name:
        details += f" · 工作表：{dataset.sheet_name}"
    st.subheader(_escape_markdown(title))
    st.caption(_escape_markdown(details))

    _render_summary(report)
    (
        risk_tab,
        metric_tab,
        profile_tab,
        execution_tab,
        agent_tab,
    ) = st.tabs(
        [
            "风险提示",
            "指标明细",
            "字段画像",
            "无法评估与运行信息",
            "Agent 解读",
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
