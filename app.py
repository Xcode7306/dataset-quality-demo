"""政务数据集质量评估的 Streamlit 网页入口。"""

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.models import QualityReport
from src.parser import DatasetReadError, UnsupportedFileTypeError
from src.presentation import (
    RISK_LEVEL_LABELS,
    build_metric_rows,
    build_profile_rows,
    build_risk_chart_rows,
    build_summary,
    serialize_report,
)
from src.resource_limits import MAX_INPUT_FILE_MIB
from src.upload_service import evaluate_uploaded_dataset, sanitize_file_name


st.set_page_config(
    page_title="政务数据集质量评估",
    page_icon="📊",
    layout="wide",
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
            with st.expander(risk.title, expanded=level == "warning"):
                st.write(risk.message)
                related = "、".join(risk.related_metrics) or "无"
                st.caption(f"关联指标：{related}")


def _render_metrics(report: QualityReport) -> None:
    rows = build_metric_rows(report)
    if not rows:
        st.info("当前报告不包含指标明细。")
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("指标值由确定性 Python 规则计算；“无法评估”不会被替换为 0。")


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
        st.warning(message)
    for message in errors:
        st.error(message)


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
        type=["csv", "xlsx", "json"],
        help=(
            "支持 CSV、Excel（.xlsx）和扁平记录型 JSON；"
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
    if uploaded_file and Path(uploaded_file.name).suffix.lower() == ".xlsx":
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
    run_evaluation = st.button(
        "运行质量评估",
        type="primary",
        width="stretch",
        disabled=uploaded_file is None,
    )
    st.caption("原始文件仅写入临时目录用于本次计算，评估结束后自动删除。")

if run_evaluation and uploaded_file is not None:
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
            st.error(f"评估未能启动：{error}")
        except Exception:  # 防止界面中断，且不暴露本地路径等环境细节
            st.session_state.pop("quality_report", None)
            st.error("评估未能启动：运行环境或临时文件不可用，请重试。")

report = st.session_state.get("quality_report")
if report is None:
    st.info("请从左侧上传 CSV、Excel 或扁平 JSON 文件，然后运行评估。")
else:
    dataset = report.dataset
    title = dataset.name
    details = f"文件：{dataset.file_name} · 类型：{dataset.file_type.upper()}"
    if dataset.sheet_name:
        details += f" · 工作表：{dataset.sheet_name}"
    st.subheader(title)
    st.caption(details)

    _render_summary(report)
    risk_tab, metric_tab, profile_tab, execution_tab = st.tabs(
        ["风险提示", "指标明细", "字段画像", "无法评估与运行信息"]
    )
    with risk_tab:
        _render_risks(report)
    with metric_tab:
        _render_metrics(report)
    with profile_tab:
        _render_profile(report)
    with execution_tab:
        _render_execution(report)

    download_file_name = sanitize_file_name(
        f"{dataset.name}_quality_report.json",
        default_name="quality_report.json",
        safe_extension=".json",
    )
    st.download_button(
        "下载 report.json",
        data=serialize_report(report),
        file_name=download_file_name,
        mime="application/json",
        type="primary",
    )
    st.caption(
        "下载报告不包含原始字段样例。"
        "未来 AI 解读层只读取该结构化报告，"
        "不参与指标计算或风险判定。"
    )
