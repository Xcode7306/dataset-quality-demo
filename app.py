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
from streamlit.runtime.scriptrunner import get_script_run_ctx

from src.agent_models import AgentAnalysis
from src.agent_providers import (
    DEFAULT_DEEPSEEK_MODEL,
    DEEPSEEK_CHAT_COMPLETIONS_URL,
    DeepSeekChatProvider,
    OpenAICompatibleChatProvider,
)
from src.agent_service import run_agent
from src.credential_store import (
    CredentialStoreError,
    credential_store_available,
    delete_model_api_key,
    load_model_api_key,
    save_model_api_key,
)
from src.metric_catalog import (
    ALL_METRIC_IDS,
    DEFAULT_SELECTED_METRIC_IDS,
    build_metric_catalog_rows,
    default_evaluation_basis,
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
from src.rule_authoring_coordinator import (
    RuleAuthoringCoordinatorError,
    RuleAuthoringRun,
    approve_rule_authoring_run,
    begin_rule_authoring_run,
    compile_rule_authoring_run,
    dry_run_rule_authoring_run,
    execute_rule_authoring_run,
    retry_rule_authoring_run,
    validate_rule_authoring_run,
)
from src.rule_authoring_providers import (
    DeepSeekRuleAuthoringProvider,
    OpenAICompatibleRuleAuthoringProvider,
    build_rule_input_guidance,
)
from src.rule_authoring_workflow import RuleAuthoringHistory
from src.rule_authoring_tools import build_rule_authoring_context
from src.rule_batch import (
    MAX_RULE_IMPORT_BYTES,
    RuleBatchInput,
    RuleBatchPreflight,
    RuleImportError,
    SUPPORTED_RULE_IMPORT_EXTENSIONS,
    compile_rule_batch,
    parse_rule_import,
    rule_inputs_from_chat_messages,
)
from src.rag.citations import response_source_summary
from src.rag.models import (
    RAG_NAMESPACE_DATA_DICTIONARY,
    RAG_NAMESPACE_STANDARDS,
    RAG_NAMESPACE_USER_SPEC,
)
from src.rag.retrieval import (
    RagKnowledgeBase,
    RagRetrievalError,
    build_default_knowledge_base,
)
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
    dry_run_uploaded_dataset_with_rule_pack,
    evaluate_uploaded_dataset_with_rule_pack,
    serialize_rule_evaluation_markdown,
    serialize_rule_evaluation_result,
    serialize_rule_issue_locations_csv,
)
from src.upload_service import evaluate_uploaded_dataset, sanitize_file_name


AGENT_STATE_KEY = "agent_ui_state"
RULE_STATE_KEY = "rule_ui_state"
RULE_AUTHORING_STATE_KEY = "rule_authoring_ui_state"
CUSTOM_RULE_STATE_KEY = "custom_rule_ui_state"
RULE_WORKFLOW_HISTORY_KEY = "rule_authoring_workflow_history"
PRE_EVALUATION_RULE_STATE_KEY = "pre_evaluation_rule_state"
PRE_EVALUATION_RULE_RESULT_KEY = "pre_evaluation_rule_result"
RAG_STATE_KEY = "rag_ui_state"
METRIC_SELECTION_KEY = "selected_metric_ids"
METRIC_SELECTION_WIDGET_PREFIX = "metric_selection_checkbox_"
METRIC_EVIDENCE_WIDGET_PREFIX = "metric_evidence_"
PRE_EVALUATION_RULE_APPROVER_KEY = "pre_evaluation_rule_approver"
PRE_EVALUATION_RULE_CONFIRM_KEY = "pre_evaluation_rule_confirmed"
PRE_EVALUATION_IMPORT_VALUES_KEY = "pre_evaluation_import_rule_values"
PRE_EVALUATION_CHAT_MESSAGES_KEY = "pre_evaluation_rule_chat_messages"
PRE_EVALUATION_CHAT_ATTACHMENTS_KEY = "pre_evaluation_rule_chat_attachments"
MODEL_API_URL_KEY = "model_api_url"
MODEL_API_KEY_KEY = "model_api_key"
MODEL_API_KEY_INPUT_KEY = "model_api_key_input"
MODEL_API_KEY_SAVE_BUTTON_KEY = "save_model_api_key"
MODEL_API_KEY_CLEAR_BUTTON_KEY = "clear_model_api_key"
MODEL_API_KEY_STORE_MESSAGE_KEY = "model_api_key_store_message"
MODEL_NAME_KEY = "model_name"
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
    "custom_rule_approver",
    "custom_rule_approval_confirmed",
    PRE_EVALUATION_RULE_APPROVER_KEY,
    PRE_EVALUATION_RULE_CONFIRM_KEY,
)
RULE_FREQUENCIES = {
    "每日（1 天）": ("daily", 1),
    "每周（7 天）": ("weekly", 7),
    "每月（31 天）": ("monthly", 31),
    "每季度（92 天）": ("quarterly", 92),
    "每年（366 天）": ("yearly", 366),
    "自定义天数": ("custom", None),
}
RAG_NAMESPACE_LABELS = {
    RAG_NAMESPACE_STANDARDS: "标准文件",
    RAG_NAMESPACE_DATA_DICTIONARY: "数据字典",
    RAG_NAMESPACE_USER_SPEC: "用户规范",
}
RAG_ALL_NAMESPACE_LABEL = "全部已批准来源"
RULE_CHAT_FILE_TYPES = tuple(
    sorted(extension.lstrip(".") for extension in SUPPORTED_RULE_IMPORT_EXTENSIONS)
)

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


def _clear_rule_state(
    *,
    preserve_rag_binding: bool = False,
    preserve_chat: bool = False,
) -> None:
    """清除当前报告的规则草案、审批、增强结果及表单缓存。"""

    st.session_state.pop(RULE_STATE_KEY, None)
    st.session_state.pop(RULE_AUTHORING_STATE_KEY, None)
    st.session_state.pop(CUSTOM_RULE_STATE_KEY, None)
    st.session_state.pop(PRE_EVALUATION_RULE_STATE_KEY, None)
    st.session_state.pop(PRE_EVALUATION_RULE_RESULT_KEY, None)
    if not preserve_chat:
        st.session_state.pop(PRE_EVALUATION_CHAT_MESSAGES_KEY, None)
        st.session_state.pop(PRE_EVALUATION_CHAT_ATTACHMENTS_KEY, None)
        st.session_state.pop(PRE_EVALUATION_IMPORT_VALUES_KEY, None)
    rag_state = st.session_state.get(RAG_STATE_KEY)
    if isinstance(rag_state, dict):
        # 文档库属于当前会话的依据来源，不因更换业务数据而丢失。
        # 评估前已完成的检索绑定也可以继续用于本次规则编制；
        # 已有报告切换到新输入时则必须清掉，避免 RuleDraft 跨报告复用。
        if not preserve_rag_binding:
            rag_state.pop("response", None)
            rag_state.pop("bound_response", None)
            rag_state.pop("selected_chunk_ids", None)
        st.session_state[RAG_STATE_KEY] = rag_state
    for key in RULE_WIDGET_KEYS:
        st.session_state.pop(key, None)


def _rag_ui_state() -> dict:
    """返回绑定当前报告会话的本地 RAG 状态。"""

    state = st.session_state.get(RAG_STATE_KEY)
    if not isinstance(state, dict):
        state = {}
    knowledge_base = state.get("knowledge_base")
    if not isinstance(knowledge_base, RagKnowledgeBase):
        state["knowledge_base"] = build_default_knowledge_base()
    st.session_state[RAG_STATE_KEY] = state
    return state


def _current_rag_binding() -> tuple[object | None, tuple[str, ...]]:
    """返回已由用户明确绑定到下一次规则编制的 RAG 响应。"""

    state = st.session_state.get(RAG_STATE_KEY)
    if not isinstance(state, dict):
        return None, ()
    response = state.get("bound_response")
    chunk_ids = tuple(
        str(item)
        for item in state.get("selected_chunk_ids", ())
        if isinstance(item, str) and item
    )
    if response is None or not chunk_ids:
        return None, ()
    return response, chunk_ids


def _rag_binding_signature() -> dict[str, object]:
    """把来源绑定纳入规则草案签名，避免沿用旧版本依据。"""

    response, chunk_ids = _current_rag_binding()
    return {
        "rag_query": getattr(response, "query", None),
        "rag_status": getattr(response, "status", None),
        "rag_chunk_ids": list(chunk_ids),
    }


def _metric_checkbox_key(metric_id: str) -> str:
    """为每一张指标卡生成稳定且独立的复选框状态键。"""

    return f"{METRIC_SELECTION_WIDGET_PREFIX}{metric_id}"


def _metric_evidence_key(metric_id: str) -> str:
    """为指标卡下的评价依据输入生成稳定控件键。"""

    return f"{METRIC_EVIDENCE_WIDGET_PREFIX}{metric_id}"


def _initialize_model_api_state() -> None:
    """初始化页面模型配置；API Key 不从环境变量回填到输入框。"""

    if MODEL_API_URL_KEY not in st.session_state:
        st.session_state[MODEL_API_URL_KEY] = os.environ.get(
            "DEEPSEEK_API_URL",
            DEEPSEEK_CHAT_COMPLETIONS_URL,
        )
    if MODEL_API_KEY_KEY not in st.session_state:
        stored_key = ""
        context = get_script_run_ctx(suppress_warning=True)
        is_app_test = bool(context and context.session_id == "test session id")
        if credential_store_available() and not is_app_test:
            try:
                stored_key = load_model_api_key() or ""
            except CredentialStoreError as error:
                st.session_state[MODEL_API_KEY_STORE_MESSAGE_KEY] = (
                    "warning",
                    str(error),
                )
        st.session_state[MODEL_API_KEY_KEY] = stored_key
    if MODEL_API_KEY_INPUT_KEY not in st.session_state:
        st.session_state[MODEL_API_KEY_INPUT_KEY] = st.session_state[
            MODEL_API_KEY_KEY
        ]
    if MODEL_NAME_KEY not in st.session_state:
        st.session_state[MODEL_NAME_KEY] = os.environ.get(
            "DEEPSEEK_MODEL",
            DEFAULT_DEEPSEEK_MODEL,
        )


def _model_api_key_format_issue(api_key: str) -> str | None:
    """验证 Bearer Token 可以安全写入 HTTP 请求头。"""

    if not api_key:
        return "请先输入 API Key，再保存到当前会话。"
    if not api_key.isascii():
        return "API Key 只能包含 ASCII 字符，请勿粘贴中文说明、引号或其他标签。"
    if any(character.isspace() for character in api_key):
        return "API Key 不能包含空格、换行或其他空白字符。"
    if any(ord(character) < 33 or ord(character) > 126 for character in api_key):
        return "API Key 只能包含可显示的 ASCII 字符。"
    return None


def _save_model_api_key() -> None:
    """将密码输入框的值保存为当前 Streamlit 会话配置。"""

    candidate = str(
        st.session_state.get(MODEL_API_KEY_INPUT_KEY, "")
    ).strip()
    if _model_api_key_format_issue(candidate) is None:
        st.session_state[MODEL_API_KEY_KEY] = candidate
        context = get_script_run_ctx(suppress_warning=True)
        is_app_test = bool(context and context.session_id == "test session id")
        if credential_store_available() and not is_app_test:
            try:
                save_model_api_key(candidate)
            except CredentialStoreError as error:
                st.session_state[MODEL_API_KEY_STORE_MESSAGE_KEY] = (
                    "warning",
                    f"API Key 已保存在当前会话，但{error}",
                )
            else:
                st.session_state[MODEL_API_KEY_STORE_MESSAGE_KEY] = (
                    "success",
                    "API Key 已保存到 macOS 钥匙串，新页面会话会自动恢复。",
                )
        else:
            st.session_state[MODEL_API_KEY_STORE_MESSAGE_KEY] = (
                "info",
                "API Key 已保存在当前会话；当前环境未启用系统凭据库。",
            )


def _clear_model_api_key() -> None:
    """只在用户显式操作时删除页面会话中的 API Key。"""

    st.session_state[MODEL_API_KEY_KEY] = ""
    st.session_state[MODEL_API_KEY_INPUT_KEY] = ""
    context = get_script_run_ctx(suppress_warning=True)
    is_app_test = bool(context and context.session_id == "test session id")
    if credential_store_available() and not is_app_test:
        try:
            delete_model_api_key()
        except CredentialStoreError as error:
            st.session_state[MODEL_API_KEY_STORE_MESSAGE_KEY] = (
                "warning",
                f"当前会话中的 API Key 已清除，但{error}",
            )
        else:
            st.session_state[MODEL_API_KEY_STORE_MESSAGE_KEY] = (
                "success",
                "API Key 已从当前会话和 macOS 钥匙串清除。",
            )
    else:
        st.session_state[MODEL_API_KEY_STORE_MESSAGE_KEY] = (
            "success",
            "API Key 已从当前会话清除。",
        )


def _model_api_configuration() -> tuple[dict[str, str] | None, str | None]:
    """读取页面配置，并兼容旧版 DeepSeek 环境变量配置。"""

    api_url = str(st.session_state.get(MODEL_API_URL_KEY, "")).strip()
    api_key = str(st.session_state.get(MODEL_API_KEY_KEY, "")).strip()
    model = str(st.session_state.get(MODEL_NAME_KEY, "")).strip()
    if api_key:
        key_issue = _model_api_key_format_issue(api_key)
        if key_issue:
            return None, key_issue
        if not api_url or not model:
            return None, "请同时填写 API 地址、API Key 和模型名称。"
        if not api_url.startswith(("http://", "https://")):
            return None, "API 地址必须以 http:// 或 https:// 开头。"
        return {
            "api_url": api_url,
            "api_key": api_key,
            "model": model,
            "source": "page",
        }, None

    if os.environ.get("QUALITY_AGENT_PROVIDER", "").strip().casefold() == "deepseek":
        environment_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not environment_key:
            return {
                "api_url": api_url or DEEPSEEK_CHAT_COMPLETIONS_URL,
                "api_key": "",
                "model": model or DEFAULT_DEEPSEEK_MODEL,
                "source": "environment",
            }, "当前选择了 DeepSeek 外部模式，但尚未配置 API Key。"
        return {
            "api_url": os.environ.get(
                "DEEPSEEK_API_URL",
                DEEPSEEK_CHAT_COMPLETIONS_URL,
            ),
            "api_key": environment_key,
            "model": os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL,
            "source": "environment",
        }, None
    return None, None


def _render_model_api_settings() -> None:
    """在初始页面提供可选的自定义模型 API 配置。"""

    with st.expander("大模型 API 配置", expanded=False):
        st.text_input(
            "API 地址",
            key=MODEL_API_URL_KEY,
            persist_state="session",
            help=(
                "可填写完整的 /chat/completions 地址，也可填写服务根地址；"
                "系统会自动补全 /chat/completions。"
            ),
        )
        st.text_input(
            "API Key",
            type="password",
            key=MODEL_API_KEY_INPUT_KEY,
            persist_state="session",
            help=(
                "点击‘保存 API Key’后保存到当前会话；macOS 上同时使用"
                "当前用户钥匙串，不写入报告、审计信息或项目文件。"
            ),
        )
        st.caption(
            "多用户边界：本 Demo 没有登录或多用户隔离；macOS 同一操作系统用户的"
            "钥匙串会被该用户的浏览器会话共享。请勿在共享操作系统账号中保存不同用户的 API Key。"
        )
        st.text_input(
            "模型名称",
            key=MODEL_NAME_KEY,
            persist_state="session",
            help="填写该 API 对应的模型标识，例如 deepseek-v4-flash。",
        )
        key_input = str(
            st.session_state.get(MODEL_API_KEY_INPUT_KEY, "")
        ).strip()
        saved_key = str(st.session_state.get(MODEL_API_KEY_KEY, "")).strip()
        key_input_issue = (
            _model_api_key_format_issue(key_input) if key_input else None
        )
        store_message = st.session_state.get(MODEL_API_KEY_STORE_MESSAGE_KEY)
        keychain_retry_available = bool(
            key_input
            and key_input == saved_key
            and isinstance(store_message, tuple)
            and len(store_message) == 2
            and store_message[0] == "warning"
        )
        save_column, clear_column = st.columns(2)
        save_column.button(
            "保存 API Key",
            key=MODEL_API_KEY_SAVE_BUTTON_KEY,
            on_click=_save_model_api_key,
            disabled=(
                not key_input
                or (key_input == saved_key and not keychain_retry_available)
                or key_input_issue is not None
            ),
            width="stretch",
        )
        clear_column.button(
            "清除 API Key",
            key=MODEL_API_KEY_CLEAR_BUTTON_KEY,
            on_click=_clear_model_api_key,
            disabled=not (saved_key or key_input),
            width="stretch",
        )
        if (
            isinstance(store_message, tuple)
            and len(store_message) == 2
            and store_message[0] in {"success", "info", "warning"}
        ):
            getattr(st, store_message[0])(store_message[1])
        if key_input_issue:
            st.warning(key_input_issue)
        elif key_input != saved_key:
            st.info("新输入的 API Key 尚未保存；当前运行仍使用上一个已保存配置。")
        configuration, issue = _model_api_configuration()
        if issue and configuration and configuration.get("source") == "environment":
            st.info(
                "仍可使用页面输入覆盖环境变量配置；当前未检测到可用的环境变量 API Key。"
            )
        elif issue:
            st.warning(issue)
        elif configuration and configuration.get("source") == "page":
            st.success(
                "API Key 已保存到当前会话；上传或替换数据时会继续保留，"
                "点击 Agent 操作时将调用该模型。"
            )
        elif configuration:
            st.info("已使用部署环境中的 DeepSeek 配置。")
        else:
            st.caption(
                "未填写 API Key 时仅使用本地模板作暂行演示；正式运行请先配置模型。"
            )


def _build_agent_provider():
    configuration, issue = _model_api_configuration()
    if issue and (
        str(st.session_state.get(MODEL_API_KEY_KEY, "")).strip()
        or configuration is not None
    ):
        raise ValueError(issue)
    if not configuration or not configuration.get("api_key"):
        return None
    if configuration["source"] == "page":
        return OpenAICompatibleChatProvider(
            api_key=configuration["api_key"],
            api_url=configuration["api_url"],
            model=configuration["model"],
        )
    return DeepSeekChatProvider(
        api_key=configuration["api_key"],
        api_url=configuration["api_url"],
        model=configuration["model"],
    )


def _build_rule_authoring_provider():
    configuration, issue = _model_api_configuration()
    if issue and (
        str(st.session_state.get(MODEL_API_KEY_KEY, "")).strip()
        or configuration is not None
    ):
        raise ValueError(issue)
    if not configuration or not configuration.get("api_key"):
        return None
    if configuration["source"] == "page":
        return OpenAICompatibleRuleAuthoringProvider(
            api_key=configuration["api_key"],
            api_url=configuration["api_url"],
            model=configuration["model"],
        )
    return DeepSeekRuleAuthoringProvider(
        api_key=configuration["api_key"],
        api_url=configuration["api_url"],
        model=configuration["model"],
    )


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


def _initialize_metric_evidence_state() -> None:
    """初始化评价依据输入；已有规则显示默认值，其余指标保持空白。"""

    for metric_id in ALL_METRIC_IDS:
        key = _metric_evidence_key(metric_id)
        if key not in st.session_state:
            st.session_state[key] = default_evaluation_basis(metric_id)


def _initialize_pre_evaluation_rule_input_state() -> None:
    """Keep optional rule text available when Streamlit temporarily hides widgets."""

    messages = st.session_state.get(PRE_EVALUATION_CHAT_MESSAGES_KEY)
    if not isinstance(messages, list):
        st.session_state[PRE_EVALUATION_CHAT_MESSAGES_KEY] = []
    attachments = st.session_state.get(PRE_EVALUATION_CHAT_ATTACHMENTS_KEY)
    if not isinstance(attachments, list):
        st.session_state[PRE_EVALUATION_CHAT_ATTACHMENTS_KEY] = []
    saved_imports = st.session_state.get(PRE_EVALUATION_IMPORT_VALUES_KEY, {})
    if isinstance(saved_imports, dict):
        for item_id, value in saved_imports.items():
            key = f"pre_import_rule_text_{item_id}"
            if key not in st.session_state and isinstance(value, str):
                st.session_state[key] = value


def _preserve_hidden_pre_evaluation_widget_state() -> None:
    """Keep optional rule text available while the report view hides its widgets.

    Streamlit normally removes widget-backed keys when a widget is not rendered
    on the next run.  The AppTest harness (and a browser event already queued
    against the previous page) can still hold the old widget node, so preserve
    these values explicitly before rendering the report view.
    """

    for item_id, value in dict(
        st.session_state.get(PRE_EVALUATION_IMPORT_VALUES_KEY, {})
    ).items():
        key = f"pre_import_rule_text_{item_id}"
        if isinstance(value, str):
            st.session_state[key] = value


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
        else "需补充评价标准"
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
        st.text_area(
            "评价依据 / 补充规则",
            key=_metric_evidence_key(metric_id),
            height=82,
            max_chars=4000,
            help=(
                "直接填写该指标的评估标准或补充规则；字段、阈值、允许值、频率、"
                "正则和比较条件会在最终评估前由 AI 检查。"
            ),
        )


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
        "右上角的“？”上可查看指标含义。每张卡片下方的“评价依据 / 补充规则”"
        "是该指标唯一的规则补充入口；勾选但未补全评价依据的指标不能启动评估。"
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
    else:
        missing_metric_evidence_ids = _missing_metric_evidence_ids(
            selected_metric_ids
        )
        if missing_metric_evidence_ids:
            missing_names = "、".join(
                str(get_metric_definition(metric_id)["name"])
                for metric_id in missing_metric_evidence_ids
                if get_metric_definition(metric_id) is not None
            )
            st.warning(
                "已勾选指标缺少评价依据，请补全评价规则后再运行："
                f"{missing_names}"
            )

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


def _render_rule_chat_input(
    *,
    header_slot=None,
    chat_shell=None,
) -> tuple[
    tuple[RuleBatchInput, ...],
    tuple[str, ...],
    tuple[str, ...],
    bool,
]:
    """Render the top-of-page rule chat and its integrated file attachment flow."""

    requests: list[RuleBatchInput] = []
    errors: list[str] = []
    warnings: list[str] = []

    if header_slot is None:
        st.markdown("## 与大模型对话创建规则")
        st.caption(
            "描述你想创建的规则；可以连续补充字段、阈值、允许值、频率或比较条件。"
            "点击输入框左侧的“＋”或直接拖拽规则文件，可批量导入规则。"
        )
    else:
        with header_slot.container():
            st.markdown("## 与大模型对话创建规则")
            st.caption(
                "描述你想创建的规则；可以连续补充字段、阈值、允许值、频率或比较条件。"
                "点击输入框左侧的“＋”或直接拖拽规则文件，可批量导入规则。"
            )
    chat_messages = list(
        st.session_state.get(PRE_EVALUATION_CHAT_MESSAGES_KEY, [])
    )
    for message in chat_messages:
        role = message.get("role") if isinstance(message, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if role not in {"user", "assistant"} or not content:
            continue
        with st.chat_message(role):
            st.text(str(content))
    # ``st.chat_input`` is pinned to the viewport bottom when rendered
    # directly in the main container.  Put it in a horizontal container so
    # Streamlit renders it inline, immediately below the guidance above.
    if chat_shell is None:
        chat_shell = st.empty()
    with chat_shell.container(horizontal=True, gap="small"):
        chat_value = st.chat_input(
            "描述你想创建的规则",
            key="pre_evaluation_rule_chat_input",
            max_chars=4000,
            accept_file="multiple",
            file_type=RULE_CHAT_FILE_TYPES,
            max_upload_size=max(1, MAX_RULE_IMPORT_BYTES // (1024 * 1024)),
            width="stretch",
        )
    if isinstance(chat_value, str):
        chat_text = chat_value.strip()
        attached_files = []
    elif chat_value is not None:
        chat_text = str(getattr(chat_value, "text", "") or "").strip()
        attached_files = list(getattr(chat_value, "files", []) or [])
    else:
        chat_text = ""
        attached_files = []
    chat_submitted = bool(chat_text or attached_files)
    if chat_text:
        chat_messages.append({"role": "user", "content": chat_text})
    if attached_files:
        attachment_names = [
            str(getattr(item, "name", "规则文件")) for item in attached_files
        ]
        chat_messages.append(
            {
                "role": "user",
                "kind": "attachment",
                "content": "已附加规则文件：" + "、".join(attachment_names),
            }
        )
        existing_attachments = list(
            st.session_state.get(PRE_EVALUATION_CHAT_ATTACHMENTS_KEY, [])
        )
        for item in attached_files:
            existing_attachments.append(
                {
                    "name": str(getattr(item, "name", "规则文件")),
                    "content": item.getvalue(),
                }
            )
        st.session_state[PRE_EVALUATION_CHAT_ATTACHMENTS_KEY] = existing_attachments
    if chat_text or attached_files:
        st.session_state[PRE_EVALUATION_CHAT_MESSAGES_KEY] = chat_messages

    if chat_messages:
        if st.button(
            "清空规则对话",
            key="clear_pre_evaluation_rule_chat",
            width="stretch",
        ):
            st.session_state[PRE_EVALUATION_CHAT_MESSAGES_KEY] = []
            st.session_state[PRE_EVALUATION_CHAT_ATTACHMENTS_KEY] = []
            st.session_state[PRE_EVALUATION_IMPORT_VALUES_KEY] = {}
            st.rerun()

    try:
        requests.extend(
            rule_inputs_from_chat_messages(
                st.session_state.get(PRE_EVALUATION_CHAT_MESSAGES_KEY, [])
            )
        )
    except RuleImportError as error:
        errors.append(str(error))

    attachment_items = list(
        st.session_state.get(PRE_EVALUATION_CHAT_ATTACHMENTS_KEY, [])
    )
    for attachment in attachment_items:
        source_name = str(attachment.get("name", "规则文件"))
        try:
            imported = parse_rule_import(attachment.get("content", b""), source_name)
        except RuleImportError as error:
            errors.append(f"规则文件“{source_name}”无法导入：{error}")
            continue
        warnings.extend(imported.warnings)
        st.caption(
            f"已附加：{source_name} · 识别 {len(imported.items)} 条规则；可在生成前逐条修改。"
        )
        with st.expander(f"检查并编辑：{source_name}", expanded=True):
            for item in imported.items:
                edit_key = f"pre_import_rule_text_{item.item_id}"
                if edit_key not in st.session_state:
                    st.session_state[edit_key] = item.user_intent
                st.text_area(
                    item.label,
                    key=edit_key,
                    height=72,
                    max_chars=4000,
                    help=(
                        f"目标指标：{item.target_metric_id}"
                        if item.target_metric_id
                        else "未指定指标时按自定义规则生成。"
                    ),
                )
                edited_intent = str(st.session_state.get(edit_key, "")).strip()
                saved_import_values = dict(
                    st.session_state.get(PRE_EVALUATION_IMPORT_VALUES_KEY, {})
                )
                saved_import_values[item.item_id] = edited_intent
                st.session_state[PRE_EVALUATION_IMPORT_VALUES_KEY] = saved_import_values
                try:
                    requests.append(
                        RuleBatchInput.create(
                            origin="file_import",
                            user_intent=edited_intent,
                            label=item.label,
                            target_metric_id=item.target_metric_id,
                            source_name=item.source_name,
                            source_location=item.source_location,
                        )
                    )
                except RuleImportError as error:
                    errors.append(f"{item.label}：{error}")

    return (
        tuple(requests),
        tuple(dict.fromkeys(errors)),
        tuple(dict.fromkeys(warnings)),
        chat_submitted,
    )


def _render_pre_evaluation_rule_inputs(
    selected_metric_ids: tuple[str, ...],
) -> tuple[
    tuple[RuleBatchInput, ...],
    tuple[str, ...],
    tuple[str, ...],
    bool,
]:
    """Collect rule inputs from card criteria, the chat, and file attachments.

    The card's ``评价依据 / 补充规则`` field is the only metric-specific
    supplement entry point.  Deterministic default text is left to the normal
    evaluator; user-entered or edited criteria become pre-evaluation RuleBatch
    inputs and are checked before the final report is allowed to start.
    """

    requests: list[RuleBatchInput] = []
    errors: list[str] = []
    warnings: list[str] = []

    for metric_id in selected_metric_ids:
        definition = get_metric_definition(metric_id)
        if definition is None:
            continue
        intent = _metric_evidence_text(metric_id)
        default_basis = default_evaluation_basis(metric_id).strip()
        if not intent or intent == default_basis:
            continue
        try:
            requests.append(
                RuleBatchInput.create(
                    origin="metric_supplement",
                    user_intent=intent,
                    label=f"指标卡片：{definition['name']}",
                    target_metric_id=metric_id,
                    source_location=f"metric:{metric_id}:evidence",
                )
            )
        except RuleImportError as error:
            errors.append(f"{definition['name']}：{error}")

    if requests:
        st.info(
            f"已从指标卡片下方识别 {len(requests)} 条待生成规则。"
            "点击左侧“AI 检查并生成规则”后，完整性问题会在本页立即列出。"
        )
    return (
        tuple(requests),
        tuple(dict.fromkeys(errors)),
        tuple(dict.fromkeys(warnings)),
        False,
    )


def _pre_evaluation_rule_signature(
    evaluation_signature: object,
    requests: tuple[RuleBatchInput, ...],
) -> str:
    return _rule_form_signature(
        {
            "evaluation": evaluation_signature,
            "rules": [item.to_dict() for item in requests],
            "model": _model_request_binding(include_environment=True),
            **_rag_binding_signature(),
        }
    )


def _model_request_binding(*, include_environment: bool) -> dict[str, object]:
    """返回不含明文密钥的模型配置绑定，用于状态哈希。

    规则预检需要绑定最终生效的环境变量配置；已生成报告的生命周期只绑定
    页面上的模型设置，避免用户临时切换 Agent 环境变量时丢失一份与数据
    输入无关的基础报告。
    """

    configuration = None
    issue = None
    if include_environment:
        configuration, issue = _model_api_configuration()
    saved_key = str(st.session_state.get(MODEL_API_KEY_KEY, "")).strip()
    effective_key = str(
        configuration.get("api_key")
        if configuration is not None
        else saved_key
    ).strip()
    if configuration is None:
        source = "page" if saved_key else None
        api_url = str(st.session_state.get(MODEL_API_URL_KEY, "")).strip()
        model = str(st.session_state.get(MODEL_NAME_KEY, "")).strip()
    else:
        source = configuration.get("source")
        api_url = configuration.get("api_url")
        model = configuration.get("model")
    return {
        "issue": issue,
        "source": source,
        "api_url": api_url,
        "model": model,
        "api_key_sha256": (
            hashlib.sha256(effective_key.encode("utf-8")).hexdigest()
            if effective_key
            else None
        ),
    }


def _render_pre_evaluation_rule_state(
    *,
    signature: str,
    uploaded_file,
    dataset_name: str,
    sheet_name: str,
    reference_date: date,
    selected_metric_ids: tuple[str, ...],
) -> None:
    """Show batch clarification/preview and execute only after local approval."""

    state = st.session_state.get(PRE_EVALUATION_RULE_STATE_KEY)
    if not isinstance(state, dict) or state.get("signature") != signature:
        return
    if state.get("error"):
        st.error(
            "规则生成预检未完成："
            f"{_escape_markdown(state['error'])}"
        )
    preflight = state.get("preflight")
    if not isinstance(preflight, RuleBatchPreflight):
        return

    st.markdown("### AI 规则生成预检结果")
    rows = []
    for item in preflight.items:
        draft = item.draft
        rows.append(
            {
                "来源": item.request.label,
                "目标指标": item.request.target_metric_id or "自定义规则",
                "状态": item.status,
                "生成类型": (
                    draft.rule_spec.rule_type
                    if draft is not None and draft.rule_spec is not None
                    else "—"
                ),
                "需处理内容": "；".join(item.messages) or "—",
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    blocking = [item for item in preflight.items if item.status != "ready"]
    if blocking:
        st.warning(
            f"有 {len(blocking)} 条规则描述不完整或不可执行，最终评估尚未启动。"
        )
        for item in blocking:
            st.markdown(f"**{_escape_markdown(item.request.label)}**")
            for message in item.messages:
                st.text(f"需要补充：{message}")
        st.caption("请直接修改上方对应描述，再重新点击“AI 检查并生成规则”。")
        return
    if not preflight.ready or preflight.draft_pack is None:
        for warning in preflight.warnings:
            st.error(_escape_markdown(warning))
        return

    st.success(
        f"{len(preflight.items)} 条描述均已生成并通过确定性校验；"
        f"合并后共 {len(preflight.draft_pack.rules)} 条可执行规则。"
    )
    for warning in preflight.warnings:
        st.warning(_escape_markdown(warning))
    with st.expander("查看合并后的 RulePack 草案", expanded=True):
        st.json(preflight.draft_pack.to_dict())
    preview = state.get("preview")
    if preview is None:
        st.error("规则尚未完成确定性试运行，不能审批或启动最终评估。")
        return
    _render_rule_dry_run(preview.to_dict())

    approved_pack = state.get("approved_pack")
    approver = st.text_input(
        "审批人标识（评估前 AI 规则，本地自声明）",
        key=PRE_EVALUATION_RULE_APPROVER_KEY,
        max_chars=100,
    )
    confirmed = st.checkbox(
        "我已核对全部生成规则和试运行摘要，并批准将其用于本次评估。",
        key=PRE_EVALUATION_RULE_CONFIRM_KEY,
    )
    pack_hash = draft_sha256(preflight.draft_pack)
    approve_clicked = st.button(
        "批准规则并运行质量评估"
        if approved_pack is None
        else "重试已批准规则的质量评估",
        key="approve_pre_evaluation_rule_batch",
        type="primary",
        width="stretch",
        disabled=(
            uploaded_file is None
            or not approver.strip()
            or not confirmed
            or state.get("draft_sha256") != pack_hash
        ),
    )
    if not approve_clicked:
        return
    report = state.get("report")
    if not isinstance(report, QualityReport):
        st.error("规则预检报告已失效，请重新生成规则。")
        return
    try:
        if approved_pack is None:
            approved_pack = approve_rule_pack(
                preflight.draft_pack,
                report,
                approver=approver,
            )
        with st.spinner("正在重新解析数据并执行全部已批准规则……"):
            result = evaluate_uploaded_dataset_with_rule_pack(
                uploaded_file.getvalue(),
                uploaded_file.name,
                approved_pack,
                dataset_name=dataset_name.strip() or None,
                sheet_name=sheet_name.strip() or None,
                reference_date=reference_date,
                selected_metric_ids=selected_metric_ids,
            )
    except (
        DatasetReadError,
        UnsupportedFileTypeError,
        RulePackValidationError,
        RulePackExecutionError,
        ValueError,
    ) as error:
        st.session_state[PRE_EVALUATION_RULE_STATE_KEY] = {
            **state,
            "approved_pack": approved_pack,
            "error": _model_error_detail(error),
        }
        st.error(
            "已批准规则的评估未完成："
            f"{_escape_markdown(_model_error_detail(error))}"
        )
        return
    st.session_state[PRE_EVALUATION_RULE_RESULT_KEY] = result
    st.session_state[PRE_EVALUATION_RULE_STATE_KEY] = {
        **state,
        "approved_pack": approved_pack,
        "result": result,
        "error": None,
    }
    st.session_state["quality_report"] = result.enhanced_report
    st.rerun()


def _escape_markdown(value: object) -> str:
    """转义会触发 Markdown 链接、图片或 HTML 的不可信展示文本。"""

    text = str(value)
    for character in ("\\", "`", "*", "_", "[", "]", "(", ")", "<", ">", "#", "!", "|"):
        text = text.replace(character, f"\\{character}")
    return text


def _model_error_detail(error: Exception) -> str:
    """返回可展示的模型错误，并隐藏当前会话中的 API Key。"""

    detail = str(error).strip() or "模型没有返回可识别的结果。"
    secrets = {
        str(st.session_state.get(MODEL_API_KEY_KEY, "")).strip(),
        str(os.environ.get("DEEPSEEK_API_KEY", "")).strip(),
    }
    for secret in secrets:
        if secret:
            detail = detail.replace(secret, "[已隐藏]")
    return detail[:800]


def _rule_missing_guidance(
    report: QualityReport,
    metric_id: str,
    user_intent: str,
) -> tuple[str, ...]:
    """生成针对当前指标的缺失规则提示，供模型失败时直接补充。"""

    try:
        context = build_rule_authoring_context(report, metric_id)
        return build_rule_input_guidance(
            context,
            user_intent=user_intent,
        )
    except Exception:
        return (
            "请补充字段名称、规则条件，以及允许值、更新时间或数值范围等可执行参数。",
        )


def _render_rule_missing_guidance(
    report: QualityReport,
    metric_id: str,
    user_intent: str,
) -> None:
    """在规则编译失败后告诉用户还需要补充什么。"""

    st.warning("当前补充评价依据还不能转换为可执行规则，请补充以下信息：")
    for item in _rule_missing_guidance(report, metric_id, user_intent):
        st.text(f"• {item}")


def _report_sha256(report: QualityReport) -> str:
    """读取由结构化报告规范化计算出的稳定哈希。"""

    return str(
        report.to_dict()
        .get("evaluation_context", {})
        .get("report_sha256", "")
    )


def _rule_workflow_history() -> RuleAuthoringHistory:
    """返回当前浏览器会话内、最多 20 条的工作流摘要历史。"""

    history = st.session_state.get(RULE_WORKFLOW_HISTORY_KEY)
    if not isinstance(history, RuleAuthoringHistory):
        history = RuleAuthoringHistory()
        st.session_state[RULE_WORKFLOW_HISTORY_KEY] = history
    return history


def _store_rule_authoring_run(run: RuleAuthoringRun) -> None:
    """只把脱敏工作流摘要写入当前会话历史，不保存上传字节。"""

    st.session_state[RULE_WORKFLOW_HISTORY_KEY] = _rule_workflow_history().upsert(
        run.workflow
    )


def _render_rule_workflow_status(run: RuleAuthoringRun) -> None:
    """展示由本地代码控制的状态与完整转换链。"""

    workflow = run.workflow
    state_labels = {
        "collecting": "收集请求",
        "retrieving": "检索依据",
        "needs_clarification": "等待补充",
        "compiling": "编译规则",
        "draft": "规则草案",
        "validated": "校验通过",
        "dry_run_complete": "试运行完成",
        "awaiting_approval": "等待审批",
        "approved": "已审批",
        "executed": "已执行",
        "rejected": "已拒绝",
        "failed": "执行失败",
    }
    st.caption(
        "v1.0 工作流 · "
        f"{state_labels.get(workflow.state, workflow.state)} · "
        f"重试 {workflow.retry_count}/1 · {workflow.workflow_id}"
    )
    with st.expander(
        "查看工作流状态与恢复记录",
        expanded=workflow.state == "failed",
    ):
        st.json(workflow.to_dict())
        st.caption("状态由本地确定性代码推进；模型不能审批、执行或选择恢复点。")


def _render_rule_workflow_history() -> None:
    """展示并导出当前会话的有界摘要历史。"""

    history = _rule_workflow_history()
    with st.expander("规则工作流历史（当前会话）"):
        st.caption(
            "最多保留 20 条脱敏摘要；不包含原始上传内容、API 密钥，关闭会话后不持久化。"
        )
        if not history.records:
            st.info("当前会话还没有规则工作流记录。")
            return
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "更新时间": item.updated_at,
                        "目标": item.target_metric_id or "自定义规则",
                        "状态": item.state,
                        "重试": f"{item.retry_count}/1",
                        "工作流": item.workflow_id,
                    }
                    for item in reversed(history.records)
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "下载工作流历史（JSON）",
            data=json.dumps(
                history.to_dict(),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ).encode("utf-8"),
            file_name="rule_authoring_history_v1.0.json",
            mime="application/json",
            key="download_rule_workflow_history_v10",
        )
        if st.button(
            "清空当前会话工作流历史",
            key="clear_rule_workflow_history_v10",
        ):
            st.session_state[RULE_WORKFLOW_HISTORY_KEY] = history.clear()
            st.success("当前会话工作流历史已清空。")


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
    standard_metric_ids = tuple(
        metric_id
        for metric_id in selected
        if (
            (definition := get_metric_definition(metric_id)) is not None
            and not bool(definition.get("auto_assessable"))
        )
    )
    auto_assessable_count = len(selected) - len(standard_metric_ids)
    st.caption(
        "本次指标选择："
        f"共 {len(selected)} 项 · 当前可直接计算 {auto_assessable_count} 项 · "
        f"需补充评价标准 {len(standard_metric_ids)} 项"
    )
    if standard_metric_ids:
        st.warning(
            "以下指标需要补充评价标准；请进入“补充评价标准”页面填写所需依据，"
            "再由规则 Agent 解析、试运行并重新评估："
        )
        for metric_id in standard_metric_ids:
            definition = get_metric_definition(metric_id)
            if definition is None:
                continue
            required_inputs = tuple(
                str(item) for item in definition.get("required_inputs", ())
            )
            requirement = "、".join(required_inputs) or "具体评价口径"
            st.markdown(
                f"- **{_escape_markdown(definition['name'])}**：需要提供"
                f"{_escape_markdown(requirement)}。"
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


def _rule_authoring_state_for(report: QualityReport) -> dict:
    """返回绑定当前报告的 v1.0 指标规则编制状态。"""

    report_sha256 = _report_sha256(report)
    state = st.session_state.get(RULE_AUTHORING_STATE_KEY)
    if not isinstance(state, dict) or state.get("report_sha256") != report_sha256:
        state = {
            "report_sha256": report_sha256,
            "runs": {},
            "drafts": {},
            "draft_signatures": {},
            "validations": {},
            "dry_runs": {},
            "approved_packs": {},
            "results": {},
            "execution_errors": {},
            "confirmed_pack_sha256": {},
        }
        st.session_state[RULE_AUTHORING_STATE_KEY] = state
    return state


def _metric_evidence_text(metric_id: str) -> str:
    value = st.session_state.get(_metric_evidence_key(metric_id), "")
    return value.strip() if isinstance(value, str) else ""


def _missing_metric_evidence_ids(
    metric_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """返回已选择但尚未填写评价依据的指标。"""

    return tuple(
        metric_id for metric_id in metric_ids if not _metric_evidence_text(metric_id)
    )


def _reset_metric_authoring_state(
    state: dict,
    metric_id: str,
    *,
    draft_signature: str | None = None,
) -> dict:
    """当用户修改评价依据时清除该指标的后续节点。"""

    updated = dict(state)
    for collection_name in (
        "runs",
        "drafts",
        "validations",
        "dry_runs",
        "approved_packs",
        "results",
        "execution_errors",
        "confirmed_pack_sha256",
    ):
        collection = dict(updated.get(collection_name, {}))
        collection.pop(metric_id, None)
        updated[collection_name] = collection
    signatures = dict(updated.get("draft_signatures", {}))
    if draft_signature is None:
        signatures.pop(metric_id, None)
    else:
        signatures[metric_id] = draft_signature
    updated["draft_signatures"] = signatures
    return updated


def _sync_metric_authoring_run(
    state: dict,
    metric_id: str,
    run: RuleAuthoringRun,
    *,
    draft_signature: str,
) -> dict:
    """把 v1.0 run 同步到兼容旧页面测试的各个视图字段。"""

    updated = _reset_metric_authoring_state(
        state,
        metric_id,
        draft_signature=draft_signature,
    )
    mappings = {
        "runs": run,
        "drafts": run.workflow.draft,
        "validations": run.workflow.validation,
        "dry_runs": run.preview.to_dict() if run.preview is not None else None,
        "approved_packs": run.approved_pack,
        "results": run.result,
        "execution_errors": (
            run.workflow.error
            if run.workflow.state in {"failed", "needs_clarification", "rejected"}
            else None
        ),
    }
    for collection_name, value in mappings.items():
        collection = dict(updated.get(collection_name, {}))
        if value is None:
            collection.pop(metric_id, None)
        else:
            collection[metric_id] = value
        updated[collection_name] = collection
    _store_rule_authoring_run(run)
    return updated


def _custom_authoring_state(
    *,
    signature: str,
    run: RuleAuthoringRun,
    error: str | None = None,
) -> dict:
    """构造 v1.0 自定义规则状态，并保留旧字段兼容。"""

    _store_rule_authoring_run(run)
    return {
        "signature": signature,
        "run": run,
        "draft": run.workflow.draft,
        "dry_run": run.preview.to_dict() if run.preview is not None else None,
        "approved_pack": run.approved_pack,
        "result": run.result,
        "error": error or (
            run.workflow.error
            if run.workflow.state in {"failed", "needs_clarification", "rejected"}
            else None
        ),
    }


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
        st.info("当前使用本地模板暂行演示模式；此结果不是外部模型生成的报告。")
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
        provider = _build_agent_provider()
        agent_kwargs = {
            "intent": intent,
            "question": question,
        }
        if provider is not None:
            agent_kwargs["provider"] = provider
            agent_kwargs["allow_template_fallback"] = False
        analysis = run_agent(report, **agent_kwargs)
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
    configuration, configuration_issue = _model_api_configuration()
    if configuration and configuration.get("source") == "page":
        st.warning(
            "当前已配置自定义大模型 API。点击快捷入口或提交问题时，"
            "会发送经过白名单过滤的报告投影；不发送原始单元格值。"
        )
    elif configuration and configuration.get("source") == "environment":
        if configuration.get("api_key"):
            st.warning(
                "当前部署已配置 DeepSeek 外部模式。点击快捷入口或提交问题时，"
                "会发送经过白名单过滤的报告投影；不发送原始单元格值。"
            )
        else:
            st.warning(
                "当前部署已选择 DeepSeek 外部模式，但尚未配置 "
                "DEEPSEEK_API_KEY；调用前必须补充 API Key，不会回退到本地模板。"
            )
    elif configuration_issue:
        st.warning(configuration_issue)
    else:
        st.caption(
            "当前未配置 API Key，仅使用本地模板作为暂行演示；正式使用 Agent 前请完成模型配置。"
        )
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

    request_failed = False
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
    except Exception as error:
        request_failed = True
        st.error(
            "外部模型解读失败，未生成模板替代结果："
            f"{_escape_markdown(_model_error_detail(error))}"
        )

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
        if not request_failed:
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


def _render_rag_panel(
    *,
    selected_metric_ids: tuple[str, ...],
) -> None:
    """渲染项目预置标准依据的检索、冲突提示和引用绑定。"""

    st.subheader("标准依据 RAG（v0.9）")
    st.caption(
        "仅从项目预置的标准、数据字典和用户规范中检索。"
        "结果带文档、版本、条款/章节和稳定 chunk ID；没有可定位来源时，不会形成标准合规依据。"
    )
    state = _rag_ui_state()
    knowledge_base = state["knowledge_base"]

    summary = knowledge_base.summary()
    st.markdown("#### 当前可检索的项目标准依据")
    documents = summary.get("documents", [])
    if documents:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "文档": item.get("title", "—"),
                        "标准号": item.get("standard_number") or "—",
                        "版本": item.get("version") or "未标注",
                        "解析时间": item.get("ingested_at", "—"),
                        "用途": RAG_NAMESPACE_LABELS.get(
                            item.get("source_namespace"),
                            item.get("source_namespace", "—"),
                        ),
                        "状态": item.get("effective_status", "—"),
                        "片段数": item.get("chunk_count", 0),
                    }
                    for item in documents
                    if isinstance(item, dict)
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("当前知识库没有已批准文档。")

    if documents:
        st.markdown("#### 文档移除")
        document_options = ["__none__", *[item.document_id for item in knowledge_base.documents]]
        selected_document = st.selectbox(
            "选择要从当前会话知识库移除的文档",
            options=document_options,
            format_func=lambda value: (
                "不移除"
                if value == "__none__"
                else next(
                    (
                        item.title
                        for item in knowledge_base.documents
                        if item.document_id == value
                    ),
                    value,
                )
            ),
            key="rag_document_remove_v09",
        )
        confirm_remove = st.selectbox(
            "移除确认",
            options=("未确认", "已确认"),
            format_func=lambda value: (
                "未确认移除"
                if value == "未确认"
                else "我确认移除该文档；已有草案中的引用需要重新检索确认"
            ),
            key="rag_document_remove_confirmed_v09",
        ) == "已确认"
        if st.button(
            "移除选中文档",
            key="rag_remove_document_v09",
            disabled=selected_document == "__none__" or not confirm_remove,
        ):
            removed = knowledge_base.remove_document(selected_document)
            if removed:
                state["response"] = None
                state["bound_response"] = None
                state["selected_chunk_ids"] = ()
                st.session_state[RAG_STATE_KEY] = state
                st.success("文档已从当前会话知识库移除。")

    st.markdown("#### 检索与绑定")
    query = st.text_input(
        "检索问题或条款关键词",
        key="rag_query_v09",
        max_chars=2_000,
        placeholder="例如：服务名称是否必填、更新频率、有效性",
    )
    metric_filter_options = ["__all__", *selected_metric_ids]
    metric_filter = st.selectbox(
        "按指标筛选（可选）",
        options=metric_filter_options,
        format_func=lambda value: "不限定" if value == "__all__" else value,
        key="rag_metric_filter_v09",
    )
    search_namespace_options = ["__all__", *RAG_NAMESPACE_LABELS]
    search_namespace = st.selectbox(
        "按来源用途筛选",
        options=search_namespace_options,
        format_func=lambda value: (
            RAG_ALL_NAMESPACE_LABEL if value == "__all__" else RAG_NAMESPACE_LABELS[value]
        ),
        key="rag_search_namespace_v09",
    )
    standard_filter = st.text_input(
        "按标准号筛选（可选）",
        key="rag_search_standard_v09",
        max_chars=100,
    )
    version_filter = st.text_input(
        "按版本筛选（可选）",
        key="rag_search_version_v09",
        max_chars=80,
    )
    if st.button(
        "检索标准依据",
        key="rag_search_v09",
        width="stretch",
        disabled=not bool(query.strip()),
    ):
        try:
            response = knowledge_base.search(
                query,
                metric_id=None if metric_filter == "__all__" else metric_filter,
                standard_number=standard_filter.strip() or None,
                version=version_filter.strip() or None,
                source_namespace=None if search_namespace == "__all__" else search_namespace,
                limit=5,
            )
        except (RagRetrievalError, ValueError) as error:
            st.error(f"检索失败：{_escape_markdown(str(error))}")
        else:
            state["response"] = response
            state["bound_response"] = None
            state["selected_chunk_ids"] = ()
            st.session_state[RAG_STATE_KEY] = state

    response = state.get("response")
    if response is None:
        bound_response, bound_ids = _current_rag_binding()
        if bound_response is not None:
            st.success(f"已绑定 {len(bound_ids)} 个标准依据片段到规则编制。")
            for source in response_source_summary(bound_response):
                st.caption(
                    f"{source['document_name']} · {source.get('version') or '未标注版本'} · "
                    f"{source['chunk_id']}"
                )
        return

    st.caption(
        f"过滤后文档 {response.filtered_document_count} 份，候选片段 "
        f"{response.total_candidate_count} 个，当前展示 {len(response.results)} 个。"
    )
    if response.status == "no_results":
        st.warning("没有命中已批准来源；不能据此声称符合标准。")
        return
    if response.status == "conflict":
        st.error(
            "检索命中多个版本或来源，当前结果不能直接绑定。请用版本/用途筛选后重新检索。"
        )
        if response.conflict is not None:
            st.text(response.conflict.reason)
            for label, version_label in zip(
                response.conflict.document_labels,
                response.conflict.versions,
            ):
                st.text(f"来源：{label} · 版本：{version_label}")
        return

    st.markdown("#### 可绑定片段")
    result_labels = {
        result.chunk.chunk_id: (
            f"{result.document.title} · {result.document.version or '未标注版本'} · "
            f"{result.chunk.section or result.chunk.clause or '正文'} · "
            f"{result.chunk.chunk_id}"
        )
        for result in response.results
    }
    for result in response.results:
        citation = result.citation
        location = " / ".join(
            item
            for item in (
                citation.section,
                citation.clause,
                f"第{citation.page}页" if citation.page is not None else None,
                f"chunk:{citation.chunk_id}",
            )
            if item
        )
        st.markdown(
            f"**{_escape_markdown(citation.document_name)}** · "
            f"{_escape_markdown(citation.document_version or '未标注版本')} · "
            f"{_escape_markdown(location)}"
        )
        st.text(result.chunk.text)
    selected_ids = tuple(
        st.multiselect(
            "选择要绑定的检索片段",
            options=tuple(result_labels),
            format_func=lambda value: result_labels[value],
            key="rag_selected_chunks_v09",
        )
    )
    if st.button(
        "绑定所选片段到规则编制",
        key="rag_bind_selected_v09",
        width="stretch",
        disabled=not bool(selected_ids),
    ):
        state["bound_response"] = response
        state["selected_chunk_ids"] = selected_ids
        st.session_state[RAG_STATE_KEY] = state
        st.success(f"已绑定 {len(selected_ids)} 个标准依据片段。")


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


def _render_rule_dry_run(preview: dict) -> None:
    """展示 v0.7 试运行摘要，不展示原始单元格值。"""

    counts = preview.get("counts", {}) if isinstance(preview, dict) else {}
    st.success("规则试运行完成；尚未审批，也未改变当前基础报告。")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "规则包": preview.get("rule_pack_id", "—"),
                    "版本": preview.get("rule_pack_version", "—"),
                    "检查数量": counts.get("checked", 0),
                    "符合数量": counts.get("compliant", 0),
                    "疑似问题": counts.get("issues", 0),
                    "无法评估": counts.get("not_assessable", 0),
                }
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    metric_rows = preview.get("metrics", []) if isinstance(preview, dict) else []
    if metric_rows:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "规则结果": item.get("name", "—"),
                        "字段": item.get("field") or "—",
                        "状态": item.get("status", "—"),
                        "结果": item.get("value")
                        if item.get("value") is not None
                        else "—",
                        "检查数量": item.get("checked_count", "—"),
                        "疑似问题": item.get("issue_count", "—"),
                        "原因": item.get("reason") or "—",
                    }
                    for item in metric_rows
                    if isinstance(item, dict)
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def _render_custom_rule_authoring(
    report: QualityReport,
    *,
    uploaded_file,
    dataset_name: str,
    sheet_name: str,
    reference_date: date,
    selected_metric_ids: tuple[str, ...],
) -> None:
    """渲染 v1.0 可恢复的自然语言自定义规则闭环。"""

    st.subheader("自定义规则（v1.0）")
    st.caption(
        "用自然语言新增一条业务规则。当前支持格式/正则、字符长度、"
        "条件必填和跨字段比较；规则仍须经过确定性校验、试运行和人工批准。"
        "工作流状态、一次失败恢复及当前会话历史均由本地代码控制。"
    )
    if uploaded_file is None:
        st.info("当前上传内容已不可用，请重新选择文件。")
        return

    rag_response, rag_chunk_ids = _current_rag_binding()
    if rag_chunk_ids:
        st.info(f"本次自定义规则将携带 {len(rag_chunk_ids)} 个已绑定标准依据片段。")

    state = st.session_state.get(CUSTOM_RULE_STATE_KEY, {})
    basis = str(st.session_state.get("custom_rule_intent", "")).strip()
    signature = _rule_form_signature(
        {
            "report_sha256": report.evaluation_context.get("report_sha256"),
            "input_sha256": report.evaluation_context.get("input_sha256"),
            "intent": basis,
            **_rag_binding_signature(),
        }
    )
    if state.get("signature") and state.get("signature") != signature:
        state = {}
        st.session_state[CUSTOM_RULE_STATE_KEY] = state

    st.sidebar.text_input(
        "自定义规则描述",
        key="custom_rule_intent",
        max_chars=4000,
        help="例如：状态为注销时，注销日期必须填写；开始日期不得晚于结束日期。",
    )
    basis = str(st.session_state.get("custom_rule_intent", "")).strip()
    signature = _rule_form_signature(
        {
            "report_sha256": report.evaluation_context.get("report_sha256"),
            "input_sha256": report.evaluation_context.get("input_sha256"),
            "intent": basis,
            **_rag_binding_signature(),
        }
    )
    if state.get("signature") and state.get("signature") != signature:
        state = {}
        st.session_state[CUSTOM_RULE_STATE_KEY] = state

    if st.button(
        "AI 解析自定义规则",
        key="compile_custom_rule_v08",
        width="stretch",
        disabled=(
            not bool(basis)
            or isinstance(state.get("run"), RuleAuthoringRun)
        ),
    ):
        run = None
        try:
            run = begin_rule_authoring_run(
                report,
                target_metric_id=None,
                user_intent=basis,
                selected_metric_ids=selected_metric_ids,
                selected_chunk_ids=rag_chunk_ids,
            )
            provider = _build_rule_authoring_provider()
            run = compile_rule_authoring_run(
                run,
                report,
                user_intent=basis,
                provider=provider,
                allow_template_fallback=provider is None,
                rag_response=rag_response,
                selected_chunk_ids=rag_chunk_ids,
            )
            if run.workflow.state == "draft":
                run = validate_rule_authoring_run(run, report)
        except (RuleAuthoringCoordinatorError, ValueError) as error:
            safe_error = _model_error_detail(error)
            if run is not None and run.workflow.state == "compiling":
                run = RuleAuthoringRun(
                    workflow=run.workflow.fail(
                        stage="compiling",
                        code="provider_configuration_failed",
                        message=safe_error,
                    )
                )
                state = _custom_authoring_state(signature=signature, run=run)
            else:
                state = {
                    "signature": signature,
                    "error": safe_error,
                }
            st.session_state[CUSTOM_RULE_STATE_KEY] = state
            st.error(
                "自定义规则解析失败："
                f"{_escape_markdown(safe_error)}"
            )
        else:
            state = _custom_authoring_state(signature=signature, run=run)
            st.session_state[CUSTOM_RULE_STATE_KEY] = state

    state = st.session_state.get(CUSTOM_RULE_STATE_KEY, state)
    run = state.get("run")
    if (
        isinstance(run, RuleAuthoringRun)
        and run.workflow.state == "failed"
        and run.workflow.recoverable_state == "compiling"
    ):
        st.error(f"规则编译失败：{_escape_markdown(run.workflow.error or '未知错误')}")
        if run.workflow.can_retry and st.button(
            "重试失败步骤（最多一次）",
            key="retry_custom_compile_v10",
            width="stretch",
        ):
            try:
                run = retry_rule_authoring_run(
                    run,
                    user_intent=basis,
                    selected_chunk_ids=rag_chunk_ids,
                )
                provider = _build_rule_authoring_provider()
                run = compile_rule_authoring_run(
                    run,
                    report,
                    user_intent=basis,
                    provider=provider,
                    allow_template_fallback=provider is None,
                    rag_response=rag_response,
                    selected_chunk_ids=rag_chunk_ids,
                )
                if run.workflow.state == "draft":
                    run = validate_rule_authoring_run(run, report)
            except (RuleAuthoringCoordinatorError, ValueError) as error:
                if run.workflow.state == "compiling":
                    run = RuleAuthoringRun(
                        workflow=run.workflow.fail(
                            stage="compiling",
                            code="provider_configuration_failed",
                            message=_model_error_detail(error),
                        )
                    )
            state = _custom_authoring_state(signature=signature, run=run)
            st.session_state[CUSTOM_RULE_STATE_KEY] = state
    run = state.get("run")
    if isinstance(run, RuleAuthoringRun):
        _render_rule_workflow_status(run)
    if state.get("error") and not (
        isinstance(run, RuleAuthoringRun) and run.workflow.state == "failed"
    ):
        st.error(f"自定义规则未完成：{_escape_markdown(state['error'])}")
    draft = state.get("draft")
    if draft is None:
        st.caption("填写规则描述后，点击“AI 解析自定义规则”。")
        return

    if draft.provider.fallback_used:
        st.info("当前未配置外部模型，使用本地模板生成候选草案；正式使用前请配置 API。")
    elif draft.provider.mode == "model":
        st.success(f"模型已生成候选自定义规则：{draft.provider.model or draft.provider.provider}")
    st.caption(f"草案状态：{draft.status}")

    if draft.status == "needs_clarification":
        st.warning("当前自定义规则缺少可执行信息，请补充后重新解析。")
        for question in draft.clarification_questions:
            st.text(f"需要补充：{question}")
        return
    if draft.status == "rejected":
        st.error(_escape_markdown(draft.unsupported_reason or "当前需求暂不支持。"))
        return
    if draft.rule_spec is None:
        st.error("Provider 未生成可展示的自定义规则草案。")
        return

    with st.expander("查看自定义规则草案", expanded=True):
        st.json(draft.rule_spec.to_dict())
    validation = run.workflow.validation if isinstance(run, RuleAuthoringRun) else None
    if validation is None or not validation.valid:
        st.error("自定义规则未通过确定性校验，不能试运行。")
        for error in validation.errors if validation is not None else ():
            st.text(error)
        return
    st.success("自定义规则已通过字段、参数和当前报告画像校验。")

    dry_run_clicked = False
    if isinstance(run, RuleAuthoringRun) and run.workflow.state == "validated":
        dry_run_clicked = st.button(
            "试运行自定义规则",
            key="dry_run_custom_rule_v08",
            width="stretch",
        )
    if dry_run_clicked:
        run = dry_run_rule_authoring_run(
            run,
            report,
            content=uploaded_file.getvalue(),
            file_name=uploaded_file.name,
            dataset_name=dataset_name.strip() or None,
            sheet_name=sheet_name.strip() or None,
            reference_date=reference_date,
            selected_metric_ids=selected_metric_ids,
        )
        state = _custom_authoring_state(signature=signature, run=run)
        st.session_state[CUSTOM_RULE_STATE_KEY] = state

    run = state.get("run")
    if (
        isinstance(run, RuleAuthoringRun)
        and run.workflow.state == "failed"
        and run.workflow.recoverable_state == "validated"
    ):
        st.error(f"试运行未完成：{_escape_markdown(run.workflow.error or '未知错误')}")
        if run.workflow.can_retry and st.button(
            "重试失败步骤（最多一次）",
            key="retry_custom_dry_run_v10",
            width="stretch",
        ):
            try:
                run = retry_rule_authoring_run(
                    run,
                    user_intent=basis,
                    selected_chunk_ids=rag_chunk_ids,
                )
                run = dry_run_rule_authoring_run(
                    run,
                    report,
                    content=uploaded_file.getvalue(),
                    file_name=uploaded_file.name,
                    dataset_name=dataset_name.strip() or None,
                    sheet_name=sheet_name.strip() or None,
                    reference_date=reference_date,
                    selected_metric_ids=selected_metric_ids,
                )
            except RuleAuthoringCoordinatorError as error:
                st.error(_escape_markdown(_model_error_detail(error)))
            state = _custom_authoring_state(signature=signature, run=run)
            st.session_state[CUSTOM_RULE_STATE_KEY] = state

    preview = state.get("dry_run")
    if preview is None:
        if isinstance(run, RuleAuthoringRun) and run.workflow.state == "failed":
            return
        st.info("规则草案已校验；点击“试运行自定义规则”查看影响摘要。")
        return
    _render_rule_dry_run(preview)

    if isinstance(run, RuleAuthoringRun) and run.workflow.state == "executed":
        if run.result is not None:
            _render_rule_result(run.result)
        return

    if (
        isinstance(run, RuleAuthoringRun)
        and run.workflow.state == "failed"
        and run.workflow.recoverable_state == "approved"
    ):
        st.error(f"正式重评未完成：{_escape_markdown(run.workflow.error or '未知错误')}")
        if run.workflow.can_retry and st.button(
            "重试失败步骤（最多一次）",
            key="retry_custom_execution_v10",
            width="stretch",
        ):
            try:
                run = retry_rule_authoring_run(
                    run,
                    user_intent=basis,
                    selected_chunk_ids=rag_chunk_ids,
                )
                run = execute_rule_authoring_run(
                    run,
                    content=uploaded_file.getvalue(),
                    file_name=uploaded_file.name,
                    dataset_name=dataset_name.strip() or None,
                    sheet_name=sheet_name.strip() or None,
                    reference_date=reference_date,
                    selected_metric_ids=selected_metric_ids,
                )
            except RuleAuthoringCoordinatorError as error:
                st.error(_escape_markdown(_model_error_detail(error)))
            state = _custom_authoring_state(signature=signature, run=run)
            st.session_state[CUSTOM_RULE_STATE_KEY] = state
        if run.result is not None:
            _render_rule_result(run.result)
        return

    approver = st.text_input(
        "审批人标识（自定义规则，本地自声明）",
        max_chars=100,
        key="custom_rule_approver",
    )
    confirmed = st.checkbox(
        "我已核对当前自定义规则和试运行摘要，并批准本次确定性重评。",
        key="custom_rule_approval_confirmed",
    )
    approve_clicked = st.button(
        "批准并重新评估（自定义规则）",
        key="approve_custom_rule_v08",
        type="primary",
        width="stretch",
        disabled=not approver.strip() or not confirmed,
    )
    if approve_clicked:
        try:
            run = approve_rule_authoring_run(
                run,
                report,
                approver=approver,
            )
            run = execute_rule_authoring_run(
                run,
                content=uploaded_file.getvalue(),
                file_name=uploaded_file.name,
                dataset_name=dataset_name.strip() or None,
                sheet_name=sheet_name.strip() or None,
                reference_date=reference_date,
                selected_metric_ids=selected_metric_ids,
            )
        except RuleAuthoringCoordinatorError as error:
            st.error(
                "自定义规则未执行："
                f"{_escape_markdown(_model_error_detail(error))}"
            )
        else:
            state = _custom_authoring_state(signature=signature, run=run)
            st.session_state[CUSTOM_RULE_STATE_KEY] = state
            if run.workflow.state == "failed":
                st.error(
                    "自定义规则已保留审批，但正式重评未完成："
                    f"{_escape_markdown(run.workflow.error or '未知错误')}"
                )

    if state.get("result") is not None:
        _render_rule_result(state["result"])


def _render_rule_authoring(
    report: QualityReport,
    *,
    uploaded_file,
    dataset_name: str,
    sheet_name: str,
    reference_date: date,
    selected_metric_ids: tuple[str, ...],
) -> None:
    """渲染 v1.0 显式状态、可恢复的规则编制闭环。"""

    st.subheader("补充评价标准")
    st.caption(
        "针对需要外部标准的指标补充评价依据，再由 Agent 生成受限规则草案。"
        "流程为：补充标准 → AI 解析 → 确定性试运行 → 批准并重新评估。"
        "模型只负责理解和编译；校验、试运行、审批、正式重评和失败恢复"
        "均由本地确定性代码完成。"
    )
    if report.status != "success":
        st.info("零配置评估成功后才能编制指标规则。")
        return
    if uploaded_file is None:
        st.info("当前上传内容已不可用，请重新选择文件。")
        return
    if not selected_metric_ids:
        st.info("请先选择至少一个指标。")
        return

    rag_response, rag_chunk_ids = _current_rag_binding()
    if rag_chunk_ids:
        st.info(f"已绑定 {len(rag_chunk_ids)} 个标准依据片段；解析规则时会写入 RuleEvidence。")

    _render_custom_rule_authoring(
        report,
        uploaded_file=uploaded_file,
        dataset_name=dataset_name,
        sheet_name=sheet_name,
        reference_date=reference_date,
        selected_metric_ids=selected_metric_ids,
    )
    st.divider()

    state = _rule_authoring_state_for(report)
    standard_metric_ids = tuple(
        metric_id
        for metric_id in selected_metric_ids
        if (
            (definition := get_metric_definition(metric_id)) is not None
            and not bool(definition.get("auto_assessable"))
        )
    )
    if standard_metric_ids:
        st.warning(
            "需要补充评价标准的指标已优先列出。请根据每项指标下方的“需要提供”"
            "说明填写具体字段、取值范围、格式或参照规则。"
        )
    else:
        st.success("当前选中指标均已有本地确定性评价依据，可直接评估。")
    ordered_metric_ids = (
        *standard_metric_ids,
        *(metric_id for metric_id in selected_metric_ids if metric_id not in standard_metric_ids),
    )
    for metric_id in ordered_metric_ids:
        definition = get_metric_definition(metric_id)
        if definition is None:
            continue
        with st.container(border=True):
            st.markdown(f"#### {_escape_markdown(definition['name'])}")
            st.caption(
                f"{_escape_markdown(definition['dimension'])} · "
                f"{_escape_markdown(definition['description'])}"
            )
            if not bool(definition.get("auto_assessable")):
                required_inputs = tuple(
                    str(item) for item in definition.get("required_inputs", ())
                )
                requirement = "、".join(required_inputs) or "具体评价口径"
                st.warning(
                    "该指标需要补充标准；需要提供："
                    f"{_escape_markdown(requirement)}。"
                )
            st.text_area(
                "评价依据",
                key=_metric_evidence_key(metric_id),
                height=100,
                max_chars=4000,
                help="评价依据只用于生成当前指标的规则草案。",
            )
            basis = _metric_evidence_text(metric_id)
            basis_signature = _rule_form_signature(
                {
                    "metric_id": metric_id,
                    "evidence": basis,
                    **_rag_binding_signature(),
                }
            )
            previous_signature = state.get("draft_signatures", {}).get(metric_id)
            if previous_signature and previous_signature != basis_signature:
                state = _reset_metric_authoring_state(state, metric_id)
                st.session_state[RULE_AUTHORING_STATE_KEY] = state
                st.info("评价依据已变化，旧规则草案、试运行和审批状态已清除。")
            run = state.get("runs", {}).get(metric_id)
            if (
                isinstance(run, RuleAuthoringRun)
                and run.workflow.state == "failed"
                and run.workflow.recoverable_state == "compiling"
            ):
                st.error(
                    "规则编译失败："
                    f"{_escape_markdown(run.workflow.error or '未知错误')}"
                )
                if run.workflow.can_retry and st.button(
                    "重试失败步骤（最多一次）",
                    key=f"retry_metric_compile_v10_{metric_id}",
                    width="stretch",
                ):
                    try:
                        run = retry_rule_authoring_run(
                            run,
                            user_intent=basis,
                            selected_chunk_ids=rag_chunk_ids,
                        )
                        provider = _build_rule_authoring_provider()
                        run = compile_rule_authoring_run(
                            run,
                            report,
                            user_intent=basis,
                            provider=provider,
                            allow_template_fallback=provider is None,
                            rag_response=rag_response,
                            selected_chunk_ids=rag_chunk_ids,
                        )
                        if run.workflow.state == "draft":
                            run = validate_rule_authoring_run(run, report)
                    except (RuleAuthoringCoordinatorError, ValueError) as error:
                        if run.workflow.state == "compiling":
                            run = RuleAuthoringRun(
                                workflow=run.workflow.fail(
                                    stage="compiling",
                                    code="provider_configuration_failed",
                                    message=_model_error_detail(error),
                                )
                            )
                    state = _sync_metric_authoring_run(
                        state,
                        metric_id,
                        run,
                        draft_signature=basis_signature,
                    )
                    st.session_state[RULE_AUTHORING_STATE_KEY] = state
            run = state.get("runs", {}).get(metric_id)
            if isinstance(run, RuleAuthoringRun):
                _render_rule_workflow_status(run)

            run = state.get("runs", {}).get(metric_id)
            draft = state.get("drafts", {}).get(metric_id)
            compile_clicked = st.button(
                "AI 解析依据",
                key=f"compile_metric_rule_{metric_id}",
                width="stretch",
                disabled=(
                    not bool(basis)
                    or isinstance(
                        state.get("runs", {}).get(metric_id),
                        RuleAuthoringRun,
                    )
                ),
            )
            if compile_clicked:
                run = None
                try:
                    run = begin_rule_authoring_run(
                        report,
                        target_metric_id=metric_id,
                        user_intent=basis,
                        selected_metric_ids=selected_metric_ids,
                        selected_chunk_ids=rag_chunk_ids,
                    )
                    provider = _build_rule_authoring_provider()
                    run = compile_rule_authoring_run(
                        run,
                        report,
                        user_intent=basis,
                        provider=provider,
                        allow_template_fallback=provider is None,
                        rag_response=rag_response,
                        selected_chunk_ids=rag_chunk_ids,
                    )
                    if run.workflow.state == "draft":
                        run = validate_rule_authoring_run(run, report)
                except (RuleAuthoringCoordinatorError, ValueError) as error:
                    if run is not None and run.workflow.state == "compiling":
                        run = RuleAuthoringRun(
                            workflow=run.workflow.fail(
                                stage="compiling",
                                code="provider_configuration_failed",
                                message=_model_error_detail(error),
                            )
                        )
                        state = _sync_metric_authoring_run(
                            state,
                            metric_id,
                            run,
                            draft_signature=basis_signature,
                        )
                    else:
                        state = _reset_metric_authoring_state(state, metric_id)
                        errors = dict(state.get("execution_errors", {}))
                        errors[metric_id] = _model_error_detail(error)
                        state["execution_errors"] = errors
                    st.session_state[RULE_AUTHORING_STATE_KEY] = state
                    st.error(
                        "规则编制未完成："
                        f"{_escape_markdown(_model_error_detail(error))}"
                    )
                else:
                    state = _sync_metric_authoring_run(
                        state,
                        metric_id,
                        run,
                        draft_signature=basis_signature,
                    )
                    st.session_state[RULE_AUTHORING_STATE_KEY] = state

            run = state.get("runs", {}).get(metric_id)
            draft = state.get("drafts", {}).get(metric_id)
            if draft is None:
                if not basis:
                    st.caption("填写评价依据后，才能启动规则编制。")
                continue

            if draft.provider.fallback_used:
                st.info(
                    "当前未配置外部模型，使用本地模板生成候选草案；正式使用前请配置 API。"
                )
            elif draft.provider.mode == "model":
                st.success(f"模型已生成候选草案：{draft.provider.model or draft.provider.provider}")
            else:
                st.info("当前使用本地模板生成候选草案。")

            st.caption(f"草案状态：{draft.status}")
            if draft.evidence:
                with st.expander("查看评价依据与模型假设"):
                    st.markdown("**用户原始依据**")
                    st.text(basis)
                    for item in draft.evidence:
                        st.text(f"{item.type} · {item.text}")
                    if draft.assumptions:
                        st.markdown("**系统/模型假设**")
                        for item in draft.assumptions:
                            st.text(item)

            if draft.status == "needs_clarification":
                st.warning("当前依据缺少可执行规则所需信息。")
                clarification_questions = tuple(
                    dict.fromkeys(
                        (
                            *draft.clarification_questions,
                            *_rule_missing_guidance(report, metric_id, basis),
                        )
                    )
                )[:5]
                for question in clarification_questions:
                    st.text(f"需要补充：{question}")
                continue
            if draft.status == "rejected":
                st.error(_escape_markdown(draft.unsupported_reason or "当前需求暂不支持。"))
                _render_rule_missing_guidance(report, metric_id, basis)
                continue
            if draft.rule_spec is None:
                st.error("Provider 未生成可展示的规则草案。")
                continue

            with st.expander("查看结构化规则草案", expanded=True):
                st.json(draft.rule_spec.to_dict())
            draft_download_name = sanitize_file_name(
                f"{report.dataset.name}_{metric_id}_rule_draft.json",
                default_name=f"{metric_id}_rule_draft.json",
                safe_extension=".json",
            )
            st.download_button(
                "下载 RuleDraft（JSON）",
                data=json.dumps(
                    draft.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ).encode("utf-8"),
                file_name=draft_download_name,
                mime="application/json",
                key=f"download_metric_rule_draft_{metric_id}",
            )
            validation = run.workflow.validation if isinstance(run, RuleAuthoringRun) else None
            if validation is None or not validation.valid:
                st.error("规则草案未通过确定性校验，不能试运行。")
                for error in validation.errors if validation is not None else ():
                    st.text(error)
                continue
            st.success("规则草案已通过字段、参数和当前报告画像校验。")

            st.caption("正式执行前必须先完成试运行和人工审批。")
            dry_run_clicked = False
            if run.workflow.state == "validated":
                dry_run_clicked = st.button(
                    "试运行规则",
                    key=f"dry_run_metric_rule_{metric_id}",
                    width="stretch",
                )
            if dry_run_clicked:
                with st.spinner("正在使用确定性规则引擎进行试运行……"):
                    run = dry_run_rule_authoring_run(
                        run,
                        report,
                        content=uploaded_file.getvalue(),
                        file_name=uploaded_file.name,
                        dataset_name=dataset_name.strip() or None,
                        sheet_name=sheet_name.strip() or None,
                        reference_date=reference_date,
                        selected_metric_ids=selected_metric_ids,
                    )
                state = _sync_metric_authoring_run(
                    state,
                    metric_id,
                    run,
                    draft_signature=basis_signature,
                )
                st.session_state[RULE_AUTHORING_STATE_KEY] = state

            run = state.get("runs", {}).get(metric_id)
            if (
                isinstance(run, RuleAuthoringRun)
                and run.workflow.state == "failed"
                and run.workflow.recoverable_state == "validated"
            ):
                st.error(
                    "试运行未完成："
                    f"{_escape_markdown(run.workflow.error or '未知错误')}"
                )
                if run.workflow.can_retry and st.button(
                    "重试失败步骤（最多一次）",
                    key=f"retry_metric_dry_run_v10_{metric_id}",
                    width="stretch",
                ):
                    try:
                        run = retry_rule_authoring_run(
                            run,
                            user_intent=basis,
                            selected_chunk_ids=rag_chunk_ids,
                        )
                        run = dry_run_rule_authoring_run(
                            run,
                            report,
                            content=uploaded_file.getvalue(),
                            file_name=uploaded_file.name,
                            dataset_name=dataset_name.strip() or None,
                            sheet_name=sheet_name.strip() or None,
                            reference_date=reference_date,
                            selected_metric_ids=selected_metric_ids,
                        )
                    except RuleAuthoringCoordinatorError as error:
                        st.error(_escape_markdown(_model_error_detail(error)))
                    state = _sync_metric_authoring_run(
                        state,
                        metric_id,
                        run,
                        draft_signature=basis_signature,
                    )
                    st.session_state[RULE_AUTHORING_STATE_KEY] = state

            preview = state.get("dry_runs", {}).get(metric_id)
            if preview is None:
                if isinstance(run, RuleAuthoringRun) and run.workflow.state == "failed":
                    continue
                st.info("规则草案已校验；点击“试运行规则”查看影响摘要。")
                continue
            _render_rule_dry_run(preview)

            if isinstance(run, RuleAuthoringRun) and run.workflow.state == "executed":
                if run.result is not None:
                    _render_rule_result(run.result)
                continue

            if (
                isinstance(run, RuleAuthoringRun)
                and run.workflow.state == "failed"
                and run.workflow.recoverable_state == "approved"
            ):
                st.error(
                    "AI 规则已保留审批，但正式重评未完成："
                    f"{_escape_markdown(run.workflow.error or '未知错误')}"
                )
                if run.workflow.can_retry and st.button(
                    "重试失败步骤（最多一次）",
                    key=f"retry_metric_execution_v10_{metric_id}",
                    width="stretch",
                ):
                    try:
                        run = retry_rule_authoring_run(
                            run,
                            user_intent=basis,
                            selected_chunk_ids=rag_chunk_ids,
                        )
                        run = execute_rule_authoring_run(
                            run,
                            content=uploaded_file.getvalue(),
                            file_name=uploaded_file.name,
                            dataset_name=dataset_name.strip() or None,
                            sheet_name=sheet_name.strip() or None,
                            reference_date=reference_date,
                            selected_metric_ids=selected_metric_ids,
                        )
                    except RuleAuthoringCoordinatorError as error:
                        st.error(_escape_markdown(_model_error_detail(error)))
                    state = _sync_metric_authoring_run(
                        state,
                        metric_id,
                        run,
                        draft_signature=basis_signature,
                    )
                    st.session_state[RULE_AUTHORING_STATE_KEY] = state
                if run.result is not None:
                    _render_rule_result(run.result)
                continue

            approver = st.text_input(
                "审批人标识（AI规则，本地自声明）",
                max_chars=100,
                key=f"v07_rule_approver_{metric_id}",
            )
            confirmed = st.checkbox(
                "我已核对当前规则、评价依据和试运行摘要，并批准本次确定性重评。",
                key=f"v07_rule_approval_confirmed_{metric_id}",
            )
            pack = run.draft_pack
            if pack is None:
                st.error("当前工作流缺少试运行 RulePack，不能审批。")
                continue
            pack_hash = draft_sha256(pack)
            confirmations = dict(state.get("confirmed_pack_sha256", {}))
            if confirmed:
                confirmations[metric_id] = pack_hash
            else:
                confirmations.pop(metric_id, None)
            state = {**state, "confirmed_pack_sha256": confirmations}
            st.session_state[RULE_AUTHORING_STATE_KEY] = state
            approve_clicked = st.button(
                "补充完成并重新评估（AI规则）",
                key=f"approve_metric_rule_{metric_id}",
                type="primary",
                width="stretch",
                disabled=(
                    not approver.strip()
                    or not confirmed
                    or confirmations.get(metric_id) != pack_hash
                ),
            )
            if approve_clicked:
                try:
                    run = approve_rule_authoring_run(
                        run,
                        report,
                        approver=approver,
                    )
                    with st.spinner("正在重新解析当前输入并执行已审批 AI 规则……"):
                        run = execute_rule_authoring_run(
                            run,
                            content=uploaded_file.getvalue(),
                            file_name=uploaded_file.name,
                            dataset_name=dataset_name.strip() or None,
                            sheet_name=sheet_name.strip() or None,
                            reference_date=reference_date,
                            selected_metric_ids=selected_metric_ids,
                        )
                except RuleAuthoringCoordinatorError as error:
                    errors = dict(state.get("execution_errors", {}))
                    safe_error = _model_error_detail(error)
                    errors[metric_id] = safe_error
                    state = {**state, "execution_errors": errors}
                    st.session_state[RULE_AUTHORING_STATE_KEY] = state
                    st.error(
                        "AI规则未执行："
                        f"{_escape_markdown(safe_error)}"
                    )
                else:
                    state = _sync_metric_authoring_run(
                        state,
                        metric_id,
                        run,
                        draft_signature=basis_signature,
                    )
                    st.session_state[RULE_AUTHORING_STATE_KEY] = state
                    if run.workflow.state == "failed":
                        st.error(
                            "AI 规则已保留审批，但正式重评未完成："
                            f"{_escape_markdown(run.workflow.error or '未知错误')}"
                        )

            result = state.get("results", {}).get(metric_id)
            if result is not None:
                _render_rule_result(result)

    st.divider()
    _render_rule_workflow_history()


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
                        "execution_error": _model_error_detail(error),
                    }
                    st.error(
                        "规则增强未执行："
                        f"{_escape_markdown(_model_error_detail(error))}"
                    )
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
        data=serialize_rule_evaluation_markdown(result),
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
) -> tuple[object, ...] | None:
    """标识当前评估请求，防止输入或模型配置变化后继续展示旧报告。"""

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
        _model_request_binding(include_environment=False),
    )


st.title("政务数据集质量评估")
st.caption(
    "v1.2 · 在首页与大模型对话创建规则，或批量导入规则，并以受控工作流执行。"
)

_initialize_metric_selection_state()
_initialize_metric_evidence_state()
_initialize_pre_evaluation_rule_input_state()
_initialize_model_api_state()
report_is_displayed = st.session_state.get("quality_report") is not None
if report_is_displayed:
    _preserve_hidden_pre_evaluation_widget_state()
pre_rule_chat_header = st.empty()
pre_rule_chat_shell = st.empty()
if report_is_displayed:
    pre_rule_chat_header.empty()
    pre_rule_chat_shell.empty()
pre_rule_requests: tuple[RuleBatchInput, ...] = ()
pre_rule_input_errors: tuple[str, ...] = ()
pre_rule_input_warnings: tuple[str, ...] = ()
pre_rule_chat_submitted = False
if not report_is_displayed:
    (
        chat_rule_requests,
        chat_rule_errors,
        chat_rule_warnings,
        pre_rule_chat_submitted,
    ) = _render_rule_chat_input(
        header_slot=pre_rule_chat_header,
        chat_shell=pre_rule_chat_shell,
    )
    selected_metric_ids = _render_metric_selection_panel()
    (
        metric_rule_requests,
        metric_rule_errors,
        metric_rule_warnings,
        _,
    ) = _render_pre_evaluation_rule_inputs(selected_metric_ids)
    pre_rule_requests = tuple((*chat_rule_requests, *metric_rule_requests))
    pre_rule_input_errors = tuple((*chat_rule_errors, *metric_rule_errors))
    pre_rule_input_warnings = tuple((*chat_rule_warnings, *metric_rule_warnings))
    for warning in pre_rule_input_warnings:
        st.warning(_escape_markdown(warning))
else:
    selected_metric_ids = tuple(
        metric_id
        for metric_id in ALL_METRIC_IDS
        if metric_id in st.session_state.get(METRIC_SELECTION_KEY, [])
    )
missing_metric_evidence_ids = _missing_metric_evidence_ids(selected_metric_ids)
preflight_rules = False
pre_rule_signature = ""

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
            "、JSONL / NDJSON 和 GeoJSON FeatureCollection；"
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
    _render_model_api_settings()
    request_signature = _evaluation_request_signature(
        uploaded_file,
        dataset_name,
        sheet_name,
        reference_date,
        selected_metric_ids,
    )
    previous_request_signature = st.session_state.get(
        "evaluation_request_signature"
    )
    if previous_request_signature != request_signature:
        had_report = st.session_state.get("quality_report") is not None
        st.session_state["evaluation_request_signature"] = request_signature
        st.session_state.pop("quality_report", None)
        _clear_agent_state()
        preserve_chat = not had_report
        _clear_rule_state(
            preserve_rag_binding=not had_report,
            # Rule descriptions and attachments remain available while the
            # user changes or first supplies the dataset.
            preserve_chat=preserve_chat,
        )
        if had_report:
            st.rerun()
    if not report_is_displayed:
        pre_rule_signature = _pre_evaluation_rule_signature(
            request_signature,
            pre_rule_requests,
        )
        previous_preflight = st.session_state.get(PRE_EVALUATION_RULE_STATE_KEY)
        if (
            isinstance(previous_preflight, dict)
            and previous_preflight.get("signature") != pre_rule_signature
        ):
            st.session_state.pop(PRE_EVALUATION_RULE_STATE_KEY, None)
            st.session_state.pop(PRE_EVALUATION_RULE_RESULT_KEY, None)
            st.session_state.pop(PRE_EVALUATION_RULE_APPROVER_KEY, None)
            st.session_state.pop(PRE_EVALUATION_RULE_CONFIRM_KEY, None)
        if missing_metric_evidence_ids:
            st.warning("请先补全所有已勾选指标的评价规则。")
        has_pre_evaluation_rules = bool(
            pre_rule_requests or pre_rule_input_errors
        )
        if has_pre_evaluation_rules:
            for error in pre_rule_input_errors:
                st.error(_escape_markdown(error))
            preflight_rules = st.button(
                "AI 检查并生成规则",
                type="primary",
                width="stretch",
                disabled=(
                    uploaded_file is None
                    or not selected_metric_ids
                    or bool(missing_metric_evidence_ids)
                    or bool(pre_rule_input_errors)
                ),
            )
            card_rule_requests_only = bool(pre_rule_requests) and all(
                item.origin == "metric_supplement"
                for item in pre_rule_requests
            )
            if card_rule_requests_only and not pre_rule_input_errors:
                st.caption(
                    "指标卡片中的补充内容默认应先用 AI 检查并生成规则；"
                    "如只想保留为普通评价依据，也可直接运行质量评估。"
                )
                run_evaluation = st.button(
                    "运行质量评估",
                    width="stretch",
                    disabled=(
                        uploaded_file is None
                        or not selected_metric_ids
                        or bool(missing_metric_evidence_ids)
                    ),
                )
            else:
                run_evaluation = False
                st.caption("全部规则通过预检和人工批准后，才会启动最终质量评估。")
        else:
            run_evaluation = st.button(
                "运行质量评估",
                type="primary",
                width="stretch",
                disabled=(
                    uploaded_file is None
                    or not selected_metric_ids
                    or bool(missing_metric_evidence_ids)
                ),
            )
        if pre_rule_chat_submitted and not pre_rule_requests:
            preflight_rules = False
            run_evaluation = False
    else:
        run_evaluation = False
    # The chat input is rendered before the sidebar uploader; carry its submit
    # event into the preflight once the uploaded data is known.
    if pre_rule_chat_submitted and uploaded_file is not None:
        preflight_rules = True
        run_evaluation = False
    st.caption("原始文件仅写入临时目录用于本次计算，评估结束后自动删除。")

if pre_rule_chat_submitted and uploaded_file is None:
    chat_messages = list(
        st.session_state.get(PRE_EVALUATION_CHAT_MESSAGES_KEY, [])
    )
    chat_messages.append(
        {
            "role": "assistant",
            "content": "我已收到规则描述。请先上传数据文件，我才能结合字段画像生成并检查规则。",
        }
    )
    st.session_state[PRE_EVALUATION_CHAT_MESSAGES_KEY] = chat_messages
    st.rerun()

if (
    (preflight_rules or pre_rule_chat_submitted)
    and uploaded_file is not None
    and selected_metric_ids
    and pre_rule_requests
    and not missing_metric_evidence_ids
    and not pre_rule_input_errors
):
    _clear_agent_state()
    st.session_state.pop(PRE_EVALUATION_RULE_RESULT_KEY, None)
    with st.spinner(
        f"正在解析数据结构并逐条检查 {len(pre_rule_requests)} 条规则……"
    ):
        try:
            preflight_report = evaluate_uploaded_dataset(
                uploaded_file.getvalue(),
                uploaded_file.name,
                dataset_name=dataset_name.strip() or None,
                sheet_name=sheet_name.strip() or None,
                reference_date=reference_date,
                selected_metric_ids=selected_metric_ids,
            )
            provider = _build_rule_authoring_provider()
            rag_response, rag_chunk_ids = _current_rag_binding()
            preflight = compile_rule_batch(
                preflight_report,
                pre_rule_requests,
                provider=provider,
                allow_template_fallback=provider is None,
                rag_response=rag_response,
                selected_chunk_ids=rag_chunk_ids,
            )
            preview = None
            if preflight.ready and preflight.draft_pack is not None:
                preview = dry_run_uploaded_dataset_with_rule_pack(
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                    preflight.draft_pack,
                    dataset_name=dataset_name.strip() or None,
                    sheet_name=sheet_name.strip() or None,
                    reference_date=reference_date,
                    selected_metric_ids=selected_metric_ids,
                )
            st.session_state[PRE_EVALUATION_RULE_STATE_KEY] = {
                "signature": pre_rule_signature,
                "report": preflight_report,
                "preflight": preflight,
                "preview": preview,
                "draft_sha256": (
                    draft_sha256(preflight.draft_pack)
                    if preflight.draft_pack is not None
                    else None
                ),
                "approved_pack": None,
                "result": None,
                "error": None,
            }
            chat_messages = list(
                st.session_state.get(PRE_EVALUATION_CHAT_MESSAGES_KEY, [])
            )
            if pre_rule_chat_submitted:
                if preflight.ready:
                    chat_messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                f"我已生成 {len(preflight.draft_pack.rules)} 条规则，"
                                "并完成确定性校验和试运行。请在下方核对后批准。"
                            ),
                        }
                    )
                else:
                    details = []
                    for item in preflight.items:
                        if item.status != "ready":
                            details.append(
                                f"{item.request.label}：" + "；".join(item.messages)
                            )
                    chat_messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                "这条规则暂时不能执行，请补充："
                                + "；".join(details[:5])
                            ),
                        }
                    )
                st.session_state[PRE_EVALUATION_CHAT_MESSAGES_KEY] = chat_messages
        except (
            DatasetReadError,
            UnsupportedFileTypeError,
            RuleImportError,
            RulePackValidationError,
            RulePackExecutionError,
            ValueError,
        ) as error:
            error_detail = _model_error_detail(error)
            st.session_state[PRE_EVALUATION_RULE_STATE_KEY] = {
                "signature": pre_rule_signature,
                "error": error_detail,
            }
            if pre_rule_chat_submitted:
                chat_messages = list(
                    st.session_state.get(PRE_EVALUATION_CHAT_MESSAGES_KEY, [])
                )
                chat_messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "这条规则暂时无法生成："
                            f"{error_detail} 请修改描述或检查模型配置后重试。"
                        ),
                    }
                )
                st.session_state[PRE_EVALUATION_CHAT_MESSAGES_KEY] = chat_messages
        except Exception:
            st.session_state[PRE_EVALUATION_RULE_STATE_KEY] = {
                "signature": pre_rule_signature,
                "error": "模型服务或临时解析环境不可用，请检查配置后重试。",
            }
            if pre_rule_chat_submitted:
                chat_messages = list(
                    st.session_state.get(PRE_EVALUATION_CHAT_MESSAGES_KEY, [])
                )
                chat_messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "这条规则暂时无法生成：模型服务或临时解析环境不可用，"
                            "请检查配置后重试。"
                        ),
                    }
                )
                st.session_state[PRE_EVALUATION_CHAT_MESSAGES_KEY] = chat_messages
    st.rerun()

if not report_is_displayed and pre_rule_signature:
    _render_pre_evaluation_rule_state(
        signature=pre_rule_signature,
        uploaded_file=uploaded_file,
        dataset_name=dataset_name,
        sheet_name=sheet_name,
        reference_date=reference_date,
        selected_metric_ids=selected_metric_ids,
    )

if (
    run_evaluation
    and uploaded_file is not None
    and selected_metric_ids
    and not missing_metric_evidence_ids
):
    _clear_agent_state()
    _clear_rule_state(preserve_rag_binding=True)
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
            _clear_rule_state(preserve_rag_binding=True)
            st.error(_escape_markdown(f"评估未能启动：{error}"))
        except Exception:  # 防止界面中断，且不暴露本地路径等环境细节
            st.session_state.pop("quality_report", None)
            _clear_agent_state()
            _clear_rule_state(preserve_rag_binding=True)
            st.error("评估未能启动：运行环境或临时文件不可用，请重试。")

report = st.session_state.get("quality_report")
if report is None:
    st.info(
        "请先从左侧上传 CSV、Excel、JSON、JSONL 或 GeoJSON 文件；"
        "如果需要自定义业务规则，可直接在首页与大模型对话创建。"
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
    pre_evaluation_rule_result = st.session_state.get(
        PRE_EVALUATION_RULE_RESULT_KEY
    )
    if pre_evaluation_rule_result is not None:
        with st.container(border=True):
            st.markdown("### 评估前生成规则的执行结果")
            st.caption(
                "以下业务规则在最终评估前已完成生成、完整性校验、试运行和人工审批。"
            )
            _render_rule_result(pre_evaluation_rule_result)
    (
        risk_tab,
        metric_tab,
        profile_tab,
        execution_tab,
        agent_tab,
        rag_tab,
        authoring_tab,
        rule_tab,
    ) = st.tabs(
        [
            "风险提示",
            "指标明细",
            "字段画像",
            "无法评估与运行信息",
            "Agent 解读",
            "标准依据 RAG",
            "补充评价标准",
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
    with rag_tab:
        _render_rag_panel(
            selected_metric_ids=selected_metric_ids,
        )
    with authoring_tab:
        _render_rule_authoring(
            report,
            uploaded_file=uploaded_file,
            dataset_name=dataset_name,
            sheet_name=sheet_name,
            reference_date=reference_date,
            selected_metric_ids=selected_metric_ids,
        )
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
