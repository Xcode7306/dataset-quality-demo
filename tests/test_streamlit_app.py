"""Streamlit 页面真实控件交互测试。"""

from datetime import date
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

from streamlit.testing.v1 import AppTest

from src.metric_catalog import (
    ALL_METRIC_IDS,
    METRIC_BY_ID,
    ORIGINAL_METRIC_IDS,
)


PROJECT_ROOT = Path(__file__).parents[1]
REFERENCE_DATE = date(2026, 7, 17)


class StreamlitAppTests(unittest.TestCase):
    @staticmethod
    def _button_by_label(app, label):
        return next(button for button in app.button if button.label == label)

    @staticmethod
    def _metric_checkbox(app, metric_id):
        definition = METRIC_BY_ID[metric_id]
        return next(
            element
            for element in app.checkbox
            if element.label == definition["name"]
        )

    @classmethod
    def _selected_metric_ids(cls, app):
        return tuple(
            metric_id
            for metric_id in ALL_METRIC_IDS
            if cls._metric_checkbox(app, metric_id).value
        )

    def _new_app(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)
        app.run()
        self.assertFalse(app.exception)
        return app

    def _upload_and_run(
        self,
        app,
        file_name,
        content,
        mime_type,
        dataset_name="",
        sheet_name=None,
    ):
        next(
            item for item in app.file_uploader if item.label == "选择数据文件"
        ).set_value((file_name, content, mime_type))
        app.run()
        app.date_input[0].set_value(REFERENCE_DATE)
        if dataset_name:
            next(
                item for item in app.text_input if item.label == "数据集名称（可选）"
            ).set_value(dataset_name)
        if sheet_name is not None:
            next(
                item for item in app.text_input if item.label == "工作表名称（可选）"
            ).set_value(sheet_name)
        run_button = self._button_by_label(app, "运行质量评估")
        self.assertFalse(run_button.disabled)
        run_button.click().run()
        self.assertFalse(app.exception)
        return app

    def _assert_report_surface(self, app, expected_type, expected_sheet=None):
        report = app.session_state["quality_report"]
        self.assertEqual(report.status, "success")
        self.assertEqual(report.dataset.file_type, expected_type)
        self.assertEqual(report.dataset.sheet_name, expected_sheet)
        self.assertEqual(len({metric.id for metric in report.metrics}), 13)
        self.assertEqual(len(app.metric), 5)
        self.assertGreaterEqual(len(app.dataframe), 2)
        metric_tables = [
            table.value
            for table in app.dataframe
            if {"指标名称", "字段名称", "状态"}.issubset(
                set(table.value.columns)
            )
        ]
        self.assertEqual(1, len(metric_tables))
        self.assertNotIn("引用键", metric_tables[0].columns)
        self.assertEqual(len(app.download_button), 3)
        self.assertNotIn(
            "疑似问题位置",
            [tab.label for tab in app.tabs],
        )
        self.assertTrue(any(item.value == "风险分布" for item in app.subheader))
        self.assertGreaterEqual(len(app.get("vega_lite_chart")), 1)
        self.assertEqual(
            [button.label for button in app.download_button],
            [
                "下载结构化报告（JSON）",
                "下载评估报告（Markdown）",
                "下载疑似问题位置（CSV）",
            ],
        )

    def test_reference_date_is_explicit_and_flows_to_report(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.csv"
        app = self._new_app()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "text/csv",
        )

        lag_metric = next(
            metric
            for metric in app.session_state["quality_report"].metrics
            if metric.id == "update_lag_days"
        )
        self.assertEqual(
            lag_metric.evidence["reference_date"], REFERENCE_DATE.isoformat()
        )

    def test_metric_selector_uses_one_unified_card_list_and_presets(self):
        app = self._new_app()
        self.assertEqual(len(app.multiselect), 0)
        self.assertEqual(len(app.checkbox), len(ALL_METRIC_IDS))
        self.assertTrue(
            app.session_state["metric_evidence_statistical_outlier_rate"]
        )
        self.assertEqual(app.session_state["metric_evidence_db31_010100"], "")
        self.assertEqual(self._selected_metric_ids(app), ORIGINAL_METRIC_IDS)
        self.assertTrue(
            any("metric-help-icon" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any("data-tooltip" in item.value for item in app.markdown)
        )
        self.assertFalse(
            any("原 v0.4" in checkbox.label for checkbox in app.checkbox)
        )
        self.assertFalse(
            any("DB31/T" in checkbox.label for checkbox in app.checkbox)
        )

        self._button_by_label(app, "全部指标").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(self._selected_metric_ids(app), ALL_METRIC_IDS)

        self._button_by_label(app, "清空选择").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(self._selected_metric_ids(app), ())

        self._button_by_label(app, "默认指标").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(self._selected_metric_ids(app), ORIGINAL_METRIC_IDS)

    def test_initial_upload_guidance_omits_zip_text(self):
        app = self._new_app()

        guidance = [item.value for item in app.info]
        self.assertTrue(any("请先从左侧上传" in value for value in guidance))
        self.assertTrue(all("ZIP" not in value and ".zip" not in value for value in guidance))

    def test_custom_model_api_settings_are_available_on_initial_page(self):
        app = self._new_app()
        labels = [item.label for item in app.text_input]
        self.assertIn("API 地址", labels)
        self.assertIn("API Key", labels)
        self.assertIn("模型名称", labels)

        next(
            item
            for item in app.text_input
            if item.label == "API 地址"
        ).set_value("https://model.example/v1")
        next(
            item for item in app.text_input if item.label == "API Key"
        ).set_value("page-only-test-key")
        next(
            item for item in app.text_input if item.label == "模型名称"
        ).set_value("custom-chat-model")
        app.run()
        self._button_by_label(app, "保存 API Key").click().run()

        self.assertFalse(app.exception)
        self.assertTrue(
            any("API Key 已保存到当前会话" in item.value for item in app.success)
        )

    def test_saved_model_api_key_survives_upload_until_explicit_clear(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.csv"
        app = self._new_app()
        next(
            item for item in app.text_input if item.label == "API Key"
        ).set_value("page-only-test-key")
        app.run()
        self._button_by_label(app, "保存 API Key").click().run()

        self.assertEqual(
            "page-only-test-key",
            app.session_state["model_api_key"],
        )
        next(
            item for item in app.file_uploader if item.label == "选择数据文件"
        ).set_value((sample.name, sample.read_bytes(), "text/csv"))
        app.run()

        self.assertFalse(app.exception)
        self.assertEqual(
            "page-only-test-key",
            app.session_state["model_api_key"],
        )
        self.assertEqual(
            "page-only-test-key",
            app.session_state["model_api_key_input"],
        )
        self._button_by_label(app, "清除 API Key").click().run()

        self.assertFalse(app.exception)
        self.assertEqual("", app.session_state["model_api_key"])
        self.assertEqual("", app.session_state["model_api_key_input"])

    def test_saved_model_api_key_can_retry_after_credential_store_warning(self):
        app = self._new_app()
        next(
            item for item in app.text_input if item.label == "API Key"
        ).set_value("page-only-test-key")
        app.run()
        self._button_by_label(app, "保存 API Key").click().run()

        app.session_state["model_api_key_store_message"] = (
            "warning",
            "API Key 已保存在当前会话，但系统凭据库暂时不可用。",
        )
        app.run()

        self.assertFalse(
            self._button_by_label(app, "保存 API Key").disabled
        )

    def test_model_api_key_rejects_non_ascii_input_before_save(self):
        app = self._new_app()
        next(
            item for item in app.text_input if item.label == "API Key"
        ).set_value("API Key：错误内容")
        app.run()

        self.assertFalse(app.exception)
        self.assertEqual("", app.session_state["model_api_key"])
        self.assertTrue(self._button_by_label(app, "保存 API Key").disabled)
        self.assertTrue(
            any("ASCII 字符" in item.value for item in app.warning)
        )

    def test_custom_rule_provider_error_hides_saved_api_key(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.csv"
        secret = "page-secret-value-12345678"
        app = self._new_app()
        next(
            item for item in app.text_input if item.label == "API Key"
        ).set_value(secret)
        app.run()
        self._button_by_label(app, "保存 API Key").click().run()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "text/csv",
        )
        next(
            item for item in app.text_input if item.label == "自定义规则描述"
        ).set_value("service_name为必填字段")
        app.run()

        with patch(
            "src.rule_authoring_coordinator.compile_rule_authoring_run",
            side_effect=ValueError(f"provider rejected API Key {secret}"),
        ):
            self._button_by_label(app, "AI 解析自定义规则").click().run()

        self.assertFalse(app.exception)
        self.assertFalse(any(secret in item.value for item in app.error))
        self.assertNotIn(
            secret,
            app.session_state["custom_rule_ui_state"]["error"],
        )

    def test_selected_metric_without_evidence_blocks_run(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.csv"
        app = self._new_app()
        for metric_id in ALL_METRIC_IDS:
            self._metric_checkbox(app, metric_id).set_value(False)
        self._metric_checkbox(app, "db31_010100").set_value(True)
        app.run()

        next(
            item for item in app.file_uploader if item.label == "选择数据文件"
        ).set_value(
            (sample.name, sample.read_bytes(), "text/csv")
        )
        app.run()

        run_button = self._button_by_label(app, "运行质量评估")
        self.assertTrue(run_button.disabled)
        self.assertTrue(
            any("补全评价规则" in warning.value for warning in app.warning)
        )

    def test_custom_metric_mix_is_the_exact_report_selection(self):
        sample = PROJECT_ROOT / "sample_data" / "bad_dataset.csv"
        app = self._new_app()
        for metric_id in ORIGINAL_METRIC_IDS:
            self._metric_checkbox(app, metric_id).set_value(False)
        for metric_id in (
            "db31_030300",
            "exact_duplicate_rate",
            "db31_010101",
            "db31_030400",
        ):
            self._metric_checkbox(app, metric_id).set_value(True)
        app.run()
        app.text_area[ALL_METRIC_IDS.index("db31_010101")].set_value(
            "按各字段约定的数据类型进行校验"
        )
        app.run()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "text/csv",
        )

        report = app.session_state["quality_report"]
        expected = [
            "exact_duplicate_rate",
            "db31_010101",
            "db31_030300",
            "db31_030400",
        ]
        self.assertEqual(
            [metric.id for metric in report.metrics],
            expected,
        )
        self.assertEqual(
            report.evaluation_context["selected_metric_ids"],
            expected,
        )
        self.assertEqual(
            next(
                metric
                for metric in report.metrics
                if metric.id == "db31_010101"
            ).status,
            "not_assessable",
        )
        self.assertEqual(
            len({metric.metric_key for metric in report.metrics}),
            4,
        )
        metric_table = next(
            table.value
            for table in app.dataframe
            if {
                "指标名称",
                "字段名称",
                "计算方式",
            }.issubset(set(table.value.columns))
        )
        self.assertNotIn("来源", metric_table.columns)
        self.assertNotIn("标准代码", metric_table.columns)

    def test_report_identifies_missing_standard_and_exposes_completion_page(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.csv"
        app = self._new_app()
        for metric_id in ORIGINAL_METRIC_IDS:
            self._metric_checkbox(app, metric_id).set_value(False)
        self._metric_checkbox(app, "db31_010101").set_value(True)
        app.run()
        app.text_area[ALL_METRIC_IDS.index("db31_010101")].set_value(
            "按各字段约定的数据类型进行校验"
        )
        app.run()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "text/csv",
        )

        self.assertTrue(
            any("需补充评价标准 1 项" in item.value for item in app.caption)
        )
        self.assertTrue(
            any("数据类型约束规范性" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any("逐字段预期数据类型标准" in item.value for item in app.warning)
        )
        self.assertIn("补充评价标准", [tab.label for tab in app.tabs])

    def test_empty_metric_selection_disables_run_and_clears_old_state(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.csv"
        app = self._new_app()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "text/csv",
        )
        self._button_by_label(app, "概括结果").click().run()
        self._button_by_label(app, "开始配置业务规则").click().run()
        self.assertIn("quality_report", app.session_state.filtered_state)
        self.assertIn("agent_ui_state", app.session_state.filtered_state)
        self.assertIn("rule_ui_state", app.session_state.filtered_state)

        next(
            item for item in app.file_uploader if item.label == "选择数据文件"
        ).set_value(None)
        app.run()
        self.assertFalse(app.exception)
        self.assertNotIn("quality_report", app.session_state.filtered_state)
        self.assertEqual(len(app.checkbox), len(ALL_METRIC_IDS))

        for metric_id in ALL_METRIC_IDS:
            self._metric_checkbox(app, metric_id).set_value(False)
        app.run()

        self.assertFalse(app.exception)
        self.assertTrue(
            self._button_by_label(app, "运行质量评估").disabled
        )
        self.assertNotIn("quality_report", app.session_state.filtered_state)
        self.assertNotIn("agent_ui_state", app.session_state.filtered_state)
        self.assertNotIn("rule_ui_state", app.session_state.filtered_state)
        self.assertEqual(len(app.download_button), 0)
        self.assertTrue(
            any(
                "至少选择一个评价指标" in warning.value
                for warning in app.warning
            )
        )

    def test_evaluation_hides_cards_until_the_uploaded_file_is_removed(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.csv"
        app = self._new_app()
        for metric_id in ORIGINAL_METRIC_IDS:
            self._metric_checkbox(app, metric_id).set_value(False)
        self._metric_checkbox(app, "exact_duplicate_rate").set_value(True)
        self._metric_checkbox(app, "db31_030300").set_value(True)
        app.run()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "text/csv",
        )
        self.assertEqual(len(app.checkbox), 0)
        self.assertIn("quality_report", app.session_state.filtered_state)
        next(
            item for item in app.file_uploader if item.label == "选择数据文件"
        ).set_value(None)
        app.run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.checkbox), len(ALL_METRIC_IDS))
        self.assertNotIn("quality_report", app.session_state.filtered_state)

    def test_supported_formats_run_through_full_report_surface(self):
        samples = PROJECT_ROOT / "sample_data"
        cases = (
            ("good_dataset.csv", "text/csv", "csv", None),
            (
                "good_dataset.xls",
                "application/vnd.ms-excel",
                "xls",
                "服务事项",
            ),
            (
                "good_dataset.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "xlsx",
                "服务事项",
            ),
            ("minimal_dataset.json", "application/json", "json", None),
            (
                "geojson_feature_collection.geojson",
                "application/geo+json",
                "geojson",
                None,
            ),
        )
        for file_name, mime_type, expected_type, sheet_name in cases:
            with self.subTest(file_name=file_name):
                app = self._new_app()
                self._upload_and_run(
                    app,
                    file_name,
                    (samples / file_name).read_bytes(),
                    mime_type,
                    dataset_name=f"验收-{expected_type}",
                    sheet_name=sheet_name,
                )
                self._assert_report_surface(app, expected_type, sheet_name)

    def test_changing_upload_clears_stale_report_before_next_run(self):
        samples = PROJECT_ROOT / "sample_data"
        app = self._new_app()
        self._upload_and_run(
            app,
            "good_dataset.csv",
            (samples / "good_dataset.csv").read_bytes(),
            "text/csv",
        )
        self.assertEqual(app.session_state["quality_report"].dataset.file_type, "csv")

        next(
            item for item in app.file_uploader if item.label == "选择数据文件"
        ).set_value(
            (
                "minimal_dataset.json",
                (samples / "minimal_dataset.json").read_bytes(),
                "application/json",
            )
        )
        app.run()
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.download_button), 0)
        self.assertNotIn("agent_ui_state", app.session_state.filtered_state)

        self._button_by_label(app, "运行质量评估").click().run()
        self._assert_report_surface(app, "json")

    def test_agent_is_user_triggered_read_only_and_uses_template_by_default(self):
        sample = PROJECT_ROOT / "sample_data" / "bad_dataset.csv"
        app = self._new_app()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "text/csv",
        )

        state = app.session_state["agent_ui_state"]
        self.assertIsNone(state["latest_analysis"])
        report_before = app.session_state["quality_report"].to_dict()
        self.assertIn("Agent 解读", [tab.label for tab in app.tabs])
        self.assertFalse(
            any(
                {
                    "疑似问题类型",
                    "字段名称",
                    "数据记录序号",
                }.issubset(set(table.value.columns))
                for table in app.dataframe
            )
        )

        with patch.dict(
            os.environ,
            {"QUALITY_AGENT_PROVIDER": "template"},
            clear=False,
        ):
            self._button_by_label(app, "概括结果").click().run()

        self.assertFalse(app.exception)
        analysis = app.session_state["agent_ui_state"]["latest_analysis"]
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.audit.mode, "template")
        self.assertGreater(len(analysis.citations), 0)
        self.assertEqual(
            report_before,
            app.session_state["quality_report"].to_dict(),
        )

    def test_deepseek_mode_is_disclosed_before_any_agent_request(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.csv"
        app = self._new_app()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "text/csv",
        )

        with patch.dict(
            os.environ,
            {"QUALITY_AGENT_PROVIDER": "deepseek"},
            clear=False,
        ):
            os.environ.pop("DEEPSEEK_API_KEY", None)
            app.run()

        self.assertIsNone(
            app.session_state["agent_ui_state"]["latest_analysis"]
        )
        self.assertTrue(
            any(
                "尚未配置 DEEPSEEK_API_KEY" in message.value
                and "调用前必须补充 API Key" in message.value
                and "不会回退到本地模板" in message.value
                for message in app.warning
            )
        )

    def test_deepseek_mode_discloses_external_projection_when_key_exists(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.csv"
        app = self._new_app()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "text/csv",
        )

        with patch.dict(
            os.environ,
            {
                "QUALITY_AGENT_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "test-only-key",
            },
            clear=False,
        ):
            app.run()

        self.assertIsNone(
            app.session_state["agent_ui_state"]["latest_analysis"]
        )
        self.assertTrue(
            any(
                "已配置 DeepSeek 外部模式" in message.value
                and "白名单过滤的报告投影" in message.value
                for message in app.warning
            )
        )

    def test_agent_question_history_is_bound_to_report_and_same_input_rerun_clears_it(self):
        sample = PROJECT_ROOT / "sample_data" / "minimal_dataset.json"
        app = self._new_app()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "application/json",
        )

        with patch.dict(
            os.environ,
            {"QUALITY_AGENT_PROVIDER": "template"},
            clear=False,
        ):
            app.chat_input[0].set_value("为什么有无法评估项？").run()

        state = app.session_state["agent_ui_state"]
        self.assertEqual(len(state["history"]), 1)
        self.assertTrue(state["history"][0]["is_question"])
        self.assertGreaterEqual(len(app.chat_message), 2)

        next(
            item for item in app.file_uploader if item.label == "选择数据文件"
        ).set_value(None)
        app.run()
        next(
            item for item in app.file_uploader if item.label == "选择数据文件"
        ).set_value(
            (sample.name, sample.read_bytes(), "application/json")
        )
        app.run()
        self._button_by_label(app, "运行质量评估").click().run()

        self.assertFalse(app.exception)
        reset_state = app.session_state["agent_ui_state"]
        self.assertEqual(reset_state["history"], [])
        self.assertIsNone(reset_state["latest_analysis"])

    def test_agent_request_failure_does_not_present_the_previous_answer_as_current(self):
        sample = PROJECT_ROOT / "sample_data" / "bad_dataset.csv"
        app = self._new_app()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "text/csv",
        )
        with patch.dict(
            os.environ,
            {"QUALITY_AGENT_PROVIDER": "template"},
            clear=False,
        ):
            self._button_by_label(app, "概括结果").click().run()
        self.assertIsNotNone(
            app.session_state["agent_ui_state"]["latest_analysis"]
        )

        with patch(
            "src.agent_service.run_agent",
            side_effect=RuntimeError("模拟 Agent 故障"),
        ):
            app.chat_input[0].set_value("这次请求会失败").run()

        state = app.session_state["agent_ui_state"]
        self.assertIsNone(state["latest_analysis"])
        self.assertEqual(len(state["history"]), 0)
        self.assertTrue(
            any(
                "外部模型解读失败" in message.value
                and "未生成模板替代结果" in message.value
                and "模拟 Agent 故障" in message.value
                for message in app.error
            )
        )

    def test_agent_evidence_uses_plain_text_for_untrusted_field_names(self):
        malicious_field = "![x](https://example.invalid/pixel.png)"
        content = f"{malicious_field},other\n,1\n,2\n".encode("utf-8")
        app = self._new_app()
        self._upload_and_run(
            app,
            "untrusted-field.csv",
            content,
            "text/csv",
        )

        with patch.dict(
            os.environ,
            {"QUALITY_AGENT_PROVIDER": "template"},
            clear=False,
        ):
            self._button_by_label(app, "概括结果").click().run()

        self.assertTrue(
            any(malicious_field in item.value for item in app.text)
        )
        self.assertFalse(
            any(malicious_field in item.value for item in app.caption)
        )
        self.assertFalse(
            any(malicious_field in item.value for item in app.markdown)
        )

    def test_jsonl_runs_through_full_report_surface_and_shows_warning(self):
        sample = PROJECT_ROOT / "sample_data" / "json_records_dataset.jsonl"
        app = self._new_app()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "application/x-ndjson",
        )

        self._assert_report_surface(app, "jsonl")
        report = app.session_state["quality_report"]
        self.assertTrue(
            any("JSON Lines" in item for item in report.execution["warnings"])
        )

    def test_json_zip_runs_through_full_report_surface_and_shows_warning(self):
        content = io.BytesIO()
        with ZipFile(content, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("001.json", '[["id","name"],[1,"A"]]')
            archive.writestr("002.json", '[["id","name"],[2,"B"]]')

        app = self._new_app()
        self._upload_and_run(
            app,
            "records.zip",
            content.getvalue(),
            "application/zip",
        )

        self._assert_report_surface(app, "zip")
        report = app.session_state["quality_report"]
        self.assertEqual(report.profile["row_count"], 2)
        self.assertTrue(
            any("ZIP JSON 分片包" in item for item in report.execution["warnings"])
        )

    def test_nested_json_returns_downloadable_failed_report(self):
        sample = PROJECT_ROOT / "sample_data" / "nested_dataset.json"
        app = self._new_app()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "application/json",
        )

        report = app.session_state["quality_report"]
        self.assertEqual(report.status, "failed")
        self.assertIn("嵌套", report.execution["errors"][0])
        self.assertEqual(len({metric.id for metric in report.metrics}), 13)
        self.assertEqual(len(app.download_button), 3)

    def test_missing_excel_sheet_returns_explainable_failed_report(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.xlsx"
        app = self._new_app()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            sheet_name="不存在的工作表",
        )

        report = app.session_state["quality_report"]
        self.assertEqual(report.status, "failed")
        self.assertIn("未找到工作表", report.execution["errors"][0])
        self.assertIn("服务事项", report.execution["errors"][0])
        self.assertEqual(len(app.download_button), 3)


if __name__ == "__main__":
    unittest.main()
