"""v0.4 引导式 RulePack 的真实 Streamlit 控件回归。"""

from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.rule_engine import RulePackExecutionError


PROJECT_ROOT = Path(__file__).parents[1]
REFERENCE_DATE = date(2026, 7, 17)


class RuleStreamlitAppTests(unittest.TestCase):
    @staticmethod
    def _button(app, label):
        return next(button for button in app.button if button.label == label)

    @staticmethod
    def _by_label(elements, label):
        return next(element for element in elements if element.label == label)

    def _evaluated_app(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.csv"
        app = AppTest.from_file(
            str(PROJECT_ROOT / "app.py"),
            default_timeout=60,
        )
        app.run()
        next(
            item for item in app.file_uploader if item.label == "选择数据文件"
        ).set_value(
            (sample.name, sample.read_bytes(), "text/csv")
        )
        app.run()
        app.date_input[0].set_value(REFERENCE_DATE)
        self._button(app, "运行质量评估").click().run()
        self.assertFalse(app.exception)
        return app

    def _draft_app(self):
        app = self._evaluated_app()
        self._button(app, "开始配置业务规则").click().run()
        self._by_label(
            app.multiselect,
            "主键字段（可组合，最多 5 个）",
        ).set_value(["record_id"])
        self._by_label(
            app.multiselect,
            "必填字段",
        ).set_value(["service_name"])
        self._by_label(
            app.selectbox,
            "更新时间字段",
        ).set_value("update_time")
        self._by_label(
            app.selectbox,
            "数值范围字段（可选）",
        ).set_value("handling_days")
        app.run()
        self._by_label(
            app.text_input,
            "数值下限（闭区间，可留空）",
        ).set_value("0")
        self._by_label(
            app.text_input,
            "数值上限（闭区间，可留空）",
        ).set_value("30")
        self._button(app, "生成并校验规则草案").click().run()
        self.assertFalse(app.exception)
        return app

    def test_rule_guidance_and_execution_are_both_user_triggered(self):
        app = self._evaluated_app()
        state = app.session_state["rule_ui_state"]
        self.assertFalse(state["guidance_started"])
        self.assertIsNone(state["draft"])
        self.assertIsNone(state["result"])
        self.assertNotIn(
            "生成并校验规则草案",
            [button.label for button in app.button],
        )

        self._button(app, "开始配置业务规则").click().run()
        self.assertTrue(
            app.session_state["rule_ui_state"]["guidance_started"]
        )
        self.assertIsNone(app.session_state["rule_ui_state"]["draft"])
        self.assertIsNone(app.session_state["rule_ui_state"]["result"])
        self.assertTrue(
            any(
                "不会启用或执行任何规则" in message.value
                for message in app.info
            )
            or "生成并校验规则草案"
            in [button.label for button in app.button]
        )

    def test_explicit_approval_recomputes_without_mutating_baseline(self):
        app = self._draft_app()
        state = app.session_state["rule_ui_state"]
        self.assertEqual(state["draft"].status, "draft")
        self.assertIsNone(state["result"])
        baseline_before = app.session_state["quality_report"].to_dict()
        self.assertIn(
            "下载规则草案（JSON）",
            [button.label for button in app.download_button],
        )

        self._by_label(
            app.text_input,
            "审批人标识（本地自声明）",
        ).set_value("测试审批人")
        self._by_label(
            app.checkbox,
            "我已核对当前 RulePack，并明确批准其仅用于当前绑定输入的确定性重评。",
        ).check()
        app.run()
        approve = self._button(app, "批准并重新评估")
        self.assertFalse(approve.disabled)
        approve.click().run()

        self.assertFalse(app.exception)
        state = app.session_state["rule_ui_state"]
        result = state["result"]
        self.assertIsNotNone(result)
        self.assertEqual(state["approved_pack"].status, "approved")
        self.assertFalse(
            state["approved_pack"].approval.identity_verified
        )
        self.assertEqual(
            app.session_state["quality_report"].to_dict(),
            baseline_before,
        )
        baseline_metric_count = len(result.baseline_report.metrics)
        self.assertEqual(
            result.enhanced_report.metrics[:baseline_metric_count],
            result.baseline_report.metrics,
        )
        self.assertTrue(
            all(
                metric_key.startswith("metric:business_")
                for metric_key in result.diff.added_metric_keys
            )
        )
        self.assertTrue(
            {
                "下载已审批 RulePack（JSON）",
                "下载规则增强结果（JSON）",
                "下载规则增强报告（Markdown）",
                "下载规则问题位置（CSV）",
            }.issubset(
                {button.label for button in app.download_button}
            )
        )

    def test_rule_edit_and_input_change_invalidate_old_state(self):
        app = self._draft_app()
        self.assertIsNotNone(app.session_state["rule_ui_state"]["draft"])

        self._by_label(
            app.multiselect,
            "必填字段",
        ).set_value(["service_name", "department"])
        app.run()
        state = app.session_state["rule_ui_state"]
        self.assertIsNone(state["draft"])
        self.assertIsNone(state["approved_pack"])
        self.assertIsNone(state["result"])
        self.assertTrue(
            any(
                "旧草案、审批和增强结果已失效" in message.value
                for message in app.info
            )
        )

        sample = PROJECT_ROOT / "sample_data" / "minimal_dataset.json"
        next(
            item for item in app.file_uploader if item.label == "选择数据文件"
        ).set_value(
            (sample.name, sample.read_bytes(), "application/json")
        )
        app.run()
        self.assertNotIn(
            "rule_ui_state",
            app.session_state.filtered_state,
        )
        self.assertNotIn(
            "quality_report",
            app.session_state.filtered_state,
        )

    def test_disabled_approval_event_cannot_bypass_confirmation(self):
        app = self._draft_app()
        self._by_label(
            app.text_input,
            "审批人标识（本地自声明）",
        ).set_value("测试审批人")
        app.run()
        approve = self._button(app, "批准并重新评估")
        self.assertTrue(approve.disabled)

        approve.click().run()

        self.assertFalse(app.exception)
        state = app.session_state["rule_ui_state"]
        self.assertIsNone(state["approved_pack"])
        self.assertIsNone(state["result"])

    def test_new_draft_requires_a_new_hash_bound_confirmation(self):
        app = self._draft_app()
        self._by_label(
            app.text_input,
            "审批人标识（本地自声明）",
        ).set_value("测试审批人")
        self._by_label(
            app.checkbox,
            "我已核对当前 RulePack，并明确批准其仅用于当前绑定输入的确定性重评。",
        ).check()
        app.run()
        old_confirmation = app.session_state["rule_ui_state"][
            "confirmed_draft_sha256"
        ]
        self.assertIsNotNone(old_confirmation)

        self._by_label(
            app.text_input,
            "规则包版本",
        ).set_value("1.0.1")
        app.run()
        self._button(app, "生成并校验规则草案").click().run()

        self.assertFalse(app.exception)
        state = app.session_state["rule_ui_state"]
        self.assertIsNotNone(state["draft"])
        self.assertIsNone(state["confirmed_draft_sha256"])
        self.assertFalse(
            self._by_label(
                app.checkbox,
                "我已核对当前 RulePack，并明确批准其仅用于当前绑定输入的确定性重评。",
            ).value
        )
        self.assertTrue(self._button(app, "批准并重新评估").disabled)

    def test_oversized_rule_number_is_a_validation_error_not_app_exception(self):
        app = self._draft_app()
        self._by_label(
            app.text_input,
            "数值下限（闭区间，可留空）",
        ).set_value("1" + "0" * 1000)
        app.run()
        self._button(app, "生成并校验规则草案").click().run()

        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state["rule_ui_state"]["draft"])
        self.assertTrue(
            any("1e308" in message.value for message in app.error)
        )

    def test_approved_pack_is_retained_when_execution_fails(self):
        app = self._draft_app()
        self._by_label(
            app.text_input,
            "审批人标识（本地自声明）",
        ).set_value("测试审批人")
        self._by_label(
            app.checkbox,
            "我已核对当前 RulePack，并明确批准其仅用于当前绑定输入的确定性重评。",
        ).check()
        app.run()

        with patch(
            "src.rule_service.evaluate_uploaded_dataset_with_rule_pack",
            side_effect=RulePackExecutionError(("forced failure",)),
        ):
            self._button(app, "批准并重新评估").click().run()

        self.assertFalse(app.exception)
        state = app.session_state["rule_ui_state"]
        self.assertIsNotNone(state["approved_pack"])
        self.assertIsNone(state["result"])
        self.assertEqual(state["execution_error"], "forced failure")
        self.assertIn(
            "下载已审批 RulePack（JSON）",
            [button.label for button in app.download_button],
        )


if __name__ == "__main__":
    unittest.main()
