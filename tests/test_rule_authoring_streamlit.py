"""v0.7 补充评价标准与规则 Agent 的 Streamlit 闭环测试。"""

from datetime import date
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from src.metric_catalog import ALL_METRIC_IDS, get_metric_definition


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DATE = date(2026, 7, 17)
TARGET_METRIC_ID = "db31_020100"


class RuleAuthoringStreamlitTests(unittest.TestCase):
    @staticmethod
    def _button(app, label):
        return next(button for button in app.button if button.label == label)

    @staticmethod
    def _by_label(elements, label):
        return next(element for element in elements if element.label == label)

    def test_metric_evidence_agent_compiles_dry_runs_and_approves(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        app.run()
        self.assertFalse(app.exception)

        self._button(app, "清空选择").click().run()
        target_name = get_metric_definition(TARGET_METRIC_ID)["name"]
        target_checkbox = next(
            checkbox for checkbox in app.checkbox if checkbox.label == target_name
        )
        target_checkbox.set_value(True)
        app.run()

        sample = ROOT / "sample_data" / "good_dataset.csv"
        next(
            item for item in app.file_uploader if item.label == "选择数据文件"
        ).set_value(
            (sample.name, sample.read_bytes(), "text/csv")
        )
        app.run()
        app.text_area[ALL_METRIC_IDS.index(TARGET_METRIC_ID)].set_value(
            "service_name为必填字段"
        )
        app.run()
        app.date_input[0].set_value(REFERENCE_DATE)
        self._button(app, "运行质量评估").click().run()
        self.assertFalse(app.exception)
        self.assertIn("补充评价标准", [tab.label for tab in app.tabs])

        self.assertEqual(len(app.text_area), 1)
        app.text_area[0].set_value("service_name为必填字段")
        app.run()
        self._button(app, "AI 解析依据").click().run()

        state = app.session_state["rule_authoring_ui_state"]
        draft = state["drafts"][TARGET_METRIC_ID]
        self.assertEqual(draft.rule_spec.rule_type, "required")
        self.assertIn("查看结构化规则草案", [item.label for item in app.expander])

        self._button(app, "试运行规则").click().run()
        state = app.session_state["rule_authoring_ui_state"]
        self.assertIn(TARGET_METRIC_ID, state["dry_runs"])
        self.assertTrue(
            any("规则试运行完成" in item.value for item in app.success)
        )

        self._by_label(
            app.text_input,
            "审批人标识（AI规则，本地自声明）",
        ).set_value("v0.7 测试审批人")
        self._by_label(
            app.checkbox,
            "我已核对当前规则、评价依据和试运行摘要，并批准本次确定性重评。",
        ).check()
        app.run()
        approve = self._button(app, "补充完成并重新评估（AI规则）")
        self.assertFalse(approve.disabled)
        approve.click().run()

        state = app.session_state["rule_authoring_ui_state"]
        self.assertEqual(state["approved_packs"][TARGET_METRIC_ID].status, "approved")
        self.assertIsNotNone(state["results"][TARGET_METRIC_ID])
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
