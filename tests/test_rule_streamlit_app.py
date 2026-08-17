"""报告后旧规则增强页签已移除的 Streamlit 回归。"""

from datetime import date
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).parents[1]
REFERENCE_DATE = date(2026, 7, 17)


class LegacyRuleEnhancementVisibilityTests(unittest.TestCase):
    def test_report_does_not_render_legacy_rule_enhancement_surface(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.csv"
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60).run()
        next(
            item for item in app.file_uploader if item.label == "选择数据文件"
        ).set_value((sample.name, sample.read_bytes(), "text/csv"))
        app.run()
        app.date_input[0].set_value(REFERENCE_DATE)
        next(
            button for button in app.button if button.label == "运行质量评估"
        ).click().run()

        self.assertFalse(app.exception)
        self.assertNotIn("规则增强", [tab.label for tab in app.tabs])
        self.assertFalse(
            any(button.label == "开始配置业务规则" for button in app.button)
        )
        self.assertFalse(
            any(
                label in {"主键字段（可组合，最多 5 个）", "必填字段"}
                for label in (item.label for item in app.multiselect)
            )
        )


if __name__ == "__main__":
    unittest.main()
