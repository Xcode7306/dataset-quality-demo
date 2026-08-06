"""最终示例报告与固定评估基准的回归测试。"""

from datetime import date
import json
from pathlib import Path
import unittest

from src.workflow import build_profile_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = PROJECT_ROOT / "sample_data"
REPORTS = PROJECT_ROOT / "reports"
REFERENCE_DATE = date(2026, 7, 17)


class ReportArtifactTests(unittest.TestCase):
    def test_checked_in_sample_reports_match_current_engine(self):
        cases = (
            ("good_dataset.csv", "good_report.json"),
            ("bad_dataset.csv", "bad_report.json"),
            ("format_messy_dataset.csv", "format_messy_report.json"),
            ("minimal_dataset.json", "json_report.json"),
            ("good_dataset.xlsx", "excel_report.json"),
        )

        for sample_name, report_name in cases:
            with self.subTest(report=report_name):
                generated = build_profile_report(
                    SAMPLES / sample_name,
                    reference_date=REFERENCE_DATE,
                ).to_dict()
                saved = json.loads(
                    (REPORTS / report_name).read_text(encoding="utf-8")
                )
                self.assertEqual(saved, generated)


if __name__ == "__main__":
    unittest.main()
