"""第一批 5 项核心指标的回归测试。"""

import tempfile
import unittest
from pathlib import Path

from src.workflow import build_profile_report


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "sample_data"


def find_metric(report, metric_id, field=None):
    for metric in report.metrics:
        if metric.id == metric_id and metric.field == field:
            return metric
    raise AssertionError(f"未找到指标 {metric_id}，字段为 {field}。")


class InitialMetricTests(unittest.TestCase):
    def test_good_dataset_has_expected_initial_metrics(self):
        report = build_profile_report(SAMPLES / "good_dataset.csv")

        self.assertEqual(find_metric(report, "file_parse_rate").value, 1.0)
        self.assertEqual(find_metric(report, "dataset_scale").value, 5)
        self.assertEqual(find_metric(report, "field_missing_rate", "department").value, 0.0)
        self.assertEqual(find_metric(report, "blank_record_rate").value, 0.0)
        self.assertEqual(
            find_metric(report, "field_type_consistency", "update_time").value,
            1.0,
        )

    def test_bad_dataset_exposes_missing_blank_and_mixed_types(self):
        report = build_profile_report(SAMPLES / "bad_dataset.csv")

        self.assertAlmostEqual(
            find_metric(report, "field_missing_rate", "department").value,
            2 / 6,
            places=6,
        )
        blank_rate = find_metric(report, "blank_record_rate")
        self.assertAlmostEqual(blank_rate.value, 1 / 6, places=6)
        self.assertNotIn("record_id", blank_rate.evidence["content_fields"])
        self.assertEqual(
            [
                location["record_number"]
                for location in blank_rate.issue_locations
            ],
            [5],
        )
        department_missing = find_metric(
            report,
            "field_missing_rate",
            "department",
        )
        self.assertEqual(
            [
                location["record_number"]
                for location in department_missing.issue_locations
            ],
            [4, 5],
        )
        self.assertAlmostEqual(
            find_metric(report, "field_type_consistency", "handling_days").value,
            3 / 5,
        )
        self.assertEqual(
            [
                location["record_number"]
                for location in find_metric(
                    report,
                    "field_type_consistency",
                    "handling_days",
                ).issue_locations
            ],
            [3, 6],
        )

    def test_empty_dataset_marks_unavailable_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.csv"
            path.write_text("only_column\n", encoding="utf-8")
            report = build_profile_report(path)

        missing_rate = find_metric(report, "field_missing_rate", "only_column")
        blank_rate = find_metric(report, "blank_record_rate")
        type_consistency = find_metric(report, "field_type_consistency", "only_column")
        self.assertEqual(missing_rate.status, "not_assessable")
        self.assertEqual(blank_rate.status, "not_assessable")
        self.assertEqual(type_consistency.status, "not_assessable")
        self.assertEqual(len(report.not_assessable), 11)
        self.assertTrue(
            {
                "field_missing_rate:only_column",
                "blank_record_rate",
                "field_type_consistency:only_column",
            }.issubset({item.id for item in report.not_assessable})
        )

    def test_empty_json_still_reports_all_13_metric_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.json"
            path.write_text("[]", encoding="utf-8")
            report = build_profile_report(path)

        metric_ids = {metric.id for metric in report.metrics}
        self.assertEqual(len(metric_ids), 13)
        self.assertIn("field_missing_rate", metric_ids)
        self.assertIn("field_type_consistency", metric_ids)
        self.assertEqual(find_metric(report, "field_missing_rate").status, "not_assessable")
        self.assertEqual(
            find_metric(report, "field_type_consistency").status,
            "not_assessable",
        )

    def test_parse_failure_still_returns_a_failed_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested.json"
            path.write_text('[{"事项": {"名称": "测试"}}]', encoding="utf-8")
            report = build_profile_report(path)

        self.assertEqual(report.status, "failed")
        self.assertEqual(find_metric(report, "file_parse_rate").value, 0.0)
        self.assertEqual(len(report.execution["errors"]), 1)
        self.assertIn("嵌套", report.execution["errors"][0])
        self.assertEqual(len(report.not_assessable), 12)
        self.assertIn("statistical_outlier_rate", {item.id for item in report.not_assessable})
