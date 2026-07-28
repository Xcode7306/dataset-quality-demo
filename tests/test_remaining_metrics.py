"""剩余 8 项零配置指标的回归测试。"""

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.metrics import (
    calculate_exact_duplicate_rate,
    calculate_normalized_duplicate_rate,
    calculate_statistical_outlier_rates,
    calculate_time_info_availability,
    calculate_update_lag_days,
)
from src.presentation import serialize_report
from src.report import save_report
from src.workflow import build_profile_report


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "sample_data"


def find_metric(report, metric_id, field=None):
    for metric in report.metrics:
        if metric.id == metric_id and metric.field == field:
            return metric
    raise AssertionError(f"未找到指标 {metric_id}，字段为 {field}。")


class RemainingMetricTests(unittest.TestCase):
    def test_bad_dataset_has_expected_format_duplicate_time_and_coverage_metrics(self):
        report = build_profile_report(SAMPLES / "bad_dataset.csv")

        self.assertAlmostEqual(
            find_metric(report, "recognizable_format_anomaly_rate", "update_time").value,
            1 / 5,
            places=6,
        )
        self.assertAlmostEqual(
            find_metric(report, "recognizable_format_anomaly_rate", "source_url").value,
            1 / 4,
            places=6,
        )
        self.assertAlmostEqual(
            find_metric(report, "recognizable_format_anomaly_rate", "handling_days").value,
            2 / 5,
            places=6,
        )
        self.assertAlmostEqual(find_metric(report, "exact_duplicate_rate").value, 1 / 6, places=6)
        self.assertAlmostEqual(
            find_metric(report, "normalized_duplicate_rate").value, 1 / 6, places=6
        )
        self.assertAlmostEqual(
            find_metric(report, "time_info_availability").value, 4 / 6, places=6
        )
        lag_metric = find_metric(report, "update_lag_days")
        reference_date = date.fromisoformat(lag_metric.evidence["reference_date"])
        latest_update_date = date.fromisoformat(
            lag_metric.evidence["latest_update_date"]
        )
        self.assertEqual(
            lag_metric.value,
            (reference_date - latest_update_date).days,
        )
        self.assertAlmostEqual(
            find_metric(report, "source_info_coverage").value, 5 / 6, places=6
        )
        self.assertAlmostEqual(
            find_metric(report, "version_info_coverage").value, 5 / 6, places=6
        )

    def test_normalized_duplicate_ignores_whitespace_and_case(self):
        dataframe = pd.DataFrame(
            {
                "record_id": [1, 2, 3],
                "name": ["政务 Service", " 政务service ", "其他事项"],
            }
        )

        self.assertEqual(calculate_exact_duplicate_rate(dataframe).value, 0.0)
        normalized = calculate_normalized_duplicate_rate(dataframe)
        self.assertAlmostEqual(normalized.value, 1 / 3, places=6)
        self.assertEqual(normalized.evidence["duplicate_groups"][0]["row_indices"], [1, 2])

    def test_exact_duplicate_preserves_leading_and_trailing_whitespace(self):
        dataframe = pd.DataFrame(
            {"record_id": [1, 2], "name": ["事项A", " 事项A "]}
        )

        self.assertEqual(calculate_exact_duplicate_rate(dataframe).value, 0.0)
        self.assertEqual(calculate_normalized_duplicate_rate(dataframe).value, 0.5)

    def test_exact_duplicate_preserves_empty_and_whitespace_only_strings(self):
        dataframe = pd.DataFrame(
            {"record_id": [1, 2], "name": ["", " "]}
        )

        self.assertEqual(calculate_exact_duplicate_rate(dataframe).value, 0.0)
        self.assertEqual(calculate_normalized_duplicate_rate(dataframe).value, 0.5)

    def test_normalized_duplicate_does_not_merge_numeric_text(self):
        dataframe = pd.DataFrame(
            {"record_id": [1, 2], "version_code": ["1.0", "10"]}
        )

        self.assertEqual(calculate_normalized_duplicate_rate(dataframe).value, 0.0)

    def test_normalized_duplicate_preserves_structured_text_semantics(self):
        cases = {
            "version": ["v1.0", "v10"],
            "source_url": [
                "https://example.gov.cn/a-b",
                "https://example.gov.cn/ab",
            ],
            "业务代码": ["ABC-12", "ABC12"],
            "资源地址": [
                "https://example.gov.cn/a-b",
                "https://example.gov.cn/ab",
            ],
            "说明": ["v1.0", "v10"],
            "备注": ["ABC-12", "ABC12"],
        }
        for field, values in cases.items():
            with self.subTest(field=field):
                dataframe = pd.DataFrame(
                    {"record_id": [1, 2], field: values}
                )
                self.assertEqual(
                    calculate_normalized_duplicate_rate(dataframe).value,
                    0.0,
                )

    def test_non_finite_numbers_produce_readable_markdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "non_finite.csv"
            input_path.write_text("amount\n1\n2\n3\n4\ninf\n", encoding="utf-8")
            report = build_profile_report(input_path)
            output_path = Path(temp_dir) / "report.md"
            save_report(report, output_path)
            report_text = output_path.read_text(encoding="utf-8")

        format_metric = find_metric(
            report, "recognizable_format_anomaly_rate", "amount"
        )
        outlier_metric = find_metric(report, "statistical_outlier_rate", "amount")
        self.assertEqual(format_metric.value, 0.2)
        self.assertEqual(outlier_metric.value, 0.2)
        self.assertEqual(outlier_metric.evidence["non_finite_count"], 1)
        self.assertIn("# 数据集质量评估报告：non_finite", report_text)
        self.assertNotIn("Infinity", report_text)
        json.loads(
            serialize_report(report).decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "extreme_finite.csv"
            input_path.write_text(
                "amount\n-1e308\n-1e308\n1e308\n1e308\n",
                encoding="utf-8",
            )
            extreme_report = build_profile_report(input_path)
            output_path = Path(temp_dir) / "report.md"
            save_report(extreme_report, output_path)
            self.assertIn(
                "# 数据集质量评估报告：extreme_finite",
                output_path.read_text(encoding="utf-8"),
            )

        extreme_outlier = find_metric(
            extreme_report,
            "statistical_outlier_rate",
            "amount",
        )
        self.assertEqual(extreme_outlier.status, "not_assessable")
        self.assertIn("数值范围过大", extreme_outlier.reason)

    def test_update_lag_uses_latest_recognized_update_date(self):
        dataframe = pd.DataFrame(
            {"update_time": ["2026-06-01", "2026-06-07", "待更新"]}
        )

        metric = calculate_update_lag_days(dataframe, reference_date=date(2026, 7, 16))
        self.assertEqual(metric.value, 39)
        self.assertEqual(metric.evidence["latest_update_date"], "2026-06-07")

        chinese_dataframe = pd.DataFrame(
            {"最后更新": ["2026-06-01", "2026-06-07"]}
        )
        chinese_report = calculate_time_info_availability(chinese_dataframe)
        chinese_lag = calculate_update_lag_days(
            chinese_dataframe, reference_date=date(2026, 7, 16)
        )
        self.assertEqual(chinese_report.value, 1.0)
        self.assertEqual(chinese_lag.value, 39)

    def test_time_metrics_accept_mixed_timezone_values(self):
        dataframe = pd.DataFrame(
            {
                "update_time": [
                    "2026-01-01",
                    "2026-01-02T00:00:00+08:00",
                ]
            }
        )

        availability = calculate_time_info_availability(dataframe)
        lag = calculate_update_lag_days(
            dataframe, reference_date=date(2026, 1, 10)
        )
        self.assertEqual(availability.value, 1.0)
        self.assertEqual(availability.evidence["latest_date"], "2026-01-02")
        self.assertEqual(lag.value, 8)

    def test_iqr_outlier_rate_is_field_level_and_excludes_identifier(self):
        dataframe = pd.DataFrame(
            {"record_id": [1, 2, 3, 4, 5], "amount": [10, 11, 12, 13, 100]}
        )

        metric = calculate_statistical_outlier_rates(dataframe)[0]
        self.assertEqual(metric.field, "amount")
        self.assertAlmostEqual(metric.value, 1 / 5, places=6)
        self.assertEqual(metric.evidence["outlier_samples"], [])

    def test_minimal_dataset_marks_semantic_metrics_not_assessable(self):
        report = build_profile_report(SAMPLES / "minimal_dataset.json")

        expected = {
            "recognizable_format_anomaly_rate",
            "time_info_availability",
            "update_lag_days",
            "source_info_coverage",
            "version_info_coverage",
            "statistical_outlier_rate",
        }
        self.assertTrue(expected.issubset({item.id for item in report.not_assessable}))


if __name__ == "__main__":
    unittest.main()
