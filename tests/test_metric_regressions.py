"""指标边界条件的专属回归测试。"""

from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.field_semantics import identify_semantic_fields
from src.metrics import (
    calculate_all_metrics,
    calculate_exact_duplicate_rate,
    calculate_normalized_duplicate_rate,
    calculate_recognizable_format_anomaly_rates,
    calculate_source_info_coverage,
    calculate_version_info_coverage,
)
from src.workflow import build_profile_report


def find_metric(metrics, metric_id, field=None):
    for metric in metrics:
        if metric.id == metric_id and metric.field == field:
            return metric
    raise AssertionError(f"未找到指标 {metric_id}，字段为 {field}。")


class MetricRegressionTests(unittest.TestCase):
    def test_invalid_ipv6_url_is_counted_as_format_anomaly(self):
        dataframe = pd.DataFrame({"source_url": ["http://[::1", "https://example.gov.cn"]})

        metric = calculate_recognizable_format_anomaly_rates(dataframe)[0]

        self.assertEqual(metric.status, "evaluated")
        self.assertEqual(metric.field, "source_url")
        self.assertEqual(metric.value, 0.5)
        self.assertEqual(metric.evidence["invalid_samples"], [])

    def test_english_keywords_only_match_field_tokens(self):
        false_positive_fields = [
            "message",
            "coverage",
            "image",
            "usage",
            "security_level",
            "resource",
            "conversion",
        ]
        dataframe = pd.DataFrame(
            {field: ["ordinary text"] for field in false_positive_fields}
        )

        format_metrics = calculate_recognizable_format_anomaly_rates(dataframe)

        self.assertEqual(len(format_metrics), 1)
        self.assertEqual(format_metrics[0].status, "not_assessable")
        self.assertEqual(calculate_source_info_coverage(dataframe).status, "not_assessable")
        self.assertEqual(calculate_version_info_coverage(dataframe).status, "not_assessable")

    def test_semantic_fields_keep_normal_english_chinese_and_camel_case(self):
        identified = identify_semantic_fields(
            [
                "updated_at",
                "handling-days",
                "sourceURL",
                "data_source",
                "release-version",
                "更新时间",
                "办理天数",
                "来源链接",
                "版本号",
            ]
        )

        self.assertIn("updated_at", identified["date"])
        self.assertIn("更新时间", identified["date"])
        self.assertIn("handling-days", identified["numeric"])
        self.assertIn("办理天数", identified["numeric"])
        self.assertIn("sourceURL", identified["url"])
        self.assertIn("来源链接", identified["url"])
        self.assertIn("data_source", identified["source"])
        self.assertIn("release-version", identified["version"])
        self.assertIn("版本号", identified["version"])

    def test_normalized_duplicate_keeps_exact_duplicates_with_mixed_missing_values(self):
        dataframe = pd.DataFrame(
            {
                "record_id": [1, 2],
                "name": ["同一事项", "同一事项"],
                "备注": pd.Series([None, float("nan")], dtype=object),
            }
        )

        exact = calculate_exact_duplicate_rate(dataframe)
        normalized = calculate_normalized_duplicate_rate(dataframe)

        self.assertEqual(exact.value, 0.5)
        self.assertEqual(normalized.value, 0.5)
        self.assertGreaterEqual(normalized.value, exact.value)
        self.assertEqual(normalized.evidence["duplicate_groups"][0]["row_indices"], [1, 2])

    def test_reference_date_flows_through_metric_collection_and_workflow(self):
        reference_date = date(2026, 7, 16)
        dataframe = pd.DataFrame({"updated_at": ["2026-06-07"]})

        direct_metric = find_metric(
            calculate_all_metrics(dataframe, reference_date=reference_date),
            "update_lag_days",
        )
        self.assertEqual(direct_metric.value, 39)
        self.assertEqual(direct_metric.evidence["reference_date"], "2026-07-16")

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "dates.csv"
            input_path.write_text("updated_at\n2026-06-07\n", encoding="utf-8")
            report = build_profile_report(input_path, reference_date=reference_date)

        workflow_metric = find_metric(report.metrics, "update_lag_days")
        self.assertEqual(workflow_metric.value, 39)
        self.assertEqual(workflow_metric.evidence["reference_date"], "2026-07-16")

    def test_reference_date_defaults_to_evaluation_day_for_compatibility(self):
        class FixedDate(date):
            @classmethod
            def today(cls):
                return cls(2030, 1, 2)

        dataframe = pd.DataFrame({"updated_at": ["2030-01-01"]})
        with patch("src.metrics.date", FixedDate):
            metric = find_metric(calculate_all_metrics(dataframe), "update_lag_days")

        self.assertEqual(metric.value, 1)
        self.assertEqual(metric.evidence["reference_date"], "2030-01-02")


if __name__ == "__main__":
    unittest.main()
