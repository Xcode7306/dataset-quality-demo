"""阶段 3：风险提示规则与阈值配置的回归测试。"""

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from src.config import RiskThresholds
from src.metrics import calculate_all_metrics
from src.models import MetricResult
from src.rules import generate_risks
from src.workflow import build_profile_report


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "sample_data"
REFERENCE_DATE = date(2026, 7, 17)


class RiskRuleTests(unittest.TestCase):
    def test_good_dataset_has_no_default_risks(self):
        report = build_profile_report(
            SAMPLES / "good_dataset.csv", reference_date=REFERENCE_DATE
        )

        self.assertEqual(report.risks, [])

    def test_bad_dataset_generates_expected_risk_levels(self):
        report = build_profile_report(
            SAMPLES / "bad_dataset.csv", reference_date=REFERENCE_DATE
        )
        risks = {risk.id: risk for risk in report.risks}

        self.assertEqual(risks["blank_records_detected"].level, "warning")
        self.assertEqual(risks["low_type_consistency:handling_days"].level, "warning")
        self.assertEqual(risks["low_time_availability"].level, "attention")
        self.assertEqual(risks["low_source_coverage"].level, "attention")
        self.assertIn("exact_duplicates_detected", risks)
        self.assertNotIn("normalized_duplicates_detected", risks)

    def test_format_messy_dataset_is_distinguished_without_missing_or_duplicates(self):
        report = build_profile_report(
            SAMPLES / "format_messy_dataset.csv", reference_date=REFERENCE_DATE
        )
        risks = {risk.id: risk for risk in report.risks}
        risk_ids = set(risks)

        self.assertEqual(len(risks), 6)
        self.assertEqual(risks["low_type_consistency:update_time"].level, "warning")
        self.assertEqual(risks["low_type_consistency:handling_days"].level, "warning")
        self.assertEqual(risks["format_anomalies_detected:update_time"].level, "warning")
        self.assertEqual(risks["format_anomalies_detected:source_url"].level, "warning")
        self.assertEqual(risks["format_anomalies_detected:handling_days"].level, "warning")
        for risk_id, risk in risks.items():
            if risk_id.startswith("format_anomalies_detected"):
                self.assertNotIn("异常样例", risk.message)
                self.assertIn("原始数据", risk.message)
        self.assertEqual(risks["low_time_availability"].level, "attention")
        self.assertFalse(any(risk_id.startswith("high_field_missing_rate") for risk_id in risk_ids))
        self.assertNotIn("blank_records_detected", risk_ids)
        self.assertNotIn("exact_duplicates_detected", risk_ids)
        self.assertNotIn("normalized_duplicates_detected", risk_ids)

    def test_publish_department_and_link_do_not_trigger_temporal_risks(self):
        dataframe = pd.DataFrame(
            {
                "标题": ["事项A", "事项B"],
                "发布部门": ["部门A", "部门B"],
                "发布链接": [
                    "https://example.gov.cn/a",
                    "https://example.gov.cn/b",
                ],
            }
        )
        metrics = calculate_all_metrics(dataframe)
        risks = generate_risks(metrics)
        format_metric = next(
            metric
            for metric in metrics
            if metric.id == "recognizable_format_anomaly_rate"
            and metric.field == "发布链接"
        )

        self.assertEqual(format_metric.evidence["expected_format"], "url")
        self.assertEqual(format_metric.value, 0.0)
        self.assertEqual(risks, [])
        self.assertEqual(
            next(metric for metric in metrics if metric.id == "time_info_availability").status,
            "not_assessable",
        )
        self.assertEqual(
            next(metric for metric in metrics if metric.id == "version_info_coverage").status,
            "not_assessable",
        )

    def test_not_assessable_metrics_do_not_generate_risks(self):
        report = build_profile_report(SAMPLES / "minimal_dataset.json")

        self.assertGreater(len(report.not_assessable), 0)
        self.assertEqual(report.risks, [])

    def test_parse_failure_generates_one_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested.json"
            path.write_text('[{"事项": {"名称": "测试"}}]', encoding="utf-8")
            report = build_profile_report(path)

        self.assertEqual(len(report.risks), 1)
        self.assertEqual(report.risks[0].id, "file_parse_failed")
        self.assertEqual(report.risks[0].level, "warning")

    def test_iqr_outlier_only_generates_info(self):
        dataframe = pd.DataFrame(
            {"record_id": [1, 2, 3, 4, 5], "amount": [10, 11, 12, 13, 100]}
        )

        risks = generate_risks(calculate_all_metrics(dataframe))
        outlier_risk = next(
            risk for risk in risks if risk.id == "statistical_outliers_detected:amount"
        )
        self.assertEqual(outlier_risk.level, "info")
        self.assertIn("不代表数据一定错误", outlier_risk.message)

    def test_custom_thresholds_can_change_risk_output(self):
        report = build_profile_report(
            SAMPLES / "bad_dataset.csv", reference_date=REFERENCE_DATE
        )
        thresholds = RiskThresholds(field_missing_attention=0.40)

        risks = generate_risks(report.metrics, thresholds)

        self.assertFalse(
            any(risk.id.startswith("high_field_missing_rate") for risk in risks)
        )

    def test_low_metric_warning_threshold_is_inclusive(self):
        metrics = [
            MetricResult(
                id="field_type_consistency",
                name="字段类型一致率",
                category="类型一致性",
                status="evaluated",
                value=0.80,
                unit="ratio",
                scope="field",
                field="示例字段",
            ),
            MetricResult(
                id="source_info_coverage",
                name="来源信息覆盖率",
                category="可溯性",
                status="evaluated",
                value=0.50,
                unit="ratio",
                scope="dataset",
            ),
        ]

        risks = {risk.id: risk for risk in generate_risks(metrics)}
        self.assertEqual(
            risks["low_type_consistency:示例字段"].level,
            "warning",
        )
        self.assertEqual(risks["low_source_coverage"].level, "warning")

    def test_normalized_duplicates_generate_only_the_additional_risk(self):
        metrics = [
            MetricResult(
                id="exact_duplicate_rate",
                name="完全重复率",
                category="唯一性",
                status="evaluated",
                value=0.0,
                unit="ratio",
                scope="dataset",
            ),
            MetricResult(
                id="normalized_duplicate_rate",
                name="规范化重复率",
                category="唯一性",
                status="evaluated",
                value=0.5,
                unit="ratio",
                scope="dataset",
            ),
        ]

        risks = {risk.id: risk for risk in generate_risks(metrics)}

        self.assertNotIn("exact_duplicates_detected", risks)
        self.assertEqual(risks["normalized_duplicates_detected"].level, "warning")
        self.assertEqual(
            risks["normalized_duplicates_detected"].evidence[
                "additional_duplicate_rate"
            ],
            0.5,
        )

    def test_update_lag_day_boundaries_and_future_date(self):
        cases = (
            (-1, "future_update_date", "attention"),
            (0, None, None),
            (364, None, None),
            (365, "long_update_lag", "attention"),
            (729, "long_update_lag", "attention"),
            (730, "long_update_lag", "warning"),
        )
        for value, expected_id, expected_level in cases:
            with self.subTest(value=value):
                metric = MetricResult(
                    id="update_lag_days",
                    name="更新滞后天数",
                    category="时效性",
                    status="evaluated",
                    value=value,
                    unit="days",
                    scope="dataset",
                )
                risks = generate_risks([metric])
                if expected_id is None:
                    self.assertEqual(risks, [])
                else:
                    self.assertEqual(len(risks), 1)
                    self.assertEqual(risks[0].id, expected_id)
                    self.assertEqual(risks[0].level, expected_level)


if __name__ == "__main__":
    unittest.main()
