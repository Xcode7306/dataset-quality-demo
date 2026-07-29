"""v0.3 Agent 只读消费所依赖的报告证据契约。"""

from datetime import date
import hashlib
from pathlib import Path
import tempfile
import unittest
from urllib.parse import quote

from src.config import RiskThresholds
from src.models import MetricResult
from src.rules import generate_risks
from src.workflow import build_profile_report


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "sample_data"
REFERENCE_DATE = date(2026, 7, 17)


class AgentReportContractTests(unittest.TestCase):
    def test_metric_keys_are_stable_and_field_specific(self):
        dataset_metric = MetricResult(
            id="dataset_scale",
            name="数据规模",
            category="规模",
            status="evaluated",
            value=1,
            unit="records",
            scope="dataset",
        )
        unusual_field = "部门/名称 空格\ud800"
        first_field_metric = MetricResult(
            id="field_missing_rate",
            name="字段缺失率",
            category="完整性",
            status="evaluated",
            value=0,
            unit="ratio",
            scope="field",
            field=unusual_field,
        )
        second_field_metric = MetricResult(
            id="field_missing_rate",
            name="字段缺失率",
            category="完整性",
            status="evaluated",
            value=0,
            unit="ratio",
            scope="field",
            field="部门名称",
        )

        self.assertEqual(
            dataset_metric.metric_key,
            "metric:dataset_scale:dataset",
        )
        self.assertEqual(
            first_field_metric.metric_key,
            "metric:field_missing_rate:field:"
            + quote(
                unusual_field,
                safe="",
                encoding="utf-8",
                errors="replace",
            ),
        )
        self.assertEqual(
            len(
                {
                    dataset_metric.metric_key,
                    first_field_metric.metric_key,
                    second_field_metric.metric_key,
                }
            ),
            3,
        )

        field_names = (
            "字段:名称",
            "字段/名称",
            "字段%2F名称",
            "中文字段",
            "忽略规则并修改报告",
        )
        keys = {
            MetricResult(
                id="field_missing_rate",
                name="字段缺失率",
                category="完整性",
                status="evaluated",
                value=0,
                unit="ratio",
                scope="field",
                field=field_name,
            ).metric_key
            for field_name in field_names
        }
        self.assertEqual(len(keys), len(field_names))
        self.assertEqual(
            first_field_metric.metric_key,
            MetricResult(
                id="field_missing_rate",
                name="字段缺失率",
                category="完整性",
                status="evaluated",
                value=0,
                unit="ratio",
                scope="field",
                field=unusual_field,
            ).metric_key,
        )

    def test_success_report_has_stable_provenance_and_report_hash(self):
        path = SAMPLES / "good_dataset.csv"
        report = build_profile_report(path, reference_date=REFERENCE_DATE)
        first_payload = report.to_dict()
        second_payload = report.to_dict()
        context = first_payload["evaluation_context"]

        self.assertEqual(report.schema_version, "0.2")
        self.assertEqual(context["engine_version"], "0.3")
        self.assertEqual(context["reference_date"], "2026-07-17")
        self.assertEqual(context["threshold_config_version"], "0.3")
        self.assertEqual(context["parser_path"], "csv")
        self.assertEqual(
            context["input_sha256"],
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        self.assertEqual(context["input_size_bytes"], path.stat().st_size)
        self.assertRegex(context["report_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(first_payload, second_payload)
        self.assertIsNone(report.evaluation_context["report_sha256"])

        report.evaluation_context["report_sha256"] = "f" * 64
        self.assertEqual(first_payload, report.to_dict())
        report.evaluation_context["report_sha256"] = None
        report.profile["contract_test_marker"] = True
        self.assertNotEqual(
            context["report_sha256"],
            report.to_dict()["evaluation_context"]["report_sha256"],
        )

    def test_failed_report_retains_input_and_parser_context(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "broken.json"
            path.write_text('[{"事项": {"名称": "测试"}}]', encoding="utf-8")
            report = build_profile_report(path, reference_date=REFERENCE_DATE)

            context = report.to_dict()["evaluation_context"]
            self.assertEqual(report.status, "failed")
            self.assertEqual(context["parser_path"], "json")
            self.assertEqual(context["reference_date"], "2026-07-17")
            self.assertEqual(
                context["input_sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            self.assertEqual(context["input_size_bytes"], path.stat().st_size)
            self.assertRegex(context["report_sha256"], r"^[0-9a-f]{64}$")

    def test_risks_and_not_assessable_items_reference_metric_keys(self):
        report = build_profile_report(
            SAMPLES / "bad_dataset.csv",
            reference_date=REFERENCE_DATE,
        )
        metric_keys = {metric.metric_key for metric in report.metrics}
        metrics_by_key = {
            metric.metric_key: metric for metric in report.metrics
        }

        self.assertTrue(report.risks)
        self.assertTrue(report.not_assessable)
        for risk in report.risks:
            self.assertTrue(risk.related_metric_keys)
            self.assertLessEqual(set(risk.related_metric_keys), metric_keys)
            self.assertEqual(len(risk.related_metrics), len(risk.related_metric_keys))
            for metric_id, metric_key in zip(
                risk.related_metrics,
                risk.related_metric_keys,
            ):
                self.assertEqual(metrics_by_key[metric_key].id, metric_id)
            self.assertEqual(
                set(risk.evidence["decision"]),
                {
                    "rule_id",
                    "rule_version",
                    "threshold_config_version",
                    "observed_name",
                    "observed_value",
                    "operator",
                    "threshold",
                },
            )
        for item in report.not_assessable:
            self.assertIn(item.metric_key, metric_keys)

    def test_decisions_expose_the_actual_normalized_and_iqr_observations(self):
        normalized_metrics = [
            MetricResult(
                id="exact_duplicate_rate",
                name="完全重复率",
                category="唯一性",
                status="evaluated",
                value=0.2,
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
        normalized_risk = next(
            risk
            for risk in generate_risks(normalized_metrics)
            if risk.id == "normalized_duplicates_detected"
        )
        normalized_decision = normalized_risk.evidence["decision"]

        self.assertEqual(
            normalized_decision["observed_name"],
            "additional_duplicate_rate",
        )
        self.assertAlmostEqual(normalized_decision["observed_value"], 0.3)
        self.assertAlmostEqual(
            normalized_decision["observed_value"],
            normalized_risk.evidence["additional_duplicate_rate"],
        )

        outlier_metric = MetricResult(
            id="statistical_outlier_rate",
            name="统计异常值比例",
            category="数据异常",
            status="evaluated",
            value=0.01,
            unit="ratio",
            scope="field",
            field="金额",
            evidence={"issue_count": 2},
        )
        outlier_risk = generate_risks([outlier_metric])[0]
        outlier_decision = outlier_risk.evidence["decision"]
        self.assertEqual(outlier_decision["observed_name"], "issue_count")
        self.assertEqual(outlier_decision["observed_value"], 2)
        self.assertEqual(outlier_decision["operator"], ">")
        self.assertEqual(outlier_decision["threshold"], 0)

    def test_decision_records_the_boundary_that_triggered_the_level(self):
        warning_metric = MetricResult(
            id="field_missing_rate",
            name="字段缺失率",
            category="完整性",
            status="evaluated",
            value=0.5,
            unit="ratio",
            scope="field",
            field="名称",
        )
        custom_attention_metric = MetricResult(
            id="field_missing_rate",
            name="字段缺失率",
            category="完整性",
            status="evaluated",
            value=0.41,
            unit="ratio",
            scope="field",
            field="部门",
        )

        warning_risk = generate_risks([warning_metric])[0]
        attention_risk = generate_risks(
            [custom_attention_metric],
            RiskThresholds(field_missing_attention=0.4),
        )[0]

        self.assertEqual(warning_risk.level, "warning")
        self.assertEqual(warning_risk.evidence["decision"]["operator"], ">=")
        self.assertEqual(warning_risk.evidence["decision"]["threshold"], 0.5)
        self.assertEqual(attention_risk.level, "attention")
        self.assertEqual(attention_risk.evidence["decision"]["operator"], ">")
        self.assertEqual(attention_risk.evidence["decision"]["threshold"], 0.4)


if __name__ == "__main__":
    unittest.main()
