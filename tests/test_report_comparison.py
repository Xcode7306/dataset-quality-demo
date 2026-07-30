"""v0.5 确定性 ReportComparison 的契约与语义回归。"""

from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator

from src.comparison_service import (
    ReportComparisonError,
    compare_reports,
    serialize_report_comparison,
    validate_report_comparison,
)
from src.models import MetricResult
from src.rule_pack import Rule, approve_rule_pack, build_rule_pack
from src.rule_service import evaluate_uploaded_dataset_with_rule_pack
from src.upload_service import evaluate_uploaded_dataset
from src.workflow import build_profile_report


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "sample_data"
SCHEMA_PATH = ROOT / "schemas" / "report-comparison.schema.json"
REFERENCE_DATE = date(2026, 7, 17)


def _report(sample: str, *, dataset_name: str = "同一治理数据集"):
    return build_profile_report(
        SAMPLES / sample,
        dataset_name=dataset_name,
        reference_date=REFERENCE_DATE,
    )


def _rehash_comparison(payload):
    hash_payload = deepcopy(payload)
    hash_payload.pop("comparison_sha256", None)
    payload["comparison_sha256"] = hashlib.sha256(
        json.dumps(
            hash_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _enhanced_required_report(sample: str):
    path = SAMPLES / sample
    content = path.read_bytes()
    baseline = evaluate_uploaded_dataset(
        content,
        path.name,
        dataset_name="同一治理数据集",
        reference_date=REFERENCE_DATE,
    )
    draft = build_rule_pack(
        baseline,
        name="跨版本必填规则",
        version="1.0",
        rules=(
            Rule(
                type="required",
                rule_id="required-service-name",
                fields=("service_name",),
            ),
        ),
        generated_at="2026-07-29T00:00:00Z",
    )
    approved = approve_rule_pack(
        draft,
        baseline,
        approver="local-test",
        approved_at="2026-07-29T00:01:00Z",
    )
    return evaluate_uploaded_dataset_with_rule_pack(
        content,
        path.name,
        approved,
        dataset_name="同一治理数据集",
        reference_date=REFERENCE_DATE,
    ).enhanced_report


class ReportComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        cls.validator = Draft202012Validator(schema)

    def compare(self, before, after):
        return compare_reports(
            before,
            after,
            dataset_series_id="政务服务事项",
            same_series_confirmed=True,
        )

    def test_bad_to_good_is_deterministic_schema_valid_and_traceable(self):
        before = _report("bad_dataset.csv")
        after = _report("good_dataset.csv")
        before_snapshot = before.to_dict()
        after_snapshot = after.to_dict()

        first = self.compare(before, after)
        second = self.compare(before, after)
        payload = first.to_dict()

        self.assertEqual(first, second)
        self.assertEqual(first.comparison_sha256, second.comparison_sha256)
        self.assertEqual(
            first.lineage["ordered_report_sha256"],
            [
                before_snapshot["evaluation_context"]["report_sha256"],
                after_snapshot["evaluation_context"]["report_sha256"],
            ],
        )
        self.assertEqual(list(self.validator.iter_errors(payload)), [])
        self.assertEqual(first.compatibility["status"], "full")
        self.assertGreater(first.summary["improved_metric_count"], 0)
        self.assertEqual(first.summary["worsened_metric_count"], 0)
        self.assertEqual(first.summary["added_risk_count"], 0)
        self.assertEqual(first.summary["resolved_risk_count"], 16)
        dataset_scale = next(
            item
            for item in first.metric_changes
            if item.metric_id == "dataset_scale"
        )
        self.assertEqual(dataset_scale.classification, "changed")
        self.assertEqual(dataset_scale.direction, "neutral")
        self.assertEqual(before.to_dict(), before_snapshot)
        self.assertEqual(after.to_dict(), after_snapshot)
        self.assertEqual(
            json.loads(serialize_report_comparison(first)),
            payload,
        )

    def test_series_confirmation_and_fixed_report_hash_are_mandatory(self):
        before = _report("bad_dataset.csv")
        after = _report("good_dataset.csv")
        with self.assertRaisesRegex(ReportComparisonError, "明确确认"):
            compare_reports(
                before,
                after,
                dataset_series_id="政务服务事项",
                same_series_confirmed=False,
            )

        tampered = after.to_dict()
        tampered["profile"]["row_count"] = 999
        with self.assertRaisesRegex(ReportComparisonError, "哈希校验失败"):
            self.compare(before, tampered)

        with self.assertRaisesRegex(ReportComparisonError, "治理对象标识"):
            compare_reports(
                before,
                after,
                dataset_series_id="\n",
                same_series_confirmed=True,
            )

    def test_assessability_transitions_are_separate_from_quality_improvement(self):
        before = _report("minimal_dataset.json")
        after = _report("good_dataset.csv")

        comparison = self.compare(before, after)

        became_assessable = {
            change.metric_key
            for change in comparison.assessability_changes
            if change.classification == "became_assessable"
        }
        self.assertIn(
            "metric:time_info_availability:dataset",
            became_assessable,
        )
        metric = next(
            change
            for change in comparison.metric_changes
            if change.metric_key == "metric:time_info_availability:dataset"
        )
        self.assertEqual(metric.classification, "became_assessable")
        self.assertNotEqual(metric.classification, "improved")

    def test_risk_disappearance_is_not_resolved_when_its_metric_disappears(self):
        before = _report("bad_dataset.csv")
        target = deepcopy(before)
        removed_key = "metric:field_missing_rate:field:service_name"
        target.metrics = [
            metric for metric in target.metrics if metric.metric_key != removed_key
        ]
        target.risks = [
            risk
            for risk in target.risks
            if risk.id != "high_field_missing_rate:service_name"
        ]

        comparison = self.compare(before, target)

        risk_change = next(
            change
            for change in comparison.risk_changes
            if change.risk_id == "high_field_missing_rate:service_name"
        )
        self.assertEqual(risk_change.classification, "not_comparable")
        self.assertIn(
            "related_metric_removed_or_not_assessable",
            risk_change.reason_codes,
        )

    def test_engine_change_suppresses_metric_and_risk_conclusions(self):
        before = _report("bad_dataset.csv")
        target = deepcopy(before)
        target.evaluation_context["engine_version"] = "0.3"

        comparison = self.compare(before, target)

        self.assertEqual(comparison.compatibility["status"], "limited")
        self.assertIn(
            "engine_version_changed",
            comparison.compatibility["reason_codes"],
        )
        self.assertTrue(
            all(
                change.classification in {"unchanged", "not_comparable"}
                for change in comparison.metric_changes
            )
        )

    def test_business_metrics_require_the_same_rule_definition_evidence(self):
        before = _report("good_dataset.csv")
        target = deepcopy(before)
        before.metrics.append(
            MetricResult(
                id="business_required_compliance",
                name="必填字段完整率",
                category="业务规则",
                status="evaluated",
                value=0.8,
                unit="ratio",
                scope="field",
                field="service_name",
                evidence={"rule_pack_sha256": "a" * 64},
            )
        )
        target.metrics.append(
            MetricResult(
                id="business_required_compliance",
                name="必填字段完整率",
                category="业务规则",
                status="evaluated",
                value=0.8,
                unit="ratio",
                scope="field",
                field="service_name",
                evidence={"rule_pack_sha256": "b" * 64},
            )
        )

        comparison = self.compare(before, target)
        change = next(
            item
            for item in comparison.metric_changes
            if item.metric_id == "business_required_compliance"
        )

        self.assertEqual(change.classification, "not_comparable")
        self.assertIn(
            "business_rule_definition_changed",
            change.reason_codes,
        )
        self.assertEqual(comparison.compatibility["status"], "limited")

    def test_unknown_business_metric_direction_remains_neutral(self):
        before = _report("good_dataset.csv")
        target = deepcopy(before)
        for report, value in ((before, 0.1), (target, 0.2)):
            report.metrics.append(
                MetricResult(
                    id="business_error_rate",
                    name="未登记的业务错误率",
                    category="业务规则",
                    status="evaluated",
                    value=value,
                    unit="ratio",
                    scope="dataset",
                    evidence={"rule_pack_sha256": "a" * 64},
                )
            )

        comparison = self.compare(before, target)
        change = next(
            item
            for item in comparison.metric_changes
            if item.metric_id == "business_error_rate"
        )

        self.assertEqual(change.direction, "neutral")
        self.assertEqual(change.classification, "changed")

    def test_same_business_rule_definition_is_comparable_across_inputs(self):
        comparison = self.compare(
            _enhanced_required_report("bad_dataset.csv"),
            _enhanced_required_report("good_dataset.csv"),
        )
        metric = next(
            item
            for item in comparison.metric_changes
            if item.metric_id == "business_required_compliance"
        )
        risk = next(
            item
            for item in comparison.risk_changes
            if item.risk_id.startswith("business_rule_violation:")
        )

        self.assertEqual(metric.classification, "improved")
        self.assertEqual(risk.classification, "resolved")
        self.assertNotIn(
            "business_rule_definition_changed",
            comparison.compatibility["reason_codes"],
        )

    def test_risk_definition_change_is_not_treated_as_persistent(self):
        before = _report("bad_dataset.csv")
        target = deepcopy(before)
        target.risks[0].evidence["decision"]["threshold"] = 0.123456
        risk_id = target.risks[0].id

        comparison = self.compare(before, target)
        change = next(
            item
            for item in comparison.risk_changes
            if item.risk_id == risk_id
        )

        self.assertEqual(change.classification, "not_comparable")
        self.assertIn(
            "risk_rule_or_threshold_changed",
            change.reason_codes,
        )
        self.assertEqual(comparison.compatibility["status"], "limited")
        self.assertIn(
            "risk_rule_definition_changed",
            comparison.compatibility["reason_codes"],
        )

    def test_reference_date_change_limits_time_based_conclusions(self):
        path = SAMPLES / "good_dataset.csv"
        before = build_profile_report(
            path,
            dataset_name="同一治理数据集",
            reference_date=date(2026, 7, 17),
        )
        after = build_profile_report(
            path,
            dataset_name="同一治理数据集",
            reference_date=date(2027, 7, 17),
        )

        comparison = self.compare(before, after)

        self.assertEqual(comparison.compatibility["status"], "limited")
        self.assertIn(
            "reference_date_changed_for_time_metrics",
            comparison.compatibility["reason_codes"],
        )
        lag_change = next(
            item
            for item in comparison.metric_changes
            if item.metric_id == "update_lag_days"
        )
        self.assertEqual(lag_change.classification, "not_comparable")
        self.assertIn("reference_date_changed", lag_change.reason_codes)
        stale_risk = next(
            item
            for item in comparison.risk_changes
            if item.risk_id == "long_update_lag"
        )
        self.assertEqual(stale_risk.classification, "not_comparable")
        self.assertIn("reference_date_changed", stale_risk.reason_codes)

    def test_future_update_dates_are_described_without_monotonic_quality_claim(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            before_path = root / "before.csv"
            after_path = root / "after.csv"
            before_path.write_text(
                "record_id,update_time\n1,2026-07-18\n",
                encoding="utf-8",
            )
            after_path.write_text(
                "record_id,update_time\n1,2026-07-17\n",
                encoding="utf-8",
            )
            before = build_profile_report(
                before_path,
                dataset_name="同一治理数据集",
                reference_date=REFERENCE_DATE,
            )
            after = build_profile_report(
                after_path,
                dataset_name="同一治理数据集",
                reference_date=REFERENCE_DATE,
            )

        comparison = self.compare(before, after)
        lag = next(
            change
            for change in comparison.metric_changes
            if change.metric_id == "update_lag_days"
        )
        self.assertEqual(lag.classification, "changed")
        self.assertIn("future_date_requires_risk_context", lag.reason_codes)

    def test_comparison_self_hash_is_rechecked(self):
        comparison = self.compare(
            _report("bad_dataset.csv"),
            _report("good_dataset.csv"),
        )
        tampered = comparison.to_dict()
        tampered["summary"]["improved_metric_count"] = 999

        with self.assertRaisesRegex(ReportComparisonError, "自身哈希"):
            validate_report_comparison(tampered)

    def test_rehashed_semantic_forgery_and_reordering_are_rejected(self):
        comparison = self.compare(
            _report("bad_dataset.csv"),
            _report("good_dataset.csv"),
        )

        forged_summary = comparison.to_dict()
        forged_summary["summary"]["improved_metric_count"] += 1
        _rehash_comparison(forged_summary)
        with self.assertRaisesRegex(ReportComparisonError, "摘要"):
            validate_report_comparison(forged_summary)

        reordered = comparison.to_dict()
        reordered["metric_changes"].reverse()
        _rehash_comparison(reordered)
        with self.assertRaisesRegex(ReportComparisonError, "排序"):
            validate_report_comparison(reordered)

        bad_lineage = comparison.to_dict()
        bad_lineage["lineage"]["ordered_report_sha256"].reverse()
        _rehash_comparison(bad_lineage)
        with self.assertRaisesRegex(ReportComparisonError, "报告顺序"):
            validate_report_comparison(bad_lineage)

        forged_direction = comparison.to_dict()
        neutral = next(
            item
            for item in forged_direction["metric_changes"]
            if item["direction"] == "neutral"
            and item["classification"] == "changed"
        )
        neutral["classification"] = "improved"
        forged_direction["summary"]["changed_metric_count"] -= 1
        forged_direction["summary"]["improved_metric_count"] += 1
        _rehash_comparison(forged_direction)
        with self.assertRaisesRegex(ReportComparisonError, "方向"):
            validate_report_comparison(forged_direction)

    def test_unordered_nested_sets_cannot_create_alternate_valid_hashes(self):
        before = _report("bad_dataset.csv")
        target = deepcopy(before)
        target.dataset.name = "同一治理数据集（别名）"
        target.evaluation_context["engine_version"] = "0.3"
        target.evaluation_context["reference_date"] = "2027-07-17"
        comparison = self.compare(before, target)

        def assert_reordering_rejected(path):
            forged = comparison.to_dict()
            values = path(forged)
            self.assertGreater(len(values), 1)
            values.reverse()
            _rehash_comparison(forged)
            with self.assertRaisesRegex(
                ReportComparisonError,
                "字典序",
            ):
                validate_report_comparison(forged)

        assert_reordering_rejected(
            lambda payload: payload["compatibility"]["reason_codes"]
        )
        assert_reordering_rejected(
            lambda payload: payload["compatibility"][
                "context_change_codes"
            ]
        )
        assert_reordering_rejected(
            lambda payload: payload["limitations"]
        )
        forged_risk_reasons = comparison.to_dict()
        risk_reasons = next(
            item["reason_codes"]
            for item in forged_risk_reasons["risk_changes"]
            if item["reason_codes"]
        )
        risk_reasons[:] = sorted(
            set(risk_reasons) | {"reference_date_changed"},
            reverse=True,
        )
        self.assertNotEqual(risk_reasons, sorted(risk_reasons))
        _rehash_comparison(forged_risk_reasons)
        with self.assertRaisesRegex(ReportComparisonError, "字典序"):
            validate_report_comparison(forged_risk_reasons)


if __name__ == "__main__":
    unittest.main()
