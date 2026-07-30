"""v0.5 会话历史的严格导入、容量和删除策略测试。"""

from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import unittest

from src.history_store import (
    DEFAULT_HISTORY_POLICY,
    HistoryPolicy,
    HistoryValidationError,
    InMemoryReportHistoryStore,
    build_version_trend,
    parse_quality_report_json,
    validate_quality_report_payload,
)
from src.workflow import build_profile_report


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "sample_data"
REPORTS = ROOT / "reports"
REFERENCE_DATE = date(2026, 7, 17)
SAVED_AT = datetime(2026, 7, 29, 10, 0, tzinfo=timezone.utc)


def _rehash_report(payload):
    hash_payload = deepcopy(payload)
    hash_payload["evaluation_context"].pop("report_sha256", None)
    payload["evaluation_context"]["report_sha256"] = hashlib.sha256(
        json.dumps(
            hash_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


class HistoryStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_profile_report(
            SAMPLES / "good_dataset.csv",
            reference_date=REFERENCE_DATE,
        )

    def test_strict_import_and_session_save_keep_an_immutable_fixed_report(self):
        imported = parse_quality_report_json(
            (REPORTS / "good_report.json").read_bytes()
        )
        store = InMemoryReportHistoryStore()

        entry = store.add_report(
            imported,
            version_label="整改前",
            dataset_series_id="政务服务事项",
            saved_at=SAVED_AT,
        )

        self.assertEqual(entry.saved_at, "2026-07-29T10:00:00Z")
        self.assertEqual(
            entry.report_sha256,
            imported["evaluation_context"]["report_sha256"],
        )
        self.assertEqual(len(store.list_entries()), 1)
        copy_payload = entry.report_payload
        copy_payload["dataset"]["name"] = "被调用方修改"
        self.assertNotEqual(
            entry.report_payload["dataset"]["name"],
            "被调用方修改",
        )
        self.assertFalse(DEFAULT_HISTORY_POLICY.raw_upload_bytes_stored)
        self.assertFalse(DEFAULT_HISTORY_POLICY.issue_location_csv_stored)
        self.assertEqual(DEFAULT_HISTORY_POLICY.storage_mode, "session_memory")

    def test_duplicate_report_is_rejected_per_series_but_allowed_in_another_series(self):
        store = InMemoryReportHistoryStore()
        store.add_report(
            self.report,
            version_label="v1",
            dataset_series_id="series-a",
            saved_at=SAVED_AT,
        )

        with self.assertRaisesRegex(HistoryValidationError, "相同报告哈希"):
            store.add_report(
                self.report,
                version_label="另一个标签",
                dataset_series_id="series-a",
                saved_at=SAVED_AT,
            )

        store.add_report(
            self.report,
            version_label="v1",
            dataset_series_id="series-b",
            saved_at=SAVED_AT,
        )
        self.assertEqual(len(store.list_entries()), 2)

    def test_import_rejects_non_utf8_duplicate_keys_constants_depth_and_tampering(self):
        invalid_cases = (
            ("非 UTF-8", b"\xff"),
            ("重复键", b'{"x":1,"x":2}'),
            ("非标准数值", b'{"x":NaN}'),
            ("嵌套深度", ("[" * 65 + "]" * 65).encode("utf-8")),
        )
        for label, content in invalid_cases:
            with self.subTest(label=label):
                with self.assertRaises(HistoryValidationError):
                    parse_quality_report_json(content)

        payload = self.report.to_dict()
        payload["profile"]["row_count"] = 999
        with self.assertRaisesRegex(HistoryValidationError, "哈希校验失败"):
            validate_quality_report_payload(payload)

    def test_cross_reference_and_uniqueness_invariants_are_rechecked(self):
        duplicate_metric_report = deepcopy(self.report)
        duplicate_metric_report.metrics.append(
            deepcopy(duplicate_metric_report.metrics[0])
        )
        with self.assertRaisesRegex(HistoryValidationError, "重复 metric_key"):
            validate_quality_report_payload(duplicate_metric_report)

        invalid_risk_report = deepcopy(self.report)
        bad_payload = invalid_risk_report.to_dict()
        bad_payload["risks"] = [
            {
                "id": "bad-risk",
                "level": "attention",
                "title": "坏引用",
                "message": "坏引用",
                "related_metrics": ["missing"],
                "related_metric_keys": ["metric:missing:dataset"],
                "evidence": {
                    "decision": {
                        "rule_id": "bad",
                        "rule_version": "0.1",
                        "threshold_config_version": "0.3",
                        "observed_name": "missing",
                        "observed_value": 1,
                        "operator": ">",
                        "threshold": 0,
                    }
                },
            }
        ]
        # 先按 QualityReport 的规范化规则重新固化哈希，确保测试命中引用复核。
        _rehash_report(bad_payload)
        with self.assertRaisesRegex(HistoryValidationError, "不存在的指标键"):
            validate_quality_report_payload(bad_payload)

    def test_structurally_raw_or_semantically_forged_reports_are_rejected(self):
        raw_profile = self.report.to_dict()
        raw_profile["profile"]["raw_rows"] = [["身份证号", "原始值"]]
        _rehash_report(raw_profile)
        with self.assertRaisesRegex(HistoryValidationError, "契约外内容"):
            validate_quality_report_payload(raw_profile)

        raw_samples = self.report.to_dict()
        raw_samples["profile"]["columns"][0]["non_null_samples"] = [
            "敏感原值"
        ]
        _rehash_report(raw_samples)
        with self.assertRaisesRegex(HistoryValidationError, "原始样例"):
            validate_quality_report_payload(raw_samples)

        raw_evidence = self.report.to_dict()
        raw_evidence["metrics"][0]["evidence"]["raw_values"] = [
            "敏感原值"
        ]
        _rehash_report(raw_evidence)
        with self.assertRaisesRegex(HistoryValidationError, "契约外内容"):
            validate_quality_report_payload(raw_evidence)

        huge_ratio = self.report.to_dict()
        ratio_metric = next(
            item
            for item in huge_ratio["metrics"]
            if item["unit"] == "ratio"
        )
        ratio_metric["value"] = 10**400
        _rehash_report(huge_ratio)
        with self.assertRaisesRegex(HistoryValidationError, "安全范围"):
            validate_quality_report_payload(huge_ratio)

        missing_profile = self.report.to_dict()
        missing_profile["profile"] = {}
        _rehash_report(missing_profile)
        with self.assertRaisesRegex(HistoryValidationError, "完整字段画像"):
            validate_quality_report_payload(missing_profile)

        bad_not_assessable = build_profile_report(
            SAMPLES / "minimal_dataset.json",
            reference_date=REFERENCE_DATE,
        ).to_dict()
        bad_not_assessable["not_assessable"][0]["reason"] = "伪造原因"
        _rehash_report(bad_not_assessable)
        with self.assertRaisesRegex(HistoryValidationError, "对应指标定义"):
            validate_quality_report_payload(bad_not_assessable)

    def test_risk_decision_context_and_metric_references_are_reconciled(self):
        report = build_profile_report(
            SAMPLES / "bad_dataset.csv",
            reference_date=REFERENCE_DATE,
        ).to_dict()
        report["risks"][0]["evidence"]["decision"][
            "threshold_config_version"
        ] = "伪造阈值版本"
        _rehash_report(report)
        with self.assertRaisesRegex(HistoryValidationError, "阈值版本"):
            validate_quality_report_payload(report)

        related = build_profile_report(
            SAMPLES / "bad_dataset.csv",
            reference_date=REFERENCE_DATE,
        ).to_dict()
        related["risks"][0]["related_metrics"] = ["伪造指标"]
        _rehash_report(related)
        with self.assertRaisesRegex(HistoryValidationError, "指标 ID"):
            validate_quality_report_payload(related)

    def test_capacity_delete_clear_and_trend_are_explicit(self):
        policy = HistoryPolicy(max_reports=1)
        store = InMemoryReportHistoryStore(policy=policy)
        first = store.add_report(
            self.report,
            version_label="整改前",
            dataset_series_id="series",
            saved_at=SAVED_AT,
        )
        changed = build_profile_report(
            SAMPLES / "bad_dataset.csv",
            dataset_name=self.report.dataset.name,
            reference_date=REFERENCE_DATE,
        )
        with self.assertRaisesRegex(HistoryValidationError, "最多保存 1 份"):
            store.add_report(
                changed,
                version_label="整改后",
                dataset_series_id="series",
                saved_at="2026-07-29T11:00:00Z",
            )

        trend = build_version_trend(
            store.list_entries(),
            dataset_series_id="series",
        )
        self.assertEqual(len(trend), 1)
        self.assertEqual(trend[0]["version_label"], "整改前")
        self.assertEqual(trend[0]["risk_count"], 0)
        self.assertTrue(store.delete(first.entry_id))
        self.assertFalse(store.delete(first.entry_id))
        self.assertEqual(store.clear(), 0)


if __name__ == "__main__":
    unittest.main()
