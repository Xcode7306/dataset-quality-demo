"""v0.6 指标目录、DB31/T 计算口径与选择链路回归。"""

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator
import pandas as pd

import src.metrics as metrics_module
from src.metric_catalog import (
    ALL_METRIC_IDS,
    DB31_METRIC_IDS,
    DEFAULT_SELECTED_METRIC_IDS,
    METRIC_BY_ID,
    METRIC_CATALOG,
    MetricSelectionError,
    ORIGINAL_METRIC_IDS,
    metric_description,
    normalize_selected_metric_ids,
)
from src.metrics import (
    calculate_all_metrics,
    calculate_failed_metrics,
)
from src.rule_engine import RulePackExecutionError
from src.rule_pack import Rule, approve_rule_pack, build_rule_pack
from src.rule_service import evaluate_uploaded_dataset_with_rule_pack
from src.upload_service import evaluate_uploaded_dataset
from src.workflow import build_profile_report


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "sample_data"
REFERENCE_DATE = date(2026, 7, 17)


def _metric(results, metric_id):
    return next(result for result in results if result.id == metric_id)


class MetricCatalogTests(unittest.TestCase):
    def test_catalog_preserves_13_original_and_adds_30_standard_metrics(self):
        self.assertEqual(len(ORIGINAL_METRIC_IDS), 13)
        self.assertEqual(len(DB31_METRIC_IDS), 30)
        self.assertEqual(len(ALL_METRIC_IDS), 43)
        self.assertEqual(len(METRIC_CATALOG), 43)
        self.assertEqual(len(set(ALL_METRIC_IDS)), 43)
        self.assertEqual(ALL_METRIC_IDS[:13], ORIGINAL_METRIC_IDS)
        self.assertEqual(ALL_METRIC_IDS[13:], DB31_METRIC_IDS)
        self.assertEqual(DEFAULT_SELECTED_METRIC_IDS, ORIGINAL_METRIC_IDS)

    def test_db31_codes_directions_and_parent_relationships_are_complete(self):
        codes = [
            METRIC_BY_ID[metric_id]["standard_code"]
            for metric_id in DB31_METRIC_IDS
        ]
        self.assertEqual(len(set(codes)), 30)
        self.assertTrue(
            all(
                METRIC_BY_ID[metric_id]["direction"] == "higher_is_better"
                for metric_id in DB31_METRIC_IDS
            )
        )
        self.assertEqual(
            {
                METRIC_BY_ID[metric_id]["parent_id"]
                for metric_id in (
                    "db31_010101",
                    "db31_010102",
                    "db31_010103",
                )
            },
            {"db31_010100"},
        )
        self.assertEqual(
            {
                METRIC_BY_ID[metric_id]["parent_id"]
                for metric_id in (
                    "db31_040201",
                    "db31_040202",
                    "db31_040203",
                    "db31_040204",
                )
            },
            {"db31_040200"},
        )

    def test_every_metric_has_a_concise_hover_description(self):
        self.assertTrue(
            all(
                isinstance(METRIC_BY_ID[metric_id]["description"], str)
                and METRIC_BY_ID[metric_id]["description"].strip()
                for metric_id in ALL_METRIC_IDS
            )
        )
        self.assertEqual(
            metric_description("db31_030300"),
            "特定字段、记录、文件或数据集意外重复较少的程度。",
        )

    def test_selection_is_validated_deduplicated_and_catalog_ordered(self):
        self.assertEqual(
            normalize_selected_metric_ids(None),
            ORIGINAL_METRIC_IDS,
        )
        self.assertEqual(
            normalize_selected_metric_ids(
                [
                    "db31_030400",
                    "field_missing_rate",
                    "db31_030300",
                    "field_missing_rate",
                ]
            ),
            (
                "field_missing_rate",
                "db31_030300",
                "db31_030400",
            ),
        )
        for invalid in ([], "dataset_scale", ["unknown_metric"], [None]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(MetricSelectionError):
                    normalize_selected_metric_ids(invalid)  # type: ignore[arg-type]


class DB31MetricCalculationTests(unittest.TestCase):
    def test_default_calculation_remains_the_original_v04_selection(self):
        dataframe = pd.DataFrame({"record_id": [1, 2], "name": ["A", "B"]})
        results = calculate_all_metrics(
            dataframe,
            reference_date=REFERENCE_DATE,
        )

        self.assertEqual(
            tuple(dict.fromkeys(result.id for result in results)),
            ORIGINAL_METRIC_IDS,
        )

    def test_selected_subset_is_the_only_output(self):
        dataframe = pd.DataFrame({"record_id": [1, 2], "name": ["A", "A"]})
        results = calculate_all_metrics(
            dataframe,
            selected_metric_ids=(
                "db31_030400",
                "exact_duplicate_rate",
                "db31_030300",
            ),
        )

        self.assertEqual(
            [result.id for result in results],
            [
                "exact_duplicate_rate",
                "db31_030300",
                "db31_030400",
            ],
        )
        self.assertEqual(len({result.metric_key for result in results}), 3)

    def test_duplicate_and_uniqueness_scores_use_exact_record_grain(self):
        dataframe = pd.DataFrame(
            {
                "record_id": [1, 2, 3, 4, 5, 6],
                "name": ["A", "A", "B", "C", "D", "E"],
                "department": ["甲", "甲", "乙", "丙", "丁", "戊"],
            }
        )
        results = calculate_all_metrics(
            dataframe,
            selected_metric_ids=(
                "exact_duplicate_rate",
                "db31_030300",
                "db31_030400",
            ),
        )

        self.assertAlmostEqual(
            float(_metric(results, "exact_duplicate_rate").value),
            1 / 6,
            places=6,
        )
        for metric_id in ("db31_030300", "db31_030400"):
            metric = _metric(results, metric_id)
            self.assertAlmostEqual(float(metric.value), 5 / 6, places=6)
            self.assertEqual(metric.status, "evaluated")
            self.assertEqual(metric.evidence["score_direction"], "higher_is_better")
            self.assertEqual(metric.evidence["grain"], "record")
            self.assertEqual(metric.evidence["equality"], "exact")
            self.assertEqual(metric.evidence["checked_count"], 6)
            self.assertEqual(metric.evidence["conforming_count"], 5)
            self.assertEqual(metric.evidence["issue_count"], 1)
            self.assertEqual(len(metric.issue_locations), 1)

    def test_related_duplicate_metrics_share_one_exact_grouping_pass(self):
        dataframe = pd.DataFrame(
            {
                "record_id": [1, 2, 3],
                "name": ["A", "A", "B"],
            }
        )
        with patch(
            "src.metrics.calculate_exact_duplicate_rate",
            wraps=metrics_module.calculate_exact_duplicate_rate,
        ) as calculate_exact:
            results = calculate_all_metrics(
                dataframe,
                selected_metric_ids=(
                    "exact_duplicate_rate",
                    "db31_030300",
                    "db31_030400",
                ),
            )

        self.assertEqual(calculate_exact.call_count, 1)
        self.assertEqual(len(results), 3)

    def test_all_identical_records_and_no_duplicates_cover_formula_edges(self):
        identical = pd.DataFrame(
            {
                "record_id": [1, 2, 3],
                "name": ["A", "A", "A"],
            }
        )
        unique = pd.DataFrame(
            {
                "record_id": [1, 2, 3],
                "name": ["A", "B", "C"],
            }
        )
        selection = ("db31_030300", "db31_030400")

        identical_results = calculate_all_metrics(
            identical,
            selected_metric_ids=selection,
        )
        unique_results = calculate_all_metrics(
            unique,
            selected_metric_ids=selection,
        )
        for metric_id in selection:
            self.assertAlmostEqual(
                float(_metric(identical_results, metric_id).value),
                1 / 3,
                places=6,
            )
            self.assertEqual(_metric(unique_results, metric_id).value, 1.0)

    def test_normalized_only_duplicates_do_not_change_db31_exact_scores(self):
        dataframe = pd.DataFrame(
            {
                "record_id": [1, 2],
                "name": ["A-B", " a b "],
            }
        )
        results = calculate_all_metrics(
            dataframe,
            selected_metric_ids=(
                "normalized_duplicate_rate",
                "db31_030300",
                "db31_030400",
            ),
        )

        self.assertEqual(
            _metric(results, "normalized_duplicate_rate").value,
            0.5,
        )
        self.assertEqual(_metric(results, "db31_030300").value, 1.0)
        self.assertEqual(_metric(results, "db31_030400").value, 1.0)

    def test_empty_dataset_marks_direct_standard_metrics_unassessable(self):
        results = calculate_all_metrics(
            pd.DataFrame(columns=["record_id", "name"]),
            selected_metric_ids=("db31_030300", "db31_030400"),
        )

        self.assertEqual(len(results), 2)
        for metric in results:
            self.assertEqual(metric.status, "not_assessable")
            self.assertIsNone(metric.value)
            self.assertEqual(metric.evidence["reason_code"], "zero_denominator")
            self.assertEqual(metric.evidence["required_inputs"], ["至少一条可评价记录"])

    def test_other_standard_metrics_remain_visible_with_required_evidence(self):
        selected = (
            "db31_010101",
            "db31_020100",
            "db31_040203",
            "db31_060100",
        )
        results = calculate_all_metrics(
            pd.DataFrame({"name": ["A"]}),
            selected_metric_ids=selected,
        )

        self.assertEqual([result.id for result in results], list(selected))
        for metric in results:
            self.assertEqual(metric.status, "not_assessable")
            self.assertIsNone(metric.value)
            self.assertTrue(metric.reason)
            self.assertTrue(metric.evidence["reason_code"])
            self.assertTrue(metric.evidence["required_inputs"])
            self.assertFalse(metric.evidence["proxy"]["standard_equivalent"])

        self.assertEqual(
            _metric(results, "db31_010101").evidence[
                "available_proxy_metric_ids"
            ],
            ["field_type_consistency"],
        )

    def test_failed_metrics_respect_the_selected_subset(self):
        results = calculate_failed_metrics(
            "测试解析失败",
            selected_metric_ids=("dataset_scale", "db31_030300"),
        )

        self.assertEqual(
            [result.id for result in results],
            ["dataset_scale", "db31_030300"],
        )
        self.assertTrue(all(result.status == "not_assessable" for result in results))
        self.assertEqual(
            _metric(results, "db31_030300").evidence["evaluation_blocker"],
            "file_parse_failed",
        )


class MetricSelectionIntegrationTests(unittest.TestCase):
    def test_workflow_records_normalized_selection_and_hashes_it(self):
        first = build_profile_report(
            SAMPLES / "good_dataset.csv",
            reference_date=REFERENCE_DATE,
            selected_metric_ids=(
                "db31_030400",
                "dataset_scale",
                "db31_030300",
                "dataset_scale",
            ),
        )
        second = build_profile_report(
            SAMPLES / "good_dataset.csv",
            reference_date=REFERENCE_DATE,
            selected_metric_ids=(
                "dataset_scale",
                "db31_030300",
                "db31_030400",
            ),
        )
        default_report = build_profile_report(
            SAMPLES / "good_dataset.csv",
            reference_date=REFERENCE_DATE,
        )

        expected = ["dataset_scale", "db31_030300", "db31_030400"]
        self.assertEqual(
            first.to_dict()["evaluation_context"]["selected_metric_ids"],
            expected,
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertNotEqual(
            first.to_dict()["evaluation_context"]["report_sha256"],
            default_report.to_dict()["evaluation_context"]["report_sha256"],
        )

    def test_uploaded_selection_matches_cli_workflow_selection(self):
        content = (SAMPLES / "good_dataset.csv").read_bytes()
        selected = ("exact_duplicate_rate", "db31_030300")
        uploaded = evaluate_uploaded_dataset(
            content,
            "good_dataset.csv",
            reference_date=REFERENCE_DATE,
            selected_metric_ids=selected,
        )

        self.assertEqual(
            uploaded.evaluation_context["selected_metric_ids"],
            list(selected),
        )
        self.assertEqual(
            [metric.id for metric in uploaded.metrics],
            list(selected),
        )

    def test_rule_pack_rebuilds_the_exact_same_selected_baseline(self):
        content = (
            "record_id,name\n"
            "1,A\n"
            "2,A\n"
        ).encode("utf-8")
        selected = ("exact_duplicate_rate", "db31_030300")
        report = evaluate_uploaded_dataset(
            content,
            "selected.csv",
            reference_date=REFERENCE_DATE,
            selected_metric_ids=selected,
        )
        draft = build_rule_pack(
            report,
            name="选择集绑定测试",
            version="1.0",
            rules=(
                Rule(
                    type="required",
                    rule_id="required-name",
                    fields=("name",),
                ),
            ),
            generated_at="2026-07-17T00:00:00Z",
        )
        approved = approve_rule_pack(
            draft,
            report,
            approver="local-tester",
            approved_at="2026-07-17T00:01:00Z",
        )

        result = evaluate_uploaded_dataset_with_rule_pack(
            content,
            "selected.csv",
            approved,
            reference_date=REFERENCE_DATE,
            selected_metric_ids=selected,
        )
        self.assertEqual(
            result.baseline_report.to_dict(),
            report.to_dict(),
        )
        self.assertEqual(
            result.baseline_report.evaluation_context["selected_metric_ids"],
            list(selected),
        )

        with self.assertRaises(RulePackExecutionError):
            evaluate_uploaded_dataset_with_rule_pack(
                content,
                "selected.csv",
                approved,
                reference_date=REFERENCE_DATE,
                selected_metric_ids=("dataset_scale",),
            )

    def test_custom_selection_report_matches_published_schema(self):
        schema = json.loads(
            (ROOT / "schemas" / "quality-report.schema.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator.check_schema(schema)
        report = build_profile_report(
            SAMPLES / "good_dataset.csv",
            reference_date=REFERENCE_DATE,
            selected_metric_ids=(
                "field_missing_rate",
                "db31_010101",
                "db31_030300",
            ),
        )

        errors = list(Draft202012Validator(schema).iter_errors(report.to_dict()))
        self.assertEqual(errors, [])

    def test_failed_workflow_keeps_only_selected_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "broken.json"
            path.write_text('{"data": [', encoding="utf-8")
            report = build_profile_report(
                path,
                reference_date=REFERENCE_DATE,
                selected_metric_ids=("dataset_scale", "db31_060100"),
            )

        self.assertEqual(report.status, "failed")
        self.assertEqual(
            [metric.id for metric in report.metrics],
            ["dataset_scale", "db31_060100"],
        )
        self.assertEqual(
            report.evaluation_context["selected_metric_ids"],
            ["dataset_scale", "db31_060100"],
        )


if __name__ == "__main__":
    unittest.main()
