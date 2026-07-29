"""v0.4 确定性业务规则引擎与上传重评服务测试。"""

import csv
from dataclasses import replace
from datetime import date
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

import src.rule_engine as rule_engine
from src.parser import parse_dataset
from src.resource_limits import ResourceLimitExceeded
from src.rule_engine import RulePackExecutionError
from src.rule_pack import Rule, approve_rule_pack, build_rule_pack
from src.rule_service import (
    evaluate_uploaded_dataset_with_rule_pack,
    serialize_rule_evaluation_result,
    serialize_rule_issue_locations_csv,
)
from src.upload_service import evaluate_uploaded_dataset
from src.workflow import build_profile_report


REFERENCE_DATE = date(2026, 7, 20)
FILE_NAME = "business-rules.csv"
CONTENT = (
    "record_id,name,updated_at,status,score,notes\n"
    "1,A,2026-07-15,active,0,private-note-alpha\n"
    "2,,bad,inactive,20,private-note-beta\n"
    "2,C,2026-07-30,other-secret,101,private-note-gamma\n"
    ",D,2026-07-01,,,private-note-delta\n"
).encode("utf-8")


def _rules():
    return (
        Rule(
            type="primary_key",
            rule_id="primary-key",
            fields=("record_id",),
        ),
        Rule(
            type="required",
            rule_id="required-name",
            fields=("name",),
        ),
        Rule(
            type="update_freshness",
            rule_id="update-weekly",
            fields=("updated_at",),
            frequency="weekly",
            max_age_days=7,
        ),
        Rule(
            type="allowed_values",
            rule_id="allowed-status",
            fields=("status",),
            allowed_values=("active", "inactive"),
        ),
        Rule(
            type="numeric_range",
            rule_id="score-range",
            fields=("score",),
            minimum=0,
            maximum=100,
        ),
    )


def _approved_pack(
    content=CONTENT,
    file_name=FILE_NAME,
    *,
    rules=None,
    reference_date=REFERENCE_DATE,
):
    report = evaluate_uploaded_dataset(
        content,
        file_name,
        reference_date=reference_date,
    )
    draft = build_rule_pack(
        report,
        name="测试业务规则",
        version="1.0",
        rules=rules or _rules(),
        generated_at="2026-07-20T00:00:00Z",
    )
    approved = approve_rule_pack(
        draft,
        report,
        approver="local-tester",
        approved_at="2026-07-20T00:01:00Z",
    )
    return report, draft, approved


def _business_metrics(result):
    return [
        metric
        for metric in result.enhanced_report.metrics
        if metric.id.startswith("business_")
    ]


def _find_metric(result, metric_id):
    for metric in _business_metrics(result):
        if metric.id == metric_id:
            return metric
    raise AssertionError(f"未找到业务指标 {metric_id}。")


class RuleEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_report, cls.draft, cls.approved = _approved_pack()

    def _evaluate(self):
        return evaluate_uploaded_dataset_with_rule_pack(
            CONTENT,
            FILE_NAME,
            self.approved,
            reference_date=REFERENCE_DATE,
        )

    def test_all_five_rule_types_append_deterministic_metrics_and_attention_risks(self):
        result = self._evaluate()
        metrics = _business_metrics(result)

        self.assertEqual(
            [metric.id for metric in metrics],
            [
                "business_primary_key_compliance",
                "business_required_compliance",
                "business_update_time_parseability",
                "business_update_frequency_compliance",
                "business_allowed_values_compliance",
                "business_numeric_range_compliance",
            ],
        )
        self.assertEqual(
            _find_metric(result, "business_primary_key_compliance").value,
            0.25,
        )
        self.assertEqual(
            _find_metric(result, "business_required_compliance").value,
            0.75,
        )
        self.assertEqual(
            _find_metric(result, "business_update_time_parseability").value,
            0.75,
        )
        freshness = _find_metric(
            result,
            "business_update_frequency_compliance",
        )
        self.assertEqual(freshness.value, 0.0)
        self.assertEqual(freshness.evidence["update_lag_days"], -10)
        self.assertTrue(freshness.evidence["future_date"])

        allowed = _find_metric(
            result,
            "business_allowed_values_compliance",
        )
        self.assertAlmostEqual(allowed.value, 2 / 3, places=6)
        self.assertEqual(allowed.evidence["excluded_missing_count"], 1)
        numeric = _find_metric(
            result,
            "business_numeric_range_compliance",
        )
        self.assertAlmostEqual(numeric.value, 2 / 3, places=6)
        self.assertEqual(numeric.evidence["excluded_missing_count"], 1)
        self.assertEqual(numeric.evidence["minimum"], 0)
        self.assertEqual(numeric.evidence["maximum"], 100)

        new_risks = result.enhanced_report.risks[
            len(result.baseline_report.risks) :
        ]
        self.assertEqual(len(new_risks), 6)
        self.assertTrue(all(risk.level == "attention" for risk in new_risks))
        self.assertTrue(
            all(
                risk.evidence["decision"]["threshold_config_version"].startswith(
                    "rule-pack:"
                )
                for risk in new_risks
            )
        )

    def test_primary_key_locations_cover_missing_and_all_duplicate_members(self):
        metric = _find_metric(
            self._evaluate(),
            "business_primary_key_compliance",
        )

        self.assertEqual(metric.evidence["issue_count"], 3)
        self.assertEqual(len(metric.issue_locations), 3)
        self.assertEqual(
            {
                (
                    location["record_number"],
                    location["issue_type"],
                )
                for location in metric.issue_locations
            },
            {
                (2, "rule_primary_key_duplicate"),
                (3, "rule_primary_key_duplicate"),
                (4, "rule_primary_key_missing"),
            },
        )
        self.assertTrue(
            all("value" not in location for location in metric.issue_locations)
        )

    def test_baseline_metrics_risks_threshold_context_and_input_are_unchanged(self):
        content_before = bytes(CONTENT)
        baseline_before = self.original_report.to_dict()
        original_location_count = sum(
            len(metric.issue_locations)
            for metric in self.original_report.metrics
        )

        result = self._evaluate()

        self.assertEqual(CONTENT, content_before)
        self.assertEqual(result.baseline_report.to_dict(), baseline_before)
        baseline_metric_count = len(result.baseline_report.metrics)
        baseline_risk_count = len(result.baseline_report.risks)
        self.assertEqual(
            result.enhanced_report.metrics[:baseline_metric_count],
            result.baseline_report.metrics,
        )
        self.assertEqual(
            result.enhanced_report.risks[:baseline_risk_count],
            result.baseline_report.risks,
        )
        self.assertEqual(
            result.enhanced_report.evaluation_context,
            result.baseline_report.evaluation_context,
        )
        self.assertEqual(
            len({metric.id for metric in result.baseline_report.metrics}),
            13,
        )
        self.assertGreater(original_location_count, 0)
        self.assertEqual(
            sum(
                len(metric.issue_locations)
                for metric in result.baseline_report.metrics
            ),
            0,
        )
        self.assertEqual(
            sum(
                len(metric.issue_locations)
                for metric in self.original_report.metrics
            ),
            original_location_count,
        )

    def test_same_approved_pack_and_input_are_deterministic(self):
        self.assertEqual(
            self._evaluate().to_dict(),
            self._evaluate().to_dict(),
        )

    def test_diff_is_v04_additions_only_and_contains_no_history_conclusion(self):
        payload = self._evaluate().diff.to_dict()

        self.assertEqual(payload["counts"]["added_metrics"], 6)
        self.assertEqual(payload["counts"]["added_evaluated_metrics"], 6)
        self.assertEqual(payload["counts"]["added_risks"], 6)
        self.assertEqual(payload["counts"]["added_not_assessable"], 0)
        self.assertEqual(payload["counts"]["added_issue_locations"], 7)
        serialized = json.dumps(payload, ensure_ascii=False)
        for forbidden in ("improved", "worsened", "已改善", "已恶化", "history"):
            self.assertNotIn(forbidden, serialized)

    def test_unapproved_stale_illegal_and_other_input_packs_are_rejected(self):
        with self.subTest(case="unapproved"):
            with self.assertRaises(RulePackExecutionError):
                evaluate_uploaded_dataset_with_rule_pack(
                    CONTENT,
                    FILE_NAME,
                    self.draft,
                    reference_date=REFERENCE_DATE,
                )

        with self.subTest(case="stale_after_edit"):
            stale = replace(self.approved, version="1.1")
            with self.assertRaises(RulePackExecutionError):
                evaluate_uploaded_dataset_with_rule_pack(
                    CONTENT,
                    FILE_NAME,
                    stale,
                    reference_date=REFERENCE_DATE,
                )

        with self.subTest(case="illegal_rule"):
            illegal = replace(
                self.approved,
                rules=(
                    *self.approved.rules,
                    Rule(
                        type="required",
                        rule_id="missing-field",
                        fields=("does_not_exist",),
                    ),
                ),
            )
            with self.assertRaises(RulePackExecutionError):
                evaluate_uploaded_dataset_with_rule_pack(
                    CONTENT,
                    FILE_NAME,
                    illegal,
                    reference_date=REFERENCE_DATE,
                )

        with self.subTest(case="input_hash_mismatch"):
            changed = CONTENT + (
                b"5,E,2026-07-20,active,50,private-note-epsilon\n"
            )
            with self.assertRaises(RulePackExecutionError):
                evaluate_uploaded_dataset_with_rule_pack(
                    changed,
                    FILE_NAME,
                    self.approved,
                    reference_date=REFERENCE_DATE,
                )

    def test_unverified_dataframe_entrypoint_is_not_public(self):
        self.assertFalse(hasattr(rule_engine, "evaluate_rule_pack"))
        self.assertTrue(
            hasattr(
                rule_engine,
                "_evaluate_rule_pack_on_verified_dataframe",
            )
        )

    def test_rule_issue_locations_have_an_independent_resource_limit(self):
        with patch(
            "src.rule_engine.MAX_RULE_ISSUE_LOCATIONS",
            2,
        ):
            with self.assertRaisesRegex(
                RulePackExecutionError,
                "疑似问题位置",
            ):
                self._evaluate()

    def test_rule_inspection_has_a_lower_independent_resource_limit(self):
        with patch(
            "src.rule_engine.MAX_RULE_INSPECTION_CELLS",
            19,
        ):
            with self.assertRaisesRegex(
                RulePackExecutionError,
                "预计检查 20",
            ):
                self._evaluate()

    def test_no_parseable_update_keeps_parse_rate_and_marks_freshness_unassessable(self):
        content = (
            "record_id,updated_at\n"
            "1,bad-time\n"
            "2,\n"
        ).encode("utf-8")
        rule = Rule(
            type="update_freshness",
            rule_id="freshness",
            fields=("updated_at",),
            frequency="daily",
            max_age_days=1,
        )
        _, _, approved = _approved_pack(content, "invalid-times.csv", rules=(rule,))

        result = evaluate_uploaded_dataset_with_rule_pack(
            content,
            "invalid-times.csv",
            approved,
            reference_date=REFERENCE_DATE,
        )

        parseability = _find_metric(
            result,
            "business_update_time_parseability",
        )
        freshness = _find_metric(
            result,
            "business_update_frequency_compliance",
        )
        self.assertEqual(parseability.status, "evaluated")
        self.assertEqual(parseability.value, 0.0)
        self.assertEqual(len(parseability.issue_locations), 2)
        self.assertEqual(freshness.status, "not_assessable")
        self.assertEqual(result.diff.counts["added_not_assessable"], 1)
        self.assertFalse(
            any(
                risk.related_metric_keys == [freshness.metric_key]
                for risk in result.enhanced_report.risks
            )
        )

    def test_allowed_values_share_numeric_semantics_but_keep_boolean_distinct(self):
        numeric_content = (
            b'[{"code": 1.0, "label": "a"}, '
            b'{"code": null, "label": "b"}]'
        )
        numeric_rule = Rule(
            type="allowed_values",
            rule_id="numeric-allowed",
            fields=("code",),
            allowed_values=(1,),
        )
        _, _, numeric_pack = _approved_pack(
            numeric_content,
            "numeric-allowed.json",
            rules=(numeric_rule,),
        )
        numeric_result = evaluate_uploaded_dataset_with_rule_pack(
            numeric_content,
            "numeric-allowed.json",
            numeric_pack,
            reference_date=REFERENCE_DATE,
        )
        numeric_metric = _find_metric(
            numeric_result,
            "business_allowed_values_compliance",
        )
        self.assertEqual(numeric_metric.value, 1.0)
        self.assertEqual(numeric_metric.evidence["excluded_missing_count"], 1)

        boolean_content = b'[{"choice": true}, {"choice": 1}]'
        boolean_rule = Rule(
            type="allowed_values",
            rule_id="boolean-allowed",
            fields=("choice",),
            allowed_values=(True,),
        )
        _, _, boolean_pack = _approved_pack(
            boolean_content,
            "boolean-allowed.json",
            rules=(boolean_rule,),
        )
        boolean_result = evaluate_uploaded_dataset_with_rule_pack(
            boolean_content,
            "boolean-allowed.json",
            boolean_pack,
            reference_date=REFERENCE_DATE,
        )
        boolean_metric = _find_metric(
            boolean_result,
            "business_allowed_values_compliance",
        )
        self.assertEqual(boolean_metric.value, 0.5)

    def test_empty_upload_business_rules_are_not_assessable(self):
        content = b"record_id,name,status\n"
        rules = (
            Rule(
                type="primary_key",
                rule_id="pk",
                fields=("record_id",),
            ),
            Rule(
                type="required",
                rule_id="required",
                fields=("name",),
            ),
            Rule(
                type="allowed_values",
                rule_id="allowed",
                fields=("status",),
                allowed_values=("active",),
            ),
        )
        _, _, approved = _approved_pack(
            content,
            "empty.csv",
            rules=rules,
        )

        result = evaluate_uploaded_dataset_with_rule_pack(
            content,
            "empty.csv",
            approved,
            reference_date=REFERENCE_DATE,
        )

        self.assertTrue(
            all(metric.status == "not_assessable" for metric in _business_metrics(result))
        )
        self.assertEqual(result.diff.counts["added_not_assessable"], 3)
        self.assertEqual(result.diff.counts["added_risks"], 0)

    def test_json_and_csv_downloads_are_strict_and_do_not_expose_raw_values(self):
        result = self._evaluate()
        json_payload = serialize_rule_evaluation_result(result).decode("utf-8")
        parsed = json.loads(
            json_payload,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
        self.assertEqual(parsed, result.to_dict())

        def assert_no_location_key(value):
            if isinstance(value, dict):
                self.assertNotIn("issue_locations", value)
                for nested in value.values():
                    assert_no_location_key(nested)
            elif isinstance(value, list):
                for nested in value:
                    assert_no_location_key(nested)

        assert_no_location_key(parsed)

        csv_payload = serialize_rule_issue_locations_csv(result).decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(csv_payload)))
        self.assertEqual(
            len(rows),
            result.diff.counts["added_issue_locations"],
        )
        for private_value in (
            "private-note-alpha",
            "private-note-beta",
            "private-note-gamma",
            "private-note-delta",
            "other-secret",
        ):
            with self.subTest(private_value=private_value):
                self.assertNotIn(private_value, json_payload)
                self.assertNotIn(private_value, csv_payload)

    def test_upload_reevaluation_reuses_workflow_parser_and_resource_limit(self):
        with (
            patch(
                "src.rule_service.validate_upload_size",
                wraps=__import__(
                    "src.resource_limits",
                    fromlist=["validate_upload_size"],
                ).validate_upload_size,
            ) as validate_size,
            patch(
                "src.rule_service.build_profile_report",
                wraps=build_profile_report,
            ) as build_report,
            patch(
                "src.rule_service.parse_dataset",
                wraps=parse_dataset,
            ) as parse,
        ):
            result = self._evaluate()

        self.assertEqual(result.baseline_report.status, "success")
        validate_size.assert_called_once_with(len(CONTENT))
        build_report.assert_called_once()
        parse.assert_called_once()

        with patch(
            "src.rule_service.validate_upload_size",
            side_effect=ResourceLimitExceeded("too large"),
        ):
            with self.assertRaisesRegex(Exception, "too large"):
                self._evaluate()


class RuleEvaluationSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema_dir = Path(__file__).resolve().parents[1] / "schemas"
        cls.schemas = [
            json.loads((schema_dir / name).read_text(encoding="utf-8"))
            for name in (
                "quality-report.schema.json",
                "rule-pack.schema.json",
                "rule-evaluation-result.schema.json",
            )
        ]
        for schema in cls.schemas:
            Draft202012Validator.check_schema(schema)
        registry = Registry().with_resources(
            (
                schema["$id"],
                Resource.from_contents(schema),
            )
            for schema in cls.schemas
        )
        cls.validator = Draft202012Validator(
            cls.schemas[-1],
            registry=registry,
        )
        _, cls.draft, cls.approved = _approved_pack()

    def _payload(self):
        return evaluate_uploaded_dataset_with_rule_pack(
            CONTENT,
            FILE_NAME,
            self.approved,
            reference_date=REFERENCE_DATE,
        ).to_dict()

    def test_result_matches_published_strict_schema(self):
        errors = sorted(
            self.validator.iter_errors(self._payload()),
            key=lambda error: list(error.absolute_path),
        )
        self.assertEqual(
            errors,
            [],
            "\n".join(
                f"{'/'.join(map(str, error.absolute_path))}: {error.message}"
                for error in errors
            ),
        )

    def test_schema_rejects_unknown_diff_fields_and_missing_approved_pack(self):
        payload = self._payload()
        payload["diff"]["historical_improvement"] = True
        self.assertTrue(list(self.validator.iter_errors(payload)))

        payload = self._payload()
        payload.pop("approved_rule_pack")
        self.assertTrue(list(self.validator.iter_errors(payload)))

        payload = self._payload()
        payload["approved_rule_pack"] = self.draft.to_dict()
        self.assertTrue(list(self.validator.iter_errors(payload)))


if __name__ == "__main__":
    unittest.main()
