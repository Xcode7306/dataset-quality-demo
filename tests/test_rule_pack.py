"""v0.4 RulePack 核心协议、校验、审批和本地引导测试。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator

from src.rule_pack import (
    FieldSemanticMapping,
    LEGACY_RULE_PACK_MISSING_METRIC_TARGETS_ERROR,
    MAX_ALLOWED_VALUES,
    MAX_RULE_PACK_EVIDENCE,
    MAX_RULE_NUMBER_ABS,
    Rule,
    RulePackValidationError,
    approve_rule_pack,
    build_rule_guidance,
    build_rule_pack,
    draft_sha256,
    is_rule_pack_executable,
    validate_rule_pack,
)
from src.rule_dsl import new_evidence
from src.workflow import build_profile_report


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "sample_data"
SCHEMAS = ROOT / "schemas"
REFERENCE_DATE = date(2026, 7, 17)
GENERATED_AT = datetime(2026, 7, 29, 8, 30, tzinfo=timezone.utc)


class DictReport:
    """为绑定与隐私边界测试提供独立报告快照。"""

    def __init__(self, payload):
        self.payload = deepcopy(payload)

    def to_dict(self):
        return deepcopy(self.payload)


def all_rule_types() -> tuple[Rule, ...]:
    return (
        Rule(
            type="primary_key",
            rule_id="pk-record",
            fields=("record_id",),
        ),
        Rule(
            type="required",
            rule_id="required-service",
            fields=("service_name",),
        ),
        Rule(
            type="update_freshness",
            rule_id="freshness-update",
            fields=("update_time",),
            frequency="monthly",
            max_age_days=31,
        ),
        Rule(
            type="allowed_values",
            rule_id="allowed-department",
            fields=("department",),
            allowed_values=("政务服务中心", "业务处室", True, 1),
        ),
        Rule(
            type="numeric_range",
            rule_id="range-days",
            fields=("handling_days",),
            minimum=0,
            maximum=365,
        ),
    )


class RulePackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_profile_report(
            SAMPLES / "good_dataset.csv",
            reference_date=REFERENCE_DATE,
        )
        cls.schema = json.loads(
            (SCHEMAS / "rule-pack.schema.json").read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def build_pack(self, rules=None):
        return build_rule_pack(
            self.report,
            name="政务服务业务规则",
            version="1.0",
            rules=rules or all_rule_types(),
            generated_at=GENERATED_AT,
        )

    def assert_schema_valid(self, payload):
        errors = sorted(
            self.validator.iter_errors(payload),
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

    def test_build_binds_success_report_and_derives_semantics(self):
        pack = self.build_pack()
        context = self.report.to_dict()["evaluation_context"]

        self.assertEqual(pack.status, "draft")
        self.assertIsNone(pack.approval)
        self.assertEqual(pack.base_report_sha256, context["report_sha256"])
        self.assertEqual(pack.base_input_sha256, context["input_sha256"])
        self.assertEqual(pack.reference_date, REFERENCE_DATE.isoformat())
        self.assertEqual(pack.field_semantics.primary_key_fields, ("record_id",))
        self.assertEqual(
            pack.field_semantics.required_fields,
            ("service_name",),
        )
        self.assertEqual(pack.field_semantics.update_time_field, "update_time")
        self.assertEqual(
            pack.field_semantics.categorical_fields,
            ("department",),
        )
        self.assertEqual(
            pack.field_semantics.numeric_fields,
            ("handling_days",),
        )
        self.assertTrue(validate_rule_pack(pack, self.report).valid)

    def test_hash_and_id_are_deterministic_and_cover_configuration(self):
        first = self.build_pack()
        second = self.build_pack()

        self.assertEqual(first, second)
        self.assertEqual(draft_sha256(first), draft_sha256(second))
        self.assertEqual(
            first.rule_pack_id,
            f"rulepack-{draft_sha256(first)[:20]}",
        )
        for changed in (
            replace(first, name="另一规则包"),
            replace(first, version="1.1"),
            replace(
                first,
                rules=(
                    *first.rules[:-1],
                    replace(first.rules[-1], maximum=366),
                ),
            ),
            replace(
                first,
                field_semantics=replace(
                    first.field_semantics,
                    numeric_fields=("record_id",),
                ),
            ),
            replace(first, base_input_sha256="f" * 64),
            replace(first, reference_date="2026-07-18"),
        ):
            self.assertNotEqual(draft_sha256(first), draft_sha256(changed))

    def test_schema_accepts_draft_and_approved_payloads(self):
        draft = self.build_pack()
        approved = approve_rule_pack(
            draft,
            self.report,
            approver="本地审批人",
            approved_at=GENERATED_AT,
        )

        self.assert_schema_valid(draft.to_dict())
        self.assert_schema_valid(approved.to_dict())

    def test_evidence_limit_matches_one_hundred_rule_batch_contract_and_schema(self):
        evidence = tuple(
            new_evidence(
                "user_statement",
                f"第 {index + 1} 条批量规则的用户依据。",
                source_id=f"batch-input-{index + 1}",
                source_label="用户评价依据",
                location=f"batch:{index + 1}",
            ).to_dict()
            for index in range(MAX_RULE_PACK_EVIDENCE)
        )
        pack = build_rule_pack(
            self.report,
            name="批量规则证据上限",
            version="1.0",
            rules=(
                Rule(
                    type="required",
                    rule_id="required-service",
                    fields=("service_name",),
                ),
            ),
            evidence=evidence,
            generated_at=GENERATED_AT,
        )
        self.assertEqual(len(pack.evidence), MAX_RULE_PACK_EVIDENCE)
        self.assert_schema_valid(pack.to_dict())

        with self.assertRaisesRegex(RulePackValidationError, "300 条依据"):
            build_rule_pack(
                self.report,
                name="超限批量规则证据",
                version="1.0",
                rules=(
                    Rule(
                        type="required",
                        rule_id="required-service",
                        fields=("service_name",),
                    ),
                ),
                evidence=(*evidence, evidence[0]),
                generated_at=GENERATED_AT,
            )

    def test_legacy_v11_payload_without_metric_targets_is_explicitly_rejected(self):
        payload = self.build_pack().to_dict()
        payload.pop("metric_targets")

        validation = validate_rule_pack(payload, self.report)

        self.assertFalse(validation.valid)
        self.assertEqual(
            validation.errors,
            (LEGACY_RULE_PACK_MISSING_METRIC_TARGETS_ERROR,),
        )
        self.assertFalse(is_rule_pack_executable(payload, self.report))
        schema_errors = list(self.validator.iter_errors(payload))
        self.assertTrue(
            any(
                error.validator == "required"
                and "metric_targets" in error.message
                for error in schema_errors
            )
        )

    def test_unapproved_pack_cannot_execute_and_local_approval_is_traceable(self):
        draft = self.build_pack()

        self.assertFalse(is_rule_pack_executable(draft, self.report))
        self.assertFalse(
            validate_rule_pack(
                draft,
                self.report,
                require_approved=True,
            ).valid
        )

        approved = approve_rule_pack(
            draft,
            self.report,
            approver=" 本地审批人 ",
            approved_at=GENERATED_AT,
        )
        self.assertTrue(is_rule_pack_executable(approved, self.report))
        self.assertEqual(approved.approval.approver, "本地审批人")
        self.assertEqual(
            approved.approval.approved_at,
            "2026-07-29T08:30:00Z",
        )
        self.assertEqual(
            approved.approval.draft_sha256,
            draft_sha256(draft),
        )
        self.assertFalse(approved.approval.identity_verified)

    def test_approval_cannot_predate_the_rule_draft(self):
        with self.assertRaisesRegex(
            RulePackValidationError,
            "不能早于",
        ):
            approve_rule_pack(
                self.build_pack(),
                self.report,
                approver="reviewer",
                approved_at="2026-07-29T08:29:59Z",
            )

    def test_approval_hash_breaks_after_any_configuration_change(self):
        approved = approve_rule_pack(
            self.build_pack(),
            self.report,
            approver="reviewer",
            approved_at=GENERATED_AT,
        )
        changed = replace(
            approved,
            rules=(
                *approved.rules[:-1],
                replace(approved.rules[-1], maximum=999),
            ),
        )

        result = validate_rule_pack(
            changed,
            self.report,
            require_approved=True,
        )
        self.assertFalse(result.valid)
        self.assertFalse(is_rule_pack_executable(changed, self.report))
        self.assertTrue(
            any("草案哈希" in error or "ID" in error for error in result.errors)
        )

    def test_report_input_and_reference_bindings_are_rechecked(self):
        approved = approve_rule_pack(
            self.build_pack(),
            self.report,
            approver="reviewer",
            approved_at=GENERATED_AT,
        )
        original = self.report.to_dict()

        for key, changed_value in (
            ("report_sha256", "a" * 64),
            ("input_sha256", "b" * 64),
            ("reference_date", "2026-07-18"),
        ):
            payload = deepcopy(original)
            payload["evaluation_context"][key] = changed_value
            with self.subTest(binding=key):
                self.assertFalse(
                    is_rule_pack_executable(approved, DictReport(payload))
                )

        failed = deepcopy(original)
        failed["status"] = "failed"
        self.assertFalse(
            is_rule_pack_executable(approved, DictReport(failed))
        )

    def test_primary_key_supports_composite_keys_up_to_five_fields(self):
        five_fields = (
            "record_id",
            "service_name",
            "department",
            "update_time",
            "version",
        )
        pack = self.build_pack(
            (
                Rule(
                    type="primary_key",
                    rule_id="composite-pk",
                    fields=five_fields,
                ),
            )
        )
        self.assertTrue(validate_rule_pack(pack, self.report).valid)

        six_fields = (*five_fields, "source_url")
        with self.assertRaises(RulePackValidationError):
            self.build_pack(
                (
                    Rule(
                        type="primary_key",
                        rule_id="too-wide-pk",
                        fields=six_fields,
                    ),
                )
            )

    def test_single_field_and_duplicate_rule_constraints(self):
        invalid_cases = (
            (
                Rule(
                    type="required",
                    rule_id="required-two",
                    fields=("service_name", "department"),
                ),
            ),
            (
                Rule(
                    type="required",
                    rule_id="same-id",
                    fields=("service_name",),
                ),
                Rule(
                    type="allowed_values",
                    rule_id="same-id",
                    fields=("department",),
                    allowed_values=("A",),
                ),
            ),
            (
                Rule(
                    type="required",
                    rule_id="required-a",
                    fields=("service_name",),
                ),
                Rule(
                    type="required",
                    rule_id="required-b",
                    fields=("service_name",),
                ),
            ),
        )
        for rules in invalid_cases:
            with self.subTest(rules=rules):
                with self.assertRaises(RulePackValidationError):
                    self.build_pack(rules)

    def test_freshness_and_numeric_range_validate_parameters_and_types(self):
        invalid_rules = (
            Rule(
                type="update_freshness",
                rule_id="bad-frequency",
                fields=("update_time",),
                frequency="whenever",
                max_age_days=7,
            ),
            Rule(
                type="update_freshness",
                rule_id="wrong-date-type",
                fields=("service_name",),
                frequency="monthly",
                max_age_days=31,
            ),
            Rule(
                type="numeric_range",
                rule_id="no-bound",
                fields=("handling_days",),
            ),
            Rule(
                type="numeric_range",
                rule_id="reversed",
                fields=("handling_days",),
                minimum=10,
                maximum=1,
            ),
            Rule(
                type="numeric_range",
                rule_id="wrong-numeric-type",
                fields=("service_name",),
                minimum=0,
            ),
        )
        for rule in invalid_rules:
            with self.subTest(rule=rule.rule_id):
                with self.assertRaises(RulePackValidationError):
                    self.build_pack((rule,))

    def test_allowed_values_are_bounded_finite_non_null_json_scalars(self):
        invalid_values = (
            (None,),
            (math.inf,),
            (MAX_RULE_NUMBER_ABS + 1,),
            ({"nested": True},),
            tuple(range(MAX_ALLOWED_VALUES + 1)),
            (1, 1.0),
        )
        for values in invalid_values:
            with self.subTest(values_type=type(values[0]).__name__):
                with self.assertRaises(RulePackValidationError):
                    self.build_pack(
                        (
                            Rule(
                                type="allowed_values",
                                rule_id="bad-values",
                                fields=("department",),
                                allowed_values=values,
                            ),
                        )
                    )

        with self.assertRaises(RulePackValidationError):
            self.build_pack(
                (
                    Rule(
                        type="numeric_range",
                        rule_id="huge-bound",
                        fields=("handling_days",),
                        minimum=MAX_RULE_NUMBER_ABS + 1,
                    ),
                )
            )

    def test_semantic_mapping_drift_is_rejected(self):
        pack = self.build_pack()
        drifted = replace(
            pack,
            field_semantics=FieldSemanticMapping(
                primary_key_fields=("record_id",),
                required_fields=("department",),
                update_time_field="update_time",
                categorical_fields=("department",),
                numeric_fields=("handling_days",),
            ),
        )
        result = validate_rule_pack(drifted, self.report)

        self.assertFalse(result.valid)
        self.assertTrue(
            any("语义映射" in error for error in result.errors)
        )

    def test_local_guidance_uses_only_profile_metadata(self):
        payload = self.report.to_dict()
        payload["profile"]["columns"][0]["non_null_samples"] = [
            "SECRET_RAW_VALUE"
        ]
        payload["profile"]["raw_records"] = [
            {"record_id": "SECRET_RAW_VALUE"}
        ]

        guidance = build_rule_guidance(DictReport(payload))
        serialized = json.dumps(guidance.to_dict(), ensure_ascii=False)

        self.assertIn("record_id", guidance.primary_key_candidates)
        self.assertIn("update_time", guidance.update_time_candidates)
        self.assertIn("handling_days", guidance.numeric_field_candidates)
        self.assertEqual(len(guidance.questions), 5)
        self.assertNotIn("SECRET_RAW_VALUE", serialized)

    def test_failed_report_cannot_build_a_rule_pack(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "broken.json"
            path.write_text('{"data": [', encoding="utf-8")
            failed_report = build_profile_report(
                path,
                reference_date=REFERENCE_DATE,
            )

        with self.assertRaises(RulePackValidationError):
            build_rule_pack(
                failed_report,
                name="invalid",
                version="1.0",
                rules=(
                    Rule(
                        type="required",
                        rule_id="required",
                        fields=("record_id",),
                    ),
                ),
                generated_at=GENERATED_AT,
            )


if __name__ == "__main__":
    unittest.main()
