"""Conservative compatibility coverage for structured model rule output."""

from __future__ import annotations

import unittest

from src.rule_authoring_providers import (
    RuleAuthoringProviderError,
    inspect_rule_intent,
    parse_provider_payload,
)


class RuleProviderCompatibilityTests(unittest.TestCase):
    def test_draft_outcome_aliases_require_and_preserve_rule_spec(self):
        for outcome in ("success", "ready", "valid", "成功"):
            with self.subTest(outcome=outcome):
                result = parse_provider_payload(
                    {
                        "outcome": outcome,
                        "rule_spec": {
                            "rule_type": "required",
                            "fields": ["指标名称"],
                            "parameters": {},
                        },
                    }
                )

                self.assertEqual(result.outcome, "draft")
                self.assertIsNotNone(result.rule_spec)
                self.assertEqual(result.rule_spec.rule_type, "required")
                self.assertEqual(result.rule_spec.fields, ("指标名称",))

    def test_common_wrapper_aliases_are_normalized_when_unambiguous(self):
        result = parse_provider_payload(
            {
                "status": "ready",
                "rule": {
                    "rule_type": "numeric_range",
                    "fields": ["总人数"],
                    "parameters": {"minimum": 0, "maximum": 10_000},
                },
            }
        )

        self.assertEqual(result.outcome, "draft")
        self.assertIsNotNone(result.rule_spec)
        self.assertEqual(
            result.rule_spec.parameters,
            {"minimum": 0, "maximum": 10_000},
        )

    def test_clarification_and_unsupported_aliases_remain_non_executable(self):
        clarification = parse_provider_payload(
            {"status": "needs-clarification", "questions": ["请补充阈值。"]}
        )
        unsupported = parse_provider_payload(
            {"status": "out-of-scope", "reason": "需要跨表查询。"}
        )

        self.assertEqual(clarification.outcome, "clarification")
        self.assertIsNone(clarification.rule_spec)
        self.assertEqual(
            clarification.clarification_questions,
            ("请补充阈值。",),
        )
        self.assertEqual(unsupported.outcome, "unsupported")
        self.assertIsNone(unsupported.rule_spec)

    def test_compatibility_layer_rejects_conflicts_and_privilege_fields(self):
        invalid_payloads = (
            {
                "outcome": "draft",
                "status": "ready",
                "rule_spec": {"rule_type": "required", "fields": ["x"]},
            },
            {"status": "success", "rule": None},
            {"outcome": "anything", "rule_spec": None},
            {
                "outcome": "success",
                "rule_spec": {"rule_type": "required", "fields": ["x"]},
                "approval": {"approved": True},
            },
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(RuleAuthoringProviderError):
                    parse_provider_payload(payload)

    def test_outcome_state_invariants_are_enforced(self):
        valid_rule = {
            "rule_type": "required",
            "fields": ["指标名称"],
            "parameters": {},
        }
        invalid_payloads = (
            {
                "outcome": "draft",
                "rule_spec": valid_rule,
                "clarification_questions": ["请补充。"],
            },
            {
                "outcome": "draft",
                "rule_spec": valid_rule,
                "unsupported_reason": "超出范围。",
            },
            {
                "outcome": "clarification",
                "rule_spec": None,
                "clarification_questions": [],
            },
            {
                "outcome": "clarification",
                "rule_spec": None,
                "clarification_questions": ["请补充。"],
                "unsupported_reason": "超出范围。",
            },
            {
                "outcome": "unsupported",
                "rule_spec": None,
                "clarification_questions": ["请补充。"],
                "unsupported_reason": "暂不支持。",
            },
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(RuleAuthoringProviderError):
                    parse_provider_payload(payload)

    def test_unknown_outcome_is_not_echoed_in_error_text(self):
        marker = "secret-looking-provider-text"

        with self.assertRaises(RuleAuthoringProviderError) as raised:
            parse_provider_payload({"outcome": marker, "rule_spec": None})

        self.assertNotIn(marker, str(raised.exception))

    def test_multi_rule_intent_is_blocked_before_provider_draft(self):
        context = {
            "fields": [
                {"name": "总人数", "inferred_type": "numeric"},
                {"name": "统计时间", "inferred_type": "text"},
            ]
        }
        inspection = inspect_rule_intent(
            context,
            user_intent="总人数必须填写；统计时间必须是四位数字。",
        )

        self.assertIsNone(inspection.recognized_rule_type)
        self.assertFalse(inspection.complete)
        self.assertIn("多条独立规则", inspection.clarification_questions[0])

    def test_single_range_with_two_bounds_is_not_treated_as_multi_rule(self):
        context = {
            "fields": [{"name": "总人数", "inferred_type": "numeric"}]
        }
        inspection = inspect_rule_intent(
            context,
            user_intent="总人数不低于0且不超过5000。",
        )

        self.assertEqual("numeric_range", inspection.recognized_rule_type)
        self.assertTrue(inspection.complete)


if __name__ == "__main__":
    unittest.main()
