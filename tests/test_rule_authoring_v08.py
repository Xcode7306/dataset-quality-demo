"""v0.8 自然语言自定义规则、协议和确定性执行回归。"""

from datetime import date
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator
from streamlit.testing.v1 import AppTest

from src.rule_authoring_service import (
    build_rule_pack_from_draft,
    compile_custom_rule_draft,
    validate_rule_draft,
)
from src.rule_pack import (
    Rule,
    RulePackValidationError,
    approve_rule_pack,
    build_rule_pack,
)
from src.rule_service import evaluate_uploaded_dataset_with_rule_pack
from src.upload_service import evaluate_uploaded_dataset


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DATE = date(2026, 7, 20)
CONTENT = (
    "status,name,start_date,end_date,code\n"
    "active,A,2026-01-02,2026-01-03,123456\n"
    "inactive,,2026-01-04,2026-01-03,12345\n"
).encode("utf-8")


class RuleAuthoringV08Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = evaluate_uploaded_dataset(
            CONTENT,
            "custom-rules.csv",
            reference_date=REFERENCE_DATE,
        )
        cls.draft_schema = json.loads(
            (ROOT / "schemas" / "rule-draft.schema.json").read_text(encoding="utf-8")
        )
        cls.pack_schema = json.loads(
            (ROOT / "schemas" / "rule-pack.schema.json").read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(cls.draft_schema)
        Draft202012Validator.check_schema(cls.pack_schema)

    def assert_valid(self, schema, payload):
        errors = list(Draft202012Validator(schema).iter_errors(payload))
        self.assertEqual([], errors, "\n".join(error.message for error in errors))

    def _approved_result(self, intent):
        draft = compile_custom_rule_draft(
            self.report,
            user_intent=intent,
            created_at="2026-08-06T00:00:00Z",
        )
        self.assertEqual("custom_rule", draft.target_type)
        self.assertIsNone(draft.target_metric_id)
        self.assertEqual("draft", draft.status)
        self.assertTrue(validate_rule_draft(draft, self.report).valid)
        self.assert_valid(self.draft_schema, draft.to_dict())
        pack = build_rule_pack_from_draft(draft, self.report)
        self.assertEqual("0.8.0", pack.version)
        self.assertEqual("quality-rule-agent-v0.8", pack.source.generator)
        self.assert_valid(self.pack_schema, pack.to_dict())
        approved = approve_rule_pack(
            pack,
            self.report,
            approver="v0.8-tester",
            approved_at="2026-08-06T00:01:00Z",
        )
        self.assert_valid(self.pack_schema, approved.to_dict())
        return evaluate_uploaded_dataset_with_rule_pack(
            CONTENT,
            "custom-rules.csv",
            approved,
            reference_date=REFERENCE_DATE,
        )

    def test_all_four_new_rule_types_compile_and_execute(self):
        cases = (
            ("code必须匹配正则 ^\\d{6}$", "regex_format", 0.5),
            ("name长度为1位", "string_length", 1.0),
            ("status为inactive时，name必须填写", "conditional_required", 0.0),
            ("start_date不得晚于end_date", "field_comparison", 0.5),
        )
        for intent, rule_type, expected_value in cases:
            with self.subTest(rule_type=rule_type):
                result = self._approved_result(intent)
                metric = next(
                    metric
                    for metric in result.enhanced_report.metrics
                    if metric.id == f"business_{rule_type}_compliance"
                )
                self.assertEqual(expected_value, metric.value)
                self.assertTrue(metric.issue_locations or expected_value == 1.0)

    def test_unsupported_cross_table_and_arbitrary_code_are_rejected(self):
        for intent in (
            "code必须存在于权威区划表",
            "运行 Python 计算这条规则",
        ):
            with self.subTest(intent=intent):
                draft = compile_custom_rule_draft(
                    self.report,
                    user_intent=intent,
                    created_at="2026-08-06T00:00:00Z",
                )
                self.assertEqual("rejected", draft.status)
                self.assertIsNone(draft.rule_spec)
                self.assertTrue(draft.unsupported_reason)

    def test_regex_safety_and_field_type_validation_are_deterministic(self):
        with self.assertRaises(RulePackValidationError):
            build_rule_pack(
                self.report,
                name="unsafe-regex",
                version="0.8.0",
                rules=(
                    Rule(
                        type="regex_format",
                        rule_id="unsafe-regex",
                        fields=("code",),
                        regex_pattern="(a+)+$",
                    ),
                ),
            )

    def test_streamlit_custom_rule_entry_reaches_dry_run(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        app.file_uploader[0].set_value(
            ("custom-rules.csv", CONTENT, "text/csv")
        ).run()
        app.date_input[0].set_value(REFERENCE_DATE)
        next(button for button in app.button if button.label == "运行质量评估").click().run()
        next(
            item for item in app.text_input if item.label == "自定义规则描述"
        ).set_value("status为inactive时，name必须填写").run()
        next(button for button in app.button if button.label == "AI 解析自定义规则").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("自定义规则已通过" in item.value for item in app.success))
        next(button for button in app.button if button.label == "试运行自定义规则").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("规则试运行完成" in item.value for item in app.success))


if __name__ == "__main__":
    unittest.main()
