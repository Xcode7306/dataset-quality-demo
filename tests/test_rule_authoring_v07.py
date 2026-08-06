"""v0.7 规则编制 Agent 的协议、回退、校验和试运行测试。"""

from datetime import date
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

from src.rule_authoring_providers import (
    RuleAuthoringProviderResult,
    TemplateRuleAuthoringProvider,
    build_rule_input_guidance,
)
from src.rule_authoring_service import (
    build_rule_pack_from_draft,
    compile_rule_draft,
    validate_rule_draft,
)
from src.rule_authoring_workflow import RuleAuthoringWorkflow
from src.rule_dsl import (
    ProviderMetadata,
    RuleSpec,
    make_workflow_id,
)
from src.rule_service import dry_run_uploaded_dataset_with_rule_pack
from src.workflow import build_profile_report


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data" / "good_dataset.csv"
REFERENCE_DATE = date(2026, 7, 17)


class RuleAuthoringV07Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_profile_report(
            SAMPLE,
            reference_date=REFERENCE_DATE,
        )
        cls.schema = json.loads(
            (ROOT / "schemas" / "rule-draft.schema.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator.check_schema(cls.schema)

    def assert_schema_valid(self, payload):
        errors = sorted(
            Draft202012Validator(self.schema).iter_errors(payload),
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

    def test_required_rule_compiles_validates_and_dry_runs(self):
        draft = compile_rule_draft(
            self.report,
            target_metric_id="db31_020100",
            user_intent="service_name为必填字段",
            created_at="2026-07-29T08:30:00Z",
        )
        self.assertEqual(draft.status, "draft")
        self.assertEqual(draft.rule_spec.rule_type, "required")
        self.assertEqual(draft.rule_spec.fields, ("service_name",))
        self.assertTrue(validate_rule_draft(draft, self.report).valid)
        self.assert_schema_valid(draft.to_dict())

        pack = build_rule_pack_from_draft(draft, self.report)
        self.assertEqual(pack.status, "draft")
        self.assertEqual(pack.source.type, "user_natural_language")
        preview = dry_run_uploaded_dataset_with_rule_pack(
            SAMPLE.read_bytes(),
            SAMPLE.name,
            pack,
            reference_date=REFERENCE_DATE,
        )
        self.assertEqual(preview.to_dict()["counts"]["checked"], 5)
        self.assertEqual(preview.to_dict()["counts"]["issues"], 0)

    def test_allowed_values_and_vague_freshness_go_to_expected_states(self):
        allowed = compile_rule_draft(
            self.report,
            target_metric_id="db31_030200",
            user_intent="department只能为政务服务中心、业务处室",
            created_at="2026-07-29T08:30:00Z",
        )
        self.assertEqual(allowed.rule_spec.rule_type, "allowed_values")
        self.assertEqual(
            allowed.rule_spec.parameters["allowed_values"],
            ["政务服务中心", "业务处室"],
        )

        vague = compile_rule_draft(
            self.report,
            target_metric_id="db31_050200",
            user_intent="更新时间应当及时",
            created_at="2026-07-29T08:30:00Z",
        )
        self.assertEqual(vague.status, "needs_clarification")
        self.assertIsNone(vague.rule_spec)
        self.assertTrue(vague.clarification_questions)
        self.assertFalse(validate_rule_draft(vague, self.report).valid)

    def test_arbitrary_code_is_rejected(self):
        draft = compile_rule_draft(
            self.report,
            target_metric_id="db31_030200",
            user_intent="调用 Python 自动判断异常",
            created_at="2026-07-29T08:30:00Z",
        )
        self.assertEqual(draft.status, "rejected")
        self.assertIsNone(draft.rule_spec)
        self.assertIn("Python", draft.unsupported_reason)

    def test_nonexistent_field_is_rejected_by_deterministic_validation(self):
        class FakeProvider:
            def generate(self, context, *, user_intent):
                return RuleAuthoringProviderResult(
                    outcome="draft",
                    rule_spec=RuleSpec(
                        rule_type="required",
                        rule_id="candidate-rule",
                        name="不存在字段必填",
                        description="测试字段校验",
                        fields=("field_not_in_profile",),
                    ),
                    metadata=ProviderMetadata(
                        provider="test",
                        model=None,
                        mode="template",
                        prompt_version="test",
                    ),
                )

        draft = compile_rule_draft(
            self.report,
            target_metric_id="db31_020100",
            user_intent="field_not_in_profile为必填字段",
            provider=FakeProvider(),
            created_at="2026-07-29T08:30:00Z",
        )
        validation = validate_rule_draft(draft, self.report)
        self.assertFalse(validation.valid)
        self.assertTrue(any("不存在的字段" in error for error in validation.errors))

    def test_workflow_state_is_controlled_locally(self):
        workflow = RuleAuthoringWorkflow(
            workflow_id=make_workflow_id("test"),
            target_metric_id="db31_020100",
        )
        compiling = workflow.start_compiling()
        draft = compile_rule_draft(
            self.report,
            target_metric_id="db31_020100",
            user_intent="service_name为必填字段",
            workflow_id=compiling.workflow_id,
            created_at="2026-07-29T08:30:00Z",
        )
        current = compiling.accept_draft(draft)
        self.assertEqual(current.state, "draft")
        current = current.mark_validated(validate_rule_draft(draft, self.report))
        self.assertEqual(current.state, "validated")
        current = current.mark_dry_run_complete({"counts": {"issues": 0}})
        current = current.await_approval().approve().execute()
        self.assertEqual(current.state, "executed")

    def test_template_provider_does_not_need_network(self):
        result = TemplateRuleAuthoringProvider().generate(
            {
                "fields": [{"name": "service_name", "inferred_type": "text"}]
            },
            user_intent="service_name为必填字段",
        )
        self.assertEqual(result.metadata.mode, "template")
        self.assertEqual(result.outcome, "draft")

    def test_missing_rule_guidance_names_required_inputs(self):
        guidance = build_rule_input_guidance(
            {
                "fields": [
                    {"name": "update_time", "inferred_type": "datetime"},
                ]
            },
            user_intent="更新时间应当及时",
        )
        self.assertTrue(any("更新频率" in item for item in guidance))
