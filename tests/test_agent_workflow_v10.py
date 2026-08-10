"""v1.0 explicit workflow, bounded history, and recovery acceptance tests."""

from datetime import date
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from streamlit.testing.v1 import AppTest

from src.rule_authoring_coordinator import (
    RuleAuthoringCoordinatorError,
    approve_rule_authoring_run,
    begin_rule_authoring_run,
    compile_rule_authoring_run,
    dry_run_rule_authoring_run,
    execute_rule_authoring_run,
    retry_rule_authoring_run,
    validate_rule_authoring_run,
)
from src.rule_authoring_providers import RuleAuthoringProviderResult
from src.rule_authoring_workflow import (
    RuleAuthoringHistory,
    RuleAuthoringWorkflow,
    RuleAuthoringWorkflowError,
    new_rule_authoring_workflow,
    validate_rule_authoring_workflow,
)
from src.rule_dsl import ProviderMetadata, RuleSpec, make_workflow_id
from src.workflow import build_profile_report


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data" / "good_dataset.csv"
REFERENCE_DATE = date(2026, 7, 17)
CREATED_AT = "2026-08-10T03:00:00Z"


class _FailingProvider:
    def generate(self, context, *, user_intent):
        raise RuntimeError(
            "provider unavailable api_key=top-secret-key "
            "Bearer top-secret-bearer sk-topsecret123456"
        )


class _UnknownFieldProvider:
    def generate(self, context, *, user_intent):
        return RuleAuthoringProviderResult(
            outcome="draft",
            rule_spec=RuleSpec(
                rule_type="required",
                rule_id="candidate-rule",
                name="未知字段必填",
                description="用于验证确定性校验状态。",
                fields=("field_not_in_profile",),
            ),
            metadata=ProviderMetadata(
                provider="test",
                model=None,
                mode="template",
                prompt_version="test",
            ),
        )


class AgentWorkflowV10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = SAMPLE.read_bytes()
        cls.report = build_profile_report(SAMPLE, reference_date=REFERENCE_DATE)
        schemas = ROOT / "schemas"
        cls.workflow_schema = json.loads(
            (schemas / "rule-authoring-workflow.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.history_schema = json.loads(
            (schemas / "rule-authoring-history.schema.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator.check_schema(cls.workflow_schema)
        Draft202012Validator.check_schema(cls.history_schema)
        cls.workflow_validator = Draft202012Validator(cls.workflow_schema)
        registry = Registry().with_resource(
            "rule-authoring-workflow.schema.json",
            Resource.from_contents(cls.workflow_schema),
        )
        cls.history_validator = Draft202012Validator(
            cls.history_schema,
            registry=registry,
        )

    def assert_valid(self, validator, payload):
        errors = sorted(
            validator.iter_errors(payload),
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

    def _begin(self, *, intent="service_name为必填字段"):
        return begin_rule_authoring_run(
            self.report,
            target_metric_id="db31_020100",
            user_intent=intent,
            created_at=CREATED_AT,
        )

    def _compile_and_validate(self, *, intent="service_name为必填字段"):
        run = self._begin(intent=intent)
        run = compile_rule_authoring_run(
            run,
            self.report,
            user_intent=intent,
            created_at=CREATED_AT,
        )
        return validate_rule_authoring_run(run, self.report, at=CREATED_AT)

    def _dry_run(self, *, intent="service_name为必填字段"):
        run = self._compile_and_validate(intent=intent)
        return dry_run_rule_authoring_run(
            run,
            self.report,
            content=self.content,
            file_name=SAMPLE.name,
            reference_date=REFERENCE_DATE,
            selected_metric_ids=run.workflow.selected_metric_ids,
            at=CREATED_AT,
        )

    def _approve(self):
        run = self._dry_run()
        return approve_rule_authoring_run(
            run,
            self.report,
            approver="v1.0-test",
            approved_at=CREATED_AT,
        )

    def test_natural_language_to_deterministic_result_is_stable_and_schema_valid(self):
        run = self._approve()
        approval_id = run.workflow.approval_id
        approved_pack = run.approved_pack
        self.assertEqual(run.workflow.state, "approved")
        self.assertIsNotNone(approved_pack)
        self.assertIs(approve_rule_authoring_run(
            run,
            self.report,
            approver="ignored-on-repeat",
        ), run)

        run = execute_rule_authoring_run(
            run,
            content=self.content,
            file_name=SAMPLE.name,
            reference_date=REFERENCE_DATE,
            selected_metric_ids=run.workflow.selected_metric_ids,
            at=CREATED_AT,
        )

        self.assertEqual(run.workflow.state, "executed")
        self.assertEqual(run.workflow.approval_id, approval_id)
        self.assertIs(run.approved_pack, approved_pack)
        self.assertIsNotNone(run.result)
        self.assertIs(execute_rule_authoring_run(
            run,
            content=self.content,
            file_name=SAMPLE.name,
        ), run)
        self.assertEqual(validate_rule_authoring_workflow(run.workflow), ())
        self.assert_valid(self.workflow_validator, run.workflow.to_dict())

        rule_id = run.workflow.draft.rule_spec.rule_id
        self.assertEqual(run.approved_pack.rules[0].rule_id, rule_id)
        added_metrics = {
            metric.metric_key: metric
            for metric in run.result.enhanced_report.metrics
            if metric.metric_key in run.result.diff.added_metric_keys
        }
        self.assertTrue(added_metrics)
        self.assertEqual(
            next(iter(added_metrics.values())).evidence["rule_id"],
            rule_id,
        )
        self.assertEqual(
            run.result.approved_rule_pack.approval.approval_id,
            approval_id,
        )
        evidence_ids = {item.id for item in run.workflow.draft.evidence}
        self.assertEqual(
            evidence_ids,
            {item["id"] for item in run.approved_pack.evidence},
        )

        history = RuleAuthoringHistory().upsert(run.workflow)
        self.assert_valid(self.history_validator, history.to_dict())

    def test_illegal_approval_and_cross_workflow_draft_are_rejected(self):
        run = self._begin()
        with self.assertRaises(RuleAuthoringCoordinatorError):
            approve_rule_authoring_run(
                run,
                self.report,
                approver="too-early",
            )
        with self.assertRaises(RuleAuthoringWorkflowError):
            run.workflow.approve()

        other = self._begin(intent="department为必填字段")
        other = compile_rule_authoring_run(
            other,
            self.report,
            user_intent="department为必填字段",
            created_at=CREATED_AT,
        )
        with self.assertRaises(RuleAuthoringWorkflowError):
            run.workflow.accept_draft(other.workflow.draft)

    def test_deterministic_validation_failure_requires_clarification(self):
        intent = "field_not_in_profile为必填字段"
        run = self._begin(intent=intent)
        run = compile_rule_authoring_run(
            run,
            self.report,
            user_intent=intent,
            provider=_UnknownFieldProvider(),
            allow_template_fallback=False,
            created_at=CREATED_AT,
        )
        run = validate_rule_authoring_run(run, self.report, at=CREATED_AT)

        self.assertEqual(run.workflow.state, "needs_clarification")
        self.assertFalse(run.workflow.validation.valid)
        self.assertFalse(run.workflow.can_retry)
        self.assertIn("不存在的字段", run.workflow.error)
        self.assert_valid(self.workflow_validator, run.workflow.to_dict())

    def test_provider_failure_retries_once_only_for_same_request(self):
        intent = "service_name为必填字段"
        run = self._begin(intent=intent)
        run = compile_rule_authoring_run(
            run,
            self.report,
            user_intent=intent,
            provider=_FailingProvider(),
            allow_template_fallback=False,
            created_at=CREATED_AT,
        )
        self.assertEqual(run.workflow.state, "failed")
        self.assertEqual(run.workflow.recoverable_state, "compiling")
        self.assertTrue(run.workflow.can_retry)
        self.assertNotIn("top-secret", run.workflow.error)

        with self.assertRaises(RuleAuthoringCoordinatorError):
            retry_rule_authoring_run(run, user_intent="service_name可以为空")

        run = retry_rule_authoring_run(
            run,
            user_intent=intent,
            at=CREATED_AT,
        )
        self.assertEqual(run.workflow.state, "compiling")
        self.assertEqual(run.workflow.retry_count, 1)
        run = compile_rule_authoring_run(
            run,
            self.report,
            user_intent=intent,
            provider=_FailingProvider(),
            allow_template_fallback=False,
            created_at=CREATED_AT,
        )
        self.assertEqual(run.workflow.state, "failed")
        self.assertFalse(run.workflow.can_retry)
        with self.assertRaises(RuleAuthoringCoordinatorError):
            retry_rule_authoring_run(run, user_intent=intent)

    def test_execution_failure_recovers_without_new_approval(self):
        run = self._approve()
        approved_pack = run.approved_pack
        approval_id = run.workflow.approval_id
        failed = execute_rule_authoring_run(
            run,
            content=b"not-a-real-dataset",
            file_name="broken.csv",
            reference_date=REFERENCE_DATE,
            selected_metric_ids=run.workflow.selected_metric_ids,
            at=CREATED_AT,
        )
        self.assertEqual(failed.workflow.state, "failed")
        self.assertEqual(failed.workflow.recoverable_state, "approved")
        self.assertIs(failed.approved_pack, approved_pack)
        self.assertEqual(failed.workflow.approval_id, approval_id)

        recovered = retry_rule_authoring_run(
            failed,
            user_intent="service_name为必填字段",
            at=CREATED_AT,
        )
        self.assertEqual(recovered.workflow.state, "approved")
        self.assertEqual(recovered.workflow.approval_id, approval_id)
        self.assertIs(recovered.approved_pack, approved_pack)
        recovered = execute_rule_authoring_run(
            recovered,
            content=self.content,
            file_name=SAMPLE.name,
            reference_date=REFERENCE_DATE,
            selected_metric_ids=recovered.workflow.selected_metric_ids,
            at=CREATED_AT,
        )
        self.assertEqual(recovered.workflow.state, "executed")
        self.assertEqual(recovered.workflow.approval_id, approval_id)

    def test_history_is_bounded_summary_only_and_redacts_failures(self):
        context = self.report.to_dict()["evaluation_context"]
        history = RuleAuthoringHistory()
        first_id = None
        for index in range(21):
            workflow = new_rule_authoring_workflow(
                target_type="custom_rule",
                target_metric_id=None,
                report_sha256=context["report_sha256"],
                input_sha256=context["input_sha256"],
                reference_date=context["reference_date"],
                selected_metric_ids=context["selected_metric_ids"],
                user_intent=f"secret raw intent {index}",
                revision=index + 1,
                created_at=CREATED_AT,
            )
            first_id = first_id or workflow.workflow_id
            history = history.upsert(workflow)
        self.assertEqual(len(history.records), 20)
        self.assertIsNone(history.get(first_id))

        failed = RuleAuthoringWorkflow(
            workflow_id=make_workflow_id("redaction-test"),
            target_metric_id="db31_020100",
            created_at=CREATED_AT,
        ).start_compiling(at=CREATED_AT).fail(
            stage="compiling",
            code="provider_failed",
            message="api_key=top-secret-value Bearer bearer-secret sk-secret12345678",
            at=CREATED_AT,
        )
        history = history.upsert(failed)
        serialized = json.dumps(history.to_dict(), ensure_ascii=False)
        self.assertNotIn("top-secret", serialized)
        self.assertNotIn("bearer-secret", serialized)
        self.assertNotIn("sk-secret", serialized)
        self.assertNotIn("secret raw intent", serialized)
        self.assertIn("[REDACTED]", serialized)
        self.assert_valid(self.history_validator, history.to_dict())

    def test_schema_rejects_unknown_workflow_fields(self):
        payload = self._begin().workflow.to_dict()
        payload["raw_uploaded_bytes"] = "forbidden"
        self.assertTrue(list(self.workflow_validator.iter_errors(payload)))

    def test_streamlit_exposes_state_history_and_single_execution(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        next(
            item for item in app.file_uploader if item.label == "选择数据文件"
        ).set_value((SAMPLE.name, self.content, "text/csv")).run()
        app.date_input[0].set_value(REFERENCE_DATE)
        next(
            button for button in app.button if button.label == "运行质量评估"
        ).click().run()
        next(
            item for item in app.text_input if item.label == "自定义规则描述"
        ).set_value("service_name为必填字段").run()
        next(
            button
            for button in app.button
            if button.label == "AI 解析自定义规则"
        ).click().run()

        state = app.session_state["custom_rule_ui_state"]
        self.assertEqual(state["run"].workflow.state, "validated")
        history = app.session_state["rule_authoring_workflow_history"]
        self.assertEqual(history.records[-1].state, "validated")
        self.assertIn(
            "查看工作流状态与恢复记录",
            [item.label for item in app.expander],
        )

        next(
            button
            for button in app.button
            if button.label == "试运行自定义规则"
        ).click().run()
        state = app.session_state["custom_rule_ui_state"]
        self.assertEqual(state["run"].workflow.state, "awaiting_approval")

        next(
            item
            for item in app.text_input
            if item.label == "审批人标识（自定义规则，本地自声明）"
        ).set_value("v1.0-ui-test")
        next(
            item
            for item in app.checkbox
            if item.label
            == "我已核对当前自定义规则和试运行摘要，并批准本次确定性重评。"
        ).check()
        app.run()
        next(
            button
            for button in app.button
            if button.label == "批准并重新评估（自定义规则）"
        ).click().run()

        state = app.session_state["custom_rule_ui_state"]
        self.assertEqual(state["run"].workflow.state, "executed")
        self.assertIsNotNone(state["result"])
        app.run()
        self.assertFalse(
            any(
                button.label == "批准并重新评估（自定义规则）"
                for button in app.button
            )
        )
        history = app.session_state["rule_authoring_workflow_history"]
        self.assertEqual(history.records[-1].state, "executed")
        self.assertEqual(len(history.records), 1)


if __name__ == "__main__":
    unittest.main()
