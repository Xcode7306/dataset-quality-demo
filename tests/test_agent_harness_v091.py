"""v0.9.1 Agent Harness goldens, traces, tool boundaries, and replay tests."""

from __future__ import annotations

from contextlib import redirect_stdout
from datetime import date
import io
import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator

from run_agent_harness import main as harness_main
from src.rule_authoring_harness import (
    compare_rule_authoring_providers,
    run_rag_retrieval_harness,
    run_rule_authoring_harness,
)
from src.rule_authoring_prompts import (
    DEFAULT_RULE_AUTHORING_PROMPT_VERSION,
    RULE_AUTHORING_PROMPT_V090,
    RULE_AUTHORING_PROMPT_V091,
    get_rule_authoring_prompt,
    list_rule_authoring_prompts,
)
from src.rule_authoring_service import compile_custom_rule_draft
from src.rule_authoring_providers import (
    DeepSeekRuleAuthoringProvider,
    RuleAuthoringProviderError,
    RuleAuthoringProviderResult,
    TemplateRuleAuthoringProvider,
    parse_provider_payload,
)
from src.rule_authoring_tools import (
    RULE_AUTHORING_TOOL_NAMES,
    RuleAuthoringToolRequestError,
    validate_rule_authoring_tool_request,
)
from src.rule_authoring_trace import MAX_TRACE_TOOL_CALLS, RuleAuthoringTraceBuilder
from src.rule_dsl import ProviderMetadata, RuleSpec, make_workflow_id, new_evidence
from src.upload_service import evaluate_uploaded_dataset
from src.workflow import build_profile_report


ROOT = Path(__file__).resolve().parents[1]


class AgentHarnessV091Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace_schema = json.loads(
            (ROOT / "schemas/rule-authoring-trace.schema.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator.check_schema(cls.trace_schema)
        cls.trace_validator = Draft202012Validator(cls.trace_schema)

    def test_versioned_prompt_registry_has_stable_fingerprints(self):
        self.assertEqual(
            DEFAULT_RULE_AUTHORING_PROMPT_VERSION,
            RULE_AUTHORING_PROMPT_V091,
        )
        old = get_rule_authoring_prompt(RULE_AUTHORING_PROMPT_V090)
        current = get_rule_authoring_prompt(RULE_AUTHORING_PROMPT_V091)
        self.assertNotEqual(old.sha256, current.sha256)
        self.assertEqual(current.sha256, get_rule_authoring_prompt().sha256)
        self.assertEqual(
            {item["version"] for item in list_rule_authoring_prompts()},
            {RULE_AUTHORING_PROMPT_V090, RULE_AUTHORING_PROMPT_V091},
        )
        legacy_result = TemplateRuleAuthoringProvider(
            prompt_version=RULE_AUTHORING_PROMPT_V090
        ).generate(
            {"fields": [{"name": "service_name", "inferred_type": "text"}]},
            user_intent="service_name为必填字段",
        )
        self.assertEqual(
            legacy_result.metadata.prompt_version,
            RULE_AUTHORING_PROMPT_V090,
        )

    def test_template_rule_goldens_reach_all_v1_thresholds(self):
        report = run_rule_authoring_harness(ROOT, replay=True)

        self.assertTrue(report.passed)
        self.assertEqual(report.total_cases, 29)
        self.assertEqual(report.passed_cases, report.total_cases)
        self.assertEqual(report.schema_valid_rate, 1.0)
        self.assertEqual(report.support_scope_accuracy, 1.0)
        self.assertEqual(report.field_mapping_accuracy, 1.0)
        self.assertEqual(report.parameter_accuracy, 1.0)
        self.assertEqual(report.deterministic_execution_rate, 1.0)
        self.assertEqual(report.replay_consistency_rate, 1.0)
        self.assertEqual(report.ungrounded_standard_claim_count, 0)
        self.assertEqual(report.unapproved_execution_count, 0)
        self.assertTrue(report.replay_executed)
        self.assertEqual(
            report.ungrounded_standard_claim_count,
            sum(item.ungrounded_standard_claim_count for item in report.cases),
        )
        self.assertEqual(
            report.unapproved_execution_count,
            sum(item.unapproved_execution_count for item in report.cases),
        )
        for case in report.cases:
            errors = list(self.trace_validator.iter_errors(case.trace.to_dict()))
            self.assertEqual([], errors, f"{case.case_id}: {errors}")

    def test_rag_goldens_cover_success_empty_conflict_and_stale_sources(self):
        report = run_rag_retrieval_harness(ROOT)

        self.assertTrue(report.passed)
        self.assertEqual(report.total_cases, 6)
        self.assertEqual(report.status_accuracy, 1.0)
        self.assertEqual(report.citation_validity_rate, 1.0)
        statuses = {case.case_id: case.actual_status for case in report.cases}
        self.assertEqual(statuses["rag-approved-v1"], "ok")
        self.assertEqual(statuses["rag-version-conflict"], "conflict")
        self.assertEqual(statuses["rag-expired-source"], "no_results")

    def test_provider_comparison_reports_a_regressing_provider(self):
        class AlwaysClarifyProvider:
            def generate(self, context, *, user_intent):
                del context, user_intent
                return RuleAuthoringProviderResult(
                    outcome="clarification",
                    rule_spec=None,
                    clarification_questions=("请补充规则。",),
                    metadata=ProviderMetadata(
                        provider="regression-test",
                        model="fake",
                        mode="model",
                        prompt_version=RULE_AUTHORING_PROMPT_V091,
                    ),
                )

        class WrongFieldProvider:
            def generate(self, context, *, user_intent):
                del context, user_intent
                return RuleAuthoringProviderResult(
                    outcome="draft",
                    rule_spec=RuleSpec(
                        rule_type="required",
                        rule_id="provider-candidate",
                        name="不存在字段必填",
                        description="用于验证确定性字段拦截。",
                        fields=("does_not_exist",),
                    ),
                    metadata=ProviderMetadata(
                        provider="wrong-field-test",
                        model="fake",
                        mode="model",
                        prompt_version=RULE_AUTHORING_PROMPT_V091,
                    ),
                )

        baseline, regression, wrong_field = compare_rule_authoring_providers(
            ROOT,
            {
                "template": TemplateRuleAuthoringProvider(),
                "regression": AlwaysClarifyProvider(),
                "wrong-field": WrongFieldProvider(),
            },
            replay=False,
        )

        self.assertTrue(baseline.passed)
        self.assertFalse(regression.passed)
        self.assertLess(regression.support_scope_accuracy, 1.0)
        self.assertEqual(regression.unapproved_execution_count, 0)
        self.assertFalse(baseline.replay_executed)
        self.assertIsNone(baseline.replay_consistency_rate)
        self.assertTrue(
            all(item.replay_consistent is None for item in baseline.cases)
        )
        self.assertFalse(wrong_field.passed)
        self.assertTrue(
            any(
                item.deterministic_validation_passed is False
                and item.trace.execution_result_id is None
                and item.trace.outcome == "failed"
                for item in wrong_field.cases
            )
        )

    def test_model_payload_rejects_abnormal_json_and_privilege_fields(self):
        invalid_payloads = (
            '{"outcome":"clarification","outcome":"draft","rule_spec":null}',
            '{"outcome":"draft","rule_spec":{"rule_type":"numeric_range","fields":["x"],"parameters":{"minimum":NaN}}}',
            '{"outcome":"draft","rule_spec":{"rule_type":"numeric_range","fields":["x"],"parameters":{"minimum":Infinity}}}',
            '{"outcome":"draft","rule_spec":{"rule_type":"numeric_range","fields":["x"],"parameters":{"minimum":-Infinity}}}',
            '{"outcome":"clarification","rule_spec":null,"approval":{"approved":true}}',
            '{"clarification_questions":["x"],"tool_calls":[{"name":"execute_rule_pack"}]}',
            '{"rule_type":"required","fields":["x"],"execution_result":"done"}',
            '{"outcome":"draft","rule_spec":{"rule_type":"required","fields":["x"],"approved":true}}',
            '{"outcome":"draft","rule_spec":{"rule_type":"required","fields":["x","x"]}}',
            '```json\n{"outcome":"clarification","rule_spec":null}\n```',
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload[:80]):
                with self.assertRaises(RuleAuthoringProviderError):
                    parse_provider_payload(payload)
        with self.assertRaises(RuleAuthoringProviderError):
            parse_provider_payload(
                {
                    "outcome": "clarification",
                    "rule_spec": None,
                    "clarification_questions": ["\ud800"],
                }
            )
        with self.assertRaises(RuleAuthoringProviderError):
            parse_provider_payload(" " * (64 * 1024 + 1))
        with self.assertRaises(RuleAuthoringProviderError):
            parse_provider_payload(
                {
                    "outcome": "clarification",
                    "rule_spec": None,
                    "assumptions": ["x" * 2001],
                    "clarification_questions": ["请补充字段。"],
                }
            )

    def test_rule_model_timeout_and_rate_limit_are_explicit_failures(self):
        class FakeResponse:
            status_code = 429

            @staticmethod
            def json():
                return {"error": {"message": "rate limited"}}

        for mode in ("timeout", "rate_limit"):
            with self.subTest(mode=mode):
                state = {"calls": 0, "closed": False}

                class FakeClient:
                    def post(self, _url, *, headers, json):
                        del headers, json
                        state["calls"] += 1
                        if mode == "timeout":
                            raise TimeoutError("simulated timeout")
                        return FakeResponse()

                    def close(self):
                        state["closed"] = True

                provider = DeepSeekRuleAuthoringProvider(
                    api_key="test-only-key",
                    client_factory=lambda **_options: FakeClient(),
                )
                with self.assertRaises(RuleAuthoringProviderError) as raised:
                    provider.generate(
                        {"fields": []},
                        user_intent="service_name为必填字段",
                    )

                self.assertTrue(state["closed"])
                self.assertGreaterEqual(state["calls"], 1)
                expected = "timeout" if mode == "timeout" else "HTTP 429"
                self.assertIn(expected, str(raised.exception))

    def test_provider_cannot_invent_standard_evidence_without_retrieval(self):
        class ForgedCitationProvider:
            def generate(self, context, *, user_intent):
                del context, user_intent
                return RuleAuthoringProviderResult(
                    outcome="draft",
                    rule_spec=RuleSpec(
                        rule_type="required",
                        rule_id="provider-candidate",
                        name="service_name必填",
                        description="模型声称来自标准。",
                        fields=("service_name",),
                    ),
                    evidence=(
                        new_evidence(
                            "standard_clause",
                            "伪造标准条款",
                            source_id="chunk-00000000000000000000",
                            document_id="doc-00000000000000000000",
                            document_name="不存在的标准",
                            document_version="v1",
                            section="1",
                            chunk_id="chunk-00000000000000000000",
                            authoritative=True,
                        ),
                    ),
                    metadata=ProviderMetadata(
                        provider="forged",
                        model="fake",
                        mode="model",
                        prompt_version=RULE_AUTHORING_PROMPT_V091,
                    ),
                )

        report = build_profile_report(
            ROOT / "harness/data/agent_harness.csv",
            reference_date=date(2026, 8, 1),
        )
        draft = compile_custom_rule_draft(
            report,
            user_intent="service_name为必填字段",
            provider=ForgedCitationProvider(),
            allow_template_fallback=False,
            created_at="2026-08-01T00:00:00Z",
        )

        self.assertFalse(
            any(
                item.type in {"standard_clause", "data_dictionary"}
                for item in draft.evidence
            )
        )

    def test_tool_allowlist_rejects_approval_execution_and_bad_arguments(self):
        self.assertEqual(
            RULE_AUTHORING_TOOL_NAMES,
            {
                "get_metric_definition",
                "get_profile_summary",
                "list_available_fields",
                "retrieve_rule_evidence",
                "validate_rule_draft",
                "dry_run_rule",
            },
        )
        validated = validate_rule_authoring_tool_request(
            "retrieve_rule_evidence",
            {"query": "必填字段", "metric_id": "db31_020100", "limit": 5},
        )
        self.assertEqual(validated["limit"], 5)
        for name in ("approve_rule_pack", "execute_rule_pack", "python"):
            with self.subTest(name=name):
                with self.assertRaises(RuleAuthoringToolRequestError):
                    validate_rule_authoring_tool_request(name, {})
        with self.assertRaises(RuleAuthoringToolRequestError):
            validate_rule_authoring_tool_request(
                "get_profile_summary", {"raw_rows": ["secret"]}
            )
        with self.assertRaises(RuleAuthoringToolRequestError):
            validate_rule_authoring_tool_request(
                "retrieve_rule_evidence", {"query": "x", "limit": 0}
            )

    def test_trace_hashes_free_text_and_redacts_credentials(self):
        workflow_id = make_workflow_id("trace-privacy")
        builder = RuleAuthoringTraceBuilder(
            workflow_id=workflow_id,
            target_type="custom_rule",
            target_metric_id=None,
            report_sha256="a" * 64,
            input_sha256="b" * 64,
            reference_date="2026-08-01",
            started_at="2026-08-01T00:00:00Z",
        )
        builder.transition("collecting", "retrieving")
        builder.tool_call(
            "retrieve_rule_evidence",
            {"query": "secret raw cell value", "limit": 5},
            result={"status": "no_results"},
        )
        builder.record_failure(
            stage="provider",
            code="unauthorized",
            message="api_key=super-secret-key Bearer sk-testsecret123456",
        )
        trace = builder.finish(
            "failed", completed_at="2026-08-01T00:00:01Z"
        )
        serialized = json.dumps(trace.to_dict(), ensure_ascii=False)

        self.assertNotIn("secret raw cell value", serialized)
        self.assertNotIn("super-secret-key", serialized)
        self.assertNotIn("sk-testsecret123456", serialized)
        self.assertIn("REDACTED", serialized)
        self.assertTrue(self.trace_validator.is_valid(trace.to_dict()))

        repeated = RuleAuthoringTraceBuilder(
            workflow_id=make_workflow_id("bounded-repeat-tools"),
            target_type="custom_rule",
            target_metric_id=None,
            report_sha256="a" * 64,
            input_sha256="b" * 64,
            reference_date="2026-08-01",
            started_at="2026-08-01T00:00:00Z",
        )
        for _ in range(MAX_TRACE_TOOL_CALLS):
            repeated.tool_call(
                "get_profile_summary",
                {},
                result={"row_count": 3},
            )
        with self.assertRaisesRegex(ValueError, "工具调用数超出上限"):
            repeated.tool_call(
                "get_profile_summary",
                {},
                result={"row_count": 3},
            )

    def test_model_failure_does_not_break_base_evaluation(self):
        class FailingProvider:
            def generate(self, context, *, user_intent):
                del context, user_intent
                raise RuleAuthoringProviderError("simulated timeout")

        report = run_rule_authoring_harness(
            ROOT,
            provider=FailingProvider(),
            provider_label="failure",
            replay=False,
        )
        self.assertFalse(report.passed)
        base = evaluate_uploaded_dataset(
            (ROOT / "harness/data/agent_harness.csv").read_bytes(),
            "agent_harness.csv",
            reference_date=date(2026, 8, 1),
        )
        self.assertEqual(base.status, "success")
        self.assertTrue(base.metrics)

    def test_cli_writes_full_trace_report_and_returns_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "harness-result.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = harness_main(
                    ["--provider", "template", "--no-replay", "--output", str(output)]
                )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertTrue(payload["passed"])
        self.assertTrue(payload["rule_authoring"][0]["cases"][0]["trace"])
        self.assertFalse(payload["rule_authoring"][0]["replay_executed"])
        self.assertIsNone(
            payload["rule_authoring"][0]["metrics"]["replay_consistency_rate"]
        )
        self.assertTrue(json.loads(stdout.getvalue())["passed"])


if __name__ == "__main__":
    unittest.main()
