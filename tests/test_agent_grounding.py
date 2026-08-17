"""Agent 报告绑定、引用与数字落地测试。"""

from __future__ import annotations

from copy import deepcopy
import json
import unittest

from src.agent_providers import ProviderResult, TemplateAgentProvider
from src.models import DatasetInfo, MetricResult, QualityReport
from src.rules import generate_risks
from src.agent_service import (
    clear_agent_cache,
    export_action_plan_markdown,
    run_agent,
    validate_agent_analysis,
)
from src.agent_tools import ReportSnapshot, numbers_are_supported


class StubReport:
    def __init__(self, payload: dict) -> None:
        self.payload = deepcopy(payload)

    def to_dict(self) -> dict:
        return deepcopy(self.payload)


def build_report_payload(report_hash: str = "a" * 64) -> dict:
    return {
        "dataset": {
            "name": "高度敏感的数据集名称",
            "file_name": "高度敏感的原始文件.csv",
            "file_type": "csv",
            "sheet_name": None,
        },
        "status": "success",
        "schema_version": "0.2",
        "profile": {
            "row_count": 20,
            "column_count": 2,
            "columns": [{"name": "不得作为原始画像外发"}],
        },
        "metrics": [
            {
                "id": "field_missing_rate",
                "metric_key": "metric:field_missing_rate:field:name",
                "name": "字段缺失率",
                "category": "完整性",
                "status": "evaluated",
                "value": 0.25,
                "unit": "ratio",
                "scope": "field",
                "field": "name",
                "evidence": {
                    "checked_count": 20,
                    "issue_count": 5,
                    "invalid_samples": ["原始值绝不能外发"],
                    "missing_row_indices": [1, 3, 5, 7, 9],
                },
                "reason": None,
            },
            {
                "id": "update_lag_days",
                "metric_key": "metric:update_lag_days:dataset",
                "name": "更新滞后天数",
                "category": "及时性",
                "status": "not_assessable",
                "value": None,
                "unit": None,
                "scope": "dataset",
                "field": None,
                "evidence": {},
                "reason": "未识别到时间字段，无法计算更新滞后天数。",
            },
        ],
        "risks": [
            {
                "id": "high_missing:name",
                "level": "warning",
                "title": "字段缺失较多",
                "message": "字段“name”的缺失率为 25.0%，建议复核。",
                "related_metrics": ["field_missing_rate"],
                "related_metric_keys": [
                    "metric:field_missing_rate:field:name"
                ],
                "evidence": {
                    "field": "name",
                    "decision": {
                        "observed_value": 0.25,
                        "threshold": 0.1,
                        "operator": ">",
                        "rule_version": "0.3",
                    },
                    "invalid_samples": ["原始值绝不能外发"],
                    "missing_row_indices": [1, 3, 5, 7, 9],
                },
            }
        ],
        "not_assessable": [
            {
                "id": "update_lag_days",
                "metric_key": "metric:update_lag_days:dataset",
                "name": "更新滞后天数",
                "reason": "未识别到时间字段，无法计算更新滞后天数。",
            }
        ],
        "evaluation_context": {
            "engine_version": "0.3",
            "reference_date": "2026-07-28",
            "threshold_config_version": "0.3",
            "parser_path": "csv",
            "report_sha256": report_hash,
        },
        "execution": {
            "warnings": [],
            "errors": ["高度敏感的原始文件.csv 在 /private/path 解析失败"],
        },
    }


def valid_model_draft() -> dict:
    return {
        "facts": [
            {
                "text": "字段缺失率为 25%，应优先复核。",
                "citation_ids": ["risk:high_missing:name"],
            }
        ],
        "actions": [
            {
                "priority": "high",
                "title": "复核字段缺失",
                "detail": "结合聚合证据与业务规则确认影响范围。",
                "citation_ids": ["risk:high_missing:name"],
            }
        ],
        "limitations": [
            {
                "text": "更新滞后天数当前无法评估。",
                "citation_ids": [
                    "not_assessable:metric:update_lag_days:dataset"
                ],
            }
        ],
        "answer": {
            "text": "当前最需要先核对字段缺失风险。",
            "citation_ids": ["risk:high_missing:name"],
        },
    }


class ValidProvider:
    cache_namespace = "valid-provider-v1"
    name = "test-model"
    model = "test-model-1"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, snapshot, *, intent, question):
        del snapshot, intent, question
        self.calls += 1
        return ProviderResult(
            payload=valid_model_draft(),
            provider=self.name,
            model=self.model,
            mode="model",
            prompt_version="test-prompt-v1",
            tool_calls=("get_report_summary", "list_priority_risks"),
            input_tokens=120,
            output_tokens=40,
            latency_ms=12,
            available_citation_ids=(
                "risk:high_missing:name",
                "not_assessable:metric:update_lag_days:dataset",
            ),
        )


class AgentGroundingTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_agent_cache()

    def test_template_analysis_is_report_bound_and_schema_valid(self):
        report = StubReport(build_report_payload())
        before = report.to_dict()

        analysis = run_agent(report, use_cache=False)
        payload = validate_agent_analysis(analysis)

        self.assertEqual("a" * 64, analysis.report_sha256)
        self.assertEqual("template", analysis.audit.mode)
        self.assertFalse(analysis.audit.fallback_used)
        self.assertEqual(before, report.to_dict())
        citation_ids = {item.id for item in analysis.citations}
        for fact in analysis.facts:
            self.assertTrue(fact.citation_ids)
            self.assertTrue(set(fact.citation_ids) <= citation_ids)
        for action in analysis.actions:
            self.assertTrue(action.citation_ids)
            self.assertTrue(set(action.citation_ids) <= citation_ids)
        self.assertEqual("0.1", payload["schema_version"])
        self.assertEqual(0, payload["audit"]["input_tokens"])
        self.assertEqual(0, payload["audit"]["output_tokens"])

    def test_numeric_field_identifier_does_not_break_template_or_fallback(self):
        payload = build_report_payload()
        metric = payload["metrics"][0]
        metric["metric_key"] = "metric:field_missing_rate:field:2024"
        metric["field"] = "2024"
        risk = payload["risks"][0]
        risk["id"] = "high_missing:2024"
        risk["message"] = "字段“2024”的缺失率为 25.0%，建议复核。"
        risk["related_metric_keys"] = [metric["metric_key"]]
        report = StubReport(payload)

        template_analysis = run_agent(report, use_cache=False)
        self.assertFalse(template_analysis.audit.fallback_used)
        self.assertTrue(
            any("2024" in fact.text for fact in template_analysis.facts)
        )

        class FailingProvider:
            cache_namespace = "numeric-field-failing-provider"

            def generate(self, snapshot, *, intent, question):
                del snapshot, intent, question
                raise RuntimeError("模拟外部提供方故障")

        fallback_analysis = run_agent(
            report,
            provider=FailingProvider(),
            use_cache=False,
        )
        self.assertTrue(fallback_analysis.audit.fallback_used)
        self.assertEqual("template", fallback_analysis.audit.mode)

    def test_template_provider_subclass_cannot_bypass_number_validation(self):
        report = StubReport(build_report_payload())
        draft = valid_model_draft()
        draft["facts"] = [
            {
                "text": "共有 20 项风险。",
                "citation_ids": ["report:summary"],
            }
        ]

        class UntrustedTemplateSubclass(TemplateAgentProvider):
            cache_namespace = "untrusted-template-subclass"

            def generate(self, snapshot, *, intent, question):
                del snapshot, intent, question
                return ProviderResult(
                    payload=draft,
                    provider="untrusted-template-subclass",
                    model=None,
                    mode="template",
                    prompt_version="test-prompt",
                    tool_calls=("get_report_summary",),
                    available_citation_ids=(
                        "report:summary",
                        "risk:high_missing:name",
                        "not_assessable:metric:update_lag_days:dataset",
                    ),
                )

        analysis = run_agent(
            report,
            provider=UntrustedTemplateSubclass(),
            use_cache=False,
        )

        self.assertTrue(analysis.audit.fallback_used)
        self.assertEqual(
            "invalid_model_output",
            analysis.audit.fallback_reason,
        )

    def test_valid_model_numbers_are_grounded_by_cited_risk(self):
        report = StubReport(build_report_payload())
        provider = ValidProvider()

        analysis = run_agent(
            report,
            intent="priority",
            provider=provider,
            use_cache=False,
        )

        self.assertFalse(analysis.audit.fallback_used)
        self.assertEqual("model", analysis.audit.mode)
        self.assertEqual(120, analysis.audit.input_tokens)
        self.assertEqual(40, analysis.audit.output_tokens)
        snapshot = ReportSnapshot.from_report(report)
        fact = analysis.facts[0]
        self.assertTrue(
            numbers_are_supported(
                fact.text,
                snapshot.numeric_values_for(fact.citation_ids),
            )
        )

    def test_model_cannot_cite_evidence_not_returned_by_its_tools(self):
        report = StubReport(build_report_payload())
        provider = ValidProvider()
        original_generate = provider.generate

        def generate_with_summary_only(snapshot, *, intent, question):
            result = original_generate(
                snapshot,
                intent=intent,
                question=question,
            )
            return ProviderResult(
                **{
                    **result.__dict__,
                    "available_citation_ids": ("report:summary",),
                }
            )

        provider.generate = generate_with_summary_only
        analysis = run_agent(
            report,
            intent="priority",
            provider=provider,
            use_cache=False,
        )

        self.assertTrue(analysis.audit.fallback_used)
        self.assertEqual(
            "invalid_model_output",
            analysis.audit.fallback_reason,
        )

    def test_dates_and_versions_do_not_authorize_unrelated_numbers(self):
        snapshot = ReportSnapshot.from_report(
            StubReport(build_report_payload())
        )
        summary_values = snapshot.numeric_values_for(["report:summary"])
        risk_values = snapshot.numeric_values_for(
            ["risk:high_missing:name"]
        )

        self.assertFalse(
            numbers_are_supported("共有 2026 项风险。", summary_values)
        )
        self.assertFalse(
            numbers_are_supported("风险率为 30%。", risk_values)
        )
        self.assertTrue(
            numbers_are_supported("字段缺失率为 25%。", risk_values)
        )
        self.assertFalse(
            snapshot.statement_numbers_are_supported(
                "共有 20 项风险。",
                ["report:summary"],
            )
        )
        self.assertTrue(
            snapshot.statement_numbers_are_supported(
                "共有 20 条记录数。",
                ["report:summary"],
            )
        )

    def test_summary_values_cannot_be_relabelled_as_other_counts(self):
        report = StubReport(build_report_payload())
        draft = valid_model_draft()
        draft["facts"] = [
            {
                "text": "共有 20 项风险。",
                "citation_ids": ["report:summary"],
            }
        ]
        provider = ValidProvider()

        def generate_collision(snapshot, *, intent, question):
            del snapshot, intent, question
            return ProviderResult(
                payload=draft,
                provider=provider.name,
                model=provider.model,
                mode="model",
                prompt_version="test-prompt-v1",
                tool_calls=("get_report_summary",),
                available_citation_ids=(
                    "report:summary",
                    "risk:high_missing:name",
                    "not_assessable:metric:update_lag_days:dataset",
                ),
            )

        provider.generate = generate_collision
        analysis = run_agent(
            report,
            provider=provider,
            use_cache=False,
        )

        self.assertTrue(analysis.audit.fallback_used)
        self.assertEqual(
            "invalid_model_output",
            analysis.audit.fallback_reason,
        )

    def test_future_update_date_template_uses_explicit_absolute_evidence(self):
        metric = MetricResult(
            id="update_lag_days",
            name="更新滞后天数",
            category="及时性",
            status="evaluated",
            value=-1,
            unit="days",
            scope="dataset",
        )
        risks = generate_risks([metric])
        report = QualityReport(
            dataset=DatasetInfo(
                name="future-date",
                file_name="future-date.csv",
                file_type="csv",
            ),
            profile={"row_count": 1, "column_count": 1, "columns": []},
            metrics=[metric],
            risks=risks,
        )

        self.assertEqual(1, risks[0].evidence["absolute_lag_days"])
        analysis = run_agent(report, use_cache=False)

        self.assertFalse(analysis.audit.fallback_used)
        self.assertTrue(
            any("晚 1 天" in fact.text for fact in analysis.facts)
        )

    def test_snapshot_tools_are_deep_copied_read_only_and_distinguish_counts(self):
        snapshot = ReportSnapshot.from_report(
            StubReport(build_report_payload())
        )
        summary = snapshot.get_report_summary()
        summary["row_count"] = 999

        self.assertEqual(20, snapshot.get_report_summary()["row_count"])
        self.assertEqual(
            2, snapshot.get_report_summary()["metric_definition_count"]
        )
        self.assertEqual(2, snapshot.get_report_summary()["metric_result_count"])
        with self.assertRaises(AttributeError):
            snapshot.report_sha256 = "b" * 64

    def test_action_plan_export_contains_only_analysis_evidence(self):
        report = StubReport(build_report_payload())
        analysis = run_agent(report, use_cache=False)

        markdown = export_action_plan_markdown(analysis)

        self.assertIn("# 数据质量改进行动计划", markdown)
        self.assertIn("复核", markdown)
        self.assertNotIn("高度敏感的原始文件.csv", markdown)
        self.assertNotIn("原始值绝不能外发", markdown)
        self.assertNotIn("/private/path", markdown)

    def test_cache_preserves_usage_and_only_marks_cache_hit(self):
        report = StubReport(build_report_payload())
        provider = ValidProvider()

        first = run_agent(report, provider=provider)
        second = run_agent(report, provider=provider)

        self.assertEqual(1, provider.calls)
        self.assertFalse(first.audit.cache_hit)
        self.assertTrue(second.audit.cache_hit)
        self.assertEqual(first.audit.input_tokens, second.audit.input_tokens)
        self.assertEqual(first.audit.output_tokens, second.audit.output_tokens)
        self.assertEqual(first.audit.latency_ms, second.audit.latency_ms)
        self.assertEqual(
            json.dumps(first.to_dict(), sort_keys=True).replace(
                '"cache_hit": false', '"cache_hit": true'
            ),
            json.dumps(second.to_dict(), sort_keys=True),
        )


if __name__ == "__main__":
    unittest.main()
