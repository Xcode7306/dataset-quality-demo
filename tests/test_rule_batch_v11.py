"""v1.1 pre-evaluation rule generation, clarification, and file batches."""

from datetime import date
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from src.rule_authoring_providers import RuleAuthoringProviderResult
from src.rule_batch import (
    MAX_RULE_IMPORT_BYTES,
    RuleBatchInput,
    RuleImportError,
    compile_rule_batch,
    parse_rule_import,
)
from src.rule_dsl import ProviderMetadata, RuleSpec
from src.rule_pack import approve_rule_pack
from src.rule_service import (
    dry_run_uploaded_dataset_with_rule_pack,
    evaluate_uploaded_dataset_with_rule_pack,
)
from src.upload_service import evaluate_uploaded_dataset


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data" / "good_dataset.csv"
REFERENCE_DATE = date(2026, 7, 17)


class _GuessingProvider:
    def generate(self, context, *, user_intent):
        return RuleAuthoringProviderResult(
            outcome="draft",
            rule_spec=RuleSpec(
                rule_type="required",
                rule_id="model-candidate",
                name="模型猜测的必填规则",
                description="模型在模糊描述下猜测字段。",
                fields=("service_name",),
            ),
            metadata=ProviderMetadata(
                provider="test-model",
                model="guessing-model",
                mode="model",
                prompt_version="test",
            ),
        )


class RuleBatchV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = SAMPLE.read_bytes()
        cls.report = evaluate_uploaded_dataset(
            cls.content,
            SAMPLE.name,
            reference_date=REFERENCE_DATE,
        )

    def test_rule_files_parse_text_csv_json_and_reject_unsafe_size(self):
        markdown = parse_rule_import(
            (
                "# 规则清单\n"
                "- service_name为必填字段\n"
                "2. handling_days数值范围为0至30\n"
            ).encode("utf-8"),
            "rules.md",
        )
        self.assertEqual(len(markdown.items), 2)
        self.assertEqual(markdown.items[0].source_location, "第 2 行")

        csv_result = parse_rule_import(
            (
                "规则描述,指标ID\n"
                "service_name为必填字段,\n"
                "department只能为人力资源和社会保障局、民政局,db31_030200\n"
            ).encode("utf-8"),
            "rules.csv",
        )
        self.assertEqual(csv_result.items[1].target_metric_id, "db31_030200")

        json_result = parse_rule_import(
            (
                '["service_name为必填字段",'
                '{"description":"handling_days数值范围为0至30"}]'
            ).encode("utf-8"),
            "rules.json",
        )
        self.assertEqual(len(json_result.items), 2)

        with self.assertRaises(RuleImportError):
            parse_rule_import(b"x" * (MAX_RULE_IMPORT_BYTES + 1), "rules.txt")

    def test_batch_requires_every_item_before_building_one_rule_pack(self):
        requests = (
            RuleBatchInput.create(
                origin="dialog",
                user_intent="service_name为必填字段",
                label="规则 1",
            ),
            RuleBatchInput.create(
                origin="dialog",
                user_intent="handling_days数值范围为0至30",
                label="规则 2",
            ),
        )
        preflight = compile_rule_batch(
            self.report,
            requests,
            created_at="2026-08-10T08:00:00Z",
        )
        self.assertTrue(preflight.ready)
        self.assertEqual(len(preflight.draft_pack.rules), 2)

        preview = dry_run_uploaded_dataset_with_rule_pack(
            self.content,
            SAMPLE.name,
            preflight.draft_pack,
            reference_date=REFERENCE_DATE,
        )
        self.assertEqual(len(preview.metrics), 2)
        approved = approve_rule_pack(
            preflight.draft_pack,
            self.report,
            approver="v1.1-test",
            approved_at="2026-08-10T08:01:00Z",
        )
        result = evaluate_uploaded_dataset_with_rule_pack(
            self.content,
            SAMPLE.name,
            approved,
            reference_date=REFERENCE_DATE,
        )
        self.assertEqual(len(result.diff.added_metric_keys), 2)

        incomplete = compile_rule_batch(
            self.report,
            (
                requests[0],
                RuleBatchInput.create(
                    origin="dialog",
                    user_intent="handling_days要合理",
                    label="缺少阈值的规则",
                ),
            ),
        )
        self.assertFalse(incomplete.ready)
        self.assertIsNone(incomplete.draft_pack)
        self.assertEqual(incomplete.items[1].status, "needs_clarification")
        self.assertTrue(incomplete.items[1].messages)

    def test_model_cannot_guess_critical_inputs_from_a_vague_description(self):
        preflight = compile_rule_batch(
            self.report,
            (
                RuleBatchInput.create(
                    origin="dialog",
                    user_intent="数据质量要好",
                    label="模糊规则",
                ),
            ),
            provider=_GuessingProvider(),
            allow_template_fallback=False,
        )
        self.assertFalse(preflight.ready)
        self.assertEqual(preflight.items[0].status, "needs_clarification")
        self.assertIsNone(preflight.items[0].draft.rule_spec)
        self.assertTrue(
            any("规则类型" in message for message in preflight.items[0].messages)
        )

    def test_streamlit_clarifies_before_report_then_executes_approved_rule(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        next(
            item for item in app.file_uploader if item.label == "选择数据文件"
        ).set_value((SAMPLE.name, self.content, "text/csv")).run()
        app.date_input[0].set_value(REFERENCE_DATE)
        next(
            item
            for item in app.text_input
            if item.label == "评估前自定义规则描述（可选）"
        ).set_value("service_name需要规范").run()
        next(
            button for button in app.button if button.label == "AI 检查并生成规则"
        ).click().run()

        self.assertNotIn("quality_report", app.session_state.filtered_state)
        preflight = app.session_state["pre_evaluation_rule_state"]["preflight"]
        self.assertEqual(preflight.items[0].status, "needs_clarification")
        self.assertTrue(
            any("最终评估尚未启动" in warning.value for warning in app.warning)
        )

        next(
            item
            for item in app.text_input
            if item.label == "评估前自定义规则描述（可选）"
        ).set_value("service_name为必填字段").run()
        next(
            button for button in app.button if button.label == "AI 检查并生成规则"
        ).click().run()
        preflight_state = app.session_state["pre_evaluation_rule_state"]
        self.assertTrue(preflight_state["preflight"].ready)
        self.assertIsNotNone(preflight_state["preview"])

        next(
            item
            for item in app.text_input
            if item.label == "审批人标识（评估前 AI 规则，本地自声明）"
        ).set_value("v1.1-ui-test")
        next(
            item
            for item in app.checkbox
            if item.label == "我已核对全部生成规则和试运行摘要，并批准将其用于本次评估。"
        ).check()
        app.run()
        next(
            button for button in app.button if button.label == "批准规则并运行质量评估"
        ).click().run()

        self.assertFalse(app.exception)
        self.assertIn("quality_report", app.session_state.filtered_state)
        report = app.session_state["quality_report"]
        self.assertTrue(
            any(metric.id == "business_required_compliance" for metric in report.metrics)
        )
        self.assertIsNotNone(app.session_state["pre_evaluation_rule_result"])

    def test_streamlit_rule_file_generates_one_batch_before_evaluation(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        next(
            item for item in app.file_uploader if item.label == "选择数据文件"
        ).set_value((SAMPLE.name, self.content, "text/csv"))
        next(
            item
            for item in app.file_uploader
            if item.label == "导入规则文件（批量，可选）"
        ).set_value(
            (
                "rules.csv",
                (
                    "规则描述\n"
                    "service_name为必填字段\n"
                    "handling_days数值范围为0至30\n"
                ).encode("utf-8"),
                "text/csv",
            )
        )
        app.run()
        app.date_input[0].set_value(REFERENCE_DATE)
        next(
            button for button in app.button if button.label == "AI 检查并生成规则"
        ).click().run()

        self.assertFalse(app.exception)
        state = app.session_state["pre_evaluation_rule_state"]
        self.assertTrue(state["preflight"].ready)
        self.assertEqual(len(state["preflight"].items), 2)
        self.assertEqual(len(state["preflight"].draft_pack.rules), 2)
        self.assertNotIn("quality_report", app.session_state.filtered_state)


if __name__ == "__main__":
    unittest.main()
