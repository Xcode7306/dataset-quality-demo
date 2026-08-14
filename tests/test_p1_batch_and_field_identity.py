"""P1 批量规则规模、超时与中文字段身份回归。"""

import csv
import io
import re
import unittest
from unittest.mock import patch

from src.rule_authoring_providers import (
    RuleAuthoringProviderResult,
    TemplateRuleAuthoringProvider,
)
from src.rule_authoring_service import compile_custom_rule_draft, validate_rule_draft
from src.rule_batch import RuleBatchInput, compile_rule_batch
from src.rule_dsl import RuleSpec
from src.upload_service import evaluate_uploaded_dataset


class _BatchRequiredProvider:
    """稳定的本地 Provider，用于测量批量编排而不调用外部模型。"""

    def __init__(self, *, failed_indexes=()):
        self.failed_indexes = set(failed_indexes)
        self.calls: list[int] = []

    def generate(self, context, *, user_intent):
        del context
        field = str(user_intent).split("为必填字段", 1)[0]
        match = re.fullmatch(r"p1_field_(\d{3})", field)
        if match is None:
            raise AssertionError(f"测试输入字段格式错误：{field}")
        index = int(match.group(1))
        self.calls.append(index)
        if index in self.failed_indexes:
            raise RuntimeError(f"provider failure {index + 1}")
        return RuleAuthoringProviderResult(
            outcome="draft",
            rule_spec=RuleSpec(
                rule_type="required",
                rule_id="model-candidate",
                name=f"{field}必填",
                description=f"字段“{field}”不得为空。",
                fields=(field,),
            ),
        )


class RuleBatchScaleP1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fields = tuple(f"p1_field_{index:03d}" for index in range(100))
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(cls.fields)
        writer.writerow(["值"] * len(cls.fields))
        cls.report = evaluate_uploaded_dataset(
            output.getvalue().encode("utf-8"),
            "p1-batch-scale.csv",
        )

    def _requests(self, count):
        return tuple(
            RuleBatchInput.create(
                origin="dialog",
                user_intent=f"{field}为必填字段",
                label=f"第 {index + 1} 条规则",
            )
            for index, field in enumerate(self.fields[:count])
        )

    def test_batch_sizes_up_to_one_hundred_build_one_complete_pack(self):
        for count in (1, 10, 50, 100):
            with self.subTest(count=count):
                provider = _BatchRequiredProvider()
                preflight = compile_rule_batch(
                    self.report,
                    self._requests(count),
                    provider=provider,
                    allow_template_fallback=False,
                    created_at="2026-08-14T00:00:00Z",
                )
                self.assertTrue(preflight.ready)
                self.assertEqual(len(provider.calls), count)
                self.assertEqual(len(preflight.items), count)
                self.assertEqual(len(preflight.draft_pack.rules), count)

    def test_failures_at_first_middle_and_last_item_are_locatable_and_atomic(self):
        requests = self._requests(100)
        for one_based_index in (1, 50, 100):
            with self.subTest(one_based_index=one_based_index):
                provider = _BatchRequiredProvider(
                    failed_indexes=(one_based_index - 1,)
                )
                preflight = compile_rule_batch(
                    self.report,
                    requests,
                    provider=provider,
                    allow_template_fallback=False,
                    created_at="2026-08-14T00:00:00Z",
                )
                self.assertFalse(preflight.ready)
                self.assertIsNone(preflight.draft_pack)
                self.assertEqual(len(provider.calls), 100)
                failed = preflight.items[one_based_index - 1]
                self.assertEqual(failed.status, "failed")
                self.assertEqual(failed.request.label, f"第 {one_based_index} 条规则")
                self.assertIn("provider failure", failed.messages[0])
                self.assertTrue(
                    all(
                        item.status == "ready"
                        for index, item in enumerate(preflight.items)
                        if index != one_based_index - 1
                    )
                )

    def test_batch_time_budget_marks_every_unfinished_item_and_never_builds_a_pack(self):
        provider = _BatchRequiredProvider()
        with patch(
            "src.rule_batch.monotonic",
            side_effect=(0.0, 0.0, 60.0),
        ):
            preflight = compile_rule_batch(
                self.report,
                self._requests(10),
                provider=provider,
                allow_template_fallback=False,
                created_at="2026-08-14T00:00:00Z",
            )

        self.assertFalse(preflight.ready)
        self.assertIsNone(preflight.draft_pack)
        self.assertEqual(provider.calls, [0])
        self.assertEqual(len(preflight.items), 10)
        self.assertTrue(all(item.status == "failed" for item in preflight.items))
        self.assertTrue(preflight.warnings)
        self.assertIn("60 秒安全时限", preflight.items[0].messages[0])


class RuleFieldIdentityP1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traditional = "服務名稱"
        cls.with_space = "服务 名称"
        cls.with_brackets = "办理（状态）"
        cls.with_punctuation = "受理-状态"
        cls.long_field = "超长字段" + "甲" * 180
        cls.fields = (
            cls.traditional,
            cls.with_space,
            cls.with_brackets,
            cls.with_punctuation,
            cls.long_field,
        )
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(cls.fields)
        writer.writerow(["值", "值", "办理", "受理", "值"])
        cls.report = evaluate_uploaded_dataset(
            output.getvalue().encode("utf-8"),
            "p1-chinese-fields.csv",
        )
        cls.provider = TemplateRuleAuthoringProvider()

    def _compile(self, intent):
        return compile_custom_rule_draft(
            self.report,
            user_intent=intent,
            provider=self.provider,
            created_at="2026-08-14T00:00:00Z",
        )

    def test_exact_chinese_field_spelling_is_preserved_in_the_dsl(self):
        for field in self.fields:
            with self.subTest(field=field):
                draft = self._compile(f"{field}为必填字段")
                self.assertEqual(draft.status, "draft")
                self.assertEqual(draft.rule_spec.fields, (field,))
                validation = validate_rule_draft(draft, self.report)
                self.assertTrue(validation.valid, validation.errors)

        long_draft = self._compile(f"{self.long_field}为必填字段")
        self.assertLessEqual(len(long_draft.rule_spec.name), 120)
        self.assertEqual(long_draft.rule_spec.fields, (self.long_field,))

    def test_near_field_spellings_enter_clarification_instead_of_guessing(self):
        near_intents = (
            "服务名称为必填字段",  # 简体不能替代传统字段
            "服务　名称为必填字段",  # 全角空格不能替代半角空格
            "办理(状态)为必填字段",  # 半角括号不能替代全角括号
            "受理－状态为必填字段",  # 全角连字符不能替代半角连字符
        )
        for intent in near_intents:
            with self.subTest(intent=intent):
                draft = self._compile(intent)
                self.assertEqual(draft.status, "needs_clarification")
                self.assertIsNone(draft.rule_spec)
                self.assertTrue(draft.clarification_questions)


if __name__ == "__main__":
    unittest.main()
