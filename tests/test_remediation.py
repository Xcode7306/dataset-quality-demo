"""v0.5 整改计划、分派、导出与治理留痕契约回归。"""

from copy import deepcopy
import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

from jsonschema import Draft202012Validator, FormatChecker

from src.comparison_presentation import (
    serialize_action_plan_csv,
    serialize_action_plan_json,
    serialize_action_plan_markdown,
    serialize_governance_record,
)
from src.comparison_service import compare_reports
from src.history_store import DEFAULT_HISTORY_POLICY
from src.remediation import (
    RemediationValidationError,
    assign_task,
    build_action_plan,
    build_governance_record,
    validate_governance_record,
    validate_remediation_plan,
)
from src.workflow import build_profile_report


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "sample_data"
PLAN_SCHEMA_PATH = ROOT / "schemas" / "remediation-plan.schema.json"
GOVERNANCE_SCHEMA_PATH = ROOT / "schemas" / "governance-record.schema.json"
REFERENCE_DATE = date(2026, 7, 17)
RECORDED_AT = datetime(2026, 7, 29, 9, 30, tzinfo=timezone.utc)


def _report(sample: str):
    return build_profile_report(
        SAMPLES / sample,
        dataset_name="同一治理数据集",
        reference_date=REFERENCE_DATE,
    )


def _comparison(
    baseline_sample: str = "good_dataset.csv",
    target_sample: str = "bad_dataset.csv",
):
    return compare_reports(
        _report(baseline_sample),
        _report(target_sample),
        dataset_series_id="政务服务事项",
        same_series_confirmed=True,
    )


def _canonical_hash(payload: dict, hash_field: str) -> str:
    hash_payload = deepcopy(payload)
    hash_payload.pop(hash_field)
    encoded = json.dumps(
        hash_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RemediationPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plan_schema = json.loads(
            PLAN_SCHEMA_PATH.read_text(encoding="utf-8")
        )
        governance_schema = json.loads(
            GOVERNANCE_SCHEMA_PATH.read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(plan_schema)
        Draft202012Validator.check_schema(governance_schema)
        cls.plan_validator = Draft202012Validator(
            plan_schema,
            format_checker=FormatChecker(),
        )
        cls.governance_validator = Draft202012Validator(
            governance_schema,
            format_checker=FormatChecker(),
        )

    def setUp(self):
        self.comparison = _comparison()
        self.plan = build_action_plan(self.comparison)

    def test_plan_is_deterministic_schema_valid_and_self_hashed(self):
        first = self.plan
        second = build_action_plan(self.comparison)
        payload = first.to_dict()

        self.assertEqual(first, second)
        self.assertEqual(first.plan_id, second.plan_id)
        self.assertEqual(first.plan_sha256, second.plan_sha256)
        self.assertEqual(
            first.comparison_sha256,
            self.comparison.comparison_sha256,
        )
        self.assertEqual(
            list(self.plan_validator.iter_errors(payload)),
            [],
        )
        self.assertEqual(
            payload["plan_sha256"],
            _canonical_hash(payload, "plan_sha256"),
        )
        self.assertEqual(validate_remediation_plan(first), payload)

        tampered = deepcopy(payload)
        tampered["tasks"][0]["detail"] += "伪造内容"
        with self.assertRaisesRegex(
            RemediationValidationError,
            "自身哈希",
        ):
            validate_remediation_plan(tampered)

    def test_task_priority_order_linkage_and_stable_ids(self):
        task_ids = [task.task_id for task in self.plan.tasks]
        rebuilt_ids = [
            task.task_id
            for task in build_action_plan(self.comparison).tasks
        ]
        priorities = [task.priority for task in self.plan.tasks]
        priority_rank = {"high": 0, "medium": 1, "low": 2}
        risk_by_change_id = {
            change.change_id: change
            for change in self.comparison.risk_changes
        }

        self.assertEqual(task_ids, rebuilt_ids)
        self.assertEqual(len(task_ids), len(set(task_ids)))
        self.assertTrue(
            all(re.fullmatch(r"task-[0-9a-f]{24}", value) for value in task_ids)
        )
        self.assertEqual(
            [priority_rank[value] for value in priorities],
            sorted(priority_rank[value] for value in priorities),
        )
        self.assertIn("high", priorities)
        self.assertIn("medium", priorities)
        self.assertTrue(all(task.change_ids for task in self.plan.tasks))

        for task in self.plan.tasks:
            if task.category != "risk":
                continue
            self.assertEqual(len(task.change_ids), 1)
            risk = risk_by_change_id[task.change_ids[0]]
            expected_priority = {
                "warning": "high",
                "attention": "medium",
            }.get(risk.target_level, "low")
            self.assertEqual(task.priority, expected_priority)

    def test_task_limit_is_prioritized_and_explicitly_disclosed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fields = [f"field_{index:02d}" for index in range(35)]
            baseline = root / "baseline.csv"
            target = root / "target.csv"
            baseline.write_text(
                ",".join(fields)
                + "\n"
                + "\n".join(
                    ",".join(f"value_{row}_{index}" for index in range(35))
                    for row in range(4)
                )
                + "\n",
                encoding="utf-8",
            )
            target.write_text(
                ",".join(fields)
                + "\n"
                + "\n".join("," * 34 for _ in range(4))
                + "\n",
                encoding="utf-8",
            )
            comparison = compare_reports(
                build_profile_report(
                    baseline,
                    dataset_name="多任务治理对象",
                    reference_date=REFERENCE_DATE,
                ),
                build_profile_report(
                    target,
                    dataset_name="多任务治理对象",
                    reference_date=REFERENCE_DATE,
                ),
                dataset_series_id="多任务治理对象",
                same_series_confirmed=True,
            )

        plan = build_action_plan(comparison)

        self.assertEqual(len(plan.tasks), 30)
        self.assertTrue(all(task.priority == "high" for task in plan.tasks))
        self.assertTrue(all(task.category == "risk" for task in plan.tasks))
        self.assertTrue(
            any(
                "候选任务共" in limitation and "未进入计划" in limitation
                for limitation in plan.improvement_summary.limitations
            )
        )
        self.assertIn(
            "当前计划已达到任务上限；处理后应重新比较并生成下一批任务。",
            plan.next_round_suggestions,
        )

    def test_assign_task_changes_only_manual_fields_and_plan_hash(self):
        task = self.plan.tasks[0]
        updated = assign_task(
            self.plan,
            task.task_id,
            assignee="  张三  ",
            due_date=date(2026, 8, 31),
            status="in_progress",
        )
        updated_task = next(
            item for item in updated.tasks if item.task_id == task.task_id
        )
        original_evidence = task.to_dict()
        updated_evidence = updated_task.to_dict()
        for field in ("assignee", "due_date", "status"):
            original_evidence.pop(field)
            updated_evidence.pop(field)

        self.assertEqual(original_evidence, updated_evidence)
        self.assertEqual(updated_task.priority, task.priority)
        self.assertEqual(updated_task.change_ids, task.change_ids)
        self.assertEqual(updated_task.assignee, "张三")
        self.assertEqual(updated_task.due_date, "2026-08-31")
        self.assertEqual(updated_task.status, "in_progress")
        self.assertEqual(updated.plan_id, self.plan.plan_id)
        self.assertNotEqual(updated.plan_sha256, self.plan.plan_sha256)
        self.assertEqual(
            updated.plan_sha256,
            _canonical_hash(updated.to_dict(), "plan_sha256"),
        )
        self.assertEqual(self.plan.tasks[0], task)

    def test_assign_task_strictly_rejects_invalid_inputs(self):
        task_id = self.plan.tasks[0].task_id
        invalid_calls = (
            lambda: assign_task(self.plan, "task-" + "0" * 24),
            lambda: assign_task(self.plan, task_id, status="closed"),
            lambda: assign_task(self.plan, task_id, assignee=123),
            lambda: assign_task(self.plan, task_id, assignee="张三\n李四"),
            lambda: assign_task(self.plan, task_id, assignee="甲" * 101),
            lambda: assign_task(
                self.plan,
                task_id,
                due_date="2026-02-30",
            ),
            lambda: assign_task(self.plan, task_id, due_date=object()),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(RemediationValidationError):
                    call()

    def test_json_markdown_and_csv_exports_are_safe_and_traceable(self):
        task_id = self.plan.tasks[0].task_id
        formula_plan = assign_task(
            self.plan,
            task_id,
            assignee="=HYPERLINK(\"https://example.invalid\",\"点击\")",
            due_date="2026-08-31",
            status="in_progress",
        )

        json_bytes = serialize_action_plan_json(formula_plan)
        self.assertIsInstance(json_bytes, bytes)
        self.assertEqual(json.loads(json_bytes), formula_plan.to_dict())

        csv_bytes = serialize_action_plan_csv(formula_plan)
        self.assertIsInstance(csv_bytes, bytes)
        self.assertTrue(csv_bytes.startswith(b"\xef\xbb\xbf"))
        rows = list(
            csv.DictReader(
                io.StringIO(csv_bytes.decode("utf-8-sig"))
            )
        )
        exported = next(row for row in rows if task_id in row.values())
        assignee_cell = next(
            value
            for value in exported.values()
            if value and "HYPERLINK" in value
        )
        self.assertTrue(assignee_cell.startswith("'="))
        self.assertFalse(assignee_cell.startswith("="))
        for row in rows:
            for value in row.values():
                if value is not None:
                    self.assertNotRegex(value.lstrip(), r"^[=+\-@]")

        markdown_assignee = (
            r"#负责人 | [链接](https://example.invalid) "
            r"<管理员> *重点* _复核_ `标记` !"
        )
        markdown_plan = assign_task(
            self.plan,
            task_id,
            assignee=markdown_assignee,
        )
        markdown_bytes = serialize_action_plan_markdown(markdown_plan)
        self.assertIsInstance(markdown_bytes, bytes)
        markdown = markdown_bytes.decode("utf-8")
        self.assertNotIn(markdown_assignee, markdown)
        for escaped in (
            r"\#",
            r"\|",
            r"\[",
            r"\]",
            r"\(",
            r"\)",
            r"\<",
            r"\>",
            r"\*",
            r"\_",
            r"\`",
            r"\!",
        ):
            self.assertIn(escaped, markdown)
        self.assertIn(markdown_plan.plan_sha256, markdown)
        self.assertIn(markdown_plan.comparison_sha256, markdown)

    def test_exporters_reject_tampered_plan(self):
        tampered = self.plan.to_dict()
        tampered["improvement_summary"]["headline"] = "伪造结论"
        for serializer in (
            serialize_action_plan_json,
            serialize_action_plan_markdown,
            serialize_action_plan_csv,
        ):
            with self.subTest(serializer=serializer.__name__):
                with self.assertRaises(RemediationValidationError):
                    serializer(tampered)

    def test_governance_record_binds_artifacts_policy_and_unverified_operator(self):
        first_task_id = self.plan.tasks[0].task_id
        assigned = assign_task(
            self.plan,
            first_task_id,
            assignee="张三",
            due_date="2026-08-31",
            status="done",
        )
        record = build_governance_record(
            self.comparison,
            assigned,
            operator="治理专员（本地输入）",
            recorded_at=RECORDED_AT,
        )
        repeated = build_governance_record(
            self.comparison,
            assigned,
            operator="治理专员（本地输入）",
            recorded_at=RECORDED_AT,
        )
        payload = record.to_dict()

        self.assertEqual(record, repeated)
        self.assertEqual(
            list(self.governance_validator.iter_errors(payload)),
            [],
        )
        self.assertEqual(
            payload["record_sha256"],
            _canonical_hash(payload, "record_sha256"),
        )
        self.assertEqual(
            payload["comparison_sha256"],
            self.comparison.comparison_sha256,
        )
        self.assertEqual(payload["plan_sha256"], assigned.plan_sha256)
        self.assertEqual(
            payload["baseline_report_sha256"],
            self.comparison.baseline.report_sha256,
        )
        self.assertEqual(
            payload["target_report_sha256"],
            self.comparison.target.report_sha256,
        )
        self.assertEqual(
            payload["dataset_series_id"],
            self.comparison.lineage["dataset_series_id"],
        )
        self.assertEqual(
            payload["history_policy"],
            DEFAULT_HISTORY_POLICY.to_dict(),
        )
        self.assertFalse(payload["operator"]["identity_verified"])
        self.assertFalse(
            payload["history_policy"]["identity_authentication"]
        )
        self.assertEqual(payload["operator"]["label"], "治理专员（本地输入）")
        self.assertEqual(payload["recorded_at"], "2026-07-29T09:30:00Z")
        self.assertEqual(payload["outcomes"]["task_status_counts"]["done"], 1)
        self.assertNotIn(
            first_task_id,
            payload["outcomes"]["open_task_ids"],
        )
        self.assertEqual(validate_governance_record(record), payload)
        self.assertEqual(
            json.loads(serialize_governance_record(record)),
            payload,
        )

    def test_governance_record_rejects_forged_or_mismatched_inputs(self):
        reverse_plan = build_action_plan(
            _comparison("bad_dataset.csv", "good_dataset.csv")
        )
        with self.assertRaisesRegex(
            RemediationValidationError,
            "不匹配",
        ):
            build_governance_record(
                self.comparison,
                reverse_plan,
                operator="治理专员",
                recorded_at=RECORDED_AT,
            )

        forged_plan = self.plan.to_dict()
        forged_plan["tasks"][0]["detail"] = "FORGED"
        forged_plan["plan_sha256"] = _canonical_hash(
            forged_plan,
            "plan_sha256",
        )
        # 自身哈希只能证明对象内部未再变化；治理记录还必须复核确定性派生。
        validate_remediation_plan(forged_plan)
        with self.assertRaisesRegex(
            RemediationValidationError,
            "确定性派生|任务证据",
        ):
            build_governance_record(
                self.comparison,
                forged_plan,
                operator="治理专员",
                recorded_at=RECORDED_AT,
            )

        for invalid_operator in ("", " \n", "甲" * 101):
            with self.subTest(operator=invalid_operator):
                with self.assertRaises(RemediationValidationError):
                    build_governance_record(
                        self.comparison,
                        self.plan,
                        operator=invalid_operator,
                        recorded_at=RECORDED_AT,
                    )
        with self.assertRaisesRegex(
            RemediationValidationError,
            "时区",
        ):
            build_governance_record(
                self.comparison,
                self.plan,
                operator="治理专员",
                recorded_at=datetime(2026, 7, 29, 9, 30),
            )

        record = build_governance_record(
            self.comparison,
            self.plan,
            operator="治理专员",
            recorded_at=RECORDED_AT,
        )
        forged_outcome = record.to_dict()
        forged_outcome["outcomes"]["task_status_counts"]["done"] = 999
        with self.assertRaisesRegex(
            RemediationValidationError,
            "自身哈希",
        ):
            validate_governance_record(forged_outcome)

        forged_outcome["record_sha256"] = _canonical_hash(
            forged_outcome,
            "record_sha256",
        )
        with self.assertRaisesRegex(
            RemediationValidationError,
            "状态计数",
        ):
            validate_governance_record(forged_outcome)

        forged_identity = record.to_dict()
        forged_identity["operator"]["identity_verified"] = True
        with self.assertRaisesRegex(
            RemediationValidationError,
            "Schema",
        ):
            validate_governance_record(forged_identity)
        with self.assertRaises(RemediationValidationError):
            serialize_governance_record(forged_identity)


if __name__ == "__main__":
    unittest.main()
