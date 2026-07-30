"""v0.5 会话历史、跨版本比较与整改闭环的 AppTest。"""

from datetime import date
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).parents[1]
REFERENCE_DATE = date(2026, 7, 17)


class V05StreamlitAppTests(unittest.TestCase):
    @staticmethod
    def _by_label(elements, label):
        return next(element for element in elements if element.label == label)

    def _button(self, app, label):
        return self._by_label(app.button, label)

    def _new_app(self):
        app = AppTest.from_file(
            str(PROJECT_ROOT / "app.py"),
            default_timeout=15,
        )
        app.run()
        self.assertFalse(app.exception)
        return app

    def _evaluate(self, app, file_name):
        sample = PROJECT_ROOT / "sample_data" / file_name
        mime_type = (
            "application/json"
            if sample.suffix == ".json"
            else "text/csv"
        )
        self._by_label(
            app.file_uploader,
            "选择数据文件",
        ).set_value((sample.name, sample.read_bytes(), mime_type))
        app.run()
        self._by_label(
            app.date_input,
            "评估基准日期",
        ).set_value(REFERENCE_DATE)
        self._button(app, "运行质量评估").click().run()
        self.assertFalse(app.exception)
        return app

    def _set_series(self, app, series_id):
        self._by_label(
            app.text_input,
            "治理对象标识",
        ).set_value(series_id)

    def _save_current(self, app, version_label):
        self._by_label(
            app.text_input,
            "当前报告版本标签",
        ).set_value(version_label)
        self._button(app, "保存当前报告到会话历史").click().run()
        self.assertFalse(app.exception)
        return app

    def _import_report(self, app, report_name, version_label):
        report_path = PROJECT_ROOT / "reports" / report_name
        self._by_label(
            app.file_uploader,
            "导入严格 QualityReport JSON",
        ).set_value(
            (
                report_path.name,
                report_path.read_bytes(),
                "application/json",
            )
        )
        self._by_label(
            app.text_input,
            "导入报告版本标签",
        ).set_value(version_label)
        self._button(app, "导入历史报告").click().run()
        self.assertFalse(app.exception)
        return app

    def test_history_is_always_visible_and_save_is_explicit(self):
        app = self._new_app()

        self.assertTrue(
            any(
                item.value == "v0.5 本地历史与整改"
                for item in app.subheader
            )
        )
        self.assertEqual(
            len(
                app.session_state[
                    "v05_report_history_store"
                ].list_entries()
            ),
            0,
        )
        self.assertEqual(len(app.download_button), 0)

        self._evaluate(app, "good_dataset.csv")
        self.assertEqual(
            [button.label for button in app.download_button],
            [
                "下载结构化报告（JSON）",
                "下载评估报告（Markdown）",
                "下载疑似问题位置（CSV）",
            ],
        )
        self.assertEqual(
            len(
                app.session_state[
                    "v05_report_history_store"
                ].list_entries()
            ),
            0,
        )

        self._set_series(app, "政务服务事项")
        self._save_current(app, "整改前")
        entries = app.session_state[
            "v05_report_history_store"
        ].list_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].version_label, "整改前")
        self.assertEqual(entries[0].dataset_series_id, "政务服务事项")
        self.assertEqual(len(app.download_button), 3)

        self._by_label(
            app.file_uploader,
            "选择数据文件",
        ).set_value(
            (
                "bad_dataset.csv",
                (
                    PROJECT_ROOT
                    / "sample_data"
                    / "bad_dataset.csv"
                ).read_bytes(),
                "text/csv",
            )
        )
        app.run()
        self.assertNotIn(
            "quality_report",
            app.session_state.filtered_state,
        )
        self.assertEqual(
            len(
                app.session_state[
                    "v05_report_history_store"
                ].list_entries()
            ),
            1,
        )
        self.assertTrue(
            any(
                "版本趋势" in item.value
                for item in app.markdown
            )
        )

    def test_strict_import_single_delete_and_clear_are_explicit(self):
        app = self._new_app()
        self._set_series(app, "导入治理对象")
        self._import_report(app, "good_report.json", "导入-v1")

        store = app.session_state["v05_report_history_store"]
        self.assertEqual(len(store.list_entries()), 1)
        delete_button = self._button(app, "删除所选历史报告")
        self.assertTrue(delete_button.disabled)
        delete_button.click().run()
        self.assertEqual(len(store.list_entries()), 1)

        self._by_label(
            app.checkbox,
            "我确认删除所选历史报告。",
        ).check()
        app.run()
        self._button(app, "删除所选历史报告").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(store.list_entries()), 0)

        self._import_report(app, "good_report.json", "导入-v1")
        self._import_report(app, "bad_report.json", "导入-v2")
        self.assertEqual(len(store.list_entries()), 2)
        clear_button = self._button(app, "清空全部会话历史")
        self.assertTrue(clear_button.disabled)
        clear_button.click().run()
        self.assertEqual(len(store.list_entries()), 2)

        self._by_label(
            app.checkbox,
            "我确认清空当前会话的全部历史报告。",
        ).check()
        app.run()
        self._button(app, "清空全部会话历史").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(store.list_entries()), 0)

    def test_invalid_import_is_rejected_without_app_exception(self):
        app = self._new_app()
        self._set_series(app, "导入治理对象")
        self._by_label(
            app.file_uploader,
            "导入严格 QualityReport JSON",
        ).set_value(
            (
                "invalid.json",
                b'{"not": "a quality report"}',
                "application/json",
            )
        )
        self._by_label(
            app.text_input,
            "导入报告版本标签",
        ).set_value("坏报告")
        self._button(app, "导入历史报告").click().run()

        self.assertFalse(app.exception)
        self.assertEqual(
            len(
                app.session_state[
                    "v05_report_history_store"
                ].list_entries()
            ),
            0,
        )
        self.assertTrue(
            any(
                "历史报告未导入" in message.value
                for message in app.error
            )
        )

    def test_comparison_requires_confirmation_and_is_selection_bound(self):
        app = self._new_app()
        self._set_series(app, "比较治理对象")
        self._import_report(app, "good_report.json", "整改前")
        self._import_report(app, "bad_report.json", "整改后")

        compare_button = self._button(app, "比较固定报告")
        self.assertTrue(compare_button.disabled)
        compare_button.click().run()
        state = app.session_state["v05_history_comparison_state"]
        self.assertIsNone(state["comparison"])
        self.assertEqual(len(app.download_button), 0)

        self._by_label(
            app.checkbox,
            "我已核对两份固定报告，并明确确认它们属于同一治理对象。",
        ).check()
        app.run()
        self._button(app, "比较固定报告").click().run()

        self.assertFalse(app.exception)
        state = app.session_state["v05_history_comparison_state"]
        comparison = state["comparison"]
        action_plan = state["action_plan"]
        self.assertIsNotNone(comparison)
        self.assertIsNotNone(action_plan)
        self.assertGreater(len(action_plan.tasks), 0)
        self.assertEqual(
            comparison.lineage["dataset_series_id"],
            "比较治理对象",
        )
        self.assertTrue(comparison.lineage["same_series_confirmed"])
        self.assertEqual(
            action_plan.comparison_sha256,
            comparison.comparison_sha256,
        )
        self.assertEqual(
            [button.label for button in app.download_button],
            [
                "下载报告比较（JSON）",
                "下载整改行动计划（JSON）",
                "下载整改行动计划（Markdown）",
                "下载整改行动计划（CSV）",
            ],
        )
        self.assertTrue(
            self._button(app, "生成治理记录").disabled
        )

        assigned_task_id = action_plan.tasks[0].task_id
        original_plan_sha256 = action_plan.plan_sha256
        self._by_label(
            app.text_input,
            "负责人",
        ).set_value("测试责任人")
        self._by_label(
            app.date_input,
            "计划完成日期",
        ).set_value(date(2026, 8, 31))
        self._by_label(
            app.selectbox,
            "任务状态",
        ).set_value("进行中")
        self._button(app, "保存任务分派").click().run()

        self.assertFalse(app.exception)
        state = app.session_state["v05_history_comparison_state"]
        assigned_plan = state["action_plan"]
        assigned_task = next(
            task
            for task in assigned_plan.tasks
            if task.task_id == assigned_task_id
        )
        self.assertNotEqual(
            assigned_plan.plan_sha256,
            original_plan_sha256,
        )
        self.assertEqual(assigned_task.assignee, "测试责任人")
        self.assertEqual(assigned_task.due_date, "2026-08-31")
        self.assertEqual(assigned_task.status, "in_progress")
        self.assertIsNone(state["governance_record"])

        self._by_label(
            app.text_input,
            "记录人标识（本地自声明）",
        ).set_value("本地测试记录人")
        self._by_label(
            app.checkbox,
            "我确认该记录人标识仅为本地自声明，系统未验证身份。",
        ).check()
        app.run()
        governance_button = self._button(app, "生成治理记录")
        self.assertFalse(governance_button.disabled)
        governance_button.click().run()

        self.assertFalse(app.exception)
        state = app.session_state["v05_history_comparison_state"]
        governance_record = state["governance_record"]
        self.assertIsNotNone(governance_record)
        self.assertEqual(
            governance_record.comparison_sha256,
            comparison.comparison_sha256,
        )
        self.assertEqual(
            governance_record.plan_sha256,
            assigned_plan.plan_sha256,
        )
        self.assertEqual(
            governance_record.dataset_series_id,
            "比较治理对象",
        )
        self.assertEqual(
            governance_record.to_dict()["operator"],
            {
                "label": "本地测试记录人",
                "identity_verified": False,
            },
        )
        self.assertEqual(
            [button.label for button in app.download_button],
            [
                "下载报告比较（JSON）",
                "下载整改行动计划（JSON）",
                "下载整改行动计划（Markdown）",
                "下载整改行动计划（CSV）",
                "下载治理记录（JSON）",
            ],
        )

        self._by_label(
            app.selectbox,
            "任务状态",
        ).set_value("已完成")
        self._button(app, "保存任务分派").click().run()
        self.assertFalse(app.exception)
        state = app.session_state["v05_history_comparison_state"]
        self.assertIsNone(state["governance_record"])
        self.assertNotEqual(
            state["action_plan"].plan_sha256,
            governance_record.plan_sha256,
        )
        self.assertNotIn(
            "下载治理记录（JSON）",
            [button.label for button in app.download_button],
        )

        baseline = self._by_label(app.selectbox, "整改前报告")
        self._by_label(
            app.selectbox,
            "整改后报告",
        ).set_value(baseline.value)
        app.run()

        self.assertFalse(app.exception)
        reset_state = app.session_state[
            "v05_history_comparison_state"
        ]
        self.assertIsNone(reset_state["comparison"])
        self.assertIsNone(reset_state["action_plan"])
        self.assertIsNone(reset_state["governance_record"])
        self.assertEqual(len(app.download_button), 0)


if __name__ == "__main__":
    unittest.main()
