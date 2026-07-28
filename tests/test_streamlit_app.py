"""Streamlit 页面真实控件交互测试。"""

from datetime import date
import io
from pathlib import Path
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).parents[1]
REFERENCE_DATE = date(2026, 7, 17)


class StreamlitAppTests(unittest.TestCase):
    def _new_app(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15)
        app.run()
        self.assertFalse(app.exception)
        return app

    def _upload_and_run(
        self,
        app,
        file_name,
        content,
        mime_type,
        dataset_name="",
        sheet_name=None,
    ):
        app.file_uploader[0].set_value((file_name, content, mime_type))
        app.run()
        app.date_input[0].set_value(REFERENCE_DATE)
        if dataset_name:
            app.text_input[0].set_value(dataset_name)
        if sheet_name is not None:
            self.assertGreaterEqual(len(app.text_input), 2)
            app.text_input[1].set_value(sheet_name)
        self.assertFalse(app.button[0].disabled)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        return app

    def _assert_report_surface(self, app, expected_type, expected_sheet=None):
        report = app.session_state["quality_report"]
        self.assertEqual(report.status, "success")
        self.assertEqual(report.dataset.file_type, expected_type)
        self.assertEqual(report.dataset.sheet_name, expected_sheet)
        self.assertEqual(len({metric.id for metric in report.metrics}), 13)
        self.assertEqual(len(app.metric), 5)
        self.assertGreaterEqual(len(app.dataframe), 2)
        self.assertEqual(len(app.download_button), 2)
        self.assertTrue(any(item.value == "风险分布" for item in app.subheader))
        self.assertGreaterEqual(len(app.get("vega_lite_chart")), 1)
        self.assertEqual(
            [button.label for button in app.download_button],
            ["下载结构化报告（JSON）", "下载评估报告（Markdown）"],
        )

    def test_reference_date_is_explicit_and_flows_to_report(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.csv"
        app = self._new_app()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "text/csv",
        )

        lag_metric = next(
            metric
            for metric in app.session_state["quality_report"].metrics
            if metric.id == "update_lag_days"
        )
        self.assertEqual(
            lag_metric.evidence["reference_date"], REFERENCE_DATE.isoformat()
        )

    def test_supported_formats_run_through_full_report_surface(self):
        samples = PROJECT_ROOT / "sample_data"
        cases = (
            ("good_dataset.csv", "text/csv", "csv", None),
            (
                "good_dataset.xls",
                "application/vnd.ms-excel",
                "xls",
                "服务事项",
            ),
            (
                "good_dataset.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "xlsx",
                "服务事项",
            ),
            ("minimal_dataset.json", "application/json", "json", None),
            (
                "geojson_feature_collection.geojson",
                "application/geo+json",
                "geojson",
                None,
            ),
        )
        for file_name, mime_type, expected_type, sheet_name in cases:
            with self.subTest(file_name=file_name):
                app = self._new_app()
                self._upload_and_run(
                    app,
                    file_name,
                    (samples / file_name).read_bytes(),
                    mime_type,
                    dataset_name=f"验收-{expected_type}",
                    sheet_name=sheet_name,
                )
                self._assert_report_surface(app, expected_type, sheet_name)

    def test_changing_upload_clears_stale_report_before_next_run(self):
        samples = PROJECT_ROOT / "sample_data"
        app = self._new_app()
        self._upload_and_run(
            app,
            "good_dataset.csv",
            (samples / "good_dataset.csv").read_bytes(),
            "text/csv",
        )
        self.assertEqual(app.session_state["quality_report"].dataset.file_type, "csv")

        app.file_uploader[0].set_value(
            (
                "minimal_dataset.json",
                (samples / "minimal_dataset.json").read_bytes(),
                "application/json",
            )
        )
        app.run()
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.download_button), 0)

        app.button[0].click().run()
        self._assert_report_surface(app, "json")

    def test_jsonl_runs_through_full_report_surface_and_shows_warning(self):
        sample = PROJECT_ROOT / "sample_data" / "json_records_dataset.jsonl"
        app = self._new_app()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "application/x-ndjson",
        )

        self._assert_report_surface(app, "jsonl")
        report = app.session_state["quality_report"]
        self.assertTrue(
            any("JSON Lines" in item for item in report.execution["warnings"])
        )

    def test_json_zip_runs_through_full_report_surface_and_shows_warning(self):
        content = io.BytesIO()
        with ZipFile(content, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("001.json", '[["id","name"],[1,"A"]]')
            archive.writestr("002.json", '[["id","name"],[2,"B"]]')

        app = self._new_app()
        self._upload_and_run(
            app,
            "records.zip",
            content.getvalue(),
            "application/zip",
        )

        self._assert_report_surface(app, "zip")
        report = app.session_state["quality_report"]
        self.assertEqual(report.profile["row_count"], 2)
        self.assertTrue(
            any("ZIP JSON 分片包" in item for item in report.execution["warnings"])
        )

    def test_nested_json_returns_downloadable_failed_report(self):
        sample = PROJECT_ROOT / "sample_data" / "nested_dataset.json"
        app = self._new_app()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "application/json",
        )

        report = app.session_state["quality_report"]
        self.assertEqual(report.status, "failed")
        self.assertIn("嵌套", report.execution["errors"][0])
        self.assertEqual(len({metric.id for metric in report.metrics}), 13)
        self.assertEqual(len(app.download_button), 2)

    def test_missing_excel_sheet_returns_explainable_failed_report(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.xlsx"
        app = self._new_app()
        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            sheet_name="不存在的工作表",
        )

        report = app.session_state["quality_report"]
        self.assertEqual(report.status, "failed")
        self.assertIn("未找到工作表", report.execution["errors"][0])
        self.assertIn("服务事项", report.execution["errors"][0])
        self.assertEqual(len(app.download_button), 2)


if __name__ == "__main__":
    unittest.main()
