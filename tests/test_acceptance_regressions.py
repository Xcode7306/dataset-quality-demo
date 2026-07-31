"""补充验收回归：编码、JSON 契约、Excel 选表与页面过期状态。"""

from datetime import date
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.cli import main as cli_main
from src.metrics import calculate_recognizable_format_anomaly_rates
from src.parser import parse_dataset
from src.workflow import build_profile_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DATE = date(2026, 7, 17)
EXCEL_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
LEGACY_EXCEL_MIME_TYPE = "application/vnd.ms-excel"


class FormatMetricAcceptanceTests(unittest.TestCase):
    def test_email_format_ignores_empty_values_and_counts_invalid_values(self):
        dataframe = pd.DataFrame(
            {
                "contact_email": [
                    "valid.user@example.gov.cn",
                    "not-an-email",
                    "",
                    None,
                ]
            }
        )

        metrics = calculate_recognizable_format_anomaly_rates(dataframe)

        self.assertEqual(len(metrics), 1)
        metric = metrics[0]
        self.assertEqual(metric.field, "contact_email")
        self.assertEqual(metric.status, "evaluated")
        self.assertEqual(metric.value, 0.5)
        self.assertEqual(metric.evidence["expected_format"], "email")
        self.assertEqual(metric.evidence["checked_count"], 2)
        self.assertEqual(metric.evidence["issue_count"], 1)

    def test_empty_email_field_is_not_assessable(self):
        dataframe = pd.DataFrame({"email": ["", None]})

        metric = calculate_recognizable_format_anomaly_rates(dataframe)[0]

        self.assertEqual(metric.field, "email")
        self.assertEqual(metric.status, "not_assessable")
        self.assertIsNone(metric.value)
        self.assertIn("没有非空值", metric.reason)


class ParserAndCliAcceptanceTests(unittest.TestCase):
    def test_utf8_bom_csv_does_not_expose_bom_in_header(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bom.csv"
            path.write_bytes(
                "﻿事项名称,状态\n户籍登记,有效\n".encode("utf-8")
            )

            parsed = parse_dataset(path)

        self.assertEqual(parsed.dataframe.columns.tolist(), ["事项名称", "状态"])
        self.assertEqual(parsed.dataframe.iloc[0].tolist(), ["户籍登记", "有效"])
        self.assertEqual(parsed.warnings, [])

    def test_gbk_csv_is_parsed_with_an_explicit_encoding_warning(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "gbk.csv"
            path.write_bytes("事项名称,状态\n户籍登记,有效\n".encode("gbk"))

            parsed = parse_dataset(path)

        self.assertEqual(parsed.dataframe.columns.tolist(), ["事项名称", "状态"])
        self.assertEqual(parsed.dataframe.iloc[0].tolist(), ["户籍登记", "有效"])
        self.assertIn("CSV 使用 GBK 编码读取。", parsed.warnings)

    def test_single_flat_json_object_becomes_one_record(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "single.json"
            path.write_text(
                json.dumps(
                    {"record_id": 7, "name": "单条事项", "enabled": True},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            parsed = parse_dataset(path)
            report = build_profile_report(path, reference_date=REFERENCE_DATE)

        self.assertEqual(parsed.dataframe.shape, (1, 3))
        self.assertEqual(parsed.dataframe.loc[0, "record_id"], 7)
        self.assertEqual(parsed.dataframe.loc[0, "name"], "单条事项")
        self.assertEqual(report.status, "success")
        self.assertEqual(report.profile["row_count"], 1)

    def test_common_invalid_json_inputs_share_the_failed_report_contract(self):
        cases = (
            ("malformed", b'[{"id": 1}', "JSON 格式错误"),
            ("scalar", b'"value"', "JSON 顶层必须"),
            ("mixed-list", b'[{"id": 1}, 2]', "每一项都必须是对象"),
            (
                "nested",
                b'[{"id": 1, "meta": {"source": "x"}}]',
                "JSON 包含嵌套对象或列表",
            ),
        )

        for name, content, expected_error in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / f"{name}.json"
                    path.write_bytes(content)

                    report = build_profile_report(path, reference_date=REFERENCE_DATE)

                self.assertEqual(report.status, "failed")
                self.assertEqual(len(report.execution["errors"]), 1)
                self.assertIn(expected_error, report.execution["errors"][0])
                self.assertEqual(len({metric.id for metric in report.metrics}), 13)
                parse_metric = next(
                    metric
                    for metric in report.metrics
                    if metric.id == "file_parse_rate"
                )
                self.assertEqual(parse_metric.status, "evaluated")
                self.assertEqual(parse_metric.value, 0.0)
                self.assertEqual(len(report.not_assessable), 12)

    def test_gb18030_json_is_parsed_with_an_explicit_warning(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "gb18030.json"
            path.write_bytes('[{"name": "中文"}]'.encode("gb18030"))

            parsed = parse_dataset(path)

        self.assertEqual(parsed.dataframe.loc[0, "name"], "中文")
        self.assertIn("JSON 使用 GB18030 编码读取。", parsed.warnings)

    def test_excel_defaults_to_first_sheet_and_accepts_explicit_second_sheet(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "multiple.xlsx"
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame({"marker": ["first"]}).to_excel(
                    writer, index=False, sheet_name="第一张"
                )
                pd.DataFrame({"marker": ["second-a", "second-b"]}).to_excel(
                    writer, index=False, sheet_name="第二张"
                )

            first = parse_dataset(path)
            second = parse_dataset(path, sheet_name="第二张")

        self.assertEqual(first.dataset.sheet_name, "第一张")
        self.assertEqual(first.dataframe["marker"].tolist(), ["first"])
        self.assertEqual(second.dataset.sheet_name, "第二张")
        self.assertEqual(
            second.dataframe["marker"].tolist(), ["second-a", "second-b"]
        )

    def test_cli_sheet_option_selects_the_requested_excel_sheet(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "multiple.xlsx"
            output_path = root / "report.md"
            with pd.ExcelWriter(input_path) as writer:
                pd.DataFrame({"marker": ["first"]}).to_excel(
                    writer, index=False, sheet_name="第一张"
                )
                pd.DataFrame({"marker": ["second-a", "second-b"]}).to_excel(
                    writer, index=False, sheet_name="第二张"
                )
            arguments = [
                "src.cli",
                str(input_path),
                "--sheet",
                "第二张",
                "--reference-date",
                REFERENCE_DATE.isoformat(),
                "--output",
                str(output_path),
            ]

            with patch.object(sys, "argv", arguments), patch("builtins.print"):
                cli_main()
            report = output_path.read_text(encoding="utf-8")

        self.assertIn("- 工作表：第二张", report)
        self.assertIn("- 记录数：2", report)


class StreamlitStateAcceptanceTests(unittest.TestCase):
    @staticmethod
    def _button_by_label(app, label):
        return next(button for button in app.button if button.label == label)

    def _new_app(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)
        app.run()
        self.assertFalse(app.exception)
        return app

    def _assert_report_is_cleared(self, app):
        self.assertNotIn("quality_report", app.session_state.filtered_state)
        self.assertNotIn("agent_ui_state", app.session_state.filtered_state)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.download_button), 0)
        self.assertTrue(
            any("请从左侧上传" in message.value for message in app.info)
        )

    def _upload_and_run(self, app, file_name, content, mime_type):
        app.file_uploader[0].set_value((file_name, content, mime_type))
        app.run()
        app.date_input[0].set_value(REFERENCE_DATE)
        self._button_by_label(app, "运行质量评估").click().run()
        self.assertFalse(app.exception)
        self.assertIn("quality_report", app.session_state.filtered_state)

    def test_dataset_name_and_reference_date_changes_clear_stale_report(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.csv"
        app = self._new_app()
        self._upload_and_run(app, sample.name, sample.read_bytes(), "text/csv")

        app.text_input[0].set_value("新数据集名称")
        app.run()
        self._assert_report_is_cleared(app)

        self._button_by_label(app, "运行质量评估").click().run()
        self.assertEqual(
            app.session_state["quality_report"].dataset.name,
            "新数据集名称",
        )

        app.date_input[0].set_value(date(2026, 7, 18))
        app.run()
        self._assert_report_is_cleared(app)

    def test_excel_sheet_name_change_clears_stale_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "multiple.xlsx"
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame({"marker": ["first"]}).to_excel(
                    writer, index=False, sheet_name="第一张"
                )
                pd.DataFrame({"marker": ["second"]}).to_excel(
                    writer, index=False, sheet_name="第二张"
                )
            content = path.read_bytes()

        app = self._new_app()
        self._upload_and_run(app, "multiple.xlsx", content, EXCEL_MIME_TYPE)
        self.assertEqual(
            app.session_state["quality_report"].dataset.sheet_name,
            "第一张",
        )

        self.assertGreaterEqual(len(app.text_input), 2)
        app.text_input[1].set_value("第二张")
        app.run()
        self._assert_report_is_cleared(app)

    def test_legacy_excel_upload_shows_sheet_input_and_runs(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.xls"
        app = self._new_app()

        self._upload_and_run(
            app,
            sample.name,
            sample.read_bytes(),
            LEGACY_EXCEL_MIME_TYPE,
        )

        self.assertGreaterEqual(len(app.text_input), 2)
        report = app.session_state["quality_report"]
        self.assertEqual(report.status, "success")
        self.assertEqual(report.dataset.file_type, "xls")
        self.assertEqual(report.dataset.sheet_name, "服务事项")
        self.assertEqual(report.profile["row_count"], 5)

    def test_oserror_during_evaluation_removes_old_report_and_shows_safe_error(self):
        sample = PROJECT_ROOT / "sample_data" / "good_dataset.csv"
        app = self._new_app()
        self._upload_and_run(app, sample.name, sample.read_bytes(), "text/csv")

        with patch(
            "src.upload_service.evaluate_uploaded_dataset",
            side_effect=OSError("临时文件写入失败"),
        ) as evaluate:
            self._button_by_label(app, "运行质量评估").click().run()

        evaluate.assert_called_once()
        self.assertFalse(app.exception)
        self._assert_report_is_cleared(app)
        self.assertTrue(
            any(
                message.value
                == "评估未能启动：运行环境或临时文件不可用，请重试。"
                for message in app.error
            )
        )
        self.assertFalse(
            any("临时文件写入失败" in message.value for message in app.error)
        )


if __name__ == "__main__":
    unittest.main()
