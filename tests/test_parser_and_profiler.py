"""阶段 1：文件解析、数据画像与 Markdown 报告的端到端测试。"""

import tempfile
import unittest
from pathlib import Path

from src.parser import DatasetReadError, UnsupportedFileTypeError, parse_dataset
from src.workflow import build_profile_report
from src.report import save_report


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "sample_data"


class ParserAndProfilerTests(unittest.TestCase):
    def test_csv_is_parsed_and_profiled(self):
        report = build_profile_report(SAMPLES / "good_dataset.csv")

        self.assertEqual(report.dataset.file_type, "csv")
        self.assertEqual(report.profile["row_count"], 5)
        self.assertEqual(report.profile["column_count"], 7)
        self.assertEqual(report.profile["columns"][0]["name"], "record_id")
        self.assertEqual(report.profile["columns"][0]["inferred_type"], "numeric")
        recognized = report.profile["recognized_fields"]
        self.assertIn("update_time", recognized["date"])
        self.assertIn("handling_days", recognized["numeric"])
        self.assertIn("source_url", recognized["url"])
        self.assertIn("department", recognized["source"])
        self.assertIn("version", recognized["version"])

    def test_json_is_parsed_and_profiled(self):
        report = build_profile_report(SAMPLES / "minimal_dataset.json")

        self.assertEqual(report.dataset.file_type, "json")
        self.assertEqual(report.profile["row_count"], 3)
        self.assertEqual(report.profile["column_count"], 2)
        self.assertEqual(report.profile["columns"][0]["inferred_type"], "text")

    def test_excel_is_parsed_and_sheet_name_is_retained(self):
        for extension in ("xls", "xlsx"):
            with self.subTest(extension=extension):
                report = build_profile_report(
                    SAMPLES / f"good_dataset.{extension}"
                )

                self.assertEqual(report.dataset.file_type, extension)
                self.assertEqual(report.dataset.sheet_name, "服务事项")
                self.assertEqual(report.profile["row_count"], 5)

    def test_nested_json_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested.json"
            path.write_text('[{"事项": {"名称": "测试"}}]', encoding="utf-8")
            with self.assertRaisesRegex(DatasetReadError, "嵌套"):
                parse_dataset(path)

    def test_zip_file_is_not_a_supported_dataset_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy-shards.zip"
            path.write_bytes(b"PK\x03\x04not-supported")
            with self.assertRaisesRegex(UnsupportedFileTypeError, r"\.zip"):
                parse_dataset(path)

    def test_zero_byte_csv_returns_failed_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.csv"
            path.write_bytes(b"")
            report = build_profile_report(path)

        self.assertEqual(report.status, "failed")
        self.assertEqual(report.metrics[0].value, 0.0)
        self.assertIn("不包含可读取的表头或记录", report.execution["errors"][0])
        self.assertEqual(len(report.not_assessable), 12)

    def test_profile_report_can_be_saved(self):
        report = build_profile_report(SAMPLES / "bad_dataset.csv")
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "report.md"
            save_report(report, output)
            saved = output.read_text(encoding="utf-8")

        self.assertIn("# 数据集质量评估报告：bad_dataset", saved)
        self.assertIn("- 文件名：bad_dataset.csv", saved)
        self.assertIn("- 记录数：6", saved)
        self.assertIn("## 指标明细", saved)
