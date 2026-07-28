"""网页上传服务和展示适配层测试。"""

from pathlib import Path
import json
import unittest

from src.models import MetricResult
from src.parser import UnsupportedFileTypeError
from src.presentation import (
    build_metric_rows,
    build_profile_rows,
    build_risk_chart_rows,
    build_summary,
    format_metric_value,
    serialize_markdown_report,
    serialize_report,
)
from src.upload_service import evaluate_uploaded_dataset


SAMPLE_DATA = Path(__file__).parents[1] / "sample_data"


class PresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = evaluate_uploaded_dataset(
            (SAMPLE_DATA / "bad_dataset.csv").read_bytes(),
            "bad_dataset.csv",
        )

    def test_metric_value_formatting(self):
        ratio = MetricResult(
            id="ratio",
            name="比例",
            category="测试",
            status="evaluated",
            value=0.125,
            unit="ratio",
            scope="dataset",
        )
        days = MetricResult(
            id="days",
            name="天数",
            category="测试",
            status="evaluated",
            value=365,
            unit="days",
            scope="dataset",
        )
        self.assertEqual(format_metric_value(ratio), "12.50%")
        self.assertEqual(format_metric_value(days), "365 天")

    def test_summary_counts_distinct_metric_ids(self):
        summary = build_summary(self.report)
        self.assertEqual(summary["row_count"], 6)
        self.assertEqual(summary["column_count"], 7)
        self.assertEqual(summary["metric_count"], 13)
        self.assertGreater(summary["risk_count"], 0)

    def test_tables_preserve_details(self):
        metric_rows = build_metric_rows(self.report)
        profile_rows = build_profile_rows(self.report)
        self.assertTrue(any(row["字段"] == "service_name" for row in metric_rows))
        self.assertTrue(any(row["字段"] == "source_url" for row in profile_rows))
        self.assertTrue(all("非空样例" not in row for row in profile_rows))

    def test_risk_chart_has_all_levels_in_stable_order(self):
        rows = build_risk_chart_rows(self.report)
        self.assertEqual([row["级别"] for row in rows], ["警告", "关注", "提示"])

    def test_report_download_is_human_readable_utf8_markdown(self):
        payload = serialize_markdown_report(self.report).decode("utf-8")
        self.assertIn("# 数据集质量评估报告：bad_dataset", payload)
        self.assertIn("## 风险提示", payload)
        self.assertIn("## 指标明细", payload)
        self.assertIn("## 字段画像", payload)
        self.assertIn("bad_dataset.csv", payload)

    def test_structured_report_download_is_strict_json(self):
        payload = serialize_report(self.report).decode("utf-8")
        structured = json.loads(
            payload,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )

        self.assertEqual(structured, self.report.to_dict())
        self.assertTrue(
            all(
                column["non_null_samples"] == []
                for column in structured["profile"]["columns"]
            )
        )


class UploadServiceTests(unittest.TestCase):
    def test_uploaded_csv_uses_existing_workflow(self):
        report = evaluate_uploaded_dataset(
            (SAMPLE_DATA / "good_dataset.csv").read_bytes(),
            "folder/good_dataset.csv",
            dataset_name="网页上传样例",
        )
        self.assertEqual(report.status, "success")
        self.assertEqual(report.dataset.name, "网页上传样例")
        self.assertEqual(report.dataset.file_name, "good_dataset.csv")
        self.assertEqual(len({metric.id for metric in report.metrics}), 13)

    def test_uploaded_legacy_excel_uses_existing_workflow(self):
        report = evaluate_uploaded_dataset(
            (SAMPLE_DATA / "good_dataset.xls").read_bytes(),
            "folder/good_dataset.xls",
            dataset_name="旧版 Excel 样例",
            sheet_name="服务事项",
        )
        self.assertEqual(report.status, "success")
        self.assertEqual(report.dataset.file_type, "xls")
        self.assertEqual(report.dataset.sheet_name, "服务事项")
        self.assertEqual(report.profile["row_count"], 5)

    def test_unsupported_upload_is_rejected_before_writing(self):
        with self.assertRaises(UnsupportedFileTypeError):
            evaluate_uploaded_dataset(b"text", "unsupported.txt")


if __name__ == "__main__":
    unittest.main()
