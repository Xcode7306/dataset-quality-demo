"""网页上传服务和展示适配层测试。"""

import csv
import io
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
    serialize_issue_locations_csv,
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
        self.assertTrue(
            any(row["字段名称"] == "service_name" for row in metric_rows)
        )
        self.assertTrue(
            any(
                row["指标名称"] == "字段缺失率"
                and row["字段名称"] == "service_name"
                for row in metric_rows
            )
        )
        self.assertTrue(all("引用键" not in row for row in metric_rows))
        self.assertTrue(any(row["字段"] == "source_url" for row in profile_rows))
        self.assertTrue(all("非空样例" not in row for row in profile_rows))

    def test_risk_chart_has_all_levels_in_stable_order(self):
        rows = build_risk_chart_rows(self.report)
        self.assertEqual([row["级别"] for row in rows], ["警告", "关注", "提示"])

    def test_issue_locations_are_human_readable_and_exclude_raw_values(self):
        payload = serialize_issue_locations_csv(self.report).decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(payload)))

        self.assertGreater(len(rows), 0)
        self.assertTrue(
            any(
                row["疑似问题类型"] == "格式异常"
                and row["字段名称"] == "handling_days"
                and row["数据记录序号"] == "3"
                for row in rows
            )
        )
        self.assertTrue(
            any(
                row["疑似问题类型"] == "完全重复记录"
                and row["数据记录序号"] == "2"
                and row["关联记录序号"] == "1"
                and row["备注"]
                == (
                    "第 2 条记录与第 1 条记录内容完全相同；"
                    "关联记录序号 1 表示这组重复数据中首次出现的记录。"
                )
                for row in rows
            )
        )
        self.assertTrue(
            any(
                row["疑似问题类型"] == "规范化后重复记录"
                and row["数据记录序号"] == "2"
                and row["关联记录序号"] == "1"
                and row["备注"]
                == (
                    "第 2 条记录与第 1 条记录在忽略自然文本中的"
                    "大小写、空白和标点差异后相同；关联记录序号 1 "
                    "表示这组重复数据中首次出现的记录。"
                )
                for row in rows
            )
        )
        self.assertTrue(
            all(
                not row["备注"]
                for row in rows
                if row["关联记录序号"] == "—"
            )
        )
        expected_location_count = sum(
            len(metric.issue_locations) for metric in self.report.metrics
        )
        self.assertEqual(len(rows), expected_location_count)
        for metric in self.report.metrics:
            issue_count = metric.evidence.get("issue_count")
            if isinstance(issue_count, int) and not isinstance(issue_count, bool):
                with self.subTest(metric=metric.metric_key):
                    self.assertEqual(
                        len(metric.issue_locations),
                        issue_count,
                    )
        for raw_value in ("invalid", "https://bad url"):
            self.assertNotIn(raw_value, payload)

    def test_report_download_is_human_readable_utf8_markdown(self):
        payload = serialize_markdown_report(self.report).decode("utf-8")
        self.assertIn("# 数据集质量评估报告：bad_dataset", payload)
        self.assertIn("## 风险提示", payload)
        self.assertNotIn("## 疑似问题位置", payload)
        self.assertNotIn("数据记录序号", payload)
        self.assertIn("## 指标明细", payload)
        self.assertIn("## 字段画像", payload)
        self.assertIn("| 指标名称 | 字段名称 |", payload)
        self.assertNotIn("| 引用键 |", payload)
        self.assertIn("bad_dataset.csv", payload)

    def test_structured_report_download_is_strict_json(self):
        payload = serialize_report(self.report).decode("utf-8")
        structured = json.loads(
            payload,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )

        self.assertEqual(structured, self.report.to_dict())
        self.assertNotIn("issue_locations", payload)
        self.assertTrue(
            all(
                column["non_null_samples"] == []
                for column in structured["profile"]["columns"]
            )
        )

    def test_markdown_report_escapes_untrusted_link_and_image_syntax(self):
        malicious_text = "![x](https://example.invalid/pixel.png)"
        report = evaluate_uploaded_dataset(
            f"{malicious_text},other\n,1\n,2\n".encode("utf-8"),
            "untrusted.csv",
            dataset_name=malicious_text,
        )

        payload = serialize_markdown_report(report).decode("utf-8")

        self.assertNotIn(malicious_text, payload)
        self.assertIn(r"\!\[x\]\(https://example.invalid/pixel.png\)", payload)


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
