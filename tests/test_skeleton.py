"""阶段 0 的最小自动化检查。"""

import tempfile
import unittest
from pathlib import Path

from src.metric_catalog import ALL_METRIC_IDS, ORIGINAL_METRIC_IDS
from src.metrics import METRIC_CALCULATORS, METRIC_CATALOG, calculate_failed_metrics
from src.models import DatasetInfo
from src.parser import UnsupportedFileTypeError, validate_file_type
from src.report import create_empty_report, save_report


class SkeletonTests(unittest.TestCase):
    def test_metric_catalog_contains_43_unique_metrics(self):
        self.assertEqual(len(METRIC_CATALOG), 43)
        self.assertEqual(len({item["id"] for item in METRIC_CATALOG}), 43)

    def test_metric_registry_matches_catalog(self):
        catalog_ids = [item["id"] for item in METRIC_CATALOG]
        registry_ids = [metric_id for metric_id, _ in METRIC_CALCULATORS]
        self.assertEqual(registry_ids, catalog_ids)

    def test_failed_metric_report_is_derived_from_catalog(self):
        catalog_ids = [item["id"] for item in METRIC_CATALOG]
        failed_ids = [
            metric.id
            for metric in calculate_failed_metrics(
                "测试失败",
                selected_metric_ids=ALL_METRIC_IDS,
            )
        ]
        self.assertEqual(failed_ids, catalog_ids)

    def test_failed_metric_default_preserves_v04_selection(self):
        failed_ids = [
            metric.id for metric in calculate_failed_metrics("测试失败")
        ]
        self.assertEqual(failed_ids, list(ORIGINAL_METRIC_IDS))

    def test_supported_file_type_is_accepted(self):
        with tempfile.NamedTemporaryFile(suffix=".csv") as file:
            path = validate_file_type(file.name)
        self.assertEqual(path.suffix, ".csv")

    def test_unsupported_file_type_is_rejected(self):
        with self.assertRaises(UnsupportedFileTypeError):
            validate_file_type("example.pdf")

    def test_empty_report_can_be_saved_as_markdown(self):
        dataset = DatasetInfo(
            name="测试数据集",
            file_name="example.csv",
            file_type="csv",
        )
        report = create_empty_report(dataset)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "report.md"
            save_report(report, output)
            saved = output.read_text(encoding="utf-8")

        self.assertIn("# 数据集质量评估报告：测试数据集", saved)
        self.assertIn("当前报告不包含指标明细。", saved)


if __name__ == "__main__":
    unittest.main()
