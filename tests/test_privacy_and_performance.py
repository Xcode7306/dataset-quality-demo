"""报告隐私与大规模覆盖率计算的回归测试。"""

import unittest

import pandas as pd

from src.metrics import (
    calculate_all_metrics,
    calculate_source_info_coverage,
    calculate_version_info_coverage,
)
from src.models import DatasetInfo, QualityReport
from src.presentation import serialize_report
from src.profiler import profile_dataframe


class PrivacyAndPerformanceTests(unittest.TestCase):
    def test_serialized_report_does_not_contain_raw_field_or_anomaly_samples(self):
        dataframe = pd.DataFrame(
            {
                "姓名": ["张三", "李四", "王五", "赵六", "陈七"],
                "身份证号": [
                    "110101199001011234",
                    "110101199002022345",
                    "110101199003033456",
                    "110101199004044567",
                    "110101199005055678",
                ],
                "邮箱": [
                    "zhangsan@example.gov.cn",
                    "lisi@example.gov.cn",
                    "private-invalid-email",
                    "zhaoliu@example.gov.cn",
                    "chenqi@example.gov.cn",
                ],
                "金额": [10, 11, 12, 13, 999999],
            }
        )
        profile = profile_dataframe(dataframe)
        metrics = calculate_all_metrics(dataframe)
        report = QualityReport(
            dataset=DatasetInfo(
                name="privacy-test",
                file_name="privacy-test.csv",
                file_type="csv",
            ),
            profile=profile,
            metrics=metrics,
        )

        serialized = serialize_report(report).decode("utf-8")

        self.assertNotIn("issue_locations", serialized)
        self.assertTrue(
            all(column["non_null_samples"] == [] for column in profile["columns"])
        )
        format_metric = next(
            metric
            for metric in metrics
            if metric.id == "recognizable_format_anomaly_rate"
            and metric.field == "邮箱"
        )
        self.assertEqual(format_metric.evidence["invalid_samples"], [])
        outlier_metric = next(
            metric
            for metric in metrics
            if metric.id == "statistical_outlier_rate" and metric.field == "金额"
        )
        self.assertEqual(outlier_metric.evidence["outlier_samples"], [])
        self.assertEqual(outlier_metric.evidence["non_finite_samples"], [])
        for private_value in dataframe[["姓名", "身份证号", "邮箱"]].to_numpy().flat:
            with self.subTest(private_value=private_value):
                self.assertNotIn(str(private_value), serialized)

    def test_large_coverage_calculation_keeps_counts_and_first_twenty_indices(self):
        # 10 万行回归用例不依赖机器耗时阈值；它验证线性实现
        # 在大规模数据上仍能正确计数，并完整保留独立 CSV 所需位置。
        row_count = 100_000
        missing_positions = set(range(0, row_count, 10))
        dataframe = pd.DataFrame(
            {
                "data_source": [
                    None if index in missing_positions else "department"
                    for index in range(row_count)
                ],
                "version": [
                    None if index in missing_positions else "v1.0"
                    for index in range(row_count)
                ],
            }
        )

        for metric in (
            calculate_source_info_coverage(dataframe),
            calculate_version_info_coverage(dataframe),
        ):
            with self.subTest(metric=metric.id):
                self.assertEqual(metric.value, 0.9)
                self.assertEqual(metric.evidence["covered_count"], 90_000)
                self.assertEqual(metric.evidence["issue_count"], 10_000)
                self.assertEqual(
                    metric.evidence["missing_row_indices"],
                    list(range(1, 192, 10)),
                )
                self.assertEqual(
                    len(metric.issue_locations),
                    10_000,
                )
                self.assertEqual(
                    metric.issue_locations[0]["record_number"],
                    1,
                )
                self.assertEqual(
                    metric.issue_locations[-1]["record_number"],
                    99_991,
                )


if __name__ == "__main__":
    unittest.main()
