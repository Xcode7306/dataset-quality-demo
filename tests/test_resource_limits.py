"""输入规模与 Excel 压缩资源边界测试。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import openpyxl.xml

from src.parser import DatasetReadError, parse_dataset
from src.resource_limits import (
    ResourceLimitExceeded,
    validate_dataframe_limits,
    validate_input_file_size,
    validate_upload_size,
    validate_xlsx_archive,
)
from src.upload_service import evaluate_uploaded_dataset
from src.workflow import build_profile_report


class ResourceLimitTests(unittest.TestCase):
    def test_openpyxl_uses_defusedxml(self):
        self.assertTrue(openpyxl.xml.DEFUSEDXML)

    def test_file_and_upload_size_limits_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "large.csv"
            path.write_bytes(b"1234")
            with patch("src.resource_limits.MAX_INPUT_FILE_BYTES", 3):
                with self.assertRaisesRegex(ResourceLimitExceeded, "\u8d85\u8fc7"):
                    validate_input_file_size(path)
                with self.assertRaisesRegex(ResourceLimitExceeded, "\u8d85\u8fc7"):
                    validate_upload_size(4)

    def test_xlsx_entry_expansion_and_compression_limits_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "archive.xlsx"
            with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr("xl/a.xml", b"0" * 4096)
                archive.writestr("xl/b.xml", b"1")

            with (
                patch("src.resource_limits.MAX_XLSX_ENTRY_COUNT", 1),
                self.assertRaisesRegex(ResourceLimitExceeded, "\u6761\u76ee"),
            ):
                validate_xlsx_archive(path)

            with (
                patch("src.resource_limits.MAX_XLSX_UNCOMPRESSED_BYTES", 100),
                self.assertRaisesRegex(ResourceLimitExceeded, "\u89e3\u538b\u540e"),
            ):
                validate_xlsx_archive(path)

            with (
                patch("src.resource_limits.MIN_RATIO_CHECK_BYTES", 1),
                patch("src.resource_limits.MAX_XLSX_COMPRESSION_RATIO", 2),
                self.assertRaisesRegex(ResourceLimitExceeded, "\u538b\u7f29\u6bd4"),
            ):
                validate_xlsx_archive(path)

    def test_dataframe_shape_and_cell_limits_are_enforced(self):
        with (
            patch("src.resource_limits.MAX_DATASET_ROWS", 1),
            self.assertRaisesRegex(ResourceLimitExceeded, "\u8bb0\u5f55"),
        ):
            validate_dataframe_limits(pd.DataFrame({"value": [1, 2]}))

        with (
            patch("src.resource_limits.MAX_DATASET_COLUMNS", 1),
            self.assertRaisesRegex(ResourceLimitExceeded, "\u5b57\u6bb5"),
        ):
            validate_dataframe_limits(pd.DataFrame({"a": [1], "b": [2]}))

        with (
            patch("src.resource_limits.MAX_CELL_TEXT_BYTES", 3),
            self.assertRaisesRegex(ResourceLimitExceeded, "\u6587\u672c"),
        ):
            validate_dataframe_limits(pd.DataFrame({"value": ["\u4e2d\u56fd"]}))

        with (
            patch("src.resource_limits.MAX_DATASET_CELLS", 3),
            self.assertRaisesRegex(ResourceLimitExceeded, "\u5355\u5143\u683c"),
        ):
            validate_dataframe_limits(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    def test_parser_converts_resource_limit_to_downloadable_failed_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "large.csv"
            path.write_text("value\n1234\n", encoding="utf-8")
            with patch("src.resource_limits.MAX_INPUT_FILE_BYTES", 3):
                report = build_profile_report(path)

            too_many_records = root / "too-many-records.json"
            too_many_records.write_text("[{},{},{}]", encoding="utf-8")
            with (
                patch("src.parser.MAX_JSON_RECORDS", 2),
                patch("src.parser.json.load") as json_load,
                self.assertRaisesRegex(DatasetReadError, "\u8bb0\u5f55"),
            ):
                parse_dataset(too_many_records)
            json_load.assert_not_called()

            nested = root / "nested.json"
            nested.write_text('[{"value":[1,2,3]}]', encoding="utf-8")
            with (
                patch("src.parser.json.load") as json_load,
                self.assertRaisesRegex(DatasetReadError, "\u5d4c\u5957"),
            ):
                parse_dataset(nested)
            json_load.assert_not_called()

            wide_object = root / "wide-object.json"
            wide_object.write_text('{"a":1,"b":2}', encoding="utf-8")
            with (
                patch("src.parser.MAX_JSON_OBJECT_PAIRS", 1),
                patch("src.parser.json.load") as json_load,
                self.assertRaisesRegex(DatasetReadError, "单个对象.*键值对"),
            ):
                parse_dataset(wide_object)
            json_load.assert_not_called()

            many_pairs = root / "many-pairs.json"
            many_pairs.write_text('[{"a":1},{"b":2}]', encoding="utf-8")
            with (
                patch("src.parser.MAX_JSON_TOTAL_PAIRS", 1),
                patch("src.parser.json.load") as json_load,
                self.assertRaisesRegex(DatasetReadError, "全文件.*键值对"),
            ):
                parse_dataset(many_pairs)
            json_load.assert_not_called()

            sparse = root / "sparse.json"
            sparse.write_text('[{"a":1},{"b":2}]', encoding="utf-8")
            with (
                patch("src.parser.MAX_DATASET_CELLS", 3),
                patch("src.parser.pd.DataFrame") as dataframe_constructor,
                self.assertRaisesRegex(DatasetReadError, "\u5355\u5143\u683c"),
            ):
                parse_dataset(sparse)
            dataframe_constructor.assert_not_called()

        self.assertEqual(report.status, "failed")
        self.assertIn("超过本地 Demo", report.execution["errors"][0])
        self.assertEqual(len({metric.id for metric in report.metrics}), 13)

    def test_upload_size_is_rejected_before_temporary_file_creation(self):
        with patch("src.resource_limits.MAX_INPUT_FILE_BYTES", 3):
            with self.assertRaisesRegex(DatasetReadError, "上传内容.*超过"):
                evaluate_uploaded_dataset(b"1234", "large.csv")


if __name__ == "__main__":
    unittest.main()
