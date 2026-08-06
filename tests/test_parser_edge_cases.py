"""解析器与上传边界的回归测试。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd

from src.models import DatasetInfo, QualityReport
from src.parser import DatasetReadError, parse_dataset
from src.presentation import serialize_report
from src.profiler import is_missing_value
from src.upload_service import evaluate_uploaded_dataset, sanitize_file_name
from src.workflow import build_profile_report


class ParserEdgeCaseTests(unittest.TestCase):
    def test_corrupt_excel_variants_return_explainable_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corrupt_xls_path = Path(temp_dir) / "corrupt.xls"
            corrupt_xls_path.write_bytes(b"not-an-ole-workbook")

            truncated_path = Path(temp_dir) / "truncated.xlsx"
            truncated_path.write_bytes(b"PK\x03\x04garbage")

            zip_only_path = Path(temp_dir) / "zip-only.xlsx"
            with ZipFile(zip_only_path, "w") as archive:
                archive.writestr("not-a-workbook.txt", "broken")

            for path in (corrupt_xls_path, truncated_path, zip_only_path):
                with self.subTest(path=path.name):
                    with self.assertRaisesRegex(DatasetReadError, "Excel 文件无法打开"):
                        parse_dataset(path)

                    report = build_profile_report(path)
                    self.assertEqual(report.status, "failed")
                    self.assertIn("Excel 文件无法打开", report.execution["errors"][0])
                    json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False)

    def test_excessively_deep_json_is_a_dataset_read_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "deep.json"
            path.write_text(
                '{"value":' * 10_000 + "1" + "}" * 10_000,
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DatasetReadError, "嵌套层级过深"):
                parse_dataset(path)

            report = build_profile_report(path)
            self.assertEqual(report.status, "failed")
            self.assertIn("嵌套层级过深", report.execution["errors"][0])
            json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False)

    def test_oversized_json_integer_is_a_dataset_read_error(self):
        get_limit = getattr(sys, "get_int_max_str_digits", None)
        set_limit = getattr(sys, "set_int_max_str_digits", None)
        previous_limit = get_limit() if get_limit is not None else None
        if set_limit is not None:
            # 模拟 Python 3.10 没有解释器级整数位数限制的行为。
            set_limit(0)
        try:
            for digit_count in (400, 5000):
                with self.subTest(digit_count=digit_count):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        path = Path(temp_dir) / "huge-integer.json"
                        path.write_text(
                            '{"value":' + "9" * digit_count + "}",
                            encoding="utf-8",
                        )

                        with self.assertRaisesRegex(
                            DatasetReadError,
                            "JSON 内容无法读取",
                        ):
                            parse_dataset(path)

                        report = build_profile_report(path)
                        self.assertEqual(report.status, "failed")
                        self.assertIn(
                            "JSON 内容无法读取",
                            report.execution["errors"][0],
                        )
                        json.dumps(
                            report.to_dict(),
                            ensure_ascii=False,
                            allow_nan=False,
                        )
        finally:
            if set_limit is not None and previous_limit is not None:
                set_limit(previous_limit)

    def test_isolated_json_surrogate_produces_serializable_failed_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "surrogate.json"
            path.write_bytes(b'[{"value":"\\ud800"}]')

            with self.assertRaisesRegex(DatasetReadError, "Unicode 代理字符"):
                parse_dataset(path)

            report = build_profile_report(path)
            self.assertEqual(report.status, "failed")
            json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False)

    def test_valid_surrogate_pair_is_normalized_to_unicode_scalar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "emoji.json"
            path.write_bytes(b'[{"value":"\\ud83d\\ude00"}]')

            parsed = parse_dataset(path)

        self.assertEqual(parsed.dataframe.loc[0, "value"], "😀")

    def test_columns_that_collide_after_stringification_are_rejected(self):
        dataframe = pd.DataFrame([[1, 2]], columns=[1, "1"])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "columns.csv"
            path.write_text("placeholder\n", encoding="utf-8")
            with patch("src.parser._read_csv", return_value=(dataframe, [])):
                with self.assertRaisesRegex(DatasetReadError, "字段名转换为文本后存在重复"):
                    parse_dataset(path)
                report = build_profile_report(path)

        self.assertEqual(report.status, "failed")
        self.assertIn("字段名转换为文本后存在重复", report.execution["errors"][0])

    def test_csv_duplicate_original_headers_are_rejected_before_pandas_mangling(self):
        for leading_lines in ("", "\n", " \t\r\n"):
            with self.subTest(leading_lines=repr(leading_lines)):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "duplicate-columns.csv"
                    path.write_text(
                        f"{leading_lines}name,name\nA,B\n",
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(
                        DatasetReadError,
                        "CSV 原始表头存在重复字段",
                    ):
                        parse_dataset(path)
                    report = build_profile_report(path)

                self.assertEqual(report.status, "failed")
                self.assertIn(
                    "CSV 原始表头存在重复字段：name",
                    report.execution["errors"][0],
                )
                json.loads(serialize_report(report).decode("utf-8"))

        # 这些 Unicode/控制空白字符不会被 pandas 当作空白行跳过，
        # 原始表头预检也必须保留它们，不能误检后续记录为重复表头。
        for retained_header in ("\u00a0", "\u3000", "\x0b", "\x0c"):
            with self.subTest(retained_header=repr(retained_header)):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "retained-header.csv"
                    path.write_text(
                        f'{retained_header}\n"name,name"\nA\n',
                        encoding="utf-8",
                    )
                    parsed = parse_dataset(path)

                self.assertEqual(parsed.dataframe.columns.tolist(), [retained_header])
                self.assertEqual(parsed.dataframe.iloc[0, 0], "name,name")

        for nul_content in (b"a\x00,a\n1,2\n", b"a,b\nx\x00y,2\n"):
            with self.subTest(nul_content=nul_content):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "nul.csv"
                    path.write_bytes(nul_content)

                    with self.assertRaisesRegex(DatasetReadError, "NUL 空字符"):
                        parse_dataset(path)
                    report = build_profile_report(path)

                self.assertEqual(report.status, "failed")
                self.assertIn("NUL 空字符", report.execution["errors"][0])
                json.loads(serialize_report(report).decode("utf-8"))

    def test_csv_accepts_leading_blanks_and_multiline_quoted_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "multiline.csv"
            path.write_text(
                '\n \t\r\nname,notes\nA,"first line\nsecond line"\n\nB,ok\n',
                encoding="utf-8",
            )

            parsed = parse_dataset(path)

        self.assertEqual(parsed.dataframe.columns.tolist(), ["name", "notes"])
        self.assertEqual(parsed.dataframe["name"].tolist(), ["A", "B"])
        self.assertEqual(
            parsed.dataframe["notes"].tolist(),
            ["first line\nsecond line", "ok"],
        )

    def test_csv_rejects_wider_logical_record_with_line_and_field_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ragged.csv"
            path.write_text(
                '\n \t\r\nname,notes\nA,"first line\nsecond line"\n\nB,ok,extra\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                DatasetReadError,
                r"第 7 行包含 3 个字段.*表头的 2 个字段",
            ):
                parse_dataset(path)
            report = build_profile_report(path)

        self.assertEqual(report.status, "failed")
        self.assertIn("第 7 行包含 3 个字段", report.execution["errors"][0])
        json.loads(serialize_report(report).decode("utf-8"))

    def test_csv_excel_and_json_preserve_na_like_text_consistently(self):
        records = [
            {"id": 1, "value": "NA"},
            {"id": 2, "value": "NULL"},
            {"id": 3, "value": "N/A"},
            {"id": 4, "value": None},
        ]
        expected_literal_values = ["NA", "NULL", "N/A"]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "equivalent.csv"
            excel_path = temp_path / "equivalent.xlsx"
            json_path = temp_path / "equivalent.json"

            csv_path.write_text(
                "id,value\n1,NA\n2,NULL\n3,N/A\n4,\n",
                encoding="utf-8",
            )
            pd.DataFrame(records).to_excel(excel_path, index=False)
            json_path.write_text(
                json.dumps(records, ensure_ascii=False),
                encoding="utf-8",
            )

            parsed_datasets = [
                parse_dataset(path)
                for path in (csv_path, excel_path, json_path)
            ]
            reports = [
                build_profile_report(path)
                for path in (csv_path, excel_path, json_path)
            ]

        for parsed in parsed_datasets:
            with self.subTest(file_type=parsed.dataset.file_type):
                values = parsed.dataframe["value"].tolist()
                self.assertEqual(values[:3], expected_literal_values)
                self.assertTrue(is_missing_value(values[3]))

        for report in reports:
            with self.subTest(file_type=report.dataset.file_type):
                value_profile = next(
                    column
                    for column in report.profile["columns"]
                    if column["name"] == "value"
                )
                self.assertEqual(value_profile["missing_count"], 1)
                self.assertEqual(value_profile["missing_rate"], 0.25)

    def test_excel_duplicate_original_headers_are_rejected_before_pandas_mangling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "duplicate-columns.xlsx"
            pd.DataFrame([["A", "B"]], columns=["name", "name"]).to_excel(
                path,
                index=False,
                sheet_name="数据",
            )

            with self.assertRaisesRegex(
                DatasetReadError, "Excel 原始表头存在重复字段"
            ):
                parse_dataset(path)
            report = build_profile_report(path)

        self.assertEqual(report.status, "failed")
        self.assertIn("Excel 原始表头存在重复字段：name", report.execution["errors"][0])
        json.loads(serialize_report(report).decode("utf-8"))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mixed-header-types.xlsx"
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                pd.DataFrame([["A", "B"]], columns=[1, "1.0"]).to_excel(
                    writer,
                    index=False,
                )
            parsed = parse_dataset(path)

        self.assertEqual(parsed.dataframe.columns.tolist(), ["1", "1.0"])

    def test_excel_na_like_headers_remain_literal_field_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "na-like-headers.xlsx"
            pd.DataFrame(
                [["A", "B", "C"]],
                columns=["NA", "NULL", "N/A"],
            ).to_excel(path, index=False)

            parsed = parse_dataset(path)

        self.assertEqual(parsed.dataframe.columns.tolist(), ["NA", "NULL", "N/A"])
        self.assertEqual(parsed.dataframe.iloc[0].tolist(), ["A", "B", "C"])

    def test_json_duplicate_keys_are_rejected_instead_of_overwriting_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "duplicate-keys.json"
            path.write_text('[{"name":"A","name":"B"}]', encoding="utf-8")

            with self.assertRaisesRegex(DatasetReadError, "JSON 对象存在重复字段"):
                parse_dataset(path)
            report = build_profile_report(path)

        self.assertEqual(report.status, "failed")
        self.assertIn("JSON 对象存在重复字段：name", report.execution["errors"][0])
        json.loads(serialize_report(report).decode("utf-8"))

    def test_json_nonstandard_numeric_constants_are_rejected(self):
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "nonstandard-number.json"
                    path.write_text(
                        f'[{{"value":{constant}}}]',
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(
                        DatasetReadError, "JSON 包含非标准数值"
                    ):
                        parse_dataset(path)
                    report = build_profile_report(path)

                self.assertEqual(report.status, "failed")
                self.assertIn(constant, report.execution["errors"][0])
                json.loads(serialize_report(report).decode("utf-8"))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "overflow-number.json"
            path.write_text('[{"value":1e9999}]', encoding="utf-8")

            with self.assertRaisesRegex(DatasetReadError, "有限浮点数范围"):
                parse_dataset(path)
            report = build_profile_report(path)

        self.assertEqual(report.status, "failed")
        self.assertIn("有限浮点数范围", report.execution["errors"][0])
        json.loads(serialize_report(report).decode("utf-8"))

    def test_isolated_surrogate_in_sheet_name_is_replaced_and_warned(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sheet-name.xlsx"
            pd.DataFrame({"name": ["A"]}).to_excel(
                path,
                index=False,
                sheet_name="数据",
            )
            report = build_profile_report(path, sheet_name="\ud800")

        self.assertEqual(report.status, "failed")
        self.assertIn("未找到工作表“�”", report.execution["errors"][0])
        self.assertTrue(
            any("工作表名称" in warning for warning in report.execution["warnings"])
        )
        json.loads(serialize_report(report).decode("utf-8"))


class UploadBoundaryTests(unittest.TestCase):
    def test_sanitizer_handles_windows_paths_reserved_names_and_limits(self):
        self.assertEqual(
            sanitize_file_name(
                r"C:\incoming\CON.csv",
                safe_extension=".csv",
            ),
            "_CON.csv",
        )

        safe_name = sanitize_file_name(
            "数据" * 100 + "\x00?.json",
            default_name="quality_report.json",
            safe_extension=".json",
        )
        self.assertLessEqual(len(safe_name), 120)
        self.assertTrue(safe_name.endswith(".json"))
        self.assertNotIn("\x00", safe_name)
        self.assertNotIn("?", safe_name)

        download_name = sanitize_file_name(
            "bad\r\nname_quality_report.json",
            default_name="quality_report.json",
            safe_extension=".json",
        )
        self.assertNotIn("\r", download_name)
        self.assertNotIn("\n", download_name)
        self.assertEqual(download_name, "bad__name_quality_report.json")

        surrogate_name = sanitize_file_name(
            chr(0xD800) + ".json",
            safe_extension=".json",
        )
        self.assertNotIn(chr(0xD800), surrogate_name)
        surrogate_name.encode("utf-8")

    def test_uploaded_report_keeps_safe_display_name(self):
        report = evaluate_uploaded_dataset(
            b"record_id,name\n1,test\n",
            "C:\\incoming\\bad\x00name?.csv",
        )

        self.assertEqual(report.status, "success")
        self.assertEqual(report.dataset.file_name, "bad_name_.csv")
        self.assertEqual(report.dataset.name, "bad_name_")

        unicode_report = evaluate_uploaded_dataset(
            b"record_id,name\n1,test\n",
            "\ud800.csv",
        )
        self.assertEqual(unicode_report.dataset.file_name, "�.csv")
        self.assertTrue(
            any(
                "上传文件名" in warning
                for warning in unicode_report.execution["warnings"]
            )
        )
        json.loads(serialize_report(unicode_report).decode("utf-8"))

        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing_\ud800.xlsx"
            failed_report = build_profile_report(
                missing_path,
                sheet_name="\ud800",
            )

        self.assertEqual(failed_report.status, "failed")
        self.assertEqual(failed_report.dataset.name, "missing_�")
        self.assertEqual(failed_report.dataset.file_name, "missing_�.xlsx")
        self.assertNotIn("\ud800", failed_report.execution["errors"][0])
        self.assertTrue(
            any(
                "文件显示名称" in warning
                for warning in failed_report.execution["warnings"]
            )
        )
        self.assertTrue(
            any(
                "工作表名称" in warning
                for warning in failed_report.execution["warnings"]
            )
        )
        json.loads(serialize_report(failed_report).decode("utf-8"))

        unicode_report = evaluate_uploaded_dataset(
            b"record_id,name\n1,test\n",
            "dataset.csv",
            dataset_name="\ud800",
        )
        self.assertEqual(unicode_report.dataset.name, "�")
        self.assertIn("Unicode 替代字符", unicode_report.execution["warnings"][0])
        json.loads(serialize_report(unicode_report).decode("utf-8"))

        workflow_report = build_profile_report(
            Path(__file__).parents[1] / "sample_data" / "good_dataset.csv",
            dataset_name="\ud800",
        )
        self.assertEqual(workflow_report.dataset.name, "�")
        self.assertIn("Unicode 替代字符", workflow_report.execution["warnings"][0])
        json.loads(serialize_report(workflow_report).decode("utf-8"))

    def test_uploaded_content_uses_fixed_temporary_file_name(self):
        dummy_report = QualityReport(
            dataset=DatasetInfo(
                name="temporary",
                file_name="temporary.csv",
                file_type="csv",
            )
        )
        with patch(
            "src.upload_service.build_profile_report",
            return_value=dummy_report,
        ) as build_report:
            report = evaluate_uploaded_dataset(
                b"record_id\n1\n",
                "folder/user-controlled.csv",
            )

        temporary_path = build_report.call_args.args[0]
        self.assertEqual(temporary_path.name, "upload.csv")
        self.assertEqual(report.dataset.file_name, "user-controlled.csv")


if __name__ == "__main__":
    unittest.main()
