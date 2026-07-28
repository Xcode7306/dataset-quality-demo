"""JSON 读取扩展阶段 A 的结构、编码与严格边界回归。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.parser import DatasetReadError, parse_dataset
from src.upload_service import evaluate_uploaded_dataset


class JsonFormatExpansionTests(unittest.TestCase):
    def test_json_integer_values_are_not_silently_rounded_by_dataframe_inference(self):
        large_values = [9007199254740992, 9007199254740993, None]
        payloads = (
            [{"amount": value} for value in large_values],
            [["amount"], *[[value] for value in large_values]],
        )
        for payload in payloads:
            with self.subTest(shape=type(payload[0]).__name__):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "large-integers.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    parsed = parse_dataset(path)

                self.assertEqual(parsed.dataframe["amount"].tolist(), large_values)
                self.assertEqual(
                    [type(value) for value in parsed.dataframe["amount"][:2]],
                    [int, int],
                )

    def test_matrix_json_uses_first_row_as_header_and_pads_short_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "wenzhou.json"
            path.write_text(
                json.dumps(
                    [["编号", "名称", "状态"], [1, "A", "有效"], [2, "B"]],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            parsed = parse_dataset(path)

        self.assertEqual(parsed.dataframe.columns.tolist(), ["编号", "名称", "状态"])
        self.assertEqual(parsed.dataframe.iloc[0].tolist(), [1, "A", "有效"])
        self.assertTrue(parsed.dataframe.isna().iloc[1, 2])
        self.assertTrue(any("二维数组" in warning for warning in parsed.warnings))

    def test_matrix_rejects_wide_rows_duplicate_headers_and_nested_cells(self):
        cases = (
            ([["a"], [1, 2]], "超过表头"),
            ([["a", "a"], [1, 2]], "重复字段"),
            ([["a"], [[1, 2]]], "嵌套"),
        )
        for payload, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "invalid-matrix.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(DatasetReadError, message):
                        parse_dataset(path)

    def test_common_api_wrappers_extract_unique_records_or_matrix(self):
        payloads = (
            (
                {"code": 1, "data": {"rows": [{"id": 1, "name": "A"}]}},
                ["id", "name"],
                [1, "A"],
                "$.data.rows",
            ),
            (
                {"code": "1", "msg": "成功", "data": [["所属", "主键"], ["市级", 7]]},
                ["所属", "主键"],
                ["市级", 7],
                "$.data",
            ),
            (
                {"result": {"record_id": 9, "name": "单条"}},
                ["record_id", "name"],
                [9, "单条"],
                "$.result",
            ),
        )
        for payload, columns, row, path_text in payloads:
            with self.subTest(path=path_text):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "wrapped.json"
                    path.write_text(
                        json.dumps(payload, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    parsed = parse_dataset(path)

                self.assertEqual(parsed.dataframe.columns.tolist(), columns)
                self.assertEqual(parsed.dataframe.iloc[0].tolist(), row)
                self.assertTrue(
                    any(path_text in warning for warning in parsed.warnings)
                )

    def test_api_wrapper_rejects_ambiguous_or_unrecognized_nested_data(self):
        cases = (
            (
                {"data": [{"id": 1}], "rows": [["id"], [2]]},
                "多个表格候选",
            ),
            ({"payload": {"id": 1}}, "未在常见接口包装"),
            ({"data": [{"id": 1, "meta": {"x": 2}}]}, "嵌套"),
        )
        for payload, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "wrapped.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(DatasetReadError, message):
                        parse_dataset(path)

    def test_ambiguous_wrapper_error_is_bounded(self):
        wrapper_keys = ("data", "rows", "records", "items", "list", "result")
        payload = {key: [{"id": 1}] for key in wrapper_keys}
        for _ in range(4):
            payload = {key: payload for key in wrapper_keys}

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "many-wrapper-candidates.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(DatasetReadError) as raised:
                parse_dataset(path)

        message = str(raised.exception)
        self.assertIn("多个表格候选", message)
        self.assertLess(len(message), 1_000)

    def test_jsonl_and_ndjson_read_one_flat_object_per_nonblank_line(self):
        content = '{"id":1,"name":"A"}\n\n{"id":2,"name":"B"}\n'
        for extension in ("jsonl", "ndjson"):
            with self.subTest(extension=extension):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / f"records.{extension}"
                    path.write_text(content, encoding="utf-8")
                    parsed = parse_dataset(path)

                self.assertEqual(parsed.dataframe["id"].tolist(), [1, 2])
                self.assertEqual(parsed.dataset.file_type, extension)
                self.assertTrue(any("JSON Lines" in item for item in parsed.warnings))

    def test_jsonl_reports_line_number_and_preserves_strict_json_rules(self):
        cases = (
            ('{"id":1}\n{"id":2,}\n', "第 2 行格式错误"),
            ('{"id":1}\n[2]\n', "第 2 行必须是"),
            ('{"id":1}\n{"id":2,"id":3}\n', "第 2 行无法读取"),
            ('{"id":1}\n{"value":NaN}\n', "第 2 行无法读取"),
            ('{"id":1}\n{"meta":{"x":2}}\n', "第 2 行包含嵌套"),
        )
        for content, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "records.jsonl"
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(DatasetReadError, message):
                        parse_dataset(path)

    def test_jsonl_only_skips_rfc_json_whitespace_lines(self):
        for invalid_whitespace in ("\v", "\f", "\u00a0"):
            with self.subTest(codepoint=ord(invalid_whitespace)):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "strict-whitespace.jsonl"
                    path.write_text(
                        '{"id":1}\n' + invalid_whitespace + '\n{"id":2}\n',
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        DatasetReadError,
                        "JSON Lines 第 2 行格式错误",
                    ):
                        parse_dataset(path)

    def test_jsonl_resource_errors_include_the_source_line(self):
        cases = (
            (
                '{"id":1}\n{"nested":{"deeper":{"value":2}}}\n',
                {"MAX_JSON_NESTING_DEPTH": 2},
            ),
            (
                '{"id":1}\n{"a":1,"b":2}\n',
                {"MAX_JSON_OBJECT_PAIRS": 1},
            ),
            (
                '{"a":1}\n{"b":2}\n',
                {"MAX_DATASET_COLUMNS": 1},
            ),
        )
        for content, limits in cases:
            with self.subTest(limits=limits):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "line-limit.jsonl"
                    path.write_text(content, encoding="utf-8")
                    patches = [patch(f"src.parser.{name}", value) for name, value in limits.items()]
                    for active_patch in patches:
                        active_patch.start()
                        self.addCleanup(active_patch.stop)
                    try:
                        with self.assertRaisesRegex(
                            DatasetReadError,
                            "JSON Lines 第 2 行",
                        ):
                            parse_dataset(path)
                    finally:
                        for active_patch in reversed(patches):
                            active_patch.stop()

    def test_targeted_outer_brace_repair_is_narrow_and_warned(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "shenzhen.json"
            path.write_text('{[{"SW":"1.20","ID":"A1"}]}', encoding="utf-8")
            parsed = parse_dataset(path)

        self.assertEqual(parsed.dataframe.loc[0, "ID"], "A1")
        self.assertTrue(any("原文件不是标准 JSON" in item for item in parsed.warnings))

        invalid_payloads = (
            '[{"id":1,}]',
            "{'id':1}",
            '// comment\n[{"id":1}]',
            '{[{"id":1,}]}',
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "invalid.json"
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaisesRegex(DatasetReadError, "JSON 格式错误"):
                        parse_dataset(path)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "duplicate-key-repair.json"
            path.write_text('{[{"id":1,"id":2}]}', encoding="utf-8")
            with self.assertRaisesRegex(DatasetReadError, "重复字段"):
                parse_dataset(path)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "spaced-repair.json"
            path.write_text('{ \n [{"id":1}] \n }', encoding="utf-8")
            parsed = parse_dataset(path)
            self.assertEqual(parsed.dataframe.loc[0, "id"], 1)
            self.assertTrue(any("原文件不是标准 JSON" in item for item in parsed.warnings))

    def test_all_wrapper_text_is_unicode_validated_before_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "wrapper-surrogate.json"
            path.write_bytes(b'{"meta":"\\ud800","data":[{"id":1}]}')
            with self.assertRaisesRegex(DatasetReadError, "孤立.*代理字符"):
                parse_dataset(path)

    def test_automatic_processing_warnings_survive_later_table_failures(self):
        cases = (
            (
                '{[{"id":1,"meta":{"x":2}}]}',
                "原文件不是标准 JSON",
            ),
            (
                '{"data":[{"id":1,"meta":{"x":2}}]}',
                "JSON 接口包装",
            ),
            (
                '{"data":[["id","id"],[1,2]]}',
                "二维数组",
            ),
        )
        for content, warning_text in cases:
            with self.subTest(warning=warning_text):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "warned-failure.json"
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(DatasetReadError) as raised:
                        parse_dataset(path)
                self.assertTrue(
                    any(warning_text in item for item in raised.exception.warnings),
                    raised.exception.warnings,
                )

    def test_oversized_duplicate_matrix_headers_do_not_amplify_error_reports(self):
        oversized_header = "x" * 32
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "oversized-header.json"
            path.write_text(
                json.dumps([[oversized_header, oversized_header], [1, 2]]),
                encoding="utf-8",
            )
            with (
                patch("src.parser.MAX_CELL_TEXT_BYTES", 16),
                self.assertRaises(DatasetReadError) as raised,
            ):
                parse_dataset(path)

        self.assertIn("单项文本", str(raised.exception))
        self.assertNotIn(oversized_header, str(raised.exception))
        self.assertLess(len(str(raised.exception)), 500)

    def test_utf16_and_gb18030_json_are_warned_and_supported(self):
        text = json.dumps([["编号", "名称"], [1, "中文"]], ensure_ascii=False)
        cases = (
            ("utf16.json", text.encode("utf-16"), "UTF-16 BOM"),
            ("gb18030.json", text.encode("gb18030"), "GB18030"),
        )
        for file_name, content, warning_text in cases:
            with self.subTest(file_name=file_name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / file_name
                    path.write_bytes(content)
                    parsed = parse_dataset(path)

                self.assertEqual(parsed.dataframe.loc[0, "名称"], "中文")
                self.assertTrue(
                    any(warning_text in warning for warning in parsed.warnings)
                )

    def test_upload_service_accepts_jsonl(self):
        report = evaluate_uploaded_dataset(
            b'{"id":1,"name":"A"}\n{"id":2,"name":"B"}\n',
            "records.jsonl",
        )

        self.assertEqual(report.status, "success")
        self.assertEqual(report.dataset.file_type, "jsonl")
        self.assertTrue(
            any("JSON Lines" in warning for warning in report.execution["warnings"])
        )

    def test_wrapped_records_are_limited_before_standard_json_materialization(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "wrapped-records.json"
            path.write_text(
                '{"data":[{"id":1},{"id":2},{"id":3}]}',
                encoding="utf-8",
            )
            with (
                patch("src.parser.MAX_JSON_RECORDS", 2),
                patch("src.parser.json.load") as json_load,
                self.assertRaisesRegex(DatasetReadError, "记录数组"),
            ):
                parse_dataset(path)
            json_load.assert_not_called()

    def test_matrix_width_is_limited_before_standard_json_materialization(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "wide-matrix.json"
            path.write_text('[["a","b","c"],[1,2,3]]', encoding="utf-8")
            with (
                patch("src.parser.MAX_DATASET_COLUMNS", 2),
                patch("src.parser.json.load") as json_load,
                self.assertRaisesRegex(DatasetReadError, "记录数组"),
            ):
                parse_dataset(path)
            json_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
