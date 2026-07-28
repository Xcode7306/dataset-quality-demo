"""JSON 读取扩展阶段 B 的 ZIP 安全与分片合并回归。"""

import json
import stat
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZIP_LZMA, ZipFile, ZipInfo

import src.parser as parser_module
from src.parser import DatasetReadError, parse_dataset
from src.resource_limits import ResourceLimitExceeded, validate_json_zip_archive
from src.upload_service import evaluate_uploaded_dataset
from src.workflow import build_profile_report


class JsonZipExpansionTests(unittest.TestCase):
    def _write_zip(
        self,
        path: Path,
        members: list[tuple[str | ZipInfo, str | bytes]],
        *,
        compression: int = ZIP_DEFLATED,
    ) -> Path:
        with ZipFile(path, "w", compression=compression) as archive:
            for member_name, content in members:
                archive.writestr(member_name, content)
        return path

    def test_matrix_shards_are_read_from_stream_and_merged_in_archive_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "wenzhou.zip"
            self._write_zip(
                path,
                [
                    ("parts/", b""),
                    ("parts/001.json", '[["编号","名称"],[1,"A"],[2,"B"]]'),
                    ("parts/002.json", '[["编号","名称"],[3,"C"]]'),
                ],
            )

            parsed = parse_dataset(path)

        self.assertEqual(parsed.dataset.file_type, "zip")
        self.assertEqual(parsed.dataframe.columns.tolist(), ["编号", "名称"])
        self.assertEqual(parsed.dataframe["编号"].tolist(), [1, 2, 3])
        self.assertTrue(
            any("2 个分片、3 条记录" in item for item in parsed.warnings)
        )
        self.assertTrue(
            any("未按包内成员路径解压落地" in item for item in parsed.warnings)
        )
        self.assertEqual(
            sum("二维数组" in item for item in parsed.warnings),
            1,
        )

    def test_record_json_and_jsonl_shards_accept_same_field_set_in_different_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "records.zip"
            self._write_zip(
                path,
                [
                    ("001.json", '[{"id":1,"name":"A"}]'),
                    ("002.jsonl", '{"name":"B","id":2}\n'),
                ],
            )

            parsed = parse_dataset(path)

        self.assertEqual(parsed.dataframe.columns.tolist(), ["id", "name"])
        self.assertEqual(parsed.dataframe.to_dict("records"), [
            {"id": 1, "name": "A"},
            {"id": 2, "name": "B"},
        ])
        self.assertTrue(any("JSON Lines" in item for item in parsed.warnings))

    def test_representative_160_matrix_shards_merge_without_member_warning_spam(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "rainfall-160.zip"
            members = [
                (
                    f"parts/{index:03d}.json",
                    json.dumps([["分片", "值"], [index, index * 10]]),
                )
                for index in range(160)
            ]
            self._write_zip(path, members)

            parsed = parse_dataset(path)

        self.assertEqual(len(parsed.dataframe), 160)
        self.assertEqual(parsed.dataframe.iloc[-1].tolist(), [159, 1590])
        self.assertEqual(sum("二维数组" in item for item in parsed.warnings), 1)
        self.assertTrue(any("160 个分片" in item for item in parsed.warnings))

    def test_shard_structure_and_schema_conflicts_are_explainable(self):
        cases = (
            (
                [
                    ("001.json", '[["id","name"],[1,"A"]]'),
                    ("002.json", '[["name","id"],["B",2]]'),
                ],
                "表头或字段顺序",
            ),
            (
                [
                    ("001.json", '[{"id":1,"name":"A"}]'),
                    ("002.json", '[{"id":2,"status":"ok"}]'),
                ],
                "字段集合",
            ),
            (
                [
                    ("001.json", '[["id"],[1]]'),
                    ("002.json", '[{"id":2}]'),
                ],
                "结构类型",
            ),
            (
                [("empty.json", "[]")],
                "没有可校验的表头或字段集合",
            ),
        )
        for members, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "conflict.zip"
                    self._write_zip(path, members)
                    with self.assertRaisesRegex(DatasetReadError, message):
                        parse_dataset(path)

    def test_unsafe_member_paths_are_rejected_without_extraction(self):
        unsafe_paths = (
            "../evil.json",
            "/absolute.json",
            "..\\evil.json",
            "C:/evil.json",
        )
        for member_name in unsafe_paths:
            with self.subTest(member_name=member_name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    archive_directory = root / "inside"
                    archive_directory.mkdir()
                    path = archive_directory / "unsafe.zip"
                    self._write_zip(path, [(member_name, '[{"id":1}]')])

                    with self.assertRaisesRegex(
                        DatasetReadError, "绝对路径|路径穿越"
                    ):
                        parse_dataset(path)

                    self.assertFalse((root / "evil.json").exists())

    def test_symlinks_duplicates_and_non_json_members_are_rejected(self):
        symlink = ZipInfo("link.json")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        cases = (
            ([(symlink, "target.json")], "符号链接"),
            ([("readme.txt", "not a shard")], "仅允许"),
        )
        for members, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "unsafe.zip"
                    self._write_zip(path, members)
                    with self.assertRaisesRegex(DatasetReadError, message):
                        parse_dataset(path)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "duplicate.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._write_zip(
                    path,
                    [
                        ("same.json", '[{"id":1}]'),
                        ("same.json", '[{"id":2}]'),
                    ],
                )
            with self.assertRaisesRegex(DatasetReadError, "重复成员路径"):
                parse_dataset(path)

        encrypted = ZipInfo("encrypted.json")
        encrypted.flag_bits = 0x1

        class ArchiveWithEncryptedMember:
            @staticmethod
            def infolist():
                return [encrypted]

        with self.assertRaisesRegex(ResourceLimitExceeded, "已加密"):
            validate_json_zip_archive(ArchiveWithEncryptedMember())

    def test_empty_and_corrupt_zip_files_are_explainable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            empty_path = root / "empty.zip"
            with ZipFile(empty_path, "w"):
                pass
            with self.assertRaisesRegex(DatasetReadError, "未找到"):
                parse_dataset(empty_path)

            corrupt_path = root / "corrupt.zip"
            corrupt_path.write_bytes(b"not-a-zip")
            with self.assertRaisesRegex(DatasetReadError, "ZIP 文件无法打开"):
                parse_dataset(corrupt_path)

    def test_entry_count_expansion_and_compression_ratio_limits_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "limits.zip"
            self._write_zip(
                path,
                [("001.json", '[{"id":1}]'), ("002.json", '[{"id":2}]')],
            )
            with (
                patch("src.resource_limits.MAX_JSON_ZIP_ENTRY_COUNT", 1),
                self.assertRaisesRegex(DatasetReadError, "条目"),
            ):
                parse_dataset(path)
            with (
                patch("src.resource_limits.MAX_JSON_ZIP_UNCOMPRESSED_BYTES", 10),
                self.assertRaisesRegex(DatasetReadError, "解压后"),
            ):
                parse_dataset(path)
            with (
                patch("src.resource_limits.MAX_JSON_ZIP_ENTRY_BYTES", 9),
                self.assertRaisesRegex(DatasetReadError, "单分片"),
            ):
                parse_dataset(path)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ratio.zip"
            payload = json.dumps([{"text": "A" * 10_000}])
            self._write_zip(path, [("data.json", payload)])
            with (
                patch("src.resource_limits.MIN_RATIO_CHECK_BYTES", 1),
                patch("src.resource_limits.MAX_JSON_ZIP_COMPRESSION_RATIO", 2),
                self.assertRaisesRegex(DatasetReadError, "压缩比"),
            ):
                parse_dataset(path)

    def test_json_record_and_structure_totals_are_global_across_shards(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "global-limits.zip"
            self._write_zip(
                path,
                [
                    ("001.json", '[{"id":1},{"id":2}]'),
                    ("002.json", '[{"id":3},{"id":4}]'),
                ],
            )
            with (
                patch("src.parser.MAX_JSON_RECORDS", 3),
                self.assertRaisesRegex(DatasetReadError, "记录数组.*1 项"),
            ):
                parse_dataset(path)
            with (
                patch("src.parser.MAX_JSON_TOTAL_PAIRS", 3),
                self.assertRaisesRegex(DatasetReadError, "键值对"),
            ):
                parse_dataset(path)

    def test_global_cell_budget_rejects_the_next_shard_before_materialization(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "global-cells.zip"
            self._write_zip(
                path,
                [
                    ("001.json", '[["id","name"],[1,"A"]]'),
                    ("002.json", '[["id","name"],[2,"B"]]'),
                ],
            )
            with (
                patch("src.parser.MAX_DATASET_CELLS", 3),
                patch(
                    "src.parser._matrix_to_dataframe",
                    wraps=parser_module._matrix_to_dataframe,
                ) as matrix_conversion,
                self.assertRaisesRegex(DatasetReadError, "记录数组"),
            ):
                parse_dataset(path)

        self.assertEqual(matrix_conversion.call_count, 1)

    def test_schema_conflict_preserves_current_shard_encoding_warning(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "warning-conflict.zip"
            self._write_zip(
                path,
                [
                    ("001.json", '[{"id":1}]'),
                    (
                        "002.json",
                        '[{"名称":"中文"}]'.encode("gb18030"),
                    ),
                ],
            )
            with self.assertRaisesRegex(DatasetReadError, "字段集合") as raised:
                parse_dataset(path)

        self.assertTrue(
            any("GB18030" in item for item in raised.exception.warnings),
            raised.exception.warnings,
        )

    def test_only_stored_and_deflated_zip_members_are_allowed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "lzma.zip"
            self._write_zip(
                path,
                [("records.json", '[{"id":1}]')],
                compression=ZIP_LZMA,
            )
            with self.assertRaisesRegex(DatasetReadError, "压缩算法"):
                parse_dataset(path)

    def test_corrupt_deflate_stream_becomes_a_downloadable_failed_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "corrupt-deflate.zip"
            self._write_zip(path, [("broken.json", '[{"id":1}]')])
            archive_bytes = bytearray(path.read_bytes())
            with ZipFile(path) as archive:
                entry = archive.getinfo("broken.json")
            header_offset = entry.header_offset
            name_length = int.from_bytes(
                archive_bytes[header_offset + 26 : header_offset + 28],
                "little",
            )
            extra_length = int.from_bytes(
                archive_bytes[header_offset + 28 : header_offset + 30],
                "little",
            )
            compressed_data_offset = header_offset + 30 + name_length + extra_length
            archive_bytes[compressed_data_offset] ^= 0xFF
            path.write_bytes(archive_bytes)

            report = build_profile_report(path)

        self.assertEqual(report.status, "failed")
        self.assertIn("broken.json", report.execution["errors"][0])

    def test_corrupt_shard_error_includes_member_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "corrupt.zip"
            self._write_zip(
                path,
                [
                    ("001.json", '[{"id":1}]'),
                    ("broken.json", '[{"id":2,}]'),
                ],
            )
            with self.assertRaisesRegex(
                DatasetReadError, "broken.json.*JSON 格式错误"
            ):
                parse_dataset(path)

    def test_upload_service_accepts_json_zip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "records.zip"
            self._write_zip(
                path,
                [
                    ("001.json", '[{"id":1,"name":"A"}]'),
                    ("002.json", '[{"id":2,"name":"B"}]'),
                ],
            )
            report = evaluate_uploaded_dataset(path.read_bytes(), "records.zip")

        self.assertEqual(report.status, "success")
        self.assertEqual(report.dataset.file_type, "zip")
        self.assertEqual(report.profile["row_count"], 2)
        self.assertTrue(
            any(
                "ZIP JSON 分片包" in item
                for item in report.execution["warnings"]
            )
        )


if __name__ == "__main__":
    unittest.main()
