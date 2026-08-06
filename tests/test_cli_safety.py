"""CLI 输出路径安全性测试。"""

from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.cli import ensure_distinct_output_path, main, paths_refer_to_same_file
from src.models import DatasetInfo
from src.report import create_empty_report, save_report


class CliSafetyTests(unittest.TestCase):
    @staticmethod
    def _empty_report():
        return create_empty_report(
            DatasetInfo(
                name="测试数据集",
                file_name="dataset.csv",
                file_type="csv",
            )
        )

    def test_rejects_same_path_even_when_spelled_differently(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "dataset.csv"
            input_path.write_text("id\n1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "不能与输入"):
                ensure_distinct_output_path(input_path, root / "." / "dataset.csv")

    def test_rejects_symbolic_link_to_input(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "dataset.csv"
            input_path.write_text("id\n1\n", encoding="utf-8")
            output_link = root / "report.json"
            output_link.symlink_to(input_path)

            self.assertTrue(paths_refer_to_same_file(input_path, output_link))
            with self.assertRaises(ValueError):
                ensure_distinct_output_path(input_path, output_link)

    def test_rejects_hard_link_to_input(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "dataset.csv"
            input_path.write_text("id\n1\n", encoding="utf-8")
            output_link = root / "report.json"
            os.link(input_path, output_link)

            self.assertTrue(paths_refer_to_same_file(input_path, output_link))
            with self.assertRaises(ValueError):
                ensure_distinct_output_path(input_path, output_link)

    @patch("src.cli.build_profile_report")
    def test_cli_rejects_same_output_before_reading_dataset(self, build_report):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "dataset.csv"
            input_path.write_text("id\n1\n", encoding="utf-8")
            arguments = [
                "src.cli",
                str(input_path),
                "--output",
                str(input_path),
            ]

            with patch.object(sys, "argv", arguments), redirect_stderr(StringIO()):
                with self.assertRaisesRegex(SystemExit, "2"):
                    main()

            build_report.assert_not_called()

    def test_allows_a_distinct_output_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "dataset.csv"
            input_path.write_text("id\n1\n", encoding="utf-8")

            ensure_distinct_output_path(input_path, root / "report.json")

    def test_atomic_save_can_replace_an_existing_regular_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "report.json"
            output_path.write_text("旧报告", encoding="utf-8")

            save_report(self._empty_report(), output_path)

            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["dataset"]["name"], "测试数据集")
            self.assertEqual(saved["schema_version"], "0.3")
            self.assertFalse(output_path.is_symlink())

    def test_save_report_selects_format_from_extension(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            json_path = root / "report.json"
            markdown_path = root / "report.md"

            save_report(self._empty_report(), json_path)
            save_report(self._empty_report(), markdown_path)

            structured = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(structured, self._empty_report().to_dict())
        self.assertIn("# 数据集质量评估报告：测试数据集", markdown)

    def test_save_report_rejects_ambiguous_output_extension(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "report.txt"
            with self.assertRaisesRegex(ValueError, r"\.json.*\.md"):
                save_report(self._empty_report(), output_path)
            self.assertFalse(output_path.exists())

    def test_atomic_save_cleans_temporary_file_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_path = root / "report.json"

            with patch("src.report.os.replace", side_effect=OSError("模拟替换失败")):
                with self.assertRaisesRegex(OSError, "模拟替换失败"):
                    save_report(self._empty_report(), output_path)

            self.assertFalse(output_path.exists())
            self.assertEqual(list(root.glob(".quality-report-*.tmp")), [])

    @patch("src.cli.build_profile_report")
    def test_cli_does_not_follow_output_symlink_created_during_evaluation(
        self,
        build_report,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "dataset.csv"
            output_path = root / "report.json"
            original_content = "id\n1\n"
            input_path.write_text(original_content, encoding="utf-8")

            def create_racing_symlink(*_args, **_kwargs):
                output_path.symlink_to(input_path)
                return self._empty_report()

            build_report.side_effect = create_racing_symlink
            arguments = [
                "src.cli",
                str(input_path),
                "--output",
                str(output_path),
            ]
            with patch.object(sys, "argv", arguments), redirect_stderr(StringIO()):
                with self.assertRaisesRegex(SystemExit, "2"):
                    main()

            self.assertEqual(input_path.read_text(encoding="utf-8"), original_content)
            self.assertTrue(output_path.is_symlink())

    @patch("src.cli.build_profile_report")
    def test_cli_detects_output_parent_symlink_swapped_to_input_directory(
        self,
        build_report,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            safe_output_directory = root / "safe-output"
            safe_output_directory.mkdir()
            output_parent = root / "current-output"
            output_parent.symlink_to(safe_output_directory, target_is_directory=True)
            input_path = root / "dataset.csv"
            output_path = output_parent / "report.json"
            original_content = "id\n1\n"
            input_path.write_text(original_content, encoding="utf-8")

            def swap_output_parent(*_args, **_kwargs):
                output_parent.unlink()
                output_parent.symlink_to(root, target_is_directory=True)
                return self._empty_report()

            build_report.side_effect = swap_output_parent
            arguments = [
                "src.cli",
                str(input_path),
                "--output",
                str(output_path),
            ]
            with patch.object(sys, "argv", arguments), redirect_stderr(StringIO()):
                with self.assertRaisesRegex(SystemExit, "2"):
                    main()

            self.assertEqual(input_path.read_text(encoding="utf-8"), original_content)
            self.assertEqual(list(safe_output_directory.iterdir()), [])
            self.assertFalse((root / "report.json").exists())

    def test_cli_accepts_explicit_reference_date(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "dataset.csv"
            output_path = root / "report.json"
            input_path.write_text("updated_at\n2026-06-07\n", encoding="utf-8")
            arguments = [
                "src.cli",
                str(input_path),
                "--reference-date",
                "2026-07-16",
                "--output",
                str(output_path),
            ]

            with patch.object(sys, "argv", arguments), patch("builtins.print"):
                main()

            report = json.loads(output_path.read_text(encoding="utf-8"))

        lag_metric = next(
            metric
            for metric in report["metrics"]
            if metric["id"] == "update_lag_days"
        )
        self.assertEqual(lag_metric["value"], 39)

    def test_cli_accepts_repeatable_metric_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "dataset.csv"
            output_path = root / "report.json"
            input_path.write_text(
                "record_id,name\n1,A\n2,A\n",
                encoding="utf-8",
            )
            arguments = [
                "src.cli",
                str(input_path),
                "--metric",
                "exact_duplicate_rate",
                "--metric",
                "db31_030300",
                "--output",
                str(output_path),
            ]

            with patch.object(sys, "argv", arguments), patch("builtins.print"):
                main()

            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            [metric["id"] for metric in report["metrics"]],
            ["exact_duplicate_rate", "db31_030300"],
        )
        self.assertEqual(
            report["evaluation_context"]["selected_metric_ids"],
            ["exact_duplicate_rate", "db31_030300"],
        )

    def test_cli_defaults_to_structured_json_and_preserves_explicit_markdown(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "dataset.csv"
            input_path.write_text("id\n1\n", encoding="utf-8")
            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                with (
                    patch.object(sys, "argv", ["src.cli", str(input_path)]),
                    patch("builtins.print"),
                ):
                    main()
            finally:
                os.chdir(previous_directory)

            default_report = json.loads(
                (root / "reports" / "report.json").read_text(encoding="utf-8")
            )
            markdown_path = root / "explicit.md"
            arguments = [
                "src.cli",
                str(input_path),
                "--output",
                str(markdown_path),
            ]
            with patch.object(sys, "argv", arguments), patch("builtins.print"):
                main()

            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(default_report["dataset"]["file_name"], "dataset.csv")
        self.assertIn("# 数据集质量评估报告：dataset", markdown)


if __name__ == "__main__":
    unittest.main()
