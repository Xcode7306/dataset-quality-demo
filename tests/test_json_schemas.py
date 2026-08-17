"""已发布 JSON Schema 与运行时报告输出的一致性测试。"""

import json
from datetime import date
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator

from src.workflow import build_profile_report


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "sample_data"
SCHEMAS = ROOT / "schemas"
REFERENCE_DATE = date(2026, 7, 17)


class JsonSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.quality_report_schema = json.loads(
            (SCHEMAS / "quality-report.schema.json").read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(cls.quality_report_schema)
        cls.quality_report_validator = Draft202012Validator(
            cls.quality_report_schema
        )

    def assert_quality_report_is_valid(self, payload):
        errors = sorted(
            self.quality_report_validator.iter_errors(payload),
            key=lambda error: list(error.absolute_path),
        )
        self.assertEqual(
            errors,
            [],
            "\n".join(
                f"{'/'.join(map(str, error.absolute_path))}: {error.message}"
                for error in errors
            ),
        )

    def test_supported_success_reports_match_quality_report_schema(self):
        sample_names = (
            "good_dataset.csv",
            "bad_dataset.csv",
            "format_messy_dataset.csv",
            "minimal_dataset.json",
            "good_dataset.xlsx",
        )

        for sample_name in sample_names:
            with self.subTest(sample=sample_name):
                report = build_profile_report(
                    SAMPLES / sample_name,
                    reference_date=REFERENCE_DATE,
                )
                self.assert_quality_report_is_valid(report.to_dict())

    def test_failed_report_matches_quality_report_schema(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "broken.json"
            path.write_text('{"data": [', encoding="utf-8")
            report = build_profile_report(path, reference_date=REFERENCE_DATE)

        self.assert_quality_report_is_valid(report.to_dict())

    def test_schema_rejects_a_metric_without_its_stable_key(self):
        payload = build_profile_report(
            SAMPLES / "good_dataset.csv",
            reference_date=REFERENCE_DATE,
        ).to_dict()
        self.assertTrue(
            all("issue_locations" not in metric for metric in payload["metrics"])
        )
        payload["metrics"][0].pop("metric_key")

        self.assertTrue(
            list(self.quality_report_validator.iter_errors(payload))
        )


if __name__ == "__main__":
    unittest.main()
