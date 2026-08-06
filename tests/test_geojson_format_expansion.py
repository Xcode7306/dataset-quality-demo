"""JSON 读取扩展阶段 C 的 GeoJSON 映射与严格边界回归。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from src.parser import DatasetReadError, parse_dataset
from src.upload_service import evaluate_uploaded_dataset


PROJECT_ROOT = Path(__file__).parents[1]


class GeoJsonFormatExpansionTests(unittest.TestCase):
    def _write_payload(self, payload, extension="geojson"):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / f"dataset.{extension}"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def _feature(feature_id, properties, geometry):
        return {
            "type": "Feature",
            "id": feature_id,
            "properties": properties,
            "geometry": geometry,
        }

    def test_feature_collection_maps_one_feature_per_row_without_coordinates(self):
        sample = PROJECT_ROOT / "sample_data" / "geojson_feature_collection.geojson"
        parsed = parse_dataset(sample)

        self.assertEqual(parsed.dataset.file_type, "geojson")
        self.assertEqual(parsed.dataframe["设施名称"].tolist(), ["服务大厅", "便民服务站"])
        self.assertEqual(
            parsed.dataframe["__geojson_feature_id"].tolist(),
            ["facility-01", "facility-02"],
        )
        self.assertEqual(
            parsed.dataframe["__geojson_geometry_type"].tolist(),
            ["Point", "LineString"],
        )
        self.assertEqual(
            parsed.dataframe["__geojson_coordinate_count"].tolist(),
            [1, 2],
        )
        self.assertNotIn("coordinates", parsed.dataframe.columns)
        self.assertEqual(parsed.dataframe.loc[1, "__geojson_min_x"], 120.15)
        self.assertEqual(parsed.dataframe.loc[1, "__geojson_max_y"], 30.28)
        self.assertTrue(any("坐标数组未展开" in item for item in parsed.warnings))

    def test_feature_collection_inside_json_is_recognized(self):
        payload = {
            "type": "FeatureCollection",
            "features": [
                self._feature(
                    7,
                    {"名称": "空间样例"},
                    {"type": "Point", "coordinates": [116.4, 39.9, 44]},
                )
            ],
        }
        parsed = parse_dataset(self._write_payload(payload, "json"))

        self.assertEqual(parsed.dataset.file_type, "json")
        self.assertEqual(parsed.dataframe.loc[0, "名称"], "空间样例")
        self.assertEqual(parsed.dataframe.loc[0, "__geojson_feature_id"], 7)
        self.assertEqual(parsed.dataframe.loc[0, "__geojson_coordinate_dimension"], 3)

    def test_geometry_collection_and_null_geometry_have_safe_summaries(self):
        payload = {
            "type": "FeatureCollection",
            "features": [
                self._feature(
                    "collection",
                    {"名称": "组合几何"},
                    {
                        "type": "GeometryCollection",
                        "geometries": [
                            {"type": "Point", "coordinates": [100, 20]},
                            {
                                "type": "LineString",
                                "coordinates": [[101, 21], [102, 22]],
                            },
                        ],
                    },
                ),
                self._feature(
                    "empty-collection",
                    {"名称": "空组合几何"},
                    {"type": "GeometryCollection", "geometries": []},
                ),
                self._feature("empty", {"名称": "无几何"}, None),
            ],
        }
        parsed = parse_dataset(self._write_payload(payload))

        self.assertEqual(parsed.dataframe.loc[0, "__geojson_coordinate_count"], 3)
        self.assertEqual(parsed.dataframe.loc[0, "__geojson_min_x"], 100.0)
        self.assertEqual(parsed.dataframe.loc[0, "__geojson_max_y"], 22.0)
        self.assertEqual(parsed.dataframe.loc[1, "__geojson_coordinate_count"], 0)
        self.assertEqual(parsed.dataframe.loc[1, "__geojson_coordinate_dimension"], 0)
        self.assertTrue(parsed.dataframe.loc[2, "__geojson_geometry_type"] is None)
        self.assertTrue(any("geometry 为 null" in item for item in parsed.warnings))

    def test_nested_properties_and_invalid_geometry_are_rejected_explainably(self):
        cases = (
            (
                self._feature(
                    "nested",
                    {"名称": "错误", "metadata": {"level": 1}},
                    {"type": "Point", "coordinates": [1, 2]},
                ),
                "properties 包含嵌套",
            ),
            (
                self._feature(
                    "reserved",
                    {"__geojson_min_x": "collision"},
                    {"type": "Point", "coordinates": [1, 2]},
                ),
                "保留字段",
            ),
            (
                self._feature(
                    "bad-position",
                    {"名称": "错误"},
                    {"type": "Point", "coordinates": [1, "x"]},
                ),
                "有限数值",
            ),
            (
                self._feature(
                    "unsupported",
                    {"名称": "错误"},
                    {"type": "Circle", "coordinates": [1, 2]},
                ),
                "不支持的几何类型",
            ),
            (
                self._feature(
                    "wrong-shape",
                    {"名称": "错误"},
                    {"type": "LineString", "coordinates": [1, 2]},
                ),
                "coordinates 必须是数组",
            ),
        )
        for feature, message in cases:
            with self.subTest(message=message):
                payload = {"type": "FeatureCollection", "features": [feature]}
                with self.assertRaisesRegex(DatasetReadError, message):
                    parse_dataset(self._write_payload(payload))

    def test_unmapped_nested_geojson_paths_are_rejected_instead_of_dropped(self):
        base_feature = self._feature(
            "point",
            {"名称": "空间样例"},
            {"type": "Point", "coordinates": [1, 2]},
        )
        cases = []

        top_level = {"type": "FeatureCollection", "features": [base_feature]}
        top_level["meta"] = {"nested": {"owner": "ignored"}}
        cases.append(top_level)

        feature_member = dict(base_feature)
        feature_member["audit"] = {"nested": {"code": 7}}
        cases.append({"type": "FeatureCollection", "features": [feature_member]})

        geometry_member = dict(base_feature)
        geometry_member["geometry"] = {
            "type": "Point",
            "coordinates": [1, 2],
            "extra": {"nested": True},
        }
        cases.append({"type": "FeatureCollection", "features": [geometry_member]})

        for payload in cases:
            with self.subTest(keys=sorted(payload.keys())):
                with self.assertRaisesRegex(DatasetReadError, "未定义的嵌套字段"):
                    parse_dataset(self._write_payload(payload))

    def test_lines_and_polygon_rings_must_have_valid_cardinality_and_closure(self):
        invalid_geometries = (
            {"type": "LineString", "coordinates": [[1, 2]]},
            {"type": "MultiLineString", "coordinates": [[[1, 2]]]},
            {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [0, 0]]],
            },
            {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
            },
            {
                "type": "MultiPolygon",
                "coordinates": [[[[0, 0], [1, 0], [1, 1], [0, 1]]]],
            },
        )
        for geometry in invalid_geometries:
            with self.subTest(geometry_type=geometry["type"]):
                payload = {
                    "type": "FeatureCollection",
                    "features": [self._feature("invalid", {}, geometry)],
                }
                with self.assertRaisesRegex(
                    DatasetReadError,
                    "至少需要|\u9996\u5c3e\u5750\u6807",
                ):
                    parse_dataset(self._write_payload(payload))

    def test_feature_requires_properties_and_a_string_or_numeric_id(self):
        missing_properties = {
            "type": "Feature",
            "id": "missing-properties",
            "geometry": {"type": "Point", "coordinates": [1, 2]},
        }
        payload = {"type": "FeatureCollection", "features": [missing_properties]}
        with self.assertRaisesRegex(DatasetReadError, "缺少 properties"):
            parse_dataset(self._write_payload(payload))

        for invalid_id in (True, None):
            with self.subTest(invalid_id=invalid_id):
                payload = {
                    "type": "FeatureCollection",
                    "features": [
                        self._feature(
                            invalid_id,
                            {},
                            {"type": "Point", "coordinates": [1, 2]},
                        )
                    ],
                }
                with self.assertRaisesRegex(DatasetReadError, "id.*字符串或有限数值"):
                    parse_dataset(self._write_payload(payload))

    def test_geojson_extension_requires_feature_collection(self):
        payload = [{"id": 1, "name": "not geojson"}]
        with self.assertRaisesRegex(DatasetReadError, "FeatureCollection"):
            parse_dataset(self._write_payload(payload))

    def test_feature_and_resource_limits_are_checked_before_json_materialization(self):
        payload = {
            "type": "FeatureCollection",
            "features": [
                self._feature(
                    index,
                    {"id": index},
                    {"type": "Point", "coordinates": [index, index]},
                )
                for index in range(3)
            ],
        }
        path = self._write_payload(payload)
        with (
            patch("src.parser.MAX_JSON_RECORDS", 2),
            patch("src.parser.json.load") as json_load,
            self.assertRaisesRegex(DatasetReadError, "记录数组"),
        ):
            parse_dataset(path)
        json_load.assert_not_called()

    def test_upload_service_accepts_geojson(self):
        sample = PROJECT_ROOT / "sample_data" / "geojson_feature_collection.geojson"
        report = evaluate_uploaded_dataset(sample.read_bytes(), sample.name)

        self.assertEqual(report.status, "success")
        self.assertEqual(report.dataset.file_type, "geojson")
        self.assertEqual(report.profile["row_count"], 2)
        self.assertTrue(
            any("坐标数组未展开" in item for item in report.execution["warnings"])
        )

    def test_json_zip_does_not_silently_enable_geojson_shards(self):
        payload = {
            "type": "FeatureCollection",
            "features": [
                self._feature(
                    "point",
                    {"名称": "空间样例"},
                    {"type": "Point", "coordinates": [1, 2]},
                )
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "geojson-shard.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("feature.json", json.dumps(payload))
            with (
                patch("src.parser._geojson_feature_collection_to_dataframe") as mapper,
                self.assertRaisesRegex(DatasetReadError, "不支持 GeoJSON"),
            ):
                parse_dataset(path)
            mapper.assert_not_called()


if __name__ == "__main__":
    unittest.main()
