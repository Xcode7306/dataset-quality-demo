"""CSV、Excel、表格型 JSON 与 GeoJSON 输入文件的解析模块。"""

import codecs
import csv
import io
import json
import math
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, ContextManager
from zipfile import BadZipFile

import pandas as pd

from .models import DatasetInfo
from .resource_limits import (
    MAX_DATASET_CELLS,
    MAX_DATASET_COLUMNS,
    MAX_DATASET_ROWS,
    MAX_JSON_ARRAY_ITEMS,
    MAX_CELL_TEXT_BYTES,
    MAX_JSON_NESTING_DEPTH,
    MAX_JSON_OBJECT_PAIRS,
    MAX_JSON_RECORDS,
    MAX_JSON_TOTAL_ARRAY_ITEMS,
    MAX_JSON_TOTAL_PAIRS,
    ResourceLimitExceeded,
    validate_dataframe_limits,
    validate_input_file_size,
    validate_xlsx_archive,
)
from .text_utils import normalize_display_text


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xls",
    ".xlsx",
    ".json",
    ".jsonl",
    ".ndjson",
    ".geojson",
}
JSON_WRAPPER_KEYS = {"data", "rows", "records", "items", "list", "result"}
GEOJSON_TECHNICAL_PREFIX = "__geojson_"
GEOJSON_ALLOWED_GEOMETRY_TYPES = {
    "Point",
    "MultiPoint",
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
    "GeometryCollection",
}
GEOJSON_COORDINATE_DEPTHS = {
    "Point": 0,
    "MultiPoint": 1,
    "LineString": 1,
    "MultiLineString": 2,
    "Polygon": 2,
    "MultiPolygon": 3,
}
ERROR_VALUE_MAX_CHARACTERS = 160
ERROR_VALUE_MAX_ITEMS = 10
BinaryOpener = Callable[[], ContextManager[BinaryIO]]


class _ReportableInputError(ValueError):
    """可安全写入 UTF-8 失败报告的输入异常基类。"""

    def __init__(self, message: str, warnings: list[str] | None = None):
        normalized_message, replaced_invalid_unicode = normalize_display_text(
            message
        )
        normalized_warnings = list(warnings or [])
        if replaced_invalid_unicode:
            normalized_warnings.append(
                "错误详情包含无法表示为 UTF-8 的字符，"
                "已替换为 Unicode 替代字符。"
            )
        super().__init__(normalized_message)
        self.warnings = list(dict.fromkeys(normalized_warnings))


class UnsupportedFileTypeError(_ReportableInputError):
    """上传的文件类型不在第一版支持范围内。"""


class DatasetReadError(_ReportableInputError):
    """支持的文件无法按第一版规则读取。"""


@dataclass
class ParsedDataset:
    """解析后统一交给数据画像与指标模块的表格型数据对象。"""

    dataframe: pd.DataFrame
    dataset: DatasetInfo
    warnings: list[str] = field(default_factory=list)


@dataclass
class _JsonReadResult:
    """单个 JSON 文档或分片的内部读取结果。"""

    dataframe: pd.DataFrame
    warnings: list[str]
    structure_kind: str
    total_pairs: int
    total_array_items: int


@dataclass
class _GeoJsonCoordinateSummary:
    """仅保留 GeoJSON 坐标的统计摘要，永不保留或展开原始坐标。"""

    coordinate_count: int = 0
    coordinate_dimension: int = 0
    min_x: float | None = None
    min_y: float | None = None
    max_x: float | None = None
    max_y: float | None = None


def _error_value_excerpt(value: object) -> str:
    """将输入派生文本限制为可安全写入失败报告的短摘要。"""

    text, _ = normalize_display_text(value)
    text = "".join(
        character if character.isprintable() else "�" for character in text
    )
    if len(text) > ERROR_VALUE_MAX_CHARACTERS:
        return f"{text[:ERROR_VALUE_MAX_CHARACTERS]}…"
    return text


def _format_error_values(values: list[str], empty_label: str) -> str:
    """有界展示字段名集合，避免攻击性输入放大错误报告。"""

    displayed = [
        _error_value_excerpt(value) or empty_label
        for value in values[:ERROR_VALUE_MAX_ITEMS]
    ]
    if len(values) > ERROR_VALUE_MAX_ITEMS:
        displayed.append(f"等 {len(values)} 个")
    return "、".join(displayed)


def validate_file_type(file_path: str | Path) -> Path:
    """检查文件存在且扩展名属于当前版本支持范围。"""

    path = Path(file_path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"暂不支持 {path.suffix or '无扩展名'} 文件；支持类型为：{supported}。"
        )
    if not path.exists():
        raise DatasetReadError(f"未找到文件：{path}。")
    if not path.is_file():
        raise DatasetReadError(f"输入路径不是文件：{path}。")
    return path


def _header_text(value: Any) -> str:
    """按最终字段名规则把原始表头转换为可比较文本。"""

    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _validate_unique_original_headers(headers: list[Any], source: str) -> None:
    """在 pandas 自动改写重复列名之前拒绝原始重复表头。"""

    header_texts = [_header_text(value) for value in headers]
    for field_index, header_text in enumerate(header_texts, start=1):
        if len(header_text.encode("utf-8", errors="replace")) > MAX_CELL_TEXT_BYTES:
            raise DatasetReadError(
                f"{source}第 {field_index} 个字段名超过 "
                f"{MAX_CELL_TEXT_BYTES} 字节的单项文本上限。"
            )
    duplicated = pd.Index(header_texts).duplicated(keep=False)
    duplicate_headers = list(
        dict.fromkeys(
            header_text
            for header_text, is_duplicate in zip(header_texts, duplicated)
            if is_duplicate
        )
    )
    if duplicate_headers:
        display_headers = _format_error_values(
            duplicate_headers,
            "（空字段名）",
        )
        raise DatasetReadError(
            f"{source}原始表头存在重复字段：{display_headers}。"
            "请在评估前将字段名修改为唯一值。"
        )


def _validate_csv_structure(path: Path, encoding: str) -> list[str]:
    """读取原始表头，并拒绝超过表头宽度的数据记录。

    pandas 会在每条数据记录都比表头多出字段时，把多出的前导
    字段静默推断为索引。这会使原始数据不进入画像与指标，因此在
    交给 pandas 之前按 CSV 逻辑记录校验宽度。
    """

    try:
        with path.open("r", encoding=encoding, newline="") as file:
            consumed_lines: list[str] = []

            def tracked_lines():
                for line in file:
                    consumed_lines.append(line)
                    yield line

            reader = csv.reader(tracked_lines())
            header: list[str] | None = None
            header_width = 0

            while True:
                consumed_lines.clear()
                start_line = reader.line_num + 1
                try:
                    row = next(reader)
                except StopIteration:
                    break
                end_line = reader.line_num
                raw_record = "".join(consumed_lines)

                # 与 pandas.read_csv 默认 skip_blank_lines=True 的常见
                # 空行规则保持一致，但保留引号内的空白或换行。
                if not raw_record.strip(" \t\r\n"):
                    continue

                if header is None:
                    header = row
                    header_width = len(row)
                    continue

                if len(row) > header_width:
                    line_label = (
                        f"第 {start_line} 行"
                        if start_line == end_line
                        else f"第 {start_line}-{end_line} 行"
                    )
                    raise DatasetReadError(
                        f"CSV {line_label}包含 {len(row)} 个字段，"
                        f"超过表头的 {header_width} 个字段。"
                        "请检查分隔符、引号或表头是否缺失。"
                    )

            if header is None:
                raise DatasetReadError("CSV 不包含可读取的表头或记录。")
            return header
    except csv.Error as error:
        raise DatasetReadError(f"CSV 格式无法解析：{error}") from error


def _read_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """按常见中文编码依次读取 CSV。"""

    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(64 * 1024), b""):
                if b"\x00" in chunk:
                    raise DatasetReadError(
                        "CSV 包含 NUL 空字符，无法在不改变内容的前提下安全解析。"
                    )
    except OSError as error:
        raise DatasetReadError(f"CSV 文件无法读取：{error}") from error

    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            _validate_unique_original_headers(
                _validate_csv_structure(path, encoding),
                "CSV ",
            )
            dataframe = pd.read_csv(
                path,
                encoding=encoding,
                keep_default_na=False,
                nrows=MAX_DATASET_ROWS + 1,
            )
            warnings = []
            if encoding != "utf-8-sig":
                warnings.append(f"CSV 使用 {encoding.upper()} 编码读取。")
            return dataframe, warnings
        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error.reason}")
        except pd.errors.EmptyDataError as error:
            raise DatasetReadError("CSV 不包含可读取的表头或记录。") from error
        except pd.errors.ParserError as error:
            raise DatasetReadError(f"CSV 格式无法解析：{error}") from error
        except OSError as error:
            raise DatasetReadError(f"CSV 文件无法读取：{error}") from error

    detail = "；".join(errors)
    raise DatasetReadError(f"无法按 UTF-8 或 GBK 读取 CSV：{detail}")


def _read_excel(path: Path, sheet_name: str | None) -> tuple[pd.DataFrame, str]:
    """读取指定 Excel 工作表；未指定时读取第一个工作表。"""

    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    try:
        # 按扩展名固定引擎，避免伪装成 .xls 的 OOXML 文件绕过
        # .xlsx 专属的 ZIP 展开体积与压缩比预检。
        workbook = pd.ExcelFile(path, engine=engine)
    # 损坏的 OLE、ZIP、XML 或工作簿元数据可以从两个引擎抛出
    # 多种异常；它们都属于用户输入无法解析，应在边界统一。
    except Exception as error:
        raise DatasetReadError(f"Excel 文件无法打开：{error}") from error

    with workbook:
        if not workbook.sheet_names:
            raise DatasetReadError("Excel 文件不包含可读取的工作表。")
        target_sheet = sheet_name or workbook.sheet_names[0]
        if target_sheet not in workbook.sheet_names:
            available = "、".join(workbook.sheet_names)
            raise DatasetReadError(
                f"未找到工作表“{target_sheet}”；可用工作表为：{available}。"
            )
        try:
            original_header = pd.read_excel(
                workbook,
                sheet_name=target_sheet,
                header=None,
                nrows=1,
                dtype=object,
                keep_default_na=False,
            )
            if not original_header.empty:
                _validate_unique_original_headers(
                    original_header.iloc[0].tolist(),
                    "Excel ",
                )
            return (
                pd.read_excel(
                    workbook,
                    sheet_name=target_sheet,
                    keep_default_na=False,
                    nrows=MAX_DATASET_ROWS + 1,
                ),
                target_sheet,
            )
        except DatasetReadError:
            raise
        except Exception as error:
            raise DatasetReadError(
                f"工作表“{target_sheet}”无法读取：{error}"
            ) from error


def _contains_nested_value(record: dict[str, Any]) -> bool:
    return any(isinstance(value, (dict, list)) for value in record.values())


def _normalize_json_string(value: str) -> str:
    """将合法的 UTF-16 surrogate pair 转换为 Unicode 标量值。

    Python 的 JSON 解码器会保留 ``\\ud800`` 这类孤立代理码位，
    它们会在后续的 UTF-8 报告序列化时再次失败。
    """

    try:
        normalized = value.encode("utf-16", errors="surrogatepass").decode(
            "utf-16"
        )
    except UnicodeDecodeError as error:
        raise DatasetReadError(
            "JSON 包含孤立的 Unicode 代理字符，无法安全输出 UTF-8 报告。"
        ) from error
    if len(normalized.encode("utf-8")) > MAX_CELL_TEXT_BYTES:
        raise DatasetReadError(
            f"JSON 包含超过 {MAX_CELL_TEXT_BYTES} 字节的单项文本，"
            "为避免内存和报告放大已停止解析。"
        )
    return normalized


def _validate_json_unicode_tree(value: Any) -> None:
    """递归校验整个 JSON 文档，包括最终不映射的包装元数据。"""

    if isinstance(value, str):
        _normalize_json_string(value)
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_unicode_tree(item)
        return
    if not isinstance(value, dict):
        return

    normalized_keys: set[str] = set()
    for key, child in value.items():
        normalized_key = _normalize_json_string(key)
        if normalized_key in normalized_keys:
            raise DatasetReadError(
                "JSON 字段名规范化后存在重复："
                f"{_error_value_excerpt(normalized_key)}。"
            )
        normalized_keys.add(normalized_key)
        _validate_json_unicode_tree(child)


def _normalize_record_unicode(record: dict[str, Any]) -> dict[str, Any]:
    """校验并规范扁平 JSON 记录中的字段名和字符串值。"""

    normalized: dict[str, Any] = {}
    for key, value in record.items():
        normalized_key = _normalize_json_string(key)
        if normalized_key in normalized:
            raise DatasetReadError(
                "JSON 字段名规范化后存在重复："
                f"{_error_value_excerpt(normalized_key)}。"
            )
        normalized[normalized_key] = (
            _normalize_json_string(value) if isinstance(value, str) else value
        )
    return normalized


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """构造 JSON 对象时立即拒绝重复键，避免静默保留最后一个值。"""

    result: dict[str, Any] = {}
    normalized_keys: set[str] = set()
    for key, value in pairs:
        normalized_key = _normalize_json_string(key)
        if key in result:
            raise DatasetReadError(
                "JSON 对象存在重复字段："
                f"{_error_value_excerpt(key)}。"
            )
        if normalized_key in normalized_keys:
            raise DatasetReadError(
                "JSON 字段名规范化后存在重复："
                f"{_error_value_excerpt(normalized_key)}。"
            )
        normalized_keys.add(normalized_key)
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    """拒绝 RFC 8259 未定义的 NaN 与 Infinity。"""

    raise DatasetReadError(f"JSON 包含非标准数值 {value}，请改为有限数值或 null。")


def _parse_json_integer(value: str) -> int:
    """在各 Python 版本上统一拒绝超出有限浮点范围的巨大整数。"""

    digits = value.removeprefix("-")
    if len(digits) > 309:
        raise DatasetReadError(
            "JSON 内容无法读取：整数位数过多，超出安全数值范围。"
        )
    parsed = int(value)
    try:
        finite_value = float(parsed)
    except OverflowError as error:
        raise DatasetReadError(
            "JSON 内容无法读取：整数超出安全数值范围。"
        ) from error
    if not math.isfinite(finite_value):
        raise DatasetReadError(
            "JSON 内容无法读取：整数超出安全数值范围。"
        )
    return parsed


def _parse_json_float(value: str) -> float:
    """拒绝语法合法但转换后溢出为无穷大的 JSON 浮点数。"""

    parsed = float(value)
    if not math.isfinite(parsed):
        raise DatasetReadError(
            "JSON 数值超出有限浮点数范围，请缩小数值或改用字符串。"
        )
    return parsed


def _path_binary_opener(path: Path) -> BinaryOpener:
    return lambda: path.open("rb")


def _can_decode_stream(open_binary: BinaryOpener, encoding: str) -> None:
    """以增量解码检查整个输入流，避免为编码探测制造文本副本。"""

    decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
    with open_binary() as file:
        for chunk in iter(lambda: file.read(64 * 1024), b""):
            decoder.decode(chunk)
    decoder.decode(b"", final=True)


def _detect_json_encoding_source(
    open_binary: BinaryOpener,
) -> tuple[str, list[str]]:
    """识别 JSON 编码；无 BOM 时只做 UTF-8 与 GB18030 的确定性回退。"""

    try:
        with open_binary() as file:
            prefix = file.read(4)
        if prefix.startswith(codecs.BOM_UTF8):
            _can_decode_stream(open_binary, "utf-8-sig")
            return "utf-8-sig", []
        if prefix.startswith(codecs.BOM_UTF32_LE) or prefix.startswith(
            codecs.BOM_UTF32_BE
        ):
            _can_decode_stream(open_binary, "utf-32")
            return "utf-32", ["JSON 使用 UTF-32 BOM 编码读取。"]
        if prefix.startswith(codecs.BOM_UTF16_LE) or prefix.startswith(
            codecs.BOM_UTF16_BE
        ):
            _can_decode_stream(open_binary, "utf-16")
            return "utf-16", ["JSON 使用 UTF-16 BOM 编码读取。"]

        for encoding in ("utf-8", "gb18030"):
            try:
                _can_decode_stream(open_binary, encoding)
                warnings = (
                    []
                    if encoding == "utf-8"
                    else ["JSON 使用 GB18030 编码读取。"]
                )
                return encoding, warnings
            except UnicodeDecodeError:
                continue
    except UnicodeDecodeError as error:
        raise DatasetReadError("JSON 声明的 BOM 编码与文件内容不匹配。") from error
    except (BadZipFile, NotImplementedError, OSError, RuntimeError) as error:
        raise DatasetReadError(f"JSON 文件无法读取：{error}") from error

    raise DatasetReadError("JSON 无法按 UTF-8 或 GB18030 编码读取。")


def _detect_json_encoding(path: Path) -> tuple[str, list[str]]:
    return _detect_json_encoding_source(_path_binary_opener(path))


def _preflight_json_chunks(
    chunks: Iterable[str],
    *,
    max_records: int | None = None,
    max_total_pairs: int | None = None,
    max_total_array_items: int | None = None,
) -> tuple[int, int]:
    """在标准解码器物化数据前流式限制 JSON 结构规模。"""

    record_limit = max(
        MAX_JSON_RECORDS if max_records is None else max_records,
        0,
    )
    total_pair_limit = (
        MAX_JSON_TOTAL_PAIRS
        if max_total_pairs is None
        else max_total_pairs
    )
    total_array_item_limit = (
        MAX_JSON_TOTAL_ARRAY_ITEMS
        if max_total_array_items is None
        else max_total_array_items
    )
    top_level: str | None = None
    depth = 0
    in_string = False
    escaped = False
    container_types: dict[int, str] = {}
    object_pair_counts: dict[int, int] = {}
    array_value_started: dict[int, bool] = {}
    array_value_kinds: dict[int, str | None] = {}
    array_item_counts: dict[int, int] = {}
    total_pair_count = 0
    total_array_item_count = 0

    def start_array_value(kind: str) -> None:
        if container_types.get(depth) != "array":
            return
        array_value_started[depth] = True
        if array_value_kinds.get(depth) is None:
            array_value_kinds[depth] = kind

    def finish_array_value(array_depth: int) -> None:
        nonlocal total_array_item_count
        if not array_value_started.get(array_depth, False):
            return
        item_count = array_item_counts[array_depth] + 1
        array_item_counts[array_depth] = item_count
        array_value_started[array_depth] = False
        total_array_item_count += 1

        first_kind = array_value_kinds.get(array_depth)
        parent_is_matrix = (
            container_types.get(array_depth - 1) == "array"
            and array_value_kinds.get(array_depth - 1) == "array"
        )
        if parent_is_matrix:
            item_limit = min(MAX_JSON_ARRAY_ITEMS, MAX_DATASET_COLUMNS)
        else:
            item_limit = min(
                MAX_JSON_ARRAY_ITEMS,
                record_limit + (1 if first_kind == "array" else 0),
            )
        if item_count > item_limit:
            raise DatasetReadError(
                f"JSON 单个记录数组包含超过 {item_limit} 项，"
                "为避免内存耗尽已停止解析。"
            )
        if total_array_item_count > total_array_item_limit:
            raise DatasetReadError(
                f"JSON 全文件数组项总数超过 "
                f"{total_array_item_limit} 项，为避免内存耗尽已停止解析。"
            )

    for chunk in chunks:
        for character in chunk:
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue

            if top_level is None:
                if character.isspace():
                    continue
                if character not in "[{":
                    # 顶层标量或非法语法交给标准解码器生成准确错误。
                    return total_pair_count, total_array_item_count
                top_level = "array" if character == "[" else "object"
                depth = 1
                container_types[depth] = top_level
                if top_level == "array":
                    array_value_started[depth] = False
                    array_value_kinds[depth] = None
                    array_item_counts[depth] = 0
                else:
                    object_pair_counts[depth] = 0
                continue

            if character == '"':
                start_array_value("scalar")
                in_string = True
                continue

            if character in "[{":
                # 顶层记录列表或二维数组中的单元格不允许再嵌套，
                # 可在物化前直接拒绝；对象顶层则保留给包装提取判断。
                if (
                    top_level == "array"
                    and depth == 2
                    and container_types.get(depth) in {"array", "object"}
                ):
                    raise DatasetReadError(
                        "JSON 包含嵌套对象或列表，"
                        "当前不支持自动展平。"
                    )
                start_array_value("array" if character == "[" else "object")
                depth += 1
                if depth > MAX_JSON_NESTING_DEPTH:
                    raise DatasetReadError("JSON 嵌套层级过深，无法安全解析。")
                container_type = "array" if character == "[" else "object"
                container_types[depth] = container_type
                if container_type == "array":
                    array_value_started[depth] = False
                    array_value_kinds[depth] = None
                    array_item_counts[depth] = 0
                else:
                    object_pair_counts[depth] = 0
                continue

            if character == ":" and container_types.get(depth) == "object":
                object_pair_counts[depth] += 1
                total_pair_count += 1
                if object_pair_counts[depth] > MAX_JSON_OBJECT_PAIRS:
                    raise DatasetReadError(
                        f"JSON 单个对象包含超过 {MAX_JSON_OBJECT_PAIRS} 个键值对，"
                        "为避免内存耗尽已停止解析。"
                    )
                if total_pair_count > total_pair_limit:
                    raise DatasetReadError(
                        f"JSON 全文件包含超过 {total_pair_limit} 个键值对，"
                        "为避免内存耗尽已停止解析。"
                    )
                continue

            if character == "," and container_types.get(depth) == "array":
                finish_array_value(depth)
                continue

            if character in "]}":
                if character == "]" and container_types.get(depth) == "array":
                    finish_array_value(depth)
                    array_value_started.pop(depth, None)
                    array_value_kinds.pop(depth, None)
                    array_item_counts.pop(depth, None)
                object_pair_counts.pop(depth, None)
                container_types.pop(depth, None)
                depth = max(depth - 1, 0)
                continue

            if not character.isspace() and character != ",":
                start_array_value("scalar")

    return total_pair_count, total_array_item_count


def _preflight_json_source(
    open_binary: BinaryOpener,
    encoding: str,
    *,
    max_records: int | None = None,
    max_total_pairs: int | None = None,
    max_total_array_items: int | None = None,
) -> tuple[int, int]:
    try:
        with open_binary() as binary_file:
            with io.TextIOWrapper(
                binary_file, encoding=encoding, newline=""
            ) as file:
                return _preflight_json_chunks(
                    iter(lambda: file.read(64 * 1024), ""),
                    max_records=max_records,
                    max_total_pairs=max_total_pairs,
                    max_total_array_items=max_total_array_items,
                )
    except UnicodeDecodeError as error:
        raise DatasetReadError("JSON 编码在解析过程中发生解码错误。") from error
    except (BadZipFile, NotImplementedError, OSError, RuntimeError) as error:
        raise DatasetReadError(f"JSON 文件无法读取：{error}") from error


def _preflight_json_structure(path: Path, encoding: str) -> tuple[int, int]:
    return _preflight_json_source(_path_binary_opener(path), encoding)


def _json_load_kwargs() -> dict[str, Any]:
    return {
        "object_pairs_hook": _object_without_duplicate_keys,
        "parse_constant": _reject_nonstandard_json_constant,
        "parse_int": _parse_json_integer,
        "parse_float": _parse_json_float,
    }


def _load_json_document_source(
    open_binary: BinaryOpener, encoding: str
) -> tuple[Any, bool]:
    """严格读取 JSON；仅修复整体外层精确为 ``{[...]}`` 的样本。"""

    try:
        with open_binary() as binary_file:
            with io.TextIOWrapper(binary_file, encoding=encoding) as file:
                return json.load(file, **_json_load_kwargs()), False
    except json.JSONDecodeError as original_error:
        try:
            with open_binary() as binary_file:
                with io.TextIOWrapper(binary_file, encoding=encoding) as file:
                    text = file.read()
        except (
            BadZipFile,
            UnicodeDecodeError,
            NotImplementedError,
            OSError,
            RuntimeError,
        ) as error:
            raise DatasetReadError(f"JSON 文件无法读取：{error}") from error
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            inner = stripped[1:-1].strip()
        else:
            inner = ""
        if inner.startswith("[") and inner.endswith("]"):
            try:
                repaired = json.loads(inner, **_json_load_kwargs())
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(repaired, list) and all(
                    isinstance(record, dict) for record in repaired
                ):
                    return repaired, True
        raise DatasetReadError(
            f"JSON 格式错误：第 {original_error.lineno} 行"
            f"第 {original_error.colno} 列。"
        ) from original_error


def _load_json_document(path: Path, encoding: str) -> tuple[Any, bool]:
    return _load_json_document_source(_path_binary_opener(path), encoding)


def _classify_tabular_value(value: Any) -> str | None:
    if isinstance(value, dict) and not _contains_nested_value(value):
        return "record"
    if not isinstance(value, list):
        return None
    if not value:
        return "empty"
    if all(isinstance(item, dict) for item in value):
        return "records"
    if all(isinstance(item, list) for item in value):
        return "matrix"
    return None


def _find_wrapped_tabular_values(
    payload: dict[str, Any], path: str = "$", *, limit: int = 2
) -> list[tuple[str, str, Any]]:
    """只收集判断唯一性所需的候选，避免歧义输入放大内存与错误文本。"""

    candidates: list[tuple[str, str, Any]] = []
    for key, value in payload.items():
        if len(candidates) >= limit:
            break
        if key not in JSON_WRAPPER_KEYS:
            continue
        child_path = f"{path}.{key}"
        kind = _classify_tabular_value(value)
        if kind is not None:
            candidates.append((child_path, kind, value))
        elif isinstance(value, dict):
            candidates.extend(
                _find_wrapped_tabular_values(
                    value,
                    child_path,
                    limit=limit - len(candidates),
                )
            )
    return candidates


def _check_json_table_size(row_count: int, column_count: int) -> None:
    if row_count > MAX_JSON_RECORDS:
        raise DatasetReadError(
            f"JSON 包含 {row_count} 条记录，超过 "
            f"{MAX_JSON_RECORDS} 条的 JSON 上限。"
        )
    if column_count > MAX_DATASET_COLUMNS:
        raise DatasetReadError(
            f"JSON 记录合并后包含 {column_count} 个字段，超过 "
            f"{MAX_DATASET_COLUMNS} 个的 Demo 上限。"
        )
    cell_count = row_count * column_count
    if cell_count > MAX_DATASET_CELLS:
        raise DatasetReadError(
            f"JSON 记录展开后需要 {cell_count} 个单元格，超过 "
            f"{MAX_DATASET_CELLS} 个的 Demo 上限。"
        )


def _records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    normalized_records: list[dict[str, Any]] = []
    seen_fields: dict[str, None] = {}
    for record in records:
        if _contains_nested_value(record):
            raise DatasetReadError(
                "JSON 包含嵌套对象或列表，当前不支持自动展平。"
            )
        normalized = _normalize_record_unicode(record)
        normalized_records.append(normalized)
        for field_name in normalized:
            seen_fields.setdefault(field_name, None)
        _check_json_table_size(len(normalized_records), len(seen_fields))
    _check_json_table_size(len(normalized_records), len(seen_fields))
    # JSON 整数与 null 混合时，pandas 默认会转为 float64，
    # 并可能将 2**53 以上的相邻整数静默舍入成同一值。
    return pd.DataFrame(
        normalized_records,
        columns=list(seen_fields),
        dtype=object,
    )


def _normalize_matrix_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        raise DatasetReadError(
            "JSON 二维数组包含嵌套单元格，当前不支持自动展平。"
        )
    return _normalize_json_string(value) if isinstance(value, str) else value


def _matrix_to_dataframe(matrix: list[list[Any]]) -> pd.DataFrame:
    if not matrix:
        return pd.DataFrame()
    normalized_header = [_normalize_matrix_cell(value) for value in matrix[0]]
    _validate_unique_original_headers(normalized_header, "JSON 二维数组 ")
    columns = [_header_text(value) for value in normalized_header]
    _check_json_table_size(len(matrix) - 1, len(columns))
    rows: list[list[Any]] = []
    for row_number, row in enumerate(matrix[1:], start=2):
        if len(row) > len(columns):
            raise DatasetReadError(
                f"JSON 二维数组第 {row_number} 行包含 {len(row)} 个值，"
                f"超过表头的 {len(columns)} 个字段。"
            )
        normalized_row = [_normalize_matrix_cell(value) for value in row]
        normalized_row.extend([None] * (len(columns) - len(normalized_row)))
        rows.append(normalized_row)
    return pd.DataFrame(rows, columns=columns, dtype=object)


def _is_finite_geojson_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _reject_unmapped_geojson_nested_members(
    value: dict[str, Any],
    allowed_nested_members: set[str],
    context: str,
) -> None:
    """拒绝白名单之外会被静默丢弃的 GeoJSON 嵌套路径。"""

    for key, child in value.items():
        if key in allowed_nested_members:
            continue
        if isinstance(child, (dict, list)):
            raise DatasetReadError(
                f"{context}包含未定义的嵌套字段"
                f"“{_error_value_excerpt(key)}”；当前不会静默忽略或自动展平。"
            )


def _add_geojson_position(
    position: list[Any], summary: _GeoJsonCoordinateSummary, feature_index: int
) -> None:
    """将一个经纬坐标位置聚合为范围和维度，不保留坐标值数组。"""

    if len(position) not in {2, 3} or not all(
        _is_finite_geojson_number(value) for value in position
    ):
        raise DatasetReadError(
            f"GeoJSON 第 {feature_index} 个 Feature 的坐标位置"
            "必须是包含 2 或 3 个有限数值的数组。"
        )
    x, y = float(position[0]), float(position[1])
    summary.coordinate_count += 1
    summary.coordinate_dimension = max(summary.coordinate_dimension, len(position))
    summary.min_x = x if summary.min_x is None else min(summary.min_x, x)
    summary.min_y = y if summary.min_y is None else min(summary.min_y, y)
    summary.max_x = x if summary.max_x is None else max(summary.max_x, x)
    summary.max_y = y if summary.max_y is None else max(summary.max_y, y)


def _summarize_geojson_coordinate_tree(
    coordinates: Any,
    summary: _GeoJsonCoordinateSummary,
    feature_index: int,
    expected_depth: int,
) -> None:
    """按几何类型校验标准坐标树，并仅计算位置数、维度和二维范围。"""

    if not isinstance(coordinates, list):
        raise DatasetReadError(
            f"GeoJSON 第 {feature_index} 个 Feature 的 coordinates 必须是数组。"
        )
    if expected_depth == 0:
        _add_geojson_position(coordinates, summary, feature_index)
        return
    if not coordinates:
        raise DatasetReadError(
            f"GeoJSON 第 {feature_index} 个 Feature 的 coordinates 不能为空。"
        )
    for child in coordinates:
        _summarize_geojson_coordinate_tree(
            child, summary, feature_index, expected_depth - 1
        )


def _validate_geojson_coordinate_cardinality(
    geometry_type: str,
    coordinates: list[Any],
    feature_index: int,
) -> None:
    """校验 LineString 和 Polygon 类型的最小基数与线性环闭合。"""

    def validate_line(line: list[Any], label: str) -> None:
        if len(line) < 2:
            raise DatasetReadError(
                f"GeoJSON 第 {feature_index} 个 Feature 的 {label} "
                "至少需要 2 个坐标位置。"
            )

    def validate_ring(ring: list[Any], label: str) -> None:
        if len(ring) < 4:
            raise DatasetReadError(
                f"GeoJSON 第 {feature_index} 个 Feature 的 {label} "
                "至少需要 4 个坐标位置。"
            )
        if ring[0] != ring[-1]:
            raise DatasetReadError(
                f"GeoJSON 第 {feature_index} 个 Feature 的 {label} "
                "首尾坐标必须相同以形成闭合线性环。"
            )

    if geometry_type == "LineString":
        validate_line(coordinates, "LineString")
    elif geometry_type == "MultiLineString":
        for line_index, line in enumerate(coordinates, start=1):
            validate_line(line, f"MultiLineString 第 {line_index} 条子线")
    elif geometry_type == "Polygon":
        for ring_index, ring in enumerate(coordinates, start=1):
            validate_ring(ring, f"Polygon 第 {ring_index} 个线性环")
    elif geometry_type == "MultiPolygon":
        for polygon_index, polygon in enumerate(coordinates, start=1):
            for ring_index, ring in enumerate(polygon, start=1):
                validate_ring(
                    ring,
                    f"MultiPolygon 第 {polygon_index} 个多边形"
                    f"的第 {ring_index} 个线性环",
                )


def _summarize_geojson_geometry(
    geometry: dict[str, Any], feature_index: int
) -> _GeoJsonCoordinateSummary:
    """验证受支持的 GeoJSON 几何，并返回不含原始坐标的摘要。"""

    summary = _GeoJsonCoordinateSummary()

    def visit(current_geometry: Any) -> None:
        if not isinstance(current_geometry, dict):
            raise DatasetReadError(
                f"GeoJSON 第 {feature_index} 个 Feature 的 geometry 必须是对象或 null。"
            )
        geometry_type = current_geometry.get("type")
        if geometry_type not in GEOJSON_ALLOWED_GEOMETRY_TYPES:
            raise DatasetReadError(
                f"GeoJSON 第 {feature_index} 个 Feature 使用了不支持的几何类型"
                f"“{_error_value_excerpt(geometry_type)}”。"
            )
        if geometry_type == "GeometryCollection":
            _reject_unmapped_geojson_nested_members(
                current_geometry,
                {"geometries"},
                f"GeoJSON 第 {feature_index} 个 Feature 的 GeometryCollection ",
            )
            geometries = current_geometry.get("geometries")
            if not isinstance(geometries, list):
                raise DatasetReadError(
                    f"GeoJSON 第 {feature_index} 个 Feature 的 GeometryCollection "
                    "必须包含 geometries 数组。"
                )
            for child_geometry in geometries:
                visit(child_geometry)
            return
        _reject_unmapped_geojson_nested_members(
            current_geometry,
            {"coordinates"},
            f"GeoJSON 第 {feature_index} 个 Feature 的 {geometry_type} geometry ",
        )
        if "coordinates" not in current_geometry:
            raise DatasetReadError(
                f"GeoJSON 第 {feature_index} 个 Feature 的 {geometry_type} "
                "缺少 coordinates。"
            )
        _summarize_geojson_coordinate_tree(
            current_geometry["coordinates"],
            summary,
            feature_index,
            GEOJSON_COORDINATE_DEPTHS[geometry_type],
        )
        _validate_geojson_coordinate_cardinality(
            geometry_type,
            current_geometry["coordinates"],
            feature_index,
        )

    visit(geometry)
    return summary


def _geojson_properties_to_record(
    properties: Any, feature_index: int
) -> dict[str, Any]:
    """仅将扁平 properties 映射为表格字段，拒绝通用嵌套展平。"""

    if properties is None:
        return {}
    if not isinstance(properties, dict):
        raise DatasetReadError(
            f"GeoJSON 第 {feature_index} 个 Feature 的 properties 必须是对象或 null。"
        )
    if _contains_nested_value(properties):
        raise DatasetReadError(
            f"GeoJSON 第 {feature_index} 个 Feature 的 properties 包含嵌套对象或列表；"
            "当前只映射扁平 properties，不自动展平。"
        )
    normalized = _normalize_record_unicode(properties)
    reserved_fields = [
        field_name
        for field_name in normalized
        if field_name.startswith(GEOJSON_TECHNICAL_PREFIX)
    ]
    if reserved_fields:
        raise DatasetReadError(
            f"GeoJSON 第 {feature_index} 个 Feature 的 properties 使用了保留字段"
            f"“{_error_value_excerpt(reserved_fields[0])}”。"
        )
    return normalized


def _normalize_geojson_feature_id(value: Any, feature_index: int) -> Any:
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (str, int, float))
    ):
        raise DatasetReadError(
            f"GeoJSON 第 {feature_index} 个 Feature 的 id "
            "必须是字符串或有限数值。"
        )
    return _normalize_json_string(value) if isinstance(value, str) else value


def _geojson_feature_collection_to_dataframe(
    payload: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    """将标准 FeatureCollection 映射为一行一个 Feature 的表格。"""

    if payload.get("type") != "FeatureCollection":
        raise DatasetReadError("GeoJSON 顶层 type 必须为 FeatureCollection。")
    _reject_unmapped_geojson_nested_members(
        payload,
        {"features"},
        "GeoJSON FeatureCollection ",
    )
    features = payload.get("features")
    if not isinstance(features, list):
        raise DatasetReadError("GeoJSON FeatureCollection 的 features 必须是数组。")

    records: list[dict[str, Any]] = []
    null_geometry_count = 0
    for feature_index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise DatasetReadError(
                f"GeoJSON features 第 {feature_index} 项必须是 type 为 Feature 的对象。"
            )
        if "geometry" not in feature:
            raise DatasetReadError(
                f"GeoJSON 第 {feature_index} 个 Feature 缺少 geometry。"
            )
        if "properties" not in feature:
            raise DatasetReadError(
                f"GeoJSON 第 {feature_index} 个 Feature 缺少 properties。"
            )
        _reject_unmapped_geojson_nested_members(
            feature,
            {"properties", "geometry"},
            f"GeoJSON 第 {feature_index} 个 Feature ",
        )

        record = _geojson_properties_to_record(
            feature["properties"], feature_index
        )
        if "id" in feature:
            record[f"{GEOJSON_TECHNICAL_PREFIX}feature_id"] = (
                _normalize_geojson_feature_id(feature["id"], feature_index)
            )

        geometry = feature["geometry"]
        if geometry is None:
            null_geometry_count += 1
            record.update(
                {
                    f"{GEOJSON_TECHNICAL_PREFIX}geometry_type": None,
                    f"{GEOJSON_TECHNICAL_PREFIX}coordinate_count": None,
                    f"{GEOJSON_TECHNICAL_PREFIX}coordinate_dimension": None,
                    f"{GEOJSON_TECHNICAL_PREFIX}min_x": None,
                    f"{GEOJSON_TECHNICAL_PREFIX}min_y": None,
                    f"{GEOJSON_TECHNICAL_PREFIX}max_x": None,
                    f"{GEOJSON_TECHNICAL_PREFIX}max_y": None,
                }
            )
        else:
            if not isinstance(geometry, dict):
                raise DatasetReadError(
                    f"GeoJSON 第 {feature_index} 个 Feature 的 geometry "
                    "必须是对象或 null。"
                )
            summary = _summarize_geojson_geometry(geometry, feature_index)
            record.update(
                {
                    f"{GEOJSON_TECHNICAL_PREFIX}geometry_type": geometry["type"],
                    f"{GEOJSON_TECHNICAL_PREFIX}coordinate_count": summary.coordinate_count,
                    f"{GEOJSON_TECHNICAL_PREFIX}coordinate_dimension": summary.coordinate_dimension,
                    f"{GEOJSON_TECHNICAL_PREFIX}min_x": summary.min_x,
                    f"{GEOJSON_TECHNICAL_PREFIX}min_y": summary.min_y,
                    f"{GEOJSON_TECHNICAL_PREFIX}max_x": summary.max_x,
                    f"{GEOJSON_TECHNICAL_PREFIX}max_y": summary.max_y,
                }
            )
        records.append(record)

    dataframe = _records_to_dataframe(records)
    warnings = [
        "检测到 GeoJSON FeatureCollection，已按每个 Feature 一行映射扁平 "
        "properties 和空间摘要；坐标数组未展开。"
    ]
    if null_geometry_count:
        warnings.append(
            f"GeoJSON 中有 {null_geometry_count} 条 Feature 的 geometry 为 null，"
            "其空间摘要留空。"
        )
    return dataframe, warnings


def _tabular_payload_to_dataframe(
    payload: Any,
) -> tuple[pd.DataFrame, list[str], str]:
    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        dataframe, warnings = _geojson_feature_collection_to_dataframe(payload)
        return dataframe, warnings, "geojson"

    warnings: list[str] = []
    kind = _classify_tabular_value(payload)
    selected = payload

    if isinstance(payload, dict) and kind != "record":
        candidates = _find_wrapped_tabular_values(payload)
        if len(candidates) > 1:
            paths = _format_error_values(
                [path for path, _, _ in candidates],
                "$",
            )
            raise DatasetReadError(
                f"JSON 接口包装中同时存在多个表格候选（示例：{paths}）。"
                "无法安全判断应读取哪一个。"
            )
        if len(candidates) == 1:
            selected_path, kind, selected = candidates[0]
            warnings.append(
                f"检测到 JSON 接口包装，已从路径 {selected_path} 提取表格记录。"
            )

    try:
        if kind == "record":
            return _records_to_dataframe([selected]), warnings, "records"
        if kind in {"records", "empty"}:
            return _records_to_dataframe(selected), warnings, "records"
        if kind == "matrix":
            warnings.append("检测到首行表头加二维数组的 JSON，已转换为表格。")
            return _matrix_to_dataframe(selected), warnings, "matrix"
    except DatasetReadError as error:
        error.warnings = list(dict.fromkeys([*warnings, *error.warnings]))
        raise

    if isinstance(payload, list):
        raise DatasetReadError(
            "JSON 顶层列表中的每一项都必须是对象记录，"
            "或每一项都必须是首行表头的二维数组行。"
        )
    if isinstance(payload, dict):
        raise DatasetReadError(
            "JSON 包含嵌套对象或列表，且未在常见接口包装"
            "路径中找到唯一表格，当前不支持自动展平。"
        )
    raise DatasetReadError(
        "JSON 顶层必须是记录列表、单条对象记录、"
        "二维数组或可识别的接口包装。"
    )


def _read_json_lines_source(
    open_binary: BinaryOpener,
    encoding: str,
    *,
    max_records: int | None = None,
    max_total_pairs: int | None = None,
    max_total_array_items: int | None = None,
) -> _JsonReadResult:
    record_limit = max(
        MAX_JSON_RECORDS if max_records is None else max_records,
        0,
    )
    total_pair_limit = (
        MAX_JSON_TOTAL_PAIRS
        if max_total_pairs is None
        else max_total_pairs
    )
    total_array_item_limit = (
        MAX_JSON_TOTAL_ARRAY_ITEMS
        if max_total_array_items is None
        else max_total_array_items
    )
    records: list[dict[str, Any]] = []
    seen_fields: dict[str, None] = {}
    total_pairs = 0
    total_array_items = 0
    try:
        with open_binary() as binary_file:
            with io.TextIOWrapper(
                binary_file, encoding=encoding, newline=""
            ) as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip(" \t\r\n"):
                        continue
                    line_prefix = f"JSON Lines 第 {line_number} 行"
                    try:
                        if len(records) >= record_limit:
                            raise DatasetReadError(
                                f"JSON Lines 包含超过 {record_limit} 条记录。"
                            )
                        pairs, array_items = _preflight_json_chunks(
                            [line],
                            max_records=max(record_limit - len(records), 0),
                            max_total_pairs=max(total_pair_limit - total_pairs, 0),
                            max_total_array_items=max(
                                total_array_item_limit - total_array_items,
                                0,
                            ),
                        )
                        total_pairs += pairs
                        total_array_items += array_items
                        try:
                            record = json.loads(line, **_json_load_kwargs())
                        except json.JSONDecodeError as error:
                            raise DatasetReadError(
                                f"{line_prefix}格式错误：第 {error.colno} 列。"
                            ) from error
                        _validate_json_unicode_tree(record)
                        if not isinstance(record, dict):
                            raise DatasetReadError(
                                f"{line_prefix}必须是单条对象记录。"
                            )
                        if _contains_nested_value(record):
                            raise DatasetReadError(
                                f"{line_prefix}包含嵌套对象或列表，"
                                "当前不支持自动展平。"
                            )
                        normalized_record = _normalize_record_unicode(record)
                    except DatasetReadError as error:
                        if str(error).startswith(line_prefix):
                            raise
                        raise DatasetReadError(
                            f"{line_prefix}无法读取：{error}",
                            warnings=error.warnings,
                        ) from error
                    records.append(normalized_record)
                    for field_name in normalized_record:
                        seen_fields.setdefault(field_name, None)
                    try:
                        _check_json_table_size(len(records), len(seen_fields))
                    except DatasetReadError as error:
                        raise DatasetReadError(
                            f"{line_prefix}无法读取：{error}",
                            warnings=error.warnings,
                        ) from error
    except UnicodeDecodeError as error:
        raise DatasetReadError("JSON Lines 在解析过程中发生解码错误。") from error
    except (
        BadZipFile,
        NotImplementedError,
        OSError,
        RuntimeError,
        zlib.error,
    ) as error:
        raise DatasetReadError(f"JSON Lines 文件无法读取：{error}") from error

    return _JsonReadResult(
        dataframe=_records_to_dataframe(records),
        warnings=["检测到 JSON Lines / NDJSON，已按每个非空行一条记录读取。"],
        structure_kind="records",
        total_pairs=total_pairs,
        total_array_items=total_array_items,
    )


def _read_json_lines(path: Path, encoding: str) -> tuple[pd.DataFrame, list[str]]:
    result = _read_json_lines_source(_path_binary_opener(path), encoding)
    return result.dataframe, result.warnings


def _read_json_source(
    open_binary: BinaryOpener,
    suffix: str,
    *,
    max_records: int | None = None,
    max_total_pairs: int | None = None,
    max_total_array_items: int | None = None,
) -> _JsonReadResult:
    """从可重复打开的二进制流读取一个 JSON 文档或 JSON Lines 分片。"""

    warnings: list[str] = []
    try:
        encoding, encoding_warnings = _detect_json_encoding_source(open_binary)
        warnings.extend(encoding_warnings)
        if suffix.lower() in {".jsonl", ".ndjson"}:
            result = _read_json_lines_source(
                open_binary,
                encoding,
                max_records=max_records,
                max_total_pairs=max_total_pairs,
                max_total_array_items=max_total_array_items,
            )
            result.warnings = [*warnings, *result.warnings]
            return result

        total_pairs, total_array_items = _preflight_json_source(
            open_binary,
            encoding,
            max_records=max_records,
            max_total_pairs=max_total_pairs,
            max_total_array_items=max_total_array_items,
        )
        payload, repaired_outer_wrapper = _load_json_document_source(
            open_binary, encoding
        )
        if repaired_outer_wrapper:
            warnings.append(
                "原文件不是标准 JSON；已仅移除整体 `{[...]}` 外层的一对花括号。"
            )
        _validate_json_unicode_tree(payload)
        is_feature_collection = (
            isinstance(payload, dict)
            and payload.get("type") == "FeatureCollection"
        )
        if (
            suffix.lower() == ".geojson"
            and not is_feature_collection
        ):
            raise DatasetReadError(
                "GeoJSON 文件必须是顶层 type 为 FeatureCollection 的对象。"
            )
        dataframe, shape_warnings, structure_kind = (
            _tabular_payload_to_dataframe(payload)
        )
        warnings.extend(shape_warnings)
        record_limit = max(
            MAX_JSON_RECORDS if max_records is None else max_records,
            0,
        )
        if len(dataframe) > record_limit:
            raise DatasetReadError(
                f"JSON 包含 {len(dataframe)} 条记录，超过"
                f"当前可用的 {record_limit} 条记录预算。"
            )
        return _JsonReadResult(
            dataframe=dataframe,
            warnings=warnings,
            structure_kind=structure_kind,
            total_pairs=total_pairs,
            total_array_items=total_array_items,
        )
    except DatasetReadError as error:
        error.warnings = list(dict.fromkeys([*warnings, *error.warnings]))
        raise
    except RecursionError as error:
        raise DatasetReadError("JSON 嵌套层级过深，无法安全解析。") from error
    except (
        BadZipFile,
        NotImplementedError,
        OSError,
        RuntimeError,
        zlib.error,
    ) as error:
        raise DatasetReadError(f"JSON 内容无法读取：{error}") from error
    except ValueError as error:
        raise DatasetReadError(f"JSON 内容无法读取：{error}") from error


def _read_json(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """读取严格 JSON、常见表格型包装或 JSON Lines。"""

    result = _read_json_source(_path_binary_opener(path), path.suffix.lower())
    return result.dataframe, result.warnings


def parse_dataset(
    file_path: str | Path,
    dataset_name: str | None = None,
    sheet_name: str | None = None,
) -> ParsedDataset:
    """将当前版本支持的文件解析为统一表格型数据对象。"""

    path = Path(file_path)
    warnings: list[str] = []
    resolved_sheet_name: str | None = None

    display_file_name, file_name_replaced = normalize_display_text(path.name)
    display_stem, stem_replaced = normalize_display_text(path.stem)
    if file_name_replaced or stem_replaced:
        warnings.append(
            "文件显示名称包含无法表示为 UTF-8 的字符，"
            "已替换为 Unicode 替代字符。"
        )

    normalized_dataset_name = dataset_name
    if dataset_name is not None and str(dataset_name).strip():
        normalized_dataset_name, replaced_invalid_unicode = normalize_display_text(
            dataset_name
        )
        if replaced_invalid_unicode:
            warnings.append(
                "数据集名称包含无法表示为 UTF-8 的字符，"
                "已替换为 Unicode 替代字符。"
            )
    elif dataset_name is not None:
        normalized_dataset_name = None

    normalized_sheet_name = sheet_name
    if sheet_name is not None:
        normalized_sheet_name, replaced_invalid_unicode = normalize_display_text(
            sheet_name
        )
        if replaced_invalid_unicode:
            warnings.append(
                "工作表名称包含无法表示为 UTF-8 的字符，"
                "已替换为 Unicode 替代字符。"
            )

    try:
        path = validate_file_type(path)
        validate_input_file_size(path)
        extension = path.suffix.lower()
        if extension == ".csv":
            dataframe, read_warnings = _read_csv(path)
            warnings.extend(read_warnings)
        elif extension in {".xls", ".xlsx"}:
            if extension == ".xlsx":
                validate_xlsx_archive(path)
            dataframe, resolved_sheet_name = _read_excel(
                path, normalized_sheet_name
            )
        else:
            dataframe, read_warnings = _read_json(path)
            warnings.extend(read_warnings)
        validate_dataframe_limits(dataframe)
    except ResourceLimitExceeded as error:
        raise DatasetReadError(str(error), warnings=warnings) from error
    except (DatasetReadError, UnsupportedFileTypeError) as error:
        error.warnings = list(dict.fromkeys([*warnings, *error.warnings]))
        raise

    string_columns = [str(column) for column in dataframe.columns]
    duplicate_columns = list(
        dict.fromkeys(
            column
            for column, duplicated in zip(
                string_columns, pd.Index(string_columns).duplicated(keep=False)
            )
            if duplicated
        )
    )
    if duplicate_columns:
        duplicates = "、".join(duplicate_columns)
        raise DatasetReadError(
            f"字段名转换为文本后存在重复：{duplicates}。"
            "请在评估前将字段名修改为唯一值。",
            warnings=warnings,
        )
    dataframe.columns = string_columns
    dataset = DatasetInfo(
        name=normalized_dataset_name or display_stem or display_file_name,
        file_name=display_file_name,
        file_type=extension.removeprefix("."),
        sheet_name=resolved_sheet_name,
    )
    return ParsedDataset(dataframe=dataframe, dataset=dataset, warnings=warnings)
