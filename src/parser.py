"""CSV、Excel、JSON 输入文件的解析模块。"""

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .models import DatasetInfo
from .resource_limits import (
    MAX_DATASET_CELLS,
    MAX_DATASET_COLUMNS,
    MAX_DATASET_ROWS,
    MAX_JSON_NESTING_DEPTH,
    MAX_JSON_OBJECT_PAIRS,
    MAX_JSON_RECORDS,
    MAX_JSON_TOTAL_PAIRS,
    ResourceLimitExceeded,
    validate_dataframe_limits,
    validate_input_file_size,
    validate_xlsx_archive,
)
from .text_utils import normalize_display_text


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".json"}


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


def validate_file_type(file_path: str | Path) -> Path:
    """检查文件存在且扩展名属于 v0.1 支持范围。"""

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
    duplicated = pd.Index(header_texts).duplicated(keep=False)
    duplicate_headers = list(
        dict.fromkeys(
            header_text
            for header_text, is_duplicate in zip(header_texts, duplicated)
            if is_duplicate
        )
    )
    if duplicate_headers:
        display_headers = "、".join(
            header or "（空字段名）" for header in duplicate_headers
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

    try:
        workbook = pd.ExcelFile(path)
    # pandas 会根据文件内容调用不同的 Excel 引擎。损坏的 ZIP、
    # XML 或工作簿元数据可以从这些引擎抛出 BadZipFile、OptionError、
    # KeyError 等多种异常；它们都属于用户输入无法解析，应在边界统一。
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
        return value.encode("utf-16", errors="surrogatepass").decode("utf-16")
    except UnicodeDecodeError as error:
        raise DatasetReadError(
            "JSON 包含孤立的 Unicode 代理字符，无法安全输出 UTF-8 报告。"
        ) from error


def _normalize_record_unicode(record: dict[str, Any]) -> dict[str, Any]:
    """校验并规范扁平 JSON 记录中的字段名和字符串值。"""

    normalized: dict[str, Any] = {}
    for key, value in record.items():
        normalized_key = _normalize_json_string(key)
        if normalized_key in normalized:
            raise DatasetReadError(
                f"JSON 字段名规范化后存在重复：{normalized_key}。"
            )
        normalized[normalized_key] = (
            _normalize_json_string(value) if isinstance(value, str) else value
        )
    return normalized


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """构造 JSON 对象时立即拒绝重复键，避免静默保留最后一个值。"""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            display_key, _ = normalize_display_text(key)
            raise DatasetReadError(f"JSON 对象存在重复字段：{display_key}。")
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


def _preflight_json_structure(path: Path) -> None:
    """在 json.load 物化数据前限制结构规模并拒绝嵌套。

    v0.1 只支持顶层扁平对象或扁平对象列表。这个轻量
    扫描器只跟踪字符串、转义和结构深度；完整语法仍交由
    标准 json 解析器校验。
    """

    top_level: str | None = None
    depth = 0
    in_string = False
    escaped = False
    array_value_started = False
    record_count = 0
    contains_nested_value = False
    container_types: dict[int, str] = {}
    object_pair_counts: dict[int, int] = {}
    total_pair_count = 0

    def finish_array_value() -> None:
        nonlocal array_value_started, record_count
        if not array_value_started:
            return
        record_count += 1
        array_value_started = False
        if record_count > MAX_JSON_RECORDS:
            raise DatasetReadError(
                f"JSON 顶层包含超过 {MAX_JSON_RECORDS} 条记录，"
                "为避免内存耗尽已停止解析。"
            )

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for chunk in iter(lambda: file.read(64 * 1024), ""):
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
                    if character == "[":
                        top_level = "array"
                        depth = 1
                        container_types[depth] = "array"
                        continue
                    if character == "{":
                        top_level = "object"
                        depth = 1
                        container_types[depth] = "object"
                        object_pair_counts[depth] = 0
                        continue
                    # 标量或其他非法顶层交给 json.load 生成统一语法错误。
                    return

                if character == '"':
                    if top_level == "array" and depth == 1:
                        array_value_started = True
                    in_string = True
                    continue

                if character in "[{":
                    if top_level == "array" and depth == 1:
                        array_value_started = True
                        if character != "{":
                            raise DatasetReadError(
                                "JSON 顶层列表中的每一项都必须是对象记录。"
                            )
                    else:
                        contains_nested_value = True
                    depth += 1
                    container_types[depth] = (
                        "array" if character == "[" else "object"
                    )
                    if character == "{":
                        object_pair_counts[depth] = 0
                    if depth > MAX_JSON_NESTING_DEPTH:
                        raise DatasetReadError("JSON 嵌套层级过深，无法安全解析。")
                    continue

                if character == ":" and container_types.get(depth) == "object":
                    object_pair_counts[depth] += 1
                    total_pair_count += 1
                    if object_pair_counts[depth] > MAX_JSON_OBJECT_PAIRS:
                        raise DatasetReadError(
                            f"JSON 单个对象包含超过 {MAX_JSON_OBJECT_PAIRS} 个键值对，"
                            "为避免内存耗尽已停止解析。"
                        )
                    if total_pair_count > MAX_JSON_TOTAL_PAIRS:
                        raise DatasetReadError(
                            f"JSON 全文件包含超过 {MAX_JSON_TOTAL_PAIRS} 个键值对，"
                            "为避免内存耗尽已停止解析。"
                        )
                    continue

                if character in "]}":
                    if top_level == "array" and depth == 1 and character == "]":
                        finish_array_value()
                        if contains_nested_value:
                            raise DatasetReadError(
                                "JSON 包含嵌套对象或列表，"
                                "第一版暂不支持自动展平。"
                            )
                        return
                    if top_level == "object" and depth == 1 and character == "}":
                        if contains_nested_value:
                            raise DatasetReadError(
                                "JSON 包含嵌套对象或列表，"
                                "第一版暂不支持自动展平。"
                            )
                        return
                    container_types.pop(depth, None)
                    object_pair_counts.pop(depth, None)
                    depth -= 1
                    continue

                if top_level == "array" and depth == 1:
                    if character == ",":
                        finish_array_value()
                    elif not character.isspace():
                        array_value_started = True


def _read_json(path: Path) -> pd.DataFrame:
    """读取记录列表或单条扁平记录 JSON。

    嵌套结构的展平规则需要单独讨论，第一版不擅自处理，避免改变业务含义。
    """

    seen_fields: set[str] = set()

    def limited_object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        record = _object_without_duplicate_keys(pairs)
        seen_fields.update(record)
        if len(seen_fields) > MAX_DATASET_COLUMNS:
            raise DatasetReadError(
                f"JSON 记录合并后超过 {MAX_DATASET_COLUMNS} 个字段，"
                "为避免内存耗尽已停止解析。"
            )
        return record

    try:
        _preflight_json_structure(path)
        with path.open("r", encoding="utf-8-sig") as file:
            payload = json.load(
                file,
                object_pairs_hook=limited_object_hook,
                parse_constant=_reject_nonstandard_json_constant,
                parse_int=_parse_json_integer,
                parse_float=_parse_json_float,
            )
    except DatasetReadError:
        raise
    except UnicodeDecodeError as error:
        raise DatasetReadError("JSON 不是 UTF-8 编码，第一版暂不支持读取。") from error
    except json.JSONDecodeError as error:
        raise DatasetReadError(f"JSON 格式错误：第 {error.lineno} 行第 {error.colno} 列。") from error
    except RecursionError as error:
        raise DatasetReadError("JSON 嵌套层级过深，无法安全解析。") from error
    except ValueError as error:
        raise DatasetReadError(f"JSON 内容无法读取：{error}") from error
    except OSError as error:
        raise DatasetReadError(f"JSON 文件无法读取：{error}") from error

    if isinstance(payload, list):
        if not all(isinstance(record, dict) for record in payload):
            raise DatasetReadError("JSON 顶层列表中的每一项都必须是对象记录。")
        if any(_contains_nested_value(record) for record in payload):
            raise DatasetReadError("JSON 包含嵌套对象或列表，第一版暂不支持自动展平。")
        cell_count = len(payload) * len(seen_fields)
        if cell_count > MAX_DATASET_CELLS:
            raise DatasetReadError(
                f"JSON 记录展开后需要 {cell_count} 个单元格，超过 "
                f"{MAX_DATASET_CELLS} 个的 Demo 上限。"
            )
        return pd.DataFrame([_normalize_record_unicode(record) for record in payload])

    if isinstance(payload, dict):
        if _contains_nested_value(payload):
            raise DatasetReadError("JSON 包含嵌套对象或列表，第一版暂不支持自动展平。")
        return pd.DataFrame([_normalize_record_unicode(payload)])

    raise DatasetReadError("JSON 顶层必须是记录列表或单条对象记录。")


def parse_dataset(
    file_path: str | Path,
    dataset_name: str | None = None,
    sheet_name: str | None = None,
) -> ParsedDataset:
    """将 v0.1 支持的文件解析为统一表格型数据对象。"""

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
        elif extension == ".xlsx":
            validate_xlsx_archive(path)
            dataframe, resolved_sheet_name = _read_excel(
                path, normalized_sheet_name
            )
        else:
            dataframe = _read_json(path)
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
