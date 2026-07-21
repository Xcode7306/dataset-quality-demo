"""输入资源上限与压缩工作簿预检。

这些上限是本地 Demo 的安全边界，不是政务数据的业务口径。
"""

from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import BadZipFile, ZipFile

if TYPE_CHECKING:
    import pandas as pd


MEBIBYTE = 1024 * 1024
MAX_INPUT_FILE_MIB = 50
MAX_INPUT_FILE_BYTES = MAX_INPUT_FILE_MIB * MEBIBYTE
MAX_XLSX_ENTRY_COUNT = 5_000
MAX_XLSX_UNCOMPRESSED_BYTES = 200 * MEBIBYTE
MAX_XLSX_COMPRESSION_RATIO = 1_000
MIN_RATIO_CHECK_BYTES = MEBIBYTE
MAX_DATASET_ROWS = 1_000_000
MAX_DATASET_COLUMNS = 10_000
MAX_DATASET_CELLS = 20_000_000
MAX_JSON_RECORDS = 200_000
MAX_JSON_NESTING_DEPTH = 100
MAX_JSON_OBJECT_PAIRS = MAX_DATASET_COLUMNS
MAX_JSON_TOTAL_PAIRS = 1_000_000
MAX_CELL_TEXT_BYTES = MEBIBYTE


class ResourceLimitExceeded(ValueError):
    """输入超出 Demo 可安全处理的资源上限。"""


def validate_input_file_size(path: Path) -> None:
    """在交给 pandas/openpyxl 前限制原始文件体积。"""

    try:
        size = path.stat().st_size
    except OSError as error:
        raise ResourceLimitExceeded(f"无法获取文件大小：{error}") from error
    if size > MAX_INPUT_FILE_BYTES:
        raise ResourceLimitExceeded(
            f"文件大小为 {size} 字节，超过本地 Demo 的 "
            f"{MAX_INPUT_FILE_MIB} MiB 上限。"
        )


def validate_upload_size(byte_count: int) -> None:
    """在写入临时目录前拒绝超限上传。"""

    if byte_count > MAX_INPUT_FILE_BYTES:
        raise ResourceLimitExceeded(
            f"上传内容为 {byte_count} 字节，超过本地 Demo 的 "
            f"{MAX_INPUT_FILE_MIB} MiB 上限。"
        )


def validate_xlsx_archive(path: Path) -> None:
    """在 openpyxl 解压前检查 ZIP 条目、展开体积和压缩比。

    损坏或非 ZIP 文件交由原有 Excel 解析边界产生统一错误。
    """

    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
    except (BadZipFile, OSError):
        return

    if len(entries) > MAX_XLSX_ENTRY_COUNT:
        raise ResourceLimitExceeded(
            f"Excel 压缩包包含 {len(entries)} 个条目，超过 "
            f"{MAX_XLSX_ENTRY_COUNT} 个的安全上限。"
        )

    expanded_size = sum(entry.file_size for entry in entries)
    if expanded_size > MAX_XLSX_UNCOMPRESSED_BYTES:
        raise ResourceLimitExceeded(
            f"Excel 解压后预计为 {expanded_size} 字节，超过 "
            f"{MAX_XLSX_UNCOMPRESSED_BYTES} 字节的安全上限。"
        )

    for entry in entries:
        if entry.file_size < MIN_RATIO_CHECK_BYTES:
            continue
        ratio = entry.file_size / max(entry.compress_size, 1)
        if ratio > MAX_XLSX_COMPRESSION_RATIO:
            raise ResourceLimitExceeded(
                "Excel 压缩包存在异常压缩比条目，"
                "为避免资源耗尽已停止解析。"
            )


def validate_dataframe_limits(dataframe: "pd.DataFrame") -> None:
    """校验解析后的行列规模与单元格文本长度。"""

    # run_demo.py 需要在系统 Python 中先读取上传常量，
    # 因此将 pandas 保留为真正解析时才加载的依赖。
    import pandas as pd

    row_count, column_count = dataframe.shape
    if row_count > MAX_DATASET_ROWS:
        raise ResourceLimitExceeded(
            f"数据集包含 {row_count} 条记录，超过 "
            f"{MAX_DATASET_ROWS} 条的 Demo 上限。"
        )
    if column_count > MAX_DATASET_COLUMNS:
        raise ResourceLimitExceeded(
            f"数据集包含 {column_count} 个字段，超过 "
            f"{MAX_DATASET_COLUMNS} 个的 Demo 上限。"
        )
    cell_count = row_count * column_count
    if cell_count > MAX_DATASET_CELLS:
        raise ResourceLimitExceeded(
            f"数据集展开后包含 {cell_count} 个单元格，超过 "
            f"{MAX_DATASET_CELLS} 个的 Demo 上限。"
        )

    for column in dataframe.columns:
        column_text = str(column)
        if len(column_text.encode("utf-8", errors="replace")) > MAX_CELL_TEXT_BYTES:
            raise ResourceLimitExceeded(
                f"字段名超过 {MAX_CELL_TEXT_BYTES} 字节的单项文本上限。"
            )
        series = dataframe[column]
        if not (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        ):
            continue
        for row_number, value in enumerate(series, start=2):
            if (
                isinstance(value, str)
                and len(value.encode("utf-8", errors="replace")) > MAX_CELL_TEXT_BYTES
            ):
                raise ResourceLimitExceeded(
                    f"第 {row_number} 行字段“{column_text}”的文本超过 "
                    f"{MAX_CELL_TEXT_BYTES} 字节上限。"
                )
