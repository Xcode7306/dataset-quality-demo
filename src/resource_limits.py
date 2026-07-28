"""输入资源上限与压缩文件预检。

这些上限是本地 Demo 的安全边界，不是政务数据的业务口径。
"""

import re
import stat
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from zipfile import (
    BadZipFile,
    ZIP_DEFLATED,
    ZIP_STORED,
    ZipFile,
    ZipInfo,
)

if TYPE_CHECKING:
    import pandas as pd


MEBIBYTE = 1024 * 1024
MAX_INPUT_FILE_MIB = 50
MAX_INPUT_FILE_BYTES = MAX_INPUT_FILE_MIB * MEBIBYTE
MAX_XLSX_ENTRY_COUNT = 5_000
MAX_XLSX_UNCOMPRESSED_BYTES = 200 * MEBIBYTE
MAX_XLSX_COMPRESSION_RATIO = 1_000
MIN_RATIO_CHECK_BYTES = MEBIBYTE
MAX_JSON_ZIP_ENTRY_COUNT = 1_000
MAX_JSON_ZIP_ENTRY_BYTES = MAX_INPUT_FILE_BYTES
MAX_JSON_ZIP_UNCOMPRESSED_BYTES = 200 * MEBIBYTE
MAX_JSON_ZIP_COMPRESSION_RATIO = 1_000
MAX_ZIP_MEMBER_NAME_BYTES = 1_024
MAX_DATASET_ROWS = 1_000_000
MAX_DATASET_COLUMNS = 10_000
MAX_DATASET_CELLS = 20_000_000
MAX_JSON_RECORDS = 200_000
MAX_JSON_ARRAY_ITEMS = MAX_JSON_RECORDS + 1
MAX_JSON_TOTAL_ARRAY_ITEMS = MAX_DATASET_CELLS + MAX_JSON_RECORDS + 1
MAX_JSON_NESTING_DEPTH = 100
MAX_JSON_OBJECT_PAIRS = MAX_DATASET_COLUMNS
MAX_JSON_TOTAL_PAIRS = 1_000_000
MAX_CELL_TEXT_BYTES = MEBIBYTE
JSON_ZIP_SHARD_EXTENSIONS = {".json", ".jsonl", ".ndjson"}
JSON_ZIP_ALLOWED_COMPRESSION_TYPES = {ZIP_STORED, ZIP_DEFLATED}
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")


class ResourceLimitExceeded(ValueError):
    """输入超出 Demo 可安全处理的资源上限。"""


def validate_input_file_size(path: Path) -> None:
    """在交给 pandas 与对应文件解析引擎前限制原始文件体积。"""

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


def _validated_zip_member_path(entry: ZipInfo) -> PurePosixPath:
    """校验 ZIP 成员名，拒绝跨平台路径穿越与路径歧义。"""

    member_name = entry.filename
    try:
        member_name_bytes = len(member_name.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ResourceLimitExceeded(
            "ZIP 包含无法安全表示为 UTF-8 的成员路径。"
        ) from error
    if member_name_bytes > MAX_ZIP_MEMBER_NAME_BYTES:
        raise ResourceLimitExceeded(
            f"ZIP 成员路径超过 {MAX_ZIP_MEMBER_NAME_BYTES} 字节上限。"
        )
    if not member_name or any(
        ord(character) < 32 or ord(character) == 127
        for character in member_name
    ):
        raise ResourceLimitExceeded("ZIP 包含空路径或控制字符路径。")

    portable_name = member_name.replace("\\", "/")
    if portable_name.startswith("/") or _WINDOWS_DRIVE_PATTERN.match(portable_name):
        raise ResourceLimitExceeded(
            f"ZIP 成员路径“{member_name}”是绝对路径，已拒绝读取。"
        )
    raw_parts = portable_name.split("/")
    if any(part in {".", ".."} for part in raw_parts):
        raise ResourceLimitExceeded(
            f"ZIP 成员路径“{member_name}”包含路径穿越片段，已拒绝读取。"
        )
    if any(":" in part for part in raw_parts):
        raise ResourceLimitExceeded(
            f"ZIP 成员路径“{member_name}”包含不安全的冒号路径片段。"
        )
    return PurePosixPath(portable_name)


def validate_json_zip_archive(archive: ZipFile) -> list[ZipInfo]:
    """预检 JSON 分片 ZIP，并返回按包内顺序排列的普通文件条目。"""

    entries = archive.infolist()
    if len(entries) > MAX_JSON_ZIP_ENTRY_COUNT:
        raise ResourceLimitExceeded(
            f"ZIP 包含 {len(entries)} 个条目，超过 "
            f"{MAX_JSON_ZIP_ENTRY_COUNT} 个的安全上限。"
        )

    expanded_size = sum(entry.file_size for entry in entries)
    if expanded_size > MAX_JSON_ZIP_UNCOMPRESSED_BYTES:
        raise ResourceLimitExceeded(
            f"ZIP 解压后预计为 {expanded_size} 字节，超过 "
            f"{MAX_JSON_ZIP_UNCOMPRESSED_BYTES} 字节的安全上限。"
        )

    compressed_size = sum(entry.compress_size for entry in entries)
    if expanded_size >= MIN_RATIO_CHECK_BYTES:
        archive_ratio = expanded_size / max(compressed_size, 1)
        if archive_ratio > MAX_JSON_ZIP_COMPRESSION_RATIO:
            raise ResourceLimitExceeded(
                "ZIP 整体压缩比异常，为避免资源耗尽已停止解析。"
            )

    shard_entries: list[ZipInfo] = []
    seen_paths: set[str] = set()
    for entry in entries:
        normalized_path = _validated_zip_member_path(entry)
        normalized_key = str(normalized_path).casefold()
        if normalized_key in seen_paths:
            raise ResourceLimitExceeded(
                f"ZIP 包含重复成员路径“{entry.filename}”，无法安全选择分片。"
            )
        seen_paths.add(normalized_key)

        unix_mode = entry.external_attr >> 16
        file_type = stat.S_IFMT(unix_mode)
        if file_type == stat.S_IFLNK:
            raise ResourceLimitExceeded(
                f"ZIP 成员“{entry.filename}”是符号链接，已拒绝读取。"
            )
        if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ResourceLimitExceeded(
                f"ZIP 成员“{entry.filename}”不是普通文件或目录。"
            )
        if entry.is_dir() or file_type == stat.S_IFDIR:
            continue
        if entry.flag_bits & 0x1:
            raise ResourceLimitExceeded(
                f"ZIP 成员“{entry.filename}”已加密，当前不支持读取。"
            )
        if entry.compress_type not in JSON_ZIP_ALLOWED_COMPRESSION_TYPES:
            raise ResourceLimitExceeded(
                f"ZIP 成员“{entry.filename}”使用了不支持的压缩算法"
                f"（方法 {entry.compress_type}）；当前仅允许 Stored 或 Deflated。"
            )
        if entry.file_size > MAX_JSON_ZIP_ENTRY_BYTES:
            raise ResourceLimitExceeded(
                f"ZIP 分片“{entry.filename}”展开后为 {entry.file_size} 字节，"
                f"超过单分片 {MAX_JSON_ZIP_ENTRY_BYTES} 字节上限。"
            )
        if entry.file_size >= MIN_RATIO_CHECK_BYTES:
            ratio = entry.file_size / max(entry.compress_size, 1)
            if ratio > MAX_JSON_ZIP_COMPRESSION_RATIO:
                raise ResourceLimitExceeded(
                    f"ZIP 分片“{entry.filename}”压缩比异常，"
                    "为避免资源耗尽已停止解析。"
                )
        if normalized_path.suffix.lower() not in JSON_ZIP_SHARD_EXTENSIONS:
            supported = "、".join(sorted(JSON_ZIP_SHARD_EXTENSIONS))
            raise ResourceLimitExceeded(
                f"ZIP 成员“{entry.filename}”不是支持的 JSON 分片；"
                f"包内仅允许 {supported} 普通文件。"
            )
        shard_entries.append(entry)

    if not shard_entries:
        raise ResourceLimitExceeded("ZIP 中未找到可读取的 JSON 分片。")
    return shard_entries


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
