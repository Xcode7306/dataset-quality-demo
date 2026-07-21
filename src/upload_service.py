"""网页上传文件到确定性质量报告的应用服务。"""

import re
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from .models import QualityReport
from .parser import DatasetReadError, SUPPORTED_EXTENSIONS, UnsupportedFileTypeError
from .resource_limits import ResourceLimitExceeded, validate_upload_size
from .text_utils import normalize_display_text
from .workflow import build_profile_report


_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_SURROGATE_PATTERN = re.compile(r"[\ud800-\udfff]")
_WINDOWS_UNSAFE_PATTERN = re.compile(r'[<>:"|?*]')
_SAFE_EXTENSION_PATTERN = re.compile(r"\.[A-Za-z0-9]{1,10}")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _clean_file_name_component(file_name: str) -> str:
    """以 POSIX 和 Windows 两种分隔符提取文件名并移除危险字符。"""

    normalized_name, _ = normalize_display_text(file_name)
    leaf_name = re.split(r"[\\/]", normalized_name)[-1]
    leaf_name = _CONTROL_CHAR_PATTERN.sub("_", leaf_name)
    leaf_name = _SURROGATE_PATTERN.sub("_", leaf_name)
    leaf_name = _WINDOWS_UNSAFE_PATTERN.sub("_", leaf_name)
    return leaf_name.strip().strip(".")


def _normalize_safe_extension(safe_extension: str | None) -> str:
    if not safe_extension:
        return ""
    extension = f".{safe_extension.lstrip('.')}"
    if not _SAFE_EXTENSION_PATTERN.fullmatch(extension):
        raise ValueError(f"不安全的文件扩展名：{safe_extension}。")
    return extension.lower()


def sanitize_file_name(
    file_name: str,
    *,
    default_name: str = "uploaded_dataset",
    safe_extension: str | None = None,
    max_length: int = 120,
    max_bytes: int = 255,
) -> str:
    """生成可用于报告展示或下载的跨平台安全文件名。

    ``safe_extension`` 由调用方在校验类型后传入，保证文件名
    被截断时仍保留该扩展名。``max_length`` 限制 Unicode
    字符数，``max_bytes`` 同时限制 UTF-8 字节数。
    """

    extension = _normalize_safe_extension(safe_extension)
    if max_length <= len(extension):
        raise ValueError("max_length 必须大于保留扩展名的长度。")
    extension_bytes = len(extension.encode("utf-8"))
    if max_bytes <= extension_bytes:
        raise ValueError("max_bytes 必须大于保留扩展名的 UTF-8 字节数。")

    cleaned_name = _clean_file_name_component(file_name)
    cleaned_default = _clean_file_name_component(default_name) or "uploaded_dataset"

    if extension:
        if cleaned_name.lower().endswith(extension):
            stem = cleaned_name[: -len(extension)]
        else:
            stem = cleaned_name
        if cleaned_default.lower().endswith(extension):
            default_stem = cleaned_default[: -len(extension)]
        else:
            default_stem = cleaned_default
    else:
        candidate_extension = Path(cleaned_name).suffix
        extension = (
            candidate_extension
            if _SAFE_EXTENSION_PATTERN.fullmatch(candidate_extension)
            else ""
        )
        stem = cleaned_name[: -len(extension)] if extension else cleaned_name
        default_stem = Path(cleaned_default).stem or "uploaded_dataset"

    stem = stem.strip().strip(".") or default_stem.strip().strip(".")
    stem = stem or "uploaded_dataset"
    if stem.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        stem = f"_{stem}"

    available_length = max_length - len(extension)
    available_bytes = max_bytes - extension_bytes
    truncated_stem: list[str] = []
    used_bytes = 0
    for character in stem[:available_length]:
        character_bytes = len(character.encode("utf-8"))
        if used_bytes + character_bytes > available_bytes:
            break
        truncated_stem.append(character)
        used_bytes += character_bytes
    stem = "".join(truncated_stem).rstrip(" .") or "_"
    return f"{stem}{extension}"


def evaluate_uploaded_dataset(
    content: bytes,
    file_name: str,
    dataset_name: str | None = None,
    sheet_name: str | None = None,
    reference_date: date | None = None,
) -> QualityReport:
    """在临时目录中评估上传内容，并返回与 CLI 相同的报告对象。

    本函数是网页层与评估引擎之间的边界。后续若接入 AI，应在本函数返回后
    只读消费 ``report.to_dict()``，将解释结果存放在独立对象中，不得改写报告。
    """

    normalized_file_name, file_name_replaced = normalize_display_text(file_name)
    cleaned_file_name = _clean_file_name_component(normalized_file_name)
    suffix = Path(cleaned_file_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"暂不支持 {suffix or '无扩展名'} 文件；支持类型为：{supported}。"
        )
    try:
        validate_upload_size(len(content))
    except ResourceLimitExceeded as error:
        raise DatasetReadError(str(error)) from error
    safe_file_name = sanitize_file_name(
        cleaned_file_name,
        default_name=f"uploaded_dataset{suffix}",
        safe_extension=suffix,
    )
    safe_dataset_name = (
        dataset_name
        if dataset_name is not None and str(dataset_name).strip()
        else Path(safe_file_name).stem
    )

    with TemporaryDirectory(prefix="dataset-quality-") as temporary_directory:
        # 用户文件名只用于报告展示，绝不参与临时路径构造。
        temporary_path = Path(temporary_directory) / f"upload{suffix}"
        temporary_path.write_bytes(content)
        report = build_profile_report(
            temporary_path,
            dataset_name=safe_dataset_name,
            sheet_name=sheet_name or None,
            reference_date=reference_date,
        )
        report.dataset.file_name = safe_file_name
        if file_name_replaced:
            warning = (
                "上传文件名包含无法表示为 UTF-8 的字符，"
                "已替换为 Unicode 替代字符。"
            )
            report.execution["warnings"] = list(
                dict.fromkeys([warning, *report.execution["warnings"]])
            )
        return report
