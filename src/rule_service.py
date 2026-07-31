"""上传文件的 v0.4 RulePack 增强重评服务与下载序列化。"""

from __future__ import annotations

import csv
from datetime import date
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

from .parser import (
    DatasetReadError,
    SUPPORTED_EXTENSIONS,
    UnsupportedFileTypeError,
    parse_dataset,
)
from .presentation import build_issue_location_rows
from .resource_limits import ResourceLimitExceeded, validate_upload_size
from .rule_engine import (
    RuleEvaluationResult,
    RulePackExecutionError,
    _evaluate_rule_pack_on_verified_dataframe,
)
from .rule_pack import RulePack, is_rule_pack_executable, validate_rule_pack
from .text_utils import normalize_display_text
from .upload_service import _clean_file_name_component, sanitize_file_name
from .workflow import build_profile_report


def _spreadsheet_safe_cell(value):
    """阻止不可信字段名在表格软件中被解释为公式。"""

    if not isinstance(value, str):
        return value
    if value.lstrip().startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return value


def _prepare_upload_name(
    file_name: str,
) -> tuple[str, str, bool]:
    normalized_file_name, file_name_replaced = normalize_display_text(file_name)
    cleaned_file_name = _clean_file_name_component(normalized_file_name)
    suffix = Path(cleaned_file_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"暂不支持 {suffix or '无扩展名'} 文件；支持类型为：{supported}。"
        )
    safe_file_name = sanitize_file_name(
        cleaned_file_name,
        default_name=f"uploaded_dataset{suffix}",
        safe_extension=suffix,
    )
    return safe_file_name, suffix, file_name_replaced


def _apply_upload_display_metadata(
    report,
    *,
    safe_file_name: str,
    file_name_replaced: bool,
) -> None:
    report.dataset.file_name = safe_file_name
    if not file_name_replaced:
        return
    warning = (
        "上传文件名包含无法表示为 UTF-8 的字符，"
        "已替换为 Unicode 替代字符。"
    )
    report.execution["warnings"] = list(
        dict.fromkeys([warning, *report.execution["warnings"]])
    )


def evaluate_uploaded_dataset_with_rule_pack(
    content: bytes,
    file_name: str,
    rule_pack: RulePack,
    dataset_name: str | None = None,
    sheet_name: str | None = None,
    reference_date: date | None = None,
    *,
    selected_metric_ids: Iterable[str] | None = None,
) -> RuleEvaluationResult:
    """从上传字节重建基线，并以当前报告重新校验后执行已审批 RulePack。

    上传内容仍使用现有临时文件、解析器、画像/基线报告流程和全部资源上限。
    原始字节只写入临时目录，函数结束后自动清理。
    """

    safe_file_name, suffix, file_name_replaced = _prepare_upload_name(file_name)
    try:
        validate_upload_size(len(content))
    except ResourceLimitExceeded as error:
        raise DatasetReadError(str(error)) from error

    safe_dataset_name = (
        dataset_name
        if dataset_name is not None and str(dataset_name).strip()
        else Path(safe_file_name).stem
    )
    with TemporaryDirectory(prefix="dataset-quality-rule-") as temporary_directory:
        temporary_path = Path(temporary_directory) / f"upload{suffix}"
        temporary_path.write_bytes(content)

        baseline_report = build_profile_report(
            temporary_path,
            dataset_name=safe_dataset_name,
            sheet_name=sheet_name or None,
            reference_date=reference_date,
            selected_metric_ids=selected_metric_ids,
        )
        _apply_upload_display_metadata(
            baseline_report,
            safe_file_name=safe_file_name,
            file_name_replaced=file_name_replaced,
        )

        # 在再次读取 DataFrame 之前先基于当前报告做一次审批、草案哈希、
        # report hash 与 input hash 绑定复核；引擎入口还会再次复核。
        validation = validate_rule_pack(
            rule_pack,
            baseline_report,
            require_approved=True,
        )
        if not validation.valid:
            raise RulePackExecutionError(validation.errors)
        if not is_rule_pack_executable(rule_pack, baseline_report):
            raise RulePackExecutionError(
                ["RulePack 未通过当前上传报告的可执行性复核。"]
            )

        parsed_dataset = parse_dataset(
            temporary_path,
            dataset_name=safe_dataset_name,
            sheet_name=sheet_name or None,
        )
        return _evaluate_rule_pack_on_verified_dataframe(
            parsed_dataset.dataframe,
            baseline_report,
            rule_pack,
        )


# 名称明确表达“重新上传并重新评估”，供界面层按语义选用。
reevaluate_uploaded_dataset = evaluate_uploaded_dataset_with_rule_pack


def serialize_rule_evaluation_result(
    result: RuleEvaluationResult,
) -> bytes:
    """序列化严格 JSON；拒绝 NaN、Infinity 和不可编码对象。"""

    return result.to_json(indent=2).encode("utf-8")


def serialize_rule_issue_locations_csv(
    result: RuleEvaluationResult,
) -> bytes:
    """导出仅含新增规则问题位置且不含任何原始单元格值的 CSV。"""

    rows = build_issue_location_rows(
        result.enhanced_report,
        metric_keys=set(result.diff.added_metric_keys),
    )
    fieldnames = [
        "疑似问题类型",
        "指标名称",
        "字段名称",
        "数据记录序号",
        "关联记录序号",
        "备注",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(
        {
            key: _spreadsheet_safe_cell(value)
            for key, value in row.items()
        }
        for row in rows
    )
    return output.getvalue().encode("utf-8-sig")
