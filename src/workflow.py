"""端到端流程：文件解析 → 数据画像 → 指标 → 风险提示 → 报告。"""

from datetime import date
import hashlib
from pathlib import Path

from .config import ENGINE_VERSION, THRESHOLD_CONFIG_VERSION
from .metrics import calculate_all_metrics, calculate_failed_metrics
from .models import DatasetInfo, NotAssessableItem
from .parser import DatasetReadError, UnsupportedFileTypeError, parse_dataset
from .profiler import profile_dataframe
from .report import create_empty_report, create_profile_report
from .resource_limits import MAX_INPUT_FILE_BYTES
from .rules import generate_risks
from .text_utils import normalize_display_text


PARSER_PATHS = {
    ".csv": "csv",
    ".xls": "excel:xls",
    ".xlsx": "excel:xlsx",
    ".json": "json",
    ".jsonl": "json_lines",
    ".ndjson": "json_lines",
    ".geojson": "geojson",
    ".zip": "json_zip",
}


def _unique_messages(*message_groups: list[str]) -> list[str]:
    """合并告警并保持首次出现顺序。"""

    return list(dict.fromkeys(message for group in message_groups for message in group))


def _sync_not_assessable(report):
    """将指标层的无法计算结果同步到报告顶层。"""

    report.not_assessable = [
        NotAssessableItem(
            id=f"{metric.id}:{metric.field}" if metric.field else metric.id,
            name=f"{metric.name}（{metric.field}）" if metric.field else metric.name,
            reason=metric.reason or "当前无法计算。",
            metric_key=metric.metric_key,
        )
        for metric in report.metrics
        if metric.status == "not_assessable"
    ]
    return report


def _input_fingerprint(path: Path) -> tuple[str | None, int | None]:
    """流式计算输入摘要；不可读输入仍返回字段完备的上下文。"""

    try:
        input_size_bytes = path.stat().st_size
    except (OSError, UnicodeError, ValueError):
        return None, None
    if input_size_bytes > MAX_INPUT_FILE_BYTES:
        return None, input_size_bytes

    try:
        digest = hashlib.sha256()
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except (OSError, UnicodeError, ValueError):
        return None, input_size_bytes
    return digest.hexdigest(), input_size_bytes


def _evaluation_context(
    path: Path,
    reference_date: date,
) -> dict[str, str | int | None]:
    input_sha256, input_size_bytes = _input_fingerprint(path)
    return {
        "engine_version": ENGINE_VERSION,
        "reference_date": reference_date.isoformat(),
        "threshold_config_version": THRESHOLD_CONFIG_VERSION,
        "parser_path": PARSER_PATHS.get(path.suffix.lower(), "unsupported"),
        "input_sha256": input_sha256,
        "input_size_bytes": input_size_bytes,
        "report_sha256": None,
    }


def build_profile_report(
    file_path: str | Path,
    dataset_name: str | None = None,
    sheet_name: str | None = None,
    reference_date: date | None = None,
):
    """构建包含数据画像和当前已实现指标的结构化报告。"""

    path = Path(file_path)
    effective_reference_date = reference_date or date.today()
    evaluation_context = _evaluation_context(path, effective_reference_date)
    normalized_dataset_name = dataset_name
    metadata_warnings: list[str] = []
    if dataset_name is not None and str(dataset_name).strip():
        normalized_dataset_name, replaced_invalid_unicode = normalize_display_text(
            dataset_name
        )
        if replaced_invalid_unicode:
            metadata_warnings.append(
                "数据集名称包含无法表示为 UTF-8 的字符，"
                "已替换为 Unicode 替代字符。"
            )
    elif dataset_name is not None:
        normalized_dataset_name = None

    try:
        parsed_dataset = parse_dataset(path, normalized_dataset_name, sheet_name)
    except (DatasetReadError, UnsupportedFileTypeError) as error:
        display_file_name, file_name_replaced = normalize_display_text(path.name)
        display_stem, stem_replaced = normalize_display_text(path.stem)
        display_file_type, file_type_replaced = normalize_display_text(
            path.suffix.lower().removeprefix(".") or "unknown"
        )
        error_message, error_replaced = normalize_display_text(str(error))
        path_warnings = []
        if file_name_replaced or stem_replaced or file_type_replaced:
            path_warnings.append(
                "文件显示名称包含无法表示为 UTF-8 的字符，"
                "已替换为 Unicode 替代字符。"
            )
        if error_replaced:
            path_warnings.append(
                "错误详情包含无法表示为 UTF-8 的字符，"
                "已替换为 Unicode 替代字符。"
            )
        dataset = DatasetInfo(
            name=normalized_dataset_name or display_stem or display_file_name,
            file_name=display_file_name,
            file_type=display_file_type,
        )
        report = create_empty_report(dataset)
        report.status = "failed"
        report.evaluation_context = evaluation_context
        report.metrics = calculate_failed_metrics(error_message)
        report.risks = generate_risks(report.metrics)
        report.execution["warnings"] = _unique_messages(
            metadata_warnings,
            getattr(error, "warnings", []),
            path_warnings,
        )
        report.execution["errors"] = [error_message]
        return _sync_not_assessable(report)

    parsed_dataset.warnings = _unique_messages(
        metadata_warnings,
        parsed_dataset.warnings,
    )
    profile = profile_dataframe(parsed_dataset.dataframe)
    report = create_profile_report(parsed_dataset, profile)
    report.evaluation_context = evaluation_context
    report.metrics = calculate_all_metrics(
        parsed_dataset.dataframe, reference_date=effective_reference_date
    )
    report.risks = generate_risks(report.metrics)
    return _sync_not_assessable(report)
