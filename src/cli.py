"""命令行运行入口：默认生成结构化 JSON 评估报告。"""

import argparse
from datetime import date
from pathlib import Path

from .report import UnsafeReportDestinationError, get_file_identity, save_report
from .workflow import build_profile_report


def paths_refer_to_same_file(input_path: Path, output_path: Path) -> bool:
    """判断输入与输出是否指向同一文件。

    ``resolve`` 覆盖相对路径与符号链接，``samefile`` 额外覆盖硬链接。
    对尚未存在的输出文件，仍可通过解析后路径防止直接覆盖。
    """

    try:
        if input_path.resolve(strict=False) == output_path.resolve(strict=False):
            return True
    except OSError:
        # 某些异常文件系统无法完成 resolve，继续尝试 samefile。
        pass

    try:
        return input_path.samefile(output_path)
    except (FileNotFoundError, OSError):
        return False


def ensure_distinct_output_path(input_path: Path, output_path: Path) -> None:
    """拒绝会覆盖原始数据的输出路径。"""

    if paths_refer_to_same_file(input_path, output_path):
        raise ValueError("输出路径不能与输入数据文件相同，以免覆盖原始数据。")


def parse_reference_date(value: str) -> date:
    """解析 ISO 日期，供命令行固定一次评估的时间基准。"""

    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "评估基准日期必须使用 YYYY-MM-DD 格式。"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description="生成政务数据集质量评估报告。")
    parser.add_argument(
        "input_file",
        help=(
            "待评估的 CSV、Excel、JSON、JSONL / NDJSON、GeoJSON "
            "或同构 JSON 分片 ZIP 文件路径。"
        ),
    )
    parser.add_argument("--name", help="报告中的数据集名称；默认使用文件名。")
    parser.add_argument("--sheet", help="Excel 工作表名称；默认读取第一个工作表。")
    parser.add_argument(
        "--reference-date",
        type=parse_reference_date,
        help="评估基准日期（YYYY-MM-DD）；默认使用运行当天。",
    )
    parser.add_argument(
        "--output",
        default="reports/report.json",
        help=(
            "报告输出路径，默认：reports/report.json；"
            "显式使用 .md 可输出 Markdown 报告。"
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output)
    try:
        ensure_distinct_output_path(input_path, output_path)
    except ValueError as error:
        parser.error(str(error))
    input_identity = get_file_identity(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_parent_identity = get_file_identity(output_path.parent)

    report = build_profile_report(
        args.input_file,
        args.name,
        args.sheet,
        reference_date=args.reference_date,
    )
    try:
        save_report(
            report,
            output_path,
            protected_file_identities=(
                (input_identity,) if input_identity is not None else ()
            ),
            expected_parent_identity=output_parent_identity,
        )
    except UnsafeReportDestinationError as error:
        parser.error(str(error))
    output_format = "结构化 JSON" if output_path.suffix.lower() == ".json" else "Markdown"
    print(f"已生成 {output_format} 质量评估报告：{output_path}")


if __name__ == "__main__":
    main()
