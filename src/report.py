"""report.json 的组装与写出。"""

import json
import os
from pathlib import Path
import secrets
import stat
import tempfile
from typing import Iterable

from .models import DatasetInfo, QualityReport
from .parser import ParsedDataset


def create_empty_report(dataset: DatasetInfo) -> QualityReport:
    """创建一次评估开始时的空报告。"""

    return QualityReport(dataset=dataset)


def create_profile_report(parsed_dataset: ParsedDataset, profile: dict) -> QualityReport:
    """将文件解析和数据画像结果组装为 report.json 的第一版内容。"""

    report = create_empty_report(parsed_dataset.dataset)
    report.profile = profile
    report.execution["warnings"] = [
        *parsed_dataset.warnings,
        *profile.get("warnings", []),
    ]
    return report


FileIdentity = tuple[int, int]


class UnsafeReportDestinationError(ValueError):
    """报告目标在安全检查中被拒绝。"""


def get_file_identity(path: str | Path) -> FileIdentity | None:
    """返回可用于识别硬链接的文件系统身份。"""

    try:
        path_stat = Path(path).stat()
    except OSError:
        return None
    return path_stat.st_dev, path_stat.st_ino


def _validate_destination_stat(
    destination_stat: os.stat_result,
    protected_file_identities: frozenset[FileIdentity],
) -> None:
    """拒绝符号链接、特殊文件和受保护的原始文件。"""

    if stat.S_ISLNK(destination_stat.st_mode):
        raise UnsafeReportDestinationError("报告输出路径不能是符号链接。")
    if not stat.S_ISREG(destination_stat.st_mode):
        raise UnsafeReportDestinationError(
            "已存在的报告输出路径不是普通文件。"
        )
    identity = destination_stat.st_dev, destination_stat.st_ino
    if identity in protected_file_identities:
        raise UnsafeReportDestinationError(
            "输出路径不能指向原始数据文件。"
        )


def _validate_destination_at(
    directory_fd: int,
    file_name: str,
    protected_file_identities: frozenset[FileIdentity],
) -> None:
    try:
        destination_stat = os.stat(
            file_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    _validate_destination_stat(destination_stat, protected_file_identities)


def _sync_directory(directory_fd: int) -> None:
    """尽力持久化目录项；不支持目录 fsync 的系统可安全忽略。"""

    try:
        os.fsync(directory_fd)
    except OSError:
        pass


def _save_report_with_directory_fd(
    payload: bytes,
    path: Path,
    protected_file_identities: frozenset[FileIdentity],
    expected_parent_identity: FileIdentity | None,
) -> None:
    """在 POSIX 系统上通过稳定的目录句柄完成原子写入。"""

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(path.parent, directory_flags)
    temporary_name = f".quality-report-{secrets.token_hex(12)}.tmp"
    temporary_fd: int | None = None
    temporary_exists = False
    try:
        opened_parent_stat = os.fstat(directory_fd)
        opened_parent_identity = (
            opened_parent_stat.st_dev,
            opened_parent_stat.st_ino,
        )
        if (
            expected_parent_identity is not None
            and opened_parent_identity != expected_parent_identity
        ):
            raise UnsafeReportDestinationError(
                "报告输出目录在评估期间发生了变化。"
            )
        _validate_destination_at(
            directory_fd,
            path.name,
            protected_file_identities,
        )
        temporary_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        temporary_flags |= getattr(os, "O_CLOEXEC", 0)
        temporary_flags |= getattr(os, "O_NOFOLLOW", 0)
        temporary_fd = os.open(
            temporary_name,
            temporary_flags,
            0o600,
            dir_fd=directory_fd,
        )
        temporary_exists = True
        with os.fdopen(temporary_fd, "wb") as temporary_file:
            temporary_fd = None
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        # 写入期间目标可能被并发替换，因此在原子替换前再检查一次。
        _validate_destination_at(
            directory_fd,
            path.name,
            protected_file_identities,
        )
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_exists = False
        _sync_directory(directory_fd)
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _save_report_portably(
    payload: bytes,
    path: Path,
    protected_file_identities: frozenset[FileIdentity],
    expected_parent_identity: FileIdentity | None,
) -> None:
    """在不支持 dir_fd 的系统上使用同目录原子替换。"""

    parent_identity = get_file_identity(path.parent)
    if (
        expected_parent_identity is not None
        and parent_identity != expected_parent_identity
    ):
        raise UnsafeReportDestinationError(
            "报告输出目录在评估期间发生了变化。"
        )
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=".quality-report-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    temporary_exists = True
    try:
        with os.fdopen(temporary_fd, "wb") as temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        # 防止临时文件创建后父目录被链接到其他位置。
        if get_file_identity(path.parent) != parent_identity:
            raise UnsafeReportDestinationError(
                "报告输出目录在写入期间发生了变化。"
            )
        try:
            destination_stat = path.lstat()
        except FileNotFoundError:
            pass
        else:
            _validate_destination_stat(
                destination_stat,
                protected_file_identities,
            )
        os.replace(temporary_path, path)
        temporary_exists = False
    finally:
        if temporary_exists:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def save_report(
    report: QualityReport,
    output_path: str | Path,
    *,
    protected_file_identities: Iterable[FileIdentity] = (),
    expected_parent_identity: FileIdentity | None = None,
) -> None:
    """将结构化报告原子保存为 UTF-8 JSON，且不跟随目标符号链接。"""

    path = Path(output_path)
    if path.name in {"", ".", ".."}:
        raise UnsafeReportDestinationError(
            "报告输出路径必须包含文件名。"
        )
    payload = json.dumps(
        report.to_dict(),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    protected_identities = frozenset(protected_file_identities)
    if os.name == "posix":
        _save_report_with_directory_fd(
            payload,
            path,
            protected_identities,
            expected_parent_identity,
        )
    else:
        _save_report_portably(
            payload,
            path,
            protected_identities,
            expected_parent_identity,
        )
