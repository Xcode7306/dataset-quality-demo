"""Store the optional page API key in the current user's OS credential store."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Callable


KEYCHAIN_SERVICE = "cn.codex.dataset-quality-demo.model-api"
KEYCHAIN_ACCOUNT = "streamlit-page-api-key"
MACOS_SECURITY_PATH = Path("/usr/bin/security")


class CredentialStoreError(RuntimeError):
    """Raised when the OS credential store cannot complete an operation."""


Runner = Callable[..., subprocess.CompletedProcess[str]]


def credential_store_available() -> bool:
    """Return whether this runtime can use the macOS login keychain."""

    return sys.platform == "darwin" and MACOS_SECURITY_PATH.is_file()


def _run_security(
    arguments: list[str],
    *,
    input_text: str | None = None,
    runner: Runner = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(
            [str(MACOS_SECURITY_PATH), *arguments],
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            # ``security ... -w`` may read from the controlling terminal even
            # when stdin is a pipe.  Detaching the child from the parent's TTY
            # makes stdin the only available password input channel.
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CredentialStoreError("系统凭据库暂时不可用。") from error


def load_model_api_key(*, runner: Runner = subprocess.run) -> str | None:
    """Load the saved API key without exposing it in command arguments."""

    if not credential_store_available():
        return None
    result = _run_security(
        [
            "find-generic-password",
            "-a",
            KEYCHAIN_ACCOUNT,
            "-s",
            KEYCHAIN_SERVICE,
            "-w",
        ],
        runner=runner,
    )
    if result.returncode == 44:
        return None
    if result.returncode != 0:
        raise CredentialStoreError("无法从系统凭据库读取 API Key。")
    value = result.stdout.rstrip("\r\n")
    return value or None


def save_model_api_key(
    api_key: str,
    *,
    runner: Runner = subprocess.run,
) -> None:
    """Save the API key to the login keychain using stdin, not argv."""

    if not credential_store_available():
        raise CredentialStoreError("当前系统不支持本机凭据库保存。")
    result = _run_security(
        [
            "add-generic-password",
            "-U",
            "-a",
            KEYCHAIN_ACCOUNT,
            "-s",
            KEYCHAIN_SERVICE,
            "-D",
            "application password",
            "-l",
            "政务数据集质量评估 Demo API Key",
            "-w",
        ],
        # With ``-w`` as the final argument and no value, ``security`` prompts
        # for the password twice.  Supplying both answers over stdin keeps the
        # secret out of argv (and therefore out of process listings) while
        # still satisfying the command's confirmation prompt.
        input_text=f"{api_key}\n{api_key}\n",
        runner=runner,
    )
    if result.returncode != 0:
        raise CredentialStoreError("无法将 API Key 保存到系统凭据库。")


def delete_model_api_key(*, runner: Runner = subprocess.run) -> None:
    """Delete the saved API key; a missing item is already cleared."""

    if not credential_store_available():
        return
    result = _run_security(
        [
            "delete-generic-password",
            "-a",
            KEYCHAIN_ACCOUNT,
            "-s",
            KEYCHAIN_SERVICE,
        ],
        runner=runner,
    )
    if result.returncode not in {0, 44}:
        raise CredentialStoreError("无法从系统凭据库删除 API Key。")


__all__ = [
    "CredentialStoreError",
    "credential_store_available",
    "delete_model_api_key",
    "load_model_api_key",
    "save_model_api_key",
]
