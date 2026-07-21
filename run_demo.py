"""使用项目虚拟环境启动 Demo，或执行交付验收测试。"""

import argparse
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

from src.resource_limits import MAX_INPUT_FILE_MIB


PROJECT_ROOT = Path(__file__).resolve().parent


def find_venv_python(project_root: Path = PROJECT_ROOT) -> Path:
    """查找项目虚拟环境中的 Python，兼容 macOS/Linux 与 Windows。"""

    candidates = (
        project_root / ".venv" / "bin" / "python",
        project_root / ".venv" / "Scripts" / "python.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "未找到项目虚拟环境。请先执行：python3 -m venv .venv，"
        "再安装 requirements.txt。"
    )


def build_command(check: bool, project_root: Path = PROJECT_ROOT) -> list[str]:
    """构造启动或验收命令，不依赖当前 shell 是否已激活环境。"""

    python = str(find_venv_python(project_root))
    if check:
        return [python, "-m", "unittest", "discover", "-s", "tests", "-v"]
    return [
        python,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
        "--server.maxUploadSize",
        str(MAX_INPUT_FILE_MIB),
        "--browser.gatherUsageStats",
        "false",
    ]


def execute_command(
    command: list[str],
    project_root: Path = PROJECT_ROOT,
    *,
    allow_user_shutdown: bool = False,
) -> int:
    """执行子进程，并按运行模式区分 Ctrl+C 的语义。

    交互式 Demo 中 Ctrl+C 是正常停服；验收检查中断则必须返回非零码。
    """

    try:
        return subprocess.run(command, cwd=project_root, check=False).returncode
    except KeyboardInterrupt:
        return 0 if allow_user_shutdown else 130


def reserve_local_port() -> int:
    """请求操作系统分配一个当前可用的本地 TCP 端口。"""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_for_streamlit_health(
    process: subprocess.Popen[bytes],
    port: int,
    *,
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.2,
) -> bool:
    """轮询 Streamlit 健康端点，子进程退出或超时时返回 False。"""

    health_url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urlopen(health_url, timeout=1.0) as response:
                if response.status == 200 and response.read().strip() == b"ok":
                    return True
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(poll_interval_seconds)
    return False


def stop_process(process: subprocess.Popen[bytes]) -> None:
    """可靠停止健康检查启动的子进程，不遗留后台服务。"""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_acceptance_check(project_root: Path = PROJECT_ROOT) -> int:
    """运行自动化回归，再启动真实 Streamlit 服务完成健康检查。"""

    test_result = execute_command(
        build_command(True, project_root),
        project_root,
        allow_user_shutdown=False,
    )
    if test_result != 0:
        return test_result

    try:
        port = reserve_local_port()
        server_command = build_command(False, project_root) + ["--server.port", str(port)]
        process = subprocess.Popen(
            server_command,
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    except OSError as error:
        print(f"Streamlit 健康检查无法启动：{error}")
        return 1
    try:
        if wait_for_streamlit_health(process, port):
            print(f"Streamlit 健康检查通过：http://127.0.0.1:{port}/_stcore/health")
            return 0
        print("Streamlit 健康检查失败：服务未在限时内就绪。")
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        stop_process(process)


def main() -> int:
    parser = argparse.ArgumentParser(description="启动或验收政务数据集质量评估 Demo。")
    parser.add_argument(
        "--check",
        action="store_true",
        help="执行自动化测试，并短暂启动本地网页完成健康检查后关闭。",
    )
    args = parser.parse_args()
    try:
        if args.check:
            return run_acceptance_check()
        command = build_command(False)
    except FileNotFoundError as error:
        parser.exit(1, f"错误：{error}\n")
    return execute_command(command, allow_user_shutdown=True)


if __name__ == "__main__":
    raise SystemExit(main())
