"""一键启动与验收入口测试。"""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from run_demo import (
    PROJECT_ROOT,
    build_command,
    execute_command,
    find_venv_python,
    run_acceptance_check,
    stop_process,
)


class RunDemoTests(unittest.TestCase):
    def test_start_command_uses_project_venv_and_streamlit(self):
        command = build_command(False)
        self.assertEqual(Path(command[0]), find_venv_python(PROJECT_ROOT))
        self.assertEqual(
            command[1:],
            [
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.headless",
                "true",
                "--server.address",
                "127.0.0.1",
                "--server.maxUploadSize",
                "50",
                "--browser.gatherUsageStats",
                "false",
            ],
        )

    def test_check_command_and_missing_environment_message(self):
        command = build_command(True)
        self.assertEqual(
            command[1:],
            ["-m", "unittest", "discover", "-s", "tests", "-v"],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "未找到项目虚拟环境"):
                find_venv_python(Path(temporary_directory))

    def test_ctrl_c_is_treated_as_normal_demo_shutdown(self):
        with patch("run_demo.subprocess.run", side_effect=KeyboardInterrupt):
            self.assertEqual(
                execute_command(["unused"], allow_user_shutdown=True),
                0,
            )

    def test_ctrl_c_fails_an_acceptance_command(self):
        with patch("run_demo.subprocess.run", side_effect=KeyboardInterrupt):
            self.assertEqual(
                execute_command(["unused"], allow_user_shutdown=False),
                130,
            )

    @patch("run_demo.stop_process")
    @patch("run_demo.wait_for_streamlit_health", return_value=True)
    @patch("run_demo.subprocess.Popen")
    @patch("run_demo.reserve_local_port", return_value=8765)
    @patch("run_demo.execute_command", return_value=0)
    def test_acceptance_runs_tests_then_real_health_check_and_stops_server(
        self,
        execute,
        _reserve_port,
        popen,
        wait_for_health,
        stop,
    ):
        process = popen.return_value

        with patch("builtins.print"):
            result = run_acceptance_check()

        self.assertEqual(result, 0)
        execute.assert_called_once()
        launched_command = popen.call_args.args[0]
        self.assertIn("--server.address", launched_command)
        self.assertEqual(launched_command[-2:], ["--server.port", "8765"])
        wait_for_health.assert_called_once_with(process, 8765)
        stop.assert_called_once_with(process)

    @patch("run_demo.subprocess.Popen")
    @patch("run_demo.execute_command", return_value=3)
    def test_acceptance_does_not_start_server_when_tests_fail(self, _execute, popen):
        self.assertEqual(run_acceptance_check(), 3)
        popen.assert_not_called()

    def test_stop_process_escalates_from_terminate_to_kill(self):
        process = unittest.mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("streamlit", 5), 0]

        stop_process(process)

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
