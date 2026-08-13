"""System credential-store behavior without touching the real keychain."""

import subprocess
import unittest
from unittest.mock import patch

from src.credential_store import (
    CredentialStoreError,
    delete_model_api_key,
    load_model_api_key,
    save_model_api_key,
)


class CredentialStoreTests(unittest.TestCase):
    @patch("src.credential_store.credential_store_available", return_value=True)
    def test_save_passes_secret_on_stdin_not_command_line(self, _available):
        calls = []

        def runner(arguments, **options):
            calls.append((arguments, options))
            return subprocess.CompletedProcess(arguments, 0, "", "")

        save_model_api_key("page-secret-value", runner=runner)

        self.assertEqual(1, len(calls))
        arguments, options = calls[0]
        self.assertNotIn("page-secret-value", arguments)
        self.assertEqual(
            "page-secret-value\npage-secret-value\n",
            options["input"],
        )
        self.assertEqual("-w", arguments[-1])

    @patch("src.credential_store.credential_store_available", return_value=True)
    def test_save_detaches_security_process_from_parent_tty(self, _available):
        calls = []

        def runner(arguments, **options):
            calls.append((arguments, options))
            return subprocess.CompletedProcess(arguments, 0, "", "")

        save_model_api_key("page-secret-value", runner=runner)

        _, options = calls[0]
        self.assertIs(options["start_new_session"], True)

    @patch("src.credential_store.credential_store_available", return_value=True)
    def test_save_failure_does_not_echo_secret_or_security_stderr(
        self,
        _available,
    ):
        secret = "page-secret-value"

        def runner(arguments, **options):
            del options
            return subprocess.CompletedProcess(
                arguments,
                1,
                "",
                f"security failed while processing {secret}",
            )

        with self.assertRaises(CredentialStoreError) as raised:
            save_model_api_key(secret, runner=runner)

        self.assertEqual(
            "无法将 API Key 保存到系统凭据库。",
            str(raised.exception),
        )
        self.assertNotIn(secret, str(raised.exception))

    @patch("src.credential_store.credential_store_available", return_value=True)
    def test_load_and_delete_use_stable_keychain_identity(self, _available):
        calls = []

        def runner(arguments, **options):
            calls.append((arguments, options))
            if "find-generic-password" in arguments:
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    "page-secret-value\n",
                    "",
                )
            return subprocess.CompletedProcess(arguments, 0, "", "")

        self.assertEqual(
            "page-secret-value",
            load_model_api_key(runner=runner),
        )
        delete_model_api_key(runner=runner)

        self.assertIn("find-generic-password", calls[0][0])
        self.assertIn("delete-generic-password", calls[1][0])
        self.assertNotIn("page-secret-value", calls[0][0])
        self.assertNotIn("page-secret-value", calls[1][0])

    @patch("src.credential_store.credential_store_available", return_value=True)
    def test_missing_keychain_item_loads_as_empty(self, _available):
        def runner(arguments, **options):
            del options
            return subprocess.CompletedProcess(arguments, 44, "", "missing")

        self.assertIsNone(load_model_api_key(runner=runner))
        delete_model_api_key(runner=runner)


if __name__ == "__main__":
    unittest.main()
