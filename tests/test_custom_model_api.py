"""页面自定义兼容模型 API 的 Provider 配置回归。"""

import unittest
from datetime import date
import json
from pathlib import Path

from src.agent_providers import OpenAICompatibleChatProvider
from src.rule_authoring_providers import OpenAICompatibleRuleAuthoringProvider
from src.rule_authoring_service import compile_rule_draft
from src.workflow import build_profile_report


ROOT = Path(__file__).resolve().parents[1]


class _FakeClient:
    def close(self):
        pass


class CustomModelApiProviderTests(unittest.TestCase):
    def test_chat_provider_uses_page_endpoint_key_and_model(self):
        provider = OpenAICompatibleChatProvider(
            api_key="page-only-key",
            api_url="https://model.example/v1",
            model="custom-chat-model",
            client_factory=lambda **options: _FakeClient(),
        )

        client, headers, endpoint = provider._make_client()

        self.assertIsInstance(client, _FakeClient)
        self.assertEqual(endpoint, "https://model.example/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer page-only-key")
        self.assertEqual(provider.model, "custom-chat-model")
        self.assertNotIn("page-only-key", provider.cache_namespace)

    def test_rule_authoring_provider_uses_page_endpoint_and_key(self):
        provider = OpenAICompatibleRuleAuthoringProvider(
            api_key="page-only-key",
            api_url="https://model.example/v1/chat/completions",
            model="custom-rule-model",
            client_factory=lambda **options: _FakeClient(),
        )

        client, headers = provider._client()

        self.assertIsInstance(client, _FakeClient)
        self.assertEqual(
            provider._api_endpoint(),
            "https://model.example/v1/chat/completions",
        )
        self.assertEqual(headers["Authorization"], "Bearer page-only-key")
        self.assertEqual(provider.model, "custom-rule-model")

    def test_custom_rule_provider_retries_without_json_mode_and_keeps_model_result(self):
        requests: list[dict] = []

        class FakeResponse:
            def __init__(self, status_code, payload):
                self.status_code = status_code
                self.payload = payload

            def json(self):
                return self.payload

            @property
            def text(self):
                return json.dumps(self.payload, ensure_ascii=False)

        class FakeClient:
            def post(self, url, *, headers, json):
                requests.append({"url": url, "headers": headers, "payload": json})
                if "response_format" in json:
                    return FakeResponse(400, {"error": "response_format unsupported"})
                return FakeResponse(
                    200,
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json_module.dumps(
                                        {
                                            "rule_type": "required",
                                            "fields": ["service_name"],
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                )

            def close(self):
                pass

        # Avoid shadowing the request parameter named json in FakeClient.post.
        json_module = json
        report = build_profile_report(
            ROOT / "sample_data" / "good_dataset.csv",
            reference_date=date(2026, 7, 17),
        )
        provider = OpenAICompatibleRuleAuthoringProvider(
            api_key="page-only-key",
            api_url="https://model.example/v1",
            model="custom-rule-model",
            client_factory=lambda **options: FakeClient(),
        )

        draft = compile_rule_draft(
            report,
            target_metric_id="db31_020100",
            user_intent="service_name为必填字段",
            provider=provider,
            allow_template_fallback=False,
        )

        self.assertEqual("draft", draft.status)
        self.assertEqual("required", draft.rule_spec.rule_type)
        self.assertEqual("model", draft.provider.mode)
        self.assertFalse(draft.provider.fallback_used)
        self.assertEqual(2, len(requests))
        self.assertIn("response_format", requests[0]["payload"])
        self.assertNotIn("response_format", requests[1]["payload"])

    def test_external_rule_failure_is_not_replaced_with_template(self):
        class FailingProvider:
            def generate(self, context, *, user_intent):
                del context, user_intent
                raise RuntimeError("rule endpoint unavailable")

        report = build_profile_report(
            ROOT / "sample_data" / "good_dataset.csv",
            reference_date=date(2026, 7, 17),
        )
        with self.assertRaisesRegex(RuntimeError, "rule endpoint unavailable"):
            compile_rule_draft(
                report,
                target_metric_id="db31_020100",
                user_intent="service_name为必填字段",
                provider=FailingProvider(),
                allow_template_fallback=False,
            )


if __name__ == "__main__":
    unittest.main()
