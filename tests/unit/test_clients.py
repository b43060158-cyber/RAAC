from __future__ import annotations

import asyncio
import http.client
import io
import json
import sys
import urllib.error
from types import ModuleType
import unittest
from unittest.mock import patch

from autogen_mas.config import DashScopeSettings
from autogen_mas.runtime.clients import (
    AnthropicMessagesCompatClient,
    AutoGenOpenAICompatClient,
    DashScopeCompatClient,
    LLMGenerationError,
    LLMProviderError,
    RoutedStructuredLLMClient,
    build_structured_client,
    _extract_json,
)


class ClientsTest(unittest.TestCase):
    def test_extract_json_repairs_invalid_code_escapes(self) -> None:
        payload = _extract_json(
            '{"code": "import re\\npattern = \\"^\\\\d+\\\\s+\\\\w+$\\"\\nreturn re.match(pattern, text)", '
            '"reasoning": "regex uses \\d and \\s", '
            '"changed_answer": false, "change_drivers": [], "change_summary": ""}'
        )

        self.assertEqual(payload["code"], 'import re\npattern = "^\\d+\\s+\\w+$"\nreturn re.match(pattern, text)')
        self.assertEqual(payload["reasoning"], "regex uses \\d and \\s")

    def test_extract_json_repairs_invalid_escapes_in_fenced_json(self) -> None:
        payload = _extract_json(
            '```json\n{"code": "return path.split(\\"\\\\_\\")[0]", "reasoning": "handles \\_ separator"}\n```'
        )

        self.assertEqual(payload["code"], 'return path.split("\\_")[0]')
        self.assertEqual(payload["reasoning"], "handles \\_ separator")

    def test_extract_json_repairs_raw_newlines_inside_strings(self) -> None:
        payload = _extract_json(
            '{"code": "def add(a, b):\n    return a + b", "reasoning": "simple"}'
        )

        self.assertEqual(payload["code"], "def add(a, b):\n    return a + b")

    def test_extract_json_repairs_other_raw_control_characters_inside_strings(self) -> None:
        payload = _extract_json(
            '{"reasoning": "line one\x0bline two\x0cline three", "final_answer": "2"}'
        )

        self.assertEqual(payload["reasoning"], "line one\x0bline two\x0cline three")
        self.assertEqual(payload["final_answer"], "2")

    def test_dashscope_client_retries_remote_disconnect(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"'
                    b'{\\"selected_option_ids\\":[\\"A\\"],\\"reasoning\\":\\"ok\\"}'
                    b'"}}]}'
                )

        calls = []

        def fake_urlopen(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) == 1:
                raise http.client.RemoteDisconnected(
                    "Remote end closed connection without response"
                )
            return FakeResponse()

        client = DashScopeCompatClient(
            DashScopeSettings(api_key="test-key", max_retries=1)
        )

        with patch("urllib.request.urlopen", fake_urlopen):
            payload = client.generate_json(
                system_prompt="system",
                user_prompt="user",
                model="model",
                temperature=0.0,
            )

        self.assertEqual(payload["selected_option_ids"], ["A"])
        self.assertEqual(len(calls), 2)

    def test_dashscope_client_rotates_to_next_key_when_out_of_balance(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"'
                    b'{\\"selected_option_ids\\":[\\"A\\"],\\"reasoning\\":\\"ok\\"}'
                    b'"}}]}'
                )

        used_keys: list[str] = []

        def fake_urlopen(request, **kwargs):
            del kwargs
            used_keys.append(request.headers["Authorization"])
            # First key has no balance; second key works.
            if request.headers["Authorization"] == "Bearer key-empty":
                raise urllib.error.HTTPError(
                    request.full_url,
                    400,
                    "Bad Request",
                    {},
                    io.BytesIO(b'{"code":"Arrearage","message":"insufficient balance"}'),
                )
            return FakeResponse()

        client = DashScopeCompatClient(
            DashScopeSettings(api_key="", api_keys=["key-empty", "key-good"], max_retries=1)
        )

        with patch("urllib.request.urlopen", fake_urlopen):
            payload = client.generate_json(
                system_prompt="system",
                user_prompt="user",
                model="model",
                temperature=0.0,
            )

        self.assertEqual(payload["selected_option_ids"], ["A"])
        # Tried the empty key once (no wasted retries on arrears), then switched.
        self.assertEqual(used_keys, ["Bearer key-empty", "Bearer key-good"])

        # A later call should keep using the good key without retrying the empty one.
        used_keys.clear()
        with patch("urllib.request.urlopen", fake_urlopen):
            client.generate_json(
                system_prompt="system",
                user_prompt="user",
                model="model",
                temperature=0.0,
            )
        self.assertEqual(used_keys, ["Bearer key-good"])

    def test_dashscope_client_raises_when_all_keys_exhausted(self) -> None:
        def fake_urlopen(request, **kwargs):
            del kwargs
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "Bad Request",
                {},
                io.BytesIO(b'{"code":"Arrearage"}'),
            )

        client = DashScopeCompatClient(
            DashScopeSettings(api_key="", api_keys=["k1", "k2"], max_retries=0)
        )

        with patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaisesRegex(LLMProviderError, "exhausted"):
                client.generate_json(
                    system_prompt="system",
                    user_prompt="user",
                    model="model",
                    temperature=0.0,
                )

    def test_dashscope_qwen3_disables_thinking_for_non_streaming_json_call(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"'
                    b'{\\"selected_option_ids\\":[\\"A\\"],\\"reasoning\\":\\"ok\\"}'
                    b'"}}]}'
                )

        captured_payloads = []

        def fake_urlopen(request, **kwargs):
            del kwargs
            captured_payloads.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse()

        client = DashScopeCompatClient(
            DashScopeSettings(
                api_key="test-key",
                provider="dashscope",
                client_backend="direct_http",
            )
        )

        with patch("urllib.request.urlopen", fake_urlopen):
            client.generate_json(
                system_prompt="system",
                user_prompt="user",
                model="qwen3-32b",
                temperature=0.0,
            )

        self.assertEqual(captured_payloads[0]["enable_thinking"], False)

    def test_dashscope_deepseek_v4_disables_thinking(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"'
                    b'{\\"selected_option_ids\\":[\\"A\\"],\\"reasoning\\":\\"ok\\"}'
                    b'"}}]}'
                )

        captured_payloads = []

        def fake_urlopen(request, **kwargs):
            del kwargs
            captured_payloads.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse()

        client = DashScopeCompatClient(
            DashScopeSettings(
                api_key="test-key",
                provider="dashscope",
                client_backend="direct_http",
            )
        )

        with patch("urllib.request.urlopen", fake_urlopen):
            client.generate_json(
                system_prompt="system",
                user_prompt="user",
                model="deepseek-v4-flash",
                temperature=0.0,
            )

        self.assertEqual(captured_payloads[0]["enable_thinking"], False)
        self.assertNotIn("thinking", captured_payloads[0])

    def test_deepseek_v4_disables_thinking(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"'
                    b'{\\"selected_option_ids\\":[\\"A\\"],\\"reasoning\\":\\"ok\\"}'
                    b'"}}]}'
                )

        captured_payloads = []

        def fake_urlopen(request, **kwargs):
            del kwargs
            captured_payloads.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse()

        client = DashScopeCompatClient(
            DashScopeSettings(
                api_key="test-key",
                provider="deepseek",
                client_backend="direct_http",
            )
        )

        with patch("urllib.request.urlopen", fake_urlopen):
            client.generate_json(
                system_prompt="system",
                user_prompt="user",
                model="deepseek-v4-flash",
                temperature=0.0,
            )

        self.assertEqual(
            captured_payloads[0]["thinking"],
            {"type": "disabled"},
        )
        self.assertNotIn("enable_thinking", captured_payloads[0])

    def test_routed_client_uses_provider_hint_from_model_name(self) -> None:
        class RecordingClient:
            def __init__(self, name: str) -> None:
                self.name = name
                self.models: list[str] = []

            def generate_json(self, *, system_prompt, user_prompt, model, temperature):
                del system_prompt, user_prompt, temperature
                self.models.append(model)
                return {"provider": self.name}

        default_client = RecordingClient("default")
        dashscope_client = RecordingClient("dashscope")
        deepseek_client = RecordingClient("deepseek")
        client = RoutedStructuredLLMClient(
            default_client=default_client,
            route_clients={
                "dashscope": dashscope_client,
                "deepseek": deepseek_client,
            },
            route_models={
                "dashscope": "qwen3-32b",
                "deepseek": "deepseek-chat",
            },
        )

        self.assertEqual(
            client.generate_json(
                system_prompt="system",
                user_prompt="user",
                model="qwen3-32b",
                temperature=0.0,
            )["provider"],
            "dashscope",
        )
        self.assertEqual(
            client.generate_json(
                system_prompt="system",
                user_prompt="user",
                model="deepseek-chat",
                temperature=0.0,
            )["provider"],
            "deepseek",
        )
        self.assertEqual(
            client.generate_json(
                system_prompt="system",
                user_prompt="user",
                model="unknown-model",
                temperature=0.0,
            )["provider"],
            "default",
        )
        self.assertEqual(dashscope_client.models, ["qwen3-32b"])
        self.assertEqual(deepseek_client.models, ["deepseek-chat"])
        self.assertEqual(default_client.models, ["unknown-model"])

    def test_routed_client_prefers_anthropic_route_for_gemini_or_claude_models(self) -> None:
        class RecordingClient:
            def __init__(self, name: str) -> None:
                self.name = name
                self.models: list[str] = []

            def generate_json(self, *, system_prompt, user_prompt, model, temperature):
                del system_prompt, user_prompt, temperature
                self.models.append(model)
                return {"provider": self.name}

        default_client = RecordingClient("default")
        anthropic_client = RecordingClient("anthropic")
        client = RoutedStructuredLLMClient(
            default_client=default_client,
            route_clients={"anthropic": anthropic_client},
            route_models={"anthropic": "claude-sonnet-5"},
        )

        self.assertEqual(
            client.generate_json(
                system_prompt="system",
                user_prompt="user",
                model="claude-opus-4-6-thinking",
                temperature=0.0,
            )["provider"],
            "anthropic",
        )
        self.assertEqual(anthropic_client.models, ["claude-opus-4-6-thinking"])
        self.assertEqual(default_client.models, [])

    def test_routed_client_prefers_ollama_route_for_gemma_models(self) -> None:
        class RecordingClient:
            def __init__(self, name: str) -> None:
                self.name = name
                self.models: list[str] = []

            def generate_json(self, *, system_prompt, user_prompt, model, temperature):
                del system_prompt, user_prompt, temperature
                self.models.append(model)
                return {"provider": self.name}

        default_client = RecordingClient("default")
        ollama_client = RecordingClient("ollama")
        client = RoutedStructuredLLMClient(
            default_client=default_client,
            route_clients={"ollama": ollama_client},
            route_models={"ollama": "gemma3:4b"},
        )

        self.assertEqual(
            client.generate_json(
                system_prompt="system",
                user_prompt="user",
                model="gemma3:4b",
                temperature=0.0,
            )["provider"],
            "ollama",
        )
        self.assertEqual(ollama_client.models, ["gemma3:4b"])
        self.assertEqual(default_client.models, [])

    def test_build_structured_client_drops_same_model_routes(self) -> None:
        client = build_structured_client(
            DashScopeSettings(
                api_key="any",
                provider="ollama",
                base_url="http://127.0.0.1:11434/v1",
                model="gemma3:4b",
                client_backend="direct_http",
                model_routes={
                    "ollama": DashScopeSettings(
                        api_key="any",
                        provider="ollama",
                        base_url="http://localhost:11434/v1",
                        model="gemma3:4b",
                        client_backend="direct_http",
                    )
                },
            )
        )

        self.assertIsInstance(client, DashScopeCompatClient)
        self.assertEqual(client.settings.base_url, "http://127.0.0.1:11434/v1")

    def test_anthropic_client_uses_messages_endpoint(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"content":[{"type":"text","text":"{\\"ok\\": true}"}]}'

        captured_requests = []

        def fake_urlopen(request, **kwargs):
            del kwargs
            captured_requests.append(request)
            return FakeResponse()

        client = AnthropicMessagesCompatClient(
            DashScopeSettings(
                api_key="test-key",
                provider="anthropic",
                base_url="https://api.anthropic.com",
                client_backend="direct_http",
            )
        )

        with patch("urllib.request.urlopen", fake_urlopen):
            payload = client.generate_json(
                system_prompt="system",
                user_prompt="user",
                model="claude-sonnet-4-6",
                temperature=0.0,
            )

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(captured_requests[0].full_url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(captured_requests[0].headers["X-api-key"], "test-key")
        self.assertEqual(captured_requests[0].headers["Anthropic-version"], "2023-06-01")
        self.assertEqual(json.loads(captured_requests[0].data.decode("utf-8"))["max_tokens"], 4096)

    def test_anthropic_client_retries_empty_response_body(self) -> None:
        class EmptyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b""

        class JsonResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"content":[{"type":"text","text":"{\\"ok\\": true}"}]}'

        responses = [EmptyResponse(), JsonResponse()]

        def fake_urlopen(request, **kwargs):
            del request, kwargs
            return responses.pop(0)

        client = AnthropicMessagesCompatClient(
            DashScopeSettings(
                api_key="test-key",
                provider="anthropic",
                base_url="https://api.anthropic.com",
                client_backend="direct_http",
                max_retries=1,
            )
        )

        with patch("urllib.request.urlopen", fake_urlopen):
            payload = client.generate_json(
                system_prompt="system",
                user_prompt="user",
                model="gemini-3-flash",
                temperature=0.0,
            )

        self.assertEqual(payload, {"ok": True})

    def test_anthropic_client_reports_non_json_text_with_stop_reason(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"content":[{"type":"text","text":"The"}],"stop_reason":"max_tokens"}'

        client = AnthropicMessagesCompatClient(
            DashScopeSettings(
                api_key="test-key",
                provider="anthropic",
                base_url="https://api.anthropic.com",
                client_backend="direct_http",
                max_retries=0,
            )
        )

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            with self.assertRaisesRegex(LLMGenerationError, "stop_reason=max_tokens"):
                client.generate_json(
                    system_prompt="system",
                    user_prompt="user",
                    model="gemini-3-flash",
                    temperature=0.0,
                )

    def test_routed_client_supports_anthropic_compatible_routes(self) -> None:
        class RecordingClient:
            def __init__(self, name: str) -> None:
                self.name = name
                self.models: list[str] = []

            def generate_json(self, *, system_prompt, user_prompt, model, temperature):
                del system_prompt, user_prompt, temperature
                self.models.append(model)
                return {"provider": self.name}

        default_client = RecordingClient("default")
        anthropic_client = RecordingClient("anthropic")
        client = RoutedStructuredLLMClient(
            default_client=default_client,
            route_clients={"anthropic": anthropic_client},
            route_models={"anthropic": "alias-local-proxy"},
        )

        self.assertEqual(
            client.generate_json(
                system_prompt="system",
                user_prompt="user",
                model="claude-sonnet-4-6",
                temperature=0.0,
            )["provider"],
            "anthropic",
        )
        self.assertEqual(
            client.generate_json(
                system_prompt="system",
                user_prompt="user",
                model="alias-local-proxy",
                temperature=0.0,
            )["provider"],
            "anthropic",
        )
        self.assertEqual(anthropic_client.models, ["claude-sonnet-4-6", "alias-local-proxy"])
        self.assertEqual(default_client.models, [])

    def test_autogen_backend_times_out(self) -> None:
        class ModelFamily:
            UNKNOWN = "unknown"

        class SystemMessage:
            def __init__(self, content):
                self.content = content

        class UserMessage:
            def __init__(self, content, source):
                self.content = content
                self.source = source

        class SlowClient:
            def __init__(self, *args, **kwargs):
                pass

            async def create(self, *args, **kwargs):
                await asyncio.sleep(0.05)

            async def close(self):
                pass

        fake_modules = {
            "autogen_core": ModuleType("autogen_core"),
            "autogen_core.models": ModuleType("autogen_core.models"),
            "autogen_ext": ModuleType("autogen_ext"),
            "autogen_ext.models": ModuleType("autogen_ext.models"),
            "autogen_ext.models.openai": ModuleType("autogen_ext.models.openai"),
        }
        fake_modules["autogen_core.models"].ModelFamily = ModelFamily
        fake_modules["autogen_core.models"].SystemMessage = SystemMessage
        fake_modules["autogen_core.models"].UserMessage = UserMessage
        fake_modules["autogen_ext.models.openai"].OpenAIChatCompletionClient = SlowClient

        previous_modules = {
            name: sys.modules.get(name)
            for name in fake_modules
        }
        sys.modules.update(fake_modules)
        try:
            client = AutoGenOpenAICompatClient(
                DashScopeSettings(
                    api_key="test-key",
                    timeout_seconds=0.01,
                    max_retries=0,
                )
            )
            with self.assertRaisesRegex(LLMGenerationError, "exceeded"):
                client.generate_json(
                    system_prompt="system",
                    user_prompt="user",
                    model="model",
                    temperature=0.0,
                )
        finally:
            for name, module in previous_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
