from __future__ import annotations

import asyncio
import http.client
import json
import logging
import socket
import string
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from autogen_mas.config import DashScopeSettings
logger = logging.getLogger(__name__)


class StructuredLLMClient(Protocol):
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float | None,
    ) -> dict[str, Any]:
        ...


# Substrings (lower-cased) that mark an "out of balance / quota" response, where
# retrying the same key is pointless but another key may still work.
_ARREARS_MARKERS = (
    "arrearage",
    "insufficient balance",
    "insufficient_quota",
    "insufficient user quota",
    "exceeded your current quota",
    "account balance",
    "欠费",
    "余额不足",
    "余额已用完",
)


def _read_http_error_body(error: urllib.error.HTTPError) -> str:
    try:
        return error.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _is_arrears_error(status_code: int, body: str) -> bool:
    """True when the response indicates the active key has no balance/quota left."""
    if status_code not in {400, 401, 402, 403, 429}:
        return False
    text = body.lower()
    return any(marker in text for marker in _ARREARS_MARKERS)


class ApiKeyPool:
    """Thread-safe rotating pool of API keys.

    When the active key runs out of balance, callers report it exhausted and the
    pool advances to the next usable key so the run continues uninterrupted.
    Shared across worker threads, so all mutation is guarded by a lock.
    """

    def __init__(self, keys: list[str]) -> None:
        seen: set[str] = set()
        ordered = [k for k in keys if k and not (k in seen or seen.add(k))]
        self._keys: list[str] = ordered or [""]
        self._index = 0
        self._exhausted: set[int] = set()
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._keys)

    def get_active(self) -> tuple[int, str]:
        with self._lock:
            return self._index, self._keys[self._index]

    def report_exhausted(self, used_index: int) -> tuple[int, str] | None:
        """Mark ``used_index`` exhausted; return the next usable (index, key) or None.

        Concurrency-safe: if another thread has already rotated past
        ``used_index``, the still-active key is returned instead of advancing
        further, so a single bad key only costs one rotation overall.
        """
        with self._lock:
            self._exhausted.add(used_index)
            index = self._index
            while index < len(self._keys) and index in self._exhausted:
                index += 1
            if index >= len(self._keys):
                self._index = len(self._keys) - 1
                return None
            self._index = index
            return index, self._keys[index]


class LLMGenerationError(RuntimeError):
    """Raised when the LLM call or response parsing fails."""


class LLMProviderError(LLMGenerationError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _provider_hint_for_model(
    model: str,
    *,
    route_models: Mapping[str, str] | None = None,
) -> str | None:
    normalized = model.strip().lower()
    if not normalized:
        return None
    if route_models:
        for provider, route_model in route_models.items():
            if normalized == route_model.strip().lower():
                return provider
    if normalized.startswith("deepseek"):
        return "deepseek"
    if normalized.startswith("qwen") or normalized.startswith("qwq"):
        return "dashscope"
    if normalized.startswith(("claude", "gemini")):
        if route_models and "anthropic" in route_models:
            return "anthropic"
        return "openai"
    if normalized.startswith("gemma") and route_models and "ollama" in route_models:
        return "ollama"
    if normalized.startswith(("gpt", "chatgpt", "o1", "o3", "o4")):
        return "openai"
    if route_models:
        for provider, route_model in route_models.items():
            route_model_normalized = route_model.strip().lower()
            if route_model_normalized and normalized.startswith(route_model_normalized):
                return provider
    return None


def _requires_dashscope_non_streaming_thinking_disabled(model: str) -> bool:
    return model.strip().lower().startswith("qwen3")


def _is_deepseek_v4(model: str) -> bool:
    return model.strip().lower().startswith("deepseek-v4-")


def _disable_thinking_for_json_request(
    payload: dict[str, Any],
    *,
    provider: str,
    model: str,
) -> None:
    """Keep structured JSON calls in the fast, non-thinking inference mode."""
    normalized_provider = provider.strip().lower()
    if normalized_provider == "dashscope" and (
        _requires_dashscope_non_streaming_thinking_disabled(model)
        or _is_deepseek_v4(model)
    ):
        payload["enable_thinking"] = False
    elif normalized_provider == "deepseek" and _is_deepseek_v4(model):
        payload["thinking"] = {"type": "disabled"}


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise LLMGenerationError("Model response is missing choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {None, "text"}
        ]
        return "".join(text_parts)
    raise LLMGenerationError("Unsupported message content format.")


def _anthropic_messages_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized + "/messages"
    return normalized + "/v1/messages"


def _extract_anthropic_message_content(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, list):
        raise LLMGenerationError("Anthropic response is missing content blocks.")
    text_parts = [
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    if not text_parts:
        raise LLMGenerationError("Anthropic response did not include text content.")
    return "".join(text_parts)


def _snippet(text: str, *, limit: int = 300) -> str:
    normalized = text.replace("\r", "\\r").replace("\n", "\\n")
    return normalized[:limit]


def _escape_invalid_json_string_escapes(text: str) -> str:
    """Preserve model output that contains code-style backslashes in JSON strings."""
    repaired: list[str] = []
    in_string = False
    index = 0
    valid_simple_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t"}
    hex_digits = set(string.hexdigits)
    control_char_escapes = {
        "\b": "\\b",
        "\f": "\\f",
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
    }

    while index < len(text):
        char = text[index]
        if not in_string:
            if char == '"':
                in_string = True
            repaired.append(char)
            index += 1
            continue

        if char == '"':
            in_string = False
            repaired.append(char)
            index += 1
            continue

        if ord(char) < 0x20:
            repaired.append(control_char_escapes.get(char, f"\\u{ord(char):04x}"))
            index += 1
            continue

        if char != "\\":
            repaired.append(char)
            index += 1
            continue

        if index + 1 >= len(text):
            repaired.append("\\\\")
            index += 1
            continue

        next_char = text[index + 1]
        if next_char in valid_simple_escapes:
            repaired.extend([char, next_char])
            index += 2
            continue
        if next_char == "u":
            hex_part = text[index + 2 : index + 6]
            if len(hex_part) == 4 and all(digit in hex_digits for digit in hex_part):
                repaired.append(text[index : index + 6])
                index += 6
            else:
                repaired.append("\\\\")
                index += 1
            continue

        repaired.append("\\\\")
        index += 1

    return "".join(repaired)


def _loads_json_with_repair(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as original_error:
        repaired = _escape_invalid_json_string_escapes(text)
        if repaired == text:
            raise
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            raise original_error


def _extract_json(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        parts = [part.strip() for part in text.split("```") if part.strip()]
        for part in parts:
            candidate = part
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            try:
                return _loads_json_with_repair(candidate)
            except json.JSONDecodeError:
                continue
    try:
        return _loads_json_with_repair(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            return _loads_json_with_repair(text[start : end + 1])
        raise


@dataclass(slots=True)
class DashScopeCompatClient:
    settings: DashScopeSettings
    _key_pool: ApiKeyPool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._key_pool = ApiKeyPool(self.settings.api_keys or [self.settings.api_key])

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float | None,
    ) -> dict[str, Any]:
        endpoint = self.settings.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        _disable_thinking_for_json_request(
            payload,
            provider=self.settings.provider,
            model=model,
        )
        body = json.dumps(payload).encode("utf-8")

        last_error: Exception | None = None
        last_error_body = ""
        # Outer loop rotates API keys; each iteration runs the full network /
        # rate-limit retry budget against the active key. A key is only ever
        # rotated when the response says it is out of balance/quota.
        while True:
            key_index, api_key = self._key_pool.get_active()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            arrears_body: str | None = None
            for attempt in range(self.settings.max_retries + 1):
                request = urllib.request.Request(
                    endpoint, data=body, headers=headers, method="POST"
                )
                try:
                    with urllib.request.urlopen(
                        request,
                        timeout=self.settings.timeout_seconds,
                    ) as response:
                        response_body = response.read().decode("utf-8")
                    payload_json = json.loads(response_body)
                    raw_content = _extract_message_content(payload_json)
                    return _extract_json(raw_content)
                except urllib.error.HTTPError as error:
                    error_body = _read_http_error_body(error)
                    last_error = error
                    last_error_body = error_body[:500]
                    if _is_arrears_error(error.code, error_body):
                        arrears_body = error_body
                        break
                    if error.code == 429 and attempt < self.settings.max_retries:
                        retry_after = error.headers.get("Retry-After")
                        delay = (
                            float(retry_after)
                            if retry_after
                            else min(10 * (attempt + 1), 30)
                        )
                        time.sleep(delay)
                        continue
                    if attempt >= self.settings.max_retries or error.code in {401, 403}:
                        break
                    time.sleep(min(2**attempt, 5))
                except (
                    http.client.RemoteDisconnected,
                    urllib.error.URLError,
                    socket.timeout,
                    TimeoutError,
                    json.JSONDecodeError,
                    LLMGenerationError,
                ) as error:
                    last_error = error
                    if attempt >= self.settings.max_retries:
                        break
                    time.sleep(min(2**attempt, 5))

            if arrears_body is None:
                # Terminal failure unrelated to balance; stop rotating.
                break

            next_key = self._key_pool.report_exhausted(key_index)
            if next_key is None:
                raise LLMProviderError(
                    f"All {self._key_pool.size} API key(s) are exhausted "
                    f"(out of balance/quota): {arrears_body[:300]}",
                    status_code=getattr(last_error, "code", None),
                )
            logger.warning(
                "API key #%d out of balance/quota; switching to key #%d "
                "and continuing the run.",
                key_index,
                next_key[0],
            )

        if isinstance(last_error, urllib.error.HTTPError):
            detail = f": {last_error_body}" if last_error_body else ""
            raise LLMProviderError(
                f"LLM generation failed after retries: HTTP Error "
                f"{last_error.code}: {last_error.reason}{detail}",
                status_code=last_error.code,
            )
        raise LLMGenerationError(f"LLM generation failed after retries: {last_error}")


@dataclass(slots=True)
class AutoGenOpenAICompatClient:
    settings: DashScopeSettings

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float | None,
    ) -> dict[str, Any]:
        try:
            from autogen_core.models import ModelFamily, SystemMessage, UserMessage
            from autogen_ext.models.openai import OpenAIChatCompletionClient
        except ModuleNotFoundError as error:
            raise LLMGenerationError(
                "AutoGen backend requested but autogen-core/autogen-ext are not installed."
            ) from error

        async def _invoke() -> dict[str, Any]:
            client_kwargs = {
                "model": model,
                "api_key": self.settings.api_key,
                "base_url": self.settings.base_url,
                "model_info": {
                    "vision": False,
                    "function_calling": False,
                    "json_output": False,
                    "family": ModelFamily.UNKNOWN,
                    "structured_output": False,
                },
            }
            if temperature is not None:
                client_kwargs["temperature"] = temperature
            client = OpenAIChatCompletionClient(**client_kwargs)
            try:
                result = await asyncio.wait_for(
                    client.create(
                        [
                            SystemMessage(content=system_prompt),
                            UserMessage(content=user_prompt, source="user"),
                        ],
                        json_output=True,
                    ),
                    timeout=self.settings.timeout_seconds,
                )
                content = result.content
                if isinstance(content, dict):
                    return content
                if hasattr(content, "model_dump"):
                    return content.model_dump()
                if hasattr(content, "dict"):
                    return content.dict()
                if isinstance(content, str):
                    if not content.strip():
                        raise LLMGenerationError("AutoGen returned empty text content.")
                    try:
                        return _extract_json(content)
                    except json.JSONDecodeError as error:
                        raise LLMGenerationError(
                            f"AutoGen returned non-JSON text: {content[:300]!r}"
                        ) from error
                raise LLMGenerationError(f"Unsupported AutoGen response content: {type(content)}")
            finally:
                close = getattr(client, "close", None)
                if close is not None:
                    await close()

        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                return asyncio.run(_invoke())
            except (asyncio.TimeoutError, TimeoutError):
                last_error = LLMGenerationError(
                    f"AutoGen request exceeded {self.settings.timeout_seconds}s"
                )
                if attempt >= self.settings.max_retries:
                    break
                time.sleep(min(2**attempt, 5))
            except LLMGenerationError as error:
                last_error = error
                if attempt >= self.settings.max_retries:
                    break
                time.sleep(min(2**attempt, 5))
        raise LLMGenerationError(f"AutoGen generation failed after retries: {last_error}")


@dataclass(slots=True)
class AnthropicMessagesCompatClient:
    settings: DashScopeSettings

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float | None,
    ) -> dict[str, Any]:
        endpoint = _anthropic_messages_endpoint(self.settings.base_url)
        payload = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": 4096,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.settings.api_key,
            "anthropic-version": "2023-06-01",
        }

        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.settings.timeout_seconds,
                ) as response:
                    response_body = response.read().decode("utf-8")
                if not response_body.strip():
                    raise LLMGenerationError("Anthropic returned empty response body.")
                payload_json = json.loads(response_body)
                raw_content = _extract_anthropic_message_content(payload_json)
                if not raw_content.strip():
                    raise LLMGenerationError("Anthropic returned empty text content.")
                try:
                    return _extract_json(raw_content)
                except json.JSONDecodeError as error:
                    stop_reason = str(payload_json.get("stop_reason", "")).strip()
                    reason_suffix = (
                        f" (stop_reason={stop_reason})" if stop_reason else ""
                    )
                    raise LLMGenerationError(
                        "Anthropic returned non-JSON text"
                        f"{reason_suffix}: {_snippet(raw_content)!r}"
                    ) from error
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code == 429 and attempt < self.settings.max_retries:
                    retry_after = error.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else min(10 * (attempt + 1), 30)
                    time.sleep(delay)
                    continue
                if attempt >= self.settings.max_retries or error.code in {400, 401, 403}:
                    break
                time.sleep(min(2**attempt, 5))
            except (
                http.client.RemoteDisconnected,
                urllib.error.URLError,
                socket.timeout,
                TimeoutError,
                json.JSONDecodeError,
                LLMGenerationError,
            ) as error:
                last_error = error
                if attempt >= self.settings.max_retries:
                    break
                time.sleep(min(2**attempt, 5))
        if isinstance(last_error, urllib.error.HTTPError):
            error_body = ""
            try:
                error_body = last_error.read().decode("utf-8")[:500]
            except Exception:
                error_body = ""
            detail = f": {error_body}" if error_body else ""
            raise LLMProviderError(
                f"LLM generation failed after retries: HTTP Error "
                f"{last_error.code}: {last_error.reason}{detail}",
                status_code=last_error.code,
            )
        raise LLMGenerationError(f"LLM generation failed after retries: {last_error}")


@dataclass(slots=True)
class RoutedStructuredLLMClient:
    default_client: StructuredLLMClient
    route_clients: dict[str, StructuredLLMClient]
    route_models: dict[str, str]

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float | None,
    ) -> dict[str, Any]:
        provider = _provider_hint_for_model(model, route_models=self.route_models)
        client = self.route_clients.get(provider or "", self.default_client)
        return client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
        )


def _build_single_structured_client(settings: DashScopeSettings) -> StructuredLLMClient:
    backend = settings.client_backend.lower()
    if settings.provider == "anthropic":
        if backend == "autogen":
            raise ValueError(
                "MAS_LLM_CLIENT backend 'autogen' is not supported for provider "
                f"'{settings.provider}'."
            )
        return AnthropicMessagesCompatClient(settings)
    if backend == "autogen":
        return AutoGenOpenAICompatClient(settings)
    if backend == "direct_http":
        return DashScopeCompatClient(settings)
    if backend == "auto":
        try:
            import autogen_core  # noqa: F401
            import autogen_ext  # noqa: F401
        except ModuleNotFoundError:
            return DashScopeCompatClient(settings)
        return AutoGenOpenAICompatClient(settings)
    raise ValueError(f"Unsupported MAS_LLM_CLIENT backend: {settings.client_backend}")


def build_structured_client(settings: DashScopeSettings) -> StructuredLLMClient:
    default_client = _build_single_structured_client(settings)
    model_routes = {
        provider: route
        for provider, route in settings.model_routes.items()
        if route.model.strip().lower() != settings.model.strip().lower()
    }
    if not model_routes:
        return default_client
    return RoutedStructuredLLMClient(
        default_client=default_client,
        route_clients={
            provider: _build_single_structured_client(route)
            for provider, route in model_routes.items()
        },
        route_models={
            provider: route.model for provider, route in model_routes.items()
        },
    )
