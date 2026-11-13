"""Explicit, bounded OpenAI-compatible cloud generation without provider fallback."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Sequence
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from page_cite_rag.generation import (
    ChatMessage,
    GenerationConfig,
    GenerationResult,
    GenerationStats,
    ModelError,
    ModelTimeoutError,
    OutputFailureAccounting,
    OutputValidationError,
    T,
    check_budget,
)


def validate_api_base(api_base: str) -> str:
    """Require HTTPS, except explicit localhost HTTP; reject embedded credentials."""
    try:
        url = urlsplit(api_base)
        valid = (
            bool(url.hostname)
            and (
                url.scheme == "https"
                or (url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost"))
            )
            and url.username is None
            and url.password is None
            and not url.query
            and not url.fragment
        )
        url.port
        # httpx also rejects malformed hosts/URLs without making a request.
        httpx.URL(api_base)
    except (ValueError, httpx.InvalidURL):
        valid = False
    if not valid:
        raise ModelError("--api-base requires an HTTPS URL (HTTP only for 127.0.0.1/localhost)")
    return api_base.rstrip("/")


class ApiChatClient:
    """Opt-in generation; keys are read only from the named environment variable.

    No prepare_budget capability is provided: budget_for_model uses check_budget's
    conservative UTF-8 byte upper estimate, reserving schema, message and template
    overhead instead of assuming a provider tokenizer. Text tokens cannot exceed
    their UTF-8 byte count for byte-backed tokenizers; provider-specific template
    expansion still depends on the configured reserve and context limit.

    max_calls caps all API dispatches in this process, including Agent calls and
    failed HTTP attempts. Recreating a client never resets this safety ceiling.
    Usage totals include every response with valid usage, including rejected output.
    """

    thinking_control = "provider_default"
    _process_lock = threading.Lock()
    _process_calls = 0
    _process_prompt_tokens = 0
    _process_completion_tokens = 0

    def __init__(
        self,
        config: GenerationConfig,
        *,
        api_base: str,
        model: str,
        api_key_env: str = "PAGECITE_API_KEY",
        max_calls: int = 100,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_base = validate_api_base(api_base)
        if not isinstance(model, str) or not model.strip():
            raise ModelError("--api-model is required for --backend api")
        if type(max_calls) is not int or max_calls < 1:
            raise ModelError("--api-max-calls must be a positive integer")
        if not isinstance(api_key_env, str) or not api_key_env.isidentifier():
            raise ModelError("--api-key-env must name an environment variable")
        self.config = config
        self._model = model
        self._api_key_env = api_key_env
        self.max_calls = max_calls
        self._read_key()
        self.identity = {"backend": "api", "api_host": urlsplit(api_base).hostname, "model": model}
        self._http = httpx.Client(
            timeout=config.timeout_seconds,
            transport=transport if transport is not None else httpx.HTTPTransport(retries=0),
            trust_env=False,
            follow_redirects=False,
        )
        self._deadline: float | None = None

    def _read_key(self) -> str:
        key = os.environ.get(self._api_key_env)
        if key is None or not key.strip():
            raise ModelError("API key environment variable is missing or empty")
        return key

    @property
    def chat_model_name(self) -> str:
        return self._model

    def chat_completions_base_url(self) -> str:
        return self._api_base

    def chat_sampling(self) -> dict:
        # GenerationConfig's existing generation profile uses temperature 0.0.
        return {"temperature": 0.0}

    @property
    def calls_used(self) -> int:
        return ApiChatClient._process_calls

    @property
    def token_totals(self) -> dict[str, int]:
        with self._process_lock:
            return {
                "prompt_tokens": ApiChatClient._process_prompt_tokens,
                "completion_tokens": ApiChatClient._process_completion_tokens,
                "total_tokens": (
                    ApiChatClient._process_prompt_tokens + ApiChatClient._process_completion_tokens
                ),
            }

    def before_agent_dispatch(self) -> None:
        """Check the process-wide ceiling; no GPU checks or consumed call yet."""
        with self._process_lock:
            if ApiChatClient._process_calls >= self.max_calls:
                raise ModelError("API process call limit reached; no retry")

    def _reserve_call(self) -> None:
        with self._process_lock:
            if ApiChatClient._process_calls >= self.max_calls:
                raise ModelError("API process call limit reached; no retry")
            ApiChatClient._process_calls += 1

    def _usage(self, payload: dict) -> tuple[int, int]:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            raise ModelError("API response omitted valid token usage")
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if type(prompt) is not int or prompt < 1 or type(completion) is not int or completion < 0:
            raise ModelError("API response omitted valid token usage")
        with self._process_lock:
            ApiChatClient._process_prompt_tokens += prompt
            ApiChatClient._process_completion_tokens += completion
        return prompt, completion

    def _validate_response(self, response: httpx.Response) -> dict:
        if not 200 <= response.status_code < 300:
            # Never echo arbitrary provider strings, even its purported error type:
            # a misconfigured provider can put credentials or content in that field.
            error_type = "provider_error"
            try:
                candidate = response.json().get("error", {}).get("type")
                if candidate in (
                    "invalid_request_error",
                    "authentication_error",
                    "permission_error",
                    "rate_limit_error",
                    "server_error",
                    "api_error",
                    "not_found_error",
                    "insufficient_quota",
                ):
                    error_type = candidate
            except (ValueError, AttributeError, TypeError):
                pass
            raise ModelError(f"API HTTP {response.status_code}: {error_type}")
        try:
            payload = response.json()
        except ValueError:
            raise ModelError("API response is not valid JSON") from None
        if not isinstance(payload, dict):
            raise ModelError("API response is not an object")
        return payload

    def dispatch_agent_request(
        self, inner: httpx.BaseTransport, request: httpx.Request
    ) -> httpx.Response:
        """Authenticate only at dispatch, keeping the key out of LangChain objects."""
        key = self._read_key()
        self._reserve_call()
        try:
            request.headers["authorization"] = f"Bearer {key}"
            response = inner.handle_request(request)
            response.read()
        except (httpx.HTTPError, OSError, ValueError, UnicodeError):
            raise ModelError("API connection failed; no retry") from None
        finally:
            request.headers.pop("authorization", None)
        payload = self._validate_response(response)
        self._usage(payload)
        return response

    def set_request_deadline(self, deadline: float) -> None:
        self._deadline = deadline

    def clear_request_deadline(self) -> None:
        self._deadline = None

    def generate(
        self,
        messages: Sequence[ChatMessage],
        response_model: type[T] | None,
        *,
        include_schema_in_prompt: bool = True,
    ) -> GenerationResult[T | str]:
        if type(include_schema_in_prompt) is not bool:
            raise ValueError("schema prompt inclusion must be boolean")
        messages = tuple(messages)
        schema = response_model.model_json_schema() if response_model is not None else None
        # The schema travels in response_format even when no prompt schema is requested.
        budget = check_budget(messages, schema, self.config)
        if self._http.is_closed:
            raise ModelError("API generation client is closed")
        timeout = self.config.timeout_seconds
        if self._deadline is not None:
            timeout = min(timeout, self._deadline - time.monotonic())
            if timeout <= 0:
                raise ModelTimeoutError("API request exceeded its total deadline")
        body = {
            "model": self._model,
            "messages": [message.model_dump() for message in messages],
            "max_tokens": self.config.num_predict,
            **self.chat_sampling(),
            "stream": False,
        }
        if response_model is not None:
            # Not strict: OpenAI-style strict mode demands every property be required and
            # additionalProperties false, which our optional/default fields are not. The
            # schema still guides the model; the reply is validated locally below.
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": response_model.__name__, "schema": schema, "strict": False},
            }
        request = self._http.build_request(
            "POST", f"{self._api_base}/chat/completions", json=body, timeout=timeout
        )
        key = self._read_key()
        self._reserve_call()
        started = time.perf_counter()
        try:
            request.headers["authorization"] = f"Bearer {key}"
            response = self._http.send(request)
        except (httpx.HTTPError, OSError, ValueError, UnicodeError):
            raise ModelError("API connection failed; no retry") from None
        finally:
            # Also scrub requests retained by transport failures, before they can escape.
            request.headers.pop("authorization", None)
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = self._validate_response(response)
        prompt, completion = self._usage(payload)
        try:
            choice = payload["choices"][0]
            message = choice["message"]
            content = message["content"]
            finish = choice["finish_reason"]
            if not isinstance(content, str):
                raise ValueError
        except (KeyError, IndexError, TypeError, ValueError):
            raise ModelError("API completion is malformed") from None
        accounting = OutputFailureAccounting(
            prompt_eval_count=prompt,
            eval_count=completion,
            elapsed_ms=elapsed_ms,
            stop_type="limit" if finish == "length" else "eos",
        )
        if finish == "length":
            raise OutputValidationError("output_limit", accounting, raw_model_output=content)
        if finish != "stop" or message.get("reasoning") or message.get("reasoning_content"):
            raise ModelError("API completion has an unsupported finish or reasoning output")
        if completion > self.config.num_predict:
            raise ModelError("API completion exceeded its output token cap")
        try:
            value = (
                content
                if response_model is None
                else response_model.model_validate_json(content, strict=True)
            )
        except ValidationError:
            raise OutputValidationError(
                "schema_invalid", accounting, raw_model_output=content
            ) from None
        duration = int(elapsed_ms * 1_000_000)
        stats = GenerationStats(
            prompt_eval_count=prompt,
            eval_count=completion,
            total_duration=duration,
            load_duration=None,
            prompt_eval_duration=0,
            eval_duration=0,
            done_reason="stop",
            elapsed_ms=elapsed_ms,
        )
        return GenerationResult(value, stats, budget, raw_model_output=content)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ApiChatClient:
        return self

    def __exit__(self, *_args) -> None:
        self.close()
