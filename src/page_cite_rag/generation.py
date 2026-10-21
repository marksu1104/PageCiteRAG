"""Small, injectable local generation boundary; no retrieval or agent workflow yet."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import time
from dataclasses import dataclass
from importlib.metadata import version
from typing import Generic, Literal, Protocol, Sequence, TypeVar
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_LLM = "qwen3.5:9b-q4_K_M"
DEFAULT_DIGEST = "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
BUDGET_VERSION = "utf8-byte-upper-estimate/message64/template-reserve/v1"
T = TypeVar("T", bound=BaseModel)


class BudgetConfig(Protocol):
    num_ctx: int
    num_predict: int
    template_reserve: int


class ModelError(RuntimeError):
    """Safe failure without prompts, answers or raw server errors in the message."""


class ContextBudgetError(ModelError):
    pass


class ModelTimeoutError(ModelError):
    pass


class StructuredOutputError(ModelError):
    pass


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    host: str = "http://127.0.0.1:11435"
    model: str = Field(default=DEFAULT_LLM, pattern=r"^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$")
    digest: str = Field(default=DEFAULT_DIGEST, pattern=r"^[0-9a-f]{64}$")
    runtime_version: str = "0.35.0"
    num_ctx: int = Field(default=16384, ge=2048, le=262144)
    num_predict: int = Field(default=2048, ge=1, le=8192)
    template_reserve: int = Field(default=1024, ge=256, le=8192)
    keep_alive: str = Field(default="30m", pattern=r"^[1-9][0-9]*(s|m|h)$")
    timeout_seconds: float = Field(default=120.0, gt=0, le=600, allow_inf_nan=False)
    num_thread: int = Field(default=6, ge=1, le=64)
    num_gpu: int = Field(default=0, ge=0, le=999)

    @model_validator(mode="after")
    def validate_local_and_budget(self) -> GenerationConfig:
        url = urlsplit(self.host)
        try:
            local = ipaddress.ip_address(url.hostname or "").is_loopback
            port = url.port
        except ValueError:
            local, port = False, None
        if (
            not local
            or url.scheme != "http"
            or port is None
            or url.username is not None
            or url.password is not None
            or url.path not in ("", "/")
            or url.query
            or url.fragment
        ):
            raise ValueError("generation host must be an explicit loopback HTTP address and port")
        if self.num_predict + self.template_reserve >= self.num_ctx:
            raise ValueError("context must leave space for the input and template")
        return self


@dataclass(frozen=True)
class PromptBudget:
    input_upper_estimate: int
    output_reserved: int
    context_limit: int
    method: str = BUDGET_VERSION
    exact_input_tokens: int | None = None


def check_budget(
    messages: Sequence[ChatMessage], schema: dict, config: BudgetConfig
) -> PromptBudget:
    """Conservative estimate, NOT a model tokenizer or proof of arbitrary template size.

    For the tested byte-level Qwen tokenizer we reserve one token per UTF-8 byte,
    plus message overhead, schema bytes and an explicit template margin. The server
    is also required to reject truncation. Reject rather than silently trim text.
    """
    if not messages or any(not isinstance(message, ChatMessage) for message in messages):
        raise ValueError("nonempty validated ChatMessage sequence required")
    input_upper = (
        sum(len(message.content.encode("utf-8")) + 64 for message in messages)
        + len(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
        + config.template_reserve
    )
    if input_upper + config.num_predict > config.num_ctx:
        raise ContextBudgetError("input exceeds the reserved context budget; split or reduce input")
    return PromptBudget(input_upper, config.num_predict, config.num_ctx)


def budget_for_model(model, messages, schema, config: BudgetConfig) -> PromptBudget:
    """Optional backend budget capability; old clients retain the byte upper bound.

    A supplied capability must return a valid budget for the same explicit
    configuration. Errors propagate; never silently fall back after a failure.
    Exact token counts still reserve the configured additional template margin.
    """
    prepare = getattr(model, "prepare_budget", None)
    if prepare is None:
        return check_budget(messages, schema, config)
    if not callable(prepare):
        raise ModelError("model budget capability is invalid")
    budget = prepare(messages, schema)
    if (
        not isinstance(budget, PromptBudget)
        or any(
            type(v) is not int or v < 0
            for v in (
                budget.input_upper_estimate,
                budget.output_reserved,
                budget.context_limit,
            )
        )
        or budget.output_reserved != config.num_predict
        or budget.context_limit != config.num_ctx
        or budget.input_upper_estimate + budget.output_reserved > budget.context_limit
        or type(budget.method) is not str
        or not budget.method
        or (
            budget.exact_input_tokens is not None
            and (
                type(budget.exact_input_tokens) is not int
                or budget.exact_input_tokens < 1
                or budget.input_upper_estimate
                != budget.exact_input_tokens + config.template_reserve
            )
        )
    ):
        raise ModelError("model budget capability returned incompatible metadata")
    return budget


class GenerationStats(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    prompt_eval_count: int = Field(ge=0)
    eval_count: int = Field(ge=0)
    total_duration: int = Field(ge=0)
    # Some resident backends cannot attribute a model load to one generation.
    load_duration: int | None = Field(ge=0)
    prompt_eval_duration: int = Field(ge=0)
    eval_duration: int = Field(ge=0)
    done_reason: Literal["stop"]
    elapsed_ms: float = Field(ge=0, allow_inf_nan=False)


class OutputFailureAccounting(BaseModel):
    """Validated usage of a returned completion, never a successful answer."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    prompt_eval_count: int = Field(ge=1)
    eval_count: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0, allow_inf_nan=False)
    stop_type: Literal["eos", "limit"]


class OutputValidationError(StructuredOutputError):
    """Only output-limit/schema failures after runtime and budget validation.

    Continuing requires an explicitly opted-in backend and runner. Response text
    is excluded from the exception message and runtime diagnostics; an
    evaluation runner may persist it with the failed sample. Other ModelError
    types remain fatal.
    """

    def __init__(
        self,
        reason: str,
        accounting: OutputFailureAccounting,
        *,
        raw_model_output: str | None = None,
        validation_errors: tuple[dict, ...] = (),
    ) -> None:
        if (
            not isinstance(accounting, OutputFailureAccounting)
            or reason not in ("output_limit", "schema_invalid")
            or accounting.stop_type != ("limit" if reason == "output_limit" else "eos")
            or (raw_model_output is not None and type(raw_model_output) is not str)
            or type(validation_errors) is not tuple
        ):
            raise ValueError("incompatible output failure accounting")
        super().__init__(f"local completion rejected: {reason}; no retry")
        self.reason = reason
        self.accounting = accounting
        # Kept off the exception message and runtime diagnostics. The local
        # evaluation runner may store it in its configured run directory.
        self.raw_model_output = raw_model_output
        self.validation_errors = validation_errors


@dataclass(frozen=True)
class GenerationResult(Generic[T]):
    value: T
    stats: GenerationStats
    budget: PromptBudget


class ModelClient(Protocol):
    """Model identity, bounded structured generation and explicit resource cleanup."""

    identity: dict

    def generate(
        self, messages: Sequence[ChatMessage], response_model: type[T]
    ) -> GenerationResult[T]: ...

    def close(self) -> None: ...


class OllamaModelClient:
    """Loopback-only, preinstalled pinned model. Never pulls, retries or executes tools.

    Uses the documented HTTP API because ollama-python 0.6.3 does not expose the
    server's truncate=False chat option. HTTP timeout/connection close is not a
    guarantee that the server immediately cancels computation.
    """

    def __init__(self, config: GenerationConfig, *, transport=None) -> None:
        import httpx

        self.config = config
        self._http = httpx.Client(
            base_url=config.host,
            timeout=httpx.Timeout(config.timeout_seconds, connect=min(5.0, config.timeout_seconds)),
            trust_env=False,
            follow_redirects=False,
            transport=transport,
        )
        self._closed = False
        try:
            runtime = self._request("/api/version").get("version")
            if runtime != config.runtime_version:
                raise ModelError("Ollama runtime version differs from the pinned configuration")
            self._verify_model()
            metadata = self._request("/api/show", {"model": config.model})
            if metadata.get("remote_model") or metadata.get("remote_host"):
                raise ModelError("remote/cloud models are not permitted by the local client")
            capabilities = metadata.get("capabilities", [])
            if not isinstance(capabilities, list) or not {"completion", "thinking"}.issubset(
                capabilities
            ):
                raise ModelError("model lacks the tested completion and thinking controls")
            details = metadata.get("details", {})
            model_info = metadata.get("model_info", {})
            if not isinstance(details, dict) or not isinstance(model_info, dict):
                raise ModelError("invalid model metadata")
            architecture = model_info.get("general.architecture")
            context = model_info.get(f"{architecture}.context_length")
            if type(context) is not int or context < config.num_ctx:
                raise ModelError("requested context is not supported by the model metadata")
            self.identity = {
                "backend": "ollama-http",
                "config": config.model_dump(),
                "runtime_version": runtime,
                "client_version": version("httpx"),
                "model_details": details,
                "model_context_limit": context,
                "template_sha256": hashlib.sha256(
                    str(metadata.get("template", "")).encode()
                ).hexdigest(),
                "budget_version": BUDGET_VERSION,
                "thinking": False,
                "truncate": False,
                "temperature": 0.0,
                "seed": 42,
            }
        except Exception:
            self.close()
            raise

    def _request(self, path: str, payload: dict | None = None) -> dict:
        import httpx

        if self._closed:
            raise ModelError("generation client is closed")
        start = time.perf_counter()
        try:
            with self._http.stream("GET" if payload is None else "POST", path, json=payload) as r:
                if not 200 <= r.status_code < 300:
                    raise ModelError(f"local Ollama request failed with HTTP {r.status_code}")
                body = bytearray()
                for chunk in r.iter_bytes():
                    if time.perf_counter() - start > self.config.timeout_seconds:
                        raise ModelTimeoutError(
                            "local generation response exceeded its time budget"
                        )
                    body.extend(chunk)
                    if len(body) > 2 * 1024 * 1024:
                        raise ModelError("local generation response exceeds its size budget")
                value = json.loads(body)
        except httpx.TimeoutException:
            raise ModelTimeoutError("local generation request timed out; not retried") from None
        except (httpx.HTTPError, OSError):
            raise ModelError("local Ollama connection failed; not retried") from None
        except (ValueError, UnicodeError):
            raise ModelError("local Ollama returned invalid JSON") from None
        if not isinstance(value, dict) or "error" in value:
            raise ModelError("local Ollama returned an invalid or failed response")
        return value

    def _verify_model(self) -> None:
        models = self._request("/api/tags").get("models")
        if not isinstance(models, list):
            raise ModelError("invalid local model list")
        matches = [m for m in models if isinstance(m, dict) and m.get("name") == self.config.model]
        if len(matches) != 1 or matches[0].get("digest") != self.config.digest:
            raise ModelError("local model is missing or its digest changed; prepare separately")
        if matches[0].get("remote_model") or matches[0].get("remote_host"):
            raise ModelError("remote/cloud models are not permitted by the local client")

    def loaded_state(self) -> list[dict]:
        """Non-content runtime observations; not proof of semantic quality or no reload."""
        models = self._request("/api/ps").get("models")
        if not isinstance(models, list):
            raise ModelError("invalid loaded model state")
        fields = ("name", "digest", "size", "size_vram", "expires_at", "context_length")
        return [
            {key: item[key] for key in fields if key in item}
            for item in models
            if isinstance(item, dict) and item.get("name") == self.config.model
        ]

    def generate(
        self, messages: Sequence[ChatMessage], response_model: type[T]
    ) -> GenerationResult[T]:
        messages = tuple(messages)
        schema = response_model.model_json_schema()
        budget = check_budget(messages, schema, self.config)
        self._verify_model()
        start = time.perf_counter()
        response = self._request(
            "/api/chat",
            {
                "model": self.config.model,
                "messages": [message.model_dump() for message in messages],
                "format": schema,
                "stream": False,
                "think": False,
                "truncate": False,
                "keep_alive": self.config.keep_alive,
                "options": {
                    "num_ctx": self.config.num_ctx,
                    "num_predict": self.config.num_predict,
                    "num_thread": self.config.num_thread,
                    "num_gpu": self.config.num_gpu,
                    "temperature": 0.0,
                    "seed": 42,
                    "presence_penalty": 0.0,
                    "repeat_penalty": 1.0,
                },
            },
        )
        message = response.get("message")
        if (
            response.get("model") != self.config.model
            or response.get("done") is not True
            or response.get("done_reason") != "stop"
            or not isinstance(message, dict)
            or message.get("role") != "assistant"
            or message.get("thinking")
            or message.get("tool_calls")
        ):
            raise StructuredOutputError(
                "generation incomplete, thinking enabled or unexpected tools"
            )
        if type(response.get("load_duration")) is not int:
            raise StructuredOutputError("Ollama load timing is missing or invalid")
        try:
            value = response_model.model_validate_json(message.get("content", ""), strict=True)
            stats = GenerationStats(
                **{
                    key: response[key]
                    for key in (
                        "prompt_eval_count",
                        "eval_count",
                        "total_duration",
                        "load_duration",
                        "prompt_eval_duration",
                        "eval_duration",
                        "done_reason",
                    )
                },
                elapsed_ms=(time.perf_counter() - start) * 1000,
            )
        except (ValidationError, TypeError, KeyError):
            raise StructuredOutputError(
                "generation did not satisfy its schema or runtime metrics"
            ) from None
        if (
            stats.prompt_eval_count < 1
            or stats.prompt_eval_count > budget.input_upper_estimate
            or stats.prompt_eval_count + self.config.num_predict > self.config.num_ctx
            or stats.eval_count > self.config.num_predict
        ):
            raise ContextBudgetError("runtime token counts violate the configured budget")
        return GenerationResult(value, stats, budget)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._http.close()

    def __enter__(self) -> OllamaModelClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()
