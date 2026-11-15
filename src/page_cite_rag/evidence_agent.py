"""Serial, non-thinking evidence gathering with a shared quote-first writer."""

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated

import httpx

from page_cite_rag.api_backend import ApiChatClient
from page_cite_rag.full_answer import FullAnswerWorkflow
from page_cite_rag.generation import ModelError
from page_cite_rag.models import Block
from page_cite_rag.ollama_backend import OllamaGgufClient
from page_cite_rag.retrieval import SearchResult
from page_cite_rag.workflows import AnswerResult, Language

EVIDENCE_AGENT_VERSION = "evidence-agent/v5"


@dataclass(frozen=True)
class EvidenceAgentLimits:
    model_calls: int = 8
    # Extra dispatches per question for re-asking after the server rejected tool-call arguments
    # (seen when text before the call used up max_tokens and cut its JSON). A reply cut before
    # any tool call is not an error: it ends the search as in LangChain, and is only recorded;
    # re-asking it did not help on dev data (fix-check 2026-11-13: one same, one worse).
    tool_call_repairs: int = 1
    tool_calls: int = 16
    passages_per_search: int = 5
    query_chars: int = 300
    finish_passages: int = 12
    max_tokens: int = 512
    timeout_seconds: int = 600


LIMITS = EvidenceAgentLimits()

# Pinned Ollama 0.35 answers HTTP 500 with this message and drops the model text when the
# generated tool-call arguments are not valid JSON. Observed rejections took as long as a full
# 512-token reply: text before the call used up the cap and cut its JSON (2026-11-13).
TOOL_CALL_REJECTED = re.compile(r'invalid tool call arguments for "([^"]+)"')

# Every search adds up to five full passages to the conversation. On documents cut into
# 2,000-character blocks the sixth request outgrows the 16,384-token window (8 of 22
# long-dependency LooGLE questions failed, 2026-11-15). The pinned Ollama 0.35 then drops the
# oldest messages to fit, the question among them, and Qwen's chat template stops with HTTP 500
# "No user query found in messages." (reproduced on CPU). A single message too long to drop
# returns llama-server's HTTP 400 instead. Requests are checked before dispatch; these server
# answers remain a second line of defence.
CONTEXT_FULL = re.compile(
    r"exceeds the available context size|input length exceeds the context length"
    r"|context size has been exceeded|no user query found in messages",
    re.IGNORECASE,
)
# Chat-template tokens around one message (role markers, tool-response tags), with room to spare.
MESSAGE_TEMPLATE_TOKENS = 32


class AgentToolCallError(ModelError):
    """The server rejected the model's tool-call arguments and no correction is left."""


class AgentContextFull(ModelError):
    """The next agent request no longer fits the context window, so the search ends there."""


@dataclass(frozen=True)
class EvidenceAgentResult:
    answer: AnswerResult
    trace: dict


SYSTEM_PROMPT = (
    "You gather evidence for a question from a large collection of short encyclopedia passages. "
    "The question may need several steps: first find the person, place or thing it refers to, "
    "then find a fact about it, and so on.\n"
    "Work one fact at a time. Each search must be a short query naming one specific entity and "
    'the relation you need, for example "Kalvex Corp founder", never the whole question. '
    "When a passage reveals the entity needed for the next step, put that entity's name in your "
    "next search.\n"
    'Example (fictional): for "When was the founder of the company that makes the Zentro phone '
    'born?" search "Zentro phone manufacturer" (a passage says Kalvex Corp makes it), then '
    '"Kalvex Corp founder" (a passage says Mira Tal), then "Mira Tal born", then call finish '
    "with the IDs of those three passages.\n"
    "You have at most 8 steps in total. Your last step must be finish with the IDs of the passages "
    "that contain the needed facts, in reasoning order. Do not answer the question yourself."
)


def _validate_model_calls(model_calls: int) -> None:
    if type(model_calls) is not int or not 8 <= model_calls <= 12:
        raise ValueError("model_calls must be an integer from 8 to 12")


def _tool_call_key(call: dict) -> str:
    """A tool call's ID, or its name and arguments when the provider gave none."""
    if call.get("id"):
        return str(call["id"])
    function = call.get("function") or {}
    return f"{function.get('name')}|{function.get('arguments')}"


def _thinking_field(client) -> tuple[str, object] | None:
    """The field that turns the model's reasoning off: the API preset's own (Gemini's lowest
    is reasoning_effort "minimal", OpenRouter's is reasoning.enabled false) or the pinned
    Ollama's reasoning_effort "none". Only "none" was sent and accepted before, so deep mode
    failed every question on the Gemini and OpenRouter presets (found 2026-11-15)."""
    if hasattr(client, "thinking_field"):
        return client.thinking_field
    if getattr(client, "thinking_control", "reasoning_effort_none") == "reasoning_effort_none":
        return ("reasoning_effort", "none")
    return None


class GuardedTransport(httpx.BaseTransport):
    """Inspect raw responses before the model adapter can discard reasoning fields."""

    def __init__(
        self,
        client: OllamaGgufClient | ApiChatClient,
        *,
        inner: httpx.BaseTransport | None = None,
        model_calls: int = LIMITS.model_calls,
        before_model_call: Callable[[], None] | None = None,
        after_model_call: Callable[[dict], None] | None = None,
    ) -> None:
        _validate_model_calls(model_calls)
        self.model_call_limit = model_calls
        self.client = client
        self.inner = inner if inner is not None else httpx.HTTPTransport(retries=0)
        self.before_model_call = before_model_call
        self.after_model_call = after_model_call
        self.records: list[dict] = []
        self.repairs = 0
        self.context_full = False
        # Admission before dispatch needs the served model's tokenizer and window; API
        # providers enforce their own (larger) limits and answer with their own errors.
        config = getattr(client, "config", None)
        self.context_window = (
            config.num_ctx
            if hasattr(client, "count_text_tokens") and hasattr(config, "num_ctx")
            else None
        )
        self.context_check: dict | None = None
        self._last_prompt: tuple[int, int, int] | None = None
        self.dispatch_error: BaseException | None = None
        # Provider data attached to a tool call that must come back unchanged with it on the
        # next turn: Gemini 3 answers HTTP 400 when a call's thought_signature is missing, and
        # the LangChain adapter drops that field (found 2026-11-15).
        self.tool_call_extras: dict[str, dict] = {}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        thinking_control = getattr(self.client, "thinking_control", "reasoning_effort_none")
        field = _thinking_field(self.client)
        if (
            (field is not None and body.get(field[0]) != field[1])
            or (field is None and thinking_control != "provider_default")
            or (thinking_control == "provider_default" and "reasoning_effort" in body)
            or body.get("stream", False)
        ):
            raise ModelError("Agent request must disable reasoning and streaming")
        # langchain-openai sends max_completion_tokens; the pinned Ollama honours only
        # max_tokens and silently ignores the newer field, which left calls uncapped.
        if "max_completion_tokens" in body:
            body["max_tokens"] = body.pop("max_completion_tokens")
        if body.get("max_tokens") != LIMITS.max_tokens:
            raise ModelError("Agent request must carry its output token cap")
        for message in body.get("messages") or []:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    extra = self.tool_call_extras.get(_tool_call_key(call))
                    if extra is not None and "extra_content" not in call:
                        call["extra_content"] = extra
        if len(self.records) - self.repairs >= self.model_call_limit:
            raise ModelError("Agent model-call ceiling reached")
        messages = body.get("messages") or []
        estimate = self._next_prompt_tokens(messages)
        if estimate is not None and estimate + LIMITS.max_tokens > self.context_window:
            self.context_full = True
            self.context_check = {
                "estimated_prompt_tokens": estimate,
                "max_tokens": LIMITS.max_tokens,
                "context_window": self.context_window,
            }
            raise AgentContextFull("the next agent request would not fit the context window")
        response, problem = self._dispatch(request, body, estimate=estimate)
        while problem is not None and self.repairs < LIMITS.tool_call_repairs:
            # The cut reply is not resent; say what failed and ask for the tool call alone.
            self.repairs += 1
            correction = {"role": "user", "content": _correction(problem)}
            body = {**body, "messages": [*body["messages"], correction]}
            response, problem = self._dispatch(request, body, repair=True)
        if problem is not None:
            error = AgentToolCallError(
                f"server rejected the {problem} tool-call arguments and no correction is left"
            )
            self.dispatch_error = error
            raise error
        record = self.records[-1]
        if type(record["prompt_eval_count"]) is int and type(record["eval_count"]) is int:
            # A repaired prompt also counts its correction, so the next estimate stays high.
            self._last_prompt = (record["prompt_eval_count"], record["eval_count"], len(messages))
        return response

    def _next_prompt_tokens(self, messages: list[dict]) -> int | None:
        """Exact tokens of the last prompt and reply, plus the new tool results counted with the
        served model's tokenizer. None before the first reply or without a local tokenizer."""
        if self.context_window is None or self._last_prompt is None:
            return None
        prompt, reply, seen = self._last_prompt
        total = prompt + reply
        for message in messages[seen:]:
            total += MESSAGE_TEMPLATE_TOKENS
            if message.get("role") == "assistant":
                continue  # its tool call is the reply already counted
            content = message.get("content")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            total += self.client.count_text_tokens(text)
        return total

    def _dispatch(
        self,
        request: httpx.Request,
        body: dict,
        *,
        repair: bool = False,
        estimate: int | None = None,
    ) -> tuple[httpx.Response, str | None]:
        """Send one model call; return the rejected tool name when the server refused its args."""
        request = httpx.Request(
            request.method,
            request.url,
            headers={k: v for k, v in request.headers.items() if k.lower() != "content-length"},
            content=json.dumps(body).encode("utf-8"),
            **(
                {"extensions": request.extensions} if isinstance(self.client, ApiChatClient) else {}
            ),
        )
        record = {
            "prompt_eval_count": None,
            "eval_count": None,
            "status": "dispatched",
            "error_type": None,
            "http_status": None,
            "error_message": None,
            "finish_reason": None,
            "repair": repair,
            # Admission estimate, kept so it can be checked against prompt_eval_count.
            "estimated_prompt_tokens": estimate,
            "requested_tool_calls": [],
            "thinking_detected": False,
        }
        self.records.append(record)
        started = time.perf_counter()
        gate_started = time.perf_counter()
        try:
            self.client.before_agent_dispatch()
            record["gate_ms"] = (time.perf_counter() - gate_started) * 1000
            if self.before_model_call is not None:
                hook_started = time.perf_counter()
                try:
                    self.before_model_call()
                finally:
                    record["hook_ms"] = (time.perf_counter() - hook_started) * 1000
            else:
                record["hook_ms"] = 0.0
            http_started = time.perf_counter()
            try:
                response = (
                    self.client.dispatch_agent_request(self.inner, request)
                    if isinstance(self.client, ApiChatClient)
                    else self.inner.handle_request(request)
                )
                response.read()
            finally:
                record["http_ms"] = (time.perf_counter() - http_started) * 1000
            response.request = request
            record["http_status"] = response.status_code
            if response.is_error:
                record["error_message"] = _error_message(response)
                rejected = TOOL_CALL_REJECTED.search(record["error_message"])
                if response.status_code == 500 and rejected:
                    record["status"] = "tool_call_rejected"
                    return response, rejected.group(1)
                if CONTEXT_FULL.search(record["error_message"]):
                    record["status"] = "context_full"
                    self.context_full = True
                    raise AgentContextFull("agent context window is full; the search ends here")
                if not isinstance(self.client, ApiChatClient):
                    # Keep the local server's reason in the error; it was lost before.
                    raise ModelError(
                        f"agent model server returned HTTP {response.status_code}: "
                        f"{record['error_message']}"
                    )
                response.raise_for_status()
            payload = response.json()
            usage = payload.get("usage") or {}
            record.update(
                prompt_eval_count=usage.get("prompt_tokens"),
                eval_count=usage.get("completion_tokens"),
            )
            if type(record["eval_count"]) is not int or record["eval_count"] > LIMITS.max_tokens:
                raise ModelError("Agent output exceeded its token cap or omitted usage")
            choice = payload["choices"][0]
            message = choice["message"]
            record["finish_reason"] = choice.get("finish_reason")
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = function.get("arguments", "")
                try:
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                except ValueError:
                    pass
                record["requested_tool_calls"].append(
                    {"id": call.get("id"), "name": function.get("name"), "arguments": arguments}
                )
            for call in message.get("tool_calls") or []:
                if isinstance(call.get("extra_content"), dict):
                    self.tool_call_extras[_tool_call_key(call)] = call["extra_content"]
            record["thinking_detected"] = bool(
                message.get("reasoning") or message.get("reasoning_content")
            )
            if record["thinking_detected"]:
                raise ModelError("thinking observed in raw Agent response; stop entire question")
            if len(record["requested_tool_calls"]) > 1:
                # Requests disable parallel tool calls, yet the model occasionally returns
                # several. One search per model call is the budget rule, so only the first
                # runs; the model can ask for the others on later steps.
                record["parallel_tools_dropped"] = len(record["requested_tool_calls"]) - 1
                message["tool_calls"] = message["tool_calls"][:1]
                response = httpx.Response(
                    response.status_code,
                    headers={
                        k: v
                        for k, v in response.headers.items()
                        if k.lower() not in ("content-length", "content-encoding")
                    },
                    content=json.dumps(payload).encode("utf-8"),
                    request=request,
                )
            # Recorded only: LangChain ends the search on a reply without a tool call.
            cut = record["finish_reason"] == "length" and not record["requested_tool_calls"]
            record["status"] = "cut_without_tool_call" if cut else "completed"
            return response, None
        except AgentContextFull:
            raise
        except BaseException as exc:
            record.update(status="failed", error_type=type(exc).__name__)
            self.dispatch_error = exc
            raise
        finally:
            record.setdefault("gate_ms", (time.perf_counter() - gate_started) * 1000)
            record.setdefault("hook_ms", 0.0)
            record.setdefault("http_ms", 0.0)
            record["elapsed_ms"] = (time.perf_counter() - started) * 1000
            if self.after_model_call is not None:
                try:
                    self.after_model_call(record)
                except BaseException as exc:
                    self.dispatch_error = exc
                    raise

    def close(self) -> None:
        self.inner.close()


def _correction(tool: str) -> str:
    return (
        f"Your previous reply was rejected: the arguments of your {tool} call were cut off or "
        f"not valid JSON (each reply is limited to {LIMITS.max_tokens} tokens). Reply again with "
        "only one tool call and no other text."
    )


def agent_token_usage(records: Sequence[dict]) -> dict:
    """Known agent tokens, plus upper bounds that also cover calls without reported usage.

    A rejected call reports no usage. The next dispatch resends its whole prompt plus at most a
    correction, so that prompt bounds its input; its output is bounded by max_tokens.
    """
    known_in = known_out = bound_in = bound_out = 0
    exact = True
    for index, record in enumerate(records):
        prompt, output = record.get("prompt_eval_count"), record.get("eval_count")
        if type(prompt) is int and type(output) is int:
            known_in, known_out = known_in + prompt, known_out + output
            if bound_in is not None:
                bound_in += prompt
            bound_out += output
            continue
        exact = False
        later = next(
            (
                r["prompt_eval_count"]
                for r in records[index + 1 :]
                if type(r.get("prompt_eval_count")) is int
            ),
            None,
        )
        bound_in = None if later is None or bound_in is None else bound_in + later
        bound_out += LIMITS.max_tokens
    return {
        "exact": exact,
        "input_tokens": known_in,
        "output_tokens": known_out,
        "input_tokens_upper_bound": bound_in,
        "output_tokens_upper_bound": bound_out,
    }


def _error_message(response: httpx.Response) -> str:
    """Keep the server's own error text (bounded) so a stopped question can be explained."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:1000]
    error = payload.get("error") if isinstance(payload, dict) else None
    message = error.get("message") if isinstance(error, dict) else error
    return str(payload if message is None else message)[:1000]


class EvidenceTools:
    """One document's passage namespace and explicit evidence-selection trace."""

    def __init__(
        self,
        blocks: Sequence[Block],
        initial: SearchResult,
        search: Callable[[str], SearchResult],
        *,
        handoff_findings: bool = False,
        model_calls: int = LIMITS.model_calls,
    ) -> None:
        _validate_model_calls(model_calls)
        self.model_call_limit = model_calls
        self.handoff_findings = handoff_findings
        self.findings: list[dict] = []
        self.findings_rejected: list[dict] = []
        self.blocks = {f"P{b.ordinal}": b for b in blocks}
        if len(self.blocks) != len(blocks):
            raise ValueError("duplicate ordinals in document")
        documents = {b.document_id for b in blocks}
        if len(documents) != 1:
            raise ValueError("passage namespace must contain exactly one document")
        for hit in initial.hits:
            if self.blocks.get(f"P{hit.block.ordinal}") != hit.block:
                raise ValueError("initial search contains a passage outside this document")
        self.search = search
        self.seen: set[str] = set()
        self.seen_order: list[str] = []
        self.finished: list[str] | None = None
        self.calls: list[dict] = []
        self.initial_ids = [
            f"P{h.block.ordinal}" for h in initial.hits[: LIMITS.passages_per_search]
        ]
        self.initial_passages = self.passages(self.initial_ids)

    def validate(self, passage_id: str) -> Block:
        if not re.fullmatch(r"P(?:0|[1-9]\d*)", passage_id) or passage_id not in self.blocks:
            raise ValueError("passage ID does not exist in this document")
        return self.blocks[passage_id]

    def passages(self, identifiers: Sequence[str]) -> str:
        lines = []
        for identifier in identifiers:
            block = self.validate(identifier)
            preview = block.text.replace("\n", " ").replace("\r", " ")
            marker = " (seen)" if identifier in self.seen else ""
            lines.append(f"{identifier}{marker}: {preview}")
            if identifier not in self.seen:
                self.seen.add(identifier)
                self.seen_order.append(identifier)
        return "\n".join(lines)

    def invoke(self, name: str, arguments: dict) -> str:
        started = time.perf_counter()
        row = {
            "name": name,
            "arguments": arguments,
            "returned_passage_ids": [],
            "search_ms": 0.0,
        }
        self.calls.append(row)
        try:
            if self.finished is not None or len(self.calls) > LIMITS.tool_calls:
                raise ValueError("tools ended or tool-call limit reached")
            if name == "search_passages":
                query = arguments["query"]
                if not query.strip() or len(query) > LIMITS.query_chars:
                    raise ValueError("query must be 1–300 characters")
                search_started = time.perf_counter()
                try:
                    result = self.search(query)
                finally:
                    row["search_ms"] = (time.perf_counter() - search_started) * 1000
                ids = [f"P{h.block.ordinal}" for h in result.hits[: LIMITS.passages_per_search]]
                for hit in result.hits[: LIMITS.passages_per_search]:
                    if self.blocks.get(f"P{hit.block.ordinal}") != hit.block:
                        raise ValueError("search contains a passage outside this document")
                output = self.passages(ids)
                used = len(self.calls)
                hint = f"Steps used: {used} of {self.model_call_limit}."
                if self.model_call_limit - used == 1:
                    hint += " Your next step is the last one: call finish now."
                output += f"\n{hint}"
            elif name == "finish":
                ids = arguments["passage_ids"]
                if len(ids) > LIMITS.finish_passages:
                    raise ValueError("finish accepts at most 12 passage IDs")
                for identifier in ids:
                    self.validate(identifier)
                    if identifier not in self.seen:
                        raise ValueError("finish requires a previously searched or read passage")
                ids = list(dict.fromkeys(ids))
                if self.handoff_findings:
                    self._accept_findings(arguments.get("findings", []))
                self.finished = ids
                output = "Evidence selection complete."
            else:
                raise ValueError("unknown tool")
            row["returned_passage_ids"] = ids
            row["status"] = "ok"
            return output
        except ValueError as exc:
            row.update(status="rejected", error=str(exc))
            return f"Rejected: {exc}"
        except BaseException as exc:
            row.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            row["elapsed_ms"] = (time.perf_counter() - started) * 1000

    def _accept_findings(self, findings) -> None:
        if not isinstance(findings, list):
            self.findings_rejected.append({"finding": findings, "reason": "expected a list"})
            return
        for index, item in enumerate(findings):
            reason = None
            if index >= 8:
                reason = "at most 8 findings"
            elif not isinstance(item, dict):
                reason = "expected a fact/passage object"
            elif (
                not isinstance(item.get("fact"), str)
                or not item["fact"].strip()
                or len(item["fact"]) > 200
            ):
                reason = "fact must be 1–200 nonblank characters"
            elif not isinstance(item.get("passage_id"), str) or item["passage_id"] not in self.seen:
                reason = "passage_id must have been seen"
            if reason is not None:
                self.findings_rejected.append({"finding": item, "reason": reason})
            else:
                self.findings.append({"fact": item["fact"], "passage_id": item["passage_id"]})

    def handoff(self) -> tuple[list[str], list[str], str]:
        selected = self.finished or []
        identifiers = list(dict.fromkeys([*selected, *self.seen_order]))
        reason = (
            "finish_called"
            if self.finished
            else "finish_empty"
            if self.finished == []
            else "finish_not_called"
        )
        return selected, identifiers, reason

    def tools(self):
        from langchain_core.tools import tool

        @tool
        def search_passages(
            query: Annotated[
                str,
                "Short query naming one specific entity and the relation needed, "
                "at most 300 characters",
            ],
        ) -> str:
            """Search the collection and return up to five full passages; seen ones are marked."""
            return self.invoke("search_passages", {"query": query})

        @tool(return_direct=True)
        def finish(
            passage_ids: Annotated[
                list[str], "Up to 12 previously seen passage IDs, most important first"
            ],
        ) -> str:
            """Finish evidence gathering and hand the selected passages to the writer."""
            return self.invoke("finish", {"passage_ids": passage_ids})

        if self.handoff_findings:
            # A JSON schema advertises the contract without rejecting the whole finish call:
            # invoke validates each finding and retains all valid siblings.
            schema = finish.args_schema.model_json_schema()
            schema["properties"]["findings"] = {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "fact": {"type": "string", "minLength": 1, "maxLength": 200},
                        "passage_id": {"type": "string"},
                    },
                    "required": ["fact", "passage_id"],
                },
            }

            def finish_with_findings(passage_ids, findings=()):
                """Finish evidence gathering and hand the selected passages to the writer."""
                arguments = {"passage_ids": passage_ids}
                if findings != ():
                    arguments["findings"] = findings
                return self.invoke("finish", arguments)

            finish = tool("finish", args_schema=schema, return_direct=True)(finish_with_findings)
        return [search_passages, finish]


def evidence_agent(llm, state):
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware

    system_prompt = SYSTEM_PROMPT.replace(
        "at most 8 steps", f"at most {state.model_call_limit} steps"
    )
    if state.handoff_findings:
        system_prompt += (
            " When calling finish, list in findings each fact found and its passage ID in "
            'reasoning order, for example "Zentro phone is made by Kalvex Corp" P12.'
        )
    return create_agent(
        llm,
        state.tools(),
        system_prompt=system_prompt,
        middleware=[
            ModelCallLimitMiddleware(run_limit=state.model_call_limit, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=LIMITS.tool_calls, exit_behavior="end"),
        ],
    )


def run_evidence_agent(
    workflow: FullAnswerWorkflow,
    client: OllamaGgufClient | ApiChatClient,
    question: str,
    *,
    language: Language,
    initial_search: SearchResult | None = None,
    handoff_findings: bool = False,
    model_calls: int = LIMITS.model_calls,
    before_model_call: Callable[[], None] | None = None,
    after_model_call: Callable[[dict], None] | None = None,
) -> EvidenceAgentResult:
    """Gather passages, then hand every seen original to the quote-first workflow."""
    _validate_model_calls(model_calls)
    total_started = time.perf_counter()
    if not isinstance(workflow, FullAnswerWorkflow) or workflow.writer != "quote_first":
        raise ValueError("evidence agent requires a quote-first FullAnswerWorkflow")
    from langchain_openai import ChatOpenAI

    workflow._validate_request(question, language)
    if initial_search is None:
        initial_search = workflow.search_sources(question)[0]
    if initial_search.document_id != workflow.document_id:
        raise ValueError("initial search contains a passage outside this document")
    state = EvidenceTools(
        workflow._store.get_blocks(workflow.document_id),
        initial_search,
        lambda query: workflow.search_sources(query)[0],
        handoff_findings=handoff_findings,
        model_calls=model_calls,
    )
    # The agent follows the owned client's sampling profile so writer and agent never diverge.
    sampling = client.chat_sampling()
    transport = GuardedTransport(
        client,
        before_model_call=before_model_call,
        after_model_call=after_model_call,
        model_calls=model_calls,
    )
    thinking = _thinking_field(client)
    with httpx.Client(transport=transport, trust_env=False) as http:
        llm = ChatOpenAI(
            base_url=client.chat_completions_base_url(),
            api_key="local",
            model=client.chat_model_name,
            **sampling,
            max_tokens=LIMITS.max_tokens,
            timeout=LIMITS.timeout_seconds,
            max_retries=0,
            **(
                {"reasoning_effort": thinking[1]}
                if thinking is not None and thinking[0] == "reasoning_effort"
                else {}
            ),
            http_client=http,
            model_kwargs={
                "parallel_tool_calls": False,
                **(
                    {thinking[0]: thinking[1]}
                    if thinking is not None and thinking[0] != "reasoning_effort"
                    else {}
                ),
            },
        )
        try:
            agent_started = time.perf_counter()
            try:
                evidence_agent(llm, state).invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    f"Question: {question}\n\nInitial search results for "
                                    "the question:\n"
                                    f"{state.initial_passages}\n\nSteps used: 0 of {model_calls}."
                                ),
                            }
                        ]
                    },
                    # Graph-step safety net only; call-limit middleware enforces the real limits.
                    config={"max_concurrency": 1, "recursion_limit": 100},
                )
            finally:
                agent_loop_ms = (time.perf_counter() - agent_started) * 1000
        except BaseException as exc:
            # The OpenAI adapter wraps transport errors; preserve the original guard/hook failure.
            if transport.dispatch_error is not None:
                raise transport.dispatch_error from None
            # A full context ends the search like the call limit: the writer gets what was seen.
            if not (transport.context_full and isinstance(exc, Exception)):
                raise

    selected, handoff, reason = state.handoff()
    if transport.context_full:
        reason = "context_full"
    steps = [
        row["arguments"]["query"]
        for row in state.calls
        if row["name"] == "search_passages" and row["status"] == "ok"
    ]
    pending = list(state.calls)
    calls = []
    for record in transport.records:
        for requested in record["requested_tool_calls"]:
            match = next(
                (
                    i
                    for i, actual in enumerate(pending)
                    if actual["name"] == requested["name"]
                    and actual["arguments"] == requested["arguments"]
                ),
                None,
            )
            actual = (
                pending.pop(match)
                if match is not None
                else {"status": "not_executed_or_schema_rejected", "returned_passage_ids": []}
            )
            calls.append({**requested, **actual})
    writer_started = time.perf_counter()
    answer = workflow.answer_from_context(
        question,
        [state.blocks[identifier] for identifier in handoff],
        language=language,
        search_steps=steps,
        **(
            {"search_findings": [(f["fact"], f["passage_id"]) for f in state.findings]}
            if handoff_findings
            else {}
        ),
    )
    writer_ms = (time.perf_counter() - writer_started) * 1000
    model_gate_ms = sum(record["gate_ms"] for record in transport.records)
    model_http_ms = sum(record["http_ms"] for record in transport.records)
    model_elapsed_ms = sum(record["elapsed_ms"] for record in transport.records)
    tool_ms = sum(row["elapsed_ms"] for row in state.calls)
    return EvidenceAgentResult(
        answer,
        {
            "version": EVIDENCE_AGENT_VERSION,
            "handoff_findings": handoff_findings,
            "findings": state.findings,
            "findings_rejected": state.findings_rejected,
            "model_call_limit": model_calls,
            "sampling": sampling,
            "model_calls": transport.records,
            "tool_call_repairs": transport.repairs,
            "token_usage": agent_token_usage(transport.records),
            "tool_calls": calls + pending,
            "selected_passage_ids": selected,
            "handoff_passage_ids": handoff,
            "handoff_reason": reason,
            "context_check": transport.context_check,
            "search_steps": steps,
            "timing": {
                "total_ms": (time.perf_counter() - total_started) * 1000,
                "agent_loop_ms": agent_loop_ms,
                "writer_ms": writer_ms,
                "model_gate_ms": model_gate_ms,
                "model_http_ms": model_http_ms,
                "tool_ms": tool_ms,
                "unattributed_ms": agent_loop_ms - model_elapsed_ms - tool_ms,
            },
        },
    )
