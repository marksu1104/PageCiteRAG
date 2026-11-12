"""Serial, non-thinking evidence gathering with a shared quote-first writer."""

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated

import httpx

from page_cite_rag.full_answer import FullAnswerWorkflow
from page_cite_rag.generation import ModelError
from page_cite_rag.models import Block
from page_cite_rag.ollama_backend import OllamaGgufClient
from page_cite_rag.retrieval import SearchResult
from page_cite_rag.workflows import AnswerResult, Language

EVIDENCE_AGENT_VERSION = "evidence-agent/v2"


@dataclass(frozen=True)
class EvidenceAgentLimits:
    model_calls: int = 8
    tool_calls: int = 16
    passages_per_search: int = 5
    query_chars: int = 300
    finish_passages: int = 12
    max_tokens: int = 512
    timeout_seconds: int = 600


LIMITS = EvidenceAgentLimits()


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


class GuardedTransport(httpx.BaseTransport):
    """Inspect raw responses before the model adapter can discard reasoning fields."""

    def __init__(
        self,
        client: OllamaGgufClient,
        *,
        inner: httpx.BaseTransport | None = None,
        before_model_call: Callable[[], None] | None = None,
        after_model_call: Callable[[dict], None] | None = None,
    ) -> None:
        self.client = client
        self.inner = inner if inner is not None else httpx.HTTPTransport(retries=0)
        self.before_model_call = before_model_call
        self.after_model_call = after_model_call
        self.records: list[dict] = []
        self.dispatch_error: BaseException | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        if body.get("reasoning_effort") != "none" or body.get("stream", False):
            raise ModelError("Agent request must disable reasoning and streaming")
        if len(self.records) >= LIMITS.model_calls:
            raise ModelError("Agent model-call ceiling reached")
        record = {
            "prompt_eval_count": None,
            "eval_count": None,
            "status": "dispatched",
            "error_type": None,
            "requested_tool_calls": [],
            "thinking_detected": False,
        }
        self.records.append(record)
        started = time.perf_counter()
        try:
            self.client.before_agent_dispatch()
            if self.before_model_call is not None:
                self.before_model_call()
            response = self.inner.handle_request(request)
            response.read()
            response.request = request
            payload = response.json()
            usage = payload.get("usage") or {}
            record.update(
                prompt_eval_count=usage.get("prompt_tokens"),
                eval_count=usage.get("completion_tokens"),
            )
            response.raise_for_status()
            message = payload["choices"][0]["message"]
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
            record["thinking_detected"] = bool(
                message.get("reasoning") or message.get("reasoning_content")
            )
            if record["thinking_detected"]:
                raise ModelError("thinking observed in raw Agent response; stop entire question")
            if len(record["requested_tool_calls"]) > 1:
                raise ModelError("Agent returned parallel tools despite serial-call policy")
            record["status"] = "completed"
            return response
        except BaseException as exc:
            record.update(status="failed", error_type=type(exc).__name__)
            self.dispatch_error = exc
            raise
        finally:
            record["elapsed_ms"] = (time.perf_counter() - started) * 1000
            if self.after_model_call is not None:
                try:
                    self.after_model_call(record)
                except BaseException as exc:
                    self.dispatch_error = exc
                    raise

    def close(self) -> None:
        self.inner.close()


class EvidenceTools:
    """One document's passage namespace and explicit evidence-selection trace."""

    def __init__(
        self,
        blocks: Sequence[Block],
        initial: SearchResult,
        search: Callable[[str], SearchResult],
    ) -> None:
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
        row = {"name": name, "arguments": arguments, "returned_passage_ids": []}
        self.calls.append(row)
        try:
            if self.finished is not None or len(self.calls) > LIMITS.tool_calls:
                raise ValueError("tools ended or tool-call limit reached")
            if name == "search_passages":
                query = arguments["query"]
                if not query.strip() or len(query) > LIMITS.query_chars:
                    raise ValueError("query must be 1–300 characters")
                result = self.search(query)
                ids = [f"P{h.block.ordinal}" for h in result.hits[: LIMITS.passages_per_search]]
                for hit in result.hits[: LIMITS.passages_per_search]:
                    if self.blocks.get(f"P{hit.block.ordinal}") != hit.block:
                        raise ValueError("search contains a passage outside this document")
                output = self.passages(ids)
                used = len(self.calls)
                hint = f"Steps used: {used} of {LIMITS.model_calls}."
                if LIMITS.model_calls - used == 1:
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

        return [search_passages, finish]


def evidence_agent(llm, state):
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware

    return create_agent(
        llm,
        state.tools(),
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            ModelCallLimitMiddleware(run_limit=LIMITS.model_calls, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=LIMITS.tool_calls, exit_behavior="end"),
        ],
    )


def run_evidence_agent(
    workflow: FullAnswerWorkflow,
    client: OllamaGgufClient,
    question: str,
    *,
    language: Language,
    initial_search: SearchResult | None = None,
    before_model_call: Callable[[], None] | None = None,
    after_model_call: Callable[[dict], None] | None = None,
) -> EvidenceAgentResult:
    """Gather passages, then hand every seen original to the quote-first workflow."""
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
    )
    transport = GuardedTransport(
        client, before_model_call=before_model_call, after_model_call=after_model_call
    )
    with httpx.Client(transport=transport, trust_env=False) as http:
        llm = ChatOpenAI(
            base_url=client.chat_completions_base_url(),
            api_key="local",
            model=client.ollama_model_name,
            temperature=0,
            seed=42,
            max_tokens=LIMITS.max_tokens,
            timeout=LIMITS.timeout_seconds,
            max_retries=0,
            reasoning_effort="none",
            http_client=http,
            model_kwargs={"parallel_tool_calls": False},
        )
        try:
            evidence_agent(llm, state).invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"Question: {question}\n\nInitial search results for "
                                "the question:\n"
                                f"{state.initial_passages}\n\nSteps used: 0 of 8."
                            ),
                        }
                    ]
                },
                # Graph-step safety net only; call-limit middleware enforces the real limits.
                config={"max_concurrency": 1, "recursion_limit": 100},
            )
        except BaseException:
            # The OpenAI adapter wraps transport errors; preserve the original guard/hook failure.
            if transport.dispatch_error is not None:
                raise transport.dispatch_error from None
            raise

    selected, handoff, reason = state.handoff()
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
    answer = workflow.answer_from_context(
        question,
        [state.blocks[identifier] for identifier in handoff],
        language=language,
        search_steps=steps,
    )
    return EvidenceAgentResult(
        answer,
        {
            "version": EVIDENCE_AGENT_VERSION,
            "model_calls": transport.records,
            "tool_calls": calls + pending,
            "selected_passage_ids": selected,
            "handoff_passage_ids": handoff,
            "handoff_reason": reason,
            "search_steps": steps,
        },
    )
