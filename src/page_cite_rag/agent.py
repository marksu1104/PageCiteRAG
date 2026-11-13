"""Shared reading-agent primitives: budgets, findings, admission and citation handles.

The evaluation runner and retained evidence workflows use these shared primitives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Sequence

from pydantic import BaseModel, Field, field_validator

from page_cite_rag.full_answer import FullAnswerWorkflow, _spans
from page_cite_rag.generation import ModelError
from page_cite_rag.models import Block
from page_cite_rag.retrieval import SearchContextGroup
from page_cite_rag.tokenize import LexicalTokenizer
from page_cite_rag.workflows import Language

MAX_AGENT_MODEL_CALLS = 7
MAX_AGENT_TOOL_CALLS = 12
MAX_AGENT_CALL_INPUT_TOKENS = 10_000
MAX_AGENT_TOTAL_INPUT_TOKENS = 20_000
MAX_INTEGRATED_AGENT_CALL_INPUT_TOKENS = 12_000
MAX_AGENT_CALL_OUTPUT_TOKENS = 512
MAX_AGENT_TOTAL_OUTPUT_TOKENS = 2_560
MAX_INTEGRATED_AGENT_CALL_OUTPUT_TOKENS = 3_072
AGENT_CONTEXT_RESERVE_TOKENS = 1_024
AGENT_WRITER_RESERVE_SECONDS = 60
InformationNeed = Annotated[str, Field(max_length=240)]


def _search_preview(text: str, query: str, tokenizer: LexicalTokenizer) -> tuple[str, str]:
    """Return a contiguous original sentence, with an explicit lexical fallback."""
    query_tokens = set(tokenizer.tokenize(query))
    sentences = [
        match.group().strip()
        for match in re.finditer(r"[^\n]+?(?:[。！？]+|[.!?]+(?=\s|$)|\n|$)", text)
    ]
    best = ""
    best_score = 0
    for sentence in sentences:
        score = len(query_tokens.intersection(tokenizer.tokenize(sentence)))
        if score > best_score:
            best, best_score = sentence, score
    if best_score:
        # A preview is a navigation hint, never a substitute for the complete source.
        if len(best) <= 480:
            return best, "lexical_match"
        positions = [best.casefold().find(token) for token in query_tokens if len(token) > 1]
        positions = [position for position in positions if position >= 0]
        start = max(0, min(positions, default=0) - 100)
        return best[start : start + 480], "lexical_match_truncated"
    return text[:240], "prefix_fallback"


class AgentWorkflowError(ModelError):
    """Fail closed when a bounded local document-navigation run is invalid."""


@dataclass(frozen=True)
class AgentContext:
    blocks: tuple[Block, ...]
    context_groups: tuple[SearchContextGroup, ...]
    exact_input_tokens: int | None
    omitted_block_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentRun:
    context: AgentContext
    trace: dict
    structured_response: object | None = None
    acquired_context: AgentContext | None = None
    navigation_note: str | None = None


class AgentRunFailure(ModelError):
    def __init__(self, error_type: str, trace: dict) -> None:
        super().__init__("bounded local agent run failed")
        self.error_type = error_type
        self.trace = trace


class ReadingFinding(BaseModel):
    """One observation tied to a question requirement and a visible original."""

    need_id: str | None = Field(
        default=None,
        max_length=16,
        description="Stable planner need ID when the initial request included a reading plan.",
    )
    need: str = Field(
        min_length=1,
        description="One neutral question part requested by the user, not an inferred answer.",
    )
    observation: str = Field(
        min_length=1,
        max_length=400,
        description="One concise fact; retain subject and conditions.",
    )
    source_ids: list[str] = Field(
        min_length=1,
        description="Visible originals supporting this finding, including all sides of a relation.",
    )
    source_span_ids: list[str] = Field(
        default_factory=list,
        description="Exact span handles read by the outline reader; empty for block readers.",
    )

    model_config = {"extra": "forbid"}

    @field_validator("need", "observation")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reading findings require nonblank text")
        return value

    @field_validator("source_ids")
    @classmethod
    def nonblank_sources(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("reading findings require nonblank source IDs")
        return list(dict.fromkeys(values))


class ReadingReport(BaseModel):
    """A source-backed reading handoff, not an independently verified answer."""

    requirements: list[ReadingFinding] = Field(
        description=(
            "One source-linked entry per requested mechanism, relation, comparison or "
            "condition. Put unsupported question parts in unresolved."
        ),
    )
    unresolved: list[str] = Field(
        description="Every requested question part still lacking original support."
    )

    model_config = {"extra": "forbid"}

    @field_validator("unresolved")
    @classmethod
    def nonblank_gaps(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("unresolved requirements must be nonblank")
        return values


class _AgentBudgetStop(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


@dataclass
class _AgentBudgetLedger:
    input_tokens: int = 0
    output_tokens: int = 0
    output_reserved_tokens: int = 0
    model_calls: int = 0
    output_usage_complete: bool = True
    model_call_limit: int | None = MAX_AGENT_MODEL_CALLS
    call_input_limit: int = MAX_AGENT_CALL_INPUT_TOKENS
    total_input_limit: int | None = MAX_AGENT_TOTAL_INPUT_TOKENS
    call_output_limit: int = MAX_AGENT_CALL_OUTPUT_TOKENS
    total_output_limit: int | None = MAX_AGENT_TOTAL_OUTPUT_TOKENS
    tool_call_limit: int | None = MAX_AGENT_TOOL_CALLS

    def admit(self, input_tokens: int, context_limit: int) -> None:
        if type(input_tokens) is not int or input_tokens < 1:
            raise AgentWorkflowError("local chat tokenizer returned invalid input-token metadata")
        if input_tokens > self.call_input_limit:
            raise _AgentBudgetStop("per_call_input_limit")
        if input_tokens + self.call_output_limit + AGENT_CONTEXT_RESERVE_TOKENS >= context_limit:
            raise _AgentBudgetStop("context_limit")
        if self.model_call_limit is not None and self.model_calls >= self.model_call_limit:
            raise _AgentBudgetStop("model_call_limit")
        if (
            self.total_input_limit is not None
            and self.input_tokens + input_tokens > self.total_input_limit
        ):
            raise _AgentBudgetStop("agent_input_budget")
        if (
            self.total_output_limit is not None
            and self.output_tokens + self.output_reserved_tokens + self.call_output_limit
            > self.total_output_limit
        ):
            raise _AgentBudgetStop("agent_output_budget")
        self.model_calls += 1
        self.input_tokens += input_tokens
        self.output_reserved_tokens += self.call_output_limit

    def complete(self, output_tokens: int | None) -> None:
        if output_tokens is None:
            self.output_usage_complete = False
            return
        if type(output_tokens) is not int or not 0 <= output_tokens <= self.call_output_limit:
            raise AgentWorkflowError("local chat model returned invalid output-token metadata")
        self.output_reserved_tokens -= self.call_output_limit
        self.output_tokens += output_tokens

    def rollback_admission(self, input_tokens: int) -> None:
        """Undo local admission when the global ledger refused pre-dispatch."""
        if (
            self.model_calls < 1
            or input_tokens < 1
            or self.input_tokens < input_tokens
            or self.output_reserved_tokens < self.call_output_limit
        ):
            raise AgentWorkflowError("cannot roll back an inconsistent model admission")
        self.model_calls -= 1
        self.input_tokens -= input_tokens
        self.output_reserved_tokens -= self.call_output_limit


def _admit(
    workflow: FullAnswerWorkflow,
    question: str,
    language: Language,
    blocks: Sequence[Block],
    context_groups: Sequence[SearchContextGroup],
) -> AgentContext:
    selected, omitted, _messages, budget = workflow._context(
        question,
        language,
        "local",
        blocks,
        None,
        context_groups=context_groups,
    )
    selected_ids = {block.block_id for block in selected}
    omitted_ids = set(omitted)
    groups = tuple(
        group for group in context_groups if not (set(group.source_block_ids) & omitted_ids)
    )
    if (
        groups
        and {block_id for group in groups for block_id in group.source_block_ids} != selected_ids
    ):
        raise AgentWorkflowError("admitted contextual chunks do not match source blocks")
    return AgentContext(tuple(selected), groups, budget.exact_input_tokens, tuple(omitted))


def _citation_handles(blocks: Sequence[Block]) -> list[dict]:
    spans = _spans(blocks)
    return [
        {
            "source_id": source_id,
            "block_id": span.block_id,
            "char_start": span.char_start,
            "char_end": span.char_end,
        }
        for source_id, span in spans.items()
    ]
