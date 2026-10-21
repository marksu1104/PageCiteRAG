"""Select immutable source spans; never ask a model to reconstruct quotations.

This is a citation presentation strategy, not a new retrieval chunker or an
entailment judge. Short blocks stay intact. Long blocks are partitioned into
bounded character ranges, explicitly without claiming sentence boundaries.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model, model_validator

from page_cite_rag.citations import (
    AnswerDraft,
    CitationError,
    CitationSource,
    ClaimDraft,
    VerifiedCitation,
    VerifiedClaim,
    _line_starts,
    validate_claims,
    verified_location_fields,
)
from page_cite_rag.models import Block, ParsedUnit

SPAN_VERSION = "whole-block-or-contiguous-4000-char-spans/v1"
MAX_SPAN_CHARS = 4000


class SpanCitationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    span_id: str = Field(pattern=r"^[0-9a-f]{64}$")


class SpanClaimDraft(ClaimDraft):
    citations: tuple[SpanCitationDraft, ...] = Field(min_length=1, max_length=8)


class SpanAnswerDraft(AnswerDraft):
    """Same answer bounds/status, but no model-proposed quote or source metadata."""

    claims: tuple[SpanClaimDraft, ...] = Field(default=(), max_length=12)


class SingleAnswerSpanDraft(SpanAnswerDraft):
    """One complete answer claim, with citations selected from supplied source spans."""

    claims: tuple[SpanClaimDraft, ...] = Field(default=(), max_length=1)


class SplitSpanAnswerDraft(SpanAnswerDraft):
    """Direct answer claims and optional cited explanation, never an unchecked answer."""

    explanation_claims: tuple[SpanClaimDraft, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_explanation(self) -> SplitSpanAnswerDraft:
        if self.status == "insufficient_evidence" and self.explanation_claims:
            raise ValueError("insufficient-evidence output must not contain explanation claims")
        if len(self.claims) + len(self.explanation_claims) > 12:
            raise ValueError("answer and explanation together are limited to12 claims")
        return self


@dataclass(frozen=True)
class SourceSpan:
    span_id: str
    block_id: str
    char_start: int
    char_end: int
    text: str


def source_spans(blocks: Sequence[Block]) -> tuple[SourceSpan, ...]:
    """Deterministic IDs bind document, block, complete block text and exact range.

    Whole original blocks remain the retrieval/budget unit. No case, whitespace,
    math or Unicode normalization is performed. Whitespace-only ranges cannot
    support a claim and are not offered as citations.
    """
    spans = []
    seen = set()
    for proposed in blocks:
        try:
            block = Block.model_validate(proposed.model_dump(warnings=False), strict=True)
        except (AttributeError, ValidationError):
            raise CitationError("invalid source block for span selection") from None
        if block.block_id in seen:
            raise CitationError("context contains duplicate block identities")
        seen.add(block.block_id)
        try:
            text_hash = hashlib.sha256(block.text.encode("utf-8")).hexdigest()
        except UnicodeError:
            raise CitationError("source block is not valid UTF-8 text") from None
        for start in range(0, len(block.text), MAX_SPAN_CHARS):
            text = block.text[start : start + MAX_SPAN_CHARS]
            if not text.strip():
                continue
            end = start + len(text)
            identity = [
                SPAN_VERSION,
                block.document_id,
                block.block_id,
                text_hash,
                block.char_start + start,
                block.char_start + end,
            ]
            span_id = hashlib.sha256(
                json.dumps(identity, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            spans.append(
                SourceSpan(
                    span_id, block.block_id, block.char_start + start, block.char_start + end, text
                )
            )
    if len({span.span_id for span in spans}) != len(spans):
        raise CitationError("source span identities collide")
    return tuple(spans)


def provided_span_answer_model(
    blocks: Sequence[Block], answer_model: type[SpanAnswerDraft] = SpanAnswerDraft
) -> type[SpanAnswerDraft]:
    """Use Pydantic's model factory to bind citation choices to this request only.

    Literal emits JSON Schema enum (const for a singleton), used by the backend's
    existing constrained decoder. No global mutation, unbounded cache, quote repair
    or semantic guarantee. Inherited bounds/validators and provenance checks remain.
    Empty context admits no claims; workflows do not generate without evidence.
    """
    return _provided_span_model(blocks, answer_model)


def provided_single_answer_model(blocks: Sequence[Block]) -> type[SingleAnswerSpanDraft]:
    """Bind a single-answer response schema to only this request's source spans."""
    return _provided_span_model(blocks, SingleAnswerSpanDraft)


def provided_split_span_answer_model(blocks: Sequence[Block]) -> type[SplitSpanAnswerDraft]:
    """Both output partitions use exactly the same request-scoped citation choices."""
    return _provided_span_model(blocks, SplitSpanAnswerDraft)


def _provided_span_model(blocks, answer_model):
    ids = tuple(span.span_id for span in source_spans(blocks))
    fields = (
        ("claims", "explanation_claims") if answer_model is SplitSpanAnswerDraft else ("claims",)
    )
    name = f"Provided{answer_model.__name__}"
    if not ids:
        return create_model(
            name,
            __base__=answer_model,
            **{
                field: (tuple[SpanClaimDraft, ...], Field(default=(), max_length=0))
                for field in fields
            },
        )
    citation = create_model(
        "ProvidedSpanCitationDraft",
        __base__=SpanCitationDraft,
        span_id=(Literal[ids], ...),
    )
    claim = create_model(
        "ProvidedSpanClaimDraft",
        __base__=SpanClaimDraft,
        citations=(tuple[citation, ...], deepcopy(SpanClaimDraft.model_fields["citations"])),
    )
    return create_model(
        name,
        __base__=answer_model,
        **{
            field: (tuple[claim, ...], deepcopy(answer_model.model_fields[field]))
            for field in fields
        },
    )


def validate_span_claims(
    store: CitationSource,
    document_id: str,
    draft: SpanAnswerDraft,
    allowed_blocks: Sequence[Block],
) -> tuple[VerifiedClaim, ...]:
    """Re-read ALL admitted source blocks, then resolve only their bound span IDs.

    A repeated quote is unambiguous here because its exact range is selected.
    Genuine source text can still fail to support a claim: semantic checking is
    separate. No fuzzy matching, normalization or repair of model output occurs.
    """
    if not isinstance(draft, SpanAnswerDraft):
        raise CitationError("validated span answer draft required")
    try:
        draft = SpanAnswerDraft.model_validate(draft.model_dump(warnings=False), strict=True)
        if any(not claim.text.strip() for claim in draft.claims):
            raise ValueError("blank claim")
    except (ValidationError, ValueError):
        raise CitationError("invalid span answer draft") from None
    # Reuse the exact route's authoritative source checks without proposing quotes.
    validate_claims(store, document_id, AnswerDraft(status="insufficient_evidence"), allowed_blocks)
    spans = {span.span_id: span for span in source_spans(allowed_blocks)}
    blocks = {block.block_id: block for block in allowed_blocks}
    verified = []
    for claim in draft.claims:
        citations = []
        seen = set()
        for proposed in claim.citations:
            span = spans.get(proposed.span_id)
            if span is None:
                raise CitationError("citation span was not provided in the context")
            if span.span_id in seen:
                raise CitationError("claim contains duplicate citation spans")
            seen.add(span.span_id)
            block = blocks[span.block_id]
            try:
                unit = store.get_unit(document_id, block.unit_index)
            except Exception:
                raise CitationError("citation source cannot be read") from None
            if (
                not isinstance(unit, ParsedUnit)
                or unit.status != "ok"
                or unit.unit_index != block.unit_index
                or unit.page_number != block.page_number
                or unit.text[block.char_start : block.char_end] != block.text
                or unit.text[span.char_start : span.char_end] != span.text
            ):
                raise CitationError("citation span differs from authoritative source")
            lines = _line_starts(unit.text)
            citations.append(
                VerifiedCitation(
                    block_id=block.block_id,
                    document_id=document_id,
                    quote=span.text,
                    unit_index=block.unit_index,
                    **verified_location_fields(store, document_id, block),
                    line_start=bisect_right(lines, span.char_start),
                    line_end=bisect_right(lines, span.char_end - 1),
                    char_start=span.char_start,
                    char_end=span.char_end,
                )
            )
        verified.append(VerifiedClaim(text=claim.text, citations=tuple(citations)))
    return tuple(verified)
