"""Exact, source-scoped citation validation; not a semantic entailment judge.

Model output supplies only a block ID and an exact quote. Source metadata and
quote positions are reconstructed from the authoritative stored unit. Passing
these checks proves provenance, not that the cited words support the claim.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from typing import Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from page_cite_rag.models import Block, ParsedUnit


class CitationError(RuntimeError):
    """Safe, fail-closed error without source text or raw storage exceptions."""


class CitationDraft(BaseModel):
    """The only source attributes a model is allowed to propose."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    block_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    quote: str = Field(min_length=1, max_length=4000)

    @field_validator("quote")
    @classmethod
    def nonblank_quote(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("quote must contain non-whitespace text")
        return value


class ClaimDraft(BaseModel):
    """One bounded claim with evidence, rather than uncited free-form prose."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    text: str = Field(min_length=1, max_length=4000)
    citations: tuple[CitationDraft, ...] = Field(min_length=1, max_length=8)

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claim must contain non-whitespace text")
        return value


class AnswerDraft(BaseModel):
    """Answer text must be derived from claims, never a separate unchecked field."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["answered", "insufficient_evidence"]
    claims: tuple[ClaimDraft, ...] = Field(default=(), max_length=12)
    limitations: tuple[str, ...] = Field(default=(), max_length=8)

    @field_validator("limitations")
    @classmethod
    def bounded_limitations(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() or len(value) > 1000 for value in values):
            raise ValueError("limitations must be nonblank bounded strings")
        return values

    @model_validator(mode="after")
    def validate_status(self) -> AnswerDraft:
        if self.status == "answered" and not self.claims:
            raise ValueError("answered output requires at least one cited claim")
        if self.status == "insufficient_evidence" and self.claims:
            raise ValueError("insufficient-evidence output must not contain answer claims")
        return self


class VerifiedCitation(BaseModel):
    """Verified quote locations, using zero-based half-open unit character offsets."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    block_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    quote: str = Field(min_length=1, max_length=4000)
    unit_index: int = Field(ge=1)
    page_number: int | None = Field(default=None, ge=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> VerifiedCitation:
        if (
            not self.quote.strip()
            or self.line_end < self.line_start
            or self.char_end - self.char_start != len(self.quote)
        ):
            raise ValueError("verified quote metadata is inconsistent")
        return self

    source_pages: tuple[int, ...] = Field(default=())
    native_refs: tuple[str, ...] = Field(default=())

    @model_validator(mode="after")
    def validate_source_locations(self) -> VerifiedCitation:
        if (
            any(type(page) is not int or page < 1 for page in self.source_pages)
            or len(set(self.source_pages)) != len(self.source_pages)
            or any(not ref for ref in self.native_refs)
            or len(set(self.native_refs)) != len(self.native_refs)
            or (len(self.source_pages) == 1 and self.page_number != self.source_pages[0])
        ):
            raise ValueError("verified source location metadata is inconsistent")
        return self


class VerifiedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    text: str = Field(min_length=1, max_length=4000)
    citations: tuple[VerifiedCitation, ...] = Field(min_length=1, max_length=8)


class CitationSource(Protocol):
    """Read-only source access; usage-scope labels are not access-control rules."""

    def get_block(self, document_id: str, block_id: str) -> Block | None: ...

    def get_unit(self, document_id: str, unit_index: int) -> ParsedUnit | None: ...


def verified_location_fields(source: CitationSource, document_id: str, block: Block) -> dict:
    """Read optional parser locations from the saved source map, never from model output."""
    location_reader = getattr(source, "get_source_location", None)
    location = location_reader(document_id, block.block_id) if callable(location_reader) else None
    pages = location.get("pages", []) if location is not None else []
    native_refs = location.get("native_refs", []) if location is not None else []
    if (
        not isinstance(pages, list)
        or any(type(page) is not int or page < 1 for page in pages)
        or len(set(pages)) != len(pages)
        or not isinstance(native_refs, list)
        or any(not isinstance(ref, str) or not ref for ref in native_refs)
        or len(set(native_refs)) != len(native_refs)
    ):
        raise CitationError("saved source location metadata is invalid")
    source_pages = tuple(pages) or ((block.page_number,) if block.page_number is not None else ())
    page_number = source_pages[0] if len(source_pages) == 1 else None
    return {
        "page_number": page_number,
        "source_pages": source_pages,
        "native_refs": tuple(native_refs),
    }


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for line in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def validate_claims(
    store: CitationSource,
    document_id: str,
    draft: AnswerDraft,
    allowed_blocks: Sequence[Block],
) -> tuple[VerifiedClaim, ...]:
    """Re-read source and require unique exact quotes within the supplied context.

    No fuzzy matching, trimming, inferred page numbers, or model-proposed offsets
    are accepted. All context blocks are checked, including those not cited. This
    is provenance validation only: a contradictory claim can still quote genuine
    source words and requires separate semantic/human evaluation.
    """
    if not isinstance(document_id, str) or re.fullmatch(r"[0-9a-f]{64}", document_id) is None:
        raise CitationError("invalid citation document scope")
    if not isinstance(draft, AnswerDraft):
        raise CitationError("validated answer draft required")
    # Revalidation also rejects objects built with Pydantic's bypass constructor.
    try:
        draft = AnswerDraft.model_validate(draft.model_dump(warnings=False))
    except Exception:
        raise CitationError("invalid answer draft") from None

    source_blocks: dict[str, tuple[Block, ParsedUnit, list[int]]] = {}
    units: dict[int, ParsedUnit] = {}
    for context_block in allowed_blocks:
        if not isinstance(context_block, Block) or context_block.document_id != document_id:
            raise CitationError("context block is outside the document scope")
        if context_block.block_id in source_blocks:
            raise CitationError("context contains duplicate block identities")
        try:
            stored = store.get_block(document_id, context_block.block_id)
            if stored is None or stored != context_block:
                raise CitationError("context block differs from authoritative source")
            unit = units.get(stored.unit_index)
            if unit is None:
                unit = store.get_unit(document_id, stored.unit_index)
                if unit is not None:
                    units[stored.unit_index] = unit
        except CitationError:
            raise
        except Exception:
            raise CitationError("citation source cannot be read") from None
        if not isinstance(unit, ParsedUnit):
            raise CitationError("citation source unit is missing")
        lines = _line_starts(unit.text)
        if (
            unit.status != "ok"
            or unit.unit_index != stored.unit_index
            or unit.page_number != stored.page_number
            or stored.char_start < 0
            or stored.char_end > len(unit.text)
            or unit.text[stored.char_start : stored.char_end] != stored.text
            or stored.line_start != bisect_right(lines, stored.char_start)
            or stored.line_end != bisect_right(lines, stored.char_end - 1)
        ):
            raise CitationError("citation block is inconsistent with its source unit")
        source_blocks[stored.block_id] = (stored, unit, lines)

    verified: list[VerifiedClaim] = []
    for claim in draft.claims:
        citations: list[VerifiedCitation] = []
        for proposed in claim.citations:
            source = source_blocks.get(proposed.block_id)
            if source is None:
                raise CitationError("citation block was not provided in the context")
            block, _unit, lines = source
            relative_start = block.text.find(proposed.quote)
            if relative_start < 0:
                raise CitationError("citation quote is not an exact source substring")
            # Starting one character later catches overlapping matches as well.
            if block.text.find(proposed.quote, relative_start + 1) >= 0:
                raise CitationError("citation quote has ambiguous source positions")
            start = block.char_start + relative_start
            end = start + len(proposed.quote)
            location_fields = verified_location_fields(store, document_id, block)
            citations.append(
                VerifiedCitation(
                    block_id=block.block_id,
                    document_id=document_id,
                    quote=proposed.quote,
                    unit_index=block.unit_index,
                    **location_fields,
                    line_start=bisect_right(lines, start),
                    line_end=bisect_right(lines, end - 1),
                    char_start=start,
                    char_end=end,
                )
            )
        verified.append(VerifiedClaim(text=claim.text, citations=tuple(citations)))
    return tuple(verified)
