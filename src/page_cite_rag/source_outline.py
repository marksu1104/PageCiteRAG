"""Versioned, source-derived section boundaries; never infer chapters from PDF pages."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from page_cite_rag.models import Block, DocumentMetadata, ParsedUnit

if TYPE_CHECKING:
    from page_cite_rag.qasper import Paper

OUTLINE_KIND = "source-outline-v1"
_ATX = re.compile(r" {0,3}(#{1,6})[ \t]+(.+?)\s*$")
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)$")


class SourceSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    section_id: str = Field(pattern=r"^section:(?:[0-9]{4,}|preamble)$")
    title: str | None = Field(default=None, max_length=512)
    start_ordinal: int = Field(ge=1)
    end_ordinal: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> SourceSection:
        if self.end_ordinal < self.start_ordinal:
            raise ValueError("section end precedes start")
        return self


class SourceOutline(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1] = 1
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    origin: Literal["qasper", "markdown_atx", "docling_headings"]
    block_count: int = Field(ge=1)
    sections: tuple[SourceSection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_partition(self) -> SourceOutline:
        cursor = 1
        seen: set[str] = set()
        for section in self.sections:
            if section.section_id in seen or section.start_ordinal != cursor:
                raise ValueError("sections must uniquely partition source block order")
            seen.add(section.section_id)
            cursor = section.end_ordinal + 1
        if cursor != self.block_count + 1:
            raise ValueError("sections must cover every source block")
        return self


def outline_from_qasper(
    paper: Paper, document: DocumentMetadata, blocks: list[Block]
) -> SourceOutline | None:
    """Preserve supplied QASPER section indices, names and paragraph order."""
    if not blocks:
        return None
    if not paper.section_names:
        return None
    if len(blocks) != len(paper.passages):
        raise ValueError("QASPER sections do not match stored passages")
    groups: list[tuple[int, int, int]] = []
    for ordinal, (passage, block) in enumerate(zip(paper.passages, blocks, strict=True), 1):
        if (
            block.document_id != document.document_id
            or block.ordinal != ordinal
            or block.text != passage.text
            or block.unit_index != 1
            or not 0 <= passage.section_index < len(paper.section_names)
        ):
            raise ValueError("QASPER section mapping differs from stored original paragraphs")
        if groups and passage.section_index < groups[-1][0]:
            raise ValueError("QASPER section order is not monotonic")
        if groups and passage.section_index == groups[-1][0]:
            index, start, _ = groups[-1]
            groups[-1] = (index, start, ordinal)
        else:
            groups.append((passage.section_index, ordinal, ordinal))
    return SourceOutline(
        document_id=document.document_id,
        source_sha256=document.source_sha256,
        origin="qasper",
        block_count=len(blocks),
        sections=tuple(
            SourceSection(
                section_id=f"section:{index:04d}",
                title=paper.section_names[index],
                start_ordinal=start,
                end_ordinal=end,
            )
            for index, start, end in groups
        ),
    )


def outline_from_markdown(
    document: DocumentMetadata, unit: ParsedUnit, blocks: list[Block]
) -> SourceOutline | None:
    """Use only explicit ATX heading lines at a verified block boundary."""
    if document.media_type != "text/markdown" or unit.status != "ok" or not blocks:
        return None
    if any(
        block.document_id != document.document_id
        or block.unit_index != 1
        or block.ordinal != ordinal
        or unit.text[block.char_start : block.char_end] != block.text
        for ordinal, block in enumerate(blocks, 1)
    ):
        raise ValueError("Markdown blocks do not match original source")
    headings: list[tuple[int, str]] = []
    offset = 0
    fence_char = ""
    fence_width = 0
    starts = {block.char_start: block.ordinal for block in blocks}
    for line in unit.text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        fence = _FENCE.fullmatch(content)
        if fence_char:
            if (
                fence is not None
                and fence.group(1)[0] == fence_char
                and len(fence.group(1)) >= fence_width
                and not fence.group(2).strip()
            ):
                fence_char = ""
            offset += len(line)
            continue
        if fence is not None:
            fence_char, fence_width = fence.group(1)[0], len(fence.group(1))
            offset += len(line)
            continue
        match = _ATX.fullmatch(content)
        if match is not None:
            title = match.group(2).rstrip(" #\t")
            if title and offset in starts:
                headings.append((starts[offset], title[:512]))
            elif title:
                # A heading inside a paragraph block cannot be mapped without splitting it.
                return None
        offset += len(line)
    if not headings:
        return None
    if len({ordinal for ordinal, _ in headings}) != len(headings):
        return None
    sections: list[SourceSection] = []
    if headings[0][0] > 1:
        sections.append(
            SourceSection(
                section_id="section:preamble",
                start_ordinal=1,
                end_ordinal=headings[0][0] - 1,
            )
        )
    for index, (start, title) in enumerate(headings):
        end = headings[index + 1][0] - 1 if index + 1 < len(headings) else len(blocks)
        sections.append(
            SourceSection(
                section_id=f"section:{index:04d}",
                title=title,
                start_ordinal=start,
                end_ordinal=end,
            )
        )
    return SourceOutline(
        document_id=document.document_id,
        source_sha256=document.source_sha256,
        origin="markdown_atx",
        block_count=len(blocks),
        sections=tuple(sections),
    )
