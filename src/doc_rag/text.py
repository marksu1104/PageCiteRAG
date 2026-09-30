"""Loss-aware paragraph boundaries over extracted source text."""

from __future__ import annotations

from hashlib import sha256
from typing import Iterator

from doc_rag.models import Block, DocumentMetadata, ParsedUnit


def iter_blocks(
    document: DocumentMetadata, unit: ParsedUnit, first_ordinal: int
) -> Iterator[Block]:
    """Split on blank lines while retaining each nonblank source slice and offsets."""
    if unit.status != "ok":
        return

    ordinal = first_ordinal
    offset = 0
    start_offset: int | None = None
    end_offset = 0
    start_line = 0
    end_line = 0

    def make_block() -> Block:
        block_text = unit.text[start_offset:end_offset]
        block_id = sha256(f"{document.document_id}\0{ordinal}".encode("utf-8")).hexdigest()
        return Block(
            document_id=document.document_id,
            block_id=block_id,
            ordinal=ordinal,
            unit_index=unit.unit_index,
            page_number=unit.page_number,
            line_start=start_line,
            line_end=end_line,
            char_start=start_offset,
            char_end=end_offset,
            text=block_text,
        )

    for line_number, line in enumerate(unit.text.splitlines(keepends=True), start=1):
        if line.strip():
            if start_offset is None:
                start_offset = offset
                start_line = line_number
            end_offset = offset + len(line)
            end_line = line_number
        elif start_offset is not None:
            yield make_block()
            ordinal += 1
            start_offset = None
        offset += len(line)

    if start_offset is not None:
        yield make_block()
