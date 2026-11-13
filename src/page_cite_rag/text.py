"""Loss-aware paragraph boundaries over extracted source text."""

from __future__ import annotations

import json
from bisect import bisect_right
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Iterator

from page_cite_rag.models import Block, DocumentMetadata, ParsedUnit

# 依據：MuSiQue 與 QASPER 原始段落多在 2,000 字元內；超長段落會讓檢索與引用只剩整份一塊。
MAX_BLOCK_CHARS = 2000


class TokenBlockFactory:
    """Split paragraphs with the prepared tokenizer; retain exact source slices."""

    def __init__(self, tokenizer_directory: Path, *, max_tokens: int = 512, overlap: int = 64):
        from tokenizers import Tokenizer

        if not 32 <= max_tokens <= 1024 or not 0 <= overlap < max_tokens - 16:
            raise ValueError("invalid text chunk token budget")
        path = tokenizer_directory / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.identity = {
            "version": "token-offset-source-blocks/v1",
            "tokenizer_sha256": sha256(path.read_bytes()).hexdigest(),
            "max_tokens": max_tokens,
            "overlap": overlap,
            "boundary_margin_tokens": 16,
            "tokenizers_version": version("tokenizers"),
        }

    def bind_document(self, document: DocumentMetadata) -> DocumentMetadata:
        parser_prefix = f"{document.parser_name}-p"
        existing_identity = document.parser_version.removeprefix(parser_prefix)
        parser_hash, separator, chunk_hash = existing_identity.partition("-c")
        parser_identity = (
            parser_hash
            if separator
            and len(parser_hash) == 64
            and all(char in "0123456789abcdef" for char in parser_hash)
            and len(chunk_hash) == 64
            and all(char in "0123456789abcdef" for char in chunk_hash)
            else sha256(document.parser_version.encode()).hexdigest()
        )
        chunk_identity = sha256(
            json.dumps(self.identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        parser_version = f"{parser_prefix}{parser_identity}-c{chunk_identity}"
        document_id = sha256(
            f"{document.source_sha256}\0{document.parser_name}\0{parser_version}".encode()
        ).hexdigest()
        return document.model_copy(
            update={"document_id": document_id, "parser_version": parser_version}
        )

    def __call__(self, document: DocumentMetadata, unit: ParsedUnit, first_ordinal: int):
        return self.split_blocks(
            document, unit, iter_blocks(document, unit, first_ordinal), first_ordinal
        )

    def split_blocks(self, document, unit, parents, first_ordinal=1):
        ordinal = first_ordinal
        line_starts = [0]
        for line in unit.text.splitlines(keepends=True):
            line_starts.append(line_starts[-1] + len(line))
        for parent in parents:
            encoded = self.tokenizer.encode(parent.text, add_special_tokens=False)
            encoded.truncate(self.max_tokens - 16, stride=self.overlap)
            previous_end = 0
            for index, window in enumerate((encoded, *encoded.overflowing)):
                offsets = [(start, end) for start, end in window.offsets if end > start]
                if not offsets:
                    raise ValueError("tokenizer produced a text window without source offsets")
                start = 0 if index == 0 else offsets[0][0]
                end = offsets[-1][1]
                if index == len(encoded.overflowing):
                    end = len(parent.text)
                if start > previous_end or end <= start:
                    raise ValueError("text chunk boundaries lose source characters")
                text = parent.text[start:end]
                if not text.strip():
                    previous_end = end
                    continue
                # Retokenization checks offsets including byte-level Unicode boundaries.
                if len(self.tokenizer.encode(text, add_special_tokens=False).ids) > self.max_tokens:
                    raise ValueError("source window exceeds its tokenizer budget")
                char_start, char_end = parent.char_start + start, parent.char_start + end
                yield Block(
                    document_id=document.document_id,
                    block_id=sha256(f"{document.document_id}\0{ordinal}".encode()).hexdigest(),
                    ordinal=ordinal,
                    unit_index=unit.unit_index,
                    page_number=unit.page_number,
                    line_start=bisect_right(line_starts, char_start),
                    line_end=bisect_right(line_starts, char_end - 1),
                    char_start=char_start,
                    char_end=char_end,
                    text=text,
                )
                ordinal += 1
                previous_end = end


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

    def make_block(block_start: int, block_end: int, block_start_line: int, block_end_line: int):
        block_text = unit.text[block_start:block_end]
        block_id = sha256(f"{document.document_id}\0{ordinal}".encode("utf-8")).hexdigest()
        return Block(
            document_id=document.document_id,
            block_id=block_id,
            ordinal=ordinal,
            unit_index=unit.unit_index,
            page_number=unit.page_number,
            line_start=block_start_line,
            line_end=block_end_line,
            char_start=block_start,
            char_end=block_end,
            text=block_text,
        )

    def split_paragraph(paragraph_start: int, paragraph_end: int):
        cursor = paragraph_start
        while paragraph_end - cursor > MAX_BLOCK_CHARS:
            limit = cursor + MAX_BLOCK_CHARS
            newline_cut = max(
                (index + 1 for index in range(cursor, limit) if unit.text[index] == "\n"),
                default=None,
            )
            if newline_cut is not None and newline_cut > cursor:
                cut = newline_cut
            else:
                sentence_cuts = []
                for index in range(cursor, limit):
                    char = unit.text[index]
                    if char in "。！？":
                        sentence_cuts.append(index + 1)
                    elif (
                        char in ".!?"
                        and index + 1 < paragraph_end
                        and unit.text[index + 1].isspace()
                    ):
                        sentence_cuts.append(index + 1)
                cut = sentence_cuts[-1] if sentence_cuts else limit
            yield cursor, cut
            cursor = cut
        if cursor < paragraph_end:
            yield cursor, paragraph_end

    line_starts = [0]
    for source_line in unit.text.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(source_line))

    def emit_paragraph(paragraph_start: int, paragraph_end: int):
        nonlocal ordinal
        for block_start, block_end in split_paragraph(paragraph_start, paragraph_end):
            block_text = unit.text[block_start:block_end]
            if not block_text.strip():
                continue
            block = make_block(
                block_start,
                block_end,
                bisect_right(line_starts, block_start),
                bisect_right(line_starts, block_end - 1),
            )
            yield block
            ordinal += 1

    for line_number, line in enumerate(unit.text.splitlines(keepends=True), start=1):
        if line.strip():
            if start_offset is None:
                start_offset = offset
            end_offset = offset + len(line)
        elif start_offset is not None:
            yield from emit_paragraph(start_offset, end_offset)
            start_offset = None
        offset += len(line)

    if start_offset is not None:
        yield from emit_paragraph(start_offset, end_offset)
