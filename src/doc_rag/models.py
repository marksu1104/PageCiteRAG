"""Validated data models for doc-rag documents and extracted text."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MediaType = Literal["text/plain", "text/markdown", "application/pdf", "application/vnd.qasper+json"]
DocumentStatus = Literal["ready", "partial", "no_text", "failed"]
UnitStatus = Literal["ok", "no_text", "error"]


class DocumentMetadata(BaseModel):
    """Stable source identity and parsing provenance, without the source path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_name: str = Field(min_length=1, max_length=512)
    media_type: MediaType
    parser_name: Literal["plain_text", "markdown", "pdfplumber", "qasper"]
    parser_version: str = Field(min_length=1, max_length=160)
    language: str | None = Field(default=None, max_length=32)
    usage_scope: str = Field(default="not-recorded", min_length=1, max_length=256)
    page_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_parser_metadata(self) -> DocumentMetadata:
        if self.media_type == "application/pdf" and self.page_count is None:
            raise ValueError("PDF metadata must include its page count")
        if self.media_type != "application/pdf" and self.page_count is not None:
            raise ValueError("text documents do not have PDF page counts")
        expected_parser = {
            "application/pdf": "pdfplumber",
            "text/plain": "plain_text",
            "text/markdown": "markdown",
            "application/vnd.qasper+json": "qasper",
        }[self.media_type]
        if self.parser_name != expected_parser:
            raise ValueError("parser name does not match the source media type")
        return self


class ParsedUnit(BaseModel):
    """One PDF page or one complete text file, including its exact extracted text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_index: int = Field(ge=1)
    page_number: int | None = Field(default=None, ge=1)
    text: str
    status: UnitStatus
    error_type: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def validate_extracted_text(self) -> ParsedUnit:
        if self.status == "ok" and not self.text.strip():
            raise ValueError("a readable unit must contain non-whitespace text")
        if self.status != "ok" and self.text:
            raise ValueError("empty or failed units must not contain extracted text")
        if self.status == "error" and not self.error_type:
            raise ValueError("failed units must record the exception type")
        if self.status != "error" and self.error_type is not None:
            raise ValueError("only failed units may record an exception type")
        return self


class Block(BaseModel):
    """A paragraph block with one-based lines and zero-based half-open text offsets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    block_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordinal: int = Field(ge=1)
    unit_index: int = Field(ge=1)
    page_number: int | None = Field(default=None, ge=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_offsets(self) -> Block:
        if self.line_end < self.line_start:
            raise ValueError("line_end must not precede line_start")
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        if not self.text.strip():
            raise ValueError("blocks must contain non-whitespace text")
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("character offsets must match the stored block text")
        return self


class DocumentRecord(DocumentMetadata):
    """Persisted document summary."""

    status: DocumentStatus
    unit_count: int = Field(ge=0)
    block_count: int = Field(ge=0)
    imported_at: datetime


class UnitSummary(BaseModel):
    """Non-content unit status safe to include in document metadata output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_index: int = Field(ge=1)
    page_number: int | None = Field(default=None, ge=1)
    status: UnitStatus
    error_type: str | None = None


class IngestResult(BaseModel):
    document: DocumentRecord
    duplicate: bool
    empty_unit_indexes: tuple[int, ...] = ()
    error_unit_indexes: tuple[int, ...] = ()
