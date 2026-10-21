from __future__ import annotations

import json
import sqlite3
from hashlib import sha256

import pytest
from pydantic import ValidationError

from page_cite_rag.citations import (
    AnswerDraft,
    CitationDraft,
    CitationError,
    ClaimDraft,
    validate_claims,
)
from page_cite_rag.models import DocumentMetadata, ParsedUnit
from page_cite_rag.store import SOURCE_LOCATIONS_KIND, SQLiteDocumentStore


def fixture_store(tmp_path, text="Heading\r\n\r\nNo improvement unless n > 64.\r\n中文細節。\r\n"):
    document_id = sha256(text.encode("utf-8")).hexdigest()
    document = DocumentMetadata(
        document_id=document_id,
        source_sha256=document_id,
        source_name="self-authored.txt",
        media_type="text/plain",
        parser_name="plain_text",
        parser_version="synthetic-1",
        language="en+zh-Hant",
        usage_scope="self-authored-test",
    )
    store = SQLiteDocumentStore(tmp_path / "citations.sqlite3")
    store.ingest(document, [ParsedUnit(unit_index=1, text=text, status="ok")])
    return store, document_id, store.get_blocks(document_id)


def answer(block, quote, text="The source reports a conditional negative result."):
    return AnswerDraft(
        status="answered",
        claims=(
            ClaimDraft(text=text, citations=(CitationDraft(block_id=block.block_id, quote=quote),)),
        ),
    )


def test_exact_quote_uses_authoritative_unit_offsets_and_quote_line_range(tmp_path):
    store, document_id, blocks = fixture_store(tmp_path)
    draft = answer(blocks[1], "n > 64.\r\n中文細節。")
    verified = validate_claims(store, document_id, draft, blocks)
    citation = verified[0].citations[0]
    unit = store.get_unit(document_id, 1)
    assert citation.quote == "n > 64.\r\n中文細節。"
    assert unit.text[citation.char_start : citation.char_end] == citation.quote
    assert citation.char_start == unit.text.index(citation.quote)
    assert citation.line_start == 3 and citation.line_end == 4
    assert citation.unit_index == 1 and citation.page_number is None
    assert citation.document_id == document_id
    assert verified[0].text == draft.claims[0].text


def test_single_line_subquote_does_not_inherit_full_block_line_range(tmp_path):
    store, document_id, blocks = fixture_store(tmp_path)
    citation = validate_claims(store, document_id, answer(blocks[1], "中文細節。"), blocks)[
        0
    ].citations[0]
    assert citation.line_start == citation.line_end == 4
    assert blocks[1].line_start == 3 and blocks[1].line_end == 4


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r", "\u2028", "\u2029", "\x85"])
def test_quote_line_positions_follow_original_splitlines_semantics(tmp_path, ending):
    text = f"Title{ending}{ending}First detail.{ending}Second detail.{ending}"
    store, document_id, blocks = fixture_store(tmp_path, text)
    quote = f"detail.{ending}Second"
    citation = validate_claims(store, document_id, answer(blocks[1], quote), blocks)[0].citations[0]
    assert citation.line_start == 3 and citation.line_end == 4
    assert text[citation.char_start : citation.char_end] == quote


def test_pdf_quote_page_is_derived_from_source_not_model(tmp_path):
    document_id = sha256(b"self-authored-two-page-fixture").hexdigest()
    document = DocumentMetadata(
        document_id=document_id,
        source_sha256=document_id,
        source_name="self-authored.pdf",
        media_type="application/pdf",
        parser_name="pdfplumber",
        parser_version="synthetic-1",
        page_count=2,
    )
    store = SQLiteDocumentStore(tmp_path / "pdf.sqlite3")
    store.ingest(
        document,
        [
            ParsedUnit(unit_index=1, page_number=1, text="First page.", status="ok"),
            ParsedUnit(unit_index=2, page_number=2, text="Second page detail.", status="ok"),
        ],
    )
    blocks = store.get_blocks(document_id)
    citation = validate_claims(store, document_id, answer(blocks[1], "detail."), blocks)[
        0
    ].citations[0]
    assert citation.unit_index == citation.page_number == 2
    assert citation.char_start == 12 and citation.char_end == 19


def test_saved_multi_page_source_map_is_preserved_without_guessing_one_page(tmp_path):
    store, document_id, blocks = fixture_store(tmp_path)
    store.save_retrieval_manifest_once(
        document_id,
        SOURCE_LOCATIONS_KIND,
        json.dumps(
            {
                "version": 1,
                "document_id": document_id,
                "source_sha256": document_id,
                "locations": {
                    blocks[1].block_id: {
                        "pages": [3, 4],
                        "native_refs": ["#/texts/2"],
                        "doc_items": [{"bbox": {"l": 10, "t": 20, "r": 30, "b": 40}}],
                    }
                },
            }
        ),
    )
    citation = validate_claims(store, document_id, answer(blocks[1], "中文細節。"), blocks)[
        0
    ].citations[0]
    assert citation.page_number is None
    assert citation.source_pages == (3, 4)
    assert citation.native_refs == ("#/texts/2",)


@pytest.mark.parametrize("quote", ["Improvement", "n >= 64", " n > 64. "])
def test_fabricated_or_nonexact_quotes_fail_closed(tmp_path, quote):
    store, document_id, blocks = fixture_store(tmp_path)
    with pytest.raises(CitationError, match="exact source substring"):
        validate_claims(store, document_id, answer(blocks[1], quote), blocks)


def test_shorter_verbatim_quote_is_valid_without_sentence_boundary_requirement(tmp_path):
    store, document_id, blocks = fixture_store(tmp_path)
    assert validate_claims(store, document_id, answer(blocks[1], "中文細節"), blocks)


@pytest.mark.parametrize(
    ("source", "quote"),
    [
        (r"Scores $\le $ 12 were included.", "Scores ≤ 12 were included."),
        (r"The directions are $\perp $.", "The directions are ⊥."),
    ],
)
def test_math_rendering_is_not_a_verbatim_quote(tmp_path, source, quote):
    store, document_id, blocks = fixture_store(tmp_path, source)
    with pytest.raises(CitationError, match="exact source substring"):
        validate_claims(store, document_id, answer(blocks[0], quote), blocks)


@pytest.mark.parametrize(("source", "quote"), [("same same", "same"), ("aaaa", "aaa")])
def test_repeated_and_overlapping_quotes_are_ambiguous(tmp_path, source, quote):
    store, document_id, blocks = fixture_store(tmp_path, source)
    with pytest.raises(CitationError, match="ambiguous"):
        validate_claims(store, document_id, answer(blocks[0], quote), blocks)


def test_matching_other_block_does_not_authorize_an_out_of_context_citation(tmp_path):
    store, document_id, blocks = fixture_store(tmp_path)
    with pytest.raises(CitationError, match="not provided"):
        validate_claims(store, document_id, answer(blocks[1], "中文細節。"), blocks[:1])


def test_cross_document_context_and_ids_are_rejected(tmp_path):
    store, document_id, blocks = fixture_store(tmp_path)
    other_id = "f" * 64
    with pytest.raises(CitationError, match="scope"):
        validate_claims(store, other_id, answer(blocks[0], "Heading"), blocks)
    foreign = blocks[0].model_copy(update={"document_id": other_id})
    with pytest.raises(CitationError, match="scope"):
        validate_claims(store, document_id, answer(foreign, "Heading"), [foreign])


def test_forged_context_text_is_rejected_even_when_quote_is_present(tmp_path):
    store, document_id, blocks = fixture_store(tmp_path)
    original = blocks[0]
    forged = original.model_copy(update={"text": "Forgery\r\n"})
    with pytest.raises(CitationError, match="authoritative"):
        validate_claims(store, document_id, answer(forged, "Forgery"), [forged])


def test_uncited_stale_context_is_also_rejected(tmp_path):
    store, document_id, blocks = fixture_store(tmp_path)
    stale = blocks[1].model_copy(update={"page_number": 99})
    with pytest.raises(CitationError, match="authoritative"):
        validate_claims(store, document_id, answer(blocks[0], "Heading"), [blocks[0], stale])


def test_source_changed_after_context_creation_is_rejected(tmp_path):
    store, document_id, blocks = fixture_store(tmp_path)
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE document_units SET text = ? WHERE document_id = ?",
            ("Changed source without changing stored blocks.", document_id),
        )
    with pytest.raises(CitationError, match="inconsistent"):
        validate_claims(store, document_id, answer(blocks[0], "Heading"), blocks)


def test_missing_block_fails_closed(tmp_path):
    store, document_id, blocks = fixture_store(tmp_path)
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DELETE FROM blocks WHERE document_id = ?", (document_id,))
    with pytest.raises(CitationError, match="authoritative"):
        validate_claims(store, document_id, answer(blocks[0], "Heading"), blocks)


def test_missing_unit_fails_closed_even_with_a_remaining_block(tmp_path):
    store, document_id, blocks = fixture_store(tmp_path)
    # A separate default SQLite connection has foreign-key enforcement off;
    # simulate corruption, never a supported store operation.
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DELETE FROM document_units WHERE document_id = ?", (document_id,))
    with pytest.raises(CitationError, match="unit is missing"):
        validate_claims(store, document_id, answer(blocks[0], "Heading"), blocks)


def test_corrupt_stored_block_metadata_cannot_supply_quote_locations(tmp_path):
    store, document_id, blocks = fixture_store(tmp_path)
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE blocks SET line_start = 99, line_end = 99 WHERE document_id = ?",
            (document_id,),
        )
    corrupted = store.get_blocks(document_id)
    with pytest.raises(CitationError, match="inconsistent"):
        validate_claims(store, document_id, answer(corrupted[0], "Heading"), corrupted)


def test_duplicate_context_id_is_rejected(tmp_path):
    store, document_id, blocks = fixture_store(tmp_path)
    with pytest.raises(CitationError, match="duplicate"):
        validate_claims(store, document_id, answer(blocks[0], "Heading"), [blocks[0], blocks[0]])


def test_source_errors_do_not_leak_private_text(tmp_path):
    store, document_id, blocks = fixture_store(tmp_path)

    class BrokenSource:
        def get_block(self, document_id, block_id):
            raise RuntimeError("private-source-canary")

    with pytest.raises(CitationError, match="cannot be read") as raised:
        validate_claims(BrokenSource(), document_id, answer(blocks[0], "Heading"), blocks)
    assert "private-source-canary" not in str(raised.value)


def test_insufficient_evidence_has_no_unchecked_answer_claims(tmp_path):
    store, document_id, _blocks = fixture_store(tmp_path)
    draft = AnswerDraft(status="insufficient_evidence", limitations=("No relevant evidence.",))
    assert validate_claims(store, document_id, draft, []) == ()
    with pytest.raises(ValidationError):
        AnswerDraft(status="answered")
    with pytest.raises(ValidationError):
        AnswerDraft(status="insufficient_evidence", claims=answer(_blocks[0], "Heading").claims)


def test_models_forbid_uncited_answer_fields_and_model_proposed_source_positions(tmp_path):
    _store, _document_id, blocks = fixture_store(tmp_path)
    with pytest.raises(ValidationError):
        AnswerDraft(status="answered", answer="Unchecked prose", claims=())
    for field, value in [("page_number", 1), ("char_start", 0), ("document_id", "a" * 64)]:
        with pytest.raises(ValidationError):
            CitationDraft(block_id=blocks[0].block_id, quote="Heading", **{field: value})
    with pytest.raises(ValidationError):
        ClaimDraft(text="Uncited", citations=())


@pytest.mark.parametrize("quote", ["", " \r\n\t", "x" * 4001, 12])
def test_empty_oversized_or_coerced_quotes_are_rejected(quote):
    with pytest.raises(ValidationError):
        CitationDraft(block_id="a" * 64, quote=quote)


def test_draft_count_bounds_and_strict_immutability():
    citation = CitationDraft(block_id="a" * 64, quote="Exact source.")
    claim = ClaimDraft(text="One supported claim.", citations=(citation,))
    for kwargs in (
        {"text": " \t", "citations": (citation,)},
        {"text": "x" * 4001, "citations": (citation,)},
        {"text": "Too many references.", "citations": (citation,) * 9},
        {"text": "Not a tuple.", "citations": [citation]},
    ):
        with pytest.raises(ValidationError):
            ClaimDraft(**kwargs)
    for kwargs in (
        {"status": "answered", "claims": (claim,) * 13},
        {"status": "answered", "claims": (claim,), "limitations": ("Limit",) * 9},
        {"status": "answered", "claims": (claim,), "limitations": (" ",)},
        {"status": "answered", "claims": (claim,), "limitations": ("x" * 1001,)},
    ):
        with pytest.raises(ValidationError):
            AnswerDraft(**kwargs)
    with pytest.raises(ValidationError):
        citation.quote = "Changed source."
    with pytest.raises(ValidationError):
        CitationDraft(block_id="A" * 64, quote="Exact source.")


def test_json_arrays_validate_as_strict_tuples_and_text_is_not_trimmed():
    draft = AnswerDraft.model_validate_json(
        json.dumps(
            {
                "status": "answered",
                "claims": [
                    {
                        "text": "  Exact claim spacing.  ",
                        "citations": [{"block_id": "a" * 64, "quote": " source spacing "}],
                    }
                ],
                "limitations": [],
            }
        )
    )
    assert isinstance(draft.claims, tuple)
    assert draft.claims[0].text == "  Exact claim spacing.  "
    assert draft.claims[0].citations[0].quote == " source spacing "


def test_bypassed_invalid_draft_is_revalidated(tmp_path):
    store, document_id, blocks = fixture_store(tmp_path)
    bypassed = AnswerDraft.model_construct(status="answered", claims=(), limitations=())
    with pytest.raises(CitationError, match="invalid answer"):
        validate_claims(store, document_id, bypassed, blocks)


def test_provenance_validation_is_not_semantic_entailment(tmp_path):
    store, document_id, blocks = fixture_store(tmp_path)
    contradictory = answer(blocks[1], "No improvement unless n > 64.", text="There is improvement.")
    assert (
        validate_claims(store, document_id, contradictory, blocks)[0].text
        == "There is improvement."
    )
