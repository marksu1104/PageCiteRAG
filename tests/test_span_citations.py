from __future__ import annotations

import json
import sqlite3

import pytest
from pydantic import ValidationError

from page_cite_rag.citations import AnswerDraft, CitationError
from page_cite_rag.span_citations import (
    MAX_SPAN_CHARS,
    SpanAnswerDraft,
    SpanCitationDraft,
    SpanClaimDraft,
    provided_single_answer_model,
    provided_span_answer_model,
    source_spans,
    validate_span_claims,
)
from test_citations import fixture_store


@pytest.mark.parametrize("count", [1, 2])
def test_provided_model_choices_and_inherited_source_validation(tmp_path, count):
    store, doc, blocks = fixture_store(tmp_path)
    blocks = blocks[:count]
    spans = source_spans(blocks)
    original_schema = SpanAnswerDraft.model_json_schema()
    scoped = provided_span_answer_model(blocks)
    schema = scoped.model_json_schema()
    choices = schema["$defs"]["ProvidedSpanCitationDraft"]["properties"]["span_id"]
    assert choices.get("enum", [choices.get("const")]) == [s.span_id for s in spans]
    assert issubclass(scoped, SpanAnswerDraft)
    draft = scoped.model_validate_json(answer(spans[-1]).model_dump_json())
    assert validate_span_claims(store, doc, draft, blocks)[0].citations[0].quote == spans[-1].text
    foreign = answer(spans[0]).model_dump(mode="json")
    foreign["claims"][0]["citations"][0]["span_id"] = "f" * 64
    with pytest.raises(ValidationError):
        scoped.model_validate_json(json.dumps(foreign))
    assert SpanAnswerDraft.model_json_schema() == original_schema


def test_single_answer_schema_preserves_one_claim_request_scope_and_abstention(tmp_path):
    _, _, blocks = fixture_store(tmp_path)
    scoped = provided_single_answer_model(blocks[:1])
    value = answer(source_spans(blocks[:1])[0]).model_dump(mode="json")
    assert len(scoped.model_validate_json(json.dumps(value)).claims) == 1
    assert scoped.model_json_schema()["properties"]["claims"]["maxItems"] == 1
    with pytest.raises(ValidationError):
        scoped.model_validate_json(json.dumps({**value, "claims": value["claims"] * 2}))
    with pytest.raises(ValidationError):
        scoped.model_validate_json(answer(source_spans(blocks[1:])[0]).model_dump_json())
    empty = provided_single_answer_model(())
    assert empty(status="insufficient_evidence").claims == ()
    with pytest.raises(ValidationError):
        empty(status="answered")


def test_provided_models_do_not_share_choices_between_requests(tmp_path):
    _, _, blocks = fixture_store(tmp_path)
    first = provided_span_answer_model(blocks[:1])
    second = provided_span_answer_model(blocks[1:])
    draft = answer(source_spans(blocks[:1])[0]).model_dump_json()
    first.model_validate_json(draft)
    with pytest.raises(ValidationError):
        second.model_validate_json(draft)
    first.model_validate_json(draft)


def test_empty_provided_model_can_only_abstain():
    scoped = provided_span_answer_model(())
    assert scoped(status="insufficient_evidence").claims == ()
    with pytest.raises(ValidationError):
        scoped(status="answered")
    assert scoped.model_json_schema()["properties"]["claims"]["maxItems"] == 0


@pytest.mark.parametrize(
    "mutation", ["no_citation", "many_citations", "many_claims", "blank", "status", "extra"]
)
def test_provided_model_retains_bounds_and_status_contract(tmp_path, mutation):
    _, _, blocks = fixture_store(tmp_path)
    scoped = provided_span_answer_model(blocks)
    value = answer(source_spans(blocks)[0]).model_dump(mode="json")
    if mutation == "no_citation":
        value["claims"][0]["citations"] = []
    elif mutation == "many_citations":
        value["claims"][0]["citations"] *= 9
    elif mutation == "many_claims":
        value["claims"] *= 13
    elif mutation == "blank":
        value["claims"][0]["text"] = " "
    elif mutation == "status":
        value["status"] = "insufficient_evidence"
    else:
        value["extra"] = "not allowed"
    with pytest.raises(ValidationError):
        scoped.model_validate_json(json.dumps(value))


def answer(span, text="The source states a conditional result."):
    return SpanAnswerDraft(
        status="answered",
        claims=(SpanClaimDraft(text=text, citations=(SpanCitationDraft(span_id=span.span_id),)),),
    )


@pytest.mark.parametrize(
    "text",
    [
        r"Scores $\le $ 12 were included; directions are $\perp $. Only if n > 64.",
        "中文：只有在 n > 64 時有效；不包含兒童。",
        "  Keep spacing, e\u0301 and é distinct.\r\nDo not normalize.  ",
        "same same same",
    ],
)
def test_span_quotes_preserve_exact_source_representation(tmp_path, text):
    store, doc, blocks = fixture_store(tmp_path, text)
    spans = source_spans(blocks)
    assert spans == source_spans(blocks) and len(spans) == 1
    citation = validate_span_claims(store, doc, answer(spans[0]), blocks)[0].citations[0]
    assert citation.quote == blocks[0].text
    assert store.get_unit(doc, 1).text[citation.char_start : citation.char_end] == citation.quote
    assert citation.block_id == blocks[0].block_id and citation.document_id == doc


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r", "\u2028", "\u2029", "\x85"])
def test_line_offsets_follow_original_unit(tmp_path, ending):
    store, doc, blocks = fixture_store(tmp_path, f"Title{ending}{ending}First.{ending}第二行。")
    span = source_spans(blocks[1:])[0]
    citation = validate_span_claims(store, doc, answer(span), blocks)[0].citations[0]
    assert citation.line_start == 3 and citation.line_end == 4


def test_long_block_is_partitioned_without_deleting_or_rewriting_text(tmp_path):
    text = "中文條件" * 1200 + r"$\le $" + "n" * 4000
    store, doc, blocks = fixture_store(tmp_path, text)
    spans = source_spans(blocks)
    assert len(spans) == 3 and "".join(span.text for span in spans) == text
    assert all(len(span.text) <= MAX_SPAN_CHARS for span in spans)
    assert [span.char_start for span in spans] == [0, 4000, 8000]
    for span in spans:
        citation = validate_span_claims(store, doc, answer(span), blocks)[0].citations[0]
        assert citation.quote == text[span.char_start : span.char_end]


@pytest.mark.parametrize("change", ["text", "position", "document", "identity"])
def test_ids_bind_source_identity_and_range(tmp_path, change):
    _, _, blocks = fixture_store(tmp_path, "Same source.")
    old = source_spans(blocks)[0]
    block = blocks[0]
    updates = {
        "text": {"text": "Same sourCe."},
        "position": {"char_start": 1, "char_end": block.char_end + 1},
        "document": {"document_id": "f" * 64},
        "identity": {"block_id": "f" * 64},
    }[change]
    assert source_spans([block.model_copy(update=updates)])[0].span_id != old.span_id


def test_out_of_context_and_cross_document_ids_are_not_authorized(tmp_path):
    store, doc, blocks = fixture_store(tmp_path)
    foreign = source_spans(blocks[1:])[0]
    with pytest.raises(CitationError, match="not provided"):
        validate_span_claims(store, doc, answer(foreign), blocks[:1])
    with pytest.raises(CitationError, match="scope"):
        validate_span_claims(store, "f" * 64, answer(foreign), blocks)


@pytest.mark.parametrize("mutation", ["text", "page", "delete_unit", "delete_block"])
def test_stale_or_corrupt_authoritative_source_fails_closed(tmp_path, mutation):
    store, doc, blocks = fixture_store(tmp_path)
    draft = answer(source_spans(blocks)[0])
    with sqlite3.connect(store.database_path) as db:
        if mutation == "text":
            db.execute("UPDATE document_units SET text = 'Changed' WHERE document_id = ?", (doc,))
        elif mutation == "page":
            db.execute("UPDATE document_units SET page_number = 99 WHERE document_id = ?", (doc,))
        else:
            table = "document_units" if mutation == "delete_unit" else "blocks"
            db.execute(f"DELETE FROM {table} WHERE document_id = ?", (doc,))
    with pytest.raises(CitationError):
        validate_span_claims(store, doc, draft, blocks)


def test_uncited_forged_block_and_duplicate_context_also_fail(tmp_path):
    store, doc, blocks = fixture_store(tmp_path)
    draft = answer(source_spans(blocks)[0])
    forged = blocks[1].model_copy(update={"page_number": 99})
    with pytest.raises(CitationError, match="authoritative"):
        validate_span_claims(store, doc, draft, [blocks[0], forged])
    with pytest.raises(CitationError, match="duplicate"):
        validate_span_claims(store, doc, draft, [blocks[0], blocks[0]])


def test_duplicate_span_per_claim_is_rejected(tmp_path):
    store, doc, blocks = fixture_store(tmp_path)
    citation = SpanCitationDraft(span_id=source_spans(blocks)[0].span_id)
    draft = SpanAnswerDraft(
        status="answered", claims=(SpanClaimDraft(text="Claim.", citations=(citation, citation)),)
    )
    with pytest.raises(CitationError, match="duplicate citation"):
        validate_span_claims(store, doc, draft, blocks)


@pytest.mark.parametrize(
    "field,value",
    [("quote", "Forged"), ("block_id", "a" * 64), ("page_number", 8), ("char_start", 9)],
)
def test_model_cannot_supply_source_text_or_positions(field, value):
    with pytest.raises(ValidationError):
        SpanCitationDraft(span_id="a" * 64, **{field: value})


@pytest.mark.parametrize("span_id", ["", "s1", "A" * 64, 123, "a" * 65])
def test_invalid_span_ids_fail_schema(span_id):
    with pytest.raises(ValidationError):
        SpanCitationDraft(span_id=span_id)


def test_inherited_answer_bounds_nonblank_and_status_rules():
    citation = SpanCitationDraft(span_id="a" * 64)
    with pytest.raises(ValidationError):
        SpanClaimDraft(text=" \t", citations=(citation,))
    with pytest.raises(ValidationError):
        SpanAnswerDraft(status="answered")
    claim = SpanClaimDraft(text="Claim.", citations=(citation,))
    with pytest.raises(ValidationError):
        SpanAnswerDraft(status="insufficient_evidence", claims=(claim,))
    with pytest.raises(ValidationError):
        SpanAnswerDraft(status="answered", claims=(claim,) * 13)
    with pytest.raises(ValidationError):
        SpanAnswerDraft(status="answered", claims=(claim,), limitations=(" ",))
    assert (
        SpanAnswerDraft.model_validate_json(
            json.dumps(
                {
                    "status": "answered",
                    "claims": [{"text": "Claim.", "citations": [{"span_id": "a" * 64}]}],
                }
            )
        )
        .claims[0]
        .citations[0]
        .span_id
        == "a" * 64
    )


def test_bypass_constructed_draft_and_wrong_strategy_are_rejected(tmp_path):
    store, doc, blocks = fixture_store(tmp_path)
    with pytest.raises(CitationError, match="required"):
        validate_span_claims(store, doc, AnswerDraft(status="insufficient_evidence"), blocks)
    bad = SpanAnswerDraft.model_construct(status="answered", claims=())
    with pytest.raises(CitationError, match="invalid span"):
        validate_span_claims(store, doc, bad, blocks)


def test_valid_provenance_is_not_semantic_entailment(tmp_path):
    store, doc, blocks = fixture_store(tmp_path, "Children were excluded.")
    claims = validate_span_claims(
        store, doc, answer(source_spans(blocks)[0], text="Children were included."), blocks
    )
    assert claims[0].text == "Children were included."  # Requires separate semantic review.


def test_source_error_does_not_leak_text(tmp_path):
    _, doc, blocks = fixture_store(tmp_path)

    class Broken:
        def get_block(self, *args):
            raise RuntimeError("PRIVATE_CANARY")

    with pytest.raises(CitationError, match="cannot be read") as raised:
        validate_span_claims(Broken(), doc, answer(source_spans(blocks)[0]), blocks)
    assert "PRIVATE_CANARY" not in str(raised.value)
