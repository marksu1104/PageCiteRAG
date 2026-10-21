"""Full answers may use multiple cited paragraphs without trusting model locations."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from page_cite_rag.full_answer import (
    EvidenceFirstRunFailure,
    EvidenceSelectionDraft,
    FullAnswerWorkflow,
)
from page_cite_rag.generation import (
    GenerationConfig,
    GenerationResult,
    GenerationStats,
    OutputFailureAccounting,
    OutputValidationError,
    check_budget,
)
from page_cite_rag.ingest import prepare_source
from page_cite_rag.retrieval import SearchContextGroup, SearchHit, SearchResult
from page_cite_rag.store import SQLiteDocumentStore
from page_cite_rag.workflows import WorkflowConfig


class _Retriever:
    def __init__(self, document_id, blocks):
        self.document_id = document_id
        self.blocks = blocks

    def search(self, query, *, top_k=5):
        return SearchResult(
            document_id=self.document_id,
            index_id="synthetic-index",
            method="hybrid",
            query=query,
            hits=tuple(
                SearchHit(rank=index, score=1.0 / index, block=block)
                for index, block in enumerate(self.blocks[:top_k], 1)
            ),
            elapsed_ms=0.1,
        )


class _Model:
    identity = {"model": "synthetic"}

    def __init__(self, config):
        self.config = config
        self.seen = None
        self.seen_request = None
        self.partial = False
        self.invalid_selection_id = False
        self.duplicate_selection_id = False
        self.select_last = False
        self.seen_requests = []
        self.fail_at = None

    def generate(self, messages, response_model):
        payload = json.loads(messages[1].content)
        self.seen_requests.append((response_model, payload))
        if issubclass(response_model, EvidenceSelectionDraft):
            if self.fail_at == "selector":
                raise _invalid_output_error()
            candidate = payload["candidates"][-1 if self.select_last else 0]
            source_id = "S999" if self.invalid_selection_id else candidate["evidence_id"]
            selected_ids = [source_id, source_id] if self.duplicate_selection_id else [source_id]
            draft = {"selected_source_ids": selected_ids, "uncovered_parts": []}
            value = response_model.model_validate_json(json.dumps(draft))
            return GenerationResult(
                value,
                GenerationStats(
                    prompt_eval_count=100,
                    eval_count=40,
                    total_duration=100,
                    load_duration=None,
                    prompt_eval_duration=40,
                    eval_duration=60,
                    done_reason="stop",
                    elapsed_ms=1.0,
                ),
                check_budget(messages, response_model.model_json_schema(), self.config),
            )
        if self.fail_at == "answer":
            raise _invalid_output_error()
        self.seen = payload
        evidence = payload.get("evidence", payload.get("sources", []))
        if not evidence:
            evidence = [
                source
                for group in payload.get("context_groups", [])
                for source in group.get("sources", [])
            ]
        self.seen_request = {
            "messages": [message.model_dump(mode="json") for message in messages],
            "schema": response_model.model_json_schema(),
            "budget": check_budget(messages, response_model.model_json_schema(), self.config),
        }
        if self.partial:
            draft = {
                "status": "partial",
                "answer_claims": [
                    {
                        "text": "The result rose by 42%.",
                        "citations": [{"source_id": evidence[0]["source_id"]}],
                    }
                ],
                "detail_claims": [],
                "limitations": ["The supplied evidence does not state the requested condition."],
            }
        else:
            draft = {
                "status": "answered",
                "answer_claims": [
                    {
                        "text": "The result rose by 42% only under condition A.",
                        "citations": [{"source_id": item["source_id"]} for item in evidence[:2]],
                    },
                ],
                "detail_claims": [
                    {
                        "text": "Condition A limits when the reported 42% increase applies.",
                        "citations": [{"source_id": item["source_id"]} for item in evidence[:2]],
                    },
                ],
            }
        value = response_model.model_validate_json(json.dumps(draft))
        return GenerationResult(
            value,
            GenerationStats(
                prompt_eval_count=100,
                eval_count=40,
                total_duration=100,
                load_duration=None,
                prompt_eval_duration=40,
                eval_duration=60,
                done_reason="stop",
                elapsed_ms=1.0,
            ),
            check_budget(messages, response_model.model_json_schema(), self.config),
        )

    def close(self):
        pass


def test_formal_answer_uses_short_ids_and_validates_multiple_source_paragraphs(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("The result rose by 42%.\n\nOnly under condition A.\n", encoding="utf-8")
    prepared = prepare_source(path)
    store = SQLiteDocumentStore(tmp_path / "paper.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    blocks = store.get_blocks(prepared.document.document_id)
    config = GenerationConfig(num_ctx=16384, num_predict=1536)
    model = _Model(config)
    workflow = FullAnswerWorkflow(
        SQLiteDocumentStore(store.database_path, read_only=True),
        prepared.document.document_id,
        _Retriever(prepared.document.document_id, blocks),
        model,
        config,
    )
    result = workflow.ask("How much, and under what condition?")
    assert result.status == "answered"
    assert result.prompt_version == "complete-cited-answer-v4"
    assert len(result.answer_claims) == 1
    assert len(result.detail_claims) == 1
    assert result.answer == "The result rose by 42% only under condition A."
    assert [item["source_id"] for item in model.seen["evidence"]] == ["S1", "S2"]
    assert result.answer_claims[0].citations[0].quote.startswith("The result rose")
    assert result.answer_claims[0].citations[1].quote.startswith("Only under")


def test_request_schema_enforces_status_claim_and_source_id_contract(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("The result rose by 42%.\n", encoding="utf-8")
    prepared = prepare_source(path)
    store = SQLiteDocumentStore(tmp_path / "paper.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    blocks = store.get_blocks(prepared.document.document_id)
    config = GenerationConfig(num_ctx=16384, num_predict=1536)
    workflow = FullAnswerWorkflow(
        SQLiteDocumentStore(store.database_path, read_only=True),
        prepared.document.document_id,
        _Retriever(prepared.document.document_id, blocks),
        _Model(config),
        config,
    )

    response_model = workflow._response_model_for(blocks)
    schema = response_model.model_json_schema()
    assert len(schema["oneOf"]) == 3
    assert schema["discriminator"]["propertyName"] == "status"
    assert schema["$defs"]["ProvidedShortCitationDraft"]["properties"]["source_id"]["const"] == "S1"

    claim = {"text": "The result rose by 42%.", "citations": [{"source_id": "S1"}]}
    with pytest.raises(ValidationError):
        response_model.model_validate_json(
            json.dumps(
                {
                    "status": "answered",
                    "answer_claims": [],
                    "detail_claims": [],
                    "limitations": [],
                }
            )
        )
    with pytest.raises(ValidationError):
        response_model.model_validate_json(
            json.dumps(
                {
                    "status": "partial",
                    "answer_claims": [claim],
                    "detail_claims": [],
                    "limitations": [],
                }
            )
        )
    with pytest.raises(ValidationError):
        response_model.model_validate_json(
            json.dumps(
                {
                    "status": "answered",
                    "answer_claims": [
                        {
                            "text": "The result rose by 42%.",
                            "citations": [{"source_id": "S999"}],
                        }
                    ],
                    "detail_claims": [],
                    "limitations": [],
                }
            )
        )
    with pytest.raises(ValidationError):
        response_model.model_validate_json(
            json.dumps(
                {
                    "status": "insufficient_evidence",
                    "answer_claims": [claim],
                    "detail_claims": [],
                    "limitations": [],
                }
            )
        )

    valid = response_model.model_validate_json(
        json.dumps(
            {
                "status": "answered",
                "answer_claims": [claim],
                "detail_claims": [],
                "limitations": [],
            }
        )
    )
    assert valid.status == "answered"


@pytest.mark.parametrize(
    ("answer_format", "payload", "expected_answer"),
    [
        (
            "short_answer",
            {
                "status": "answered",
                "short_answer": {
                    "text": "42% under condition A",
                    "citations": [{"source_id": "S1"}],
                },
                "detail_claims": [],
                "limitations": [],
            },
            "42% under condition A",
        ),
        (
            "multiple_choice",
            {
                "status": "answered",
                "choice": "C",
                "choice_citations": [{"source_id": "S1"}],
                "detail_claims": [],
                "limitations": [],
            },
            "The correct answer is (C).",
        ),
    ],
)
def test_benchmark_answer_formats_are_schema_constrained_and_normalized(
    tmp_path, answer_format, payload, expected_answer
):
    path = tmp_path / "paper.txt"
    path.write_text("The result rose by 42% under condition A.\n", encoding="utf-8")
    prepared = prepare_source(path)
    store = SQLiteDocumentStore(tmp_path / "paper.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    blocks = store.get_blocks(prepared.document.document_id)
    config = GenerationConfig(num_ctx=16384, num_predict=1536)

    class _TaskModel(_Model):
        def generate(self, messages, response_model):
            request = json.loads(messages[1].content)
            self.seen_request = {
                "messages": [message.model_dump(mode="json") for message in messages],
                "schema": response_model.model_json_schema(),
            }
            evidence = request["evidence"]
            source_id = evidence[0]["source_id"]
            if answer_format == "short_answer":
                draft = {
                    "status": "answered",
                    "short_answer": {
                        "text": expected_answer,
                        "citations": [{"source_id": source_id}],
                    },
                    "detail_claims": [],
                    "limitations": [],
                }
            else:
                draft = {
                    "status": "answered",
                    "choice": "C",
                    "choice_citations": [{"source_id": source_id}],
                    "detail_claims": [],
                    "limitations": [],
                }
            value = response_model.model_validate_json(json.dumps(draft))
            budget = check_budget(messages, response_model.model_json_schema(), self.config)
            return GenerationResult(
                value,
                GenerationStats(
                    prompt_eval_count=budget.exact_input_tokens or 100,
                    eval_count=40,
                    total_duration=100,
                    load_duration=None,
                    prompt_eval_duration=40,
                    eval_duration=60,
                    done_reason="stop",
                    elapsed_ms=1.0,
                ),
                budget,
            )

    workflow = FullAnswerWorkflow(
        SQLiteDocumentStore(store.database_path, read_only=True),
        prepared.document.document_id,
        _Retriever(prepared.document.document_id, blocks),
        _TaskModel(config),
        config,
        answer_format=answer_format,
    )

    response_model = workflow._response_model_for(blocks)
    schema = response_model.model_json_schema()
    assert len(schema["oneOf"]) == 3
    if answer_format == "multiple_choice":
        choice_schema = schema["$defs"]["AnsweredMultipleChoiceBenchmarkAnswerDraft"]
        assert choice_schema["properties"]["choice"]["enum"] == ["A", "B", "C", "D"]
        invalid = {**payload, "choice": "E"}
        with pytest.raises(ValidationError):
            response_model.model_validate_json(json.dumps(invalid))

    result = workflow.answer_from_context("What is the supported answer?", blocks)
    assert result.status == "answered"
    assert result.answer == expected_answer
    assert result.answer_claims[0].citations[0].quote == blocks[0].text
    assert result.detail_claims == ()


def test_evidence_first_checks_exact_quote_and_answers_from_selected_complete_block(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("The result rose by 42%.\n\nOnly under condition A.\n", encoding="utf-8")
    prepared = prepare_source(path)
    store = SQLiteDocumentStore(tmp_path / "paper.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    blocks = store.get_blocks(prepared.document.document_id)
    config = GenerationConfig(num_ctx=16384, num_predict=1536)
    model = _Model(config)
    workflow = FullAnswerWorkflow(
        SQLiteDocumentStore(store.database_path, read_only=True),
        prepared.document.document_id,
        _Retriever(prepared.document.document_id, blocks),
        model,
        config,
    )

    result, trace = workflow.ask_evidence_first(
        "What was the result?", before_completion=lambda: 0.1
    )

    assert result.status == "answered"
    assert trace["completion_calls"] == 2
    assert trace["cooldown_seconds"] == [0.1, 0.1]
    assert trace["selection"]["selected_source_ids"] == ["S1"]
    assert result.included_block_ids == (blocks[0].block_id,)
    assert len(model.seen_requests) == 2
    assert issubclass(model.seen_requests[0][0], EvidenceSelectionDraft)


def test_evidence_first_deduplicates_repeated_valid_source_ids(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("The result rose by 42%.\n", encoding="utf-8")
    prepared = prepare_source(path)
    store = SQLiteDocumentStore(tmp_path / "paper.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    blocks = store.get_blocks(prepared.document.document_id)
    config = GenerationConfig(num_ctx=16384, num_predict=1536)
    model = _Model(config)
    model.duplicate_selection_id = True
    workflow = FullAnswerWorkflow(
        SQLiteDocumentStore(store.database_path, read_only=True),
        prepared.document.document_id,
        _Retriever(prepared.document.document_id, blocks),
        model,
        config,
    )

    result, trace = workflow.ask_evidence_first("What was the result?")

    assert result.status == "answered"
    assert trace["selection"]["selected_source_ids"] == ["S1"]
    assert len(result.included_block_ids) == 1


def test_evidence_selection_ids_remain_stable_after_selecting_a_later_block(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("First source.\n\nSecond source.\n", encoding="utf-8")
    prepared = prepare_source(path)
    store = SQLiteDocumentStore(tmp_path / "paper.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    blocks = store.get_blocks(prepared.document.document_id)
    config = GenerationConfig(num_ctx=16384, num_predict=1536)
    model = _Model(config)
    model.select_last = True
    workflow = FullAnswerWorkflow(
        SQLiteDocumentStore(store.database_path, read_only=True),
        prepared.document.document_id,
        _Retriever(prepared.document.document_id, blocks),
        model,
        config,
    )

    result, trace = workflow.ask_evidence_first("What was the result?")

    selected_id = trace["selection"]["selected_source_ids"][0]
    final_evidence_id = model.seen["evidence"][0]["source_id"]
    assert selected_id == "S2"
    assert final_evidence_id == selected_id
    assert result.included_block_ids == (blocks[1].block_id,)


def test_precomputed_search_roundtrip_preserves_private_docling_context_groups(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("First source.\n\nSecond source.\n", encoding="utf-8")
    prepared = prepare_source(path)
    store = SQLiteDocumentStore(tmp_path / "paper.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    blocks = store.get_blocks(prepared.document.document_id)
    question = "What does the paper say?"
    group = SearchContextGroup(
        chunk_index=0,
        text="paper context",
        source_refs=("#/texts/0", "#/texts/1"),
        source_block_ids=tuple(block.block_id for block in blocks),
    )
    search = SearchResult(
        document_id=prepared.document.document_id,
        index_id="docling-index",
        method="hybrid",
        query=question,
        hits=tuple(
            SearchHit(rank=index, score=1 / index, block=block)
            for index, block in enumerate(blocks, 1)
        ),
        elapsed_ms=0.1,
        context_groups=(group,),
    )
    config = GenerationConfig(num_ctx=16384, num_predict=1536)
    model = _Model(config)
    workflow = FullAnswerWorkflow(
        SQLiteDocumentStore(store.database_path, read_only=True),
        prepared.document.document_id,
        _Retriever(prepared.document.document_id, blocks),
        model,
        config,
        WorkflowConfig(top_k=1),
    )

    class _NativeSource:
        def location_for(self, _document_id, _block_id):
            return {"native_ref": "#/texts/0", "pages": [1], "cell": None}

    workflow._native_source = _NativeSource()
    workflow._validate_context_group = lambda _group: None
    result, trace = workflow.ask_evidence_first(
        question, search_result=search, source_blocks=blocks
    )

    assert result.status == "answered"
    assert len(model.seen_requests) == 2
    assert trace["completion_calls"] == 2
    assert result.included_block_ids == tuple(block.block_id for block in blocks)


def test_docling_context_presents_each_exact_source_text_next_to_its_id(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("A result was 42%.\n\nA limitation was noted.\n", encoding="utf-8")
    prepared = prepare_source(path)
    store = SQLiteDocumentStore(tmp_path / "paper.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    blocks = store.get_blocks(prepared.document.document_id)
    config = GenerationConfig(num_ctx=16384, num_predict=1536)
    workflow = FullAnswerWorkflow(
        SQLiteDocumentStore(store.database_path, read_only=True),
        prepared.document.document_id,
        _Retriever(prepared.document.document_id, blocks),
        _Model(config),
        config,
    )

    class _NativeSource:
        def location_for(self, document_id, block_id):
            block = next(item for item in blocks if item.block_id == block_id)
            return {
                "native_ref": "#/texts/0",
                "pages": [1],
                "cell": {
                    "row_start": block.ordinal - 1,
                    "row_end": block.ordinal,
                    "column_start": 0,
                    "column_end": 1,
                },
            }

    workflow._native_source = _NativeSource()
    workflow._validate_context_group = lambda group: None
    groups = tuple(
        SearchContextGroup(
            chunk_index=index,
            text=block.text,
            source_refs=("#/texts/0",),
            source_block_ids=(block.block_id,),
        )
        for index, block in enumerate(blocks)
    )
    messages = workflow._messages_with_context_groups(
        "What does the paper say?", "en", "local", blocks, None, groups
    )
    payload = json.loads(messages[1].content)

    assert "sources" not in payload
    assert len(payload["context_groups"]) == 2
    first = payload["context_groups"][0]["sources"][0]
    assert first["source_id"] == "S1" and first["text"] == blocks[0].text
    assert first["table_row"] == [0, 1] and first["table_column"] == [0, 1]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert serialized.count(blocks[0].text.rstrip("\n")) == 1


def test_evidence_first_fails_closed_on_an_unknown_source_id(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("The result rose by 42%.\n", encoding="utf-8")
    prepared = prepare_source(path)
    store = SQLiteDocumentStore(tmp_path / "paper.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    blocks = store.get_blocks(prepared.document.document_id)
    config = GenerationConfig(num_ctx=16384, num_predict=1536)
    model = _Model(config)
    model.invalid_selection_id = True
    workflow = FullAnswerWorkflow(
        SQLiteDocumentStore(store.database_path, read_only=True),
        prepared.document.document_id,
        _Retriever(prepared.document.document_id, blocks),
        model,
        config,
    )

    try:
        workflow.ask_evidence_first("What was the result?")
    except Exception as exc:
        assert getattr(exc, "trace", {}).get("completion_calls") == 1
        assert len(model.seen_requests) == 1
    else:
        raise AssertionError("a nonverbatim evidence quote must fail closed")


def _invalid_output_error():
    return OutputValidationError(
        "schema_invalid",
        OutputFailureAccounting(
            prompt_eval_count=100,
            eval_count=3,
            elapsed_ms=1.0,
            stop_type="eos",
        ),
        raw_model_output='{"invalid":true}',
        validation_errors=({"type": "missing", "loc": ["answer_claims"]},),
    )


@pytest.mark.parametrize(
    ("failure_stage", "completion_calls"),
    [("selector", 1), ("final_answer", 2)],
)
def test_evidence_first_serializes_pydantic_failure_accounting_and_raw_output(
    tmp_path, failure_stage, completion_calls
):
    path = tmp_path / "paper.txt"
    path.write_text("The result rose by 42%.\n", encoding="utf-8")
    prepared = prepare_source(path)
    store = SQLiteDocumentStore(tmp_path / "paper.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    blocks = store.get_blocks(prepared.document.document_id)
    config = GenerationConfig(num_ctx=16384, num_predict=1536)
    model = _Model(config)
    model.fail_at = "selector" if failure_stage == "selector" else "answer"
    workflow = FullAnswerWorkflow(
        SQLiteDocumentStore(store.database_path, read_only=True),
        prepared.document.document_id,
        _Retriever(prepared.document.document_id, blocks),
        model,
        config,
    )

    with pytest.raises(EvidenceFirstRunFailure) as raised:
        workflow.ask_evidence_first("What was the result?")

    trace = raised.value.trace
    assert trace["failure_stage"] == failure_stage
    assert trace["completion_calls"] == completion_calls
    assert trace["raw_model_output"] == '{"invalid":true}'
    assert trace["validation_errors"] == [{"type": "missing", "loc": ["answer_claims"]}]
    assert trace["failure_accounting"] == {
        "prompt_eval_count": 100,
        "eval_count": 3,
        "elapsed_ms": 1.0,
        "stop_type": "eos",
    }


def test_partial_answer_keeps_supported_direct_answer_and_states_the_gap(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("The result rose by 42%.\n", encoding="utf-8")
    prepared = prepare_source(path)
    store = SQLiteDocumentStore(tmp_path / "paper.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    blocks = store.get_blocks(prepared.document.document_id)
    config = GenerationConfig(num_ctx=16384, num_predict=1536)
    model = _Model(config)
    model.partial = True
    workflow = FullAnswerWorkflow(
        SQLiteDocumentStore(store.database_path, read_only=True),
        prepared.document.document_id,
        _Retriever(prepared.document.document_id, blocks),
        model,
        config,
    )

    result = workflow.ask("What was the result, and under what condition?")

    assert result.status == "partial"
    assert result.stop_reason == "partial"
    assert result.answer == "The result rose by 42%."
    assert "does not state the requested condition" in result.limitations[-1]
    assert result.answer_claims and not result.detail_claims
