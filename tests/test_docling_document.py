"""Formal Docling source persistence without parser weights or real PDF fixtures."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import numpy as np

from page_cite_rag.api import create_app
from page_cite_rag.cli import main
from page_cite_rag.docling_document import _recover_omitted_items, bundle_from_native, load_bundle
from page_cite_rag.docling_retrieval import build_index, load_retriever
from page_cite_rag.full_answer import PROMPT_VERSION, FullAnswerWorkflow
from page_cite_rag.generation import GenerationConfig
from page_cite_rag.service import DocumentService, ServiceConfig
from page_cite_rag.store import SQLiteDocumentStore
from page_cite_rag.workflows import AnswerWorkflow, read_verified_source


def _bundle(*, search_text=None):
    document = {
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "section_header",
                "text": "Results",
                "prov": [{"page_no": 1, "bbox": {"l": 1, "t": 10, "r": 90, "b": 8}}],
            },
            {
                "self_ref": "#/texts/1",
                "label": "text",
                "text": "The increase is 42% only under condition A.",
                "prov": [{"page_no": 2, "bbox": {"l": 1, "t": 8, "r": 90, "b": 3}}],
            },
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "label": "table",
                "prov": [{"page_no": 2, "bbox": {"l": 1, "t": 6, "r": 90, "b": 2}}],
                "data": {
                    "num_rows": 1,
                    "num_cols": 1,
                    "table_cells": [
                        {
                            "text": "42%",
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                        }
                    ],
                },
            }
        ],
    }
    chunks = [
        {
            "chunk": {
                "meta": {
                    "headings": ["Results"],
                    "doc_items": [
                        {"self_ref": "#/texts/0"},
                        {"self_ref": "#/texts/1"},
                        {"self_ref": "#/tables/0"},
                    ],
                }
            },
            "search_text": search_text
            or ("Results. The increase is 42% only under condition A. 42%"),
            "search_tokens": 16,
        }
    ]
    return bundle_from_native(
        document,
        chunks,
        source_sha256="a" * 64,
        source_name="example.pdf",
        page_count=2,
        parser_version="docling-2.137.0;core-2.101.1;hybrid-496;source-v1",
    )


def test_enumerated_source_keeps_legacy_and_new_versions_readable(tmp_path):
    base = _bundle()
    document = json.loads(json.dumps(base.document))
    item = document["texts"][1]
    item.update(label="list_item", enumerated=True, marker="124.", orig="124. " + item["text"])
    versions = ("docling-native-item-or-table/v2", "docling-native-item-or-table/v3")
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")
    bundles = []
    for version in versions:
        chunks = [{**chunk, "profile": version} for chunk in base.chunks]
        bundle = bundle_from_native(
            document,
            chunks,
            source_sha256="a" * 64,
            source_name="example.pdf",
            page_count=2,
            parser_version=version,
        )
        store.ingest(
            bundle.metadata,
            bundle.units(),
            block_factory=bundle.block_factory,
            source_manifests=bundle.source_manifests(),
        )
        bundles.append(load_bundle(store, bundle.metadata.document_id))
    assert bundles[0].metadata.document_id != bundles[1].metadata.document_id
    assert bundles[0].blocks[1].text == item["text"]
    assert bundles[1].blocks[1].text == item["orig"]


def test_native_pdf_round_trip_preserves_source_scope_and_location(tmp_path, capsys):
    bundle = _bundle()
    database = tmp_path / "document.sqlite3"
    store = SQLiteDocumentStore(database)
    result = store.ingest(
        bundle.metadata,
        bundle.units(),
        block_factory=bundle.block_factory,
        source_manifests=bundle.source_manifests(),
    )
    assert result.document.status == "ready"
    assert result.document.page_count == 2
    assert result.document.unit_count == 3
    reopened = SQLiteDocumentStore(database, read_only=True)
    saved = load_bundle(reopened, bundle.metadata.document_id)
    assert saved.outline.sections[0].title == "Results"
    assert saved.coverage["readable_source_units"] == 3
    block = saved.blocks[-1]
    assert read_verified_source(reopened, bundle.metadata.document_id, block.block_id) == block
    location = saved.source.location_for(bundle.metadata.document_id, block.block_id)
    assert location["kind"] == "table"
    assert location["source_text_scope"] == "saved_table_cells_with_coordinates"
    assert location["table"]["cell_page_mapping"] == "unverified"
    assert location["pdf_character_offsets"] == "unavailable"
    workflow = AnswerWorkflow(
        reopened,
        bundle.metadata.document_id,
        None,
        object(),
        SimpleNamespace(num_ctx=4096, num_predict=512, template_reserve=128),
    )
    section, section_blocks = workflow.read_outline_section(saved.outline.sections[0].section_id)
    assert section.title == "Results"
    assert section_blocks[0].text == "Results"
    formal = FullAnswerWorkflow(
        reopened,
        bundle.metadata.document_id,
        None,
        object(),
        SimpleNamespace(num_ctx=4096, num_predict=512, template_reserve=128),
    )
    prompt = formal._messages("What is the value?", "en", "local", (block,), None)
    assert '"table_scope": "whole_table"' in prompt[1].content
    assert '"cell_page_mapping": "unverified"' in prompt[1].content

    assert main(["show", bundle.metadata.document_id, "--db", str(database)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["sections"][0]["title"] == "Results"
    assert report["tables"] == ["#/tables/0"]
    assert (
        main(
            [
                "source",
                bundle.metadata.document_id,
                block.block_id,
                "--db",
                str(database),
                "--format",
                "json",
            ]
        )
        == 0
    )
    citation = json.loads(capsys.readouterr().out)
    assert "42%" in citation["text"]
    assert citation["native_location"]["native_ref"] == "#/tables/0"


class _Encoder:
    identity = {"name": "synthetic", "precision": "test"}
    dimension = 2

    def __init__(self):
        self.document_calls = 0

    def encode_documents(self, texts):
        self.document_calls += 1
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_queries(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_chunk_index_reloads_without_reembedding_documents(tmp_path):
    bundle = _bundle()
    database = tmp_path / "document.sqlite3"
    store = SQLiteDocumentStore(database)
    store.ingest(
        bundle.metadata,
        bundle.units(),
        block_factory=bundle.block_factory,
        source_manifests=bundle.source_manifests(),
    )
    encoder = _Encoder()
    report = build_index(store, bundle.metadata.document_id, encoder)
    assert report["duplicate"] is False
    assert encoder.document_calls == 1

    reopened = SQLiteDocumentStore(database, read_only=True)
    retriever = load_retriever(reopened, bundle.metadata.document_id, encoder)
    result = retriever.search("condition A", top_k=3)
    assert result.method == "hybrid"
    assert result.document_id == bundle.metadata.document_id
    assert retriever.last_query_plan["profile"]["candidates_per_branch"] == 50
    assert retriever.last_query_plan["profile"]["rrf_constant"] == 60
    assert {hit.block.text for hit in result.hits} == {
        "Results",
        "The increase is 42% only under condition A.",
        bundle.blocks[-1].text,
    }
    assert result.context_groups[0].text == bundle.chunks[0]["search_text"]
    assert result.context_groups[0].source_refs == ("#/texts/0", "#/texts/1", "#/tables/0")
    assert encoder.document_calls == 1

    from test_full_answer import _Model

    config = GenerationConfig(num_ctx=16384, num_predict=1536)
    model = _Model(config)
    workflow = FullAnswerWorkflow(
        SQLiteDocumentStore(database, read_only=True),
        bundle.metadata.document_id,
        retriever,
        model,
        config,
    )
    selected, _, messages, budget = workflow._context(
        "What increased?",
        "en",
        "local",
        workflow._context_blocks(result, tuple(hit.block for hit in result.hits)),
        None,
        context_groups=result.context_groups,
    )
    payload = json.loads(messages[1].content)
    assert len(selected) == 3
    context_group = payload["context_groups"][0]
    assert context_group["text"] == result.context_groups[0].text
    sources = {source["source_id"]: source for source in context_group["sources"]}
    assert set(sources) == {"S1", "S2", "S3"}
    assert sources["S2"]["text"] == "The increase is 42% only under condition A."
    assert "evidence" not in payload
    assert messages[1].content.count("The increase is 42% only under condition A.") == 2
    assert "42%" in sources["S3"]["text"]
    assert sources["S3"]["native_ref"] == "#/tables/0"
    assert sources["S3"]["table_scope"] == "whole_table"
    assert sources["S3"]["table_dimensions"] == [1, 1]
    assert sources["S3"]["cell_page_mapping"] == "unverified"
    assert budget.input_upper_estimate > 0

    result_answer = workflow.ask("What increased?", language="en")
    assert result_answer.included_block_ids == tuple(block.block_id for block in selected)
    expected_messages = [message.model_dump(mode="json") for message in messages]
    assert model.seen_request["messages"] == expected_messages
    expected_schema = workflow._response_model_for(selected).model_json_schema()
    assert model.seen_request["schema"] == expected_schema
    assert model.seen_request["budget"] == budget
    assert result_answer.budget == budget

    partial = workflow._messages_with_context_groups(
        "What increased?", "en", "local", selected[:1], None, result.context_groups
    )
    assert "context_groups" not in json.loads(partial[1].content)


def test_context_budget_admits_or_omits_a_docling_chunk_as_a_whole(tmp_path):
    text = "Results. The increase is 42% only under condition A. 42% " + "detail " * 1200
    bundle = _bundle(search_text=text)
    database = tmp_path / "large-context.sqlite3"
    store = SQLiteDocumentStore(database)
    store.ingest(
        bundle.metadata,
        bundle.units(),
        block_factory=bundle.block_factory,
        source_manifests=bundle.source_manifests(),
    )
    encoder = _Encoder()
    build_index(store, bundle.metadata.document_id, encoder)
    reopened = SQLiteDocumentStore(database, read_only=True)
    retriever = load_retriever(reopened, bundle.metadata.document_id, encoder)
    result = retriever.search("condition A", top_k=1)
    assert len(result.context_groups) == 1
    assert len(result.hits) == 3

    from test_full_answer import _Model

    config = GenerationConfig(num_ctx=4096, num_predict=512, template_reserve=256)
    workflow = FullAnswerWorkflow(
        reopened,
        bundle.metadata.document_id,
        retriever,
        _Model(config),
        config,
    )
    source_blocks = workflow._context_blocks(result, tuple(hit.block for hit in result.hits))
    from page_cite_rag.generation import check_budget

    empty_budget = check_budget(
        workflow._messages("What increased?", "en", "local", (), None),
        workflow._response_model_for(()).model_json_schema(),
        config.model_copy(update={"num_ctx": 32768}),
    )
    workflow._budget_config = config.model_copy(
        update={"num_ctx": empty_budget.input_upper_estimate + config.num_predict}
    )
    selected, omitted, messages, _ = workflow._context(
        "What increased?",
        "en",
        "local",
        source_blocks,
        None,
        context_groups=result.context_groups,
    )
    assert selected == ()
    assert set(omitted) == set(result.context_groups[0].source_block_ids)
    assert "context_groups" not in json.loads(messages[1].content)


def test_chunker_omission_is_recovered_as_search_only_source():
    document = {
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "section_header",
                "text": "Important condition",
                "prov": [{"page_no": 1, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}],
            }
        ]
    }

    class Tokenizer:
        def encode(self, value, *, add_special_tokens):
            assert not add_special_tokens
            return value.split()

        def decode(self, tokens):
            return " ".join(tokens)

    chunks = []
    assert _recover_omitted_items(document, chunks, page_count=1, tokenizer=Tokenizer()) == 1
    assert chunks[0]["origin"] == "native-item-recovery-v2"
    assert chunks[0]["chunk"]["meta"]["doc_items"] == [{"self_ref": "#/texts/0"}]
    bundle = bundle_from_native(
        document,
        chunks,
        source_sha256="a" * 64,
        source_name="paper.pdf",
        page_count=1,
        parser_version="test-parser",
    )
    assert bundle.blocks[0].text == "Important condition"
    assert bundle.coverage["recovered_search_windows"] == 1


def test_docling_body_order_controls_source_and_chunk_order():
    bundle = _bundle()
    document = dict(bundle.document)
    document["body"] = {
        "children": [
            {"$ref": "#/texts/1"},
            {"$ref": "#/tables/0"},
            {"$ref": "#/texts/0"},
        ]
    }
    chunks = [
        {
            "chunk": {"meta": {"doc_items": [{"self_ref": ref}], "headings": []}},
            "search_text": search_text,
        }
        for ref, search_text in (
            ("#/tables/0", "Metric 42%"),
            ("#/texts/0", "Results"),
            ("#/texts/1", "The increase is 42% only under condition A."),
        )
    ]
    reordered = bundle_from_native(
        document,
        chunks,
        source_sha256="b" * 64,
        source_name="ordered.pdf",
        page_count=2,
        parser_version="test-parser",
    )

    assert [block.text for block in reordered.blocks] == [
        "The increase is 42% only under condition A.",
        bundle.blocks[-1].text,
        "Results",
    ]
    assert reordered.chunks[0]["search_text"] == "The increase is 42% only under condition A."


def test_recovered_search_windows_retain_body_heading_path():
    document = {
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "section_header",
                "level": 1,
                "text": "Important condition",
                "prov": [{"page_no": 1, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}],
            },
            {
                "self_ref": "#/texts/1",
                "label": "text",
                "text": "The result is valid only under condition A.",
                "prov": [{"page_no": 1, "bbox": {"l": 1, "t": 5, "r": 3, "b": 4}}],
            },
        ],
        "body": {"children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}]},
    }

    class Tokenizer:
        def encode(self, value, *, add_special_tokens):
            assert not add_special_tokens
            return value.split()

        def decode(self, tokens):
            return " ".join(tokens)

    chunks = []
    assert _recover_omitted_items(document, chunks, page_count=1, tokenizer=Tokenizer()) == 2
    recovered_body = chunks[1]
    assert recovered_body["chunk"]["meta"]["headings"] == ["Important condition"]
    assert "Important condition" in recovered_body["search_text"]
    assert "condition A" in recovered_body["search_text"]


def test_recovered_table_keeps_headers_and_complete_native_rows():
    document = {
        "tables": [
            {
                "self_ref": "#/tables/0",
                "label": "table",
                "prov": [{"page_no": 1, "bbox": {"l": 1, "t": 8, "r": 9, "b": 2}}],
                "data": {
                    "num_rows": 3,
                    "num_cols": 2,
                    "table_cells": [
                        {
                            "text": "Method",
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                            "column_header": True,
                        },
                        {
                            "text": "Score",
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 1,
                            "end_col_offset_idx": 2,
                            "column_header": True,
                        },
                        {
                            "text": "Baseline",
                            "start_row_offset_idx": 1,
                            "end_row_offset_idx": 2,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                            "row_header": True,
                        },
                        {
                            "text": "0.71",
                            "start_row_offset_idx": 1,
                            "end_row_offset_idx": 2,
                            "start_col_offset_idx": 1,
                            "end_col_offset_idx": 2,
                        },
                        {
                            "text": "Ours",
                            "start_row_offset_idx": 2,
                            "end_row_offset_idx": 3,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                            "row_header": True,
                        },
                        {
                            "text": "0.83",
                            "start_row_offset_idx": 2,
                            "end_row_offset_idx": 3,
                            "start_col_offset_idx": 1,
                            "end_col_offset_idx": 2,
                        },
                    ],
                },
            }
        ],
        "body": {"children": [{"$ref": "#/tables/0"}]},
    }

    class Tokenizer:
        def encode(self, value, *, add_special_tokens):
            assert not add_special_tokens
            return value.split()

        def decode(self, tokens):
            return " ".join(tokens)

    chunks = []
    assert _recover_omitted_items(document, chunks, page_count=1, tokenizer=Tokenizer()) == 1
    text = chunks[0]["search_text"]
    assert "Column header" in text
    assert "Method" in text and "Score" in text
    assert "Baseline" in text and "0.71" in text
    assert "Ours" in text and "0.83" in text


def test_page_furniture_is_excluded_from_recovery_and_source_blocks():
    document = {
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "section_header",
                "text": "Results",
                "prov": [{"page_no": 1, "bbox": {"l": 1, "t": 8, "r": 9, "b": 6}}],
            },
            {
                "self_ref": "#/texts/1",
                "label": "page_footer",
                "text": "Confidential footer text",
                "prov": [{"page_no": 1, "bbox": {"l": 1, "t": 2, "r": 9, "b": 1}}],
            },
        ],
        "body": {"children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}]},
    }

    class Tokenizer:
        def encode(self, value, *, add_special_tokens):
            assert not add_special_tokens
            return value.split()

        def decode(self, tokens):
            return " ".join(tokens)

    chunks = []
    assert _recover_omitted_items(document, chunks, page_count=1, tokenizer=Tokenizer()) == 1
    bundle = bundle_from_native(
        document,
        chunks,
        source_sha256="c" * 64,
        source_name="furniture.pdf",
        page_count=1,
        parser_version="test-parser",
    )
    assert [block.text for block in bundle.blocks] == ["Results"]
    assert bundle.coverage["excluded_native_refs"] == [
        {"ref": "#/texts/1", "reason": "docling_page_furniture"}
    ]


def test_docling_staged_api_uses_formal_answer_and_native_source(tmp_path):
    from test_full_answer import _Model

    bundle = _bundle()
    database = tmp_path / "document.sqlite3"
    store = SQLiteDocumentStore(database)
    store.ingest(
        bundle.metadata,
        bundle.units(),
        block_factory=bundle.block_factory,
        source_manifests=bundle.source_manifests(),
    )
    encoder = _Encoder()
    build_index(store, bundle.metadata.document_id, encoder)
    read_only = SQLiteDocumentStore(database, read_only=True)
    retriever = load_retriever(read_only, bundle.metadata.document_id, encoder)
    config = GenerationConfig(num_ctx=16384, num_predict=1536)
    service = DocumentService(
        ServiceConfig(
            database_path=database,
            document_id=bundle.metadata.document_id,
            method="hybrid",
            top_k=3,
        ),
        config,
        lambda: _Model(config),
        staged_search=lambda query, limit: retriever.search(query, top_k=limit),
    )
    app = create_app(service, api_token="test-local-token", allow_test_client=True)

    async def check():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1",
                headers={"Authorization": "Bearer test-local-token"},
            ) as client:
                answer = await client.post("/v1/ask", json={"question": "What condition?"})
                assert answer.status_code == 200
                assert answer.json()["result"]["prompt_version"] == PROMPT_VERSION
                source = await client.get(f"/v1/source/{bundle.blocks[-1].block_id}")
                assert source.status_code == 200
                assert source.json()["native_location"]["native_ref"] == "#/tables/0"

    asyncio.run(check())
