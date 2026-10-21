"""Authoritative section mapping and bounded navigation, with synthetic sources only."""

from __future__ import annotations

import hashlib

import pytest

from page_cite_rag.full_answer import FullAnswerWorkflow
from page_cite_rag.generation import GenerationConfig
from page_cite_rag.ingest import prepare_source
from page_cite_rag.qasper import ingest_paper, paper_from_row
from page_cite_rag.source_outline import OUTLINE_KIND, SourceOutline
from page_cite_rag.span_citations import source_spans
from page_cite_rag.store import SQLiteDocumentStore, StoreError
from page_cite_rag.workflows import AnswerWorkflow, WorkflowError
from test_workflows import FakeModel


def _workflow(store, document_id):
    model = FakeModel()
    return AnswerWorkflow(store, document_id, None, model, model.config)


def _paper(heading="Methods"):
    return paper_from_row(
        {
            "id": "synthetic-paper",
            "full_text": {
                "section_name": ["Introduction", heading, "Results"],
                "paragraphs": [
                    ["The question is about samples."],
                    ["Only adults were included.", "Children were excluded."],
                    ["The result was not significant."],
                ],
            },
            "qas": {"question_id": [], "question": [], "answers": []},
        }
    )


def test_qasper_original_section_names_and_boundaries_are_readable(tmp_path):
    store = SQLiteDocumentStore(tmp_path / "source.sqlite3")
    document_id, _ = ingest_paper(store, _paper())
    raw = store.get_retrieval_manifest(document_id, OUTLINE_KIND)
    outline = SourceOutline.model_validate_json(raw)
    assert [section.title for section in outline.sections] == ["Introduction", "Methods", "Results"]
    assert [(section.start_ordinal, section.end_ordinal) for section in outline.sections] == [
        (1, 1),
        (2, 3),
        (4, 4),
    ]
    workflow = _workflow(store, document_id)
    assert workflow.source_outline() == outline
    assert outline.origin == "qasper"
    assert [item.section_id for item in outline.sections] == [
        "section:0000",
        "section:0001",
        "section:0002",
    ]
    section, blocks = workflow.read_outline_section("section:0001")
    assert section.title == "Methods"
    assert [block.ordinal for block in blocks] == [2, 3]
    with pytest.raises(WorkflowError, match="outside the document"):
        workflow.read_outline_section("section:0003")


def test_full_answer_evidence_carries_qasper_section_heading(tmp_path):
    store = SQLiteDocumentStore(tmp_path / "source.sqlite3")
    document_id, _ = ingest_paper(store, _paper())
    blocks = store.get_blocks(document_id)
    workflow = FullAnswerWorkflow(
        store,
        document_id,
        None,
        FakeModel(),
        GenerationConfig(),
    )

    evidence = workflow._evidence("S1", source_spans([blocks[1]])[0], blocks, None)

    assert evidence["section"] == "Methods"


def test_qasper_conflicting_heading_cannot_silently_change_same_document(tmp_path):
    store = SQLiteDocumentStore(tmp_path / "source.sqlite3")
    document_id, _ = ingest_paper(store, _paper())
    before = store.get_retrieval_manifest(document_id, OUTLINE_KIND)
    with pytest.raises(StoreError, match="conflicts"):
        ingest_paper(store, _paper("Different methods"))
    assert store.get_retrieval_manifest(document_id, OUTLINE_KIND) == before


def test_legacy_qasper_without_heading_manifest_stays_explicit_fallback(tmp_path):
    store = SQLiteDocumentStore(tmp_path / "source.sqlite3")
    document_id, _ = ingest_paper(store, _paper())
    with store._connect() as connection:
        connection.execute(
            "DELETE FROM retrieval_indexes WHERE document_id = ? AND kind = ?",
            (document_id, OUTLINE_KIND),
        )
    ingest_paper(store, _paper("Unverifiable replacement"))
    assert store.get_retrieval_manifest(document_id, OUTLINE_KIND) is None
    assert _workflow(store, document_id).source_outline() is None


def test_markdown_atx_outline_skips_fenced_code_and_preserves_preamble(tmp_path):
    path = tmp_path / "source.md"
    path.write_text(
        "Introductory context.\n\n# Methods\n\nOnly adults.\n\n"
        "```md\n\n# Not a section\n\n```\n\n# Results\n\nNo effect.\n",
        encoding="utf-8",
    )
    source = prepare_source(path)
    store = SQLiteDocumentStore(tmp_path / "source.sqlite3")
    store.ingest(source.document, source.iter_units())
    outline = SourceOutline.model_validate_json(
        store.get_retrieval_manifest(source.document.document_id, OUTLINE_KIND)
    )
    assert [section.title for section in outline.sections] == [None, "Methods", "Results"]
    assert [section.section_id for section in outline.sections] == [
        "section:preamble",
        "section:0000",
        "section:0001",
    ]
    assert (
        outline.sections[-1].end_ordinal
        == store.get_document(source.document.document_id).block_count
    )
    assert _workflow(store, source.document.document_id).source_outline() == outline
    assert outline.origin == "markdown_atx"


def test_read_only_old_markdown_without_outline_derives_without_writing(tmp_path):
    path = tmp_path / "old.md"
    path.write_text("# First\n\nA.\n\n# Second\n\nB.", encoding="utf-8")
    source = prepare_source(path)
    writable = SQLiteDocumentStore(tmp_path / "source.sqlite3")
    writable.ingest(source.document, source.iter_units())
    with writable._connect() as connection:
        connection.execute(
            "DELETE FROM retrieval_indexes WHERE document_id = ? AND kind = ?",
            (source.document.document_id, OUTLINE_KIND),
        )
    before = hashlib.sha256(writable.database_path.read_bytes()).hexdigest()
    read_only = SQLiteDocumentStore(writable.database_path, read_only=True)
    outline = _workflow(read_only, source.document.document_id).source_outline()
    assert [section.title for section in outline.sections] == ["First", "Second"]
    assert hashlib.sha256(writable.database_path.read_bytes()).hexdigest() == before


def test_markdown_heading_inside_unsplit_paragraph_falls_back_to_unit(tmp_path):
    path = tmp_path / "ambiguous.md"
    path.write_text("Introductory context.\n# Ambiguous heading\nContinuation.")
    source = prepare_source(path)
    store = SQLiteDocumentStore(tmp_path / "source.sqlite3")
    store.ingest(source.document, source.iter_units())
    assert store.get_retrieval_manifest(source.document.document_id, OUTLINE_KIND) is None
    workflow = _workflow(store, source.document.document_id)
    assert workflow.source_outline() is None
    unit, blocks = workflow.read_source_unit(1)
    assert unit.unit_index == 1 and len(blocks) == 1
