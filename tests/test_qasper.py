from __future__ import annotations

import hashlib
import io

import pytest

from doc_rag import qasper
from doc_rag.qasper import Annotation, Paper, Passage, Question, align_question, ingest_paper
from doc_rag.retrieval import BM25Retriever, build_bm25_index
from doc_rag.store import SQLiteDocumentStore, StoreError


def make_row(paper_id="paper", paragraphs=None, questions=None):
    return {
        "id": paper_id,
        "full_text": {
            "section_name": ["Results"],
            "paragraphs": [paragraphs or ["Alpha samples.", "Beta controls."]],
        },
        "qas": questions
        or {
            "question_id": [paper_id + "-q1", paper_id + "-q2"],
            "question": ["Alpha?", "Beta?"],
            "answers": [
                {"answer": [{"unanswerable": False, "evidence": [text]}]}
                for text in ("Alpha samples.", "Beta controls.")
            ],
        },
    }


def test_whitespace_alignment_retains_original_paragraph_and_coordinates():
    paper = qasper.paper_from_row(make_row(paragraphs=["", "Alpha\n\nsamples.", "Beta controls."]))
    assert paper.passages[0].passage_id == "paper/s0000/p0001"
    aligned = align_question(paper, paper.questions[0])
    assert aligned.gold_passage_ids == ("paper/s0000/p0001",)
    assert aligned.exclusion_reason is None
    assert paper.passages[0].text == "Alpha\n\nsamples."


@pytest.mark.parametrize("change", ["paragraphs", "evidence", "question_lengths"])
def test_malformed_arrow_row_is_not_silently_accepted(change):
    row = make_row()
    if change == "paragraphs":
        row["full_text"]["paragraphs"] = ["not a paragraph list"]
    elif change == "evidence":
        row["qas"]["answers"][0]["answer"][0]["evidence"] = "not an evidence list"
    else:
        row["qas"]["question_id"].pop()
    with pytest.raises(qasper.QasperError, match="schema"):
        qasper.paper_from_row(row)


@pytest.mark.parametrize(
    ("annotations", "reason"),
    [
        ((Annotation(True, ()),), "unanswerable"),
        ((Annotation(False, ()),), "no_evidence"),
        ((Annotation(False, ("FLOAT SELECTED: table",)),), "non_text_evidence"),
        ((Annotation(False, ("missing evidence",)),), "unmatched_evidence"),
        ((), "no_annotations"),
    ],
)
def test_exclusions_are_explicit(annotations, reason):
    paper = qasper.paper_from_row(make_row())
    aligned = align_question(paper, Question("id", "question", annotations))
    assert aligned.exclusion_reason == reason
    assert aligned.gold_passage_ids == ()


def test_duplicates_are_ambiguous_and_partial_annotations_are_not_silently_scored():
    paper = Paper("paper", (Passage("a", 0, 0, "same text"), Passage("b", 0, 1, "same  text")), ())
    assert (
        align_question(
            paper, Question("q", "question", (Annotation(False, ("same text",)),))
        ).exclusion_reason
        == "ambiguous_evidence"
    )
    original = qasper.paper_from_row(make_row())
    partial = Question(
        "q", "question", (Annotation(False, ("Alpha samples.", "FLOAT SELECTED: table")),)
    )
    assert align_question(original, partial).gold_passage_ids == ()
    multiple = Question(
        "q",
        "question",
        (
            Annotation(False, ("Alpha samples.", "Alpha  samples.")),
            Annotation(False, ("Beta controls.",)),
            Annotation(False, ("unmatched",)),
            Annotation(True, ()),
        ),
    )
    aligned = align_question(original, multiple)
    assert aligned.gold_passage_ids == ("paper/s0000/p0000", "paper/s0000/p0001")
    assert aligned.exclusion_reason is None
    assert set(aligned.annotation_issues) == {"unmatched_evidence", "unanswerable"}


def test_evidence_alignment_does_not_fold_case_or_punctuation():
    paper = qasper.paper_from_row(make_row())
    for evidence in ("alpha samples.", "Alpha samples"):
        assert (
            align_question(
                paper, Question("q", "q", (Annotation(False, (evidence,)),))
            ).exclusion_reason
            == "unmatched_evidence"
        )


def test_qasper_store_keeps_paragraph_boundaries_and_does_not_index_questions(tmp_path):
    row = make_row(paragraphs=["Alpha\n\nsamples.", "Beta controls."])
    row["qas"]["question"][0] = "secretquestioncanary"
    paper = qasper.paper_from_row(row)
    store = SQLiteDocumentStore(tmp_path / "qasper.sqlite3")
    document_id, mapping = ingest_paper(store, paper)
    assert len(store.get_blocks(document_id)) == 2
    assert store.get_blocks(document_id)[0].text == "Alpha\n\nsamples."
    unit = store.get_unit(document_id, 1)
    for block in store.get_blocks(document_id):
        assert unit.text[block.char_start : block.char_end] == block.text
        assert block.page_number is None
    build_bm25_index(store, document_id)
    retriever = BM25Retriever(store, document_id)
    assert not retriever.search("secretquestioncanary").hits
    assert mapping[retriever.search("alpha").hits[0].block.block_id] == "paper/s0000/p0000"
    ingest_paper(store, paper)
    assert store.count_documents() == 1


def test_invalid_custom_block_factory_rolls_back(tmp_path):
    from doc_rag.ingest import prepare_source
    from doc_rag.models import Block

    path = tmp_path / "source.txt"
    path.write_text("Actual source.")
    prepared = prepare_source(path)
    store = SQLiteDocumentStore(tmp_path / "docs.sqlite3")

    def invalid(document, unit, first):
        yield Block(
            document_id=document.document_id,
            block_id="0" * 64,
            ordinal=first,
            unit_index=1,
            line_start=1,
            line_end=1,
            char_start=0,
            char_end=5,
            text="WRONG",
        )

    with pytest.raises(StoreError, match="outside"):
        store.ingest(prepared.document, prepared.iter_units(), block_factory=invalid)
    assert store.count_documents() == 0


def test_parquet_reader_uses_local_verified_file(tmp_path, monkeypatch):
    parquet = pytest.importorskip("pyarrow.parquet")
    import pyarrow

    row = make_row()
    snapshot = qasper.snapshot_path(tmp_path, "train")
    snapshot.parent.mkdir(parents=True)
    parquet.write_table(pyarrow.Table.from_pylist([row]), snapshot)
    monkeypatch.setitem(
        qasper.SOURCES,
        "train",
        {
            "filename": snapshot.name,
            "size": snapshot.stat().st_size,
            "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        },
    )
    assert list(qasper.read_snapshot(tmp_path, "train"))[0] == qasper.paper_from_row(row)
    snapshot.write_bytes(b"corrupt")
    with pytest.raises(qasper.QasperError, match="checksum"):
        list(qasper.read_snapshot(tmp_path, "train"))


def test_downloader_is_explicit_validates_and_never_overwrites_corrupt_file(tmp_path, monkeypatch):
    payload = b"synthetic parquet bytes"
    for split in qasper.SOURCES:
        monkeypatch.setitem(
            qasper.SOURCES,
            split,
            {
                "filename": f"{split}.parquet",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            },
        )
    requests = []

    def fake_open(request, timeout):
        requests.append(request.full_url)
        return io.BytesIO(payload)

    monkeypatch.setattr(qasper.urllib.request, "urlopen", fake_open)
    qasper.download_qasper(tmp_path)
    assert len(requests) == 2
    assert all(qasper.REVISION in url for url in requests)
    qasper.download_qasper(tmp_path)
    assert len(requests) == 2
    corrupted = qasper.snapshot_path(tmp_path, "train")
    corrupted.write_bytes(b"user-owned contents")
    with pytest.raises(qasper.QasperError, match="checksum"):
        qasper.download_qasper(tmp_path)
    assert corrupted.read_bytes() == b"user-owned contents"
    assert len(requests) == 2
