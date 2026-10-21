"""Compare RAG retrieval context with complete source context on frozen cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unicodedata
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from page_cite_rag.answer_metrics import (
    AnswerPrediction,
    evaluate_answers,
    read_answer_references,
    score_prediction,
)
from page_cite_rag.application import FrozenSearchRetriever, document_retriever, document_workflow
from page_cite_rag.docling_document import DEFAULT_ARTIFACTS, convert_pdf, load_bundle
from page_cite_rag.docling_retrieval import build_index as build_docling_index
from page_cite_rag.embedding import DEFAULT_MODEL_DIR
from page_cite_rag.embedding_native import NativeEmbeddingEncoder
from page_cite_rag.embedding_server import DEFAULT_GGUF_PATH, NativeEmbeddingServer
from page_cite_rag.evidence_metrics import align_references, reference_metrics
from page_cite_rag.full_answer import EVIDENCE_SELECTION_PROMPT_VERSION
from page_cite_rag.generation import ContextBudgetError, budget_for_model
from page_cite_rag.llama import DEFAULT_BUNDLE, LlamaConfig, LlamaModelClient
from page_cite_rag.models import Block, DocumentMetadata, ParsedUnit
from page_cite_rag.qasper import align_question, ingest_paper, read_snapshot
from page_cite_rag.reranker_server import DEFAULT_MODEL_PATH as DEFAULT_RERANKER_PATH
from page_cite_rag.reranker_server import MODEL_SHA256 as RERANKER_SHA256
from page_cite_rag.reranker_server import NativeRerankerServer
from page_cite_rag.retrieval import SearchContextGroup, SearchHit, SearchResult, build_bm25_index
from page_cite_rag.store import SOURCE_LOCATIONS_KIND, SQLiteDocumentStore

ROOT = Path(__file__).resolve().parents[1]
P0_PATH = ROOT / "outputs/reference-baseline/p0-cases.json"
XRAG_PATH = ROOT / "data/xrag/ead86612ac265e578ffee6f838c2180fec6428d9/xrag.dev.jsonl"
XRAG_SHA256 = "7c7977b93ce205069b91fc701a4a5df10ec30055945869804589403bcd432a2c"
DRCD_SHA256 = "e236df03861ba241bd7b29628cd48b3b8e339c60137e041a4358b5320ed61928"
GPU_UUID = os.environ.get("PAGE_CITE_RAG_GPU_UUID", "")
Q3_MANIFEST_PATH = ROOT / "outputs/quality-convergence/manifests/q3-final-20261018.json"
Q4_MANIFEST_PATH = ROOT / "outputs/quality-convergence/manifests/q4-agent-20261018.json"
Q5_MANIFEST_PATH = ROOT / "outputs/quality-convergence/manifests/q5-evidence-first-20261018.json"
Q6_MANIFEST_PATH = ROOT / "outputs/quality-convergence/manifests/q6-a-b-20261019.json"
Q6_CONFIRMATION_MANIFEST_PATH = (
    ROOT / "outputs/quality-convergence/manifests/q6-a1-confirmation-20261019.json"
)
Q3_MANIFEST_SHA256 = "2c1dd858cc5b08c97c61d21f2f03fc514725ed356055cc3097d40fd6b0f0e8cf"
Q4_MANIFEST_SHA256 = "2b3e63fefa83e628b5a79d122e83904b74a0524290be4b843719f045f1bfce98"
Q5_MANIFEST_SHA256 = "c8430fcacec18f3fb104d9c9dde2b55d018700bd6edd051102b9587d48078484"
Q6_MANIFEST_SHA256 = "c06382ce03397fd5a77a7f9f1a46324779b2499eb6acd47a74c633034b0979fb"
Q6_CONFIRMATION_MANIFEST_SHA256 = "f846d4ea1444bd5d46520b37c05f3413692ac4bcab4daf5fb449471b72808bbe"
DEFAULT_FINAL_K = 10
RERANK_CANDIDATE_K = 20
INPUT_LIMIT = 10_000
OUTPUT_TOKENS = 1_536
CONTEXT_TOKENS = 16_384
TEMPLATE_RESERVE = 1_024
MAX_WALL_SECONDS = 50 * 60
MAX_GENERATION_SECONDS = 40 * 60
MAX_OUTPUT_BYTES = 100 * 1024 * 1024
COOL_TARGET_C = 60
COOL_WAIT_SECONDS = 180


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def file_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_p0() -> dict:
    expected = "2665144bcf7bd1d6a59b31e83fce2de8669590a9635d4d5cda3ad567137214d3"
    if sha256(P0_PATH) != expected:
        raise ValueError("the frozen P0 manifest checksum changed")
    manifest = json.loads(P0_PATH.read_text(encoding="utf-8"))
    if manifest.get("case_count") != 20 or len(manifest.get("cases", [])) != 20:
        raise ValueError("the frozen P0 case count changed")
    if sha256(XRAG_PATH) != XRAG_SHA256:
        raise ValueError("the pinned XRAG input checksum changed")
    for item in manifest["pdfs"].values():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("a frozen public PDF checksum changed")
    return manifest


def read_manifest(path: Path | None = None) -> dict:
    """Read a frozen evaluation manifest and verify its lineage and source pins."""
    if path is None:
        return read_p0()
    if path.resolve() == P0_PATH.resolve():
        return read_p0()
    if path.is_symlink() or not path.is_file():
        raise ValueError("the frozen evaluation manifest is missing or is a symlink")
    frozen_hashes = {
        Q3_MANIFEST_PATH.resolve(): Q3_MANIFEST_SHA256,
        Q4_MANIFEST_PATH.resolve(): Q4_MANIFEST_SHA256,
        Q5_MANIFEST_PATH.resolve(): Q5_MANIFEST_SHA256,
        Q6_MANIFEST_PATH.resolve(): Q6_MANIFEST_SHA256,
        Q6_CONFIRMATION_MANIFEST_PATH.resolve(): Q6_CONFIRMATION_MANIFEST_SHA256,
    }
    expected_hash = frozen_hashes.get(path.resolve())
    if expected_hash is None or sha256(path) != expected_hash:
        raise ValueError("the frozen evaluation manifest checksum changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    version = manifest.get("version")
    if version not in (
        "quality-convergence-q3/v1",
        "quality-convergence-q4/v1",
        "quality-convergence-q5/v1",
        "quality-convergence-q6/v1",
        "quality-convergence-q6-confirmation/v1",
    ):
        raise ValueError("unsupported frozen evaluation manifest")
    if manifest.get("case_count") != len(manifest.get("cases", [])):
        raise ValueError("the frozen evaluation case count changed")
    if len({case.get("id") for case in manifest["cases"]}) != len(manifest["cases"]):
        raise ValueError("the frozen evaluation manifest contains duplicate case IDs")
    if version == "quality-convergence-q3/v1":
        if manifest.get("arms") != ["A"]:
            raise ValueError("Q3 must use the single frozen product configuration")
    elif version == "quality-convergence-q4/v1" and manifest.get("arms") != ["fixed", "agent"]:
        raise ValueError("Q4 must compare the fixed and bounded-agent arms")
    elif version == "quality-convergence-q5/v1" and manifest.get("arms") != ["A", "B"]:
        raise ValueError("Q5 must compare the fixed and evidence-first answer arms")
    elif version == "quality-convergence-q6/v1" and manifest.get("arms") != ["A1", "B1"]:
        raise ValueError("Q6 must compare the shared direct-answer and selector arms")
    if manifest.get("qasper", {}).get("split") != "validation":
        raise ValueError("the frozen evaluation must use the pinned QASPER validation split")
    if (
        manifest.get("qasper", {}).get("sha256")
        != "089781b91c337d348dd9e8b57cc8adc100ed2d9cab84a6127402bcccf1559222"
    ):
        raise ValueError("the frozen QASPER validation checksum changed")
    if version == "quality-convergence-q3/v1":
        if sha256(XRAG_PATH) != manifest.get("xrag", {}).get("sha256"):
            raise ValueError("the pinned XRAG input checksum changed")
        drcd_path = ROOT / manifest.get("drcd", {}).get("path", "")
        if drcd_path.is_symlink() or not drcd_path.is_file() or sha256(drcd_path) != DRCD_SHA256:
            raise ValueError("the pinned DRCD dev split is missing or has changed")
    elif version == "quality-convergence-q4/v1":
        if len(manifest["cases"]) != 8 or any(
            case.get("kind") != "qasper" for case in manifest["cases"]
        ):
            raise ValueError("Q4 must contain exactly eight frozen QASPER cases")
        if sha256(Q3_MANIFEST_PATH) != Q3_MANIFEST_SHA256:
            raise ValueError("the frozen Q3 source manifest changed")
        q3 = json.loads(Q3_MANIFEST_PATH.read_text(encoding="utf-8"))
        q3_answerable = [case for case in q3["cases"] if case["kind"] == "qasper"][:12]
        expected_ids = [case["id"] for case in q3_answerable[:8]]
        if [case["id"] for case in manifest["cases"]] != expected_ids:
            raise ValueError("Q4 cases no longer match the first eight frozen Q3 answerable cases")
    elif version == "quality-convergence-q5/v1":
        if manifest.get("phase") != "Q5" or manifest.get("arms") != ["A", "B"]:
            raise ValueError("Q5 must compare the fixed and evidence-first answer arms")
        if len(manifest["cases"]) != 44:
            raise ValueError("Q5 must contain the frozen 44-case evaluation slice")
        if sha256(Q3_MANIFEST_PATH) != Q3_MANIFEST_SHA256:
            raise ValueError("the frozen Q3 source manifest changed")
        if manifest.get("parent_q3_sha256") != Q3_MANIFEST_SHA256:
            raise ValueError("Q5 does not reference the frozen Q3 manifest")
        if manifest.get("parent_p0_sha256") != sha256(P0_PATH):
            raise ValueError("Q5 does not reference the frozen P0 PDF cases")
        q3 = json.loads(Q3_MANIFEST_PATH.read_text(encoding="utf-8"))
        p0 = read_p0()
        q3_cases = q3["cases"]
        p0_pdf_cases = [case for case in p0["cases"] if case["kind"] == "pdf"]
        if manifest["cases"] != [*q3_cases, *p0_pdf_cases]:
            raise ValueError("Q5 case IDs or order differ from the frozen Q3 and P0 cases")
        if (
            manifest.get("qasper") != q3.get("qasper")
            or manifest.get("xrag") != q3.get("xrag")
            or manifest.get("drcd") != q3.get("drcd")
        ):
            raise ValueError("Q5 public benchmark pins differ from Q3")
        expected_pdf_keys = {case["document"] for case in p0_pdf_cases}
        if set(manifest.get("pdfs", {})) != expected_pdf_keys:
            raise ValueError("Q5 PDF assets differ from the selected P0 PDFs")
        if any(manifest["pdfs"][key] != p0["pdfs"][key] for key in expected_pdf_keys):
            raise ValueError("Q5 PDF asset pins differ from P0")
    elif version == "quality-convergence-q6/v1":
        if manifest.get("phase") != "Q6" or len(manifest["cases"]) != 44:
            raise ValueError("Q6 must contain the frozen 44-case development slice")
        if sha256(Q5_MANIFEST_PATH) != Q5_MANIFEST_SHA256:
            raise ValueError("the frozen Q5 source manifest changed")
        q5 = json.loads(Q5_MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("parent_q5_sha256") != Q5_MANIFEST_SHA256:
            raise ValueError("Q6 does not reference the frozen Q5 manifest")
        if manifest["cases"] != q5["cases"]:
            raise ValueError("Q6 case IDs or order differ from frozen Q5")
        if (
            manifest.get("qasper") != q5.get("qasper")
            or manifest.get("xrag") != q5.get("xrag")
            or manifest.get("drcd") != q5.get("drcd")
            or manifest.get("pdfs") != q5.get("pdfs")
        ):
            raise ValueError("Q6 source pins differ from frozen Q5")
        if manifest.get("smoke_case_ids") != [
            "pdf-nist-core",
            "pdf-nist-measure",
            "pdf-sbert-objectives",
            "pdf-lost-manipulation",
            "xrag-zh-en-13",
            "qasper-noanswer-206739417251064b910ae9e5ff096e867ee10fb8",
        ]:
            raise ValueError("Q6 smoke cases differ from the frozen plan")
    else:
        if manifest.get("phase") != "Q6-confirmation" or manifest.get("arms") != ["A1"]:
            raise ValueError("Q6 confirmation must validate the A1-only frozen candidate")
        if len(manifest["cases"]) != 8 or any(
            case.get("kind") != "qasper" for case in manifest["cases"]
        ):
            raise ValueError("Q6 confirmation must contain exactly eight QASPER cases")
        if sha256(Q5_MANIFEST_PATH) != Q5_MANIFEST_SHA256:
            raise ValueError("the frozen Q5 source manifest changed")
        if sha256(Q6_MANIFEST_PATH) != Q6_MANIFEST_SHA256:
            raise ValueError("the frozen Q6 source manifest changed")
        if (
            manifest.get("parent_q5_sha256") != Q5_MANIFEST_SHA256
            or manifest.get("parent_q6_sha256") != Q6_MANIFEST_SHA256
        ):
            raise ValueError("Q6 confirmation lineage differs from the frozen Q5/Q6 manifests")
        q5 = json.loads(Q5_MANIFEST_PATH.read_text(encoding="utf-8"))
        q6 = json.loads(Q6_MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("qasper") != q6.get("qasper"):
            raise ValueError("Q6 confirmation QASPER source pin differs from Q6")
        excluded_papers = {case["paper_id"] for case in q5["cases"] if case.get("kind") == "qasper"}
        expected_cases = q6_confirmation_case_specs(
            read_snapshot(ROOT / "data/qasper", "validation"), excluded_papers
        )
        if manifest.get("cases") != expected_cases:
            raise ValueError("Q6 confirmation case selection differs from its frozen algorithm")
    return manifest


def q6_confirmation_case_specs(papers, excluded_paper_ids: set[str]) -> list[dict]:
    """Select one deterministically ranked, unused validation question per paper."""
    prefix = "page-cite-rag-q6-a1-confirmation-v1:"
    selected = {"answerable": [], "noanswer": []}
    ordered_papers = sorted(
        (paper for paper in papers if paper.paper_id not in excluded_paper_ids),
        key=lambda paper: hashlib.sha256(f"{prefix}{paper.paper_id}".encode()).hexdigest(),
    )
    for paper in ordered_papers:
        questions = sorted(
            paper.questions,
            key=lambda question: hashlib.sha256(
                f"{prefix}{question.question_id}".encode()
            ).hexdigest(),
        )
        for question in questions:
            labels = {annotation.unanswerable for annotation in question.annotations}
            group = "answerable" if labels == {False} else "noanswer" if labels == {True} else None
            limit = 6 if group == "answerable" else 2
            if group is None or len(selected[group]) >= limit:
                continue
            if group == "answerable" and align_question(paper, question).exclusion_reason:
                continue
            selected[group].append(
                {
                    "id": f"qasper-{question.question_id}",
                    "kind": "qasper",
                    "paper_id": paper.paper_id,
                    "question_id": question.question_id,
                    "language": "en",
                }
            )
            break
        if len(selected["answerable"]) == 6 and len(selected["noanswer"]) == 2:
            break
    cases = selected["answerable"] + selected["noanswer"]
    if len(cases) != 8 or len({case["paper_id"] for case in cases}) != 8:
        raise ValueError("QASPER snapshot cannot provide the frozen Q6 confirmation sample")
    return cases


@dataclass
class Record:
    record_id: str
    parent_id: str
    text: str
    page_numbers: tuple[int, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass
class Case:
    case_id: str
    kind: str
    question: str
    language: str
    document_id: str
    store: SQLiteDocumentStore
    parent_by_block: dict[str, str] = field(default_factory=dict)
    parent_text: dict[str, str] = field(default_factory=dict)
    pages_by_block: dict[str, tuple[int, ...]] = field(default_factory=dict)
    locations_by_block: dict[str, dict] = field(default_factory=dict)
    paper_id: str | None = None
    question_id: str | None = None
    gold_ids: tuple[str, ...] = ()
    qasper_question: object | None = None
    qasper_paper: object | None = None
    a_result: SearchResult | None = None
    candidate_result: SearchResult | None = None
    r_result: SearchResult | None = None
    source_blocks: tuple[Block, ...] = ()
    source_context_groups: tuple[SearchContextGroup, ...] = ()


def ingest_records(
    store: SQLiteDocumentStore,
    source_name: str,
    records: list[Record],
    *,
    parser: str,
    media_type: str = "text/plain",
    parser_version: str = "source-records-v1",
    page_count: int | None = None,
) -> tuple[str, dict[str, str]]:
    if not records or any(not record.text.strip() for record in records):
        raise ValueError("source records must contain nonempty text")
    payload = [
        (record.record_id, record.parent_id, record.text, record.page_numbers, record.metadata)
        for record in records
    ]
    source_sha = file_digest(payload)
    document_id = hashlib.sha256(f"{parser_version}\0{source_sha}".encode()).hexdigest()
    document = DocumentMetadata(
        document_id=document_id,
        source_sha256=source_sha,
        source_name=source_name,
        media_type=media_type,
        parser_name=parser,
        parser_version=parser_version,
        usage_scope="public benchmark data; local quality evaluation",
        page_count=page_count,
    )
    source_parts = []
    ranges = []
    cursor = 0
    for index, record in enumerate(records):
        if index:
            source_parts.append("\n\n")
            cursor += 2
        start = cursor
        source_parts.append(record.text)
        cursor += len(record.text)
        ranges.append((start, cursor))
    source_text = "".join(source_parts)
    line_starts = [0]
    for line in source_text.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    blocks = []
    record_to_block = {}
    for ordinal, (record, (start, end)) in enumerate(zip(records, ranges, strict=True), 1):
        block_id = hashlib.sha256(
            f"{document_id}\0{record.record_id}\0{ordinal}".encode()
        ).hexdigest()
        blocks.append(
            Block(
                document_id=document_id,
                block_id=block_id,
                ordinal=ordinal,
                unit_index=1,
                page_number=None,
                line_start=bisect_right(line_starts, start),
                line_end=bisect_right(line_starts, end - 1),
                char_start=start,
                char_end=end,
                text=record.text,
            )
        )
        record_to_block[record.record_id] = block_id
    unit = ParsedUnit(unit_index=1, page_number=None, text=source_text, status="ok")

    def block_factory(_metadata: DocumentMetadata, unit: ParsedUnit, first: int):
        if first != 1 or unit.unit_index != 1 or unit.text != source_text:
            raise ValueError("source record and unit differ")
        yield from blocks

    locations = {
        record_to_block[record.record_id]: {
            "record_id": record.record_id,
            "parent_id": record.parent_id,
            "pages": list(record.page_numbers),
            "metadata": record.metadata,
        }
        for record in records
    }
    manifests = {
        SOURCE_LOCATIONS_KIND: json.dumps(
            {
                "version": 1,
                "document_id": document_id,
                "source_sha256": source_sha,
                "locations": locations,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    }
    store.ingest(document, [unit], block_factory=block_factory, source_manifests=manifests)
    return document_id, {record_to_block[record.record_id]: record.parent_id for record in records}


def load_xrag_rows(manifest: dict) -> dict[tuple[str, str, int], dict]:
    wanted = {
        (item["task"], item.get("dataset_language", "zh"), item["example_id"])
        for item in manifest["cases"]
        if item["kind"] == "xrag"
    }
    result = {}
    with XRAG_PATH.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            key = (row.get("task"), row.get("language"), row.get("id"))
            if key in wanted:
                if key in result:
                    raise ValueError("duplicate XRAG row in the pinned task/language slice")
                result[key] = row
    if set(result) != wanted:
        raise ValueError("one or more frozen XRAG questions are missing")
    return result


def prepare_inputs(manifest: dict, store: SQLiteDocumentStore) -> dict[str, Case]:
    case_specs = {item["id"]: item for item in manifest["cases"]}
    qasper_specs = [item for item in manifest["cases"] if item["kind"] == "qasper"]
    paper_ids = {item["paper_id"] for item in qasper_specs}
    papers = {
        paper.paper_id: paper
        for paper in read_snapshot(
            ROOT / "data/qasper", manifest.get("qasper", {}).get("split", "train")
        )
        if paper.paper_id in paper_ids
    }
    if set(papers) != paper_ids:
        raise ValueError("one or more frozen QASPER papers are missing")

    prepared = {}
    qasper_sources = {}
    for paper_id, paper in papers.items():
        document_id, block_to_passage = ingest_paper(store, paper)
        passage_text = {passage.passage_id: passage.text for passage in paper.passages}
        qasper_sources[paper_id] = (document_id, block_to_passage, passage_text)
    for item in qasper_specs:
        paper = papers[item["paper_id"]]
        question = next(q for q in paper.questions if q.question_id == item["question_id"])
        unanswerable = bool(question.annotations) and all(
            annotation.unanswerable for annotation in question.annotations
        )
        if align_question(paper, question).exclusion_reason is not None and not unanswerable:
            raise ValueError(f"frozen QASPER case is not evidence-aligned: {item['id']}")
        document_id, block_map, passage_text = qasper_sources[paper.paper_id]
        prepared[item["id"]] = Case(
            case_id=item["id"],
            kind="qasper",
            question=question.text,
            language="en",
            document_id=document_id,
            store=store,
            parent_by_block=block_map,
            parent_text=passage_text,
            paper_id=paper.paper_id,
            question_id=question.question_id,
            qasper_question=question,
            qasper_paper=paper,
        )

    xrag_specs = [item for item in manifest["cases"] if item["kind"] == "xrag"]
    xrag_rows = load_xrag_rows(manifest) if xrag_specs else {}
    for item in xrag_specs:
        row = xrag_rows[(item["task"], item.get("dataset_language", "zh"), item["example_id"])]
        records = [
            Record(f"article-{index}", f"article-{index}", row["articles"][index])
            for index in row["article_order"]
        ]
        document_id, block_map = ingest_records(
            store,
            f"xrag-{item['example_id']}",
            records,
            parser="plain_text",
            parser_version="xrag-source-records-v1",
        )
        prepared[item["id"]] = Case(
            case_id=item["id"],
            kind="xrag",
            question=row[item.get("question_field", "question")],
            language=item.get("language", "zh-Hant"),
            document_id=document_id,
            store=store,
            parent_by_block=block_map,
            parent_text={record.parent_id: record.text for record in records},
            gold_ids=tuple(
                f"article-{index}"
                for index, is_support in enumerate(row["is_support"])
                if is_support
            ),
        )

    drcd_specs = [item for item in manifest["cases"] if item["kind"] == "drcd"]
    if drcd_specs:
        drcd_path = ROOT / manifest["drcd"]["path"]
        if sha256(drcd_path) != DRCD_SHA256:
            raise ValueError("the pinned DRCD dev split checksum changed")
        drcd_data = json.loads(drcd_path.read_text(encoding="utf-8"))
        articles = {article["id"]: article for article in drcd_data["data"]}
        by_article: dict[str, list[dict]] = {}
        for item in drcd_specs:
            by_article.setdefault(item["article_id"], []).append(item)
        for article_id, specs in by_article.items():
            article = articles.get(article_id)
            if article is None:
                raise ValueError("a frozen DRCD article is missing")
            paragraph_by_id = {paragraph["id"]: paragraph for paragraph in article["paragraphs"]}
            records = [
                Record(
                    f"paragraph-{paragraph['id']}",
                    f"paragraph-{paragraph['id']}",
                    paragraph["context"],
                    metadata={"title": article["title"]},
                )
                for paragraph in article["paragraphs"]
            ]
            document_id, block_map = ingest_records(
                store,
                f"drcd-{article_id}",
                records,
                parser="plain_text",
                parser_version=f"drcd-{manifest['drcd']['revision']}-article-v1",
            )
            questions_by_id = {
                qa["id"]: qa for paragraph in article["paragraphs"] for qa in paragraph["qas"]
            }
            for item in specs:
                question = questions_by_id.get(item["qa_id"])
                paragraph = paragraph_by_id.get(item["paragraph_id"])
                if question is None or paragraph is None or not question["answers"]:
                    raise ValueError("a frozen DRCD QA reference is missing")
                if not any(
                    paragraph["context"][
                        answer["answer_start"] : answer["answer_start"] + len(answer["text"])
                    ]
                    == answer["text"]
                    for answer in question["answers"]
                ):
                    raise ValueError("a frozen DRCD answer span no longer aligns")
                prepared[item["id"]] = Case(
                    case_id=item["id"],
                    kind="drcd",
                    question=question["question"],
                    language=item["language"],
                    document_id=document_id,
                    store=store,
                    parent_by_block=block_map,
                    parent_text={record.parent_id: record.text for record in records},
                    gold_ids=(f"paragraph-{item['paragraph_id']}",),
                )

    pdf_assets = {}
    for item in manifest["cases"]:
        if item["kind"] != "pdf" or item["document"] in pdf_assets:
            continue
        pdf = manifest["pdfs"][item["document"]]
        bundle = convert_pdf(
            ROOT / pdf["path"],
            artifacts=DEFAULT_ARTIFACTS,
            tokenizer_dir=DEFAULT_MODEL_DIR,
            language=None,
            usage_scope="public benchmark PDF",
        )
        ingested = store.ingest(
            bundle.metadata,
            bundle.units(),
            block_factory=bundle.block_factory,
            source_manifests=bundle.source_manifests(),
        )
        source = load_bundle(store, ingested.document.document_id)
        blocks = source.blocks
        locations = {
            block.block_id: source.source.location_for(
                ingested.document.document_id, block.block_id
            )
            or {}
            for block in blocks
        }
        context_groups = tuple(
            SearchContextGroup(
                chunk_index=index,
                text=source.source.contextualized_chunk_text(index),
                source_refs=source.source.context_refs_for_chunk(index),
                source_block_ids=source.source.context_block_ids_for_chunk(index),
            )
            for index in range(len(source.chunks))
        )
        pdf_assets[item["document"]] = {
            "document_id": ingested.document.document_id,
            "parent_by_block": {block.block_id: f"native:{block.block_id}" for block in blocks},
            "parent_text": {f"native:{block.block_id}": block.text for block in blocks},
            "pages_by_block": {
                block_id: tuple(location.get("pages", []))
                for block_id, location in locations.items()
            },
            "locations_by_block": locations,
            "context_groups": context_groups,
        }
    for item in manifest["cases"]:
        if item["kind"] != "pdf":
            continue
        asset = pdf_assets[item["document"]]
        prepared[item["id"]] = Case(
            case_id=item["id"],
            kind="pdf",
            question=item["question"],
            language=item["language"],
            document_id=asset["document_id"],
            store=store,
            parent_by_block=asset["parent_by_block"],
            parent_text=asset["parent_text"],
            pages_by_block=asset["pages_by_block"],
            locations_by_block=asset["locations_by_block"],
            source_context_groups=asset["context_groups"],
        )

    if set(prepared) != set(case_specs):
        raise ValueError("not every frozen evaluation case was prepared")
    return prepared


def rank_cases(
    cases: dict[str, Case],
    encoder: NativeEmbeddingEncoder,
    *,
    final_k: int,
) -> list[dict]:
    text_documents = sorted({case.document_id for case in cases.values() if case.kind != "pdf"})
    for document_id in text_documents:
        store = next(case.store for case in cases.values() if case.document_id == document_id)
        build_bm25_index(store, document_id)
        from page_cite_rag.dense import build_dense_index

        build_dense_index(store, document_id, encoder, versioned=True)
    for case in cases.values():
        if case.kind == "pdf":
            build_docling_index(case.store, case.document_id, encoder)

    rows = []
    for case in cases.values():
        if case.kind == "pdf":
            from page_cite_rag.docling_retrieval import load_retriever

            retriever = load_retriever(case.store, case.document_id, encoder)
        else:
            retriever = document_retriever(
                case.store, case.document_id, "hybrid", encoder=encoder, versioned_dense=True
            )
        candidate_result = retriever.search(case.question, top_k=RERANK_CANDIDATE_K)
        result = _limit_result(candidate_result, final_k)
        case.candidate_result = candidate_result
        case.a_result = result
        case.source_blocks = tuple(case.store.get_blocks(case.document_id))
        candidate_units = (
            [
                {
                    "unit_id": f"chunk:{group.chunk_index}",
                    "source_block_ids": list(group.source_block_ids),
                }
                for group in candidate_result.context_groups
            ]
            if candidate_result.context_groups
            else [
                {
                    "unit_id": f"block:{hit.block.block_id}",
                    "source_block_ids": [hit.block.block_id],
                }
                for hit in candidate_result.hits
            ]
        )
        rows.append(
            {
                "case_id": case.case_id,
                "arm": "A",
                "method": result.method,
                "candidate_count": len(result.hits),
                "candidate_pool_count": len(candidate_units),
                "candidate_pool": candidate_units,
                "top_k": final_k,
                "retrieval_ms": result.elapsed_ms,
                "branches": [
                    {
                        "name": branch.name,
                        "index_id": branch.index_id,
                        "elapsed_ms": branch.elapsed_ms,
                        "block_ids": list(branch.block_ids),
                    }
                    for branch in result.branches
                ],
                "ranking": [
                    {
                        "rank": hit.rank,
                        "block_id": hit.block.block_id,
                        "parent_id": case.parent_by_block.get(hit.block.block_id),
                        "source_pages": list(case.pages_by_block.get(hit.block.block_id, ())),
                        "source_location": case.locations_by_block.get(hit.block.block_id),
                        "score": hit.score,
                    }
                    for hit in result.hits
                ],
                "full_document_source_order": [block.block_id for block in case.source_blocks],
            }
        )
    return rows


def _limit_result(result: SearchResult, top_k: int) -> SearchResult:
    if not result.context_groups:
        return result.model_copy(update={"hits": result.hits[:top_k]})
    groups = result.context_groups[:top_k]
    hits_by_id = {hit.block.block_id: hit for hit in result.hits}
    ordered_ids = tuple(
        dict.fromkeys(block_id for group in groups for block_id in group.source_block_ids)
    )
    if any(block_id not in hits_by_id for block_id in ordered_ids):
        raise ValueError("a retrieved Docling chunk refers to a missing source block")
    hits = tuple(
        SearchHit(
            rank=rank,
            score=hits_by_id[block_id].score,
            block=hits_by_id[block_id].block,
        )
        for rank, block_id in enumerate(ordered_ids, 1)
    )
    return result.model_copy(update={"hits": hits, "context_groups": groups})


def rerank_cases(
    cases: dict[str, Case], server: NativeRerankerServer, *, final_k: int
) -> list[dict]:
    rows = []
    for case in cases.values():
        candidate_result = case.candidate_result
        if candidate_result is None:
            raise ValueError("hybrid candidates were not prepared for reranking")
        if candidate_result.context_groups:
            unit_ids = [f"chunk:{group.chunk_index}" for group in candidate_result.context_groups]
            source_block_ids = [
                list(group.source_block_ids) for group in candidate_result.context_groups
            ]
            documents = [group.text for group in candidate_result.context_groups]
        else:
            unit_ids = [f"block:{hit.block.block_id}" for hit in candidate_result.hits]
            source_block_ids = [[hit.block.block_id] for hit in candidate_result.hits]
            documents = [hit.block.text for hit in candidate_result.hits]
        started = time.perf_counter()
        try:
            ranked = server.rerank(case.question, documents)
        except Exception as exc:
            raise RuntimeError(
                f"reranking failed for {case.case_id} with {len(documents)} candidate units: {exc}"
            ) from exc
        rerank_ms = (time.perf_counter() - started) * 1000
        selected = ranked[:final_k]
        if candidate_result.context_groups:
            selected_groups = tuple(
                candidate_result.context_groups[index] for index, _score in selected
            )
            original_hits = {hit.block.block_id: hit for hit in candidate_result.hits}
            ordered_ids = tuple(
                dict.fromkeys(
                    block_id for group in selected_groups for block_id in group.source_block_ids
                )
            )
            if any(block_id not in original_hits for block_id in ordered_ids):
                raise ValueError("reranked Docling chunk refers to a missing source block")
            block_scores = {}
            for index, score in selected:
                for block_id in candidate_result.context_groups[index].source_block_ids:
                    block_scores[block_id] = max(block_scores.get(block_id, float("-inf")), score)
            hits = tuple(
                SearchHit(
                    rank=rank,
                    score=block_scores[block_id],
                    block=original_hits[block_id].block,
                )
                for rank, block_id in enumerate(ordered_ids, 1)
            )
            context_groups = selected_groups
        else:
            hits = tuple(
                SearchHit(
                    rank=rank,
                    score=score,
                    block=candidate_result.hits[index].block,
                )
                for rank, (index, score) in enumerate(selected, 1)
            )
            context_groups = ()
        index_identity = hashlib.sha256(
            json.dumps(
                {
                    "candidate_index": candidate_result.index_id,
                    "reranker_sha256": RERANKER_SHA256,
                    "unit_ids": unit_ids,
                    "ranking": [index for index, _score in ranked],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        case.r_result = candidate_result.model_copy(
            update={
                "index_id": index_identity,
                "hits": hits,
                "context_groups": context_groups,
                "elapsed_ms": candidate_result.elapsed_ms + rerank_ms,
            }
        )
        rows.append(
            {
                "case_id": case.case_id,
                "arm": "R",
                "candidate_pool_count": len(documents),
                "rerank_elapsed_ms": rerank_ms,
                "pair_token_counts": list(server.last_pair_tokens),
                "reranked_order": [
                    {
                        "unit_id": unit_ids[index],
                        "source_block_ids": source_block_ids[index],
                        "score": score,
                        "rank": rank,
                    }
                    for rank, (index, score) in enumerate(ranked, 1)
                ],
                "top_k": final_k,
                "ranking": [
                    {
                        "rank": hit.rank,
                        "block_id": hit.block.block_id,
                        "parent_id": case.parent_by_block.get(hit.block.block_id),
                        "source_pages": list(case.pages_by_block.get(hit.block.block_id, ())),
                        "source_location": case.locations_by_block.get(hit.block.block_id),
                        "score": hit.score,
                    }
                    for hit in hits
                ],
            }
        )
    return rows


class InputCappedModel:
    """Enforce the frozen exact input cap while sharing the resident model."""

    def __init__(self, model: LlamaModelClient) -> None:
        self.model = model
        self.identity = model.identity
        self.last_request = None

    def prepare_budget(self, messages, schema):
        budget = self.model.prepare_budget(messages, schema)
        if budget.exact_input_tokens is None or budget.exact_input_tokens > INPUT_LIMIT:
            raise ContextBudgetError("input exceeds the frozen 10,000-token limit")
        return budget

    def generate(self, messages, response_model):
        self.last_request = None
        budget = self.prepare_budget(messages, response_model.model_json_schema())
        self.last_request = {
            "messages": [message.model_dump(mode="json") for message in messages],
            "response_schema": response_model.model_json_schema(),
            "exact_input_tokens": budget.exact_input_tokens,
        }
        return self.model.generate(messages, response_model)


def preflight_workflows(
    cases: dict[str, Case],
    model: LlamaModelClient,
    budget_config: LlamaConfig,
    *,
    final_k: int,
    arms: tuple[str, ...] = ("A", "R"),
    include_full_document: bool = True,
    q6_shared_context: bool = False,
) -> tuple[dict[str, dict], dict[str, dict]]:
    capped_model = InputCappedModel(model)
    ready = {}
    rows = {}
    for case in cases.values():
        if (
            not case.source_blocks
            or case.a_result is None
            or ("R" in arms and case.r_result is None)
        ):
            raise ValueError("retrieval did not prepare the frozen case")
        workflows = {}
        shared_result = case.candidate_result if q6_shared_context else case.a_result
        result_by_arm = {
            "A": case.a_result,
            "B": case.a_result,
            "R": case.r_result,
            "A1": shared_result,
            "B1": shared_result,
        }
        for arm in arms:
            result = result_by_arm[arm]
            if result is None:
                raise ValueError(f"retrieval result for arm {arm} is unavailable")
            workflow_top_k = RERANK_CANDIDATE_K if q6_shared_context else final_k
            retriever = FrozenSearchRetriever(result, top_k=workflow_top_k)
            workflow = document_workflow(
                case.store,
                case.document_id,
                retriever,
                capped_model,
                budget_config,
                top_k=workflow_top_k,
            )
            blocks = workflow._context_blocks(result, workflow._read_hits(result, case.question))
            if q6_shared_context:
                prepared = workflow._prepare_evidence_selection(
                    case.question, case.language, blocks, result.context_groups
                )
                selected = prepared["blocks"]
                omitted = prepared["omitted"]
                budget = prepared["answer_budget"]
                answer_messages = workflow._messages_with_context_groups(
                    case.question,
                    case.language,
                    "local",
                    selected,
                    None,
                    prepared["context_groups"],
                )
                final_budget = budget_for_model(
                    capped_model,
                    answer_messages,
                    workflow._response_model_for(selected).model_json_schema(),
                    budget_config,
                )
                if final_budget != budget:
                    raise ValueError("A1 and B1 final-answer preflight budgets differ")
                rows[f"{case.case_id}/{arm}"] = {
                    "eligible": True,
                    "exact_input_tokens": final_budget.exact_input_tokens,
                    "context_limit": final_budget.context_limit,
                    "output_reserved": final_budget.output_reserved,
                    "retrieved_blocks": len(blocks),
                    "admitted_blocks": len(selected),
                    "omitted_blocks": len(omitted),
                    "admitted_block_ids": [block.block_id for block in selected],
                    "omitted_block_ids": list(omitted),
                    "candidate_count": len(prepared["units"]),
                    "candidate_ids": list(prepared["units"]),
                    "A1_final_input_tokens": final_budget.exact_input_tokens,
                    "B1_selector_input_tokens": (
                        prepared["selection_budget"].exact_input_tokens
                        if prepared["units"]
                        else None
                    ),
                    "B1_final_input_tokens": final_budget.exact_input_tokens,
                    "common_context": True,
                    "prompt_version": workflow.prompt_version,
                    "selection_prompt_version": EVIDENCE_SELECTION_PROMPT_VERSION,
                }
                ready[f"{case.case_id}/{arm}"] = {
                    "workflow": workflow,
                    "search_result": result,
                    "blocks": blocks,
                    "prepared": prepared,
                }
                workflows[arm] = workflow
                continue
            selected, omitted, _messages, budget = workflow._context(
                case.question,
                case.language,
                "local",
                blocks,
                None,
                context_groups=result.context_groups,
            )
            rows[f"{case.case_id}/{arm}"] = {
                "eligible": True,
                "exact_input_tokens": budget.exact_input_tokens,
                "context_limit": budget.context_limit,
                "output_reserved": budget.output_reserved,
                "retrieved_blocks": len(blocks),
                "admitted_blocks": len(selected),
                "omitted_blocks": len(omitted),
                "admitted_block_ids": [block.block_id for block in selected],
                "omitted_block_ids": list(omitted),
                "prompt_version": workflow.prompt_version,
            }
            if arm == "B":
                rows[f"{case.case_id}/{arm}"].update(
                    workflow.evidence_first_preflight(
                        case.question,
                        case.language,
                        blocks,
                        result.context_groups,
                    )
                )
            ready[f"{case.case_id}/{arm}"] = workflow
            workflows[arm] = workflow
        if q6_shared_context or not include_full_document:
            continue
        workflow = workflows["A"]

        covered_ids = {
            block_id for group in case.source_context_groups for block_id in group.source_block_ids
        }
        source_ids = {block.block_id for block in case.source_blocks}
        if case.source_context_groups and covered_ids != source_ids:
            rows[f"{case.case_id}/C"] = {
                "eligible": False,
                "reason": "complete_document_chunk_coverage_incomplete",
                "source_blocks": len(source_ids),
                "covered_blocks": len(covered_ids),
                "prompt_version": workflow.prompt_version,
            }
            continue
        try:
            c_selected, c_omitted, _c_messages, c_budget = workflow._context(
                case.question,
                case.language,
                "local",
                case.source_blocks,
                None,
                context_groups=case.source_context_groups,
            )
        except ContextBudgetError as exc:
            rows[f"{case.case_id}/C"] = {
                "eligible": False,
                "reason": "complete_document_exceeds_admission_limit",
                "detail": str(exc)[:300],
                "source_blocks": len(case.source_blocks),
                "prompt_version": workflow.prompt_version,
            }
            continue
        if c_omitted or {block.block_id for block in c_selected} != source_ids:
            rows[f"{case.case_id}/C"] = {
                "eligible": False,
                "reason": "complete_document_not_fully_admitted",
                "exact_input_tokens": c_budget.exact_input_tokens,
                "source_blocks": len(case.source_blocks),
                "admitted_blocks": len(c_selected),
                "omitted_blocks": len(c_omitted),
                "prompt_version": workflow.prompt_version,
            }
            continue
        rows[f"{case.case_id}/C"] = {
            "eligible": True,
            "exact_input_tokens": c_budget.exact_input_tokens,
            "context_limit": c_budget.context_limit,
            "output_reserved": c_budget.output_reserved,
            "source_blocks": len(case.source_blocks),
            "admitted_blocks": len(case.source_blocks),
            "omitted_blocks": 0,
            "admitted_block_ids": [block.block_id for block in case.source_blocks],
            "omitted_block_ids": [],
            "prompt_version": workflow.prompt_version,
        }
        ready[f"{case.case_id}/C"] = workflow
    return ready, rows


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(temp, 0o600)
    os.replace(temp, path)


def append_jsonl_record(path: Path, value: dict) -> None:
    """Append one mode-0600 evaluation ledger event without rewriting history."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as ledger:
            descriptor = -1
            ledger.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def run_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def wait_for_gpu_cool(model: LlamaModelClient) -> float:
    """Require a cool, single-owner GPU before each completion."""
    server_pid = model.diagnostics().get("server_pid")
    if type(server_pid) is not int or server_pid < 1:
        raise RuntimeError("the owned model server PID is unavailable")
    deadline = time.monotonic() + COOL_WAIT_SECONDS
    started = time.monotonic()
    while True:
        try:
            values = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    GPU_UUID,
                    "--query-gpu=memory.used,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=3,
            ).strip()
            memory, temperature = map(int, values.split(","))
            raw_pids = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    GPU_UUID,
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=3,
            ).strip()
            pids = {int(pid) for pid in raw_pids.splitlines() if pid}
        except (OSError, ValueError, subprocess.SubprocessError):
            raise RuntimeError("GPU telemetry failed; stop rather than assume safety") from None
        if pids - {server_pid} or memory > 12_000 or temperature > 80:
            raise RuntimeError("GPU resource or temperature guard was reached")
        if temperature <= COOL_TARGET_C:
            return time.monotonic() - started
        if time.monotonic() >= deadline:
            raise TimeoutError("GPU did not cool to 60 C within the three-minute guard")
        time.sleep(5)


def static_configuration(
    manifest: dict, *, final_k: int, manifest_path: Path | None = None
) -> dict:
    phase = manifest.get("phase", "Q2")
    if phase in ("Q6", "Q6-confirmation"):
        confirmation = phase == "Q6-confirmation"
        return {
            "run_version": (
                "quality-convergence-q6-a1-confirmation-v1"
                if confirmation
                else "quality-convergence-q6-a1-b1-k20-v1"
            ),
            "manifest_sha256": sha256(
                manifest_path
                or (Q6_CONFIRMATION_MANIFEST_PATH if confirmation else Q6_MANIFEST_PATH)
            ),
            "parent_q5_sha256": manifest["parent_q5_sha256"],
            **({"parent_q6_sha256": manifest["parent_q6_sha256"]} if confirmation else {}),
            "qasper_revision": manifest["qasper"]["revision"],
            "qasper_sha256": manifest["qasper"]["sha256"],
            **(
                {
                    "xrag_revision": manifest["xrag"]["revision"],
                    "xrag_sha256": manifest["xrag"]["sha256"],
                    "drcd_revision": manifest["drcd"]["revision"],
                    "drcd_sha256": manifest["drcd"]["sha256"],
                    "pdf_sha256": {key: value["sha256"] for key, value in manifest["pdfs"].items()},
                }
                if not confirmation
                else {}
            ),
            "source_code_sha256": {
                "runner": sha256(Path(__file__).resolve()),
                "full_answer": sha256(ROOT / "src/page_cite_rag/full_answer.py"),
                "workflows": sha256(ROOT / "src/page_cite_rag/workflows.py"),
                "retrieval": sha256(ROOT / "src/page_cite_rag/retrieval.py"),
                "docling_document": sha256(ROOT / "src/page_cite_rag/docling_document.py"),
                "docling_retrieval": sha256(ROOT / "src/page_cite_rag/docling_retrieval.py"),
                "embedding_server": sha256(ROOT / "src/page_cite_rag/embedding_server.py"),
                "answer_metrics": sha256(ROOT / "src/page_cite_rag/answer_metrics.py"),
                "llama": sha256(ROOT / "src/page_cite_rag/llama.py"),
                "pyproject": sha256(ROOT / "pyproject.toml"),
                "uv_lock": sha256(ROOT / "uv.lock"),
            },
            "models": {
                "generator": "Qwen3.5-9B-Q4_K_M",
                "embedding": "Qwen3-Embedding-0.6B-f16",
                "runtime": "pinned local llama.cpp CUDA bundle",
                "device_uuid": GPU_UUID,
            },
            "arms": {
                "A1": "candidate A1: common top-20 admitted source context; direct answer"
                if confirmation
                else "common top-20 admitted source context; direct answer",
                **(
                    {}
                    if confirmation
                    else {
                        "B1": (
                            "same common context; select at most 12 source IDs; same final answerer"
                        )
                    }
                ),
            },
            "shared": {
                "candidate_pool_k": 20,
                "reranker": False,
                "query_rewrite": False,
                "answer_contract": manifest["shared"]["answer_contract"],
                "exact_input_token_cap": manifest["shared"]["exact_input_token_cap"],
                "output_tokens_per_call": manifest["shared"]["output_tokens_per_call"],
                "template_reserve": manifest["shared"]["template_reserve"],
                "temperature": 0.0,
                "thinking": False,
                "no_gold_in_generation": True,
                "no_retries": True,
                "case_count": manifest["case_count"],
                "common_context_preflight": manifest["shared"]["common_context_preflight"],
            },
            "selection": manifest["selection_policy"],
            **({"confirmation_selection": manifest["selection"]} if confirmation else {}),
            "fallback_policy": manifest["fallback_policy"],
            "resource_limits": manifest["resource_limits"],
            "wall_limit_seconds": manifest["generation_limits"]["invocation_wall_seconds"],
            "generation_limit_seconds": manifest["generation_limits"][
                "invocation_generation_seconds"
            ],
        }
    if phase == "Q5":
        sources = {
            "manifest_sha256": sha256(manifest_path or Q5_MANIFEST_PATH),
            "parent_q3_sha256": manifest["parent_q3_sha256"],
            "parent_p0_sha256": manifest["parent_p0_sha256"],
            "qasper_revision": manifest["qasper"]["revision"],
            "qasper_split": manifest["qasper"]["split"],
            "qasper_sha256": manifest["qasper"]["sha256"],
            "xrag_revision": manifest["xrag"]["revision"],
            "xrag_sha256": manifest["xrag"]["sha256"],
            "drcd_revision": manifest["drcd"]["revision"],
            "drcd_sha256": manifest["drcd"]["sha256"],
            "pdf_sha256": {key: value["sha256"] for key, value in manifest["pdfs"].items()},
        }
        return {
            "run_version": f"quality-convergence-q5-hybrid-evidence-first-k{final_k}-v3",
            **sources,
            "source_code_sha256": {
                "runner": sha256(Path(__file__).resolve()),
                "full_answer": sha256(ROOT / "src/page_cite_rag/full_answer.py"),
                "workflows": sha256(ROOT / "src/page_cite_rag/workflows.py"),
                "retrieval": sha256(ROOT / "src/page_cite_rag/retrieval.py"),
                "docling_document": sha256(ROOT / "src/page_cite_rag/docling_document.py"),
                "docling_retrieval": sha256(ROOT / "src/page_cite_rag/docling_retrieval.py"),
                "embedding_server": sha256(ROOT / "src/page_cite_rag/embedding_server.py"),
                "answer_metrics": sha256(ROOT / "src/page_cite_rag/answer_metrics.py"),
                "llama": sha256(ROOT / "src/page_cite_rag/llama.py"),
                "pyproject": sha256(ROOT / "pyproject.toml"),
                "uv_lock": sha256(ROOT / "uv.lock"),
            },
            "models": {
                "generator": "Qwen3.5-9B-Q4_K_M",
                "embedding": "Qwen3-Embedding-0.6B-f16",
                "runtime": "pinned local llama.cpp CUDA bundle",
                "device_uuid": GPU_UUID,
            },
            "arms": {
                "A": f"shared BM25S + Qwen dense + RRF top-{RERANK_CANDIDATE_K}; direct answer",
                "B": (
                    "same retrieved context; exact-quote-verified evidence selection; "
                    "answer from selected complete source units"
                ),
            },
            "shared": {
                "retrieval": "identical pinned hybrid ranking and top-k for A/B",
                "candidate_pool_k": RERANK_CANDIDATE_K,
                "reranker": False,
                "query_rewrite": False,
                "answer_contract": "complete-cited-answer-v3",
                "context_limit": CONTEXT_TOKENS,
                "exact_input_token_cap": INPUT_LIMIT,
                "output_tokens_per_call": OUTPUT_TOKENS,
                "template_reserve": TEMPLATE_RESERVE,
                "temperature": 0.0,
                "thinking": False,
                "no_gold_in_generation": True,
                "no_retries": True,
                "case_count": manifest["case_count"],
                "same_final_answer_schema_and_prompt": True,
                "continue_after_recorded_non_guard_case_failure": True,
                "stop_after_generation_client_closes": True,
            },
            "evidence_first": {
                "prompt_version": "evidence-selection/v1",
                "inspiration": "PaperQA2 gather-evidence then answer; partial local adaptation",
                "max_completion_calls_per_case": 2,
                "quote_validation": (
                    "exact contiguous substring of a verified candidate source unit"
                ),
                "selection_limit": 12,
                "annotations_are_hypotheses": True,
                "final_context": (
                    "complete original selected block or Docling chunk, retaining source mapping"
                ),
            },
            "wall_limit_seconds": MAX_WALL_SECONDS,
            "generation_limit_seconds": MAX_GENERATION_SECONDS,
        }
    if phase == "Q4":
        sources = {
            "manifest_sha256": sha256(manifest_path or Q4_MANIFEST_PATH),
            "qasper_revision": manifest["qasper"]["revision"],
            "qasper_split": manifest["qasper"]["split"],
            "qasper_sha256": manifest["qasper"]["sha256"],
        }
        source_hashes = {
            "runner": sha256(Path(__file__).resolve()),
            "agent": sha256(ROOT / "src/page_cite_rag/agent.py"),
            "llama": sha256(ROOT / "src/page_cite_rag/llama.py"),
            "full_answer": sha256(ROOT / "src/page_cite_rag/full_answer.py"),
            "workflows": sha256(ROOT / "src/page_cite_rag/workflows.py"),
            "retrieval": sha256(ROOT / "src/page_cite_rag/retrieval.py"),
            "embedding_server": sha256(ROOT / "src/page_cite_rag/embedding_server.py"),
            "answer_metrics": sha256(ROOT / "src/page_cite_rag/answer_metrics.py"),
            "pyproject": sha256(ROOT / "pyproject.toml"),
            "uv_lock": sha256(ROOT / "uv.lock"),
        }
        return {
            "run_version": f"quality-convergence-q4-fixed-agent-k{final_k}-v1",
            **sources,
            "source_code_sha256": source_hashes,
            "models": {
                "generator": "Qwen3.5-9B-Q4_K_M",
                "embedding": "Qwen3-Embedding-0.6B-f16",
                "runtime": "pinned local llama.cpp CUDA bundle",
                "device_uuid": GPU_UUID,
            },
            "arms": {
                "fixed": "hybrid top-k plus one deterministic next block in the top-hit section",
                "agent": "same hybrid top-k plus LangChain bounded scoped source reads",
            },
            "shared": {
                "retrieval": "same BM25S + Qwen dense RRF candidate pool and final top-k",
                "candidate_pool_k": RERANK_CANDIDATE_K,
                "query_rewrite": False,
                "answer_contract": "complete-cited-answer-v2",
                "context_limit": CONTEXT_TOKENS,
                "exact_input_token_cap": INPUT_LIMIT,
                "output_tokens": OUTPUT_TOKENS,
                "template_reserve": TEMPLATE_RESERVE,
                "temperature": 0.0,
                "thinking": False,
                "no_gold_in_generation": True,
                "no_retries": True,
                "case_count": manifest["case_count"],
            },
            "agent": {
                "framework": "langchain",
                "langchain_version": "1.4.6",
                "langchain_openai_version": "1.7.2",
                "model_call_limit": 2,
                "tool_call_limit": 4,
                "final_answer_calls": 1,
                "external_tools": False,
                "tool_policy": "search_sources then scoped read_outline_section",
            },
            "wall_limit_seconds": MAX_WALL_SECONDS,
            "generation_limit_seconds": MAX_GENERATION_SECONDS,
        }
    is_q3 = phase == "Q3"
    arms = tuple(manifest.get("arms", ("A", "R")))
    sources = {
        "manifest_sha256": sha256(manifest_path or P0_PATH),
        "qasper_revision": manifest["qasper"]["revision"],
        "qasper_split": manifest["qasper"].get("split", "train"),
        "qasper_sha256": manifest["qasper"]["sha256"],
        "xrag_revision": manifest["xrag"]["revision"],
        "xrag_sha256": sha256(XRAG_PATH),
    }
    if not is_q3:
        sources["p0_sha256"] = sha256(P0_PATH)
        sources["pdf_sha256"] = {key: value["sha256"] for key, value in manifest["pdfs"].items()}
    else:
        sources["drcd_revision"] = manifest["drcd"]["revision"]
        sources["drcd_sha256"] = sha256(ROOT / manifest["drcd"]["path"])
    source_hashes = {
        "runner": sha256(Path(__file__).resolve()),
        "workflows": sha256(ROOT / "src/page_cite_rag/workflows.py"),
        "full_answer": sha256(ROOT / "src/page_cite_rag/full_answer.py"),
        "retrieval": sha256(ROOT / "src/page_cite_rag/retrieval.py"),
        "embedding_server": sha256(ROOT / "src/page_cite_rag/embedding_server.py"),
        "answer_metrics": sha256(ROOT / "src/page_cite_rag/answer_metrics.py"),
        "uv_lock": sha256(ROOT / "uv.lock"),
    }
    if "R" in arms:
        source_hashes["reranker_server"] = sha256(ROOT / "src/page_cite_rag/reranker_server.py")
    static = {
        "run_version": (
            f"quality-convergence-q3-a-k{final_k}-v1"
            if is_q3
            else f"quality-convergence-q2-ar-k{final_k}-v1"
        ),
        **sources,
        "source_code_sha256": {
            **source_hashes,
        },
        "models": {
            "generator": "Qwen3.5-9B-Q4_K_M",
            "embedding": "Qwen3-Embedding-0.6B-f16",
            "runtime": "pinned local llama.cpp CUDA bundle",
            "device_uuid": GPU_UUID,
        },
        "arms": {"A": f"BM25S + Qwen dense + RRF top-{RERANK_CANDIDATE_K}, take top-{final_k}"},
        "shared": {
            "parser_and_source_representation": "same",
            "candidate_pool_k": RERANK_CANDIDATE_K,
            "answer_contract": "complete-cited-answer-v2",
            "context_limit": CONTEXT_TOKENS,
            "exact_input_token_cap": INPUT_LIMIT,
            "output_tokens": OUTPUT_TOKENS,
            "template_reserve": TEMPLATE_RESERVE,
            "temperature": 0.0,
            "seed": 42,
            "thinking": False,
            "no_gold_in_generation": True,
            "no_retries": True,
            "case_count": manifest["case_count"],
        },
        "wall_limit_seconds": MAX_WALL_SECONDS,
        "generation_limit_seconds": MAX_GENERATION_SECONDS,
    }
    if "R" in arms:
        static["models"]["reranker"] = {
            "model": "Qwen3-Reranker-0.6B-Q8_0-GGUF",
            "sha256": RERANKER_SHA256,
            "runtime": "pinned local llama.cpp CUDA bundle",
        }
        static["arms"]["R"] = (
            f"same hybrid top-{RERANK_CANDIDATE_K} candidates, Qwen rerank, take top-{final_k}"
        )
        static["shared"]["reranker_instruction"] = (
            "GGUF template default: retrieve relevant passages that answer the query"
        )
        static["shared"]["reranker_pair_token_limit"] = 4096
    return static


def _q6_failure_accounting_code_update(existing: dict, current: dict) -> bool:
    """Permit only the reviewed error-accounting serialization fix on a Q6 resume."""
    supported_versions = {
        "quality-convergence-q6-a1-b1-k20-v1",
        "quality-convergence-q6-a1-confirmation-v1",
    }
    if not (
        existing.get("run_version") in supported_versions
        and current.get("run_version") == existing.get("run_version")
    ):
        return False
    old_hashes = existing.get("source_code_sha256", {})
    new_hashes = current.get("source_code_sha256", {})
    if not isinstance(old_hashes, dict) or set(old_hashes) != set(new_hashes):
        return False
    changed = {name for name in old_hashes if old_hashes[name] != new_hashes[name]}
    if not changed or not changed.issubset({"runner", "full_answer"}):
        return False
    old_values = {key: value for key, value in existing.items() if key != "source_code_sha256"}
    new_values = {key: value for key, value in current.items() if key != "source_code_sha256"}
    return old_values == new_values


def validate_output_target(output: Path, resume: bool, static: dict) -> dict | None:
    """Reject ambiguous output reuse before starting either model server."""
    if output.is_symlink() and not output.exists():
        raise ValueError("the output path is a dangling symlink")
    output = output.resolve()
    if not output.exists():
        if resume:
            raise ValueError("cannot resume because the output directory does not exist")
        return None
    if not output.is_dir():
        raise ValueError("the output path exists but is not a directory")
    config_path = output / "configuration.json"
    existing_files = tuple(output.iterdir())
    if not config_path.exists():
        if existing_files:
            raise ValueError("the output directory is non-empty but has no configuration.json")
        if resume:
            raise ValueError("cannot resume because the output directory has no configuration")
        return None
    old = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(old, dict) or (
        old.get("static") != static
        and not (
            resume
            and isinstance(old.get("static"), dict)
            and _q6_failure_accounting_code_update(old["static"], static)
        )
    ):
        raise ValueError("resume configuration differs; refusing to mix run artifacts")
    if not resume:
        raise ValueError("the output directory already contains a run; pass --resume")
    return old


def result_path(output: Path, arm: str, case_id: str) -> Path:
    return output / arm / f"{case_id}.json"


def resumed_record_token_usage(record: dict) -> tuple[int, int]:
    """Read token counts from both success and failure/legacy result records."""
    result = record.get("result")
    if not isinstance(result, dict):
        result = {}
    budget = result.get("budget")
    if not isinstance(budget, dict):
        budget = {}
    stats = result.get("generation_stats")
    if not isinstance(stats, dict):
        stats = {}
    diagnostics = record.get("model_diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    completion = diagnostics.get("last_completion")
    if not isinstance(completion, dict):
        completion = {}
    input_tokens = record.get("input_tokens")
    output_tokens = record.get("output_tokens")
    if input_tokens is None:
        input_tokens = completion.get("tokens_evaluated", budget.get("exact_input_tokens", 0))
    if output_tokens is None:
        output_tokens = completion.get("tokens_predicted", stats.get("eval_count", 0))
    return int(input_tokens or 0), int(output_tokens or 0)


def continue_after_case_failure(
    phase: str | None, guard_failures: list, *, client_closed: bool, error: BaseException
) -> bool:
    """Q5/Q6 may continue recorded ordinary errors only with a healthy local client."""
    return (
        phase in ("Q5", "Q6")
        and not guard_failures
        and not client_closed
        and isinstance(error, Exception)
    )


def client_closed_before_generation(error_type: str, error: str) -> bool:
    """Identify a closed-client error that did not issue a new model completion."""
    return error_type == "ModelError" and error == "generation client is closed"


def qasper_prediction_text(result: dict) -> str:
    """Map the product's explicit abstention status to QASPER's answer label."""
    if result.get("status") == "insufficient_evidence":
        return "Unanswerable"
    answer = result.get("answer", "")
    return answer if isinstance(answer, str) else ""


def evidence_selection_generation_stats(trace: dict) -> dict:
    """Return selector stats when present; failed selectors have a null selection."""
    selection = trace.get("selection")
    if not isinstance(selection, dict):
        return {}
    stats = selection.get("generation_stats")
    return stats if isinstance(stats, dict) else {}


def score_qasper(
    output: Path,
    qasper_cases: list[dict],
    *,
    split: str = "train",
    arms: tuple[str, ...] = ("A", "R"),
) -> dict:
    question_ids = {case["question_id"] for case in qasper_cases}
    gold = read_answer_references(ROOT / "data/qasper", split, question_ids)
    preflight_path = output / "preflight.json"
    preflight = (
        json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else {}
    )
    scores = {}
    for arm in arms:
        stored_arm = "B1" if arm == "B1_fallback_served" else arm
        fallback_only = arm == "B1_fallback_served"
        predictions = {}
        applicable_gold = {}
        not_applicable = 0
        no_answer_cases = correct_refusals = false_refusals = ambiguous_refusals = 0
        for case in qasper_cases:
            case_id = case["id"]
            question_id = case["question_id"]
            admission = preflight.get(f"{case_id}/{arm}", {})
            path = result_path(output, stored_arm, case_id)
            record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            declared_ineligible = admission.get("eligible") is False
            recorded_ineligible = record.get("execution") == "not_applicable"
            if declared_ineligible and path.is_file() and not recorded_ineligible:
                raise ValueError(f"preflight and execution disagree for {case_id}/{arm}")
            if recorded_ineligible and admission.get("eligible") is True:
                raise ValueError(f"preflight and execution disagree for {case_id}/{arm}")
            applicable_gold[question_id] = gold[question_id]
            is_no_answer = all(reference.answer_type == "none" for reference in gold[question_id])
            no_answer_cases += int(is_no_answer)
            if declared_ineligible or recorded_ineligible:
                not_applicable += 1
                continue
            if not path.is_file():
                continue
            if fallback_only:
                fallback = record.get("fallback", {})
                if fallback.get("execution") != "ok":
                    continue
                result = fallback.get("result", {})
            elif record.get("execution") == "ok":
                result = record.get("result", {})
            else:
                continue
            refused = result.get("status") == "insufficient_evidence"
            correct_refusals += int(is_no_answer and refused)
            has_answer_reference = any(
                reference.answer_type != "none" for reference in gold[question_id]
            )
            has_unanswerable_reference = any(
                reference.answer_type == "none" for reference in gold[question_id]
            )
            ambiguous = has_answer_reference and has_unanswerable_reference
            false_refusals += int(has_answer_reference and not ambiguous and refused)
            ambiguous_refusals += int(ambiguous and refused)
            claims = result.get("answer_claims", []) + result.get("detail_claims", [])
            if not claims:
                claims = result.get("claims", [])
            cited = []
            for claim in claims:
                for citation in claim.get("citations", []):
                    quote = citation.get("quote")
                    if isinstance(quote, str) and quote not in cited:
                        cited.append(quote)
            predictions[question_id] = AnswerPrediction(
                qasper_prediction_text(result), tuple(cited)
            )
        answerable_ids = {
            question_id
            for question_id, references in applicable_gold.items()
            if any(reference.answer_type != "none" for reference in references)
        }
        ambiguous_ids = {
            question_id
            for question_id, references in applicable_gold.items()
            if any(reference.answer_type != "none" for reference in references)
            and any(reference.answer_type == "none" for reference in references)
        }
        unambiguous_answerable_ids = answerable_ids - ambiguous_ids
        unanswerable_ids = set(applicable_gold) - answerable_ids

        def fixed_group(question_ids: set[str]) -> dict:
            selected_gold = {key: applicable_gold[key] for key in question_ids}
            selected_predictions = {
                key: predictions[key] for key in question_ids if key in predictions
            }
            return {
                "questions": len(selected_gold),
                "metrics": evaluate_answers(selected_gold, selected_predictions)
                if selected_gold
                else None,
            }

        per_question = {}
        for question_id, references in applicable_gold.items():
            prediction = predictions.get(question_id)
            metrics = score_prediction(references, prediction)
            per_question[question_id] = {
                "answer_f1": metrics["answer_f1"],
                "evidence_f1": metrics["evidence_f1"],
                "evidence_precision": metrics["evidence_precision"],
                "evidence_recall": metrics["evidence_recall"],
                "answer_type": metrics["answer_type"],
                "prediction_present": prediction is not None,
            }

        scores[arm] = {
            "applicable_questions": len(applicable_gold),
            "not_applicable_questions": not_applicable,
            "metrics": evaluate_answers(applicable_gold, predictions) if applicable_gold else None,
            "gold_fixed_groups": {
                "answerable_including_ambiguous": fixed_group(answerable_ids),
                "answerable_unambiguous": fixed_group(unambiguous_answerable_ids),
                "ambiguous_reference": fixed_group(ambiguous_ids),
                "unanswerable": fixed_group(unanswerable_ids),
            },
            "no_answer_questions": no_answer_cases,
            "correct_refusals": correct_refusals,
            "correct_refusal_rate": (
                correct_refusals / no_answer_cases if no_answer_cases else None
            ),
            "incorrect_refusals_on_unambiguous_answerable": false_refusals,
            "ambiguous_reference_refusals": ambiguous_refusals,
            "answerable_questions_including_ambiguous": len(answerable_ids),
            "incorrect_refusal_rate_on_answerable": (
                false_refusals / len(unambiguous_answerable_ids)
                if unambiguous_answerable_ids
                else None
            ),
            "per_question": per_question,
        }
    return {
        "scope": f"{len(qasper_cases)} frozen QASPER {split} questions; selected evaluation slice",
        "metric": "official-compatible QASPER Answer F1 and raw-passage Evidence F1",
        "scorer_version": "gold-fixed-subgroups/status-normalized/evidence-pr-v4",
        "not_official_leaderboard": True,
        "scores": scores,
    }


def score_qasper_retrieval(
    cases: dict[str, Case],
    output: Path,
    *,
    final_k: int,
    arms: tuple[str, ...] = ("A", "R"),
) -> dict:
    scores = {}
    unanswerable = 0
    for arm in arms:
        rows = []
        for case in cases.values():
            if case.kind != "qasper" or case.candidate_result is None:
                continue
            references = align_references(case.qasper_paper, case.qasper_question)
            if not references:
                unanswerable += int(arm == arms[0])
                continue
            candidate_ids = tuple(
                dict.fromkeys(
                    case.parent_by_block[hit.block.block_id]
                    for hit in case.candidate_result.hits
                    if hit.block.block_id in case.parent_by_block
                )
            )
            result = (
                case.candidate_result
                if arm in ("A1", "B1")
                else case.a_result
                if arm in ("A", "B", "fixed", "agent")
                else case.r_result
            )
            if result is None:
                raise ValueError("both retrieval arms must be prepared before scoring")
            final = tuple(
                dict.fromkeys(
                    case.parent_by_block[hit.block.block_id]
                    for hit in result.hits
                    if hit.block.block_id in case.parent_by_block
                )
            )
            path = result_path(output, arm, case.case_id)
            admitted = None
            if path.is_file():
                generated = json.loads(path.read_text(encoding="utf-8"))
                if generated.get("execution") == "ok":
                    ids = generated.get("result", {}).get("included_block_ids", [])
                    admitted = tuple(
                        dict.fromkeys(
                            case.parent_by_block[x] for x in ids if x in case.parent_by_block
                        )
                    )
            pool_metrics = reference_metrics(candidate_ids, references, k=len(candidate_ids) or 1)
            final_metrics = reference_metrics(final, references, k=final_k)
            admitted_metrics = (
                reference_metrics(admitted, references, k=len(admitted) or 1)
                if admitted is not None
                else None
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    **final_metrics,
                    "stages": {
                        "candidate_pool": pool_metrics,
                        "final_retrieval": final_metrics,
                        "admitted_context": admitted_metrics,
                    },
                }
            )
        count = len(rows)
        scores[arm] = {
            "questions": count,
            "mean_union_recall": (
                sum(row["union_recall"] for row in rows) / count if count else None
            ),
            "mean_best_reference_recall": (
                sum(row["best_reference_recall"] for row in rows) / count if count else None
            ),
            "complete_reference_rate": (
                sum(row["any_reference_complete"] for row in rows) / count if count else None
            ),
            "per_question": rows,
        }
    return {
        "scope": (
            f"QASPER answerable cases; {unanswerable} no-answer cases excluded from evidence recall"
        ),
        "metric": (
            f"QASPER reference-aware recall in shared top-{RERANK_CANDIDATE_K} pool, "
            f"final top-{final_k}, and admitted context"
        ),
        "arms": scores,
        "interpretation_limit": (
            "Evidence annotations may be incomplete and do not judge semantic support."
        ),
    }


def _answer_characters(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(char for char in normalized if char.isalnum())


def character_f1(prediction: str, reference: str) -> float:
    predicted = _answer_characters(prediction)
    expected = _answer_characters(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    return 2 * overlap / (len(predicted) + len(expected))


def retrieval_summary(case: Case, *, cutoffs: tuple[int, ...] = (3, 5, 10)) -> dict:
    if case.a_result is None:
        raise ValueError("the frozen hybrid result is unavailable")
    ranked_ids = [
        case.parent_by_block[hit.block.block_id]
        for hit in case.a_result.hits
        if hit.block.block_id in case.parent_by_block
    ]
    gold = set(case.gold_ids)
    if not gold:
        raise ValueError("retrieval metrics require at least one gold source ID")
    first_relevant = next((rank for rank, item in enumerate(ranked_ids, 1) if item in gold), None)
    per_cutoff = {}
    for cutoff in cutoffs:
        selected = ranked_ids[:cutoff]
        matched = len(set(selected) & gold)
        per_cutoff[str(cutoff)] = {
            "recall": matched / len(gold),
            "any_relevant": matched > 0,
            "complete_support_set": gold <= set(selected),
        }
    return {
        "gold_source_count": len(gold),
        "mrr": 1 / first_relevant if first_relevant else 0.0,
        "first_relevant_rank": first_relevant,
        "at_k": per_cutoff,
    }


def score_auxiliary_cases(output: Path, cases: dict[str, Case], manifest: dict) -> dict:
    """Report benchmark answer overlap and supporting-source retrieval separately."""
    xrag_cases = [item for item in manifest["cases"] if item["kind"] == "xrag"]
    xrag_rows = load_xrag_rows(manifest) if xrag_cases else {}
    drcd_cases = [item for item in manifest["cases"] if item["kind"] == "drcd"]
    drcd_rows = {}
    if drcd_cases:
        drcd_path = ROOT / manifest["drcd"]["path"]
        data = json.loads(drcd_path.read_text(encoding="utf-8"))
        qa_by_id = {
            qa["id"]: qa
            for article in data["data"]
            for paragraph in article["paragraphs"]
            for qa in paragraph["qas"]
        }
        drcd_rows = {item["id"]: qa_by_id[item["qa_id"]] for item in drcd_cases}

    arms = tuple(manifest.get("arms", ("A",)))
    dataset_cases = {"xrag": xrag_cases, "drcd": drcd_cases}
    summaries = {}
    for dataset, items in dataset_cases.items():
        if not items:
            continue
        arm_rows = {arm: [] for arm in arms}
        retrieval_rows = []
        for item in items:
            case = cases[item["id"]]
            retrieval = retrieval_summary(case)
            retrieval_rows.append(retrieval)
            if dataset == "xrag":
                row = xrag_rows[(item["task"], item["dataset_language"], item["example_id"])]
                references = (row[item["answer_field"]],)
            else:
                row = drcd_rows[item["id"]]
                references = tuple(answer["text"] for answer in row["answers"])
            for arm in arms:
                stored_arm = "B1" if arm == "B1_fallback_served" else arm
                fallback_only = arm == "B1_fallback_served"
                path = result_path(output, stored_arm, item["id"])
                record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
                if fallback_only:
                    fallback = record.get("fallback", {})
                    result = fallback.get("result", {}) if fallback.get("execution") == "ok" else {}
                    execution = (
                        "fallback_served" if fallback.get("execution") == "ok" else "missing"
                    )
                else:
                    result = record.get("result", {}) if record.get("execution") == "ok" else {}
                    execution = record.get("execution", "missing")
                prediction = result.get("answer", "")
                arm_rows[arm].append(
                    {
                        "case_id": item["id"],
                        "direction": item.get("direction", "zh-zh"),
                        "answer_language": item.get("language"),
                        "paired_base_id": item.get("paired_base_id"),
                        "query_translation": item.get("query_translation", False),
                        "execution": execution,
                        "answer_exact_match": any(
                            _answer_characters(prediction) == _answer_characters(reference)
                            for reference in references
                        ),
                        "answer_character_f1": max(
                            (character_f1(prediction, reference) for reference in references),
                            default=0.0,
                        ),
                    }
                )

        arm_summaries = {}
        for arm, rows in arm_rows.items():
            directions = {}
            for direction in dict.fromkeys(row["direction"] for row in rows):
                group = [row for row in rows if row["direction"] == direction]
                directions[direction] = {
                    "questions": len(group),
                    "execution_successes": sum(row["execution"] == "ok" for row in group),
                    "exact_match": sum(row["answer_exact_match"] for row in group) / len(group),
                    "character_f1": sum(row["answer_character_f1"] for row in group) / len(group),
                }
            arm_summaries[arm] = {
                "execution_successes": sum(row["execution"] == "ok" for row in rows),
                "exact_match": sum(row["answer_exact_match"] for row in rows) / len(rows),
                "character_f1": sum(row["answer_character_f1"] for row in rows) / len(rows),
                "by_direction": directions,
                "per_case": rows,
            }
        summaries[dataset] = {
            "questions": len(items),
            "arms": arm_summaries,
            "shared_retrieval": {
                "mean_recall_at_3": sum(row["at_k"]["3"]["recall"] for row in retrieval_rows)
                / len(retrieval_rows),
                "complete_support_at_10": sum(
                    row["at_k"]["10"]["complete_support_set"] for row in retrieval_rows
                )
                / len(retrieval_rows),
                "per_case": retrieval_rows,
            },
        }
    return {
        "metric_scope": (
            "Answer exact/character overlap is a lexical diagnostic, not semantic correctness. "
            "Answer references follow each frozen case's requested output language. Retrieval "
            "labels are dataset support annotations, not exhaustive evidence judgments."
        ),
        "datasets": summaries,
        "translation_pairs_are_not_independent": True,
    }


def q4_context_preflight(cases, model, budget_config, *, final_k):
    """Freeze both Q4 contexts with the same exact source and token admission path."""
    from page_cite_rag.agent import _admit, fixed_section_supplement

    capped_model = InputCappedModel(model)
    ready = {}
    rows = {}
    for case in cases.values():
        if case.a_result is None or not case.source_blocks:
            raise ValueError("Q4 retrieval did not prepare every frozen case")
        workflow = document_workflow(
            case.store,
            case.document_id,
            FrozenSearchRetriever(case.a_result, top_k=final_k),
            capped_model,
            budget_config,
            top_k=final_k,
        )
        retrieved = workflow._read_hits(case.a_result, case.question)
        base_blocks = workflow._context_blocks(case.a_result, retrieved)
        base = _admit(
            workflow,
            case.question,
            case.language,
            base_blocks,
            case.a_result.context_groups,
        )
        fixed, fixed_trace = fixed_section_supplement(
            workflow,
            case.question,
            case.language,
            case.a_result,
            base_blocks,
        )
        ready[case.case_id] = {
            "workflow": workflow,
            "base_blocks": base_blocks,
            "agent_context": base,
            "fixed_context": fixed,
            "fixed_trace": fixed_trace,
        }
        rows[f"{case.case_id}/fixed"] = {
            "eligible": True,
            "exact_input_tokens": fixed.exact_input_tokens,
            "admitted_block_ids": [block.block_id for block in fixed.blocks],
            "context_group_ids": [group.chunk_index for group in fixed.context_groups],
            "retrieval_block_count": len(base_blocks),
            "added_block_ids": fixed_trace.get("admitted_block_ids", []),
            "policy": fixed_trace,
            "prompt_version": workflow.prompt_version,
        }
        rows[f"{case.case_id}/agent"] = {
            "eligible": True,
            "exact_input_tokens": base.exact_input_tokens,
            "admitted_block_ids": [block.block_id for block in base.blocks],
            "context_group_ids": [group.chunk_index for group in base.context_groups],
            "retrieval_block_count": len(base_blocks),
            "dynamic_reads_checked_at_runtime": True,
            "prompt_version": workflow.prompt_version,
        }
    return ready, rows


def q4_quality_summary(output: Path, cases: list[dict], split: str) -> dict:
    """Return official-compatible lexical scores and paired operational diagnostics."""
    question_ids = {case["question_id"] for case in cases}
    gold = read_answer_references(ROOT / "data/qasper", split, question_ids)
    per_case = []
    arm_stats = {
        arm: {
            "successful_slots": 0,
            "model_calls_attempted": 0,
            "agent_model_calls_attempted": 0,
            "tool_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cooldown_seconds": 0.0,
            "answer_elapsed_ms": [],
        }
        for arm in ("fixed", "agent")
    }
    for case in cases:
        row = {"case_id": case["id"], "question_id": case["question_id"], "arms": {}}
        for arm in ("fixed", "agent"):
            path = result_path(output, arm, case["id"])
            record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            stats = arm_stats[arm]
            stats["successful_slots"] += int(record.get("execution") == "ok")
            stats["model_calls_attempted"] += record.get("completion_calls", 0)
            stats["agent_model_calls_attempted"] += record.get("agent_model_calls_attempted", 0)
            stats["tool_calls"] += record.get("agent_tool_calls", 0)
            stats["input_tokens"] += record.get("input_tokens", 0)
            stats["output_tokens"] += record.get("output_tokens", 0)
            stats["cooldown_seconds"] += record.get("cooldown_seconds_total", 0.0)
            if record.get("execution") == "ok":
                stats["answer_elapsed_ms"].append(record.get("answer_elapsed_ms", 0.0))
                result = record.get("result", {})
                claims = result.get("answer_claims", []) + result.get("detail_claims", [])
                cited = []
                for claim in claims:
                    for citation in claim.get("citations", []):
                        quote = citation.get("quote")
                        if isinstance(quote, str) and quote not in cited:
                            cited.append(quote)
                prediction = AnswerPrediction(result.get("answer", ""), tuple(cited))
                lexical = score_prediction(gold[case["question_id"]], prediction)
                row["arms"][arm] = {
                    "execution": "ok",
                    "status": result.get("status"),
                    "answer_f1": lexical["answer_f1"],
                    "evidence_f1": lexical["evidence_f1"],
                    "answer_elapsed_ms": record.get("answer_elapsed_ms"),
                    "answer_input_tokens": record.get("answer_input_tokens"),
                    "context_block_count": len(result.get("included_block_ids", [])),
                    "agent_model_calls": record.get("agent_model_calls_attempted", 0),
                    "agent_tool_calls": record.get("agent_tool_calls", 0),
                    "agent_trace": record.get("agent_trace"),
                    "fixed_context_trace": record.get("fixed_context_trace"),
                }
            else:
                row["arms"][arm] = {
                    "execution": record.get("execution", "missing"),
                    "error_type": record.get("error_type"),
                }
        fixed = row["arms"].get("fixed", {})
        agent = row["arms"].get("agent", {})
        row["paired_delta"] = (
            {
                "answer_f1_agent_minus_fixed": agent["answer_f1"] - fixed["answer_f1"],
                "evidence_f1_agent_minus_fixed": agent["evidence_f1"] - fixed["evidence_f1"],
            }
            if fixed.get("execution") == agent.get("execution") == "ok"
            else None
        )
        per_case.append(row)

    for stats in arm_stats.values():
        values = stats.pop("answer_elapsed_ms")
        stats["answer_latency_p50_ms"] = float(np.median(values)) if values else None
        stats["answer_latency_max_ms"] = max(values) if values else None
    return {
        "scope": "eight exposed QASPER validation questions; paired development diagnostic",
        "metrics": "QASPER official-compatible Answer F1 / Evidence F1; lexical overlap only",
        "not_semantic_quality_validation": True,
        "arms": arm_stats,
        "paired_successes": sum(row["paired_delta"] is not None for row in per_case),
        "per_case": per_case,
    }


def execute_q4(args, manifest, static, output, old_config, started) -> dict:
    """Run one frozen fixed-read versus bounded-agent evaluation, without hidden retries."""
    with tempfile.TemporaryDirectory(prefix="page-cite-rag-q4-") as temporary:
        store = SQLiteDocumentStore(Path(temporary) / "sources.sqlite3")
        slot = _EmbeddingSlot()
        encoder = NativeEmbeddingEncoder(slot, DEFAULT_MODEL_DIR, batch_size=4)
        with NativeEmbeddingServer(DEFAULT_GGUF_PATH, GPU_UUID) as embed_server:
            slot.active = embed_server
            cases = prepare_inputs(manifest, store)
            rankings = rank_cases(cases, encoder, final_k=args.final_k)
            embedding_samples = list(embed_server.samples)
            slot.active = None
        if not embedding_samples:
            raise RuntimeError("embedding telemetry was not recorded")
        if time.monotonic() - started > MAX_WALL_SECONDS:
            raise TimeoutError("Q4 retrieval preparation exceeded the 50-minute wall limit")

        config = LlamaConfig(
            bundle_dir=DEFAULT_BUNDLE,
            device="cuda",
            gpu_uuid=GPU_UUID,
            num_ctx=CONTEXT_TOKENS,
            num_predict=OUTPUT_TOKENS,
            template_reserve=TEMPLATE_RESERVE,
            timeout_seconds=120,
            max_gpu_memory_mib=12_000,
            max_gpu_temperature_c=80,
        )
        with LlamaModelClient(
            config,
            budget_policy="exact",
            recover_output_errors=False,
            sampling_profile="greedy",
        ) as model:
            contexts, preflight = q4_context_preflight(cases, model, config, final_k=args.final_k)
            full = {
                "generator_identity": model.identity,
                "embedding_identity": encoder.identity,
                "arms": list(manifest["arms"]),
                "case_count": len(cases),
                "eligible_completion_slots": len(cases) * len(manifest["arms"]),
                "preflight": preflight,
                "agent_max_calls_per_case": 3,
            }
            if old_config and old_config.get("full") != full:
                raise ValueError("Q4 resume preflight differs; refusing to mix run artifacts")
            output.mkdir(mode=0o700, parents=True, exist_ok=True)
            atomic_json(output / "configuration.json", {"static": static, "full": full})
            atomic_json(output / "retrieval.json", rankings)
            atomic_json(output / "preflight.json", preflight)
            if args.preflight_only:
                tokens = [row["exact_input_tokens"] for row in preflight.values()]
                return {
                    "preflight_only": True,
                    "cases": len(cases),
                    "arms": list(manifest["arms"]),
                    "completion_slots": len(cases) * 2,
                    "input_token_min": min(tokens) if tokens else None,
                    "input_token_max": max(tokens) if tokens else None,
                    "embedding_peak_mib": max(row["memory_mib"] for row in embedding_samples),
                    "embedding_peak_c": max(row["temperature_c"] for row in embedding_samples),
                    "wall_seconds": time.monotonic() - started,
                }

            from page_cite_rag.agent import (
                AgentRunFailure,
                local_chat_model,
                run_bounded_reading_agent,
            )
            from page_cite_rag.workflows import RetrievalTrace

            generation_started = time.monotonic()
            ordered_cases = list(cases.values())
            for index, case in enumerate(ordered_cases):
                ordered_arms = ("fixed", "agent") if index % 2 == 0 else ("agent", "fixed")
                ready = contexts[case.case_id]
                workflow = ready["workflow"]
                for arm in ordered_arms:
                    target = result_path(output, arm, case.case_id)
                    if target.exists():
                        record = json.loads(target.read_text(encoding="utf-8"))
                        if record.get("arm") != arm or record.get("case_id") != case.case_id:
                            raise ValueError("existing Q4 result identity differs from its path")
                        continue
                    if time.monotonic() - started > MAX_WALL_SECONDS:
                        raise TimeoutError("the 50-minute Q4 batch wall limit has been reached")
                    if time.monotonic() - generation_started > MAX_GENERATION_SECONDS:
                        raise TimeoutError("the 40-minute Q4 generation limit has been reached")
                    call_started = time.monotonic()
                    agent_run = None
                    final_attempted = False
                    cooldown_total = 0.0
                    cooldown_failures = []
                    try:

                        def cooled() -> float:
                            if time.monotonic() - started > MAX_WALL_SECONDS:
                                raise TimeoutError(
                                    "the 50-minute Q4 batch wall limit has been reached"
                                )
                            if time.monotonic() - generation_started > MAX_GENERATION_SECONDS:
                                raise TimeoutError(
                                    "the 40-minute Q4 generation limit has been reached"
                                )
                            try:
                                return wait_for_gpu_cool(model)
                            except Exception as exc:
                                cooldown_failures.append(type(exc).__name__)
                                raise

                        if arm == "fixed":
                            selected = ready["fixed_context"]
                            fixed_trace = ready["fixed_trace"]
                            agent_model_calls = agent_tool_calls = 0
                            agent_trace = None
                            agent_input_tokens = agent_output_tokens = 0
                            cooldown_times = []
                        else:
                            with local_chat_model(model) as chat_model:
                                agent_run = run_bounded_reading_agent(
                                    workflow,
                                    case.question,
                                    case.language,
                                    case.a_result,
                                    ready["base_blocks"],
                                    chat_model,
                                    cooldown=cooled,
                                )
                            selected = agent_run.context
                            fixed_trace = None
                            agent_trace = agent_run.trace
                            agent_model_calls = agent_trace["model_calls_attempted"]
                            agent_tool_calls = agent_trace["tool_calls_recorded"]
                            agent_input_tokens = agent_trace["agent_input_tokens"] or 0
                            agent_output_tokens = agent_trace["agent_output_tokens"] or 0
                            cooldown_times = agent_trace["cooldown_seconds"]
                        cooldown_total = sum(cooldown_times)
                        final_cooldown = cooled()
                        cooldown_total += final_cooldown
                        final_attempted = True
                        result = workflow.answer_from_context(
                            case.question,
                            selected.blocks,
                            language=case.language,
                            context_groups=selected.context_groups,
                        )
                        result = result.model_copy(
                            update={
                                "retrieval": RetrievalTrace(
                                    method=case.a_result.method,
                                    index_id=case.a_result.index_id,
                                    query=case.a_result.query,
                                    hit_count=len(case.a_result.hits),
                                    elapsed_ms=case.a_result.elapsed_ms,
                                    query_plan=workflow._query_plan(),
                                    branches=case.a_result.branches,
                                )
                            }
                        )
                        stats = result.generation_stats
                        record = {
                            "arm": arm,
                            "case_id": case.case_id,
                            "execution": "ok",
                            "completion_calls": agent_model_calls + 1,
                            "agent_model_calls_attempted": agent_model_calls,
                            "agent_tool_calls": agent_tool_calls,
                            "agent_input_tokens": agent_trace.get("agent_input_tokens")
                            if agent_trace
                            else None,
                            "agent_output_tokens": agent_trace.get("agent_output_tokens")
                            if agent_trace
                            else None,
                            "agent_trace": agent_trace,
                            "fixed_context_trace": fixed_trace,
                            "final_answer_attempted": True,
                            "cooldown_seconds_total": cooldown_total,
                            "answer_elapsed_ms": stats.elapsed_ms if stats else None,
                            "answer_input_tokens": result.budget.exact_input_tokens,
                            "input_tokens": (stats.prompt_eval_count if stats else 0)
                            + agent_input_tokens,
                            "output_tokens": (stats.eval_count if stats else 0)
                            + agent_output_tokens,
                            "llm_request": workflow._model.last_request,
                            "result": result.model_dump(mode="json"),
                        }
                    except Exception as exc:
                        failure_trace = exc.trace if isinstance(exc, AgentRunFailure) else None
                        cooldown_error = (
                            failure_trace.get("cooldown_error_type") if failure_trace else None
                        )
                        if cooldown_failures and cooldown_error is None:
                            cooldown_error = cooldown_failures[-1]
                        agent_attempts = (
                            failure_trace.get("model_calls_attempted", 0) if failure_trace else 0
                        )
                        record = {
                            "arm": arm,
                            "case_id": case.case_id,
                            "execution": "guard_stopped" if cooldown_error else "failed",
                            "completion_calls": agent_attempts + int(final_attempted),
                            "agent_model_calls_attempted": agent_attempts,
                            "agent_tool_calls": len(failure_trace.get("tool_trace", []))
                            if failure_trace
                            else 0,
                            "agent_trace": failure_trace,
                            "final_answer_attempted": final_attempted,
                            "cooldown_seconds_total": cooldown_total,
                            "error_type": (
                                cooldown_error
                                or (
                                    exc.error_type
                                    if isinstance(exc, AgentRunFailure)
                                    else type(exc).__name__
                                )
                            ),
                            "model_diagnostics": model.diagnostics(),
                            "elapsed_ms": (time.monotonic() - call_started) * 1000,
                        }
                        atomic_json(target, record)
                        if cooldown_error:
                            raise RuntimeError(
                                f"GPU cooldown guard stopped Q4: {cooldown_error}"
                            ) from None
                        raise
                    atomic_json(target, record)
                    atomic_json(output / "runtime-diagnostics.json", model.diagnostics())
                    print(
                        f"Q4 {case.case_id} {arm}: {record['execution']}, "
                        f"{record['completion_calls']} model calls, "
                        f"{record['answer_elapsed_ms']:.0f} ms final generation",
                        flush=True,
                    )
                    if run_size(output) > MAX_OUTPUT_BYTES:
                        raise RuntimeError("the 100 MiB Q4 artifact cap has been reached")

            diagnostics = model.diagnostics()
            records = [
                json.loads(path.read_text(encoding="utf-8"))
                for arm in manifest["arms"]
                for path in sorted((output / arm).glob("*.json"))
            ]
            completed = sum(record.get("execution") == "ok" for record in records)
            attempt = {
                "model_call_attempts": sum(record.get("completion_calls", 0) for record in records),
                "successful_slots": completed,
                "planned_slots": len(cases) * 2,
                "agent_model_call_attempts": sum(
                    record.get("agent_model_calls_attempted", 0) for record in records
                ),
                "agent_tool_calls": sum(record.get("agent_tool_calls", 0) for record in records),
                "input_tokens": sum(record.get("input_tokens", 0) for record in records),
                "output_tokens": sum(record.get("output_tokens", 0) for record in records),
                "cooldown_seconds_total": sum(
                    record.get("cooldown_seconds_total", 0.0) for record in records
                ),
                "answer_contract": "complete-cited-answer-v2",
                "generation_seconds": time.monotonic() - generation_started,
                "embedding_peak_mib": max(row["memory_mib"] for row in embedding_samples),
                "embedding_peak_c": max(row["temperature_c"] for row in embedding_samples),
                "gpu_samples": diagnostics.get("monitor_sample_count"),
                "gpu_observed_peak": diagnostics.get("observed_gpu_peak"),
                "wall_seconds": time.monotonic() - started,
                "no_retries": True,
            }
            atomic_json(output / "runtime-diagnostics.json", diagnostics)
            atomic_json(output / "attempt.json", attempt)
            qasper_cases = [item for item in manifest["cases"] if item["kind"] == "qasper"]
            answer_scores = score_qasper(
                output,
                qasper_cases,
                split=manifest["qasper"]["split"],
                arms=tuple(manifest["arms"]),
            )
            retrieval_scores = score_qasper_retrieval(
                cases,
                output,
                final_k=args.final_k,
                arms=tuple(manifest["arms"]),
            )
            quality_summary = q4_quality_summary(output, qasper_cases, manifest["qasper"]["split"])
            atomic_json(output / "qasper-scores.json", answer_scores)
            atomic_json(output / "qasper-retrieval-scores.json", retrieval_scores)
            atomic_json(output / "q4-comparison.json", quality_summary)
            return {
                "output": str(output),
                "model_call_attempts": attempt["model_call_attempts"],
                "successful_slots": completed,
                "planned_slots": attempt["planned_slots"],
                "qasper_scores": answer_scores,
                "quality_summary": quality_summary,
            }


def _q6_retrieval_trace(workflow, search_result):
    from page_cite_rag.workflows import RetrievalTrace

    return RetrievalTrace(
        method=search_result.method,
        index_id=search_result.index_id,
        query=search_result.query,
        hit_count=len(search_result.hits),
        elapsed_ms=search_result.elapsed_ms,
        query_plan=workflow._query_plan(),
        branches=search_result.branches,
    )


def _q6_output_validation_trace(error, arm: str) -> dict:
    """Serialize a failed structured completion and retain its raw output."""
    return {
        "failure_stage": "A1-final" if arm == "A1" else "B1-final",
        "error_type": type(error).__name__,
        "raw_model_output": error.raw_model_output,
        "validation_errors": list(error.validation_errors),
        "failure_accounting": error.accounting.model_dump(mode="json"),
    }


def _q6_stage_usage(arm: str, trace: dict | None, result, fallback_result=None) -> list[dict]:
    stages = []
    if arm == "A1":
        stats = result.generation_stats if result is not None else None
        if stats is not None:
            stages.append(
                {
                    "stage": "A1-final",
                    "calls": 1,
                    "status": "ok",
                    "input_tokens": stats.prompt_eval_count,
                    "output_tokens": stats.eval_count,
                    "elapsed_ms": stats.elapsed_ms,
                }
            )
        return stages
    trace = trace or {}
    selection = trace.get("selection")
    selection_stats = selection.get("generation_stats", {}) if isinstance(selection, dict) else {}
    failure = trace.get("failure_accounting", {})
    if selection_stats:
        stages.append(
            {
                "stage": "B1-selector",
                "calls": 1,
                "status": "ok",
                "input_tokens": selection_stats.get("prompt_eval_count", 0),
                "output_tokens": selection_stats.get("eval_count", 0),
                "elapsed_ms": selection_stats.get("elapsed_ms", 0),
            }
        )
    elif failure and trace.get("failure_stage") == "selector":
        stages.append(
            {
                "stage": "B1-selector",
                "calls": 1,
                "status": "failed",
                "input_tokens": failure.get("prompt_eval_count", 0),
                "output_tokens": failure.get("eval_count", 0),
                "elapsed_ms": failure.get("elapsed_ms", 0),
            }
        )
    final_stats = trace.get("final_generation_stats", {})
    if final_stats:
        stages.append(
            {
                "stage": "B1-final",
                "calls": 1,
                "status": "ok",
                "input_tokens": final_stats.get("prompt_eval_count", 0),
                "output_tokens": final_stats.get("eval_count", 0),
                "elapsed_ms": final_stats.get("elapsed_ms", 0),
            }
        )
    elif failure and trace.get("failure_stage") == "final_answer":
        stages.append(
            {
                "stage": "B1-final",
                "calls": 1,
                "status": "failed",
                "input_tokens": failure.get("prompt_eval_count", 0),
                "output_tokens": failure.get("eval_count", 0),
                "elapsed_ms": failure.get("elapsed_ms", 0),
            }
        )
    elif fallback_result is not None and fallback_result.generation_stats is not None:
        stats = fallback_result.generation_stats
        stages.append(
            {
                "stage": "A1-fallback-final",
                "calls": 1,
                "status": "ok",
                "input_tokens": stats.prompt_eval_count,
                "output_tokens": stats.eval_count,
                "elapsed_ms": stats.elapsed_ms,
            }
        )
    return stages


def _q6_aggregate(
    output: Path, case_ids: list[str], arms: tuple[str, ...], *, wall_seconds: float
) -> dict:
    records = []
    for arm in arms:
        for case_id in case_ids:
            path = result_path(output, arm, case_id)
            if path.is_file():
                records.append(json.loads(path.read_text(encoding="utf-8")))
    stages = [stage for record in records for stage in record.get("stage_usage", [])]
    ledger_path = output / "invocation-ledger.jsonl"
    ledger_events = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    completed_wall = sum(
        event.get("wall_seconds", 0.0)
        for event in ledger_events
        if event.get("event") in ("invocation_end", "invocation_stopped")
    )
    starts = {
        event["invocation_id"]: datetime.fromisoformat(event["started_utc"])
        for event in ledger_events
        if event.get("event") == "invocation_start"
    }
    ended = {
        event["invocation_id"]
        for event in ledger_events
        if event.get("event") in ("invocation_end", "invocation_stopped")
    }
    open_starts = [started for key, started in starts.items() if key not in ended]
    if open_starts:
        completed_wall += sum(
            max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
            for started in open_starts
        )
    unknown_usage_stages = sum(bool(stage.get("usage_unknown")) for stage in stages)
    return {
        "phase": "Q6",
        "completion_calls": sum(stage.get("calls", 0) for stage in stages),
        "planned_model_calls": len(case_ids)
        * (3 if set(arms) == {"A1", "B1"} else 2 if "B1" in arms else 1),
        "successful_slots": sum(record.get("execution") == "ok" for record in records),
        "fallback_served_slots": sum(
            record.get("fallback", {}).get("execution") == "ok" for record in records
        ),
        "failed_slots": sum(record.get("execution") == "failed" for record in records),
        "attempted_stage_count": len(stages),
        "input_tokens": sum(stage.get("input_tokens", 0) for stage in stages),
        "output_tokens": sum(stage.get("output_tokens", 0) for stage in stages),
        "unknown_usage_stages": unknown_usage_stages,
        "token_totals_complete": unknown_usage_stages == 0,
        "aggregate_tokens": sum(
            stage.get("input_tokens", 0) + stage.get("output_tokens", 0) for stage in stages
        ),
        "cooldown_seconds_total": sum(
            record.get("cooldown_seconds_total", 0) for record in records
        ),
        "unknown_cooldown_slots": sum(
            bool(record.get("cooldown_usage_unknown")) for record in records
        ),
        "no_retries": True,
        "this_invocation_wall_seconds": wall_seconds,
        "cross_resume_wall_seconds": completed_wall,
        "cross_resume_wall_complete": not any(
            event.get("wall_seconds_incomplete")
            for event in ledger_events
            if event.get("event") in ("invocation_end", "invocation_stopped")
        ),
        "cross_resume_wall_source": "invocation-ledger.jsonl",
    }


def _q6_latency_ms(record: dict) -> float:
    return sum(stage.get("elapsed_ms", 0.0) for stage in record.get("stage_usage", []))


def _q6_comparison_summary(
    output: Path,
    case_ids: list[str],
    arms: tuple[str, ...],
    qasper_scores: dict | None,
    auxiliary_scores: dict,
) -> dict:
    per_arm = {}
    records_by_arm = {arm: {} for arm in arms}
    for arm in arms:
        records = {}
        for case_id in case_ids:
            path = result_path(output, arm, case_id)
            if path.is_file():
                records[case_id] = json.loads(path.read_text(encoding="utf-8"))
        records_by_arm[arm] = records
        stages = [stage for record in records.values() for stage in record.get("stage_usage", [])]
        successful = [record for record in records.values() if record.get("execution") == "ok"]
        per_arm[arm] = {
            "planned_slots": len(case_ids),
            "recorded_slots": len(records),
            "successful_raw_slots": len(successful),
            "fallback_served_slots": sum(
                record.get("fallback", {}).get("execution") == "ok" for record in records.values()
            ),
            "failed_or_unusable_slots": sum(
                record.get("execution") != "ok" for record in records.values()
            ),
            "completion_calls": sum(stage.get("calls", 0) for stage in stages),
            "input_tokens": sum(stage.get("input_tokens", 0) for stage in stages),
            "output_tokens": sum(stage.get("output_tokens", 0) for stage in stages),
            "unknown_usage_stages": sum(bool(stage.get("usage_unknown")) for stage in stages),
            "token_totals_complete": not any(stage.get("usage_unknown") for stage in stages),
            "unknown_cooldown_slots": sum(
                bool(record.get("cooldown_usage_unknown")) for record in records.values()
            ),
            "total_tokens": sum(
                stage.get("input_tokens", 0) + stage.get("output_tokens", 0) for stage in stages
            ),
            "generation_seconds_p50": (
                float(np.percentile([_q6_latency_ms(record) / 1000 for record in successful], 50))
                if successful
                else None
            ),
            "generation_seconds_p95": (
                float(np.percentile([_q6_latency_ms(record) / 1000 for record in successful], 95))
                if successful
                else None
            ),
            "cooldown_seconds": sum(
                record.get("cooldown_seconds_total", 0.0) for record in records.values()
            ),
        }

    common_successes = [
        case_id
        for case_id in case_ids
        if all(records_by_arm[arm].get(case_id, {}).get("execution") == "ok" for arm in arms)
    ]
    paired_token_ratios = []
    paired_incremental_seconds = []
    if set(arms) == {"A1", "B1"}:
        for case_id in common_successes:
            a_record = records_by_arm["A1"][case_id]
            b_record = records_by_arm["B1"][case_id]
            a_tokens = a_record.get("input_tokens", 0) + a_record.get("output_tokens", 0)
            b_tokens = b_record.get("input_tokens", 0) + b_record.get("output_tokens", 0)
            if a_tokens:
                paired_token_ratios.append(b_tokens / a_tokens)
            paired_incremental_seconds.append(
                (_q6_latency_ms(b_record) - _q6_latency_ms(a_record)) / 1000
            )

    qasper = (qasper_scores or {}).get("scores", {})
    answer_deltas = []
    answer_wins = answer_losses = 0
    qasper_evidence_delta = None
    if "A1" in qasper and "B1" in qasper:
        a_questions = qasper["A1"].get("per_question", {})
        b_questions = qasper["B1"].get("per_question", {})
        for question_id in a_questions.keys() & b_questions.keys():
            delta = b_questions[question_id]["answer_f1"] - a_questions[question_id]["answer_f1"]
            answer_deltas.append(delta)
            answer_wins += delta >= 0.05
            answer_losses += delta <= -0.05
        a_metrics = qasper["A1"].get("metrics")
        b_metrics = qasper["B1"].get("metrics")
        if a_metrics and b_metrics:
            qasper_evidence_delta = b_metrics["Evidence F1"] - a_metrics["Evidence F1"]

    chinese_deltas = {}
    datasets = auxiliary_scores.get("datasets", {})
    for dataset, result in datasets.items():
        a_cases = {
            row["case_id"]: row
            for row in result.get("arms", {}).get("A1", {}).get("per_case", [])
            if row.get("answer_language") == "zh-Hant"
        }
        b_cases = {
            row["case_id"]: row
            for row in result.get("arms", {}).get("B1", {}).get("per_case", [])
            if row.get("answer_language") == "zh-Hant"
        }
        by_direction = {}
        for case_id in a_cases.keys() & b_cases.keys():
            direction = a_cases[case_id].get("direction", "unknown")
            delta = (
                b_cases[case_id]["answer_character_f1"] - a_cases[case_id]["answer_character_f1"]
            )
            by_direction.setdefault(direction, []).append(delta)
        chinese_deltas[dataset] = {
            direction: {
                "questions": len(deltas),
                "mean_character_f1_delta": sum(deltas) / len(deltas),
                "lexical_guard_pass": sum(deltas) / len(deltas) >= -0.05,
            }
            for direction, deltas in by_direction.items()
        }

    token_ratio_mean = (
        sum(paired_token_ratios) / len(paired_token_ratios) if paired_token_ratios else None
    )
    incremental_p95 = (
        float(np.percentile(paired_incremental_seconds, 95)) if paired_incremental_seconds else None
    )
    qasper_mean_delta = sum(answer_deltas) / len(answer_deltas) if answer_deltas else None
    raw_b1_failures = per_arm.get("B1", {}).get("failed_or_unusable_slots")
    mechanical_gates = {
        "b1_raw_failures_and_fallbacks_zero": (
            raw_b1_failures == 0 if raw_b1_failures is not None else None
        ),
        "qasper_answer_f1_wins_exceed_losses": (
            answer_wins > answer_losses if answer_deltas else None
        ),
        "qasper_mean_answer_f1_delta_nonnegative": (
            qasper_mean_delta >= 0 if qasper_mean_delta is not None else None
        ),
        "qasper_evidence_f1_delta_at_least_minus_005": (
            qasper_evidence_delta >= -0.05 if qasper_evidence_delta is not None else None
        ),
        "chinese_direction_lexical_guards_pass": (
            all(
                row["lexical_guard_pass"]
                for directions in chinese_deltas.values()
                for row in directions.values()
            )
            if any(chinese_deltas.values())
            else None
        ),
        "mean_paired_total_token_ratio_at_most_25": (
            token_ratio_mean <= 2.5 if token_ratio_mean is not None else None
        ),
        "p95_incremental_generation_seconds_at_most_30": (
            incremental_p95 <= 30 if incremental_p95 is not None else None
        ),
    }
    gates_applicable = [value for value in mechanical_gates.values() if value is not None]
    return {
        "scope": "Q6 fixed development slice; no independent validation claim",
        "arms": per_arm,
        "paired_successful_cases": len(common_successes),
        "paired_mean_total_token_ratio_b1_over_a1": token_ratio_mean,
        "paired_p95_incremental_generation_seconds": incremental_p95,
        "qasper_answer_f1": {
            "paired_wins_delta_at_least_005": answer_wins,
            "paired_losses_delta_at_most_minus_005": answer_losses,
            "mean_delta": qasper_mean_delta,
            "evidence_f1_delta": qasper_evidence_delta,
        },
        "zh_hant_character_f1_delta_by_dataset_and_direction": chinese_deltas,
        "mechanical_adoption_gates": mechanical_gates,
        "all_applicable_mechanical_gates_pass": (
            all(gates_applicable) if gates_applicable else None
        ),
        "source_review_required": True,
        "semantic_quality_limit": (
            "Answer F1, exact-evidence overlap and character F1 are benchmark/lexical proxies; "
            "they do not establish semantic correctness or safety."
        ),
    }


def execute_q6(args, manifest, static, output, old_config, started) -> dict:
    """Run the frozen shared-context A1/B1 comparison with explicit stage accounting."""
    from page_cite_rag.full_answer import EvidenceFirstRunFailure
    from page_cite_rag.generation import OutputValidationError

    invocation_id = str(time.time_ns())
    invocation_ledger = output / "invocation-ledger.jsonl"
    append_jsonl_record(
        invocation_ledger,
        {
            "event": "invocation_start",
            "invocation_id": invocation_id,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "resume": bool(args.resume),
            "smoke_only": bool(args.smoke_only),
        },
    )
    with tempfile.TemporaryDirectory(prefix="page-cite-rag-q6-") as temporary:
        store = SQLiteDocumentStore(Path(temporary) / "sources.sqlite3")
        slot = _EmbeddingSlot()
        encoder = NativeEmbeddingEncoder(slot, DEFAULT_MODEL_DIR, batch_size=4)
        smoke_ids = manifest.get("smoke_case_ids", [])
        selected_specs = (
            [case for case in manifest["cases"] if case["id"] == args.smoke_case_id]
            if args.smoke_case_id
            else [case for case in manifest["cases"] if case["id"] in smoke_ids]
            if args.smoke_only
            else manifest["cases"]
        )
        run_arms = (
            (args.smoke_arm,)
            if args.smoke_case_id
            else ("A1", "B1")
            if args.smoke_only
            else tuple(manifest["arms"])
        )
        run_manifest = {**manifest, "cases": selected_specs, "case_count": len(selected_specs)}
        with NativeEmbeddingServer(DEFAULT_GGUF_PATH, GPU_UUID) as embed_server:
            slot.active = embed_server
            cases = prepare_inputs(run_manifest, store)
            rankings = rank_cases(cases, encoder, final_k=RERANK_CANDIDATE_K)
            embedding_samples = list(embed_server.samples)
            slot.active = None
        if not embedding_samples:
            raise RuntimeError("Q6 embedding telemetry was not recorded")
        if time.monotonic() - started > MAX_WALL_SECONDS:
            raise TimeoutError("Q6 retrieval preparation exceeded the 50-minute wall limit")

        config = LlamaConfig(
            bundle_dir=DEFAULT_BUNDLE,
            device="cuda",
            gpu_uuid=GPU_UUID,
            num_ctx=CONTEXT_TOKENS,
            num_predict=OUTPUT_TOKENS,
            template_reserve=TEMPLATE_RESERVE,
            timeout_seconds=120,
            max_gpu_memory_mib=12_000,
            max_gpu_temperature_c=80,
        )
        with LlamaModelClient(
            config,
            budget_policy="exact",
            recover_output_errors=True,
            sampling_profile="greedy",
        ) as model:
            ready, admission = preflight_workflows(
                cases,
                model,
                config,
                final_k=RERANK_CANDIDATE_K,
                arms=run_arms,
                include_full_document=False,
                q6_shared_context=True,
            )
            full = {
                "generator_identity": model.identity,
                "embedding_identity": encoder.identity,
                "arms": list(run_arms),
                "case_count": len(cases),
                "eligible_completion_slots": len(cases) * len(run_arms),
                "planned_max_model_calls": len(cases)
                * (3 if set(run_arms) == {"A1", "B1"} else 2 if "B1" in run_arms else 1),
                "admission": admission,
                "smoke_only": bool(args.smoke_only),
                "targeted_smoke_case_id": args.smoke_case_id,
                "targeted_smoke_arm": args.smoke_arm if args.smoke_case_id else None,
                "executed_case_ids": list(cases),
            }
            if old_config and old_config.get("full") != full:
                raise ValueError("Q6 resume preflight differs; refusing to mix run artifacts")
            output.mkdir(mode=0o700, parents=True, exist_ok=True)
            saved_static = old_config.get("static", static) if old_config else static
            code_snapshots = (
                list(old_config.get("source_code_continuations", ())) if old_config else []
            )
            if not code_snapshots:
                code_snapshots.append(
                    {
                        "invocation_id": None,
                        "role": "original-static-snapshot",
                        "source_code_sha256": saved_static.get("source_code_sha256", {}),
                    }
                )
            current_code = static.get("source_code_sha256", {})
            if code_snapshots[-1].get("source_code_sha256") != current_code:
                code_snapshots.append(
                    {
                        "invocation_id": invocation_id,
                        "role": "failure-accounting-only-continuation",
                        "source_code_sha256": current_code,
                    }
                )
            atomic_json(
                output / "configuration.json",
                {
                    "static": saved_static,
                    "full": full,
                    "source_code_continuations": code_snapshots,
                },
            )
            atomic_json(output / "retrieval.json", rankings)
            atomic_json(output / "preflight.json", admission)
            case_ids = list(cases)
            if args.preflight_only:
                tokens = [
                    row["A1_final_input_tokens"]
                    for key, row in admission.items()
                    if key.endswith("/A1") and row.get("eligible")
                ]
                summary = {
                    "preflight_only": True,
                    "cases": len(cases),
                    "arms": list(run_arms),
                    "planned_max_model_calls": full["planned_max_model_calls"],
                    "input_token_min": min(tokens) if tokens else None,
                    "input_token_max": max(tokens) if tokens else None,
                    "embedding_peak_mib": max(row["memory_mib"] for row in embedding_samples),
                    "embedding_peak_c": max(row["temperature_c"] for row in embedding_samples),
                    "wall_seconds": time.monotonic() - started,
                }
                append_jsonl_record(
                    invocation_ledger,
                    {
                        "event": "invocation_end",
                        "invocation_id": invocation_id,
                        "ended_utc": datetime.now(timezone.utc).isoformat(),
                        "outcome": "preflight_only",
                        "wall_seconds": time.monotonic() - started,
                    },
                )
                return summary

            generation_started = time.monotonic()
            attempted = 0
            for index, case_id in enumerate(case_ids):
                case = cases[case_id]
                arm_order = (
                    (("A1", "B1") if index % 2 == 0 else ("B1", "A1"))
                    if len(run_arms) == 2
                    else run_arms
                )
                for arm in arm_order:
                    target = result_path(output, arm, case_id)
                    if target.exists():
                        previous = json.loads(target.read_text(encoding="utf-8"))
                        if previous.get("arm") != arm or previous.get("case_id") != case_id:
                            raise ValueError("Q6 result identity differs from its path")
                        continue
                    if time.monotonic() - started > MAX_WALL_SECONDS:
                        append_jsonl_record(
                            invocation_ledger,
                            {
                                "event": "invocation_stopped",
                                "invocation_id": invocation_id,
                                "ended_utc": datetime.now(timezone.utc).isoformat(),
                                "reason": "invocation_wall_limit",
                                "completion_calls_this_invocation": attempted,
                                "wall_seconds": time.monotonic() - started,
                            },
                        )
                        raise TimeoutError("Q6 invocation reached the 50-minute wall limit")
                    if time.monotonic() - generation_started > MAX_GENERATION_SECONDS:
                        append_jsonl_record(
                            invocation_ledger,
                            {
                                "event": "invocation_stopped",
                                "invocation_id": invocation_id,
                                "ended_utc": datetime.now(timezone.utc).isoformat(),
                                "reason": "invocation_generation_limit",
                                "completion_calls_this_invocation": attempted,
                                "wall_seconds": time.monotonic() - started,
                            },
                        )
                        raise TimeoutError("Q6 invocation reached the 40-minute generation limit")
                    context = ready[f"{case_id}/{arm}"]
                    workflow = context["workflow"]
                    prepared = context["prepared"]
                    result = None
                    fallback_result = None
                    trace = None
                    cooldown = []
                    guard_failures = []
                    call_started = time.monotonic()
                    calls = 0

                    def before_completion() -> float:
                        if time.monotonic() - started > MAX_WALL_SECONDS:
                            raise TimeoutError("Q6 invocation reached the 50-minute wall limit")
                        if time.monotonic() - generation_started > MAX_GENERATION_SECONDS:
                            raise TimeoutError(
                                "Q6 invocation reached the 40-minute generation limit"
                            )
                        try:
                            waited = wait_for_gpu_cool(model)
                        except Exception as exc:
                            guard_failures.append(type(exc).__name__)
                            raise
                        cooldown.append(waited)
                        return waited

                    try:
                        if arm == "A1":
                            before_completion()
                            calls = 1
                            result = workflow.answer_from_context(
                                case.question,
                                prepared["blocks"],
                                language=case.language,
                                context_groups=prepared["context_groups"],
                            )
                            result = result.model_copy(
                                update={
                                    "retrieval": _q6_retrieval_trace(
                                        workflow, context["search_result"]
                                    )
                                }
                            )
                        else:
                            try:
                                result, trace = workflow.ask_evidence_first(
                                    case.question,
                                    language=case.language,
                                    search_result=context["search_result"],
                                    source_blocks=context["blocks"],
                                    before_completion=before_completion,
                                )
                                calls = trace["completion_calls"]
                            except EvidenceFirstRunFailure as exc:
                                trace = exc.trace
                                calls = int(trace.get("completion_calls", 0))
                                cause = getattr(exc, "cause", exc)
                                diagnostics = model.diagnostics()
                                selector_output_failure = (
                                    trace.get("failure_stage") == "selector"
                                    and isinstance(cause, OutputValidationError)
                                    and not guard_failures
                                    and diagnostics.get("closed") is not True
                                    and not diagnostics.get("guard_errors")
                                )
                                if not selector_output_failure:
                                    raise
                                before_completion()
                                calls += 1
                                fallback_result = workflow.answer_from_context(
                                    case.question,
                                    prepared["blocks"],
                                    language=case.language,
                                    context_groups=prepared["context_groups"],
                                )
                                fallback_result = fallback_result.model_copy(
                                    update={
                                        "retrieval": _q6_retrieval_trace(
                                            workflow, context["search_result"]
                                        )
                                    }
                                )

                        stage_usage = _q6_stage_usage(arm, trace, result, fallback_result)
                        if arm == "B1" and fallback_result is not None:
                            trace["fallback"] = "A1 after selector schema/output failure"
                        input_tokens = sum(stage.get("input_tokens", 0) for stage in stage_usage)
                        output_tokens = sum(stage.get("output_tokens", 0) for stage in stage_usage)
                        record = {
                            "arm": arm,
                            "case_id": case_id,
                            "execution": "ok" if result is not None else "fallback",
                            "completion_calls": calls,
                            "cooldown_seconds": sum(cooldown),
                            "cooldown_seconds_total": sum(cooldown),
                            "elapsed_ms": (time.monotonic() - call_started) * 1000,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "stage_usage": stage_usage,
                            "evidence_first_trace": trace,
                            "result": (
                                result.model_dump(mode="json") if result is not None else None
                            ),
                            "fallback": (
                                {
                                    "execution": "ok",
                                    "reason": "selector_output_validation_failure",
                                    "result": fallback_result.model_dump(mode="json"),
                                }
                                if fallback_result is not None
                                else {"execution": "not_used"}
                            ),
                        }
                    except BaseException as exc:
                        failure_trace = getattr(exc, "trace", trace)
                        cause = getattr(exc, "cause", exc)
                        error_type = getattr(exc, "cause_type", type(cause).__name__)
                        diagnostics = model.diagnostics()
                        if isinstance(cause, OutputValidationError) and not failure_trace:
                            failure_trace = _q6_output_validation_trace(cause, arm)
                        stage_usage = _q6_stage_usage(arm, failure_trace, None)
                        # Direct A1 failures are recorded as a single attempted final call.
                        if arm == "A1" and calls and not stage_usage:
                            accounting = getattr(cause, "accounting", None)
                            stage_usage = [
                                {
                                    "stage": "A1-final",
                                    "calls": 1,
                                    "status": "failed",
                                    "input_tokens": (
                                        accounting.prompt_eval_count if accounting else 0
                                    ),
                                    "output_tokens": accounting.eval_count if accounting else 0,
                                    "elapsed_ms": accounting.elapsed_ms if accounting else 0,
                                }
                            ]
                        failure_calls = calls
                        if isinstance(failure_trace, dict):
                            failure_calls = max(
                                failure_calls, int(failure_trace.get("completion_calls", 0))
                            )
                        accounted_calls = sum(stage.get("calls", 0) for stage in stage_usage)
                        if failure_calls > accounted_calls:
                            stage_name = (
                                "A1-final"
                                if arm == "A1"
                                else "B1-selector"
                                if isinstance(failure_trace, dict)
                                and failure_trace.get("failure_stage") == "selector"
                                and accounted_calls == 0
                                else "B1-final"
                            )
                            stage_usage.append(
                                {
                                    "stage": stage_name,
                                    "calls": failure_calls - accounted_calls,
                                    "status": "failed_unaccounted",
                                    "input_tokens": 0,
                                    "output_tokens": 0,
                                    "usage_unknown": True,
                                }
                            )
                        failure_tokens_in = sum(
                            stage.get("input_tokens", 0) for stage in stage_usage
                        )
                        failure_tokens_out = sum(
                            stage.get("output_tokens", 0) for stage in stage_usage
                        )
                        safety_stop = (
                            bool(guard_failures)
                            or diagnostics.get("closed") is True
                            or bool(diagnostics.get("guard_errors"))
                            or isinstance(cause, TimeoutError)
                            or not isinstance(cause, Exception)
                        )
                        record = {
                            "arm": arm,
                            "case_id": case_id,
                            "execution": "guard_stopped" if safety_stop else "failed",
                            "completion_calls": failure_calls,
                            "cooldown_seconds": sum(cooldown),
                            "cooldown_seconds_total": sum(cooldown),
                            "input_tokens": failure_tokens_in,
                            "output_tokens": failure_tokens_out,
                            "stage_usage": stage_usage,
                            "evidence_first_trace": failure_trace,
                            "error_type": error_type,
                            "error": str(exc)[:500],
                            "model_diagnostics": diagnostics,
                            "elapsed_ms": (time.monotonic() - call_started) * 1000,
                            "fallback": {"execution": "not_served"},
                        }
                        if failure_calls or not safety_stop:
                            atomic_json(target, record)
                            append_jsonl_record(
                                invocation_ledger,
                                {
                                    "event": "case_attempt",
                                    "invocation_id": invocation_id,
                                    "case_id": case_id,
                                    "arm": arm,
                                    "execution": record["execution"],
                                    "completion_calls": failure_calls,
                                    "stage_usage": stage_usage,
                                    "elapsed_ms": record["elapsed_ms"],
                                },
                            )
                            attempted += failure_calls
                        if safety_stop:
                            append_jsonl_record(
                                invocation_ledger,
                                {
                                    "event": "invocation_stopped",
                                    "invocation_id": invocation_id,
                                    "ended_utc": datetime.now(timezone.utc).isoformat(),
                                    "reason": error_type,
                                    "completion_calls_this_invocation": attempted,
                                    "wall_seconds": time.monotonic() - started,
                                },
                            )
                            raise RuntimeError(
                                f"Q6 stopped by execution/resource guard: {error_type}"
                            ) from None
                        if diagnostics.get("closed") is True:
                            raise RuntimeError(
                                "Q6 model client closed; stop before touching unattempted slots"
                            ) from None
                        print(
                            f"Q6 {case_id} {arm}: failed ({error_type}); recorded without retry",
                            flush=True,
                        )
                        continue

                    atomic_json(target, record)
                    atomic_json(output / "runtime-diagnostics.json", model.diagnostics())
                    append_jsonl_record(
                        invocation_ledger,
                        {
                            "event": "case_attempt",
                            "invocation_id": invocation_id,
                            "case_id": case_id,
                            "arm": arm,
                            "execution": record["execution"],
                            "completion_calls": calls,
                            "stage_usage": stage_usage,
                            "elapsed_ms": record["elapsed_ms"],
                        },
                    )
                    attempted += calls
                    print(
                        f"Q6 {case_id} {arm}: {record['execution']}, {calls} model calls, "
                        f"{record['elapsed_ms']:.0f} ms",
                        flush=True,
                    )
                    if run_size(output) > MAX_OUTPUT_BYTES:
                        raise RuntimeError("Q6 reached the 100 MiB artifact cap")

            diagnostics = model.diagnostics()
            atomic_json(output / "runtime-diagnostics.json", diagnostics)
            attempt = _q6_aggregate(
                output, case_ids, run_arms, wall_seconds=time.monotonic() - started
            )
            attempt["phase"] = manifest.get("phase")
            attempt["embedding_peak_mib"] = max(row["memory_mib"] for row in embedding_samples)
            attempt["embedding_peak_c"] = max(row["temperature_c"] for row in embedding_samples)
            attempt["gpu_samples"] = diagnostics.get("monitor_sample_count")
            attempt["gpu_observed_peak"] = diagnostics.get("observed_gpu_peak")
            atomic_json(output / "attempt.json", attempt)

            qasper_cases = [item for item in run_manifest["cases"] if item["kind"] == "qasper"]
            answer_scores = (
                score_qasper(
                    output,
                    qasper_cases,
                    split=manifest["qasper"]["split"],
                    arms=(
                        *run_arms,
                        *(("B1_fallback_served",) if "B1" in run_arms else ()),
                    ),
                )
                if qasper_cases
                else None
            )
            retrieval_scores = (
                score_qasper_retrieval(
                    cases,
                    output,
                    final_k=RERANK_CANDIDATE_K,
                    arms=run_arms,
                )
                if qasper_cases
                else None
            )
            score_manifest = {**run_manifest, "arms": list(run_arms)}
            auxiliary_scores = (
                {"datasets": {}}
                if manifest.get("phase") == "Q6-confirmation"
                else score_auxiliary_cases(output, cases, score_manifest)
            )
            comparison = (
                {
                    "scope": "Eight newly unused QASPER validation questions; A1-only confirmation",
                    "arm": "A1",
                    "paired_against": None,
                    "interpretation_limit": (
                        "One small public-data confirmation sample; no B1 re-run, no independent "
                        "benchmark claim, and possible pretraining exposure."
                    ),
                }
                if manifest.get("phase") == "Q6-confirmation"
                else _q6_comparison_summary(
                    output, case_ids, run_arms, answer_scores, auxiliary_scores
                )
            )
            if answer_scores is not None:
                atomic_json(output / "qasper-scores.json", answer_scores)
                atomic_json(output / "qasper-retrieval-scores.json", retrieval_scores)
            if manifest.get("phase") == "Q6-confirmation":
                atomic_json(output / "confirmation-summary.json", comparison)
            else:
                atomic_json(output / "auxiliary-scores.json", auxiliary_scores)
                atomic_json(output / "q6-comparison.json", comparison)
            append_jsonl_record(
                invocation_ledger,
                {
                    "event": "invocation_end",
                    "invocation_id": invocation_id,
                    "ended_utc": datetime.now(timezone.utc).isoformat(),
                    "outcome": "completed",
                    "completion_calls_this_invocation": attempted,
                    "wall_seconds": time.monotonic() - started,
                },
            )
            return {
                "output": str(output),
                "invocation_id": invocation_id,
                "completion_calls_this_invocation": attempted,
                "summary": attempt,
                "qasper_scores": answer_scores,
                "qasper_retrieval_scores": retrieval_scores,
                "auxiliary_scores": auxiliary_scores,
                "comparison": comparison,
            }


def execute(args) -> dict:
    started = time.monotonic()
    if args.final_k >= RERANK_CANDIDATE_K:
        raise ValueError("final_k must be smaller than the frozen reranker candidate pool")
    manifest_path = getattr(args, "manifest", None)
    manifest = read_manifest(manifest_path)
    if getattr(args, "smoke_only", False) and manifest.get("phase") != "Q6":
        raise ValueError("--smoke-only is supported only by the frozen Q6 manifest")
    arms = tuple(manifest.get("arms", ("A", "R")))
    static = (
        static_configuration(manifest, final_k=args.final_k)
        if manifest_path is None
        else static_configuration(manifest, final_k=args.final_k, manifest_path=manifest_path)
    )
    output = args.output_dir.resolve()
    if output.is_relative_to(ROOT / "docs"):
        raise ValueError("results must use a supported output location")
    old_config = validate_output_target(output, args.resume, static)

    if manifest.get("phase") == "Q4":
        return execute_q4(args, manifest, static, output, old_config, started)
    if manifest.get("phase") in ("Q6", "Q6-confirmation"):
        if manifest.get("phase") == "Q6-confirmation" and (
            getattr(args, "smoke_only", False)
            or getattr(args, "smoke_case_id", None)
            or getattr(args, "smoke_arm", None)
        ):
            raise ValueError("Q6 confirmation has its own frozen eight-question candidate set")
        if getattr(args, "smoke_only", False) and args.resume:
            raise ValueError("Q6 smoke uses its own output directory; do not mix with resume")
        return execute_q6(args, manifest, static, output, old_config, started)

    with tempfile.TemporaryDirectory(prefix="page-cite-rag-quality-") as temporary:
        store = SQLiteDocumentStore(Path(temporary) / "sources.sqlite3")
        slot = _EmbeddingSlot()
        encoder = NativeEmbeddingEncoder(slot, DEFAULT_MODEL_DIR, batch_size=4)
        with NativeEmbeddingServer(DEFAULT_GGUF_PATH, GPU_UUID) as embed_server:
            slot.active = embed_server
            cases = prepare_inputs(manifest, store)
            rankings = rank_cases(cases, encoder, final_k=args.final_k)
            embedding_samples = list(embed_server.samples)
            slot.active = None
        if not embedding_samples:
            raise RuntimeError("embedding telemetry was not recorded")
        if time.monotonic() - started > MAX_WALL_SECONDS:
            raise TimeoutError("retrieval preparation exceeded the 50-minute wall limit")

        reranker_samples = []
        reranker_startup_ms = None
        if "R" in arms:
            with NativeRerankerServer(DEFAULT_RERANKER_PATH, GPU_UUID) as reranker:
                reranking = rerank_cases(cases, reranker, final_k=args.final_k)
                reranker_samples = list(reranker.samples)
                reranker_startup_ms = reranker.startup_ms
            if not reranker_samples:
                raise RuntimeError("reranker telemetry was not recorded")
            rankings.extend(reranking)
            if time.monotonic() - started > MAX_WALL_SECONDS:
                raise TimeoutError("retrieval and reranking exceeded the 50-minute wall limit")

        config = LlamaConfig(
            bundle_dir=DEFAULT_BUNDLE,
            device="cuda",
            gpu_uuid=GPU_UUID,
            num_ctx=CONTEXT_TOKENS,
            num_predict=OUTPUT_TOKENS,
            template_reserve=TEMPLATE_RESERVE,
            timeout_seconds=120,
            max_gpu_memory_mib=12_000,
            max_gpu_temperature_c=80,
        )
        with LlamaModelClient(
            config,
            budget_policy="exact",
            recover_output_errors=False,
            sampling_profile="greedy",
        ) as model:
            workflows, admission = preflight_workflows(
                cases,
                model,
                config,
                final_k=args.final_k,
                arms=arms,
                include_full_document=manifest.get("phase") not in ("Q3", "Q5"),
            )
            c_fit_count = sum(
                row.get("eligible") is True for key, row in admission.items() if key.endswith("/C")
            )
            scheduled = len(cases) * len(arms)
            planned_model_calls = len(cases) * 3 if manifest.get("phase") == "Q5" else scheduled
            full = {
                "generator_identity": model.identity,
                "embedding_identity": encoder.identity,
                "reranker_sha256": RERANKER_SHA256 if "R" in arms else None,
                "arms": list(arms),
                "case_count": len(cases),
                "eligible_completion_slots": scheduled,
                "admission": admission,
            }
            if old_config and old_config.get("full") != full:
                raise ValueError("resume preflight differs; refusing to mix run artifacts")
            output.mkdir(mode=0o700, parents=True, exist_ok=True)
            atomic_json(output / "configuration.json", {"static": static, "full": full})
            atomic_json(output / "retrieval.json", rankings)
            if "R" in arms:
                atomic_json(
                    output / "reranker.json",
                    {
                        "model_sha256": RERANKER_SHA256,
                        "startup_ms": reranker_startup_ms,
                        "candidate_pool_k": RERANK_CANDIDATE_K,
                        "temperature_peak_c": max(row["temperature_c"] for row in reranker_samples),
                        "memory_peak_mib": max(row["memory_mib"] for row in reranker_samples),
                        "cases": reranking,
                    },
                )
            atomic_json(output / "preflight.json", admission)
            if args.preflight_only:
                whole_document_applicable = manifest.get("phase") not in ("Q3", "Q5")
                eligible_tokens = [
                    row["exact_input_tokens"]
                    for row in admission.values()
                    if row.get("eligible") and row.get("exact_input_tokens") is not None
                ]
                return {
                    "preflight_only": True,
                    "cases": len(cases),
                    "arms": list(arms),
                    "whole_document_fit_cases": c_fit_count if whole_document_applicable else None,
                    "whole_document_not_fit_cases": (
                        len(cases) - c_fit_count if whole_document_applicable else None
                    ),
                    "completion_slots": scheduled,
                    "input_token_min": min(eligible_tokens) if eligible_tokens else None,
                    "input_token_max": max(eligible_tokens) if eligible_tokens else None,
                    "embedding_peak_mib": max(row["memory_mib"] for row in embedding_samples),
                    "embedding_peak_c": max(row["temperature_c"] for row in embedding_samples),
                    "reranker_peak_mib": (
                        max(row["memory_mib"] for row in reranker_samples)
                        if reranker_samples
                        else None
                    ),
                    "reranker_peak_c": (
                        max(row["temperature_c"] for row in reranker_samples)
                        if reranker_samples
                        else None
                    ),
                    "wall_seconds": time.monotonic() - started,
                }

            generation_started = time.monotonic()
            attempted = successful = 0
            input_tokens = output_tokens = 0
            cooldown_total = 0.0
            ordered_cases = list(cases.values())
            for index, case in enumerate(ordered_cases):
                case_arms = arms if len(arms) == 1 or index % 2 == 0 else tuple(reversed(arms))
                for arm in case_arms:
                    target = result_path(output, arm, case.case_id)
                    if target.exists():
                        record = json.loads(target.read_text(encoding="utf-8"))
                        if record.get("arm") != arm or record.get("case_id") != case.case_id:
                            raise ValueError("existing result identity differs from its path")
                        calls = record.get("completion_calls", 0)
                        attempted += calls
                        successful += int(record.get("execution") == "ok")
                        record_input_tokens, record_output_tokens = resumed_record_token_usage(
                            record
                        )
                        input_tokens += record_input_tokens
                        output_tokens += record_output_tokens
                        cooldown_total += record.get(
                            "cooldown_seconds_total", record.get("cooldown_seconds", 0.0)
                        )
                        continue
                    if time.monotonic() - started > MAX_WALL_SECONDS:
                        raise TimeoutError("the 50-minute batch wall limit has been reached")
                    if time.monotonic() - generation_started > MAX_GENERATION_SECONDS:
                        raise TimeoutError("the 40-minute generation limit has been reached")
                    cooled = 0.0
                    guard_failures = []

                    def before_completion() -> float:
                        if time.monotonic() - started > MAX_WALL_SECONDS:
                            raise TimeoutError("the 50-minute batch wall limit has been reached")
                        if time.monotonic() - generation_started > MAX_GENERATION_SECONDS:
                            raise TimeoutError("the 40-minute generation limit has been reached")
                        try:
                            return wait_for_gpu_cool(model)
                        except Exception as exc:
                            guard_failures.append(type(exc).__name__)
                            raise

                    if arm != "B":
                        try:
                            cooled = before_completion()
                        except BaseException as exc:
                            atomic_json(
                                target,
                                {
                                    "arm": arm,
                                    "case_id": case.case_id,
                                    "execution": "guard_stopped",
                                    "completion_calls": 0,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc)[:500],
                                    "model_diagnostics": model.diagnostics(),
                                },
                            )
                            raise
                        cooldown_total += cooled
                    workflow = workflows[f"{case.case_id}/{arm}"]
                    call_started = time.monotonic()
                    try:
                        evidence_trace = None
                        if arm == "B":
                            result, evidence_trace = workflow.ask_evidence_first(
                                case.question,
                                language=case.language,
                                before_completion=before_completion,
                            )
                            calls = evidence_trace["completion_calls"]
                            cooled = sum(evidence_trace["cooldown_seconds"])
                            cooldown_total += cooled
                            selection_stats = evidence_selection_generation_stats(evidence_trace)
                            final_stats = (
                                result.generation_stats.model_dump(mode="json")
                                if result.generation_stats
                                else {}
                            )
                            input_tokens_for_case = selection_stats.get("prompt_eval_count", 0)
                            input_tokens_for_case += final_stats.get("prompt_eval_count", 0)
                            output_tokens_for_case = selection_stats.get("eval_count", 0)
                            output_tokens_for_case += final_stats.get("eval_count", 0)
                        else:
                            result = workflow.ask(case.question, language=case.language)
                            calls = int(result.generation_stats is not None)
                            stats = result.generation_stats
                            input_tokens_for_case = (
                                stats.prompt_eval_count if stats is not None else 0
                            )
                            output_tokens_for_case = stats.eval_count if stats is not None else 0
                        record = {
                            "arm": arm,
                            "case_id": case.case_id,
                            "execution": "ok",
                            "completion_calls": calls,
                            "cooldown_seconds": cooled,
                            "cooldown_seconds_total": cooled,
                            "elapsed_ms": (time.monotonic() - call_started) * 1000,
                            "answer_elapsed_ms": result.elapsed_ms,
                            "input_tokens": input_tokens_for_case,
                            "output_tokens": output_tokens_for_case,
                            "result": result.model_dump(mode="json"),
                        }
                        if evidence_trace is not None:
                            record["evidence_first_trace"] = evidence_trace
                        if manifest.get("phase") != "Q5":
                            record["llm_request"] = workflow._model.last_request
                    except BaseException as exc:
                        failure_trace = getattr(exc, "trace", None)
                        error_type = getattr(exc, "cause_type", type(exc).__name__)
                        closed_before_generation = client_closed_before_generation(
                            error_type, str(exc)
                        )
                        failure_calls = (
                            failure_trace.get("completion_calls", 0)
                            if isinstance(failure_trace, dict)
                            else (0 if closed_before_generation else int(arm != "B"))
                        )
                        failure_cooldown = (
                            sum(failure_trace.get("cooldown_seconds", []))
                            if isinstance(failure_trace, dict)
                            else cooled
                        )
                        diagnostics = model.diagnostics()
                        client_closed = diagnostics.get("closed") is True
                        if closed_before_generation:
                            raise RuntimeError(
                                "generation client was closed before this slot; "
                                "no completion was issued, stop and resume this untouched slot"
                            ) from None
                        completion_stats = diagnostics.get("last_completion") or {}
                        failure_input_tokens = (
                            completion_stats.get("tokens_evaluated", 0) if failure_calls else 0
                        )
                        failure_output_tokens = (
                            completion_stats.get("tokens_predicted", 0) if failure_calls else 0
                        )
                        record = {
                            "arm": arm,
                            "case_id": case.case_id,
                            "execution": "guard_stopped" if guard_failures else "failed",
                            "completion_calls": failure_calls,
                            "cooldown_seconds": failure_cooldown,
                            "cooldown_seconds_total": failure_cooldown,
                            "input_tokens": failure_input_tokens,
                            "output_tokens": failure_output_tokens,
                            "evidence_first_trace": failure_trace,
                            "error_type": error_type,
                            "error": str(exc)[:500],
                            "model_diagnostics": diagnostics,
                            "elapsed_ms": (time.monotonic() - call_started) * 1000,
                        }
                        atomic_json(target, record)
                        attempted += failure_calls
                        input_tokens += failure_input_tokens
                        output_tokens += failure_output_tokens
                        if arm == "B":
                            cooldown_total += failure_cooldown
                        if guard_failures:
                            raise RuntimeError(
                                f"GPU or execution guard stopped the batch: {guard_failures[-1]}"
                            ) from None
                        if continue_after_case_failure(
                            manifest.get("phase"),
                            guard_failures,
                            client_closed=client_closed,
                            error=exc,
                        ):
                            print(
                                f"{attempted}/{planned_model_calls} model calls; "
                                f"{case.case_id} {arm}: failed ({type(exc).__name__}); "
                                "recorded without retry",
                                flush=True,
                            )
                            if run_size(output) > MAX_OUTPUT_BYTES:
                                raise RuntimeError("the 100 MiB artifact cap has been reached")
                            continue
                        if client_closed and manifest.get("phase") == "Q5":
                            raise RuntimeError(
                                "generation client closed after a recorded model failure; "
                                "stop before touching unattempted slots"
                            ) from None
                        raise
                    atomic_json(target, record)
                    atomic_json(output / "runtime-diagnostics.json", model.diagnostics())
                    successful += int(record["execution"] == "ok")
                    attempted += calls
                    stats = result.generation_stats
                    input_tokens += record.get("input_tokens", 0)
                    output_tokens += record.get("output_tokens", 0)
                    print(
                        f"{attempted}/{planned_model_calls} model calls; {case.case_id} {arm}: "
                        f"{record['execution']}, {record['elapsed_ms']:.0f} ms; "
                        f"cooled {cooled:.1f} s",
                        flush=True,
                    )
                    if run_size(output) > MAX_OUTPUT_BYTES:
                        raise RuntimeError("the 100 MiB artifact cap has been reached")

            diagnostics = model.diagnostics()
            atomic_json(output / "runtime-diagnostics.json", diagnostics)
            whole_document_applicable = manifest.get("phase") not in ("Q3", "Q5")
            atomic_json(
                output / "attempt.json",
                {
                    "completion_calls": attempted,
                    "planned_model_calls": (planned_model_calls),
                    "successful_slots": successful,
                    "planned_slots": scheduled,
                    "whole_document_fit_cases": c_fit_count if whole_document_applicable else None,
                    "whole_document_not_fit_cases": (
                        len(cases) - c_fit_count if whole_document_applicable else None
                    ),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "aggregate_tokens": input_tokens + output_tokens,
                    "cooldown_seconds_total": cooldown_total,
                    "answer_contract": (
                        "complete-cited-answer-v3"
                        if manifest.get("phase") == "Q5"
                        else "complete-cited-answer-v2"
                    ),
                    "generation_seconds": time.monotonic() - generation_started,
                    "embedding_peak_mib": max(row["memory_mib"] for row in embedding_samples),
                    "embedding_peak_c": max(row["temperature_c"] for row in embedding_samples),
                    "reranker_peak_mib": (
                        max(row["memory_mib"] for row in reranker_samples)
                        if reranker_samples
                        else None
                    ),
                    "reranker_peak_c": (
                        max(row["temperature_c"] for row in reranker_samples)
                        if reranker_samples
                        else None
                    ),
                    "reranker_startup_ms": reranker_startup_ms,
                    "gpu_samples": diagnostics.get("monitor_sample_count"),
                    "gpu_observed_peak": diagnostics.get("observed_gpu_peak"),
                    "wall_seconds": time.monotonic() - started,
                    "no_retries": True,
                },
            )
            qasper_cases = [item for item in manifest["cases"] if item["kind"] == "qasper"]
            answer_scores = None
            if qasper_cases:
                qasper_split = manifest.get("qasper", {}).get("split", "train")
                answer_scores = score_qasper(output, qasper_cases, split=qasper_split, arms=arms)
                atomic_json(output / "qasper-scores.json", answer_scores)
                atomic_json(
                    output / "qasper-retrieval-scores.json",
                    score_qasper_retrieval(cases, output, final_k=args.final_k, arms=arms),
                )
            auxiliary_scores = score_auxiliary_cases(output, cases, manifest)
            if auxiliary_scores["datasets"]:
                atomic_json(output / "crosslingual-scores.json", auxiliary_scores)
            return {
                "output": str(output),
                "completion_calls": attempted,
                "planned_model_calls": (planned_model_calls),
                "successful_slots": successful,
                "planned_slots": scheduled,
                "whole_document_fit_cases": c_fit_count if whole_document_applicable else None,
                "whole_document_not_fit_cases": (
                    len(cases) - c_fit_count if whole_document_applicable else None
                ),
                "qasper_scores": answer_scores,
                "auxiliary_scores": auxiliary_scores["datasets"],
            }


class _EmbeddingSlot:
    active: NativeEmbeddingServer | None = None

    def token_ids(self, text: str) -> list[int]:
        if self.active is None:
            raise RuntimeError("embedding server is not active")
        return self.active.token_ids(text)

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.active is None:
            raise RuntimeError("embedding server is not active")
        return self.active.embed(texts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/quality-convergence/q2-reranker-20261018-1",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="optional frozen evaluation manifest; omit to use the historical P0 manifest",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="run only the six frozen Q6 compatibility cases; requires the Q6 manifest",
    )
    parser.add_argument(
        "--smoke-case-id",
        choices=(
            "pdf-nist-core",
            "pdf-nist-measure",
            "pdf-sbert-objectives",
            "pdf-lost-manipulation",
            "xrag-zh-en-13",
            "qasper-noanswer-206739417251064b910ae9e5ff096e867ee10fb8",
        ),
        help="isolate one frozen smoke case after a technical fix; requires --smoke-arm",
    )
    parser.add_argument(
        "--smoke-arm",
        choices=("A1", "B1"),
        help="run only this arm for --smoke-case-id to avoid repeating the other arm",
    )
    parser.add_argument("--final-k", type=int, choices=(10,), default=DEFAULT_FINAL_K)
    args = parser.parse_args()
    try:
        if not args.score_only and not GPU_UUID:
            raise ValueError("set PAGE_CITE_RAG_GPU_UUID to an explicit GPU UUID")
        if args.smoke_only and (args.manifest is None or args.score_only):
            raise ValueError("--smoke-only requires --manifest with the frozen Q6 manifest")
        if bool(args.smoke_case_id) != bool(args.smoke_arm) or (
            args.smoke_case_id and not args.smoke_only
        ):
            raise ValueError(
                "--smoke-case-id and --smoke-arm must be used together with --smoke-only"
            )
        if args.score_only:
            manifest = read_manifest(args.manifest)
            output = args.output_dir.resolve()
            if not output.is_dir() or not (output / "configuration.json").is_file():
                raise ValueError("score-only requires an existing run directory")
            saved = json.loads((output / "configuration.json").read_text(encoding="utf-8"))
            static = saved.get("static", {})
            if manifest.get("phase") == "Q3":
                if (
                    static.get("manifest_sha256") != sha256(Q3_MANIFEST_PATH)
                    or static.get("qasper_revision") != manifest["qasper"]["revision"]
                    or static.get("qasper_sha256") != manifest["qasper"]["sha256"]
                    or not static.get("run_version", "").startswith("quality-convergence-q3-a-")
                    or set(static.get("arms", {})) != {"A"}
                ):
                    raise ValueError("score-only run does not match the frozen Q3 manifest")
                previous = output / "qasper-scores.json"
                if not previous.is_file():
                    raise ValueError("Q3's original score artifact is missing")
                qasper_cases = [item for item in manifest["cases"] if item["kind"] == "qasper"]
                scores = score_qasper(
                    output,
                    qasper_cases,
                    split=manifest["qasper"]["split"],
                    arms=("A",),
                )
                raw_hashes = [
                    {"case_id": case["id"], "sha256": sha256(result_path(output, "A", case["id"]))}
                    for case in qasper_cases
                    if result_path(output, "A", case["id"]).is_file()
                ]
                scores["rescore_lineage"] = {
                    "reason": (
                        "map explicit insufficient_evidence status to the official QASPER "
                        "Unanswerable label"
                    ),
                    "previous_score_sha256": sha256(previous),
                    "raw_answer_hashes_sha256": file_digest(raw_hashes),
                    "raw_answers_overwritten": False,
                }
                target = output / "qasper-scores-status-normalized-v3.json"
                if target.exists():
                    raise ValueError("the versioned Q3 rescore artifact already exists")
                atomic_json(target, scores)
                result = {
                    "output": str(output),
                    "new_score_artifact": str(target),
                    "previous_score_sha256": scores["rescore_lineage"]["previous_score_sha256"],
                    "qasper_scores": scores,
                }
            elif manifest.get("phase") in (None, "Q2"):
                if (
                    static.get("p0_sha256") != sha256(P0_PATH)
                    or static.get("qasper_revision") != manifest["qasper"]["revision"]
                    or static.get("qasper_train_sha256") != manifest["qasper"]["sha256"]
                    or not static.get("run_version", "").startswith("quality-convergence-q2-ar-")
                    or set(static.get("arms", {})) != {"A", "R"}
                ):
                    raise ValueError("score-only run does not match the frozen QASPER manifest")
                qasper_cases = [item for item in manifest["cases"] if item["kind"] == "qasper"]
                scores = score_qasper(output, qasper_cases)
                target = output / "qasper-scores-status-normalized-v3.json"
                if target.exists():
                    raise ValueError("the versioned QASPER rescore artifact already exists")
                atomic_json(target, scores)
                result = {
                    "output": str(output),
                    "new_score_artifact": str(target),
                    "qasper_scores": scores,
                }
            else:
                raise ValueError("score-only is supported only for frozen Q2 or Q3 runs")
        else:
            result = execute(args)
    except BaseException as exc:
        print(f"quality comparison stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
