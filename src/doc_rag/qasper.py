"""Pinned QASPER Parquet preparation and paragraph/evidence alignment.

Dataset: allenai/qasper (CC BY 4.0), Dasigi et al., 2021.
Only the explicit download operation uses the network. No dataset scripts run.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from doc_rag.models import Block, DocumentMetadata, ParsedUnit
from doc_rag.store import SQLiteDocumentStore

DATASET = "allenai/qasper"
REVISION = "16ac8fdf81e7e3ad213ead4b60fdcc9dc8ca40e9"
ALIGNMENT_VERSION = "whitespace-only-unique-complete-annotation-union-v1"
PARSER_VERSION = "qasper-original-paragraphs-v1"
SOURCES = {
    "train": {
        "filename": "qasper-train.parquet",
        "sha256": "9af08092ee26c4f700202c1f90d1592b662926f23f3a308a10ff0a53345e37fe",
        "size": 14374550,
    },
    "validation": {
        "filename": "qasper-validation.parquet",
        "sha256": "089781b91c337d348dd9e8b57cc8adc100ed2d9cab84a6127402bcccf1559222",
        "size": 4749127,
    },
}


class QasperError(RuntimeError):
    """The snapshot or its schema cannot support this evaluation."""


def source_url(split: str) -> str:
    return (
        f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"
        f"/qasper/{SOURCES[split]['filename']}"
    )


def snapshot_path(data_dir: Path, split: str) -> Path:
    return data_dir.expanduser() / REVISION / SOURCES[split]["filename"]


def verify_snapshot(path: Path, split: str) -> None:
    source = SOURCES[split]
    if path.is_symlink() or not path.is_file():
        raise QasperError(f"{split} snapshot is missing; run doc-rag eval download-qasper first")
    with path.open("rb") as file:
        checksum = hashlib.file_digest(file, "sha256").hexdigest()
    if path.stat().st_size != source["size"] or checksum != source["sha256"]:
        raise QasperError(f"{split} snapshot checksum/size mismatch; use a verified snapshot")


def download_qasper(data_dir: Path) -> dict:
    """Download the two fixed public files; never replace a mismatched local file."""
    files = []
    for split, source in SOURCES.items():
        target = snapshot_path(data_dir, split)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() and not target.is_symlink():
            with tempfile.TemporaryDirectory(prefix=".download-", dir=target.parent) as temporary:
                staging = Path(temporary) / source["filename"]
                request = urllib.request.Request(
                    source_url(split), headers={"User-Agent": "doc-rag-qasper-preparation/1"}
                )
                with (
                    urllib.request.urlopen(request, timeout=60) as response,
                    staging.open("xb") as out,
                ):
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > source["size"]:
                            raise QasperError(f"{split} download exceeded its pinned size")
                        out.write(chunk)
                verify_snapshot(staging, split)
                try:
                    os.link(staging, target)  # Exclusive publication on the same filesystem.
                except FileExistsError:
                    verify_snapshot(target, split)
        verify_snapshot(target, split)
        files.append({"split": split, "path": str(target), **source})
    return {"dataset": DATASET, "revision": REVISION, "license": "CC-BY-4.0", "files": files}


@dataclass(frozen=True)
class Passage:
    passage_id: str
    section_index: int
    paragraph_index: int
    text: str


@dataclass(frozen=True)
class Annotation:
    unanswerable: bool
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    annotations: tuple[Annotation, ...]


@dataclass(frozen=True)
class Paper:
    paper_id: str
    passages: tuple[Passage, ...]
    questions: tuple[Question, ...]


@dataclass(frozen=True)
class Alignment:
    gold_passage_ids: tuple[str, ...]
    annotation_issues: tuple[str, ...]
    exclusion_reason: str | None


def paper_from_row(row: dict) -> Paper:
    """Decode the pinned Arrow struct layout; questions/answers stay outside passages."""
    try:
        paper_id = row["id"]
        sections = row["full_text"]["paragraphs"]
        section_names = row["full_text"]["section_name"]
        qas = row["qas"]
        if (
            not isinstance(paper_id, str)
            or not paper_id
            or not isinstance(sections, list)
            or not isinstance(section_names, list)
            or len(sections) != len(section_names)
        ):
            raise ValueError("invalid paper identity or sections")
        passages = []
        for section_index, paragraphs in enumerate(sections):
            if not isinstance(paragraphs, list):
                raise ValueError("section paragraphs must be a list")
            for paragraph_index, text in enumerate(paragraphs):
                if not isinstance(text, str):
                    raise ValueError("paragraph is not text")
                if text.strip():
                    passages.append(
                        Passage(
                            f"{paper_id}/s{section_index:04d}/p{paragraph_index:04d}",
                            section_index,
                            paragraph_index,
                            text,
                        )
                    )
        questions = []
        if not all(isinstance(qas[key], list) for key in ("question_id", "question", "answers")):
            raise ValueError("question arrays must be lists")
        for question_id, text, references in zip(
            qas["question_id"], qas["question"], qas["answers"], strict=True
        ):
            if not isinstance(question_id, str) or not question_id or not isinstance(text, str):
                raise ValueError("invalid question identity or text")
            annotations = []
            if not isinstance(references["answer"], list):
                raise ValueError("answer annotations must be a list")
            for answer in references["answer"]:
                if (
                    type(answer["unanswerable"]) is not bool
                    or not isinstance(answer["evidence"], list)
                    or not all(isinstance(item, str) for item in answer["evidence"])
                ):
                    raise ValueError("invalid evidence annotation")
                annotations.append(Annotation(answer["unanswerable"], tuple(answer["evidence"])))
            questions.append(Question(question_id, text, tuple(annotations)))
        if len({question.question_id for question in questions}) != len(questions):
            raise ValueError("duplicate question IDs")
        return Paper(paper_id, tuple(passages), tuple(questions))
    except (KeyError, TypeError, ValueError) as exc:
        raise QasperError("unsupported QASPER row schema") from exc


def read_snapshot(data_dir: Path, split: str) -> Iterator[Paper]:
    path = snapshot_path(data_dir, split)
    verify_snapshot(path, split)
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise QasperError("Parquet reading requires uv sync --group eval") from exc
    seen = set()
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            paper = paper_from_row(row)
            if paper.paper_id in seen:
                raise QasperError("duplicate paper ID in QASPER snapshot")
            seen.add(paper.paper_id)
            yield paper


def normalize_evidence(text: str) -> str:
    return " ".join(text.split())


def align_question(paper: Paper, question: Question) -> Alignment:
    """Union complete, uniquely aligned text annotations; log excluded alternatives."""
    lookup: dict[str, list[str]] = defaultdict(list)
    for passage in paper.passages:
        lookup[normalize_evidence(passage.text)].append(passage.passage_id)
    gold: set[str] = set()
    issues: list[str] = []
    for annotation in question.annotations:
        if annotation.unanswerable:
            issues.append("unanswerable")
            continue
        evidence = {normalize_evidence(text) for text in annotation.evidence if text.strip()}
        if not evidence:
            issues.append("no_evidence")
            continue
        annotation_gold: set[str] = set()
        annotation_issues: set[str] = set()
        for text in evidence:
            if "FLOAT SELECTED" in text:
                annotation_issues.add("non_text_evidence")
            elif not lookup[text]:
                annotation_issues.add("unmatched_evidence")
            elif len(lookup[text]) > 1:
                annotation_issues.add("ambiguous_evidence")
            else:
                annotation_gold.add(lookup[text][0])
        if annotation_issues:
            issues.extend(sorted(annotation_issues))
        else:
            gold.update(annotation_gold)
    reason = None
    if not question.text.strip():
        reason = "empty_question"
    elif not paper.passages:
        reason = "no_passages"
    elif not gold:
        counts = Counter(issues)
        reason = next(
            (
                item
                for item in (
                    "ambiguous_evidence",
                    "unmatched_evidence",
                    "non_text_evidence",
                    "no_evidence",
                    "unanswerable",
                )
                if counts[item]
            ),
            "no_annotations",
        )
    return Alignment(tuple(sorted(gold)), tuple(issues), reason)


def ingest_paper(store: SQLiteDocumentStore, paper: Paper) -> tuple[str, dict[str, str]]:
    """Use the core store/index while keeping each original QASPER paragraph intact."""
    payload = json.dumps(
        {"paper_id": paper.paper_id, "passages": [(p.passage_id, p.text) for p in paper.passages]},
        ensure_ascii=False,
        sort_keys=True,
    )
    source_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    document_id = hashlib.sha256(f"{REVISION}/{PARSER_VERSION}/{source_hash}".encode()).hexdigest()
    document = DocumentMetadata(
        document_id=document_id,
        source_sha256=source_hash,
        source_name=paper.paper_id + ".qasper",
        media_type="application/vnd.qasper+json",
        parser_name="qasper",
        parser_version=PARSER_VERSION,
        language="en",
        usage_scope="QASPER CC-BY-4.0 evaluation",
    )
    source_text = "\n\n".join(passage.text for passage in paper.passages)
    unit = ParsedUnit(
        unit_index=1, text=source_text, status="ok" if source_text.strip() else "no_text"
    )
    blocks = []
    block_to_passage = {}
    offset = 0
    line_starts = [0]
    for line in source_text.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    for ordinal, passage in enumerate(paper.passages, start=1):
        block_id = hashlib.sha256(f"{document_id}/{passage.passage_id}".encode()).hexdigest()
        line_start = bisect_right(line_starts, offset)
        line_end = bisect_right(line_starts, offset + len(passage.text) - 1)
        blocks.append(
            Block(
                document_id=document_id,
                block_id=block_id,
                ordinal=ordinal,
                unit_index=1,
                line_start=line_start,
                line_end=line_end,
                char_start=offset,
                char_end=offset + len(passage.text),
                text=passage.text,
            )
        )
        block_to_passage[block_id] = passage.passage_id
        offset += len(passage.text) + 2

    def paragraph_factory(metadata: DocumentMetadata, source_unit: ParsedUnit, first: int):
        yield from blocks

    store.ingest(document, (unit,), block_factory=paragraph_factory)
    return document_id, block_to_passage
