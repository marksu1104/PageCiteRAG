"""Freeze a cross-lingual reading-evaluation slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path

from page_cite_rag.answer_metrics import read_answer_references
from page_cite_rag.qasper import align_question, read_snapshot, snapshot_path

ROOT = Path(__file__).resolve().parents[1]
SEED = "pagecite-reading-eval-20261020-v3"
OUTPUT_DEFAULT = ROOT / "outputs/reading-eval-20261020-v3"
XRAG_DEFAULT = OUTPUT_DEFAULT / "xrag-zh-mono-dev-v1"
LONG_BENCH_V2_FULL_DEFAULT = (
    ROOT / "outputs/quality-cycle-20261024/data/longbench-v2-single-document-qa-175"
)
XRAG = {
    "dataset": "AmazonScience/XRAG",
    "revision": "ead86612ac265e578ffee6f838c2180fec6428d9",
    "path": "xrag.dev.jsonl",
    "url": "https://huggingface.co/datasets/AmazonScience/XRAG/resolve/ead86612ac265e578ffee6f838c2180fec6428d9/xrag.dev.jsonl",
    "sha256": "7c7977b93ce205069b91fc701a4a5df10ec30055945869804589403bcd432a2c",
    "size": 250_082_145,
    "license": "CC-BY-NC-4.0",
}
LONG_BENCH_V1 = {
    "dataset": "zai-org/LongBench",
    "revision": "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
    "path": "data.zip",
    "url": "https://huggingface.co/datasets/zai-org/LongBench/resolve/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip",
    "sha256": "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64",
    "size": 113_932_529,
    "license": "not specified on the official dataset card; redistribution terms unresolved",
}
LONG_BENCH_V2 = {
    "dataset": "zai-org/LongBench-v2",
    "revision": "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
    "path": "data.json",
    "url": "https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json",
    "sha256": "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2",
    "size": 465_490_535,
    "license": "Apache-2.0",
}
QASPER = {
    "dataset": "allenai/qasper",
    "split": "validation",
    "revision": "16ac8fdf81e7e3ad213ead4b60fdcc9dc8ca40e9",
    "url": "https://huggingface.co/datasets/allenai/qasper/resolve/16ac8fdf81e7e3ad213ead4b60fdcc9dc8ca40e9/qasper/qasper-validation.parquet",
    "license": "CC-BY-4.0",
}
_ID_PATTERNS = (
    re.compile(rb"(?<![0-9a-f])[0-9a-f]{24}(?![0-9a-f])"),
    re.compile(rb"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])"),
    re.compile(rb"(?<![0-9a-f])[0-9a-f]{48}(?![0-9a-f])"),
)
_TEXT_ARTIFACT_SUFFIXES = {
    ".csv",
    ".diff",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".patch",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}
_FROZEN_INPUT_NAMES = {"gold.jsonl", "inputs.jsonl", "manifest.json"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256(payload.encode("utf-8"))


def _verify_source(path: Path, source: dict) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"pinned source file is missing or not regular: {path.name}")
    if path.stat().st_size != source["size"] or _file_sha256(path) != source["sha256"]:
        raise ValueError(f"pinned source checksum/size mismatch: {path.name}")


def _exposed_ids(output_dir: Path, known_ids: set[str]) -> tuple[set[str], int, int]:
    exposed: set[str] = set()
    files = 0
    frozen_manifest_files_skipped = 0
    frozen_input_parents: set[Path] = set()
    pattern_width = (
        max(
            int(match.group(1))
            for pattern in _ID_PATTERNS
            if (match := re.search(rb"\{([0-9]+)\}", pattern.pattern)) is not None
        )
        + 1
    )
    for path in output_dir.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() not in _TEXT_ARTIFACT_SUFFIXES:
            continue
        if path.name in _FROZEN_INPUT_NAMES and path.parent not in frozen_input_parents:
            manifest_path = path.parent / "manifest.json"
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    manifest = {}
                if manifest.get("state") == "frozen_inputs_not_run":
                    frozen_input_parents.add(path.parent)
        if path.name in _FROZEN_INPUT_NAMES and path.parent in frozen_input_parents:
            frozen_manifest_files_skipped += 1
            continue
        try:
            with path.open("rb") as source:
                carry = b""
                while chunk := source.read(1024 * 1024):
                    payload = carry + chunk
                    for pattern in _ID_PATTERNS:
                        exposed.update(
                            value.decode("ascii")
                            for value in pattern.findall(payload)
                            if value.decode("ascii") in known_ids
                        )
                    carry = payload[-pattern_width:]
        except OSError:
            continue
        files += 1
    return exposed, files, frozen_manifest_files_skipped


def _rank(kind: str, identifier: str, *, seed: str = SEED) -> str:
    return _sha256(f"{seed}:{kind}:{identifier}".encode("utf-8"))


def _xrag_exposed_ids(output_dir: Path, known_ids: set[int]) -> tuple[set[int], int]:
    pattern = re.compile(r"xrag(?:-[a-z0-9_]+)*-(\d+)(?:-[A-Z])?$", re.IGNORECASE)
    exposed: set[int] = set()
    scanned = 0
    for path in output_dir.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        scanned += 1
        match = pattern.search(path.stem)
        if match is not None and int(match.group(1)) in known_ids:
            exposed.add(int(match.group(1)))
    return exposed, scanned


def _xrag_bundle(row: dict) -> tuple[str, list[dict]]:
    articles = row.get("articles")
    order = row.get("article_order")
    if (
        not isinstance(articles, list)
        or len(articles) != 8
        or any(not isinstance(article, str) or not article.strip() for article in articles)
        or not isinstance(order, list)
        or sorted(order) != list(range(8))
    ):
        raise ValueError("XRAG row has an invalid article list or presentation order")

    pieces = []
    ranges = []
    offset = 0
    for presented_index, source_index in enumerate(order, start=1):
        heading = f"[Article {presented_index}]\n\n"
        pieces.append(heading)
        offset += len(heading)
        article = articles[source_index]
        article_text = article if article.endswith("\n") else article + "\n"
        start = offset
        pieces.append(article_text)
        offset += len(article_text)
        ranges.append(
            {
                "article_id": f"article-{presented_index}",
                "source_index": source_index,
                "char_start": start,
                "char_end": offset,
            }
        )
        pieces.append("\n\n")
        offset += 2
    return "".join(pieces), ranges


def _xrag_cases(
    rows: list[dict], exposed_ids: set[int], *, per_question_type: int = 3
) -> tuple[list[dict], list[dict], dict]:
    question_types = ("aggregation", "comparison", "multihop", "set")
    candidates = {question_type: [] for question_type in question_types}
    for row in rows:
        question_type = row.get("q_type")
        row_id = row.get("id")
        question = row.get("question")
        if (
            row.get("language") != "zh"
            or row.get("task") != "mono_retrieval"
            or question_type not in candidates
            or type(row_id) is not int
            or row_id in exposed_ids
            or not isinstance(question, str)
            or not question.strip()
        ):
            continue
        text, ranges = _xrag_bundle(row)
        article_hashes = {_sha256(article.encode("utf-8")) for article in row["articles"]}
        rank = _rank("xrag-zh-mono", str(row_id))
        candidates[question_type].append((rank, row, text, ranges, article_hashes))

    selected = []
    used_articles: set[str] = set()
    for question_type in question_types:
        count = 0
        for item in sorted(candidates[question_type], key=lambda value: value[0]):
            _rank_hash, _row, _text, _ranges, article_hashes = item
            if used_articles & article_hashes:
                continue
            selected.append((question_type, *item))
            used_articles.update(article_hashes)
            count += 1
            if count == per_question_type:
                break
        if count != per_question_type:
            raise ValueError("XRAG development split cannot supply the frozen stratified sample")

    inputs = []
    gold = []
    selected_metadata = []
    for question_type, rank, row, text, ranges, article_hashes in selected:
        row_id = row["id"]
        case_id = f"xrag-dev-zh-mono-{row_id}"
        inputs.append(
            {
                "case_id": case_id,
                "dataset": "xrag-zh-mono-crosslingual",
                "source_id": case_id,
                "language": "zh-Hant",
                "question": row["question"],
                "source": {
                    "format": "xrag_article_bundle",
                    "text": text,
                    "article_ranges": ranges,
                },
            }
        )
        support = row.get("is_support")
        if (
            not isinstance(support, list)
            or len(support) != 8
            or any(type(value) is not int or value not in (0, 1, 2) for value in support)
        ):
            raise ValueError("XRAG row has invalid support labels")
        support_by_article = {
            entry["article_id"]: support[entry["source_index"]] for entry in ranges
        }
        gold.append(
            {
                "case_id": case_id,
                "answer": row["answer"],
                "english_answer": row["en_answer"],
                "question_type": question_type,
                "crossdoc_type": row["crossdoc_type"],
                "article_support_levels": support_by_article,
            }
        )
        selected_metadata.append(
            {
                "case_id": case_id,
                "rank_sha256": rank,
                "question_type": question_type,
                "article_content_sha256": sorted(article_hashes),
                "character_count": len(text),
            }
        )
    return (
        inputs,
        gold,
        {
            "eligible_rows_by_question_type": {
                key: len(value) for key, value in candidates.items()
            },
            "selected": selected_metadata,
            "unique_article_contents": len(used_articles),
        },
    )


def freeze_xrag(args: argparse.Namespace) -> dict:
    output = args.output_dir.expanduser().resolve()
    source_path = args.xrag_jsonl.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("refusing to overwrite an existing frozen evaluation directory")
    _verify_source(source_path, XRAG)
    rows = []
    with source_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("language") == "zh" and row.get("task") == "mono_retrieval":
                rows.append(row)
    known_ids = {row["id"] for row in rows if type(row.get("id")) is int}
    exposed_ids, scanned_files = _xrag_exposed_ids(ROOT / "outputs", known_ids)
    inputs, gold, selection = _xrag_cases(rows, exposed_ids)
    input_bytes, gold_bytes = _jsonl_bytes(inputs), _jsonl_bytes(gold)
    gold_by_id = {row["case_id"]: row for row in gold}
    cases = []
    for row in inputs:
        source_bytes = row["source"]["text"].encode("utf-8")
        cases.append(
            {
                "case_id": row["case_id"],
                "dataset": row["dataset"],
                "source_id": row["source_id"],
                "language": row["language"],
                "question_type": gold_by_id[row["case_id"]]["question_type"],
                "question_sha256": _sha256(row["question"].encode("utf-8")),
                "source_sha256": _sha256(source_bytes),
                "gold_sha256": _canonical_hash(gold_by_id[row["case_id"]]),
            }
        )
    manifest = {
        "format": "pagecite-xrag-eval/v1",
        "created_date_utc": "2026-10-22",
        "seed": f"{SEED}:xrag-zh-mono-dev-v1",
        "state": "frozen_development_inputs_not_run",
        "dataset": XRAG,
        "attribution": (
            "Liu et al., XRAG: Cross-lingual Retrieval-Augmented Generation, EMNLP Findings 2025"
        ),
        "handling": "CC BY-NC 4.0; local non-commercial research only; retain attribution.",
        "selection_policy": (
            "Chinese monolingual-retrieval development rows; three hash-ranked rows per "
            "question type; exclude IDs exposed in prior XRAG-named output files; greedily "
            "require distinct article text hashes across selected questions. Selection does "
            "not inspect answer/support labels."
        ),
        "exposure_scan": {
            "xrag_output_files_scanned": scanned_files,
            "previously_exposed_zh_mono_ids_excluded": sorted(exposed_ids),
        },
        "selection": selection,
        "evaluation_scope": (
            "Cross-lingual multi-article stress diagnostic. Eight XRAG articles are bundled into "
            "one synthetic text input to exercise this single-source runner; not the official XRAG "
            "protocol, a single-PDF benchmark, or product-wide quality validation."
        ),
        "protocol_validation": {
            "source_task_label": "mono_retrieval",
            "article_languages_verified": False,
            "official_mono_retrieval_eligible": False,
            "reason": (
                "The source task label is not article-language validation. Original development "
                "rows can include non-English distractors; an input-side language audit is "
                "required before claiming the official all-English article setting."
            ),
        },
        "metrics": {
            "retrieval": (
                "Article-level support labels from XRAG is_support; diagnostic ranking "
                "metrics, not official XRAG retrieval scores."
            ),
            "answer": (
                "Local Chinese character-overlap F1 and exact match after script "
                "normalization; not the official XRAG LLM-judge metric."
            ),
            "citation": (
                "Cited article support level is an article-label proxy, not claim-level "
                "semantic support."
            ),
        },
        "gold_not_loaded_during_indexing_or_generation": True,
        "cases": cases,
        "files": {
            "inputs.jsonl": {
                "sha256": _sha256(input_bytes),
                "bytes": len(input_bytes),
                "rows": len(inputs),
            },
            "gold.jsonl": {
                "sha256": _sha256(gold_bytes),
                "bytes": len(gold_bytes),
                "rows": len(gold),
            },
        },
        "preparation_code_sha256": _file_sha256(Path(__file__).resolve()),
        "project_lock_sha256": _file_sha256(ROOT / "uv.lock"),
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".xrag-eval-", dir=output.parent) as temporary:
        staging = Path(temporary) / "frozen"
        staging.mkdir(mode=0o700)
        for name, payload in (
            ("inputs.jsonl", input_bytes),
            ("gold.jsonl", gold_bytes),
            ("manifest.json", manifest_bytes),
        ):
            path = staging / name
            with path.open("xb") as target:
                target.write(payload)
            path.chmod(0o600)
        os.rename(staging, output)
    return manifest


def _qasper_cases(
    data_dir: Path,
    exposed_papers: set[str],
    *,
    answerable: int = 6,
    no_answer: int = 2,
    count: int | None = None,
    seed: str = SEED,
) -> tuple[list[dict], list[dict]]:
    papers = list(read_snapshot(data_dir, "validation"))
    candidates = []
    for paper in papers:
        for question in paper.questions:
            if not question.annotations or not paper.passages:
                continue
            label = (
                "no_answer"
                if all(annotation.unanswerable for annotation in question.annotations)
                else "answerable"
            )
            candidates.append(
                (
                    paper.paper_id in exposed_papers,
                    _rank("qasper", question.question_id, seed=seed),
                    label,
                    paper,
                    question,
                )
            )

    selected = []
    used_papers: set[str] = set()
    if count is not None:
        ordered = sorted(candidates, key=lambda item: (item[0], item[1]))
        for previously_exposed, rank, label, paper, question in ordered:
            if paper.paper_id in used_papers:
                continue
            selected.append((previously_exposed, rank, label, paper, question))
            used_papers.add(paper.paper_id)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(
                f"QASPER validation can supply only {len(selected)} unique papers "
                f"for the requested sample of {count}"
            )
    else:
        eligible = [item for item in candidates if not item[0]]
        for label, limit in (("answerable", answerable), ("no_answer", no_answer)):
            if limit == 0:
                continue
            group = sorted(
                (item for item in eligible if item[2] == label), key=lambda item: item[1]
            )
            for _, rank, _, paper, question in group:
                if paper.paper_id in used_papers:
                    continue
                selected.append((False, rank, label, paper, question))
                used_papers.add(paper.paper_id)
                if sum(item[2] == label for item in selected) == limit:
                    break
        if len(selected) != answerable + no_answer:
            raise ValueError("the unused QASPER split cannot supply the requested frozen sample")

    question_ids = {question.question_id for _, _, _, _, question in selected}
    references = read_answer_references(data_dir, "validation", question_ids)
    inputs, gold = [], []
    for _, rank, label, paper, question in selected:
        case_id = f"qasper-{question.question_id}"
        source = [
            {
                "passage_id": passage.passage_id,
                "section_index": passage.section_index,
                "text": passage.text,
            }
            for passage in paper.passages
        ]
        alignment = align_question(paper, question)
        inputs.append(
            {
                "case_id": case_id,
                "dataset": "qasper",
                "source_id": paper.paper_id,
                "language": "en",
                "question": question.text,
                "source": {
                    "format": "qasper_paragraphs",
                    "section_names": list(paper.section_names),
                    "paragraphs": source,
                },
            }
        )
        gold.append(
            {
                "case_id": case_id,
                "stratum": label,
                "references": [asdict(reference) for reference in references[question.question_id]],
                "aligned_evidence_passage_ids": list(alignment.gold_passage_ids),
                "evidence_alignment_issues": list(alignment.annotation_issues),
            }
        )
    return inputs, gold


def _longbench_v2_cases(
    rows: list[dict], exposed: set[str], *, count: int = 8, seed: str = SEED
) -> tuple[list[dict], list[dict], dict]:
    candidates = []
    for row in rows:
        words = len(row["context"].split())
        if row["domain"] == "Single-Document QA" and 8_000 <= words <= 32_000:
            choices = {letter: row[f"choice_{letter}"] for letter in "ABCD"}
            source_hash = _sha256(row["context"].encode("utf-8"))
            candidates.append(
                (
                    row["_id"] in exposed,
                    _rank("longbench-v2", row["_id"], seed=seed),
                    row,
                    words,
                    choices,
                    source_hash,
                )
            )
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected, seen_sources = [], set()
    for item in candidates:
        if item[5] in seen_sources:
            continue
        selected.append(item)
        seen_sources.add(item[5])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"LongBench v2 has only {len(selected)} eligible unique contexts "
            f"for the requested sample of {count}"
        )

    inputs, gold = [], []
    for previously_exposed, rank, row, words, choices, source_hash in selected:
        case_id = f"longbench-v2-{row['_id']}"
        question = f"{row['question']}\n\n" + "\n".join(
            f"({letter}) {choice}" for letter, choice in choices.items()
        )
        inputs.append(
            {
                "case_id": case_id,
                "dataset": "longbench-v2-single-document-qa",
                "source_id": row["_id"],
                "language": "en",
                "question": question,
                "source": {"format": "plain_text", "text": row["context"]},
            }
        )
        gold.append(
            {
                "case_id": case_id,
                "answer": row["answer"],
                "domain": row["domain"],
                "sub_domain": row["sub_domain"],
                "difficulty": row["difficulty"],
                "length": row["length"],
            }
        )
    metadata = {
        "source_rows": len(rows),
        "eligible_rows": len(candidates),
        "selected": [
            {
                "case_id": f"longbench-v2-{row['_id']}",
                "rank_sha256": rank,
                "context_sha256": source_hash,
                "word_count": words,
                "difficulty": row["difficulty"],
                "length": row["length"],
                "previously_exposed": previously_exposed,
            }
            for previously_exposed, rank, row, words, _choices, source_hash in selected
        ],
    }
    return inputs, gold, metadata


def _longbench_v2_full_cases(
    rows: list[dict], exposed: set[str], *, expected_count: int = 175
) -> tuple[list[dict], list[dict], dict]:
    """Keep the complete Single-Document QA split and isolate its answer key."""
    selected = [row for row in rows if row.get("domain") == "Single-Document QA"]
    if len(selected) != expected_count:
        raise ValueError("LongBench-v2 Single-Document QA count differs from the frozen protocol")
    identifiers = [row.get("_id") for row in selected]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("LongBench-v2 contains an invalid source ID")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("LongBench-v2 Single-Document QA IDs are not unique")

    inputs, gold = [], []
    selection = []
    for row in selected:
        choices = {letter: row[f"choice_{letter}"] for letter in "ABCD"}
        if any(not isinstance(value, str) or not value.strip() for value in choices.values()):
            raise ValueError("LongBench-v2 choices must be nonempty text")
        question = row.get("question")
        context = row.get("context")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("LongBench-v2 question must be nonempty text")
        if not isinstance(context, str) or not context.strip():
            raise ValueError("LongBench-v2 context must be nonempty text")

        source_id = row["_id"]
        case_id = f"longbench-v2-full-{source_id}"
        retrieval_question = (
            question + "\n\n" + "\n".join(f"({letter}) {choices[letter]}" for letter in "ABCD")
        )
        source_sha256 = _sha256(context.encode("utf-8"))
        inputs.append(
            {
                "case_id": case_id,
                "dataset": "longbench-v2-single-document-qa",
                "source_id": source_id,
                "language": "en",
                "question": retrieval_question,
                "benchmark_question": question,
                "choices": choices,
                "task_format": "multiple_choice",
                "domain": row["domain"],
                "sub_domain": row["sub_domain"],
                "difficulty": row["difficulty"],
                "length": row["length"],
                "source": {"format": "plain_text", "text": context},
            }
        )
        gold.append(
            {
                "case_id": case_id,
                "answer": row["answer"],
                "domain": row["domain"],
                "sub_domain": row["sub_domain"],
                "difficulty": row["difficulty"],
                "length": row["length"],
            }
        )
        selection.append(
            {
                "case_id": case_id,
                "source_id": source_id,
                "context_sha256": source_sha256,
                "context_characters": len(context),
                "difficulty": row["difficulty"],
                "length": row["length"],
                "previously_exposed": source_id in exposed,
            }
        )
    return (
        inputs,
        gold,
        {
            "source_rows": len(rows),
            "selected_rows": len(selected),
            "cases": selection,
        },
    )


def freeze_longbench_v2_full(args: argparse.Namespace) -> dict:
    """Freeze all 175 single-document items without placing answers in model inputs."""
    output = args.output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("refusing to overwrite an existing frozen evaluation directory")
    source_path = args.longbench_v2_full_json.expanduser().resolve()
    _verify_source(source_path, LONG_BENCH_V2)
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    known_ids = {row["_id"] for row in rows if row.get("domain") == "Single-Document QA"}
    exposed, scanned_files, skipped_frozen_files = _exposed_ids(ROOT / "outputs", known_ids)
    inputs, gold, selection = _longbench_v2_full_cases(rows, exposed)
    input_bytes, gold_bytes = _jsonl_bytes(inputs), _jsonl_bytes(gold)
    gold_by_id = {item["case_id"]: item for item in gold}
    cases = []
    for item in inputs:
        context = item["source"]["text"]
        cases.append(
            {
                "case_id": item["case_id"],
                "source_id": item["source_id"],
                "context_sha256": _sha256(context.encode("utf-8")),
                "question_sha256": _sha256(item["benchmark_question"].encode("utf-8")),
                "choices_sha256": _canonical_hash(item["choices"]),
                "gold_sha256": _canonical_hash(gold_by_id[item["case_id"]]),
                "difficulty": item["difficulty"],
                "length": item["length"],
                "previously_exposed": item["source_id"] in exposed,
            }
        )

    manifest = {
        "format": "pagecite-longbench-v2-rag-eval/v1",
        "created_date_utc": "2026-10-24",
        "state": "frozen_inputs_not_run",
        "handling": "Private local benchmark artifacts; do not publish dataset text or answers.",
        "protocol_scope": (
            "All 175 Single-Document QA items. Official LongBench RAG prompt and answer extractor "
            "are reused with PageCiteRAG-produced retrieved chunks; this is a local retrieval "
            "adaptation, not the official leaderboard run or a canonical LongBench-v2 RAG split."
        ),
        "source": {
            **LONG_BENCH_V2,
            "local_file": source_path.relative_to(ROOT).as_posix(),
            "source_sha256": _file_sha256(source_path),
        },
        "exposure_scan": {
            "outputs_files_scanned": scanned_files,
            "unrun_frozen_input_gold_manifests_found": skipped_frozen_files,
            "exposed_ids": len(exposed),
            "unexposed_ids": len(known_ids - exposed),
        },
        "selection": selection,
        "cases": cases,
        "metrics": {
            "primary": "LongBench-v2 exact multiple-choice accuracy; official extraction contract",
            "invalid_or_missing_choice": "zero",
            "missing_execution_slot": "protocol error; do not write score",
            "strata": ["difficulty", "length", "previously_exposed"],
        },
        "files": {
            "inputs.jsonl": {
                "sha256": _sha256(input_bytes),
                "bytes": len(input_bytes),
                "rows": len(inputs),
            },
            "gold.jsonl": {
                "sha256": _sha256(gold_bytes),
                "bytes": len(gold_bytes),
                "rows": len(gold),
            },
        },
        "preparation_code_sha256": _file_sha256(Path(__file__).resolve()),
        "project_lock_sha256": _file_sha256(ROOT / "uv.lock"),
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".longbench-v2-", dir=output.parent) as temporary:
        staging = Path(temporary) / "frozen"
        staging.mkdir(mode=0o700)
        for name, payload in (
            ("inputs.jsonl", input_bytes),
            ("gold.jsonl", gold_bytes),
            ("manifest.json", manifest_bytes),
        ):
            path = staging / name
            with path.open("xb") as target:
                target.write(payload)
            path.chmod(0o600)
        os.rename(staging, output)
    return manifest


def _multifieldqa_zh_cases(
    zip_path: Path, exposed: set[str], *, count: int = 8, seed: str = SEED
) -> tuple[list[dict], list[dict], dict]:
    with zipfile.ZipFile(zip_path) as archive:
        member = "data/multifieldqa_zh.jsonl"
        try:
            rows = [json.loads(line) for line in archive.open(member) if line.strip()]
        except KeyError as error:
            raise ValueError("pinned LongBench archive has no MultiFieldQA-zh data") from error
    candidates = []
    for row in rows:
        source_hash = _sha256(row["context"].encode("utf-8"))
        candidates.append(
            (
                row["_id"] in exposed,
                _rank("multifieldqa-zh", row["_id"], seed=seed),
                row,
                source_hash,
            )
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected, seen_sources = [], set()
    for item in candidates:
        if item[3] in seen_sources:
            continue
        selected.append(item)
        seen_sources.add(item[3])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"MultiFieldQA-zh has only {len(selected)} eligible unique contexts "
            f"for the requested sample of {count}"
        )

    inputs, gold = [], []
    for _previously_exposed, _rank_hash, row, _source_hash in selected:
        case_id = f"multifieldqa-zh-{row['_id']}"
        inputs.append(
            {
                "case_id": case_id,
                "dataset": "longbench-v1-multifieldqa-zh",
                "source_id": row["_id"],
                "language": "zh-Hant",
                "question": row["input"],
                "source": {"format": "plain_text", "text": row["context"]},
            }
        )
        gold.append({"case_id": case_id, "answers": row["answers"]})
    metadata = {
        "source_rows": len(rows),
        "unique_source_rows": len(candidates),
        "unexposed_rows": sum(not item[0] for item in candidates),
        "archive_member": "data/multifieldqa_zh.jsonl",
        "selected": [
            {
                "case_id": f"multifieldqa-zh-{row['_id']}",
                "rank_sha256": rank,
                "context_sha256": source_hash,
                "character_count": len(row["context"]),
                "previously_exposed": previously_exposed,
            }
            for previously_exposed, rank, row, source_hash in selected
        ],
    }
    return inputs, gold, metadata


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )


def freeze_product_combination(
    benchmark_root: Path,
    product_cases_path: Path,
    product_gold_path: Path,
    output_dir: Path,
) -> dict:
    """Append an explicitly exposed product suite to an immutable benchmark freeze."""
    benchmark_root = benchmark_root.expanduser().resolve()
    product_cases_path = product_cases_path.expanduser().resolve()
    product_gold_path = product_gold_path.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("refusing to overwrite an existing combined frozen dataset")

    base_manifest_path = benchmark_root / "manifest.json"
    base_inputs_path = benchmark_root / "inputs.jsonl"
    base_gold_path = benchmark_root / "gold.jsonl"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    for name, path in (("inputs.jsonl", base_inputs_path), ("gold.jsonl", base_gold_path)):
        expected = base_manifest["files"][name]
        if (
            path.is_symlink()
            or path.stat().st_size != expected["bytes"]
            or _file_sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"benchmark freeze failed integrity check: {name}")

    def read_rows(path: Path) -> list[dict]:
        if path.is_symlink() or not path.is_file():
            raise ValueError("product case and gold files must be regular local files")
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    benchmark_inputs = read_rows(base_inputs_path)
    benchmark_gold = read_rows(base_gold_path)
    product_inputs = read_rows(product_cases_path)
    product_gold = read_rows(product_gold_path)
    product_sources_path = product_cases_path.with_name("sources.json")
    if product_sources_path.is_symlink() or not product_sources_path.is_file():
        raise ValueError("product source attribution manifest is missing")
    product_sources = json.loads(product_sources_path.read_text(encoding="utf-8"))
    if len(product_inputs) != 12 or len(product_gold) != 12:
        raise ValueError("the product evaluation requires exactly twelve frozen cases")
    product_ids = [row.get("case_id") for row in product_inputs]
    if any(not isinstance(value, str) or not value for value in product_ids):
        raise ValueError("product cases require non-empty stable case IDs")
    if len(set(product_ids)) != len(product_ids):
        raise ValueError("product case IDs must be unique")
    if set(product_ids) & {row["case_id"] for row in benchmark_inputs}:
        raise ValueError("product cases duplicate a public benchmark case ID")
    if product_ids != [row.get("case_id") for row in product_gold]:
        raise ValueError("product rubric rows must match the frozen case order")

    source_files: dict[str, dict[str, str | int]] = {}
    product_case_specs = []
    for case, gold in zip(product_inputs, product_gold, strict=True):
        if case.get("dataset") != "product-functional":
            raise ValueError("product cases must use the product-functional dataset label")
        source = case.get("source")
        source_record = product_sources.get(case.get("source_id"))
        if not isinstance(source, dict) or source.get("format") != "file":
            raise ValueError("product cases must use the real local file ingestion path")
        relative = source.get("relative_path")
        expected_sha = source.get("sha256")
        path = Path(relative) if isinstance(relative, str) else Path()
        if (
            not isinstance(relative, str)
            or path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or not isinstance(expected_sha, str)
            or not isinstance(source_record, dict)
            or source_record.get("relative_path") != relative
            or source_record.get("sha256") != expected_sha
            or not isinstance(source_record.get("citation"), str)
            or not isinstance(source_record.get("rights_status"), str)
            or not isinstance(source_record.get("sharing"), str)
        ):
            raise ValueError("product source file identity is invalid")
        source_path = ROOT / path
        if (
            source_path.is_symlink()
            or not source_path.is_file()
            or source_path.resolve() != source_path
        ):
            raise ValueError("product source must be a regular project-local file")
        actual_sha = _file_sha256(source_path)
        if actual_sha != expected_sha:
            raise ValueError("product source file checksum differs from its annotation")
        source_files[relative] = {
            "sha256": actual_sha,
            "bytes": source_path.stat().st_size,
        }
        if (
            not isinstance(case.get("question"), str)
            or case.get("language") not in ("en", "zh-Hant")
            or case.get("question_language") not in ("en", "zh-Hant")
            or case.get("source_language") not in ("en", "zh-Hant")
            or not isinstance(gold.get("expected_facts"), list)
            or not gold["expected_facts"]
            or not isinstance(gold.get("source_locations"), list)
            or not gold["source_locations"]
        ):
            raise ValueError("product case is missing its frozen question, language, or rubric")
        product_case_specs.append(
            {
                "case_id": case["case_id"],
                "dataset": case["dataset"],
                "document_cluster": case["source_id"],
                "source_id": case["source_id"],
                "source_sha256": actual_sha,
                "question_sha256": _sha256(case["question"].encode("utf-8")),
                "gold_sha256": _canonical_hash(gold),
                "language": case["language"],
                "question_language": case["question_language"],
                "source_language": case["source_language"],
                "task_category": case["task_category"],
                "previously_exposed": True,
            }
        )

    all_inputs = benchmark_inputs + product_inputs
    all_gold = benchmark_gold + product_gold
    input_bytes = _jsonl_bytes(all_inputs)
    gold_bytes = _jsonl_bytes(all_gold)
    manifest = {
        **base_manifest,
        "format": "pagecite-reading-eval/v1-quality-finalization",
        "state": "frozen_inputs_not_run",
        "handling": "private local evaluation; do not publish dataset or source artifacts",
        "combined_from": {
            "benchmark_manifest_sha256": _file_sha256(base_manifest_path),
            "benchmark_inputs_sha256": base_manifest["files"]["inputs.jsonl"]["sha256"],
            "benchmark_gold_sha256": base_manifest["files"]["gold.jsonl"]["sha256"],
            "product_cases_sha256": _file_sha256(product_cases_path),
            "product_gold_sha256": _file_sha256(product_gold_path),
            "product_sources_sha256": _file_sha256(product_sources_path),
        },
        "product_case_scope": {
            "case_count": len(product_inputs),
            "all_cases_previously_exposed": True,
            "all_source_files_verified_by_sha256": True,
            "source_files": source_files,
            "source_attribution_and_rights": product_sources,
            "metrics": (
                "operational answer availability and refusal only; source-grounded fact, "
                + "completeness, citation support, and requested-language review are uncalibrated"
            ),
        },
        "cases": list(base_manifest["cases"]) + product_case_specs,
        "files": {
            "inputs.jsonl": {
                "sha256": _sha256(input_bytes),
                "bytes": len(input_bytes),
                "rows": len(all_inputs),
            },
            "gold.jsonl": {
                "sha256": _sha256(gold_bytes),
                "bytes": len(gold_bytes),
                "rows": len(all_gold),
            },
        },
        "preparation_code_sha256": _file_sha256(Path(__file__).resolve()),
        "project_lock_sha256": _file_sha256(ROOT / "uv.lock"),
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reading-eval-", dir=output.parent) as temporary:
        staging = Path(temporary) / "frozen"
        staging.mkdir(mode=0o700)
        for name, payload in (
            ("inputs.jsonl", input_bytes),
            ("gold.jsonl", gold_bytes),
            ("manifest.json", manifest_bytes),
        ):
            path = staging / name
            with path.open("xb") as target:
                target.write(payload)
            path.chmod(0o600)
        os.rename(staging, output)
    output.chmod(0o700)
    return manifest


def freeze(args: argparse.Namespace) -> dict:
    output = args.output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("refusing to overwrite an existing frozen evaluation directory")
    qasper_path = snapshot_path(args.qasper_data_dir, "validation")
    v1_path = args.longbench_v1_zip.expanduser().resolve()
    v2_path = args.longbench_v2_json.expanduser().resolve()
    _verify_source(v1_path, LONG_BENCH_V1)
    _verify_source(v2_path, LONG_BENCH_V2)

    papers = list(read_snapshot(args.qasper_data_dir, "validation"))
    qasper_ids = {question.question_id for paper in papers for question in paper.questions}
    v2_rows = json.loads(v2_path.read_text(encoding="utf-8"))
    v2_ids = {row["_id"] for row in v2_rows}
    with zipfile.ZipFile(v1_path) as archive:
        v1_member = "data/multifieldqa_zh.jsonl"
        v1_rows = [json.loads(line) for line in archive.open(v1_member) if line.strip()]
    v1_ids = {row["_id"] for row in v1_rows}
    exposed, scanned_files, skipped_frozen_files = _exposed_ids(
        ROOT / "outputs", qasper_ids | v2_ids | v1_ids
    )
    exposed_qasper = exposed & qasper_ids
    exposed_v2 = exposed & v2_ids
    exposed_v1 = exposed & v1_ids
    exposed_papers = {
        paper.paper_id
        for paper in papers
        if any(question.question_id in exposed_qasper for question in paper.questions)
    }

    selection_seed = f"pagecite-v1-quality-{getattr(args, 'selection_seed', 42)}"
    qasper_count = getattr(args, "qasper_count", None)
    qasper_answerable = getattr(args, "qasper_answerable_count", 6)
    qasper_no_answer = getattr(args, "qasper_no_answer_count", 2)
    longbench_v2_count = getattr(args, "longbench_v2_count", 8)
    multifieldqa_zh_count = getattr(args, "multifieldqa_zh_count", 8)
    expected_count = (
        (qasper_count if qasper_count is not None else qasper_answerable + qasper_no_answer)
        + longbench_v2_count
        + multifieldqa_zh_count
    )
    q_inputs, q_gold = _qasper_cases(
        args.qasper_data_dir,
        exposed_papers,
        answerable=qasper_answerable,
        no_answer=qasper_no_answer,
        count=qasper_count,
        seed=selection_seed,
    )
    v2_inputs, v2_gold, v2_meta = _longbench_v2_cases(
        v2_rows, exposed_v2, count=longbench_v2_count, seed=selection_seed
    )
    v1_inputs, v1_gold, v1_meta = _multifieldqa_zh_cases(
        v1_path, exposed_v1, count=multifieldqa_zh_count, seed=selection_seed
    )
    inputs = q_inputs + v2_inputs + v1_inputs
    gold = q_gold + v2_gold + v1_gold
    if len(inputs) != expected_count or len(gold) != expected_count:
        raise ValueError("frozen evaluation input/gold counts differ from the requested sample")
    if len({row["case_id"] for row in inputs}) != expected_count:
        raise ValueError("frozen evaluation contains duplicate case IDs")

    input_bytes, gold_bytes = _jsonl_bytes(inputs), _jsonl_bytes(gold)
    case_specs = []
    gold_by_id = {row["case_id"]: row for row in gold}
    for row in inputs:
        source = row["source"]
        if row["dataset"] == "qasper":
            previously_exposed = row["source_id"] in exposed_papers
        elif row["dataset"] == "longbench-v2-single-document-qa":
            previously_exposed = row["source_id"] in exposed_v2
        else:
            previously_exposed = row["source_id"] in exposed_v1
        source_bytes = (
            _jsonl_bytes(source["paragraphs"])
            if source["format"] == "qasper_paragraphs"
            else source["text"].encode("utf-8")
        )
        case_specs.append(
            {
                "case_id": row["case_id"],
                "dataset": row["dataset"],
                "source_id": row["source_id"],
                "language": row["language"],
                "previously_exposed": previously_exposed,
                "question_sha256": _sha256(row["question"].encode("utf-8")),
                "source_sha256": _sha256(source_bytes),
                "gold_sha256": _canonical_hash(gold_by_id[row["case_id"]]),
            }
        )

    manifest = {
        "format": "pagecite-reading-eval/v5",
        "created_date_utc": "2026-10-31",
        "seed": selection_seed,
        "state": "frozen_inputs_not_run",
        "handling": (
            "Redistribute benchmark inputs, gold, and outputs only as permitted by source terms."
        ),
        "selection_policy": {
            "qasper": (
                "hash-rank validation questions with at most one question per paper; when a total "
                "count is requested, prefer papers without any prior exposure and fill any "
                "shortfall from exposed papers; record exposure per case; do not stratify on "
                "answer label "
                "label or filter on evidence alignment"
            ),
            "longbench_v2": (
                "prefer unseen IDs, then hash-rank exposed IDs; single-document QA; "
                "8000-32000 whitespace words; unique context; record exposure per case; "
                f"{longbench_v2_count} rows"
            ),
            "multifieldqa_zh": (
                "prefer unseen IDs, then hash-rank exposed IDs if needed; unique context; "
                "record exposure per case; "
                f"{multifieldqa_zh_count} rows"
            ),
        },
        "exposure_scan": {
            "outputs_files_scanned": scanned_files,
            "unrun_frozen_input_gold_manifests_found": skipped_frozen_files,
            "qasper_exposed_question_ids": len(exposed_qasper),
            "qasper_previously_exposed_papers": len(exposed_papers),
            "qasper_unexposed_papers": len(papers) - len(exposed_papers),
            "qasper_selected_previously_exposed_cases": sum(
                row["source_id"] in exposed_papers for row in q_inputs
            ),
            "qasper_selected_unexposed_cases": sum(
                row["source_id"] not in exposed_papers for row in q_inputs
            ),
            "longbench_v2_exposed_ids": len(exposed_v2),
            "longbench_v2_selected_previously_exposed_cases": sum(
                row["source_id"] in exposed_v2 for row in v2_inputs
            ),
            "multifieldqa_zh_exposed_ids": len(exposed_v1),
            "multifieldqa_zh_selected_previously_exposed_cases": sum(
                row["source_id"] in exposed_v1 for row in v1_inputs
            ),
        },
        "sources": {
            "qasper": {
                **QASPER,
                "file": (
                    "data/qasper/16ac8fdf81e7e3ad213ead4b60fdcc9dc8ca40e9/qasper-validation.parquet"
                ),
                "sha256": "089781b91c337d348dd9e8b57cc8adc100ed2d9cab84a6127402bcccf1559222",
            },
            "longbench_v1_multifieldqa_zh": LONG_BENCH_V1,
            "longbench_v2": LONG_BENCH_V2,
        },
        "cases": case_specs,
        "dataset_selection_details": {"longbench_v2": v2_meta, "multifieldqa_zh": v1_meta},
        "metrics": {
            "qasper": (
                "QASPER Answer-F1 and paragraph-evidence P/R/F1; evidence N/A when original "
                "evidence cannot align to text paragraphs"
            ),
            "longbench_v2": (
                "official exact-choice accuracy after the pinned LongBench-v2 answer extraction "
                "contract; invalid/missing choice scores zero"
            ),
            "multifieldqa_zh": "official LongBench qa_f1_zh_score, jieba tokenizer; best reference",
            "claims": (
                "source provenance and claim-support are separate; "
                "no automated semantic-support claim"
            ),
        },
        "official_metric_sources": {
            "qasper": {
                "implementation": "src/page_cite_rag/answer_metrics.py",
                "compatible_evaluator_revision": "e996b6c7b1b5f95d9308a74e3586416c6e780df1",
                "compatible_evaluator_sha256": (
                    "781aba7cd8e524bef4f0a1b4bf3504e5b02cb1d8d5bf32a8f0a89dfa83e86bfe"
                ),
            },
            "longbench_v1": {
                "repository": "https://github.com/THUDM/LongBench",
                "revision": "2e00731f8d0bff23dc4325161044d0ed8af94c1e",
                "path": "LongBench/metrics.py",
                "metric": "qa_f1_zh_score",
            },
            "longbench_v2": {
                "repository": "https://github.com/THUDM/LongBench",
                "path": "pred.py",
                "extraction_contract": (
                    "The correct answer is (A|B|C|D); no valid extraction scores zero"
                ),
            },
        },
        "non_comparability": (
            "small local task-specific subset; model/prompt/retrieval differ from official "
            "leaderboard setup; not a leaderboard score or SOTA claim"
        ),
        "files": {
            "inputs.jsonl": {
                "sha256": _sha256(input_bytes),
                "bytes": len(input_bytes),
                "rows": expected_count,
            },
            "gold.jsonl": {
                "sha256": _sha256(gold_bytes),
                "bytes": len(gold_bytes),
                "rows": expected_count,
            },
        },
        "preparation_code_sha256": _file_sha256(Path(__file__).resolve()),
        "qasper_snapshot_sha256": _file_sha256(qasper_path),
        "project_lock_sha256": _file_sha256(ROOT / "uv.lock"),
        "longbench_v2_details": v2_meta,
        "multifieldqa_zh_details": v1_meta,
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reading-eval-", dir=output.parent) as temporary:
        staging = Path(temporary) / "frozen"
        staging.mkdir(mode=0o700)
        for name, payload in (
            ("inputs.jsonl", input_bytes),
            ("gold.jsonl", gold_bytes),
            ("manifest.json", manifest_bytes),
        ):
            path = staging / name
            with path.open("xb") as target:
                target.write(payload)
            path.chmod(0o600)
        os.rename(staging, output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--product-cases", type=Path)
    parser.add_argument("--product-gold", type=Path)
    parser.add_argument("--qasper-data-dir", type=Path, default=ROOT / "data/qasper")
    parser.add_argument("--longbench-v1-zip", type=Path)
    parser.add_argument("--longbench-v2-json", type=Path)
    parser.add_argument("--longbench-v2-full-json", type=Path)
    parser.add_argument("--xrag-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--qasper-count", type=int)
    parser.add_argument("--qasper-answerable-count", type=int, default=6)
    parser.add_argument("--qasper-no-answer-count", type=int, default=2)
    parser.add_argument("--longbench-v2-count", type=int, default=8)
    parser.add_argument("--multifieldqa-zh-count", type=int, default=8)
    args = parser.parse_args()
    if args.benchmark_root is not None:
        if (
            args.product_cases is None
            or args.product_gold is None
            or args.longbench_v2_full_json is not None
            or args.xrag_jsonl is not None
            or args.longbench_v1_zip is not None
            or args.longbench_v2_json is not None
        ):
            parser.error(
                "combine mode requires --benchmark-root, --product-cases, --product-gold, "
                "and --output-dir only"
            )
        if args.output_dir is None:
            parser.error("combine mode requires --output-dir")
        manifest = freeze_product_combination(
            args.benchmark_root,
            args.product_cases,
            args.product_gold,
            args.output_dir,
        )
    elif args.longbench_v2_full_json is not None:
        if args.xrag_jsonl is not None or args.longbench_v1_zip is not None:
            parser.error("use --longbench-v2-full-json by itself for the 175-item track")
        args.output_dir = args.output_dir or LONG_BENCH_V2_FULL_DEFAULT
        manifest = freeze_longbench_v2_full(args)
    elif args.xrag_jsonl is not None:
        args.output_dir = args.output_dir or XRAG_DEFAULT
        manifest = freeze_xrag(args)
    else:
        if args.longbench_v1_zip is None or args.longbench_v2_json is None:
            parser.error("provide both LongBench sources or use --xrag-jsonl")
        args.output_dir = args.output_dir or OUTPUT_DEFAULT
        manifest = freeze(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "state": manifest["state"],
                "case_count": len(manifest["cases"]),
                "input_sha256": manifest["files"]["inputs.jsonl"]["sha256"],
                "gold_sha256": manifest["files"]["gold.jsonl"]["sha256"],
                "exposure_scan": manifest.get("exposure_scan"),
                "product_case_count": manifest.get("product_case_scope", {}).get("case_count", 0),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
