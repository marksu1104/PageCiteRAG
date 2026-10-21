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
    for path in output_dir.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.parent.name.startswith("reading-eval-") and path.name in {
            "inputs.jsonl",
            "gold.jsonl",
            "manifest.json",
        }:
            frozen_manifest_files_skipped += 1
            continue
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        files += 1
        for pattern in _ID_PATTERNS:
            exposed.update(
                value.decode("ascii")
                for value in pattern.findall(payload)
                if value.decode("ascii") in known_ids
            )
    return exposed, files, frozen_manifest_files_skipped


def _rank(kind: str, identifier: str) -> str:
    return _sha256(f"{SEED}:{kind}:{identifier}".encode("utf-8"))


def _qasper_cases(data_dir: Path, exposed_papers: set[str]) -> tuple[list[dict], list[dict]]:
    papers = list(read_snapshot(data_dir, "validation"))
    candidates = []
    for paper in papers:
        if paper.paper_id in exposed_papers:
            continue
        for question in paper.questions:
            if not question.annotations or not paper.passages:
                continue
            label = (
                "no_answer"
                if all(annotation.unanswerable for annotation in question.annotations)
                else "answerable"
            )
            candidates.append((_rank("qasper", question.question_id), label, paper, question))

    selected = []
    used_papers: set[str] = set()
    for label, limit in (("answerable", 6), ("no_answer", 2)):
        group = sorted((item for item in candidates if item[1] == label), key=lambda item: item[0])
        for rank, _, paper, question in group:
            if paper.paper_id in used_papers:
                continue
            selected.append((rank, label, paper, question))
            used_papers.add(paper.paper_id)
            if sum(item[1] == label for item in selected) == limit:
                break
    if len(selected) != 8 or len(used_papers) != 8:
        raise ValueError("the unused QASPER split cannot supply the frozen 6/2 sample")

    question_ids = {question.question_id for _, _, _, question in selected}
    references = read_answer_references(data_dir, "validation", question_ids)
    inputs, gold = [], []
    for rank, label, paper, question in selected:
        case_id = f"qasper-{question.question_id}"
        source = [
            {"passage_id": passage.passage_id, "text": passage.text} for passage in paper.passages
        ]
        alignment = align_question(paper, question)
        inputs.append(
            {
                "case_id": case_id,
                "dataset": "qasper",
                "source_id": paper.paper_id,
                "language": "en",
                "question": question.text,
                "source": {"format": "qasper_paragraphs", "paragraphs": source},
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


def _longbench_v2_cases(rows: list[dict], exposed: set[str]) -> tuple[list[dict], list[dict], dict]:
    candidates = []
    for row in rows:
        words = len(row["context"].split())
        if (
            row["domain"] == "Single-Document QA"
            and 8_000 <= words <= 32_000
            and row["_id"] not in exposed
        ):
            choices = {letter: row[f"choice_{letter}"] for letter in "ABCD"}
            source_hash = _sha256(row["context"].encode("utf-8"))
            candidates.append((_rank("longbench-v2", row["_id"]), row, words, choices, source_hash))
    candidates.sort(key=lambda item: item[0])
    selected, seen_sources = [], set()
    for item in candidates:
        if item[4] in seen_sources:
            continue
        selected.append(item)
        seen_sources.add(item[4])
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise ValueError("LongBench v2 has fewer than eight eligible unseen contexts")

    inputs, gold = [], []
    for rank, row, words, choices, source_hash in selected:
        case_id = f"longbench-v2-{row['_id']}"
        question = (
            f"{row['question']}\n\nSelect exactly one choice and include the exact phrase "
            "“The correct answer is (X)” with X replaced by A, B, C, or D.\n"
            + "\n".join(f"({letter}) {choice}" for letter, choice in choices.items())
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
            }
            for rank, row, words, _choices, source_hash in selected
        ],
    }
    return inputs, gold, metadata


def _multifieldqa_zh_cases(
    zip_path: Path, exposed: set[str]
) -> tuple[list[dict], list[dict], dict]:
    with zipfile.ZipFile(zip_path) as archive:
        member = "data/multifieldqa_zh.jsonl"
        try:
            rows = [json.loads(line) for line in archive.open(member) if line.strip()]
        except KeyError as error:
            raise ValueError("pinned LongBench archive has no MultiFieldQA-zh data") from error
    candidates = []
    for row in rows:
        if row["_id"] in exposed:
            continue
        source_hash = _sha256(row["context"].encode("utf-8"))
        candidates.append((_rank("multifieldqa-zh", row["_id"]), row, source_hash))
    candidates.sort(key=lambda item: item[0])
    selected, seen_sources = [], set()
    for item in candidates:
        if item[2] in seen_sources:
            continue
        selected.append(item)
        seen_sources.add(item[2])
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise ValueError("MultiFieldQA-zh has fewer than eight eligible unseen contexts")

    inputs, gold = [], []
    for _rank_hash, row, _source_hash in selected:
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
        "archive_member": "data/multifieldqa_zh.jsonl",
        "selected": [
            {
                "case_id": f"multifieldqa-zh-{row['_id']}",
                "rank_sha256": rank,
                "context_sha256": source_hash,
                "character_count": len(row["context"]),
            }
            for rank, row, source_hash in selected
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

    q_inputs, q_gold = _qasper_cases(args.qasper_data_dir, exposed_papers)
    v2_inputs, v2_gold, v2_meta = _longbench_v2_cases(v2_rows, exposed_v2)
    v1_inputs, v1_gold, v1_meta = _multifieldqa_zh_cases(v1_path, exposed_v1)
    inputs = q_inputs + v2_inputs + v1_inputs
    gold = q_gold + v2_gold + v1_gold
    if len(inputs) != 24 or len(gold) != 24:
        raise ValueError("frozen evaluation must contain exactly 24 input/gold pairs")
    if len({row["case_id"] for row in inputs}) != 24:
        raise ValueError("frozen evaluation contains duplicate case IDs")

    input_bytes, gold_bytes = _jsonl_bytes(inputs), _jsonl_bytes(gold)
    case_specs = []
    gold_by_id = {row["case_id"]: row for row in gold}
    for row in inputs:
        source = row["source"]
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
                "question_sha256": _sha256(row["question"].encode("utf-8")),
                "source_sha256": _sha256(source_bytes),
                "gold_sha256": _canonical_hash(gold_by_id[row["case_id"]]),
            }
        )

    manifest = {
        "format": "pagecite-reading-eval/v4",
        "created_date_utc": "2026-10-20",
        "seed": SEED,
        "state": "frozen_inputs_not_run",
        "handling": (
            "Redistribute benchmark inputs, gold, and outputs only as permitted by source terms."
        ),
        "selection_policy": {
            "qasper": (
                "hash-rank questions; exclude every paper with a previously exposed validation "
                "question; one question per paper; 6 answerable and 2 all-annotator-unanswerable; "
                "do not filter on evidence alignment"
            ),
            "longbench_v2": (
                "hash-rank unseen IDs; single-document QA; 8000-32000 whitespace words; "
                "unique context; eight rows"
            ),
            "multifieldqa_zh": "hash-rank unseen IDs; unique context; eight rows",
        },
        "exposure_scan": {
            "outputs_files_scanned": scanned_files,
            "unrun_frozen_input_gold_manifests_found": skipped_frozen_files,
            "qasper_exposed_question_ids": len(exposed_qasper),
            "qasper_exposed_papers_excluded": len(exposed_papers),
            "qasper_unexposed_papers": len(papers) - len(exposed_papers),
            "longbench_v2_exposed_ids": len(exposed_v2),
            "multifieldqa_zh_exposed_ids": len(exposed_v1),
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
            "inputs.jsonl": {"sha256": _sha256(input_bytes), "bytes": len(input_bytes), "rows": 24},
            "gold.jsonl": {"sha256": _sha256(gold_bytes), "bytes": len(gold_bytes), "rows": 24},
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
    parser.add_argument("--qasper-data-dir", type=Path, default=ROOT / "data/qasper")
    parser.add_argument("--longbench-v1-zip", type=Path, required=True)
    parser.add_argument("--longbench-v2-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    manifest = freeze(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "state": manifest["state"],
                "case_count": len(manifest["cases"]),
                "input_sha256": manifest["files"]["inputs.jsonl"]["sha256"],
                "gold_sha256": manifest["files"]["gold.jsonl"]["sha256"],
                "exposure_scan": manifest["exposure_scan"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
