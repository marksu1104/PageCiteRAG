"""Freeze exposure-screened QASPER strata for local reading evaluation.

``--stratum answerable`` freezes questions every annotator marked answerable.
``--stratum no-answer`` freezes questions every annotator marked unanswerable
(plan 5.17), to check that answering more often does not turn refusals into guesses.
``--exposure exposed`` draws only from papers already seen, for development sets.

Plan 5.10 step 4. The validation split was exposed during development, so this reads a
local, pinned QASPER split (``train`` by default; no model is trained, and the
annotations are human-made in every split). Papers seen in any ``outputs`` artifact and
papers in reserved manifests are excluded; at most one question per paper is taken in
seeded hash order. Gold evidence only decides which questions qualify; it never enters
retrieval, reading or generation. Not comparable with the official test-split leaderboard.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from page_cite_rag.answer_metrics import read_answer_references  # noqa: E402
from page_cite_rag.qasper import (  # noqa: E402
    align_question,
    read_snapshot,
    snapshot_path,
    source_url,
)

_spec = importlib.util.spec_from_file_location("prepare", ROOT / "scripts/prepare_reading_eval.py")
prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare)


def select_cases(
    papers,
    exposed: set[str],
    reserved: frozenset[str],
    count: int,
    seed: str,
    *,
    stratum: str = "multi-section",
    exposure: str = "unexposed",
):
    candidates = []
    for paper in papers:
        seen = paper.paper_id in exposed
        if (seen != (exposure == "exposed")) or paper.paper_id in reserved or not paper.passages:
            continue
        section_of = {passage.passage_id: passage.section_index for passage in paper.passages}
        for question in paper.questions:
            if not question.annotations:
                continue
            if stratum == "no-answer":
                if all(annotation.unanswerable for annotation in question.annotations):
                    rank = prepare._rank("qasper", question.question_id, seed=seed)
                    candidates.append((rank, paper, question, None, 0))
                continue
            if stratum == "answerable" and not all(
                not annotation.unanswerable for annotation in question.annotations
            ):
                continue
            alignment = align_question(paper, question)
            if stratum == "answerable":
                rank = prepare._rank("qasper", question.question_id, seed=seed)
                section_count = len(
                    {section_of[pid] for pid in alignment.gold_passage_ids if pid in section_of}
                )
                candidates.append((rank, paper, question, alignment, section_count))
                continue
            if alignment.exclusion_reason or not alignment.gold_passage_ids:
                continue
            sections = {section_of[pid] for pid in alignment.gold_passage_ids if pid in section_of}
            if len(sections) >= 2:
                rank = prepare._rank("qasper", question.question_id, seed=seed)
                candidates.append((rank, paper, question, alignment, len(sections)))
    selected, used = [], set()
    for item in sorted(candidates, key=lambda item: item[0]):
        if item[1].paper_id in used:
            continue
        selected.append(item)
        used.add(item[1].paper_id)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} eligible papers for the requested {count}")
    return selected, len(candidates), len({item[1].paper_id for item in candidates})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qasper-data-dir", type=Path, default=ROOT / "data/qasper")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument(
        "--stratum",
        choices=("answerable", "multi-section", "no-answer"),
        default="multi-section",
    )
    parser.add_argument("--exposure", choices=("unexposed", "exposed"), default="unexposed")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit("refusing to overwrite an existing frozen evaluation directory")
    seed = str(args.selection_seed)
    source_path = snapshot_path(args.qasper_data_dir.expanduser().resolve(), args.split)
    papers = list(read_snapshot(args.qasper_data_dir, args.split))
    question_paper = {q.question_id: p.paper_id for p in papers for q in p.questions}
    exposed_questions, scanned, skipped = prepare._exposed_ids(
        ROOT / "outputs", set(question_paper) | {p.paper_id for p in papers}
    )
    exposed = {question_paper.get(value, value) for value in exposed_questions}
    reserved = prepare._reserved_ids(args.exclude_manifest)
    selected, eligible_questions, eligible_papers = select_cases(
        papers, exposed, reserved, args.count, seed, stratum=args.stratum, exposure=args.exposure
    )
    references = read_answer_references(
        args.qasper_data_dir, args.split, {item[2].question_id for item in selected}
    )
    inputs, gold = [], []
    for _rank, paper, question, alignment, sections in selected:
        case_id = f"qasper-{question.question_id}"
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
                    "paragraphs": [
                        {
                            "passage_id": passage.passage_id,
                            "section_index": passage.section_index,
                            "text": passage.text,
                        }
                        for passage in paper.passages
                    ],
                },
            }
        )
        gold.append(
            {
                "case_id": case_id,
                "stratum": {
                    "answerable": "answerable",
                    "multi-section": "multi_section_evidence",
                    "no-answer": "no_answer",
                }[args.stratum],
                "evidence_section_count": sections,
                "references": [asdict(ref) for ref in references[question.question_id]],
                "aligned_evidence_passage_ids": (
                    list(alignment.gold_passage_ids) if alignment is not None else []
                ),
                "evidence_alignment_issues": (
                    list(alignment.annotation_issues) if alignment is not None else []
                ),
            }
        )
    input_bytes, gold_bytes = prepare._jsonl_bytes(inputs), prepare._jsonl_bytes(gold)
    gold_by_id = {row["case_id"]: row for row in gold}
    cases = [
        {
            "case_id": row["case_id"],
            "dataset": "qasper",
            "source_id": row["source_id"],
            "language": "en",
            "previously_exposed": args.exposure == "exposed",
            "question_sha256": prepare._sha256(row["question"].encode("utf-8")),
            "source_sha256": prepare._sha256(prepare._jsonl_bytes(row["source"]["paragraphs"])),
            "gold_sha256": prepare._canonical_hash(gold_by_id[row["case_id"]]),
        }
        for row in inputs
    ]
    manifest = {
        "format": "pagecite-reading-eval/v5",
        "created_date_utc": datetime.now(timezone.utc).date().isoformat(),
        "seed": args.selection_seed,
        "reserved_ids_excluded": sorted(reserved),
        "state": "frozen_inputs_not_run",
        "handling": "Redistribute QASPER inputs, gold and outputs only as CC-BY-4.0 permits.",
        "selection_policy": {
            "qasper": (
                f"{args.split} split; "
                + (
                    "only papers already seen in outputs"
                    if args.exposure == "exposed"
                    else "exclude papers seen in outputs"
                )
                + " and reserved manifests; "
                + (
                    "keep questions every annotator marked unanswerable"
                    if args.stratum == "no-answer"
                    else (
                        "keep questions every annotator marked answerable"
                        if args.stratum == "answerable"
                        else "keep questions whose aligned human evidence spans >=2 sections"
                    )
                )
                + f"; at most one question per paper in seeded hash order; {args.count} cases"
            )
        },
        "exposure_scan": {
            "outputs_files_scanned": scanned,
            "unrun_frozen_input_gold_manifests_found": skipped,
            "qasper_previously_exposed_papers": len(exposed & {p.paper_id for p in papers}),
            "eligible_questions": eligible_questions,
            "eligible_papers": eligible_papers,
            "qasper_selected_previously_exposed_cases": (
                len(cases) if args.exposure == "exposed" else 0
            ),
        },
        "sources": {
            "qasper": {
                **prepare.QASPER,
                "split": args.split,
                "url": source_url(args.split),
                "file": source_path.relative_to(ROOT).as_posix(),
                "sha256": prepare._file_sha256(source_path),
            }
        },
        "cases": cases,
        "metrics": {
            "qasper": (
                "QASPER Answer-F1 and paragraph-evidence P/R/F1; evidence N/A when original "
                "evidence cannot align to text paragraphs"
            )
        },
        "non_comparability": (
            f"{args.split}-split {args.stratum} subset; local model, prompt and retrieval; "
            "not a leaderboard score"
        ),
        "files": {
            "inputs.jsonl": {
                "sha256": prepare._sha256(input_bytes),
                "bytes": len(input_bytes),
                "rows": len(inputs),
            },
            "gold.jsonl": {
                "sha256": prepare._sha256(gold_bytes),
                "bytes": len(gold_bytes),
                "rows": len(gold),
            },
        },
        "preparation_code_sha256": prepare._file_sha256(Path(__file__).resolve()),
        "qasper_snapshot_sha256": prepare._file_sha256(source_path),
        "project_lock_sha256": prepare._file_sha256(ROOT / "uv.lock"),
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
            with (staging / name).open("xb") as target:
                target.write(payload)
            (staging / name).chmod(0o600)
        os.rename(staging, output)
    print(json.dumps({"output_dir": str(output), "cases": len(cases), **manifest["exposure_scan"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
