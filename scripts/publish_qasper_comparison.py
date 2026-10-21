"""Validate and publish identifier-only comparison artifacts, never corpus/index files."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from page_cite_rag.evaluation import QueryRecord, _json_hash, _summarize


def publish(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise ValueError("publication destination already exists; refusing to overwrite")
    methods = {}
    records = {}
    selection_sha = None
    row_fields = set(QueryRecord.model_fields) | {"recall_at_k"}
    summary_fields = {
        "run_id",
        "selection_sha256",
        "platform",
        "configuration",
        "coverage",
        "results",
        "this_invocation",
    }
    for directory in sorted(source.iterdir()):
        if not directory.is_dir() or len(directory.name) != 64:
            continue
        summary = json.loads((directory / "summary.json").read_text())
        if set(summary) != summary_fields:
            raise ValueError("unexpected summary fields")
        config = summary["configuration"]
        method = config["retriever"]["method"]
        if method not in ("bm25", "dense", "hybrid") or method in methods:
            raise ValueError("expected exactly one complete run for each method")
        selection = json.loads((directory / "selection.json").read_text())
        if (
            _json_hash(selection) != summary["selection_sha256"]
            or _json_hash({"configuration": config, "selection": selection}) != summary["run_id"]
        ):
            raise ValueError("run/selection checksum mismatch")
        if selection_sha is not None and selection_sha != summary["selection_sha256"]:
            raise ValueError("methods do not share frozen selection")
        selection_sha = summary["selection_sha256"]
        parsed = []
        expected = {
            (r["split"], r["paper_id"], r["question_id"]): r["gold_passage_ids"] for r in selection
        }
        seen = set()
        for line in (directory / "results.jsonl").read_text().splitlines():
            row = json.loads(line)
            if set(row) != row_fields:
                raise ValueError("unexpected result fields; refusing an invalid result record")
            record = QueryRecord.model_validate({k: row[k] for k in QueryRecord.model_fields})
            key = (record.split, record.paper_id, record.question_id)
            if (
                key in seen
                or list(record.gold_passage_ids) != expected.get(key)
                or any(
                    not p.startswith(record.paper_id + "/") for p in record.retrieved_passage_ids
                )
                or record.status != "ok"
            ):
                raise ValueError("query identity, scope or status mismatch")
            seen.add(key)
            parsed.append(record)
        if seen != set(expected) or _summarize(parsed) != summary["results"]:
            raise ValueError("published metrics do not match per-question results")
        methods[method] = summary
        records[method] = parsed
    if set(methods) != {"bm25", "dense", "hybrid"}:
        raise ValueError("comparison is incomplete")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "comparison"
        staging.mkdir()
        report = {"selection_sha256": selection_sha, "methods": methods}
        (staging / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        for method, rows in records.items():
            with (staging / (method + ".jsonl")).open("w") as out:
                for row in rows:
                    out.write(row.model_dump_json() + "\n")
        staging.rename(destination)
    print(f"Published {sum(map(len, records.values()))} identifier-only query records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/qasper-comparison"))
    parser.add_argument("--destination", type=Path, default=Path("benchmarks/qasper-retrieval"))
    args = parser.parse_args()
    publish(args.source, args.destination)
