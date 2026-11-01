import hashlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from page_cite_rag.models import Block
from page_cite_rag.retrieval import SearchBranchTrace, SearchContextGroup, SearchHit, SearchResult
from page_cite_rag.source_outline import OUTLINE_KIND, SourceOutline
from page_cite_rag.store import SQLiteDocumentStore

_RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts/run_reading_eval.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location("run_reading_eval", _RUNNER_PATH)
run_reading_eval = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = run_reading_eval
_RUNNER_SPEC.loader.exec_module(run_reading_eval)


def _write_frozen_fixture(root, case_ids, row_ids=None):
    rows = [{"case_id": case_id} for case_id in (case_ids if row_ids is None else row_ids)]
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    identity = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    for filename in ("inputs.jsonl", "gold.jsonl"):
        (root / filename).write_bytes(payload)
    manifest = {
        "files": {filename: dict(identity) for filename in ("inputs.jsonl", "gold.jsonl")},
        "cases": [{"case_id": case_id} for case_id in case_ids],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, rows


def test_diagnostics_preserve_dispatch_wait_and_guard_snapshot():
    value = {
        "dispatch_wait_seconds_total": 12.5,
        "last_dispatch_wait": {"seconds": 4.5, "dispatch_temperature_limit_c": 69},
        "last_gpu_check": {"temperature_c": 87, "sw_thermal_slowdown": True},
        "guard_error": "gpu_thermal_slowdown",
    }
    result = run_reading_eval._safe_model_diagnostics(SimpleNamespace(diagnostics=lambda: value))
    for key in value:
        assert result[key] == value[key]


def test_cached_hybrid_pool_reconstruction_preserves_old_top_k(tmp_path):
    document_id = "a" * 64
    blocks = [
        Block(
            document_id=document_id,
            block_id=f"{index:064x}",
            ordinal=index,
            unit_index=index,
            line_start=1,
            line_end=1,
            char_start=0,
            char_end=len(f"text {index}"),
            text=f"text {index}",
        )
        for index in range(1, 4)
    ]
    by_id = {block.block_id: block for block in blocks}

    class Store:
        def get_block(self, requested_document_id, block_id):
            return by_id.get(block_id) if requested_document_id == document_id else None

    lexical_ids = (blocks[0].block_id, blocks[1].block_id)
    dense_ids = (blocks[2].block_id, blocks[1].block_id)
    lexical = SearchResult(
        document_id=document_id,
        index_id="lexical-index",
        method="hybrid",
        query="question",
        hits=tuple(
            SearchHit(rank=rank, score=0.0, block=by_id[block_id])
            for rank, block_id in enumerate(lexical_ids, 1)
        ),
        elapsed_ms=1,
    )
    dense = SearchResult(
        document_id=document_id,
        index_id="dense-index",
        method="hybrid",
        query="question",
        hits=tuple(
            SearchHit(rank=rank, score=0.0, block=by_id[block_id])
            for rank, block_id in enumerate(dense_ids, 1)
        ),
        elapsed_ms=2,
    )
    old_top = run_reading_eval.reciprocal_rank_fusion([lexical, dense], top_k=2)
    cached = SearchResult(
        document_id=document_id,
        index_id="cached-hybrid-index",
        method="hybrid",
        query="question",
        hits=old_top,
        elapsed_ms=3,
        branches=(
            SearchBranchTrace(
                name="lexical", index_id="lexical-index", elapsed_ms=1, block_ids=lexical_ids
            ),
            SearchBranchTrace(
                name="dense", index_id="dense-index", elapsed_ms=2, block_ids=dense_ids
            ),
        ),
    )

    expanded = run_reading_eval._expand_cached_hybrid_candidates(
        cached, Store(), top_k=4, constant=60
    )

    assert [hit.block.block_id for hit in expanded.hits[:2]] == [
        hit.block.block_id for hit in old_top
    ]
    assert [hit.block.ordinal for hit in expanded.hits] == [2, 1, 3]


def test_product_file_source_uses_the_project_ingestion_path(tmp_path, monkeypatch):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = source_root / "sample.md"
    source.write_text("# Heading\n\nA verified source sentence.", encoding="utf-8")
    monkeypatch.setattr(run_reading_eval, "ROOT", tmp_path)
    case = {
        "case_id": "product-example",
        "dataset": "product-functional",
        "source_language": "en",
        "source": {
            "format": "file",
            "relative_path": "sources/sample.md",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    }
    store = SQLiteDocumentStore(tmp_path / "reading.sqlite3")

    document_id, block_map = run_reading_eval._case_document(store, case)

    document = store.get_document(document_id)
    assert document.parser_name == "markdown"
    assert document.source_sha256 == case["source"]["sha256"]
    extracted = " ".join(block.text for block in store.get_blocks(document_id))
    assert "verified source sentence" in extracted
    assert block_map == {}


def test_product_file_source_rejects_changed_bytes_and_path_escape(tmp_path, monkeypatch):
    source = tmp_path / "sample.txt"
    source.write_text("Original source", encoding="utf-8")
    monkeypatch.setattr(run_reading_eval, "ROOT", tmp_path)
    store = SQLiteDocumentStore(tmp_path / "reading.sqlite3")
    case = {
        "case_id": "product-example",
        "source": {
            "format": "file",
            "relative_path": "sample.txt",
            "sha256": hashlib.sha256(b"different source").hexdigest(),
        },
    }

    with pytest.raises(ValueError, match="checksum"):
        run_reading_eval._case_document(store, case)
    case["source"]["relative_path"] = "../outside.txt"
    with pytest.raises(ValueError, match="project-relative"):
        run_reading_eval._case_document(store, case)


def test_selected_reranker_is_shared_with_agent_initial_context():
    row = {
        "result": {"method": "hybrid", "hits": []},
        "reranked_result": {
            "method": "hybrid",
            "hits": [{"block": {"block_id": "ranked-block"}}],
        },
        "block_to_passage_id": {"ranked-block": "passage-1"},
    }

    assert run_reading_eval._row_for_arm(row, "single-agent")["result"] == row["result"]
    shared = run_reading_eval._row_for_arm(row, "single-agent", shared_reranker=True)
    fixed = run_reading_eval._row_for_arm(row, "fixed-rag-reranked")

    assert shared["result"] == row["reranked_result"]
    assert shared["retrieved_passage_ids"] == ["passage-1"]
    assert fixed["result"] == shared["result"]


def test_fixed_followup_selects_next_frozen_hybrid_blocks_once():
    document_id = "a" * 64
    blocks = [
        Block(
            document_id=document_id,
            block_id=hashlib.sha256(f"followup-{ordinal}".encode()).hexdigest(),
            ordinal=ordinal,
            unit_index=1,
            line_start=ordinal,
            line_end=ordinal,
            char_start=ordinal - 1,
            char_end=ordinal,
            text=str(ordinal),
        )
        for ordinal in range(1, 5)
    ]
    initial = SearchResult(
        document_id=document_id,
        index_id="reranked-index",
        method="hybrid",
        query="question",
        hits=(
            SearchHit(rank=1, score=0.9, block=blocks[1]),
            SearchHit(rank=2, score=0.8, block=blocks[0]),
        ),
        elapsed_ms=1.0,
    )
    candidate_pool = SearchResult(
        document_id=document_id,
        index_id="hybrid-index",
        method="hybrid",
        query="question",
        hits=tuple(
            SearchHit(rank=rank, score=1 / rank, block=block)
            for rank, block in enumerate(blocks, start=1)
        ),
        elapsed_ms=1.0,
    )

    followup = run_reading_eval._fixed_followup_search(
        initial,
        candidate_pool,
        initial_admitted_block_ids=[blocks[1].block_id],
        limit=1,
    )

    assert followup is not None
    assert [hit.block.block_id for hit in followup.hits] == [blocks[0].block_id]
    assert [hit.rank for hit in followup.hits] == [1]
    assert (
        run_reading_eval._fixed_followup_search(
            initial,
            candidate_pool,
            initial_admitted_block_ids=[blocks[1].block_id],
            limit=2,
        )
        .hits[1]
        .block
        == blocks[2]
    )


def test_fixed_followup_rejects_candidate_pool_that_does_not_contain_initial_hits():
    document_id = "b" * 64
    first = Block(
        document_id=document_id,
        block_id=hashlib.sha256(b"initial").hexdigest(),
        ordinal=1,
        unit_index=1,
        line_start=1,
        line_end=1,
        char_start=0,
        char_end=1,
        text="a",
    )
    other = first.model_copy(
        update={"block_id": hashlib.sha256(b"other").hexdigest(), "text": "b", "char_end": 1}
    )
    initial = SearchResult(
        document_id=document_id,
        index_id="reranked-index",
        method="hybrid",
        query="question",
        hits=(SearchHit(rank=1, score=1.0, block=first),),
        elapsed_ms=1.0,
    )
    candidates = SearchResult(
        document_id=document_id,
        index_id="hybrid-index",
        method="hybrid",
        query="question",
        hits=(SearchHit(rank=1, score=1.0, block=other),),
        elapsed_ms=1.0,
    )

    with pytest.raises(ValueError, match="invalid scope or ranking"):
        run_reading_eval._fixed_followup_search(
            initial,
            candidates,
            initial_admitted_block_ids=[first.block_id],
        )


def test_prepared_reranker_persists_each_result_and_updates_global_ledger(tmp_path, monkeypatch):
    import page_cite_rag.reranker_server as reranker_server
    from page_cite_rag.ingest import prepare_source

    source = tmp_path / "source.md"
    source.write_text("# Method\n\nA simple source passage.\n", encoding="utf-8")
    prepared = prepare_source(source, usage_scope="self-authored-test")
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    block = store.get_blocks(prepared.document.document_id)[1]
    search = SearchResult(
        document_id=block.document_id,
        index_id="candidate-index",
        method="hybrid",
        query="What is the method?",
        hits=(SearchHit(rank=1, score=1.0, block=block),),
        elapsed_ms=1.0,
    )
    case = {"case_id": "case-1"}
    row = {"case_id": "case-1", "candidate_result": search.model_dump(mode="json")}
    run_root = tmp_path / "answer-run"
    prepared_dir = run_root / "prepared"
    prepared_dir.mkdir(parents=True)
    retrieval_path = prepared_dir / "retrieval.jsonl"
    retrieval_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "usage": {
                    "reranker_pair_tokens_confirmed": 353,
                    "reranker_pair_tokens_unknown_upper_bound": 163840,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_reading_eval,
        "_load_prepared",
        lambda *_args, **_kwargs: (store, {"case-1": row}),
    )

    class FakeRanker:
        def __init__(self, *_args, **_kwargs):
            self.last_pair_tokens = ()
            self.last_batch_sizes = ()
            self.last_tokenization_ms = 1.0
            self.last_forward_ms = 2.0
            self.last_total_ms = 3.0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def rerank(self, _query, documents):
            self.last_pair_tokens = tuple(5 for _ in documents)
            self.last_batch_sizes = (len(documents),)
            return [(index, float(len(documents) - index)) for index in range(len(documents))]

        def diagnostics(self):
            return {"closed": True, "gpu_temperature_limit_c": 88}

    monkeypatch.setattr(reranker_server, "NativeRerankerServer", FakeRanker)
    result = run_reading_eval._rank_prepared_retrieval(
        run_root, {"retrieval": {"reranker": True}}, [case]
    )

    reranked_path = prepared_dir / "reranked" / "results.jsonl"
    ranked_row = json.loads(reranked_path.read_text(encoding="utf-8"))
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert result["state"] == "complete"
    assert result["pair_tokens"] == 5
    assert ranked_row["candidate_sha256"] == run_reading_eval.canonical_hash(
        search.model_dump(mode="json")
    )
    assert progress["usage"]["reranker_pair_tokens_confirmed"] == 358
    assert progress["usage"]["reranker_pair_tokens_inflight_upper_bound"] == 0


@pytest.mark.parametrize(
    ("dataset", "source_language", "question_language", "expected"),
    [
        ("qasper", "en", "en", "qasper"),
        ("qasper", "en", "zh-Hant", "derived-query-qasper-source-en"),
        (
            "longbench-v1-multifieldqa-zh",
            "zh-Hant",
            "zh-Hant",
            "longbench-v1-multifieldqa-zh",
        ),
        (
            "longbench-v1-multifieldqa-zh",
            "zh-Hant",
            "en",
            "derived-query-longbench-v1-multifieldqa-zh-source-zh-Hant",
        ),
    ],
)
def test_translated_queries_are_separate_from_native_benchmark_scores(
    dataset, source_language, question_language, expected
):
    assert (
        run_reading_eval._score_dataset_group(
            {"dataset": dataset},
            {"source_language": source_language, "question_language": question_language},
        )
        == expected
    )


def test_unique_document_embedding_plan_deduplicates_repeated_cases():
    result = run_reading_eval._unique_document_passages(
        {
            "case-1": ("document-a", {"p1": "first"}),
            "case-2": ("document-a", {"p1": "same source"}),
            "case-3": ("document-b", {"p2": "other"}),
        }
    )
    assert result == {"document-a": {"p1": "first"}, "document-b": {"p2": "other"}}


def test_protocol_configuration_hash_must_match_before_longbench_work():
    static = {"retrieval": {"top_k": 10}, "answer": {"temperature": 0.1}}
    expected = run_reading_eval.canonical_hash(static)

    run_reading_eval.verify_protocol_configuration_hash(static, expected, required=True)

    with pytest.raises(ValueError, match="does not match"):
        run_reading_eval.verify_protocol_configuration_hash(static, "0" * 64, required=True)
    with pytest.raises(ValueError, match="require the protocol-frozen"):
        run_reading_eval.verify_protocol_configuration_hash(static, None, required=True)


def test_protocol_configuration_hash_must_be_sha256_hex():
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        run_reading_eval.verify_protocol_configuration_hash(
            {"static": True}, "NOT-A-HASH", required=False
        )


def test_retrieval_gpu_guard_stops_the_batch_without_a_model_guard():
    assert run_reading_eval._gpu_guard_detected(
        {"guard_error": None}, {"guard_error": "owned_service_exited"}
    )
    assert not run_reading_eval._gpu_guard_detected({"guard_error": None}, {"guard_error": None})


def test_missing_gpu_selection_stops_before_loading_inputs(monkeypatch, capsys):
    monkeypatch.setattr(run_reading_eval, "GPU_UUID", "")
    monkeypatch.setattr(sys, "argv", [str(_RUNNER_PATH), "--phase", "prepare"])
    monkeypatch.setattr(
        run_reading_eval, "load_frozen", lambda *_: pytest.fail("loaded inputs without a GPU")
    )

    assert run_reading_eval.main() == 1
    assert "PAGE_CITE_RAG_GPU_UUID" in capsys.readouterr().err


@pytest.mark.parametrize("expected_hash", [None, "0" * 64])
def test_longbench_prepare_refuses_missing_or_mismatched_protocol_hash(
    monkeypatch, tmp_path, capsys, expected_hash
):
    _write_frozen_fixture(tmp_path, ["case-1"])
    run_root = tmp_path / "run"
    argv = [
        str(_RUNNER_PATH),
        "--phase",
        "prepare",
        "--frozen-root",
        str(tmp_path),
        "--run-root",
        str(run_root),
        "--compare-arms",
        run_reading_eval.LONG_BENCH_OFFICIAL_ARM,
    ]
    if expected_hash is not None:
        argv.extend(("--expected-static-configuration-sha256", expected_hash))
    monkeypatch.setattr(run_reading_eval, "GPU_UUID", "GPU-test")
    monkeypatch.setattr(sys, "argv", argv)

    assert run_reading_eval.main() == 1
    assert "protocol-frozen" in capsys.readouterr().err
    assert not (run_root / "configuration.json").exists()


def test_agent_final_input_limit_matches_fixed_rag_budget():
    from page_cite_rag.agent import MAX_EXACT_INPUT_TOKENS

    assert run_reading_eval.EXACT_INPUT_LIMIT == 12_000
    assert run_reading_eval.AGENT_FINAL_INPUT_LIMIT == MAX_EXACT_INPUT_TOKENS
    assert run_reading_eval.AGENT_FINAL_INPUT_LIMIT == run_reading_eval.EXACT_INPUT_LIMIT


@pytest.mark.parametrize("case_count", [1, 3, 24])
def test_frozen_loader_never_parses_gold(monkeypatch, tmp_path, case_count):
    manifest, rows = _write_frozen_fixture(
        tmp_path, [f"case-{index}" for index in range(case_count)]
    )

    original_read = run_reading_eval.read_jsonl

    def read_inputs_only(path):
        assert path.name == "inputs.jsonl"
        return original_read(path)

    monkeypatch.setattr(run_reading_eval, "read_jsonl", read_inputs_only)
    loaded_manifest, loaded_rows = run_reading_eval.load_frozen(tmp_path)

    assert loaded_manifest == manifest
    assert loaded_rows == rows


@pytest.mark.parametrize("case_count", [1, 3, 24])
def test_gold_loader_uses_manifest_case_count(tmp_path, case_count):
    manifest, rows = _write_frozen_fixture(
        tmp_path, [f"case-{index}" for index in range(case_count)]
    )

    assert run_reading_eval.load_gold(tmp_path, manifest) == rows


@pytest.mark.parametrize("filename", ["inputs.jsonl", "gold.jsonl"])
@pytest.mark.parametrize("case_ids", [[], ["duplicate", "duplicate"], [""]])
def test_frozen_loaders_reject_empty_or_duplicate_manifest_ids(tmp_path, filename, case_ids):
    manifest, _ = _write_frozen_fixture(tmp_path, case_ids)

    with pytest.raises(ValueError, match="manifest case IDs must be nonempty strings and unique"):
        if filename == "inputs.jsonl":
            run_reading_eval.load_frozen(tmp_path)
        else:
            run_reading_eval.load_gold(tmp_path, manifest)


@pytest.mark.parametrize("filename", ["inputs.jsonl", "gold.jsonl"])
@pytest.mark.parametrize(
    "row_ids",
    [
        ["case-a"],
        ["case-a", "case-b", "extra"],
        ["case-b", "case-a"],
        ["case-a", "case-a"],
        ["case-a", "unknown"],
    ],
)
def test_frozen_loaders_require_exact_manifest_ids_and_order(tmp_path, filename, row_ids):
    manifest, _ = _write_frozen_fixture(tmp_path, ["case-a", "case-b"], row_ids)

    with pytest.raises(ValueError, match="rows do not match the manifest"):
        if filename == "inputs.jsonl":
            run_reading_eval.load_frozen(tmp_path)
        else:
            run_reading_eval.load_gold(tmp_path, manifest)


@pytest.mark.parametrize("filename", ["inputs.jsonl", "gold.jsonl"])
@pytest.mark.parametrize("mismatch", ["bytes", "sha256", "symlink"])
def test_frozen_loaders_retain_file_identity_checks(tmp_path, filename, mismatch):
    manifest, _ = _write_frozen_fixture(tmp_path, ["case-a"])
    if mismatch == "symlink":
        path = tmp_path / filename
        target = tmp_path / "original.jsonl"
        path.rename(target)
        path.symlink_to(target)
    else:
        manifest["files"][filename][mismatch] = 0 if mismatch == "bytes" else "0" * 64
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen file (identity|checksum) differs"):
        if filename == "inputs.jsonl":
            run_reading_eval.load_frozen(tmp_path)
        else:
            run_reading_eval.load_gold(tmp_path, manifest)


def test_qasper_evidence_uses_original_paragraph_not_quote_span():
    passage = "The method improves exact-match by 2.1 points under the stated setting."
    result = {
        "answer_claims": [{"citations": [{"block_id": "block-1", "quote": "2.1 points"}]}],
        "detail_claims": [{"citations": [{"block_id": "block-2", "quote": "additional context"}]}],
    }

    evidence = run_reading_eval._qasper_evidence_texts(
        result,
        {"block-1": "passage-1", "block-2": "passage-2"},
        {"passage-1": passage, "passage-2": "A detail-only paragraph."},
    )

    assert evidence == (passage,)


def test_qasper_evidence_rejects_unmapped_citations():
    result = {"answer_claims": [{"citations": [{"block_id": "unknown"}]}]}

    with pytest.raises(ValueError, match="no original paragraph mapping"):
        run_reading_eval._qasper_evidence_texts(result, {}, {})


def test_xrag_article_mapping_uses_source_character_ranges():
    source = {
        "format": "xrag_article_bundle",
        "article_ranges": [
            {"article_id": "article-1", "char_start": 10, "char_end": 20},
            {"article_id": "article-2", "char_start": 30, "char_end": 40},
        ]
        + [
            {
                "article_id": f"article-{index}",
                "char_start": 50 + index * 20,
                "char_end": 60 + index * 20,
            }
            for index in range(3, 9)
        ],
    }

    class Store:
        def get_blocks(self, _document_id):
            return [
                SimpleNamespace(block_id="first", char_start=11, char_end=19),
                SimpleNamespace(block_id="second", char_start=31, char_end=39),
                SimpleNamespace(block_id="heading", char_start=22, char_end=25),
            ]

    assert run_reading_eval._xrag_block_article_map(Store(), "document", source) == {
        "first": "article-1",
        "second": "article-2",
    }


def test_xrag_article_mapping_rejects_blocks_crossing_article_boundaries():
    source = {
        "format": "xrag_article_bundle",
        "article_ranges": [
            {
                "article_id": f"article-{index}",
                "char_start": index * 20,
                "char_end": index * 20 + 10,
            }
            for index in range(8)
        ],
    }

    class Store:
        def get_blocks(self, _document_id):
            return [SimpleNamespace(block_id="crossing", char_start=5, char_end=25)]

    with pytest.raises(ValueError, match="crosses XRAG article boundaries"):
        run_reading_eval._xrag_block_article_map(Store(), "document", source)


def test_xrag_character_metrics_normalize_chinese_script_and_punctuation():
    score = run_reading_eval._xrag_zh_char_metrics("繁體回答，保留細節！", "繁体回答保留细节")

    assert score == {
        "answer_char_f1_nonofficial": 1.0,
        "answer_exact_match_nonofficial": 1.0,
    }


def test_xrag_two_partial_support_articles_require_both_for_hit():
    metrics = run_reading_eval._xrag_article_metrics(
        {
            "crossdoc_type": 2,
            "article_support_levels": {
                "article-1": 1,
                "article-2": 1,
                "article-3": 0,
                "article-4": 0,
                "article-5": 0,
                "article-6": 0,
                "article-7": 0,
                "article-8": 0,
            },
        },
        {"article-1", "article-2"},
        {"article-1"},
        {"article-1"},
        {"article-1"},
        {"article-1"},
    )

    assert metrics["sufficient_article_recall_at_10"] is None
    assert metrics["required_article_recall_candidate_union"] == 1.0
    assert metrics["required_article_recall_at_10"] == 0.5
    assert metrics["required_article_hit_at_10"] == 0.0
    assert metrics["citation_required_article_recall"] == 0.5
    assert metrics["citation_supporting_article_precision"] == 1.0


def test_search_result_roundtrip_validates_json_arrays_in_strict_mode():
    block_id = "b" * 64
    value = {
        "document_id": "a" * 64,
        "index_id": "index-id",
        "method": "hybrid",
        "query": "question",
        "hits": [
            {
                "rank": 1,
                "score": 0.5,
                "block": {
                    "document_id": "a" * 64,
                    "block_id": block_id,
                    "ordinal": 1,
                    "unit_index": 1,
                    "page_number": None,
                    "line_start": 1,
                    "line_end": 1,
                    "char_start": 0,
                    "char_end": 5,
                    "text": "hello",
                },
            }
        ],
        "elapsed_ms": 1.0,
        "branches": [
            {
                "name": "lexical",
                "index_id": "lexical-index",
                "elapsed_ms": 0.1,
                "block_ids": [block_id],
            }
        ],
    }

    search = run_reading_eval._search_result_from_json(value)

    assert len(search.hits) == 1
    assert search.branches[0].block_ids == (block_id,)


def test_longbench_v2_answer_extraction_requires_official_phrase():
    assert run_reading_eval._longbench_v2_choice("The correct answer is (C).") == "C"
    assert run_reading_eval._longbench_v2_choice("The correct answer is D") == "D"
    assert run_reading_eval._longbench_v2_choice("I think the answer is A") is None


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        ("qasper", "short_answer"),
        ("longbench-v1-multifieldqa-zh", "short_answer"),
        ("longbench-v2-single-document-qa", "multiple_choice"),
        ("xrag-zh-mono-crosslingual", "short_answer"),
        ("derived-crosslingual", "short_answer"),
    ],
)
def test_benchmark_dataset_uses_its_scoring_answer_format(dataset, expected):
    assert run_reading_eval._answer_format_for({"dataset": dataset}) == expected


def test_unknown_benchmark_has_no_implicit_answer_format():
    with pytest.raises(ValueError, match="unknown frozen dataset task"):
        run_reading_eval._answer_format_for({"dataset": "new-task"})


def test_rerank_preflight_keeps_each_arm_own_context():
    baseline = {"admitted_block_ids": ["a"]}
    reranked = {"admitted_block_ids": ["b"]}
    preflight = {
        "cases": {
            "q": {
                "by_arm": {
                    "fixed-rag": baseline,
                    "fixed-rag-reranked": reranked,
                }
            }
        }
    }
    assert (
        run_reading_eval._preflight_for_arm(preflight, "q", "fixed-rag", "contract-consistent")
        == baseline
    )
    assert (
        run_reading_eval._preflight_for_arm(
            preflight, "q", "fixed-rag-reranked", "contract-consistent"
        )
        == reranked
    )


def test_prompt_budget_context_supports_reading_plan_preflight_schema():
    legacy_preflight = {
        "admitted_block_ids": ["common-a"],
        "prompt_budget_admitted_block_ids": ["prompt-a", "prompt-b"],
    }
    reading_plan_preflight = {"admitted_block_ids": ["planned-a", "planned-b"]}

    assert run_reading_eval._prompt_budget_admitted_block_ids(legacy_preflight) == [
        "prompt-a",
        "prompt-b",
    ]
    assert run_reading_eval._prompt_budget_admitted_block_ids(reading_plan_preflight) == [
        "planned-a",
        "planned-b",
    ]


def test_static_configuration_records_xrag_answer_contract(tmp_path):
    manifest = {"files": {"inputs.jsonl": {"sha256": "a" * 64}}, "cases": []}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    static = run_reading_eval.static_configuration(tmp_path, manifest)

    assert static["answer"]["task_formats"]["xrag-zh-mono-crosslingual"] == "short_answer"
    assert static["retrieval"]["token_chunking"] is True
    assert static["retrieval"]["text_chunking"] is not None
    assert static["retrieval"]["reranker"] is False


def test_static_configuration_freezes_optional_nonofficial_metric(tmp_path):
    metric = run_reading_eval._ZH_T2S_F1_AUXILIARY
    manifest = {
        "files": {"inputs.jsonl": {"sha256": "a" * 64}},
        "cases": [],
        "auxiliary_metrics": [metric],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    static = run_reading_eval.static_configuration(tmp_path, manifest)

    assert static["auxiliary_metrics"] == [metric]
    with pytest.raises(ValueError, match="unsupported or invalid frozen auxiliary metrics"):
        run_reading_eval.static_configuration(
            tmp_path, {**manifest, "auxiliary_metrics": ["other"]}
        )


def test_static_configuration_pins_integrated_agent_limits_and_primary_arms(tmp_path):
    manifest = {
        "files": {"inputs.jsonl": {"sha256": "a" * 64}},
        "cases": [{"case_id": f"case-{index}"} for index in range(24)],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    static = run_reading_eval.static_configuration(
        tmp_path,
        manifest,
        selected_case_ids=[f"case-{index}" for index in range(24)],
        selected_arms=["fixed-rag-reranked", "single-agent"],
    )

    assert static["run_selection"]["arms"] == ["fixed-rag-reranked", "single-agent"]
    assert static["runner_version"] == "frozen-reading-eval-v30-agent-control-followup-v1"
    assert static["agent"]["architecture_by_arm"] == {
        "fixed-rag-reranked": "fixed_hybrid_retrieval_then_one_structured_answer",
        "single-agent": "single_agent_navigation_then_shared_answer_writer",
    }
    assert static["retrieval"]["reranker"] is True
    assert static["agent"]["resource_profiles_by_arm"]["fixed-rag-reranked"] == {
        "model_calls_per_case_max": 1,
        "data_tool_events_per_case_max": 0,
        "total_input_tokens_per_case_max": 12_000,
        "total_output_tokens_per_case_max": 3_072,
        "initial_evidence": "shared_hybrid_then_rerank",
    }
    integrated = static["agent"]["resource_profiles_by_arm"]["single-agent"]
    assert integrated["model_call_limit_per_case"] is None
    assert integrated["common_writer_calls_per_case"] == 1
    assert integrated["call_input_tokens_max"] == 12_000
    assert integrated["call_output_tokens_max"] == 3_072
    assert integrated["data_tool_event_limit_per_case"] is None
    assert integrated["total_input_tokens_per_case_limit"] is None
    assert integrated["total_output_tokens_per_case_limit"] is None
    assert integrated["case_deadline_seconds"] == 600
    assert static["agent"]["shared_budget_by_arm"]["single-agent"] is True
    assert "delegated-agent" not in static["run_selection"]["arms"]
    assert static["answer_call_budget"] == {
        "cases": 24,
        "per_case_model_call_limit_by_arm": {"fixed-rag-reranked": 1, "single-agent": None},
        "single_agent_per_case_call_limit": None,
        "case_deadline_seconds": 600,
        "batch_generation_dispatch_limit": "frozen_global_and_phase_ledgers",
    }


def test_three_workflow_static_budget_is_bounded_before_execution(tmp_path):
    manifest = {"files": {"inputs.jsonl": {"sha256": "a" * 64}}, "cases": []}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    arms = ["fixed-rag-reranked", "fixed-rag-followup-once", "single-agent"]

    static = run_reading_eval.static_configuration(
        tmp_path,
        manifest,
        selected_case_ids=[f"case-{index}" for index in range(24)],
        selected_arms=arms,
    )

    assert static["answer_call_budget"] == {
        "cases": 24,
        "per_case_model_call_limit_by_arm": {
            "fixed-rag-reranked": 1,
            "fixed-rag-followup-once": 2,
            "single-agent": None,
        },
        "single_agent_per_case_call_limit": None,
        "case_deadline_seconds": 600,
        "batch_generation_dispatch_limit": "frozen_global_and_phase_ledgers",
    }
    assert static["retrieval"]["reranker"] is True

    with pytest.raises(ValueError, match="24-case development ceiling"):
        run_reading_eval.static_configuration(
            tmp_path,
            manifest,
            selected_case_ids=[f"case-{index}" for index in range(25)],
            selected_arms=arms,
        )


def test_quality_v1_static_budget_accounts_for_the_shared_planner(tmp_path):
    manifest = {
        "files": {"inputs.jsonl": {"sha256": "a" * 64}},
        "cases": [{"case_id": f"case-{index}"} for index in range(72)],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    static = run_reading_eval.static_configuration(
        tmp_path,
        manifest,
        selected_case_ids=[f"case-{index}" for index in range(72)],
        selected_arms=list(run_reading_eval.QUALITY_V1_ARMS),
    )

    assert static["reading_plan_call_budget"] == {
        "planner_calls_max": 72,
        "planner_calls_limit": 72,
        "planner_phase_generation_dispatch_limit": 72,
        "model_calls_per_agent_case_limit": None,
        "case_deadline_seconds": 600,
    }
    assert static["reading_plan"]["context_group_limit"] == 100
    assert static["reading_plan"]["fused_candidate_count"] == 100
    assert static["retrieval"]["candidates_per_branch"] == 50
    assert static["retrieval"]["final_k"] == 100
    assert static["reading_plan"]["retrieval_candidates_per_branch"] == 50
    assert static["reading_plan"]["retrieval_candidate_count_per_query"] == 100
    assert static["answer_call_budget"]["per_case_model_call_limit_by_arm"] == {
        "fixed-rag": 1,
        "fixed-rag-followup-once": 2,
        "decomposition-fixed-rag": 1,
        "decomposition-agent": None,
    }
    assert static["answer_call_budget"]["single_agent_per_case_call_limit"] is None
    assert static["answer_call_budget"]["phase_generation_dispatch_limit"] == 720
    assert static["reading_plan"]["planned_query_input_tokens_max"] == 25_000

    with pytest.raises(ValueError, match="72-case ceiling"):
        run_reading_eval.static_configuration(
            tmp_path,
            manifest,
            selected_case_ids=[f"case-{index}" for index in range(73)],
            selected_arms=list(run_reading_eval.QUALITY_V1_ARMS),
        )


def test_quality_comparison_can_freeze_shared_initial_reranking(tmp_path):
    manifest = {
        "files": {"inputs.jsonl": {"sha256": "a" * 64}},
        "cases": [{"case_id": f"case-{index}"} for index in range(72)],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    static = run_reading_eval.static_configuration(
        tmp_path,
        manifest,
        selected_case_ids=[f"case-{index}" for index in range(72)],
        selected_arms=list(run_reading_eval.QUALITY_V1_ARMS),
        shared_initial_reranker=True,
    )

    assert static["retrieval"]["reranker"] is False
    assert static["retrieval"]["shared_initial_reranker"] is True
    assert static["retrieval"]["reranker_scope"].endswith("Agent follow-up search stays hybrid")
    assert static["models"]["reranker"]["name"] == "Qwen3-Reranker-0.6B-Q8_0"
    assert static["preparation_budget"]["reranker_pairs_max"] == 7_200
    assert static["preparation_budget"]["reranker_pair_tokens_max"] == 30_000_000

    with pytest.raises(ValueError, match="reserved for the four reading-plan arms"):
        run_reading_eval.static_configuration(
            tmp_path,
            manifest,
            selected_case_ids=[f"case-{index}" for index in range(72)],
            selected_arms=["fixed-rag", "single-agent"],
            shared_initial_reranker=True,
        )


def test_shared_initial_reranker_counts_completed_pairs_once_and_reuses_manifest(
    tmp_path, monkeypatch
):
    from page_cite_rag import reranker_server

    batch_root = tmp_path / "quality-repair-closure-v2"
    run_root = batch_root / "runs" / "formal-v11"
    planned_dir = run_root / "planned-retrieval"
    planned_dir.mkdir(parents=True)
    planned_path = planned_dir / "results.jsonl"
    planned_path.write_text("{}\n", encoding="utf-8")
    progress_path = batch_root / "progress.json"
    progress_path.write_text(
        json.dumps({"counters": {"reranker_pairs": 0, "reranker_pair_tokens": 0}}),
        encoding="utf-8",
    )

    manifest = {
        "files": {"inputs.jsonl": {"sha256": "a" * 64}},
        "cases": [{"case_id": "case-1"}],
    }
    (tmp_path / "dataset" / "manifest.json").parent.mkdir(parents=True)
    (tmp_path / "dataset" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    static = run_reading_eval.static_configuration(
        tmp_path / "dataset",
        manifest,
        selected_case_ids=["case-1"],
        selected_arms=list(run_reading_eval.QUALITY_V1_ARMS),
        shared_initial_reranker=True,
    )
    monkeypatch.setattr(run_reading_eval, "GPU_UUID", "GPU-test")

    document_id = "a" * 64
    blocks = tuple(
        Block(
            document_id=document_id,
            block_id=hashlib.sha256(f"block-{index}".encode()).hexdigest(),
            ordinal=index,
            unit_index=index,
            line_start=index,
            line_end=index,
            char_start=0,
            char_end=6,
            text=f"text {index}",
        )
        for index in (1, 2)
    )

    def search_result(index_id):
        return SearchResult(
            document_id=document_id,
            index_id=index_id,
            query="question",
            hits=tuple(
                SearchHit(rank=rank, score=1 / rank, block=block)
                for rank, block in enumerate(blocks, 1)
            ),
            elapsed_ms=1,
        )

    planned = {
        "case-1": {
            "status": "complete",
            "document_id": document_id,
            "original_query_candidate_result": search_result("original").model_dump(mode="json"),
            "candidate_result": search_result("decomposed").model_dump(mode="json"),
        }
    }
    monkeypatch.setattr(
        run_reading_eval,
        "_load_prepared",
        lambda *args, **kwargs: (object(), None),
    )
    monkeypatch.setattr(
        run_reading_eval,
        "_load_planned_retrieval",
        lambda *args, **kwargs: planned,
    )

    class FakeRanker:
        dispatches = 0

        def __init__(self, *args, **kwargs):
            self.last_pair_tokens = ()
            self.last_batch_sizes = ()
            self.last_tokenization_ms = 0.5
            self.last_forward_ms = 1.0
            self.last_total_ms = 1.5

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def rerank(self, query, documents):
            type(self).dispatches += 1
            self.last_pair_tokens = tuple(7 for _ in documents)
            self.last_batch_sizes = (len(documents),)
            return [(index, float(len(documents) - index)) for index in range(len(documents))]

        def diagnostics(self):
            return {"temperature_c": 55, "memory_used_mib": 1000}

    monkeypatch.setattr(reranker_server, "NativeRerankerServer", FakeRanker)
    case = {"case_id": "case-1", "question": "question"}

    first = run_reading_eval._rank_planned_initial_retrieval(run_root, static, [case])
    after_first = json.loads(progress_path.read_text(encoding="utf-8"))
    second = run_reading_eval._rank_planned_initial_retrieval(run_root, static, [case])
    after_second = json.loads(progress_path.read_text(encoding="utf-8"))

    assert first["candidate_pairs"] == 4
    assert first["pair_tokens"] == 28
    assert second == first
    assert FakeRanker.dispatches == 2
    assert after_first["counters"]["reranker_pairs"] == 4
    assert after_first["counters"]["reranker_pair_tokens"] == 28
    assert after_second["counters"]["reranker_pairs"] == 4
    assert after_second["counters"]["reranker_pair_tokens"] == 28


def test_quality_v1_global_call_ledger_stops_before_dispatch_ceiling(tmp_path):
    from page_cite_rag.generation import ModelError

    class Schema(BaseModel):
        value: str

    wrapper = run_reading_eval._CountingModelClient(
        object(), global_call_ledger=tmp_path / "global-calls.jsonl", global_call_limit=1
    )
    wrapper.begin_case(call_context={"phase": "plan", "arm": "shared-planner", "case_id": "c1"})

    assert wrapper._reserve_global_call(Schema) == 1
    with pytest.raises(ModelError, match="model-call ceiling"):
        wrapper._reserve_global_call(Schema)


def test_v4_writer_reservations_cover_active_followup_and_agent_writer(tmp_path):
    static = {
        "run_selection": {
            "arms": list(run_reading_eval.QUALITY_V1_ARMS),
            "case_ids": ["c1", "c2"],
        }
    }

    future_only = run_reading_eval._remaining_quality_writer_reservation(
        tmp_path,
        static,
        current_arm="fixed-rag",
        current_case_id="c1",
    )
    followup_first_answer = run_reading_eval._remaining_quality_writer_reservation(
        tmp_path,
        static,
        current_arm="fixed-rag-followup-once",
        current_case_id="c1",
        current_case_remaining_calls=1,
    )
    agent_navigation = run_reading_eval._remaining_quality_writer_reservation(
        tmp_path,
        static,
        current_arm="decomposition-agent",
        current_case_id="c1",
        current_case_remaining_calls=1,
    )
    agent_writer = run_reading_eval._remaining_quality_writer_reservation(
        tmp_path,
        static,
        current_arm="decomposition-agent",
        current_case_id="c1",
    )

    assert future_only == 11
    assert followup_first_answer == future_only
    assert agent_navigation == future_only
    assert agent_writer == future_only - 1


def test_counting_client_reservation_can_drop_after_first_case_call():
    wrapper = run_reading_eval._CountingModelClient(object())
    wrapper.begin_case(
        required_remaining_calls=lambda: max(
            0, 1 - (wrapper.generate_calls - wrapper._case_start_calls)
        )
    )

    assert wrapper._required_remaining_calls() == 1
    wrapper.generate_calls += 1
    assert wrapper._required_remaining_calls() == 0


def test_quality_v1_global_call_ledger_excludes_audited_pre_dispatch_voids(tmp_path):
    from page_cite_rag.generation import ModelError

    class Schema(BaseModel):
        value: str

    ledger = tmp_path / "global-calls.jsonl"
    run_reading_eval.atomic_jsonl_once(
        ledger,
        [
            {"call_index": 1, "state": "reserved_before_dispatch"},
            {"state": "voided_before_dispatch", "voids_call_indices": [1]},
        ],
    )
    wrapper = run_reading_eval._CountingModelClient(
        object(), global_call_ledger=ledger, global_call_limit=1
    )

    assert wrapper._reserve_global_call(Schema) == 2
    with pytest.raises(ModelError, match="model-call ceiling"):
        wrapper._reserve_global_call(Schema)


def test_v2_generation_budget_covers_non_planner_phases_and_tracks_usage(tmp_path):
    class Schema(BaseModel):
        value: str

    batch_root = tmp_path / "quality-repair-closure-v2"
    batch_root.mkdir()
    progress = {
        "revision": "quality-repair-closure-v2",
        "state": "in_progress",
        "scope": {"maximum_new_generation_dispatches": 696},
        "counters": {
            "generation_dispatches": 0,
            "generation_input_tokens": 0,
            "generation_output_tokens": 0,
            "unknown_generation_usage": 0,
        },
    }
    (batch_root / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    answer_run = batch_root / "runs" / "answer"
    ledger, limit = run_reading_eval._global_generation_budget(answer_run, {"reading_plan": None})

    assert ledger == batch_root / "global-model-calls.jsonl"
    assert limit == 696
    wrapper = run_reading_eval._CountingModelClient(
        object(), global_call_ledger=ledger, global_call_limit=limit
    )
    wrapper.begin_case(call_context={"phase": "answer", "arm": "fixed-rag", "case_id": "c1"})
    assert wrapper._reserve_global_call(Schema) == 1

    after_reservation = json.loads((batch_root / "progress.json").read_text(encoding="utf-8"))
    assert after_reservation["counters"]["generation_dispatches"] == 1
    assert after_reservation["counters"]["unknown_generation_usage"] == 1

    wrapper._record_global_usage(1, {"stats": {"prompt_eval_count": 321, "eval_count": 23}})
    after_usage = json.loads((batch_root / "progress.json").read_text(encoding="utf-8"))
    assert after_usage["counters"]["generation_input_tokens"] == 321
    assert after_usage["counters"]["generation_output_tokens"] == 23
    assert after_usage["counters"]["unknown_generation_usage"] == 0


def test_v2_generation_budget_rejects_progress_with_changed_scope(tmp_path):
    batch_root = tmp_path / "quality-repair-closure-v2"
    batch_root.mkdir()
    (batch_root / "progress.json").write_text(
        json.dumps(
            {
                "revision": "quality-repair-closure-v2",
                "state": "in_progress",
                "scope": {"maximum_new_generation_dispatches": 697},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="approved frozen limit"):
        run_reading_eval._global_generation_budget(
            batch_root / "runs" / "answer", {"reading_plan": None}
        )


def test_v3_generation_budget_enforces_total_and_phase_limits(tmp_path):
    from page_cite_rag.generation import ModelError

    class Schema(BaseModel):
        value: str

    batch_root = tmp_path / "agent-quality-closure-v3"
    batch_root.mkdir()
    progress = {
        "revision": "agent-quality-closure-v3",
        "state": "in_progress",
        "scope": {
            "maximum_new_generation_dispatches": 1134,
            "phase_generation_dispatch_limits": {
                "protocol": 1,
                "development": 42,
                "regression_answers": 720,
                "regression_planners": 72,
                "new_holdout_planners_and_two_flows": 288,
            },
        },
        "counters": {},
    }
    (batch_root / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    protocol_run = batch_root / "phase1-protocol" / "run"
    ledger, total_limit = run_reading_eval._global_generation_budget(
        protocol_run, {"reading_plan": None}
    )
    phase = run_reading_eval._global_generation_phase(protocol_run, "run")
    wrapper = run_reading_eval._CountingModelClient(
        object(),
        global_call_ledger=ledger,
        global_call_limit=total_limit,
        global_call_phase=phase,
    )
    wrapper.begin_case(call_context={"phase": "answer", "case_id": "case-1"})

    assert total_limit == 1134
    assert phase == "protocol"
    assert wrapper._reserve_global_call(Schema) == 1
    with pytest.raises(ModelError, match="phase generation ceiling"):
        wrapper._reserve_global_call(Schema)

    row = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    updated = json.loads((batch_root / "progress.json").read_text(encoding="utf-8"))
    assert row["budget_phase"] == "protocol"
    assert row["phase"] == "answer"
    assert updated["counters"]["generation_dispatches_by_phase"]["protocol"] == 1


def test_v3_ledger_corrects_duplicate_writer_observer_reservation(tmp_path):
    class Schema(BaseModel):
        value: str

    batch_root = tmp_path / "agent-quality-closure-v3"
    batch_root.mkdir()
    progress = {
        "revision": "agent-quality-closure-v3",
        "state": "in_progress",
        "scope": {
            "maximum_new_generation_dispatches": 1134,
            "phase_generation_dispatch_limits": {
                "protocol": 3,
                "development": 42,
                "regression_answers": 720,
                "regression_planners": 72,
                "new_holdout_planners_and_two_flows": 288,
            },
        },
        "counters": {},
    }
    (batch_root / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    ledger = batch_root / "global-model-calls.jsonl"
    rows = []
    for index in (1, 2, 3):
        rows.append(
            {
                "call_index": index,
                "state": "reserved_before_dispatch",
                "budget_phase": "protocol",
                "arm": "single-agent",
                "case_id": "case-1",
            }
        )
        rows.append(
            {
                "call_index": index,
                "state": "usage_recorded",
                "input_tokens": 100 if index == 1 else 200,
                "output_tokens": 10 if index == 1 else 20,
                "usage_unknown": False,
            }
        )
    rows.append(
        {
            "state": "duplicate_observer_correction",
            "duplicate_call_index": 2,
            "actual_call_index": 3,
        }
    )
    run_reading_eval.atomic_jsonl_once(ledger, rows)
    wrapper = run_reading_eval._CountingModelClient(
        object(),
        global_call_ledger=ledger,
        global_call_limit=1134,
        global_call_phase="protocol",
    )

    assert wrapper._reserve_global_call(Schema) == 4
    updated = json.loads((batch_root / "progress.json").read_text(encoding="utf-8"))
    assert updated["counters"]["generation_dispatches"] == 3
    assert updated["counters"]["generation_input_tokens"] == 300
    assert updated["counters"]["generation_output_tokens"] == 30
    assert updated["counters"]["generation_dispatches_by_phase"]["protocol"] == 3


def test_v4_progress_total_dispatches_match_global_ledger(tmp_path):
    class Schema(BaseModel):
        value: str

    batch_root = tmp_path / "agent-production-alignment-v4"
    batch_root.mkdir()
    progress = {
        "revision": "agent-production-alignment-v4",
        "state": "in_progress",
        "generation_dispatch_limits": {
            "protocol": 2,
            "development": 3,
            "regression_planners": 4,
            "regression_answers": 5,
            "new_diagnostics": 6,
            "total": 20,
        },
        "counters": {},
    }
    (batch_root / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    ledger = batch_root / "global-model-calls.jsonl"
    wrapper = run_reading_eval._CountingModelClient(
        object(),
        global_call_ledger=ledger,
        global_call_limit=20,
        global_call_phase="development",
    )
    wrapper.begin_case(call_context={"phase": "answer", "case_id": "case-1"})

    assert wrapper._reserve_global_call(Schema) == 1
    updated = json.loads((batch_root / "progress.json").read_text(encoding="utf-8"))
    assert updated["counters"]["generation_dispatches"] == 1
    assert updated["counters"]["generation_dispatches_by_phase"] == {
        "protocol": 0,
        "development": 1,
        "regression_planners": 0,
        "regression_answers": 0,
        "new_diagnostics": 0,
        "total": 1,
    }


def test_agent_followup_generation_budget_uses_its_own_approved_phase_ledger(tmp_path):
    class Schema(BaseModel):
        value: str

    parent = tmp_path / "agent-production-alignment-v4"
    batch_root = parent / "agent-control-followup-v1"
    batch_root.mkdir(parents=True)
    limits = {
        "development": 123,
        "regression_planners": 72,
        "regression_answers": 864,
        "new_diagnostics": 240,
        "total": 1299,
    }
    progress = {
        "revision": "agent-control-followup-v1",
        "state": "in_progress",
        "generation_dispatch_limits": limits,
        "counters": {},
    }
    (batch_root / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    development_run = batch_root / "development" / "agent-six"
    ledger, total_limit = run_reading_eval._global_generation_budget(
        development_run, {"reading_plan": None}
    )
    phase = run_reading_eval._global_generation_phase(development_run, "run")
    wrapper = run_reading_eval._CountingModelClient(
        object(),
        global_call_ledger=ledger,
        global_call_limit=total_limit,
        global_call_phase=phase,
    )

    assert run_reading_eval._generation_batch_root(development_run) == batch_root
    assert ledger == batch_root / "global-model-calls.jsonl"
    assert total_limit == 1299
    assert phase == "development"
    assert wrapper._reserve_global_call(Schema) == 1
    updated = json.loads((batch_root / "progress.json").read_text(encoding="utf-8"))
    assert updated["counters"]["generation_dispatches_by_phase"]["development"] == 1

    regression_plan = batch_root / "regression" / "plan"
    assert run_reading_eval._global_generation_phase(regression_plan, "plan") == (
        "regression_planners"
    )
    assert (
        run_reading_eval._global_generation_phase(batch_root / "regression" / "answers", "run")
        == "regression_answers"
    )

    single_agent_run = batch_root / "development" / "agent-six"
    selection = {
        "run_selection": {
            "arms": ["single-agent"],
            "case_ids": ["case-1", "case-2", "case-3"],
        }
    }
    (single_agent_run / "single-agent" / ".attempts").mkdir(parents=True)
    (single_agent_run / "single-agent" / ".attempts" / "case-3.json").write_text("{}")
    assert (
        run_reading_eval._remaining_quality_writer_reservation(
            single_agent_run,
            selection,
            current_arm="single-agent",
            current_case_id="case-1",
            current_case_remaining_calls=1,
        )
        == 2
    )


def test_static_configuration_freezes_four_way_retrieval_grid(tmp_path):
    manifest = {"files": {"inputs.jsonl": {"sha256": "a" * 64}}, "cases": []}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    static = run_reading_eval.static_configuration(tmp_path, manifest, retrieval_grid=True)

    assert static["run_selection"]["arms"] == ["fixed-rag", "fixed-rag-reranked"]
    assert static["models"]["reranker"] is not None
    assert static["retrieval"]["reranker_request"] == {
        "documents_per_request": 8,
        "pair_max_tokens": 4096,
        "request_timeout_seconds": 60,
    }
    assert static["retrieval"]["query_rewrite"]["max_calls"] == 240
    assert static["retrieval"]["query_rewrite"]["num_predict"] == 128
    assert static["retrieval"]["retrieval_grid"]["variants"] == [
        "original_hybrid",
        "rewrite_bm25_only",
        "rewrite_dense_only",
        "rewrite_both",
    ]


def test_eval_top_k_keeps_full_source_mapping_for_selected_chunks():
    document_id = "a" * 64
    blocks = tuple(
        Block(
            block_id=hashlib.sha256(f"source-{ordinal}".encode()).hexdigest(),
            document_id=document_id,
            unit_index=1,
            ordinal=ordinal,
            text=f"source {ordinal}",
            char_start=0,
            char_end=len(f"source {ordinal}"),
            line_start=ordinal,
            line_end=ordinal,
        )
        for ordinal in range(1, 5)
    )
    groups = (
        SearchContextGroup(
            chunk_index=0,
            text="chunk one",
            source_refs=("#/texts/0", "#/texts/1"),
            source_block_ids=(blocks[0].block_id, blocks[1].block_id),
        ),
        SearchContextGroup(
            chunk_index=1,
            text="chunk two",
            source_refs=("#/texts/2", "#/texts/3"),
            source_block_ids=(blocks[2].block_id, blocks[3].block_id),
        ),
    )
    result = SearchResult(
        document_id=document_id,
        index_id="test-index",
        query="question",
        hits=tuple(
            SearchHit(rank=index, score=1 / index, block=block)
            for index, block in enumerate(blocks, 1)
        ),
        elapsed_ms=1.0,
        context_groups=groups,
    )

    limited = run_reading_eval._limit_search_result_to_context_groups(result, 1)

    assert limited.context_groups == (groups[0],)
    assert [hit.block.block_id for hit in limited.hits] == list(groups[0].source_block_ids)
    assert [hit.rank for hit in limited.hits] == [1, 2]


def test_static_configuration_records_no_tools_same_context_diagnostic(tmp_path):
    manifest = {"files": {"inputs.jsonl": {"sha256": "a" * 64}}, "cases": []}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    static = run_reading_eval.static_configuration(
        tmp_path,
        manifest,
        selected_case_ids=["exposed-case"],
        selected_arms=["fixed-rag", "single-agent-no-tools"],
    )

    assert static["agent"]["architecture_by_arm"]["single-agent-no-tools"] == (
        "single_agent_structured_answer_over_frozen_initial_evidence"
    )
    profile = static["agent"]["resource_profiles_by_arm"]["single-agent-no-tools"]
    assert profile["model_calls_per_case_max"] == 1
    assert profile["data_tool_events_per_case_max"] == 0
    assert static["agent"]["shared_budget_by_arm"]["single-agent-no-tools"] is True
    assert static["run_selection"]["arms"] == ["fixed-rag", "single-agent-no-tools"]


def test_static_configuration_pins_selected_generator_bundle(tmp_path, monkeypatch):
    manifest = {"files": {"inputs.jsonl": {"sha256": "a" * 64}}, "cases": []}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        run_reading_eval,
        "GPU_UUID",
        "GPU-00000000-0000-0000-0000-000000000000",
    )

    static = run_reading_eval.static_configuration(
        tmp_path, manifest, selected_arms=["fixed-rag"], generator_model_size="4b"
    )
    config = run_reading_eval._llama_config(static)

    assert static["models"]["generator"] == {
        "name": "Qwen3.5-4B-Q4_K_M",
        "size": "4b",
        "revision": "e87f176479d0855a907a41277aca2f8ee7a09523",
        "file": "Qwen3.5-4B-Q4_K_M.gguf",
        "sha256": "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
        "llama_commit": run_reading_eval.LLAMA_COMMIT,
    }
    assert config.model_size == "4b"
    assert config.bundle_dir.name == "7fe450e1-e87f1764"


def test_official_longbench_adapter_rejects_4b_model(tmp_path):
    manifest = {"files": {"inputs.jsonl": {"sha256": "a" * 64}}, "cases": []}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="uses the pinned 9B model"):
        run_reading_eval.static_configuration(
            tmp_path,
            manifest,
            selected_arms=[run_reading_eval.LONG_BENCH_OFFICIAL_ARM],
            generator_model_size="4b",
        )


def test_prompt_comparison_is_frozen_against_the_current_fixed_rag(tmp_path):
    from page_cite_rag.full_answer import PROMPT_VERSION

    manifest = {"files": {"inputs.jsonl": {"sha256": "a" * 64}}, "cases": []}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    arms = ["fixed-rag", "fixed-rag-prompt-task-aligned"]

    static = run_reading_eval.static_configuration(tmp_path, manifest, selected_arms=arms)

    assert static["run_selection"]["arms"] == arms
    assert static["answer"]["prompt_revisions_by_arm"] == {
        arms[0]: "contract-consistent",
        arms[1]: "task-aligned-v5",
    }
    assert static["answer"]["contracts_by_arm"] == {
        arms[0]: PROMPT_VERSION,
        arms[1]: "complete-cited-answer-v5-task-aligned",
    }


def test_prompt_comparison_uses_only_evidence_admitted_by_both_prompts():
    common = run_reading_eval._common_admitted_block_ids(
        {
            "contract-consistent": {
                "admitted_block_ids": ["first", "second", "third"],
            },
            "task-aligned-v5": {
                "admitted_block_ids": ["first", "second"],
            },
        },
        ["first", "second", "third", "fourth"],
    )

    assert common == ["first", "second"]


def test_longbench_official_prompt_matches_pinned_rag_template():
    prompt = run_reading_eval._longbench_v2_official_prompt(
        "Retrieved chunk 1: A $5 fee applies.",
        "Which answer is supported?",
        {"A": "one", "B": "two", "C": "three", "D": "four"},
    )

    assert prompt == (
        "Please read the following retrieved text chunks and answer the question below.\n\n"
        "<text>\nRetrieved chunk 1: A $5 fee applies.\n</text>\n\n"
        "What is the correct answer to this question: Which answer is supported?\n"
        "Choices:\n(A) one\n(B) two\n(C) three\n(D) four\n\n"
        'Format your response as follows: "The correct answer is (insert answer here)".\n'
    )
    prompt_sha = hashlib.sha256(run_reading_eval.LONG_BENCH_V2_RAG_PROMPT.encode()).hexdigest()
    assert prompt_sha == run_reading_eval.LONG_BENCH_V2_RAG_PROMPT_SHA256


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("The correct answer is (B).", "B"),
        ("The correct answer is C because...", "C"),
        ("**The correct answer is (D)**", "D"),
        ("I choose B.", None),
        ("The correct answer is (E).", None),
    ],
)
def test_longbench_official_extractor_matches_upstream(response, expected):
    assert run_reading_eval._longbench_v2_official_choice(response) == expected


def test_official_longbench_arm_requires_an_isolated_protocol(tmp_path):
    manifest = {"files": {"inputs.jsonl": {"sha256": "a" * 64}}, "cases": []}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="must run in its own frozen batch"):
        run_reading_eval.static_configuration(
            tmp_path,
            manifest,
            selected_arms=[run_reading_eval.LONG_BENCH_OFFICIAL_ARM, "fixed-rag"],
        )

    static = run_reading_eval.static_configuration(
        tmp_path,
        manifest,
        selected_arms=[run_reading_eval.LONG_BENCH_OFFICIAL_ARM],
        document_embedding_token_limit=23_000_000,
        query_embedding_token_limit=35_000,
    )
    assert static["answer"]["sampling_profile"] == "longbench-official"
    assert static["answer"]["num_predict"] == 128
    assert static["answer"]["prompt_template"]["sha256"] == (
        run_reading_eval.LONG_BENCH_V2_RAG_PROMPT_SHA256
    )
    assert static["preparation_budget"]["document_embedding_input_tokens_max"] == 23_000_000
    assert static["preparation_budget"]["query_embedding_input_tokens_max"] == 35_000
    assert static["preparation_budget"]["reranker_pair_tokens_max"] == 200_000_000
    assert static["embedding_input"]["document_max_tokens"] == 512
    assert static["embedding_input"]["query_max_tokens"] == 1024
    assert static["embedding_input"]["request_timeout_seconds"] == 60
    with pytest.raises(ValueError, match="global ceiling"):
        run_reading_eval.static_configuration(
            tmp_path,
            manifest,
            selected_arms=[run_reading_eval.LONG_BENCH_OFFICIAL_ARM],
            document_embedding_token_limit=40_000_001,
        )
    with pytest.raises(ValueError, match="global ceiling"):
        run_reading_eval.static_configuration(
            tmp_path,
            manifest,
            selected_arms=[run_reading_eval.LONG_BENCH_OFFICIAL_ARM],
            query_embedding_token_limit=100_001,
        )


def test_prompt_comparison_rejects_reordered_common_evidence():
    with pytest.raises(ValueError, match="different orders"):
        run_reading_eval._common_admitted_block_ids(
            {
                "contract-consistent": {"admitted_block_ids": ["first", "second"]},
                "task-aligned-v5": {"admitted_block_ids": ["second", "first"]},
            },
            ["first", "second"],
        )


def test_qasper_document_rebuild_restores_section_titles(tmp_path):
    store = SQLiteDocumentStore(tmp_path / "qasper.sqlite3")
    case = {
        "case_id": "qasper-question",
        "dataset": "qasper",
        "source_id": "paper",
        "language": "en",
        "question": "Question?",
        "source": {
            "format": "qasper_paragraphs",
            "section_names": ["Introduction", "Methods"],
            "paragraphs": [
                {
                    "passage_id": "paper/s0001/p0000",
                    "section_index": 1,
                    "text": "The method uses two stages.",
                }
            ],
        },
    }

    document_id, block_to_passage = run_reading_eval._case_document(store, case)
    outline = SourceOutline.model_validate_json(
        store.get_retrieval_manifest(document_id, OUTLINE_KIND)
    )

    assert outline.sections[0].title == "Methods"
    assert block_to_passage[store.get_blocks(document_id)[0].block_id] == "paper/s0001/p0000"


def test_identical_benchmark_sources_reuse_stable_document_metadata(tmp_path):
    from page_cite_rag.text import iter_blocks

    class StableTestChunker:
        def bind_document(self, document):
            parser_version = f"{document.parser_version};token-blocks-test-v1"
            document_id = hashlib.sha256(
                f"{document.source_sha256}\0{document.parser_name}\0{parser_version}".encode()
            ).hexdigest()
            return document.model_copy(
                update={"document_id": document_id, "parser_version": parser_version}
            )

        def __call__(self, document, unit, first_ordinal):
            yield from iter_blocks(document, unit, first_ordinal)

    text = "The same source document is referenced by multiple questions."
    base = {
        "dataset": "longbench-v2-single-document-qa",
        "language": "en",
        "source": {"format": "plain_text", "text": text},
    }
    store = SQLiteDocumentStore(tmp_path / "same-source.sqlite3")
    chunker = StableTestChunker()

    first_id, _ = run_reading_eval._case_document(
        store, base | {"case_id": "case-one", "source_id": "source-one"}, chunker=chunker
    )
    second_id, _ = run_reading_eval._case_document(
        store, base | {"case_id": "case-two", "source_id": "source-two"}, chunker=chunker
    )

    assert first_id == second_id
    record = store.get_document(first_id)
    assert record is not None
    expected_hash = hashlib.sha256(text.encode()).hexdigest()
    assert record.source_name == f"benchmark-{expected_hash[:16]}.txt"
    assert len(store.get_blocks(first_id)) == 1


def test_longbench_chinese_f1_matches_official_punctuation_normalization():
    score = run_reading_eval._longbench_zh_f1("逐字引用，保留原文！", ["逐字引用保留原文"])

    assert score == 1.0


def test_longbench_chinese_f1_removes_official_corner_brackets():
    score = run_reading_eval._longbench_zh_f1("【中文】引用", ["中文引用"])

    assert score == 1.0


def test_longbench_chinese_f1_keeps_official_opening_book_title_mark():
    # The pinned upstream scorer removes 》 but preserves 《.
    score = run_reading_eval._longbench_zh_f1("《中文》", ["中文"])

    assert score == pytest.approx(2 / 3)


def test_nonofficial_script_normalized_f1_preserves_official_score():
    prediction, references = "廈門大學", ["厦门大学。"]

    assert run_reading_eval._longbench_zh_f1(prediction, references) == 0.0
    assert run_reading_eval._longbench_zh_t2s_f1_nonofficial(prediction, references) == 1.0


def test_nonofficial_normalization_is_frozen_and_applies_to_both_arms(tmp_path):
    metric = run_reading_eval._ZH_T2S_F1_AUXILIARY
    cases = [
        {"case_id": "native", "dataset": "longbench-v1-multifieldqa-zh"},
        {"case_id": "cross", "dataset": "longbench-v1-multifieldqa-zh"},
    ]
    manifest = {"cases": cases, "auxiliary_metrics": [metric]}
    inputs = [
        {"case_id": "native", "language": "zh-Hant", "source_language": "zh-Hant"},
        {"case_id": "cross", "language": "en", "source_language": "zh-Hant"},
    ]
    gold = [{"case_id": case["case_id"], "answers": ["厦门大学。"]} for case in cases]
    arms = ("fixed-rag", "single-agent")
    for arm in arms:
        directory = tmp_path / arm
        directory.mkdir()
        for case in cases:
            case_id = case["case_id"]
            (directory / f"{case_id}.json").write_text(
                json.dumps(
                    {
                        "arm": arm,
                        "case_id": case_id,
                        "execution": "ok",
                        "result": {"answer": "廈門大學"},
                    }
                ),
                encoding="utf-8",
            )
    with pytest.raises(ValueError, match="frozen run configuration"):
        run_reading_eval.score_run(tmp_path, manifest, inputs, gold, arms=arms)
    (tmp_path / "configuration.json").write_text(
        json.dumps({"static": {"auxiliary_metrics": []}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="differ from the frozen run configuration"):
        run_reading_eval.score_run(tmp_path, manifest, inputs, gold, arms=arms)
    (tmp_path / "configuration.json").write_text(
        json.dumps({"static": {"auxiliary_metrics": [metric]}}), encoding="utf-8"
    )

    report = run_reading_eval.score_run(tmp_path, manifest, inputs, gold, arms=arms)
    assert report["auxiliary_metrics"] == [metric]
    for arm in arms:
        rows = {
            row["case_id"]: row
            for dataset in report["arms"][arm]["datasets"].values()
            for row in dataset["per_case"]
        }
        assert rows["native"]["answer_f1"] == 0.0
        assert rows["native"][metric] == 1.0
        assert rows["cross"]["answer_f1"] is None
        assert rows["cross"][metric] is None


def test_embedding_budget_is_checked_before_dispatch():
    class FakeTransport:
        def __init__(self):
            self.dispatches = 0

        def token_ids(self, text):
            return list(range(len(text)))

        def embed(self, texts):
            self.dispatches += 1
            return texts

    transport = FakeTransport()
    counter = run_reading_eval._CountingTransport(transport, document_token_limit=3)

    with pytest.raises(ValueError, match="budget exceeded before dispatch"):
        counter.embed(["four"])

    assert transport.dispatches == 0
    assert counter.document_tokens == 0


@pytest.mark.parametrize(
    "checkpoint_state",
    [
        "valid",
        "valid_lower_embedding_budget",
        "valid_reranker_disabled",
        "over_embedding_budget",
        "wrong_model",
        "corrupt_vectors",
    ],
)
def test_prepared_checkpoint_reuses_only_verified_artifacts(
    tmp_path, monkeypatch, checkpoint_state
):
    import numpy as np

    from page_cite_rag.dense import build_dense_index
    from page_cite_rag.retrieval import build_bm25_index, index_root

    class Server:
        requests = []

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def token_ids(self, text):
            return list(range(len(text)))

        def embed(self, texts):
            self.requests.extend(texts)
            return np.array([[1.0, 0.0]] * len(texts))

        def diagnostics(self):
            return {"guard_error": None}

    class Encoder:
        dimension = 2

        def __init__(self, transport, *args, **kwargs):
            self.transport = transport
            self.identity = {
                "query_prompt": "Q:",
                "max_tokens": 512,
                "query_max_tokens": 512,
                "model": "test",
            }

        def split_document(self, text):
            return [(0, len(text))]

        def encode_documents(self, texts):
            return self.transport.embed(texts)

        def encode_queries(self, texts):
            return self.transport.embed(["Q:" + text for text in texts])

    monkeypatch.setattr(run_reading_eval, "NativeEmbeddingServer", Server)
    monkeypatch.setattr(run_reading_eval, "NativeEmbeddingEncoder", Encoder)
    source_root = tmp_path / "previous"
    source_prepared = source_root / "prepared"
    source_prepared.mkdir(parents=True)
    source_store = SQLiteDocumentStore(source_root / "prepared/reading.sqlite3")
    cases = [
        {
            "case_id": "one",
            "dataset": "longbench-v2-single-document-qa",
            "language": "en",
            "question": "Question one",
            "source": {"format": "plain_text", "text": "Already indexed text."},
        }
    ]
    doc_id, _ = run_reading_eval._case_document(source_store, cases[0])
    build_bm25_index(source_store, doc_id)
    dense = build_dense_index(source_store, doc_id, Encoder(Server()), versioned=True)
    Server.requests.clear()
    static = {
        "source_code_sha256": {"scripts/run_reading_eval.py": "new"},
        "dataset_manifest_sha256": "frozen-manifest-with-review-metadata",
        "auxiliary_metrics": ["longbench_zh_t2s_f1_nonofficial_v1"],
        "models": {"generator": "unchanged"},
        "retrieval": {"token_chunking": False, "reranker": False},
        "embedding_input": {
            "document_max_tokens": 512,
            "query_max_tokens": 512,
            "request_timeout_seconds": 15,
        },
        "preparation_budget": {
            "document_embedding_input_tokens_max": 1000,
            "query_embedding_input_tokens_max": 1000,
        },
        "run_selection": {
            "case_ids": ["one"],
            "arms": ["fixed-rag-reranked", "single-agent"],
        },
        "answer": {"default_contract": "new-shared-writer"},
        "agent": {"architecture": "navigation-then-writer", "call_limit": 6},
        "answer_call_budget": {"total_worst_case": 7},
    }
    previous = {
        **static,
        "source_code_sha256": {"scripts/run_reading_eval.py": "old"},
        "dataset_manifest_sha256": "previous-manifest",
        "run_selection": {
            "case_ids": ["one"],
            "arms": ["fixed-rag-reranked", "fixed-rag-followup-once", "single-agent"],
        },
        "answer": {"default_contract": "old-writer"},
        "agent": {"architecture": "old-navigation", "call_limit": 8},
        "answer_call_budget": {"total_worst_case": 9},
    }
    previous.pop("auxiliary_metrics")
    if checkpoint_state == "valid_reranker_disabled":
        # Reranking consumes the frozen base candidate pool; the reusable
        # checkpoint does not contain reranker-specific state.
        previous["retrieval"] = {
            **previous["retrieval"],
            "reranker": True,
            "reranker_request": {"top_n": 10},
        }
        previous["models"] = {
            **previous["models"],
            "reranker": {"name": "test-reranker"},
        }
    run_reading_eval.atomic_json_once(source_root / "configuration.json", {"static": previous})
    source_database = source_root / "prepared/reading.sqlite3"
    original_db_hash = run_reading_eval.sha256(source_database)
    retrieval_path = source_prepared / "retrieval.jsonl"
    retrieval_row = {"case_id": "one", "result": {"hits": []}}
    if previous["retrieval"].get("reranker"):
        retrieval_row["candidate_result"] = {"hits": []}
    run_reading_eval.atomic_jsonl_once(retrieval_path, [retrieval_row])
    preflight_path = source_prepared / "embedding-preflight.json"
    run_reading_eval.atomic_json_once(
        preflight_path,
        {
            "static_configuration_sha256": run_reading_eval.canonical_hash(previous),
            "new_document_embedding_input_tokens_expected": 21,
            "query_embedding_input_tokens": 14,
        },
    )
    source_ready = {
        "retrieval_sha256": run_reading_eval.sha256(retrieval_path),
        "database_files_sha256": run_reading_eval._file_hashes(source_database),
        "index_files_sha256": run_reading_eval._file_hashes(index_root(source_store)),
        "embedding_preflight_sha256": run_reading_eval.sha256(preflight_path),
        "static_configuration_sha256": run_reading_eval.canonical_hash(previous),
        "embedding_inputs": {"reused_document_count": 1},
    }
    run_reading_eval.atomic_json_once(source_prepared / "prepared.json", source_ready)
    if checkpoint_state == "wrong_model":
        static = {**static, "models": {"generator": "different"}}
    elif checkpoint_state == "valid_lower_embedding_budget":
        static = {
            **static,
            "preparation_budget": {
                "document_embedding_input_tokens_max": 25,
                "query_embedding_input_tokens_max": 20,
            },
        }
    elif checkpoint_state == "over_embedding_budget":
        static = {
            **static,
            "preparation_budget": {
                "document_embedding_input_tokens_max": 20,
                "query_embedding_input_tokens_max": 20,
            },
        }
    elif checkpoint_state == "corrupt_vectors":
        (index_root(source_store) / dense.index_id / "vectors.npy").write_bytes(b"corrupt")

    run_root = tmp_path / "continuation"
    if checkpoint_state in {
        "valid",
        "valid_lower_embedding_budget",
        "valid_reranker_disabled",
    }:
        ready = run_reading_eval._prepare(
            tmp_path, run_root, {}, cases, static, reuse_from=source_root
        )
        assert ready["checkpoint_reuse"]["base_retrieval_reused"] is True
        assert Server.requests == []
        assert SQLiteDocumentStore(run_root / "prepared/reading.sqlite3").count_documents() == 1
        assert (
            run_reading_eval._file_hashes(
                index_root(SQLiteDocumentStore(run_root / "prepared/reading.sqlite3"))
            )
            == source_ready["index_files_sha256"]
        )
    else:
        expected = (
            "embedding usage exceeds the active preparation budget"
            if checkpoint_state == "over_embedding_budget"
            else "configuration changed|checksum"
        )
        with pytest.raises(ValueError, match=expected):
            run_reading_eval._prepare(tmp_path, run_root, {}, cases, static, reuse_from=source_root)
        assert Server.requests == []
    assert source_store.count_documents() == 1
    assert run_reading_eval.sha256(source_database) == original_db_hash


@pytest.mark.parametrize("failure_kind", ["terminal", "schema"])
def test_reading_planner_records_failures_and_never_retries_failed_cases(
    tmp_path, monkeypatch, failure_kind
):
    from page_cite_rag import application
    from page_cite_rag.generation import ModelError, OutputFailureAccounting, OutputValidationError
    from page_cite_rag.reading_plan import ReadingNeed, ReadingPlan

    cases = [
        {"case_id": "first", "dataset": "qasper", "language": "en", "question": "First?"},
        {"case_id": "second", "dataset": "qasper", "language": "en", "question": "Second?"},
    ]
    search = SearchResult(
        document_id="a" * 64,
        index_id="b" * 32,
        method="hybrid",
        query="question",
        hits=(),
        elapsed_ms=0.0,
    )
    retrieval = {
        case["case_id"]: {"document_id": search.document_id, "result": search.model_dump()}
        for case in cases
    }

    class Store:
        def get_document(self, _document_id):
            return SimpleNamespace(language="en")

    class Workflow:
        def source_outline(self):
            return None

    monkeypatch.setattr(application, "document_workflow", lambda *args, **kwargs: Workflow())
    monkeypatch.setattr(run_reading_eval, "_load_prepared", lambda *args: (Store(), retrieval))

    models = []

    class Model:
        identity = {"model": "test-model"}
        config = object()

        def __init__(self, fail_first):
            self.fail_first = fail_first
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def diagnostics(self):
            return {"guard_error": None}

        def generate(self, _messages, _response_model):
            self.calls += 1
            if self.fail_first and self.calls == 1:
                if failure_kind == "terminal":
                    raise ModelError("model transport failed")
                raise OutputValidationError(
                    "schema_invalid",
                    OutputFailureAccounting(
                        prompt_eval_count=4,
                        eval_count=3,
                        elapsed_ms=2.0,
                        stop_type="eos",
                    ),
                    raw_model_output="{}",
                )
            plan = ReadingPlan(
                requirements=(ReadingNeed(need="Find the requested fact", queries=("fact",)),)
            )
            stats = SimpleNamespace(
                model_dump=lambda **_kwargs: {"prompt_eval_count": 5, "eval_count": 2}
            )
            return SimpleNamespace(value=plan, stats=stats, raw_model_output=plan.model_dump_json())

    def model_client(*_args, **_kwargs):
        assert _kwargs["recover_output_errors"] is True
        model = Model(fail_first=not models)
        models.append(model)
        return model

    monkeypatch.setattr(run_reading_eval, "LlamaModelClient", model_client)
    monkeypatch.setattr(run_reading_eval, "_llama_config", lambda *_args, **_kwargs: None)
    static = {
        "reading_plan": {"num_predict": 64, "planner_budget": {"total_calls_limit": 4}},
        "answer": {"sampling_profile": "greedy"},
        "run_selection": {"case_ids": [case["case_id"] for case in cases]},
    }

    results_path = tmp_path / "run/reading-plans/results.jsonl"
    if failure_kind == "terminal":
        with pytest.raises(ModelError, match="unstarted case"):
            run_reading_eval._run_reading_plans(tmp_path / "run", static, cases)
        failed = json.loads(results_path.read_text().splitlines()[0])
        assert failed["case_id"] == "first"
        assert failed["failure_type"] == "ModelError"
        assert not (tmp_path / "run/reading-plans/ready.json").exists()
        assert not (tmp_path / "run/reading-plans/.attempts/second.json").exists()
        ready = run_reading_eval._run_reading_plans(tmp_path / "run", static, cases)
    else:
        ready = run_reading_eval._run_reading_plans(tmp_path / "run", static, cases)

    completed = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert [row["case_id"] for row in completed] == ["first", "second"]
    assert [model.calls for model in models] == ([1, 1] if failure_kind == "terminal" else [2])
    assert ready["successful_plans"] == 1
    assert ready["failed_plans"] == 1
    assert ready["planner_calls_attempted"] == 2
    assert ready["input_tokens"] == (9 if failure_kind == "schema" else 5)
    assert ready["output_tokens"] == (5 if failure_kind == "schema" else 2)
    assert ready["usage_unknown"] is (failure_kind == "terminal")


def test_same_pool_reranker_freezes_candidates_and_admits_exact_contexts(tmp_path, monkeypatch):
    from page_cite_rag import hybrid, llama, reranker_server

    document_id = "a" * 64
    block = Block(
        block_id="b" * 64,
        document_id=document_id,
        unit_index=1,
        ordinal=1,
        text="The measured value is 12 units.",
        char_start=0,
        char_end=31,
        line_start=1,
        line_end=1,
    )
    search = SearchResult(
        document_id=document_id,
        index_id="c" * 64,
        method="hybrid",
        query="What is the measured value?",
        hits=(SearchHit(rank=1, score=1.0, block=block),),
        elapsed_ms=1.0,
    )
    serialized = search.model_dump(mode="json")
    variants = ("original_hybrid", "rewrite_dense_only")
    case_id = "case-1"
    static = {
        "models": {"generator": {"size": "9b"}},
        "answer": {"sampling_profile": "greedy"},
        "retrieval": {
            "reranker": True,
            "retrieval_grid": {"protocol": "full-sentence-rewrite-routing/v1"},
        },
    }
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "configuration.json").write_text(json.dumps({"static": static}))
    grid_dir = run_root / "retrieval-grid"
    grid_dir.mkdir()
    grid_path = grid_dir / "retrieval-grid.jsonl"
    run_reading_eval.atomic_jsonl_once(
        grid_path,
        [
            {
                "case_id": case_id,
                "candidate_results": {name: serialized for name in variants},
                "context_results": {name: serialized for name in variants},
            }
        ],
    )
    route_score_path = grid_dir / "retrieval-score.json"
    run_reading_eval.atomic_json_once(
        route_score_path, {"selection": {"selected_variant": "rewrite_dense_only"}}
    )

    class FakeRanker:
        def __init__(self, *args, **kwargs):
            self.last_pair_tokens = ()
            self.last_batch_sizes = ()
            self.last_tokenization_ms = 0.1
            self.last_forward_ms = 0.2
            self.last_total_ms = 0.3

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def diagnostics(self):
            return {"guard_error": None}

    class FakeModel:
        config = object()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def diagnostics(self):
            return {"guard_error": None}

    monkeypatch.setattr(reranker_server, "NativeRerankerServer", FakeRanker)
    monkeypatch.setattr(llama, "LlamaModelClient", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(run_reading_eval, "_llama_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        run_reading_eval,
        "_load_prepared",
        lambda *args, **kwargs: (object(), {case_id: {}}),
    )
    monkeypatch.setattr(
        run_reading_eval,
        "_exact_context_admission",
        lambda store, result, case, model: {
            "admitted_block_ids": [block.block_id],
            "omitted_block_ids": [],
            "exact_input_tokens": 500,
            "context_group_count": 0,
        },
    )

    def rerank(candidate, ranker, *, top_k, store=None):
        ranker.last_pair_tokens = (17,)
        ranker.last_batch_sizes = (1,)
        return candidate

    monkeypatch.setattr(hybrid, "rerank_search_result", rerank)
    case = {
        "case_id": case_id,
        "question": search.query,
        "language": "en",
        "dataset": "qasper",
        "source": {"paragraphs": []},
    }
    result = run_reading_eval._rank_full_sentence_retrieval_grid(run_root, static, [case])

    assert result["state"] == "complete"
    assert result["candidate_pool_count"] == 2
    assert result["confirmed_pair_tokens"] == 34
    assert result["generation_calls"] == 0
    rows = run_reading_eval.load_unique_jsonl(
        grid_dir / "same-pool-reranker/results.jsonl", key="pool_id"
    )
    admissions = run_reading_eval.load_unique_jsonl(
        grid_dir / "same-pool-reranker/admissions.jsonl", key="pool_id"
    )
    assert set(rows) == {f"{case_id}::{name}" for name in variants}
    assert len({row["candidate_sha256"] for row in rows.values()}) == 1
    assert len(admissions) == 2
    assert all(row["before"]["exact_input_tokens"] == 500 for row in admissions.values())


def test_downstream_continuation_allows_only_runner_hash_change(tmp_path):
    frozen = {
        "source_code_sha256": {
            "scripts/run_reading_eval.py": "a" * 64,
            "src/page_cite_rag/retrieval.py": "b" * 64,
        },
        "retrieval": {"top_k": 10},
    }
    run_root = tmp_path / "frozen-run"
    run_root.mkdir()
    (run_root / "configuration.json").write_text(json.dumps({"static": frozen}), encoding="utf-8")
    current = json.loads(json.dumps(frozen))
    current["source_code_sha256"]["scripts/run_reading_eval.py"] = "c" * 64

    assert (
        run_reading_eval._validate_downstream_stage_configuration(
            run_root, current, stage="same-pool-reranker"
        )
        == frozen
    )
    audit = json.loads(
        (run_root / "stage-continuation-audits/same-pool-reranker.json").read_text(encoding="utf-8")
    )
    assert audit["changed_source_paths"] == ["scripts/run_reading_eval.py"]
    assert audit["frozen_static_configuration_sha256"] == run_reading_eval.canonical_hash(frozen)

    changed_retrieval = json.loads(json.dumps(current))
    changed_retrieval["retrieval"]["top_k"] = 20
    with pytest.raises(ValueError, match="differs beyond the runner source hash"):
        run_reading_eval._validate_downstream_stage_configuration(
            run_root, changed_retrieval, stage="same-pool-reranker"
        )

    changed_index_code = json.loads(json.dumps(current))
    changed_index_code["source_code_sha256"]["src/page_cite_rag/retrieval.py"] = "d" * 64
    with pytest.raises(ValueError, match="source changed beyond the evaluation runner"):
        run_reading_eval._validate_downstream_stage_configuration(
            run_root, changed_index_code, stage="same-pool-reranker"
        )


def test_alignment_issue_counts_are_per_case_not_per_retrieval_variant():
    counts = run_reading_eval._alignment_issue_counts(
        {
            "case-1": {"evidence_alignment_issues": ["unmatched", "unmatched", "non_text"]},
            "case-2": {"evidence_alignment_issues": ["unmatched"]},
        },
        ["case-1", "case-1", "case-2"],
    )

    assert counts == {"unmatched": 2, "non_text": 1}


def test_embedding_prepare_failure_records_attempt_and_gpu_trace(tmp_path):
    class FakeTransport:
        def token_ids(self, text):
            return list(range(len(text)))

        def embed(self, texts):
            raise TimeoutError("request timed out")

    counter = run_reading_eval._CountingTransport(FakeTransport())
    with pytest.raises(TimeoutError):
        counter.embed(["private source text"])

    class FakeServer:
        def diagnostics(self):
            return {"guard_error": "gpu_thermal_slowdown", "last_gpu_sample": {"temperature_c": 83}}

    run_root = tmp_path / "run"
    prepared = run_root / "prepared"
    prepared.mkdir(parents=True)
    (prepared / "embedding-preflight.json").write_text("{}", encoding="utf-8")
    run_reading_eval._record_preparation_failure(
        run_root,
        {"runner": "test"},
        {"stage": "building_document_indexes", "document_ordinal": 2, "document_count": 3},
        TimeoutError("request timed out"),
        FakeServer(),
        counter,
    )

    record = json.loads((run_root / "PREPARATION-FAILURE.json").read_text(encoding="utf-8"))
    assert record["failure_stage"] == "building_document_indexes"
    assert record["document_ordinal"] == 2
    assert record["embedding_request_accounting"]["requests_attempted"] == 1
    assert record["embedding_request_accounting"]["requests_succeeded"] == 0
    assert record["embedding_request_accounting"]["document_input_tokens_attempted"] == len(
        "private source text"
    )
    assert record["gpu_trace"]["guard_error"] == "gpu_thermal_slowdown"
    assert "private source text" not in json.dumps(record)


def test_embedding_startup_guard_failure_records_exception_sample(tmp_path):
    class StartupGuardError(RuntimeError):
        category = "gpu_thermal_slowdown"
        sample = {"temperature_c": 83, "max_operating_c": 93, "sw_thermal_slowdown": True}

    run_root = tmp_path / "run"
    run_root.mkdir()
    _write_frozen_fixture(run_root, ["case-1"])
    run_reading_eval._record_preparation_failure(
        run_root,
        {"runner": "test"},
        {"stage": "exact_embedding_preflight", "document_ordinal": None, "document_count": 1},
        StartupGuardError("GPU thermal slowdown"),
        None,
        None,
    )

    record = json.loads((run_root / "PREPARATION-FAILURE.json").read_text(encoding="utf-8"))
    assert record["gpu_trace"]["guard_error"] == "gpu_thermal_slowdown"
    assert record["gpu_trace"]["last_gpu_sample"] == StartupGuardError.sample


@pytest.mark.parametrize("arms", [("fixed-rag",), ("fixed-rag", "single-agent")])
def test_score_run_compares_only_selected_arms_and_labels_case_scope(tmp_path, arms):
    case_id = "longbench-v2-case"
    manifest = {"cases": [{"case_id": case_id, "dataset": "longbench-v2-single-document-qa"}]}
    manifest["files"] = {"inputs.jsonl": {"rows": 24}}
    inputs = [{"case_id": case_id}]
    gold = [{"case_id": case_id, "answer": "B"}]
    for arm, answer in (("fixed-rag", "B"), ("single-agent", "A")):
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        (arm_dir / f"{case_id}.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "case_id": case_id,
                    "execution": "ok",
                    "result": {"answer": f"The correct answer is ({answer})."},
                    "gpu_diagnostics": {"observed_gpu_peak": {}},
                }
            ),
            encoding="utf-8",
        )

    result = run_reading_eval.score_run(
        tmp_path,
        manifest,
        inputs,
        gold,
        arms=arms,
    )

    assert result["state"] == "scored_complete"
    assert result["evaluation_scope"] == {
        "kind": "selected_case_and_arm_slice",
        "case_ids": [case_id],
        "arms": list(arms),
        "case_count": 1,
        "frozen_dataset_case_count": 24,
    }
    assert set(result["arms"]) == set(arms)
    assert json.loads((tmp_path / "scores.json").read_text(encoding="utf-8")) == result
    assert (
        result["arms"]["fixed-rag"]["datasets"]["longbench-v2-single-document-qa"]["per_case"][0][
            "case_id"
        ]
        == case_id
    )
    assert (
        result["arms"]["fixed-rag"]["datasets"]["longbench-v2-single-document-qa"]["metrics"][
            "accuracy"
        ]
        == 1.0
    )
    if "single-agent" in arms:
        assert (
            result["arms"]["single-agent"]["datasets"]["longbench-v2-single-document-qa"][
                "metrics"
            ]["accuracy"]
            == 0.0
        )


def test_product_score_reports_availability_without_fabricating_correctness(tmp_path):
    case_id = "product-answerable"
    manifest = {
        "cases": [{"case_id": case_id, "dataset": "product-functional"}],
        "files": {"inputs.jsonl": {"rows": 1}},
    }
    inputs = [
        {
            "case_id": case_id,
            "dataset": "product-functional",
            "source_language": "en",
            "question_language": "en",
            "language": "zh-Hant",
        }
    ]
    gold = [{"case_id": case_id, "expected_facts": ["a numeric comparison"]}]
    arm_dir = tmp_path / "fixed-rag"
    arm_dir.mkdir()
    (arm_dir / f"{case_id}.json").write_text(
        json.dumps(
            {
                "arm": "fixed-rag",
                "case_id": case_id,
                "execution": "ok",
                "result": {
                    "status": "answered",
                    "answer": "A numeric comparison.",
                    "answer_claims": [{"citations": [{"block_id": "source-1"}]}],
                    "detail_claims": [],
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_reading_eval.score_run(tmp_path, manifest, inputs, gold, arms=("fixed-rag",))

    row = result["arms"]["fixed-rag"]["datasets"]["product-functional"]["per_case"][0]
    assert row["answer_available"] is True
    assert row["refused"] is False
    assert row["claim_count"] == 1
    assert row["citation_count"] == 1
    assert row["official_correctness_metric"] is None
    assert row["review_required"] is True


def test_score_run_uses_qasper_unanswerable_gold_for_refusal_metrics(tmp_path):
    case_id = "qasper-no-answer"
    arm = "fixed-rag"
    manifest = {
        "cases": [{"case_id": case_id, "dataset": "qasper"}],
        "files": {"inputs.jsonl": {"rows": 1}},
    }
    inputs = [
        {
            "case_id": case_id,
            "dataset": "qasper",
            "source": {"paragraphs": [{"passage_id": "p1", "text": "A source paragraph."}]},
        }
    ]
    gold = [
        {
            "case_id": case_id,
            "references": [{"answer": "Unanswerable", "answer_type": "none", "evidence": []}],
            "aligned_evidence_passage_ids": [],
            "evidence_alignment_issues": ["unanswerable"],
            "unanswerable": True,
        }
    ]
    arm_dir = tmp_path / arm
    arm_dir.mkdir()
    (arm_dir / f"{case_id}.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "case_id": case_id,
                "execution": "ok",
                "result": {
                    "answer": "",
                    "status": "insufficient_evidence",
                    "included_block_ids": [],
                    "answer_claims": [],
                },
            }
        ),
        encoding="utf-8",
    )

    report = run_reading_eval.score_run(tmp_path, manifest, inputs, gold, arms=(arm,))

    metrics = report["arms"][arm]["datasets"]["qasper"]["per_case"][0]
    assert metrics["correct_refusal"] is True
    assert metrics["false_refusal"] is False


def test_official_longbench_score_uses_original_text_extractor_even_at_output_limit(tmp_path):
    case_id = "longbench-v2-full-case"
    arm = run_reading_eval.LONG_BENCH_OFFICIAL_ARM
    manifest = {
        "cases": [{"case_id": case_id}],
        "files": {"inputs.jsonl": {"rows": 175}},
    }
    inputs = [{"case_id": case_id, "dataset": "longbench-v2-single-document-qa"}]
    gold = [{"case_id": case_id, "answer": "B"}]
    arm_dir = tmp_path / arm
    arm_dir.mkdir()
    (arm_dir / f"{case_id}.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "case_id": case_id,
                "execution": "failed",
                "failure": {"reason": "output_limit"},
                "result": None,
                "raw_model_output": "The correct answer is (B).",
                "completion_calls_attempted": 1,
                "input_tokens_lower_bound": 50,
                "output_tokens_lower_bound": 128,
                "context_trace": {
                    "retrieved_block_ids": ["included", "omitted"],
                    "admitted_block_ids": ["included"],
                    "omitted_block_ids": ["omitted"],
                    "exact_input_tokens": 50,
                },
                "gpu_diagnostics": {"observed_gpu_peak": {}},
            }
        ),
        encoding="utf-8",
    )

    result = run_reading_eval.score_run(
        tmp_path,
        manifest,
        inputs,
        gold,
        arms=(arm,),
    )

    metric = result["arms"][arm]["datasets"]["longbench-v2-single-document-qa"]
    assert metric["n"] == 1
    assert metric["successful"] == 0
    assert metric["metrics"]["accuracy"] == 1.0
    assert metric["per_case"][0]["official_protocol_raw_output_scored"] is True
    admission = result["arms"][arm]["context_admission"]
    assert admission["case_count"] == 1
    assert admission["initial_omitted_blocks"] == admission["final_omitted_blocks"] == 1


@pytest.mark.parametrize("failure_kind", ["output_limit", "transport"])
def test_official_longbench_keeps_healthy_client_but_stops_on_fatal_failure(
    tmp_path, monkeypatch, failure_kind
):
    from page_cite_rag.generation import ModelError, OutputFailureAccounting, OutputValidationError
    from page_cite_rag.retrieval import SearchHit, SearchResult

    arm = run_reading_eval.LONG_BENCH_OFFICIAL_ARM
    store = SQLiteDocumentStore(tmp_path / "source.sqlite3")
    cases = [
        {
            "case_id": name,
            "dataset": "longbench-v2-single-document-qa",
            "language": "en",
            "benchmark_question": "Question " + name,
            "question": "Question " + name,
            "choices": {k: k for k in "ABCD"},
            "source": {"format": "plain_text", "text": "Evidence " + name},
        }
        for name in ("first", "second")
    ]
    retrieval = {}
    for case in cases:
        document_id, _ = run_reading_eval._case_document(store, case)
        block = store.get_blocks(document_id)[0]
        search = SearchResult(
            document_id=document_id,
            index_id="test",
            query=case["question"],
            hits=(SearchHit(rank=1, score=1.0, block=block),),
            elapsed_ms=0,
        )
        retrieval[case["case_id"]] = {"result": search.model_dump(mode="json")}

    class Model:
        identity = {"name": "test"}
        calls = 0

        def __init__(self, *args, **kwargs):
            assert kwargs.get("recover_output_errors", True) is True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def prepare_budget(self, *args):
            return SimpleNamespace(exact_input_tokens=100)

        def generate(self, *args):
            type(self).calls += 1
            if self.calls == 1:
                if failure_kind == "transport":
                    raise ModelError("transport failed")
                raise OutputValidationError(
                    "output_limit",
                    OutputFailureAccounting(
                        prompt_eval_count=100, eval_count=128, elapsed_ms=1, stop_type="limit"
                    ),
                    raw_model_output="Unfinished response",
                )
            stats = SimpleNamespace(
                model_dump=lambda **kw: {"prompt_eval_count": 100, "eval_count": 8}
            )
            return SimpleNamespace(raw_model_output="The correct answer is (B).", stats=stats)

        def diagnostics(self):
            return {"guard_error": None}

    static = {"run_selection": {"arms": [arm]}}
    preflight = {}
    for case in cases:
        search = run_reading_eval._search_result_from_json(retrieval[case["case_id"]]["result"])
        packet = run_reading_eval._longbench_v2_admission(Model(), search, case)
        preflight[case["case_id"]] = {
            k: packet[k]
            for k in (
                "admitted_block_ids",
                "omitted_block_ids",
                "retrieved_block_ids",
                "exact_input_tokens",
                "prompt_sha256",
            )
        }
        preflight[case["case_id"]]["retrieval_sha256"] = run_reading_eval.canonical_hash(
            retrieval[case["case_id"]]["result"]
        )
    (tmp_path / "preflight.json").write_text(
        json.dumps(
            {
                "static_configuration_sha256": run_reading_eval.canonical_hash(static),
                "generator_identity": Model.identity,
                "cases": preflight,
            }
        )
    )
    monkeypatch.setattr(run_reading_eval, "LlamaModelClient", Model)
    monkeypatch.setattr(run_reading_eval, "_llama_config", lambda *a, **kw: None)
    monkeypatch.setattr(run_reading_eval, "_load_prepared", lambda *a: (store, retrieval))
    result = run_reading_eval._run_longbench_official_arm(tmp_path, static, cases, None)
    assert Model.calls == (2 if failure_kind == "output_limit" else 1)
    assert result["stopped"] == (None if failure_kind == "output_limit" else "model_client_failed")
    assert result["remaining_slots"] == (0 if failure_kind == "output_limit" else 1)


@pytest.mark.parametrize("arms", [("fixed-rag",), ("fixed-rag", "single-agent")])
def test_score_run_reports_missing_and_interrupted_slots_only_for_selected_arms(tmp_path, arms):
    for arm in arms:
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        (arm_dir / "done.json").write_text(
            json.dumps({"arm": arm, "case_id": "done"}), encoding="utf-8"
        )
    attempts = tmp_path / arms[0] / ".attempts"
    attempts.mkdir()
    (attempts / "missing.json").write_text("{}", encoding="utf-8")
    excluded_arm = next(arm for arm in run_reading_eval.ARMS if arm not in arms)
    (tmp_path / excluded_arm).mkdir()
    (tmp_path / excluded_arm / "unrelated.json").write_text("invalid JSON", encoding="utf-8")

    result = run_reading_eval.score_run(
        tmp_path,
        {"cases": [{"case_id": "missing"}, {"case_id": "done"}]},
        [],
        [],
        arms=arms,
    )

    assert result == {
        "state": "incomplete",
        "missing_slots_by_arm": {arm: ["missing"] for arm in arms},
        "unresolved_attempt_slots_by_arm": {
            arm: ["missing"] if arm == arms[0] else [] for arm in arms
        },
        "score_artifact_written": False,
    }
    assert not (tmp_path / "scores.json").exists()


@pytest.mark.parametrize("arms", [("fixed-rag",), ("fixed-rag", "single-agent")])
@pytest.mark.parametrize("wrong_field", ["arm", "case_id"])
def test_score_run_selected_arms_retain_raw_identity_checks(tmp_path, arms, wrong_field):
    arm_dir = tmp_path / arms[-1]
    arm_dir.mkdir()
    raw = {"arm": arms[-1], "case_id": "selected", wrong_field: "wrong"}
    (arm_dir / "selected.json").write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its frozen arm/case identity"):
        run_reading_eval.score_run(
            tmp_path,
            {"cases": [{"case_id": "selected"}]},
            [],
            [],
            arms=arms,
        )


@pytest.mark.parametrize("arms", [(), ("fixed-rag", "fixed-rag"), ("unknown",)])
def test_score_run_rejects_invalid_arm_selection(tmp_path, arms):
    with pytest.raises(ValueError, match="nonempty unique subset"):
        run_reading_eval.score_run(tmp_path, {"cases": []}, [], [], arms=arms)


def test_score_run_rejects_raw_results_outside_selected_cases(tmp_path):
    arm_dir = tmp_path / "fixed-rag"
    arm_dir.mkdir()
    (arm_dir / "unexpected.json").write_text(
        json.dumps({"arm": "fixed-rag", "case_id": "unexpected"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="outside the selected evaluation scope"):
        run_reading_eval.score_run(
            tmp_path,
            {"cases": [{"case_id": "selected", "dataset": "longbench-v2-single-document-qa"}]},
            [{"case_id": "selected"}],
            [{"case_id": "selected", "answer": "A"}],
            arms=("fixed-rag",),
        )


def test_partition_loader_checks_master_manifest_without_reading_gold(tmp_path):
    partition = tmp_path / "holdout"
    partition.mkdir()
    part_manifest, rows = _write_frozen_fixture(partition, ["one", "two"])
    (partition / "manifest.json").unlink()
    master = {"partitions": {"holdout": part_manifest}}
    (tmp_path / "manifest.json").write_text(json.dumps(master), encoding="utf-8")

    loaded, inputs = run_reading_eval.load_frozen(partition)

    assert loaded["partition"] == "holdout"
    assert loaded["cases"] == part_manifest["cases"]
    assert inputs == rows
    assert run_reading_eval.load_gold(partition, loaded) == rows


def test_task_format_override_is_explicit_and_validated():
    assert (
        run_reading_eval._answer_format_for(
            {"dataset": "derived-crosslingual", "task_format": "multiple_choice"}
        )
        == "multiple_choice"
    )
    assert run_reading_eval._answer_format_for({"dataset": "longbench-v1-qmsum"}) == "general"
    with pytest.raises(ValueError, match="frozen task_format"):
        run_reading_eval._answer_format_for({"dataset": "qasper", "task_format": "summary"})


@pytest.mark.parametrize(
    ("arm", "guard_failure"),
    [
        ("repair-once", False),
        ("reading-agent", False),
        ("delegated-reading-agent", False),
        ("reading-agent", True),
    ],
)
def test_review_arms_share_product_flow_and_account_every_stage(
    tmp_path, monkeypatch, arm, guard_failure
):
    from pydantic import BaseModel

    import page_cite_rag.agent as agent_module
    import page_cite_rag.review as review_module
    from page_cite_rag.generation import (
        ChatMessage,
        GenerationConfig,
        GenerationResult,
        GenerationStats,
        PromptBudget,
    )
    from page_cite_rag.ingest import prepare_source
    from page_cite_rag.retrieval import SearchHit, SearchResult
    from page_cite_rag.workflows import AnswerResult

    source = tmp_path / "source.md"
    source.write_text("# Method\n\nUse only under condition A.\n", encoding="utf-8")
    prepared = prepare_source(source, usage_scope="self-authored-test")
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    block = store.get_blocks(prepared.document.document_id)[1]
    question = "Which conditions apply?"
    search = SearchResult(
        document_id=block.document_id,
        index_id="test-index",
        method="hybrid",
        query=question,
        hits=(SearchHit(rank=1, score=1.0, block=block),),
        elapsed_ms=1.0,
    )
    config = GenerationConfig(num_predict=1536)
    budget = PromptBudget(1124, 1536, 16384, "exact-test", 100)
    stats = GenerationStats(
        prompt_eval_count=100,
        eval_count=10,
        total_duration=0,
        load_duration=None,
        prompt_eval_duration=0,
        eval_duration=0,
        done_reason="stop",
        elapsed_ms=1.0,
    )
    staged_instances = []

    class FakeStagedSearch:
        def __init__(self, *_args, **_kwargs):
            self.active = False
            self.session_calls = 0
            self.last_trace = {}
            staged_instances.append(self)

        @contextmanager
        def session(self):
            self.session_calls += 1
            self.active = True
            try:
                yield self
            finally:
                self.active = False

    class WriterValue(BaseModel):
        text: str

    class FakeModel:
        identity = {"model": "scripted-test"}

        def __init__(self, *_args, **_kwargs):
            self.config = config
            self.call_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def prepare_budget(self, *_args):
            return budget

        def generate(self, _messages, response_model):
            assert staged_instances[-1].active
            self.call_count += 1
            is_checker = response_model is review_module.SourceReview
            value = (
                review_module.SourceReview(verdict="supported", needs_reading=False, issues=())
                if is_checker
                else WriterValue(text="source-backed answer")
            )
            raw = f"{'checker' if is_checker else 'answer'}-{self.call_count}"
            return GenerationResult(value, stats, budget, raw)

        def diagnostics(self):
            return {"guard_error": "gpu_thermal_slowdown"} if guard_failure else {}

    @contextmanager
    def fake_local_chat(model):
        assert isinstance(model, run_reading_eval._CountingModelClient)
        yield object()

    observed = []

    def shared_review(workflow, original_question, **kwargs):
        observed.append(kwargs)
        assert original_question == question
        assert kwargs["initial_blocks"] == (block,)
        if guard_failure:
            workflow._model.generate(
                (ChatMessage(role="user", content="actual stage input"),), WriterValue
            )
            failed_navigation = {
                "agent_input_tokens_exact": 40,
                "model_call_records": [
                    {
                        "status": "not_dispatched_cooldown_failed",
                        "input_tokens_exact": 40,
                    }
                ],
                "cooldown_error_type": "ModelError",
            }
            failure = agent_module.AgentRunFailure("ModelError", failed_navigation)
            raise review_module.ReviewRunFailure(
                failure, {"failed_stage_trace": failed_navigation, "stages": []}
            ) from failure
        for schema in (WriterValue, review_module.SourceReview) * 2:
            workflow._model.generate(
                (ChatMessage(role="user", content="actual stage input"),), schema
            )
        navigation = None
        if arm != "repair-once":
            navigation = {
                "agent_input_tokens_exact": 80,
                "agent_output_tokens_observed": 10,
                "agent_output_tokens_reserved": 10,
                "model_call_records": [
                    {
                        "status": "completed",
                        "input_tokens_exact": 40,
                        "output_tokens_observed": 5,
                        "elapsed_ms": 1.0,
                    }
                    for _ in range(2)
                ],
                "selected_block_ids": [block.block_id],
            }
        result = AnswerResult(
            document_id=block.document_id,
            mode="local",
            question=question,
            language="en",
            status="insufficient_evidence",
            answer="Source does not establish the complete answer.",
            answer_claims=(),
            detail_claims=(),
            limitations=(),
            included_block_ids=(block.block_id,),
            omitted_block_ids=(),
            stop_reason="model_insufficient_evidence",
            retrieval=None,
            generation_stats=stats,
            model_identity=workflow._model.identity,
            budget=budget,
            elapsed_ms=1.0,
        )
        return review_module.ReviewedAnswer(result, {"agent": navigation, "stages": []})

    case = {
        "case_id": "review-case",
        "dataset": "qasper",
        "question": question,
        "language": "en",
        "task_format": "general",
    }
    static = {
        "run_selection": {"arms": [arm]},
        "answer": {
            "prompt_revisions_by_arm": {arm: "contract-consistent"},
            "sampling_profile": "greedy",
        },
    }
    preflight_case = {
        "admitted_block_ids": [block.block_id],
        "omitted_block_ids": [],
        "retrieved_block_ids": [block.block_id],
        "prompt_budget_admitted_block_ids": [block.block_id],
        "exact_final_input_tokens": 100,
    }
    (tmp_path / "preflight.json").write_text(
        json.dumps(
            {
                "static_configuration_sha256": run_reading_eval.canonical_hash(static),
                "generator_identity": FakeModel.identity,
                "cases": {
                    "review-case": {"by_prompt_revision": {"contract-consistent": preflight_case}}
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_reading_eval, "LlamaModelClient", FakeModel)
    monkeypatch.setattr(run_reading_eval, "_llama_config", lambda _static: config)
    monkeypatch.setattr(run_reading_eval, "StagedGpuSearch", FakeStagedSearch)
    monkeypatch.setattr(
        run_reading_eval,
        "_load_prepared",
        lambda *_: (
            store,
            {
                "review-case": {
                    "document_id": block.document_id,
                    "result": search.model_dump(mode="json"),
                }
            },
        ),
    )
    monkeypatch.setattr(agent_module, "local_chat_model", fake_local_chat)
    monkeypatch.setattr(review_module, "answer_with_review", shared_review)

    inputs = [case, {**case, "case_id": "next-case"}] if guard_failure else [case]
    outcome = run_reading_eval._run_arm(tmp_path, static, inputs, arm, None)

    raw = json.loads((tmp_path / arm / "review-case.json").read_text(encoding="utf-8"))
    assert observed[0]["strategy"] == run_reading_eval.REVIEW_ARMS[arm]
    assert staged_instances[0].session_calls == 1
    assert not staged_instances[0].active
    if guard_failure:
        assert outcome["successful_slots_this_invocation"] == 0
        assert outcome["stopped"] == "gpu_guard_stopped"
        assert outcome["remaining_slots"] == 1
        assert not (tmp_path / arm / ".attempts" / "next-case.json").exists()
        assert raw["failure"]["reading_trace"]["failed_stage_trace"]
        assert raw["completion_calls_attempted"] == 1
        assert raw["input_tokens_lower_bound"] == 100
        assert raw["output_tokens_lower_bound"] == 10
        return
    assert outcome["successful_slots_this_invocation"] == 1
    assert len(raw["structured_generation_records"]) == 4
    assert raw["structured_generation_records"][0]["messages"][0]["content"] == "actual stage input"
    assert raw["structured_generation_records"][-1]["raw_model_output"] == "checker-4"
    assert raw["raw_model_output"] == "answer-3"
    assert raw["completion_calls_attempted"] == (4 if arm == "repair-once" else 6)
    assert raw["input_tokens_lower_bound"] == (400 if arm == "repair-once" else 480)
    assert raw["output_tokens_lower_bound"] == (40 if arm == "repair-once" else 50)
    assert raw["output_usage_unknown"] is False
    assert raw["reading_trace"] is not None


def test_official_summary_track_scores_visible_details_and_not_metadata(tmp_path):
    pytest.importorskip("rouge")
    result = {
        "answer": "A precise overview.",
        "detail_claims": [
            {"text": "A necessary qualification.", "citations": [{"quote": "secret"}]}
        ],
        "limitations": ["not part of the summary"],
    }
    text = run_reading_eval._summary_scoring_text(result)
    assert text == "A precise overview.\nA necessary qualification."
    metric = run_reading_eval._longbench_summary_score(text, [text], "en")
    assert metric["rouge_l"] == pytest.approx(1.0)
    zh = run_reading_eval._longbench_summary_score("完整中文摘要", ["完整中文摘要"], "zh")
    assert zh["rouge_l"] == pytest.approx(1.0)


@pytest.mark.parametrize("eligible,language", [(False, "en"), (True, "en"), (True, "zh-Hant")])
def test_cross_language_f1_uses_frozen_contract_even_if_answer_matches_gold(
    tmp_path, eligible, language
):
    case_id, arm = "chinese-source", "fixed-rag"
    manifest = {
        "cases": [
            {
                "case_id": case_id,
                "dataset": "longbench-v1-multifieldqa-zh",
                "native_metric_eligible": eligible,
            }
        ]
    }
    inputs = [{"case_id": case_id, "language": language, "source_language": "zh-Hant"}]
    gold = [{"case_id": case_id, "answers": ["由首相指派"]}]
    directory = tmp_path / arm
    directory.mkdir()
    (directory / f"{case_id}.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "case_id": case_id,
                "execution": "ok",
                "result": {"answer": "由首相指派"},
            }
        ),
        encoding="utf-8",
    )
    report = run_reading_eval.score_run(tmp_path, manifest, inputs, gold, arms=(arm,))
    dataset = next(iter(report["arms"][arm]["datasets"].values()))
    row = dataset["per_case"][0]
    expected = eligible and language == "zh-Hant"
    assert row["native_metric_eligible"] is expected
    assert row["answer_f1"] == (1.0 if expected else None)
    assert ("answer_f1" in dataset["metrics"]) is expected
