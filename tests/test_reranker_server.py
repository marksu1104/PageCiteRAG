from __future__ import annotations

import httpx
import pytest

from page_cite_rag.reranker_server import NativeRerankerServer
from page_cite_rag.retrieval import RetrievalError


def test_reranker_sorts_by_score_and_preserves_original_order_for_ties():
    server = NativeRerankerServer.__new__(NativeRerankerServer)

    def post(path, _payload):
        if path == "/tokenize":
            return {"tokens": [1, 2, 3]}
        return {
            "results": [
                {"index": 2, "relevance_score": 0.7},
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.7},
            ]
        }

    server._post = post

    assert server.rerank("question", ["a", "b", "c"]) == [
        (0, 0.9),
        (1, 0.7),
        (2, 0.7),
    ]
    assert server.last_pair_tokens == (3, 3, 3)
    assert server.last_batch_sizes == (3,)
    assert server.last_attempted_batch_count == server.last_completed_batch_count == 1


def test_reranker_scores_all_candidates_in_bounded_batches_and_restores_global_indexes():
    server = NativeRerankerServer.__new__(NativeRerankerServer)
    request_sizes = []

    def post(path, payload):
        if path == "/tokenize":
            return {"tokens": [1, 2, 3]}
        documents = payload["documents"]
        request_sizes.append(len(documents))
        offset = sum(request_sizes[:-1])
        return {
            "results": [
                {"index": index, "relevance_score": float(offset + index)}
                for index in reversed(range(len(documents)))
            ]
        }

    server._post = post
    documents = [f"document-{index}" for index in range(19)]

    ranked = server.rerank("question", documents)

    assert request_sizes == [8, 8, 3]
    assert [index for index, _ in ranked] == list(reversed(range(19)))
    assert server.last_pair_tokens == (3,) * 19
    assert server.last_batch_sizes == (8, 8, 3)
    assert server.last_attempted_batch_count == server.last_completed_batch_count == 3


@pytest.mark.parametrize(
    "results",
    [
        [],
        [{"index": 0, "relevance_score": 1.0}, {"index": 0, "relevance_score": 0.5}],
        [{"index": 0, "relevance_score": float("nan")}, {"index": 1, "relevance_score": 0.5}],
    ],
)
def test_reranker_rejects_incomplete_or_invalid_response(results):
    server = NativeRerankerServer.__new__(NativeRerankerServer)

    def post(path, _payload):
        return {"tokens": [1]} if path == "/tokenize" else {"results": results}

    server._post = post

    with pytest.raises(RetrievalError):
        server.rerank("question", ["a", "b"])


def test_reranker_rejects_oversized_pair_before_scoring():
    server = NativeRerankerServer.__new__(NativeRerankerServer)

    def post(path, _payload):
        if path == "/tokenize":
            return {"tokens": [1] * 4097}
        pytest.fail("oversized pair reached reranker")

    server._post = post

    with pytest.raises(RetrievalError, match="exceeds 4096 tokens"):
        server.rerank("question", ["a"])


def test_local_model_timeout_identifies_endpoint_without_exposing_payload():
    server = NativeRerankerServer.__new__(NativeRerankerServer)
    server._check = lambda: None

    class TimedOutClient:
        def post(self, *_args, **_kwargs):
            raise httpx.ReadTimeout("request timed out")

    server._http = TimedOutClient()

    with pytest.raises(RetrievalError, match="timed out at /tokenize"):
        server._post("/tokenize", {"content": "private document text"})


def test_reranker_diagnostics_keep_completed_batch_count_when_a_later_batch_fails():
    server = NativeRerankerServer.__new__(NativeRerankerServer)
    rerank_requests = 0

    def post(path, _payload):
        nonlocal rerank_requests
        if path == "/tokenize":
            return {"tokens": [1, 2]}
        rerank_requests += 1
        if rerank_requests == 2:
            raise RetrievalError("native local request timed out at /v1/rerank")
        return {
            "results": [{"index": index, "relevance_score": float(index)} for index in range(8)]
        }

    server._post = post
    with pytest.raises(RetrievalError, match="timed out at /v1/rerank"):
        server.rerank("question", [f"document-{index}" for index in range(16)])

    assert server.last_pair_tokens == (2,) * 16
    assert server.last_batch_sizes == (8, 8)
    assert server.last_attempted_batch_count == 2
    assert server.last_completed_batch_count == 1
