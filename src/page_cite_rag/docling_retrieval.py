"""Reusable hybrid index over Docling search chunks with native source results."""

from __future__ import annotations

import base64
import hashlib
import json
import zlib

import numpy as np

from page_cite_rag.docling_adapter import DoclingChunkRetriever
from page_cite_rag.docling_document import DoclingDocumentBundle, load_bundle
from page_cite_rag.embedding import EmbeddingEncoder, normalized_vectors
from page_cite_rag.query import QueryRewriter
from page_cite_rag.retrieval import RetrievalError, SearchResult
from page_cite_rag.retrieval_profile import FORMAL_RETRIEVAL
from page_cite_rag.store import SQLiteDocumentStore

_INDEX_VERSION = "docling-chunk-dense-v1"


def _identity(encoder: EmbeddingEncoder) -> str:
    encoded = json.dumps(encoder.identity, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_fingerprint(bundle: DoclingDocumentBundle) -> str:
    texts = [entry["search_text"] for entry in bundle.chunks]
    return hashlib.sha256(
        json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _kind(encoder: EmbeddingEncoder) -> str:
    return f"{_INDEX_VERSION}:{_identity(encoder)}"


def build_index(store: SQLiteDocumentStore, document_id: str, encoder: EmbeddingEncoder) -> dict:
    """Encode once and activate an immutable, model-specific chunk matrix."""
    bundle = load_bundle(store, document_id)
    existing = store.get_retrieval_manifest(document_id, _kind(encoder))
    if existing is not None:
        return {
            key: value
            for key, value in json.loads(existing).items()
            if key != "vectors_zlib_base64"
        } | {"duplicate": True}
    texts = [entry["search_text"] for entry in bundle.chunks]
    vectors = normalized_vectors(encoder.encode_documents(texts), len(texts), encoder.dimension)
    raw = vectors.astype("<f4", copy=False).tobytes()
    manifest = {
        "version": _INDEX_VERSION,
        "document_id": document_id,
        "source_fingerprint": _source_fingerprint(bundle),
        "encoder_identity": encoder.identity,
        "shape": [len(texts), encoder.dimension],
        "vectors_sha256": hashlib.sha256(raw).hexdigest(),
        "vectors_zlib_base64": base64.b64encode(zlib.compress(raw, level=6)).decode("ascii"),
    }
    serialized = json.dumps(manifest, sort_keys=True, ensure_ascii=False, allow_nan=False)
    store.save_retrieval_manifest_once(document_id, _kind(encoder), serialized)
    return {key: value for key, value in manifest.items() if key != "vectors_zlib_base64"} | {
        "duplicate": False
    }


class _IndexedEncoder:
    def __init__(self, encoder: EmbeddingEncoder, vectors: np.ndarray, texts: list[str]) -> None:
        self._encoder = encoder
        self._vectors = vectors
        self._texts = texts
        self.identity = encoder.identity
        self.dimension = encoder.dimension

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if texts != self._texts:
            raise RetrievalError("Docling indexed search texts changed")
        return self._vectors

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encoder.encode_queries(texts)


class _ProfiledDoclingRetriever(DoclingChunkRetriever):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        allowed_chunk_indices: tuple[int, ...] | None = None,
    ) -> SearchResult:
        result = super().search(query, top_k=top_k, allowed_chunk_indices=allowed_chunk_indices)
        self.last_query_plan = {
            **(self.last_query_plan or {}),
            "profile": {
                **FORMAL_RETRIEVAL.as_dict(),
                "candidates_per_branch": self.candidates,
                "rrf_constant": self.constant,
            },
            "requested_top_k": top_k,
            "scope_chunk_indexes": list(allowed_chunk_indices)
            if allowed_chunk_indices is not None
            else None,
        }
        return result


def load_retriever(
    store: SQLiteDocumentStore,
    document_id: str,
    encoder: EmbeddingEncoder,
    *,
    candidates: int = 20,
    constant: int = 60,
) -> DoclingChunkRetriever:
    bundle = load_bundle(store, document_id)
    raw = store.get_retrieval_manifest(document_id, _kind(encoder))
    if raw is None:
        raise RetrievalError("Docling hybrid index is missing; run index --method hybrid")
    try:
        manifest = json.loads(raw)
        shape = manifest["shape"]
        if (
            manifest["version"] != _INDEX_VERSION
            or manifest["document_id"] != document_id
            or manifest["source_fingerprint"] != _source_fingerprint(bundle)
            or manifest["encoder_identity"] != encoder.identity
            or shape != [len(bundle.chunks), encoder.dimension]
        ):
            raise RetrievalError("Docling index identity differs from the source or encoder")
        compressed = base64.b64decode(manifest["vectors_zlib_base64"], validate=True)
        expected_bytes = len(bundle.chunks) * encoder.dimension * 4
        decompressor = zlib.decompressobj()
        vector_bytes = decompressor.decompress(compressed, expected_bytes + 1)
        if (
            len(vector_bytes) != expected_bytes
            or not decompressor.eof
            or hashlib.sha256(vector_bytes).hexdigest() != manifest["vectors_sha256"]
        ):
            raise RetrievalError("Docling index vectors changed")
        vectors = np.frombuffer(vector_bytes, dtype="<f4").reshape(shape).copy()
    except (KeyError, ValueError, TypeError, zlib.error) as exc:
        raise RetrievalError("Docling index manifest is invalid") from exc
    return _ProfiledDoclingRetriever(
        bundle.source,
        bundle.chunks,
        _IndexedEncoder(encoder, vectors, [entry["search_text"] for entry in bundle.chunks]),
        candidates=candidates,
        constant=constant,
    )


class RewritingDoclingRetriever:
    """Apply the existing bounded English lexical rewrite to Docling's hybrid search."""

    def __init__(self, retriever: DoclingChunkRetriever, rewriter: QueryRewriter) -> None:
        self._retriever = retriever
        self._rewriter = rewriter
        self.last_query_plan: dict | None = None

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        allowed_chunk_indices: tuple[int, ...] | None = None,
    ) -> SearchResult:
        plan = self._rewriter.rewrite(query)
        self._retriever.lexical_overrides[query] = plan.lexical_query
        try:
            result = self._retriever.search(
                query, top_k=top_k, allowed_chunk_indices=allowed_chunk_indices
            )
            self.last_query_plan = {
                **(self._retriever.last_query_plan or {}),
                "rewrite": plan.model_dump(mode="json"),
                "scope_chunk_indexes": list(allowed_chunk_indices)
                if allowed_chunk_indices is not None
                else None,
            }
            return result
        finally:
            self._retriever.lexical_overrides.pop(query, None)
