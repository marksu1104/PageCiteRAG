"""Exact cosine retrieval with safe persistent snapshots and parent-block ranking."""

from __future__ import annotations

import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from doc_rag.embedding import EmbeddingEncoder, normalized_vectors
from doc_rag.retrieval import (
    RetrievalError,
    SearchHit,
    SearchResult,
    _file_hash,
    _fingerprint,
    index_root,
)
from doc_rag.store import SQLiteDocumentStore


class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    block_index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)


class DenseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[1] = 1
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder: dict
    dimension: int = Field(gt=0)
    block_ids: tuple[str, ...]
    segments: tuple[Segment, ...]
    vectors_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime


def _validate_segments(segments: tuple[Segment, ...], blocks: list) -> None:
    previous = -1
    covered = [0] * len(blocks)
    for segment in segments:
        index = segment.block_index
        if (
            index < previous
            or index >= len(blocks)
            or segment.char_start > covered[index]
            or segment.char_end <= max(segment.char_start, covered[index])
            or segment.char_end > len(blocks[index].text)
        ):
            raise RetrievalError("dense segment mapping is invalid; rebuild the index")
        covered[index] = segment.char_end
        previous = index
    if covered != [len(block.text) for block in blocks]:
        raise RetrievalError("dense segments do not cover all source blocks; rebuild the index")


def build_dense_index(
    store: SQLiteDocumentStore, document_id: str, encoder: EmbeddingEncoder
) -> DenseManifest:
    blocks = store.get_blocks(document_id)
    if not blocks:
        raise RetrievalError("document has no extractable blocks to index")
    segments = tuple(
        Segment(block_index=index, char_start=start, char_end=end)
        for index, block in enumerate(blocks)
        for start, end in encoder.split_document(block.text)
    )
    _validate_segments(segments, blocks)
    texts = [blocks[s.block_index].text[s.char_start : s.char_end] for s in segments]
    vectors = normalized_vectors(encoder.encode_documents(texts), len(texts), encoder.dimension)
    root = index_root(store)
    root.mkdir(parents=True, exist_ok=True)
    index_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix=".dense-", dir=root) as temporary:
        from pathlib import Path

        staging = Path(temporary) / "index"
        staging.mkdir()
        np.save(staging / "vectors.npy", vectors, allow_pickle=False)
        manifest = DenseManifest(
            document_id=document_id,
            index_id=index_id,
            source_fingerprint=_fingerprint(blocks),
            encoder=encoder.identity,
            dimension=encoder.dimension,
            block_ids=tuple(block.block_id for block in blocks),
            segments=segments,
            vectors_sha256=_file_hash(staging / "vectors.npy"),
            created_at=datetime.now(timezone.utc),
        )
        staging.rename(root / index_id)
        store.save_retrieval_manifest(document_id, "dense", manifest.model_dump_json())
    return manifest


class DenseRetriever:
    """Reuse one encoder and mmap snapshot; aggregate subchunks before top-k."""

    def __init__(
        self, store: SQLiteDocumentStore, document_id: str, encoder: EmbeddingEncoder
    ) -> None:
        raw = store.get_retrieval_manifest(document_id, "dense")
        if raw is None:
            raise RetrievalError("no dense index; build one explicitly before searching")
        try:
            manifest = DenseManifest.model_validate_json(raw)
        except ValidationError as exc:
            raise RetrievalError("invalid dense manifest; rebuild the index") from exc
        blocks = store.get_blocks(document_id)
        if (
            manifest.document_id != document_id
            or manifest.encoder != encoder.identity
            or manifest.dimension != encoder.dimension
            or manifest.block_ids != tuple(block.block_id for block in blocks)
            or manifest.source_fingerprint != _fingerprint(blocks)
        ):
            raise RetrievalError(
                "dense source or embedding configuration changed; rebuild the index"
            )
        _validate_segments(manifest.segments, blocks)
        directory = index_root(store) / manifest.index_id
        path = directory / "vectors.npy"
        try:
            if (
                directory.is_symlink()
                or path.is_symlink()
                or _file_hash(path) != manifest.vectors_sha256
            ):
                raise RetrievalError("dense checksum mismatch; rebuild the index")
            vectors = np.load(path, allow_pickle=False, mmap_mode="r")
            if (
                vectors.dtype != np.dtype("float32")
                or vectors.shape != (len(manifest.segments), encoder.dimension)
                or not np.all(np.isfinite(vectors))
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
            ):
                raise RetrievalError(
                    "dense vector shape or normalization invalid; rebuild the index"
                )
        except (OSError, ValueError, TypeError) as exc:
            raise RetrievalError("dense vectors unreadable; rebuild the index") from exc
        self.manifest = manifest
        self._blocks = blocks
        self._vectors = vectors
        self._encoder = encoder
        self._parents = np.array([s.block_index for s in manifest.segments], dtype=np.int64)

    def search(self, query: str, *, top_k: int = 5) -> SearchResult:
        if not isinstance(query, str) or type(top_k) is not int or top_k < 1:
            raise ValueError("query must be a string and top_k a positive integer")
        start = time.perf_counter()
        hits = ()
        if query.strip():
            vector = normalized_vectors(
                self._encoder.encode_queries([query]), 1, self.manifest.dimension
            )
            scores = self._vectors @ vector[0]
            block_scores = np.full(len(self._blocks), -np.inf, dtype=np.float32)
            np.maximum.at(block_scores, self._parents, scores)
            order = np.argsort(-block_scores, kind="stable")[:top_k]
            hits = tuple(
                SearchHit(rank=rank, score=float(block_scores[i]), block=self._blocks[i])
                for rank, i in enumerate(order, 1)
            )
        return SearchResult(
            document_id=self.manifest.document_id,
            index_id=self.manifest.index_id,
            method="dense",
            query=query,
            hits=hits,
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
