"""Explicitly built, persisted BM25 indexes over one document's source blocks."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Iterable, Literal

import bm25s
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from doc_rag.models import Block
from doc_rag.store import SQLiteDocumentStore
from doc_rag.tokenize import LexicalTokenizer, tokenizer_version

_INDEX_FILES = frozenset(
    {
        "data.csc.index.npy",
        "indices.csc.index.npy",
        "indptr.csc.index.npy",
        "vocab.index.json",
        "params.index.json",
    }
)
_CONFIG_VERSION = "paragraph-v1/lucene-k1-1.5-b-0.75/numpy-float32-v1"


class RetrievalError(RuntimeError):
    """A missing, incompatible or damaged index needs an explicit rebuild."""


class IndexManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[1] = 1
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    engine_version: str
    tokenizer_version: str
    config_version: str
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    block_ids: tuple[str, ...]
    file_sha256: dict[str, str]
    created_at: datetime


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)
    block: Block


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    index_id: str
    method: Literal["bm25", "dense", "hybrid"] = "bm25"
    query: str
    hits: tuple[SearchHit, ...]
    elapsed_ms: float = Field(ge=0)


def _fingerprint(blocks: list[Block]) -> str:
    digest = hashlib.sha256()
    for block in blocks:
        digest.update(
            json.dumps(block.model_dump(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def index_root(store: SQLiteDocumentStore) -> Path:
    return store.database_path.with_name(store.database_path.name + ".indexes")


def build_bm25_index(store: SQLiteDocumentStore, document_id: str) -> IndexManifest:
    """Build new files before atomically changing the active SQLite manifest."""
    document = store.get_document(document_id)
    if document is None:
        raise RetrievalError("document was not found")
    blocks = store.get_blocks(document_id)
    if not blocks:
        raise RetrievalError("document has no extractable blocks to index")
    tokenizer = LexicalTokenizer()
    corpus = [tokenizer.tokenize(block.text) for block in blocks]
    if not any(corpus):
        raise RetrievalError("document has no searchable words to index")

    engine = bm25s.BM25(method="lucene", k1=1.5, b=0.75, backend="numpy")
    engine.index(corpus, create_empty_token=False, show_progress=False)
    root = index_root(store)
    root.mkdir(parents=True, exist_ok=True)
    index_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix=".build-", dir=root) as temporary:
        staging = Path(temporary) / "index"
        engine.save(staging, allow_pickle=False)
        if {path.name for path in staging.iterdir()} != _INDEX_FILES:
            raise RetrievalError("BM25 produced an unsupported index format")
        manifest = IndexManifest(
            document_id=document_id,
            index_id=index_id,
            engine_version=version("bm25s"),
            tokenizer_version=tokenizer_version(),
            config_version=_CONFIG_VERSION,
            source_fingerprint=_fingerprint(blocks),
            block_ids=tuple(block.block_id for block in blocks),
            file_sha256={name: _file_hash(staging / name) for name in sorted(_INDEX_FILES)},
            created_at=datetime.now(timezone.utc),
        )
        staging.rename(root / index_id)
        # An activation failure leaves an unused directory, not a partial active index.
        store.save_retrieval_manifest(document_id, "bm25", manifest.model_dump_json())
    return manifest


class BM25Retriever:
    """One loaded snapshot for many queries; never builds or downloads on search."""

    def __init__(self, store: SQLiteDocumentStore, document_id: str) -> None:
        raw = store.get_retrieval_manifest(document_id, "bm25")
        if raw is None:
            raise RetrievalError("no BM25 index; run doc-rag index for this document first")
        try:
            manifest = IndexManifest.model_validate_json(raw)
        except ValidationError as exc:
            raise RetrievalError("invalid index manifest; rebuild the index") from exc
        if manifest.document_id != document_id:
            raise RetrievalError("index belongs to a different document; rebuild the index")
        if (
            manifest.engine_version != version("bm25s")
            or manifest.tokenizer_version != tokenizer_version()
            or manifest.config_version != _CONFIG_VERSION
        ):
            raise RetrievalError(
                "index configuration or package version changed; rebuild the index"
            )
        blocks = store.get_blocks(document_id)
        if (
            tuple(block.block_id for block in blocks) != manifest.block_ids
            or _fingerprint(blocks) != manifest.source_fingerprint
        ):
            raise RetrievalError("index does not match stored source blocks; rebuild the index")
        directory = index_root(store) / manifest.index_id
        try:
            if set(manifest.file_sha256) != _INDEX_FILES:
                raise RetrievalError("index file manifest is incomplete; rebuild the index")
            for name, expected in manifest.file_sha256.items():
                path = directory / name
                if path.is_symlink() or _file_hash(path) != expected:
                    raise RetrievalError("index checksum mismatch; rebuild the index")
            engine = bm25s.BM25.load(directory, mmap=True, load_corpus=False, allow_pickle=False)
            self._validate_engine(engine, len(blocks))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RetrievalError("index files are unreadable; rebuild the index") from exc
        self.manifest = manifest
        self._blocks = blocks
        self._engine = engine
        self._tokenizer = LexicalTokenizer()

    @staticmethod
    def _validate_engine(engine: bm25s.BM25, count: int) -> None:
        """Validate the sparse snapshot before using its positional block mapping."""
        data, indices, pointers = (engine.scores[key] for key in ("data", "indices", "indptr"))
        vocab = engine.vocab_dict
        valid = (
            engine.method == "lucene"
            and engine.idf_method == "lucene"
            and engine.k1 == 1.5
            and engine.b == 0.75
            and engine.backend == "numpy"
            and engine.scores["num_docs"] == count
            and data.ndim == indices.ndim == pointers.ndim == 1
            and data.dtype == np.dtype("float32")
            and indices.dtype == pointers.dtype == np.dtype("int32")
            and len(data) == len(indices)
            and len(pointers) == len(vocab) + 1
            and pointers[0] == 0
            and pointers[-1] == len(data)
            and np.all(np.diff(pointers.astype(np.int64)) >= 0)
            and np.all((indices >= 0) & (indices < count))
            and np.all(np.isfinite(data) & (data >= 0))
            and all(isinstance(token, str) for token in vocab)
            and all(type(value) is int for value in vocab.values())
            and set(vocab.values()) == set(range(len(vocab)))
        )
        if not valid:
            raise RetrievalError("index arrays or parameters are incompatible; rebuild the index")

    def search(self, query: str, *, top_k: int = 5) -> SearchResult:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        start = time.perf_counter()
        tokens = [
            token for token in self._tokenizer.tokenize(query) if token in self._engine.vocab_dict
        ]
        hits: tuple[SearchHit, ...] = ()
        if tokens:
            scores = self._engine.get_scores(tokens)
            if scores.shape != (len(self._blocks),) or not np.all(np.isfinite(scores)):
                raise RetrievalError("BM25 returned invalid scores")
            candidates = np.flatnonzero(scores > 0)
            # Stable paragraph order resolves ties; never return zero-match filler.
            ordered = candidates[np.argsort(-scores[candidates], kind="stable")][:top_k]
            hits = tuple(
                SearchHit(rank=rank, score=float(scores[index]), block=self._blocks[index])
                for rank, index in enumerate(ordered, start=1)
            )
        return SearchResult(
            document_id=self.manifest.document_id,
            index_id=self.manifest.index_id,
            query=query,
            hits=hits,
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )

    def search_many(self, queries: Iterable[str], *, top_k: int = 5) -> Iterable[SearchResult]:
        """Stream results while sharing this index, tokenizer and source snapshot."""
        for query in queries:
            yield self.search(query, top_k=top_k)
