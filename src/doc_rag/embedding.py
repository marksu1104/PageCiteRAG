"""Pinned, explicitly prepared embeddings; inference only reads local model files."""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

import numpy as np

from doc_rag.retrieval import RetrievalError

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DEFAULT_MODEL_DIR = Path("models/qwen3-embedding-0.6b") / MODEL_REVISION
# Git blob hashes for small files; SHA-256 LFS hashes for large objects.
MODEL_FILES = {
    "1_Pooling/config.json": (313, "b6291baabbc39f9792d6759883a47d7d5bd0fbcf"),
    "README.md": (17237, "00a82ea60dfb1d8aed899c16e2b4f12673cd2f79"),
    "config.json": (727, "cef2749ee93607b8f9a58ec72f4f6bfaf874e71d"),
    "config_sentence_transformers.json": (215, "76aef3ade63553ebb698fe3c2a3264040ed093f8"),
    "merges.txt": (1671853, "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7"),
    "model.safetensors": (
        1191586416,
        "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
    ),
    "modules.json": (349, "952a9b81c0bfd99800fabf352f69c7ccd46c5e43"),
    "tokenizer.json": (
        11423705,
        "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
    ),
    "tokenizer_config.json": (9706, "7345216a0785dc7086e8c245b2a9d3896ce2b756"),
    "vocab.json": (2776833, "4783fe10ac3adce15ac8f358ef5462739852c569"),
}


def verify_model(directory: Path) -> dict[str, str]:
    """Validate allowlisted files against the pinned upstream snapshot, without network."""
    if directory.is_symlink() or not directory.is_dir():
        raise RetrievalError("local embedding model is missing; run model prepare-embedding")
    actual_files = {
        str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()
    }
    if actual_files != set(MODEL_FILES) or any(path.is_symlink() for path in directory.rglob("*")):
        raise RetrievalError("local model file list changed; prepare a clean model directory")
    hashes = {}
    for name, (size, expected) in MODEL_FILES.items():
        path = directory / name
        if path.stat().st_size != size:
            raise RetrievalError("local embedding model size mismatch")
        digest = hashlib.sha256() if len(expected) == 64 else hashlib.sha1()
        if len(expected) == 40:
            digest.update(f"blob {size}\0".encode())
        sha256 = hashlib.sha256()
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
                sha256.update(chunk)
        if digest.hexdigest() != expected:
            raise RetrievalError("local embedding model checksum mismatch")
        hashes[name] = sha256.hexdigest()
    return hashes


def prepare_embedding_model(directory: Path = DEFAULT_MODEL_DIR) -> dict:
    """The sole network operation for models; never overwrite an existing snapshot."""
    directory = directory.expanduser()
    if directory.exists() or directory.is_symlink():
        hashes = verify_model(directory)
    else:
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".model-", dir=directory.parent) as temporary:
            staging = Path(temporary) / "snapshot"
            staging.mkdir()
            for name, (size, _) in MODEL_FILES.items():
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
                with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as out:
                    received = 0
                    while chunk := response.read(1024 * 1024):
                        received += len(chunk)
                        if received > size:
                            raise RetrievalError("embedding download exceeded its expected size")
                        out.write(chunk)
            hashes = verify_model(staging)
            staging.rename(directory)
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": "Apache-2.0",
        "directory": str(directory),
        "file_sha256": hashes,
    }


class EmbeddingEncoder(Protocol):
    """Injectable model boundary; core tests need neither Torch nor model downloads."""

    identity: dict
    dimension: int

    def split_document(self, text: str) -> list[tuple[int, int]]: ...

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


def normalized_vectors(value: np.ndarray, rows: int, dimension: int) -> np.ndarray:
    vectors = np.asarray(value, dtype=np.float32)
    if vectors.shape != (rows, dimension) or not np.all(np.isfinite(vectors)):
        raise RetrievalError("embedding returned invalid shape or nonfinite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise RetrievalError("embedding returned zero or invalid norms")
    return vectors / norms


class QwenEmbeddingEncoder:
    """One CPU model reused across documents and queries, with explicit token limits."""

    dimension = 1024

    def __init__(
        self,
        directory: Path = DEFAULT_MODEL_DIR,
        *,
        max_tokens: int = 512,
        overlap: int = 64,
        batch_size: int = 4,
        threads: int = 6,
    ) -> None:
        if (
            type(max_tokens) is not int
            or not 32 <= max_tokens <= 32768
            or type(overlap) is not int
            or not 0 <= overlap < max_tokens - 16
            or type(batch_size) is not int
            or batch_size < 1
            or type(threads) is not int
            or threads < 1
        ):
            raise ValueError("invalid embedding token, overlap, batch or thread budget")
        hashes = verify_model(directory.expanduser())
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RetrievalError("embedding requires the optional ml-cpu extra") from exc
        torch.set_num_threads(threads)
        self._model = SentenceTransformer(
            str(directory.expanduser().resolve()),
            device="cpu",
            local_files_only=True,
            trust_remote_code=False,
            model_kwargs={"dtype": torch.float32, "use_safetensors": True},
            processor_kwargs={"padding_side": "left"},
        )
        self._model.max_seq_length = max_tokens
        self._batch_size = batch_size
        self._max_tokens = max_tokens
        self._overlap = overlap
        self._prompt = self._model.prompts.get("query")
        if not self._prompt or self._model.get_embedding_dimension() != self.dimension:
            raise RetrievalError("local embedding prompt or dimension is incompatible")
        self.identity = {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "file_sha256": hashes,
            "dimension": self.dimension,
            "query_prompt": self._prompt,
            "document_prompt": "",
            "max_tokens": max_tokens,
            "overlap": overlap,
            "chunking": "original-character-offset-windows/max-score-per-block/v1",
            "device": "cpu",
            "dtype": "float32",
            "batch_size": batch_size,
            "threads": threads,
            "versions": {
                name: version(name)
                for name in ("sentence-transformers", "transformers", "torch", "safetensors")
            },
        }

    def _length(self, text: str) -> int:
        return len(self._model.tokenizer(text, add_special_tokens=True, verbose=False)["input_ids"])

    def split_document(self, text: str) -> list[tuple[int, int]]:
        if self._length(text) <= self._max_tokens:
            return [(0, len(text))]
        offsets = self._model.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )["offset_mapping"]
        # Leave room for boundary re-tokenization and model special tokens.
        window = self._max_tokens - 16
        spans = []
        start = 0
        while start < len(offsets):
            end = min(start + window, len(offsets))
            char_start = 0 if start == 0 else offsets[start][0]
            if spans:
                char_start = min(char_start, spans[-1][1])
            char_end = len(text) if end == len(offsets) else offsets[end - 1][1]
            while self._length(text[char_start:char_end]) > self._max_tokens:
                end -= 1
                if end <= start + self._overlap:
                    raise RetrievalError("cannot split passage within its embedding budget")
                char_end = offsets[end - 1][1]
            spans.append((char_start, char_end))
            if end == len(offsets):
                break
            start = end - self._overlap
        return spans

    def _encode(self, texts: list[str], *, query: bool) -> np.ndarray:
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("embedding texts must be nonempty strings")
        if any(
            self._length((self._prompt if query else "") + text) > self._max_tokens
            for text in texts
        ):
            raise RetrievalError("embedding input exceeds the explicit token budget; not truncated")
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        options = {"prompt_name": "query"} if query else {"prompt": ""}
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            **options,
        )
        return normalized_vectors(vectors, len(texts), self.dimension)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, query=False)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, query=True)
