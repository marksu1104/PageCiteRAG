from __future__ import annotations

import struct

import pytest
from tokenizers import pre_tokenizers

from page_cite_rag.gguf_tokenizer import gguf_token_ids, read_gguf_metadata


def _string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _entry(key: str, kind: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", kind) + payload


def _string_array(values: list[str]) -> bytes:
    return struct.pack("<IQ", 8, len(values)) + b"".join(_string(v) for v in values)


def write_gguf(path, *, pre="qwen35"):
    """Tiny byte-level BPE vocabulary in GGUF metadata form (no tensors)."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokens = [*alphabet, "ab", "Ġab", "<|im_end|>"]
    types = [1] * (len(tokens) - 1) + [3]
    entries = [
        _entry("general.architecture", 8, _string("qwen35")),
        _entry("tokenizer.ggml.model", 8, _string("gpt2")),
        _entry("tokenizer.ggml.pre", 8, _string(pre)),
        _entry("tokenizer.ggml.tokens", 9, _string_array(tokens)),
        _entry(
            "tokenizer.ggml.token_type",
            9,
            struct.pack("<IQ", 5, len(types)) + b"".join(struct.pack("<i", t) for t in types),
        ),
        _entry("tokenizer.ggml.merges", 9, _string_array(["a b", "Ġ ab"])),
    ]
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(entries)) + b"".join(entries))
    return {token: index for index, token in enumerate(tokens)}


def test_gguf_tokenizer_applies_merges_and_parses_special_tokens(tmp_path):
    path = tmp_path / "model.gguf"
    ids = write_gguf(path)
    assert read_gguf_metadata(path)["tokenizer.ggml.pre"] == "qwen35"
    assert gguf_token_ids(path, "ab ab<|im_end|>") == [ids["ab"], ids["Ġab"], ids["<|im_end|>"]]


def test_gguf_tokenizer_refuses_an_unverified_pre_tokenizer(tmp_path):
    path = tmp_path / "model.gguf"
    write_gguf(path, pre="llama-bpe")
    with pytest.raises(ValueError, match="no verified tokenizer"):
        gguf_token_ids(path, "ab")


def test_gguf_reader_rejects_other_files(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"not a gguf")
    with pytest.raises(ValueError, match="not a GGUF"):
        read_gguf_metadata(path)
