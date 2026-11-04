"""Exact prompt token counts from the tokenizer stored in a GGUF model file.

A server such as Ollama reports prompt tokens only after generating. Admission needs the
count before dispatch, so the byte-level BPE vocabulary and merges are read from the same
GGUF file the server loads. Only pre-tokenizers whose splitting rule has been checked
against llama.cpp ``/tokenize`` are accepted; anything else raises instead of estimating.
"""

from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path

# Qwen2 split rule. For Qwen3.5 (`qwen35`) it reproduced llama.cpp token IDs exactly on
# 2,628 strings: 2,418 Chinese/English document blocks, edge cases and 200 full prompts
# with special tokens (2026-11-04). A mismatch is still caught after each call, because
# the server's own prompt count must equal this count.
_QWEN_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
VERIFIED_PRE_TOKENIZERS = {"qwen2": _QWEN_SPLIT, "qwen35": _QWEN_SPLIT}
_SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_CONTROL, _USER_DEFINED = 3, 4


def read_gguf_metadata(path: Path, *, prefixes: tuple[str, ...] = ("tokenizer.ggml.",)) -> dict:
    """Read GGUF key/value metadata with the given key prefixes; tensors are not read."""
    out: dict = {}
    with Path(path).open("rb") as handle:

        def read(fmt: str):
            size = struct.calcsize(fmt)
            data = handle.read(size)
            if len(data) != size:
                raise ValueError("truncated GGUF metadata")
            return struct.unpack(fmt, data)[0]

        def read_string() -> str:
            return handle.read(read("<Q")).decode("utf-8")

        def read_value(kind: int):
            if kind in _SCALARS:
                return read(_SCALARS[kind])
            if kind == 8:
                return read_string()
            if kind == 9:
                item, count = read("<I"), read("<Q")
                return [read_value(item) for _ in range(count)]
            raise ValueError(f"unsupported GGUF metadata type {kind}")

        if handle.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, _tensors, entries = read("<I"), read("<Q"), read("<Q")
        if version not in (2, 3):
            raise ValueError("unsupported GGUF version")
        for _ in range(entries):
            key = read_string()
            value = read_value(read("<I"))
            if key.startswith(prefixes):
                out[key] = value
    return out


def _tokenizer(path: Path):
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers

    meta = read_gguf_metadata(path)
    pre = meta.get("tokenizer.ggml.pre")
    if meta.get("tokenizer.ggml.model") != "gpt2" or pre not in VERIFIED_PRE_TOKENIZERS:
        raise ValueError(f"no verified tokenizer for GGUF pre-tokenizer {pre!r}")
    tokens = meta["tokenizer.ggml.tokens"]
    types = meta["tokenizer.ggml.token_type"]
    merges = meta["tokenizer.ggml.merges"]
    if len(types) != len(tokens) or meta.get("tokenizer.ggml.add_bos_token"):
        raise ValueError("unsupported GGUF tokenizer layout")
    tokenizer = Tokenizer(
        models.BPE(
            vocab={token: index for index, token in enumerate(tokens)},
            merges=[tuple(merge.split(" ", 1)) for merge in merges],
            ignore_merges=False,
            byte_fallback=False,
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(VERIFIED_PRE_TOKENIZERS[pre]), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(
        [
            AddedToken(tokens[index], special=True, normalized=False)
            for index, kind in enumerate(types)
            if kind in (_CONTROL, _USER_DEFINED)
        ]
    )
    return tokenizer, pre


@lru_cache(maxsize=4)
def _cached(path: str, size: int, mtime_ns: int):
    return _tokenizer(Path(path))


def gguf_tokenizer(path: Path):
    """Return ``(tokenizer, pre_tokenizer_name)``, cached per file identity."""
    stat = Path(path).stat()
    return _cached(str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)


def gguf_token_ids(path: Path, text: str) -> list[int]:
    """Token IDs with special-token parsing, as llama.cpp ``/tokenize`` with parse_special."""
    tokenizer, _pre = gguf_tokenizer(path)
    return tokenizer.encode(text, add_special_tokens=False).ids
