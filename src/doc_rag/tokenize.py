"""Versioned lexical tokenization; source text is never changed in the store."""

from __future__ import annotations

import re
import tempfile
import unicodedata
from importlib.metadata import version

_HAN = r"[\u3400-\u4dbf\u4e00-\u9fff]"
_WORD = rf"(?:(?!{_HAN})[^\W_])+"
_TOKEN = re.compile(rf"{_HAN}+|-?\d+(?:[.,]\d+)*|{_WORD}(?:[.'-]{_WORD})*")
_HAN_RUN = re.compile(rf"{_HAN}+")


def tokenizer_version() -> str:
    return (
        f"nfkc-casefold-mathminus-jieba-search-no-hmm-v1/jieba-{version('jieba')}"
        f"/unicode-{unicodedata.unidata_version}"
    )


class LexicalTokenizer:
    """Keep negations, numbers and English terms; segment Han runs with jieba."""

    def __init__(self) -> None:
        self._segmenter = None

    def tokenize(self, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text).casefold().replace("\u2212", "-")
        tokens: list[str] = []
        for match in _TOKEN.finditer(normalized):
            token = match.group()
            if _HAN_RUN.fullmatch(token):
                if self._segmenter is None:
                    import jieba

                    self._segmenter = jieba.Tokenizer()
                    # Do not read a shared, potentially stale /tmp/jieba.cache.
                    with tempfile.TemporaryDirectory(prefix="doc-rag-jieba-") as cache:
                        self._segmenter.tmp_dir = cache
                        self._segmenter.initialize()
                tokens.extend(self._segmenter.cut_for_search(token, HMM=False))
            else:
                tokens.append(token)
        return tokens
