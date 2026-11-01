from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def embedding_tokenizer_dir(tmp_path_factory):
    """A tiny tokenizer, so runner tests never depend on ignored local model assets."""
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    directory = tmp_path_factory.mktemp("embedding-tokenizer")
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    return directory
