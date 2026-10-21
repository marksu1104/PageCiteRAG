"""The product CLI uses one cited-answer contract across retrieval choices."""

import pytest

from page_cite_rag import cli
from test_answer_cli import FakeClient
from test_workflows import setup as setup


def test_cli_uses_unified_cited_answer_contract(setup, monkeypatch, capsys):
    from page_cite_rag.retrieval import build_bm25_index

    build_bm25_index(setup[0], setup[1])
    monkeypatch.setattr(cli, "_open_generation_client", lambda args, config: FakeClient(config))
    assert (
        cli.main(
            [
                "ask",
                setup[1],
                "Were children included?",
                "--db",
                str(setup[0].database_path),
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert '"prompt_version": "complete-cited-answer-v4"' in capsys.readouterr().out


@pytest.mark.parametrize("command", ["ask", "explain"])
def test_retired_citation_strategy_flag_is_rejected_before_storage(setup, monkeypatch, command):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid policy must not open storage or model")

    monkeypatch.setattr(cli, "SQLiteDocumentStore", forbidden)
    arguments = [command, setup[1], "Question?" if command == "ask" else setup[2][0].block_id]
    with pytest.raises(SystemExit) as error:
        cli.main(arguments + ["--citation-strategy", "spans"])
    assert error.value.code == 2
