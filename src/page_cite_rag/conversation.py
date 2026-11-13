"""Bounded conversation history and standalone follow-up questions."""

from dataclasses import dataclass
from typing import Sequence

from page_cite_rag.generation import ChatMessage, ModelError

# Source: langchain condense question prompt (ConversationalRetrievalChain).
CONDENSE_QUESTION_PROMPT = (
    "Given the following conversation and a follow up question, rephrase the follow up "
    "question to be a standalone question, in its original language.\n\n"
    "Chat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
)


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str


def format_history(turns: Sequence[Turn]) -> str:
    return "\n".join(
        f"Human: {turn.question}\nAssistant: "
        + (turn.answer[:1500] + "…" if len(turn.answer) > 1500 else turn.answer)
        for turn in turns[-6:]
    )


def condense_question(client, turns: Sequence[Turn], question: str) -> tuple[str, dict]:
    record = {"rewritten": False, "standalone": question, "error": None, "stats": None}
    if not turns:
        return question, record
    prompt = CONDENSE_QUESTION_PROMPT.format(chat_history=format_history(turns), question=question)
    try:
        generated = client.generate([ChatMessage(role="user", content=prompt)], None)
        record["stats"] = generated.stats.model_dump(mode="json")
        standalone = next(
            (line.strip() for line in generated.value.splitlines() if line.strip()), ""
        )
        standalone = standalone.strip("\"'").strip()
        if standalone and len(standalone) <= 1000:
            record.update(rewritten=standalone != question, standalone=standalone)
    except (ModelError, OSError, ValueError) as exc:
        record["error"] = type(exc).__name__
    return record["standalone"], record
