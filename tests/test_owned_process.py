from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="Linux parent-death signal semantics")
def test_owned_server_survives_langgraph_tool_worker_exit(tmp_path):
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    from page_cite_rag.owned_process import launch_owned_llama_process

    children: list[subprocess.Popen] = []

    @tool
    def start_owned_child() -> str:
        """Start a short-lived CPU child through the production ownership wrapper."""
        child = launch_owned_llama_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            env={"PATH": os.defpath, "LANG": "C.UTF-8"},
            stdout=subprocess.DEVNULL,
        )
        children.append(child)
        return str(child.pid)

    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode([start_owned_child]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)

    try:
        graph.compile().invoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "start_owned_child",
                                "args": {},
                                "id": "launch-child",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            }
        )
        assert len(children) == 1
        time.sleep(0.2)
        assert children[0].poll() is None
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
