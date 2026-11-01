"""Launch native servers from a thread whose lifetime is the application process."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

_PROCESS_LAUNCHER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="owned-process-launcher")


def launch_owned_llama_process(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str], stdout
) -> subprocess.Popen:
    """Start an owned llama.cpp server from a persistent launcher thread.

    Linux PR_SET_PDEATHSIG is tied to the thread that creates the child process.
    LangGraph's tool workers are short lived, so they must not create these servers.
    """
    if not command or any(not isinstance(argument, str) for argument in command):
        raise ValueError("owned server command must contain strings")

    launcher_command = [
        sys.executable,
        "-I",
        "-m",
        "page_cite_rag.llama_process",
        str(os.getpid()),
        *command,
    ]
    return _PROCESS_LAUNCHER.submit(
        subprocess.Popen,
        launcher_command,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    ).result()
