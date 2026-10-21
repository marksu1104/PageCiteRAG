"""Linux exec wrapper: the owned native server receives TERM if its parent dies."""

from __future__ import annotations

import ctypes
import os
import signal
import sys


def main() -> None:
    if sys.platform != "linux" or len(sys.argv) < 3:
        raise SystemExit("owned llama process requires Linux and an explicit parent")
    parent_pid = int(sys.argv[1])
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int
    # PR_SET_PDEATHSIG. No preexec_fn in a potentially multi-threaded application.
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != parent_pid:
        raise SystemExit("could not establish native process ownership")
    os.execve(sys.argv[2], sys.argv[2:], os.environ)


if __name__ == "__main__":
    main()
