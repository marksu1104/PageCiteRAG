"""Track only GPU servers started and owned by the current process."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable

_lock = threading.RLock()
_processes: dict[int, subprocess.Popen] = {}


@contextmanager
def gpu_process_lifecycle() -> Iterator[None]:
    """Serialize owned process changes with GPU PID snapshots."""
    with _lock:
        yield


def snapshot_gpu_processes(
    snapshot: Callable[[], dict], *, expected_process: subprocess.Popen | None = None
) -> tuple[dict, dict]:
    """Sample GPU state and classify PIDs while owned-process lifecycle is stable.

    A process that exits while a snapshot is being collected may remain in the
    driver's one-sample PID list. Recheck once only in that case; a still-visible
    PID is unknown and remains blocked. The expected service's exit is always
    surfaced separately so it cannot be hidden as a stale sample.
    """
    with _lock:
        registered = dict(_processes)
        value = snapshot()
        if not isinstance(value, dict):
            raise ValueError("GPU snapshot must be an object")
        sampled = set(value.get("pids", ()))
        if any(type(pid) is not int or pid < 1 for pid in sampled):
            raise ValueError("GPU snapshot contains an invalid PID")

        live = {pid for pid, process in registered.items() if process.poll() is None}
        exited_owned = set(registered) - live
        stale = sampled & exited_owned
        latest = sampled
        if stale:
            refreshed = snapshot()
            if not isinstance(refreshed, dict):
                raise ValueError("GPU refresh snapshot must be an object")
            latest = set(refreshed.get("pids", ()))
            if any(type(pid) is not int or pid < 1 for pid in latest):
                raise ValueError("GPU refresh snapshot contains an invalid PID")
            value = {**value, "pids": sorted(latest)}
            stale &= sampled - latest

        for pid in exited_owned:
            if _processes.get(pid) is registered[pid]:
                del _processes[pid]

        expected_exited = False
        if expected_process is not None:
            current = _processes.get(expected_process.pid)
            expected_exited = expected_process.poll() is not None or (
                current is not None and current is not expected_process
            )
        unexpected = latest - live - stale
        return value, {
            "sampled_pids": sorted(sampled),
            "allowed_pids": sorted(live),
            "stale_owned_pids": sorted(stale),
            "unexpected_pids": sorted(unexpected),
            "owned_service_exited": expected_exited,
        }


def register_owned_gpu_process(process: subprocess.Popen) -> None:
    """Allow one process we launched to share a GPU with another owned service."""
    pid = process.pid
    if type(pid) is not int or pid < 1:
        raise ValueError("owned GPU process has an invalid PID")
    with _lock:
        current = _processes.get(pid)
        if current is not None and current is not process:
            raise RuntimeError("GPU PID is already registered to another process")
        _processes[pid] = process


def unregister_owned_gpu_process(process: subprocess.Popen | None) -> None:
    if process is None or type(process.pid) is not int:
        return
    with _lock:
        if _processes.get(process.pid) is process:
            del _processes[process.pid]


def owned_gpu_pids() -> set[int]:
    """Return live PIDs from Popen handles created by this process only."""
    with _lock:
        stopped = [pid for pid, process in _processes.items() if process.poll() is not None]
        for pid in stopped:
            del _processes[pid]
        return set(_processes)
