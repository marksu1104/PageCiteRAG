"""Track only GPU servers started and owned by the current process."""

from __future__ import annotations

import math
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable

_lock = threading.RLock()
_processes: dict[int, subprocess.Popen] = {}

MAX_GPU_MEMORY_MIB = 24_000
GPU_DISPATCH_HEADROOM_C = 15


class GpuDispatchTimeout(TimeoutError):
    """No inference was sent because the dispatch headroom wait expired."""


class GpuDispatchCancelled(TimeoutError):
    """No inference was sent because the owning operation was cancelled."""


def stable_gpu_headroom_window(
    samples: list[dict],
    *,
    now: float,
    max_temperature_c: int,
    headroom_c: int = GPU_DISPATCH_HEADROOM_C,
    stable_seconds: float = 3.0,
    max_sample_gap_seconds: float = 1.25,
) -> dict | None:
    """Return a reusable recent safe window, or ``None`` when it is unproven.

    This does not replace a fresh per-request ownership/thermal check. It only
    avoids sleeping through another three seconds when the owned monitor has
    continuously observed the same safe headroom window between adjacent calls.
    """
    if (
        type(samples) is not list
        or type(now) not in (int, float)
        or not math.isfinite(now)
        or type(max_temperature_c) is not int
        or not 40 <= max_temperature_c <= 88
        or type(headroom_c) is not int
        or not 1 <= headroom_c <= 30
        or type(stable_seconds) not in (int, float)
        or not math.isfinite(stable_seconds)
        or stable_seconds <= 0
        or type(max_sample_gap_seconds) not in (int, float)
        or not math.isfinite(max_sample_gap_seconds)
        or max_sample_gap_seconds <= 0
    ):
        return None

    window = []
    cutoff = now - stable_seconds
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("phase") not in {"monitor", "dispatch"}:
            continue
        sampled_at = sample.get("sampled_monotonic")
        if type(sampled_at) not in (int, float) or not math.isfinite(sampled_at):
            return None
        if sampled_at <= now and sampled_at >= cutoff - max_sample_gap_seconds:
            window.append(sample)
    window.sort(key=lambda item: item["sampled_monotonic"])
    if len(window) < 3:
        return None
    if window[0]["sampled_monotonic"] > cutoff:
        return None
    if now - window[-1]["sampled_monotonic"] > max_sample_gap_seconds:
        return None

    limits = []
    for left, right in zip(window, window[1:]):
        if right["sampled_monotonic"] - left["sampled_monotonic"] > max_sample_gap_seconds:
            return None
    for sample in window:
        temperature = sample.get("temperature_c")
        target = sample.get("target_temperature_c")
        memory = sample.get("memory_mib")
        if (
            type(temperature) is not int
            or type(memory) is not int
            or memory > MAX_GPU_MEMORY_MIB
            or type(sample.get("hw_thermal_slowdown")) is not bool
            or type(sample.get("sw_thermal_slowdown")) is not bool
            or sample["hw_thermal_slowdown"]
            or sample["sw_thermal_slowdown"]
        ):
            return None
        process_state = sample.get("process_state")
        if (
            not isinstance(process_state, dict)
            or process_state.get("unexpected_pids")
            or process_state.get("owned_service_exited")
        ):
            return None
        dispatch_limit = max_temperature_c - 3
        if target is not None:
            if type(target) is not int:
                return None
            dispatch_limit = min(dispatch_limit, target - headroom_c)
        if temperature > dispatch_limit:
            return None
        limits.append(dispatch_limit)
    if not limits:
        return None
    return {
        "seconds": 0.0,
        "dispatch_temperature_limit_c": min(limits),
        "observed_temperature_c": window[-1]["temperature_c"],
        "safe_sample_count": len(window),
        "stable_window_seconds": window[-1]["sampled_monotonic"] - window[0]["sampled_monotonic"],
        "reused_monitor_window": True,
    }


def wait_for_gpu_dispatch(
    check: Callable[[], dict],
    wait: Callable[[float], bool],
    *,
    max_temperature_c: int,
    timeout_seconds: float = 120.0,
    headroom_c: int = 15,
    recent_samples: Callable[[], list[dict]] | None = None,
) -> dict:
    """Apply the shared soft dispatch gate while leaving hard GPU guards in ``check``.

    The caller's check must take a fresh, fail-closed sample. A wait is only a
    scheduling pause; it never clears a thermal or ownership error.
    """
    if (
        type(max_temperature_c) is not int
        or not 40 <= max_temperature_c <= 88
        or type(headroom_c) is not int
        or not 1 <= headroom_c <= 30
        or type(timeout_seconds) not in (int, float)
        or not 3 <= timeout_seconds <= 120
    ):
        raise ValueError("invalid GPU dispatch policy")

    started = time.monotonic()
    safe_samples = 0
    dispatch_limit = max_temperature_c - 3
    first_sample = None
    if recent_samples is not None:
        first_sample = check()
        reusable = stable_gpu_headroom_window(
            recent_samples(),
            now=time.monotonic(),
            max_temperature_c=max_temperature_c,
            headroom_c=headroom_c,
        )
        if reusable is not None:
            reusable["seconds"] = time.monotonic() - started
            return reusable
    while True:
        sample = first_sample if first_sample is not None else check()
        first_sample = None
        temperature = sample.get("temperature_c")
        target = sample.get("target_temperature_c")
        if type(temperature) is not int:
            raise ValueError("GPU dispatch temperature is unavailable")
        if target is not None:
            if type(target) is not int:
                raise ValueError("GPU dispatch target temperature is invalid")
            dispatch_limit = min(dispatch_limit, target - headroom_c)
        safe_samples = safe_samples + 1 if temperature <= dispatch_limit else 0
        elapsed = time.monotonic() - started
        if elapsed >= 3 and safe_samples >= 2:
            return {
                "seconds": elapsed,
                "dispatch_temperature_limit_c": dispatch_limit,
                "observed_temperature_c": temperature,
                "safe_sample_count": safe_samples,
            }
        if elapsed >= timeout_seconds:
            raise GpuDispatchTimeout("GPU dispatch headroom wait timed out; no inference sent")
        if wait(min(1.0, timeout_seconds - elapsed)):
            raise GpuDispatchCancelled("GPU dispatch wait cancelled")


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
