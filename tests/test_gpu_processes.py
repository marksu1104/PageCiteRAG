from __future__ import annotations

import threading

import pytest

from page_cite_rag import embedding_server, gpu_processes
from page_cite_rag.retrieval import RetrievalError


class FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode


def test_registry_only_allows_live_processes_started_here():
    process = FakeProcess(12345)
    gpu_processes.register_owned_gpu_process(process)
    try:
        assert gpu_processes.owned_gpu_pids() == {12345}
        process.returncode = 0
        assert gpu_processes.owned_gpu_pids() == set()
    finally:
        gpu_processes.unregister_owned_gpu_process(process)


def test_registry_does_not_remove_a_different_handle_with_same_pid():
    first = FakeProcess(23456)
    second = FakeProcess(23456)
    gpu_processes.register_owned_gpu_process(first)
    try:
        with pytest.raises(RuntimeError, match="already registered"):
            gpu_processes.register_owned_gpu_process(second)
        gpu_processes.unregister_owned_gpu_process(second)
        assert gpu_processes.owned_gpu_pids() == {23456}
    finally:
        gpu_processes.unregister_owned_gpu_process(first)


def test_gpu_pid_sampling_waits_until_owned_process_is_registered():
    process = FakeProcess(34567)
    started = threading.Event()
    result = []

    def sample_owned_pids():
        started.set()
        result.append(gpu_processes.owned_gpu_pids())

    with gpu_processes.gpu_process_lifecycle():
        thread = threading.Thread(target=sample_owned_pids)
        thread.start()
        assert started.wait(timeout=1)
        gpu_processes.register_owned_gpu_process(process)
    thread.join(timeout=1)

    try:
        assert not thread.is_alive()
        assert result == [{34567}]
    finally:
        gpu_processes.unregister_owned_gpu_process(process)


def test_gpu_snapshot_and_close_share_one_lifecycle_lock():
    process = FakeProcess(45678)
    snapshot_started = threading.Event()
    allow_snapshot_to_finish = threading.Event()
    closed = threading.Event()
    result = []
    gpu_processes.register_owned_gpu_process(process)

    def snapshot():
        snapshot_started.set()
        assert allow_snapshot_to_finish.wait(timeout=1)
        return {"pids": [process.pid], "memory_mib": 5000, "temperature_c": 65}

    def sample_gpu():
        result.append(gpu_processes.snapshot_gpu_processes(snapshot, expected_process=process))

    def close_gpu():
        with gpu_processes.gpu_process_lifecycle():
            process.returncode = 0
            gpu_processes.unregister_owned_gpu_process(process)
        closed.set()

    sampling = threading.Thread(target=sample_gpu)
    sampling.start()
    assert snapshot_started.wait(timeout=1)
    closing = threading.Thread(target=close_gpu)
    closing.start()
    assert not closed.wait(timeout=0.05)
    allow_snapshot_to_finish.set()
    sampling.join(timeout=1)
    closing.join(timeout=1)

    try:
        assert not sampling.is_alive() and not closing.is_alive()
        sample, ownership = result[0]
        assert sample["pids"] == [process.pid]
        assert ownership["allowed_pids"] == [process.pid]
        assert ownership["unexpected_pids"] == []
        assert ownership["owned_service_exited"] is False
        assert gpu_processes.owned_gpu_pids() == set()
    finally:
        gpu_processes.unregister_owned_gpu_process(process)


def test_gpu_snapshot_rechecks_exited_owned_pid_before_allowing_it():
    process = FakeProcess(56789)
    gpu_processes.register_owned_gpu_process(process)
    calls = 0

    def snapshot():
        nonlocal calls
        calls += 1
        if calls == 1:
            process.returncode = 0
            return {"pids": [process.pid], "memory_mib": 5000, "temperature_c": 65}
        return {"pids": [], "memory_mib": 5000, "temperature_c": 65}

    try:
        sample, ownership = gpu_processes.snapshot_gpu_processes(snapshot, expected_process=process)
        assert calls == 2
        assert sample["pids"] == []
        assert ownership["stale_owned_pids"] == [process.pid]
        assert ownership["unexpected_pids"] == []
        assert ownership["owned_service_exited"] is True
    finally:
        gpu_processes.unregister_owned_gpu_process(process)


def test_gpu_snapshot_keeps_a_reused_or_unknown_pid_blocked():
    process = FakeProcess(67890)
    gpu_processes.register_owned_gpu_process(process)
    calls = 0

    def snapshot():
        nonlocal calls
        calls += 1
        if calls == 1:
            process.returncode = 0
        return {"pids": [process.pid], "memory_mib": 5000, "temperature_c": 65}

    try:
        _sample, ownership = gpu_processes.snapshot_gpu_processes(snapshot)
        assert calls == 2
        assert ownership["stale_owned_pids"] == []
        assert ownership["unexpected_pids"] == [process.pid]
    finally:
        gpu_processes.unregister_owned_gpu_process(process)


def test_gpu_snapshot_still_reports_foreign_processes():
    process = FakeProcess(78901)
    gpu_processes.register_owned_gpu_process(process)
    try:
        _sample, ownership = gpu_processes.snapshot_gpu_processes(
            lambda: {"pids": [process.pid, 99999], "memory_mib": 5000, "temperature_c": 65}
        )
        assert ownership["allowed_pids"] == [process.pid]
        assert ownership["unexpected_pids"] == [99999]
    finally:
        gpu_processes.unregister_owned_gpu_process(process)


def test_gpu_sample_allows_explicit_88_with_read_only_headroom(monkeypatch):
    commands = []

    def check_output(args, **_kwargs):
        commands.append(args)
        if "--query-compute-apps=pid" in args:
            return "12345\n"
        return "8000,84,Not Active,Not Active"

    monkeypatch.setattr(embedding_server.subprocess, "check_output", check_output)
    sample = embedding_server._gpu_sample(
        "GPU-test",
        max_gpu_temperature_c=88,
        temperature_limits={"max_operating_c": 93, "target_temperature_c": 84},
    )

    assert sample["temperature_c"] == 84
    assert sample["max_operating_c"] == 93
    assert sample["pids"] == {12345}
    assert all(args[0] == "nvidia-smi" for args in commands)
    assert all(
        not any(flag.startswith(("--power-limit", "--gpu-target", "--reset")) for flag in args)
        for args in commands
    )


def test_temperature_limit_reader_enforces_driver_reported_headroom(monkeypatch):
    commands = []

    def check_output(args, **_kwargs):
        commands.append(args)
        return "GPU Target Temperature : 84 C\nGPU Max Operating Temp : 93 C\n"

    monkeypatch.setattr(embedding_server.subprocess, "check_output", check_output)
    limits = embedding_server._temperature_limits("GPU-test", 88)

    assert limits == {"max_operating_c": 93, "target_temperature_c": 84}
    assert commands == [["nvidia-smi", "-q", "-i", "GPU-test", "-d", "TEMPERATURE"]]


@pytest.mark.parametrize(
    "temperature,limit,limits",
    [
        (89, 88, {"max_operating_c": 93, "target_temperature_c": 84}),
        (88, 88, {"max_operating_c": 92, "target_temperature_c": 84}),
    ],
)
def test_gpu_sample_stops_above_limit_or_without_five_degree_headroom(
    monkeypatch, temperature, limit, limits
):
    def check_output(args, **_kwargs):
        if "--query-compute-apps=pid" in args:
            return ""
        return f"8000,{temperature},Not Active,Not Active"

    monkeypatch.setattr(embedding_server.subprocess, "check_output", check_output)
    with pytest.raises(RetrievalError):
        embedding_server._gpu_sample(
            "GPU-test", max_gpu_temperature_c=limit, temperature_limits=limits
        )


def test_gpu_sample_requires_valid_thermal_status(monkeypatch):
    def check_output(args, **_kwargs):
        if "--query-compute-apps=pid" in args:
            return ""
        return "8000,84,Not Active,N/A"

    monkeypatch.setattr(embedding_server.subprocess, "check_output", check_output)
    with pytest.raises(RetrievalError, match="thermal telemetry unavailable"):
        embedding_server._gpu_sample(
            "GPU-test",
            max_gpu_temperature_c=88,
            temperature_limits={"max_operating_c": 93, "target_temperature_c": 84},
        )
