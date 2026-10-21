from __future__ import annotations

import subprocess

import pytest
from pydantic import ValidationError

import page_cite_rag.embedding_server as embedding_server
import page_cite_rag.llama as llama
from page_cite_rag import cli
from page_cite_rag.generation import ModelError
from test_llama import make_client

UUID = "GPU-" + "0" * 36
THERMAL = "GPU Max Operating Temp : 93 C\nGPU Target Temperature : 84 C\n"


def higher_client(tmp_path, monkeypatch):
    client, _, _ = make_client(tmp_path, monkeypatch)
    client.config = llama.LlamaConfig(
        bundle_dir=client.config.bundle_dir,
        device="cuda",
        gpu_uuid=UUID,
        max_gpu_temperature_c=88,
    )
    return client


def health_sample(**change):
    return {
        "memory_mib": 6361,
        "temperature_c": 84,
        "pids": [],
        "max_operating_c": 93,
        "target_temperature_c": 84,
        "hw_thermal_slowdown": False,
        "sw_thermal_slowdown": False,
        "power_draw_w": 250.5,
        "power_limit_w": 300.0,
        "fan_speed_percent": 80.0,
        "utilization_gpu_percent": 99.0,
        **change,
    }


@pytest.mark.parametrize("value", [89, 93, 100, 39, True, 88.0, "88"])
def test_temperature_policy_is_bounded_and_strict(value):
    with pytest.raises(ValidationError):
        llama.LlamaConfig(device="cuda", gpu_uuid=UUID, max_gpu_temperature_c=value)


def test_default_80_preserved_and_higher_cpu_rejected():
    assert llama.LlamaConfig().max_gpu_temperature_c == 80
    with pytest.raises(ValidationError):
        llama.LlamaConfig(max_gpu_temperature_c=88)


@pytest.mark.parametrize("temp", [81, 84, 88])
def test_normal_temperatures_allowed_and_telemetry_kept(tmp_path, monkeypatch, temp):
    client = higher_client(tmp_path, monkeypatch)
    monkeypatch.setattr(client, "_gpu_snapshot", lambda: health_sample(temperature_c=temp))
    with client:
        client._check_gpu(before=True)
        sample = client.diagnostics()["monitor_samples"][-1]
        assert sample["temperature_c"] == temp
        assert sample["power_draw_w"] == 250.5
        assert sample["fan_speed_percent"] == 80
        assert sample["hw_thermal_slowdown"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"temperature_c": 89},
        {"memory_mib": 12001},
        {"pids": [999]},
        {"hw_thermal_slowdown": True},
        {"sw_thermal_slowdown": True},
        {"hw_thermal_slowdown": None},
        {"max_operating_c": 90},
    ],
)
def test_higher_policy_still_stops_on_invalid_or_unsafe_state(tmp_path, monkeypatch, change):
    client = higher_client(tmp_path, monkeypatch)
    monkeypatch.setattr(client, "_gpu_snapshot", lambda: health_sample(**change))
    with client, pytest.raises(ModelError):
        client._check_gpu(before=True)


def stub_queries(monkeypatch, *, thermal=THERMAL, telemetry="80,250,300,99,Not Active,Not Active"):
    calls = []

    def query(args, **kwargs):
        calls.append(args)
        assert args[0] == "nvidia-smi" and "-i" in args and UUID in args
        assert kwargs["timeout"] == 3
        assert not any(arg.startswith(("--power-limit", "--gpu-target", "--reset")) for arg in args)
        if "-q" in args:
            return thermal
        if any(arg.startswith("--query-gpu=memory") for arg in args):
            return "6361,84"
        if any(arg.startswith("--query-compute") for arg in args):
            return ""
        return telemetry

    monkeypatch.setattr(llama.subprocess, "check_output", query)
    return calls


def test_read_only_limits_cached_health_fresh_each_sample(tmp_path, monkeypatch):
    client = higher_client(tmp_path, monkeypatch)
    calls = stub_queries(monkeypatch)
    with client:
        client._check_gpu(before=True)
        client._check_gpu(before=True)
    assert sum("-q" in args for args in calls) == 1
    assert len(calls) == 7


@pytest.mark.parametrize(
    "thermal,telemetry",
    [
        ("GPU Max Operating Temp : N/A\nGPU Target Temperature : 84 C", ""),
        (THERMAL + THERMAL, ""),
        (THERMAL.replace("93", "90"), ""),
        (THERMAL, "80,NaN,300,99,Not Active,Not Active"),
        (THERMAL, "80,250,300,99,N/A,Not Active"),
        (THERMAL, "80,250,300,99,Not Active"),
    ],
)
def test_missing_or_malformed_monitoring_fails_closed(tmp_path, monkeypatch, thermal, telemetry):
    client = higher_client(tmp_path, monkeypatch)
    stub_queries(monkeypatch, thermal=thermal, telemetry=telemetry)
    with client, pytest.raises(ModelError):
        client._check_gpu(before=True)


def test_monitor_timeout_is_not_ignored(tmp_path, monkeypatch):
    client = higher_client(tmp_path, monkeypatch)

    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 3)

    monkeypatch.setattr(llama.subprocess, "check_output", timed_out)
    with client, pytest.raises(ModelError):
        client._check_gpu(before=True)


@pytest.mark.parametrize(
    ("telemetry", "category"),
    [
        ("12001,40,Not Active,Not Active", "gpu_memory_limit"),
        ("6000,89,Not Active,Not Active", "gpu_temperature_limit"),
        ("6000,40,Active,Not Active", "gpu_thermal_slowdown"),
    ],
)
def test_embedding_guard_retains_the_triggering_safe_snapshot(monkeypatch, telemetry, category):
    responses = iter((telemetry, "12345\n"))
    monkeypatch.setattr(
        embedding_server.subprocess,
        "check_output",
        lambda *_args, **_kwargs: next(responses),
    )

    with pytest.raises(embedding_server._GpuGuardTriggered) as raised:
        embedding_server._gpu_sample(
            UUID,
            max_gpu_temperature_c=88,
            temperature_limits={"max_operating_c": 93, "target_temperature_c": 84},
        )

    assert raised.value.category == category
    assert raised.value.sample["pids"] == [12345]
    assert raised.value.sample["phase"] == "guard"


@pytest.mark.parametrize("backend,device", [("ollama", "cpu"), ("llama", "cpu")])
def test_cli_rejects_override_without_cuda(backend, device, capsys):
    args = cli.build_parser().parse_args(
        [
            "ask",
            "a" * 64,
            "Question?",
            "--backend",
            backend,
            "--device",
            device,
            "--gpu-temperature-limit",
            "88",
        ]
    )
    with pytest.raises(ValueError, match="requires explicit llama CUDA"):
        cli._generation_config(args)
    assert (
        cli.main(
            [
                "ask",
                "a" * 64,
                "Question?",
                "--backend",
                backend,
                "--device",
                device,
                "--gpu-temperature-limit",
                "88",
            ]
        )
        == 1
    )
    assert "invalid configuration or local resource" in capsys.readouterr().err
