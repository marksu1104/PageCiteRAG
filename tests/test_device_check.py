from __future__ import annotations

import scripts.device_check as device_check


def test_system_info_reads_linux_cpu_and_memory_sources(monkeypatch):
    monkeypatch.setattr(device_check.sys, "platform", "linux")
    monkeypatch.setattr(device_check.platform, "platform", lambda: "Linux test")
    monkeypatch.setattr(device_check.platform, "python_version", lambda: "3.11.test")
    monkeypatch.setattr(device_check.os, "cpu_count", lambda: 12)

    def fake_read_text(path, *, encoding=None, **_kwargs):
        assert encoding == "utf-8"
        if str(path) == "/proc/cpuinfo":
            return "processor: 0\nmodel name: Test CPU 5600\n"
        if str(path) == "/proc/meminfo":
            return "MemTotal:       16384000 kB\n"
        raise AssertionError(path)

    monkeypatch.setattr(device_check.Path, "read_text", fake_read_text)
    assert device_check.system_info() == {
        "platform": "Linux test",
        "python_version": "3.11.test",
        "cpu_model": "Test CPU 5600",
        "logical_cores": 12,
        "total_memory_bytes": 16384000 * 1024,
    }


def test_cli_failure_is_recorded_without_raising(monkeypatch):
    report = {"errors": []}

    def failed_run(_command, *, input_text=None):
        raise OSError("synthetic subprocess failure")

    monkeypatch.setattr(device_check, "_run_command", failed_run)
    result, timing = device_check._call_cli(report, "ingest", ["python", "cli"])

    assert result is None
    assert timing[0] == 0
    assert report["errors"] == [
        {
            "step": "ingest",
            "error_class": "OSError",
            "message": "synthetic subprocess failure",
        }
    ]


def test_report_error_message_is_limited_to_300_characters():
    report = {"errors": []}
    device_check._record_error(report, "setup", ValueError("x" * 400))
    assert len(report["errors"][0]["message"]) == 300


def test_report_redacts_provider_api_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key")
    result = device_check._redact_api_keys({"answer": "value secret-test-key hidden"})
    assert result == {"answer": "value [REDACTED] hidden"}
