"""
Windows 使用方式（PowerShell）：先安裝 uv，再 clone PageCiteRAG 專案並進入資料夾。
執行 `uv sync --locked --no-default-groups` 安裝本流程需要的套件；BM25、
pdfplumber 與 API 生成 transport 都是核心依賴，因此不需額外 extras。
自行安裝 Ollama 後執行 `ollama pull <模型>`，再使用例如：
`uv run python scripts/device_check.py --document C:\\資料\\paper.pdf
--provider ollama-local --model <模型>`。完成後請把輸出的 `device-check.json` 傳回，
以便比較這台電腦的實際處理時間與結果。使用免費 API 時請以 provider 選擇 gemini、
groq 或 openrouter；金鑰由 PageCiteRAG 原有環境變數設定讀取，不會寫入報告。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

DEFAULT_QUESTIONS = [
    "請指出文件中一項重要細節，並說明它如何支持作者的論點。",
    "這篇文件的主旨與主要貢獻是什麼？",
    "把其中最重要的概念詳細解釋給外行人聽。",
    "那它有什麼限制？",
]
PROVIDERS = ("ollama-local", "gemini", "groq", "openrouter")


def _cpu_model() -> str | None:
    if sys.platform == "win32":
        return platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER")
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.partition(":")[2].strip()
        except (OSError, UnicodeError):
            return None
    return platform.processor() or None


def _total_memory_bytes() -> int | None:
    if sys.platform == "win32":

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullTotalPhys)
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (OSError, UnicodeError, ValueError, IndexError):
            return None
    return None


def system_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "total_memory_bytes": _total_memory_bytes(),
    }


def _windows_process_memory(pid: int) -> int | None:
    if sys.platform != "win32":
        return None

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.windll.psapi
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)


def _run_command(
    command: list[str], *, input_text: str | None = None
) -> tuple[int, str, str, float, int | None, bool]:
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if input_text is not None:
        assert process.stdin is not None
        process.stdin.write(input_text)
        process.stdin.close()
        process.stdin = None

    peak_bytes: int | None = None
    memory_unavailable = False
    while process.poll() is None:
        if sys.platform == "win32":
            measured = _windows_process_memory(process.pid)
            if measured is None:
                memory_unavailable = True
            elif peak_bytes is None or measured > peak_bytes:
                peak_bytes = measured
        time.sleep(0.1)
    stdout, stderr = process.communicate()
    if sys.platform == "win32":
        measured = _windows_process_memory(process.pid)
        if measured is not None:
            peak_bytes = max(peak_bytes or 0, measured)
        else:
            memory_unavailable = True
    return (
        process.returncode,
        stdout,
        stderr,
        time.perf_counter() - started,
        peak_bytes,
        memory_unavailable,
    )


def _record_error(report: dict[str, Any], step: str, error: BaseException) -> None:
    report["errors"].append(
        {"step": step, "error_class": type(error).__name__, "message": str(error)[:300]}
    )


def _redact_api_keys(value: Any) -> Any:
    secrets = [
        os.environ.get(name, "")
        for name in (
            "PAGECITE_API_KEY",
            "GEMINI_API_KEY",
            "GROQ_API_KEY",
            "OPENROUTER_API_KEY",
        )
    ]
    secrets = [secret for secret in secrets if secret]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [_redact_api_keys(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_api_keys(item) for key, item in value.items()}
    return value


def _cli_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "page_cite_rag.cli", *args]


def _call_cli(
    report: dict[str, Any], step: str, command: list[str], input_text: str | None = None
) -> tuple[dict[str, Any] | None, tuple[float, int | None, bool]]:
    try:
        returncode, stdout, stderr, elapsed, memory, unavailable = _run_command(
            command, input_text=input_text
        )
        if returncode != 0:
            raise RuntimeError(f"CLI exited with status {returncode}")
        result = json.loads(stdout.strip().splitlines()[-1])
        return result, (elapsed, memory, unavailable)
    except Exception as exc:
        _record_error(report, step, exc)
        return None, (0.0, None, sys.platform == "win32")


def _collect_peak_memory(report: dict[str, Any]) -> None:
    if sys.platform.startswith("linux"):
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            report["peak_memory_bytes"] = int(usage.ru_maxrss * 1024)
        except (ImportError, OSError, ValueError) as exc:
            report["memory_note"] = f"Linux 子程序峰值記憶體無法讀取：{type(exc).__name__}"
    elif sys.platform == "win32" and report["peak_memory_bytes"] is None:
        report["memory_note"] = "Windows 子程序記憶體無法讀取；已依要求記錄為 null。"


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "system": None,
        "document": Path(args.document).name,
        "provider": args.provider,
        "model": args.model,
        "steps": [],
        "errors": [],
        "peak_memory_bytes": None,
        "memory_note": None,
    }
    try:
        report["system"] = system_info()
    except Exception as exc:
        _record_error(report, "system_info", exc)
        report["memory_note"] = "系統資訊讀取失敗。"
        return report
    try:
        questions = DEFAULT_QUESTIONS
        if args.questions is not None:
            loaded = json.loads(Path(args.questions).read_text(encoding="utf-8"))
            if not isinstance(loaded, list) or not all(isinstance(q, str) for q in loaded):
                raise ValueError("questions file must contain a JSON array of strings")
            questions = loaded
        with tempfile.TemporaryDirectory(prefix="page-cite-rag-device-check-") as temporary:
            database = Path(temporary) / "device-check.sqlite3"
            ingest_arguments = ["ingest", args.document, "--db", str(database)]
            if Path(args.document).suffix.casefold() == ".pdf":
                ingest_arguments.extend(("--parser", "pdfplumber"))
            ingest, timing = _call_cli(
                report,
                "ingest",
                _cli_command(*ingest_arguments),
            )
            report["steps"].append({"step": "ingest", "wall_seconds": timing[0]})
            if ingest is None:
                _collect_peak_memory(report)
                return report
            document_id = ingest["document"]["document_id"]
            indexed, timing = _call_cli(
                report,
                "index",
                _cli_command("index", document_id, "--method", "bm25", "--db", str(database)),
            )
            report["steps"].append({"step": "index", "wall_seconds": timing[0]})
            if indexed is None:
                _collect_peak_memory(report)
                return report

            previous_question: str | None = None
            for number, question in enumerate(questions):
                is_followup = number == 3 and previous_question is not None
                command = _cli_command(
                    "chat" if is_followup else "ask",
                    document_id,
                    *([] if is_followup else [question]),
                    "--mode",
                    "auto",
                    "--format",
                    "json",
                    "--backend",
                    "api",
                    "--api-provider",
                    args.provider,
                    "--api-model",
                    args.model,
                    "--method",
                    "bm25",
                    "--db",
                    str(database),
                )
                chat_input = f"{previous_question}\n{question}\n/exit\n" if is_followup else None
                answer_report, timing = _call_cli(
                    report, f"question_{number + 1}", command, input_text=chat_input
                )
                step: dict[str, Any] = {
                    "step": f"question_{number + 1}",
                    "question": question,
                    "wall_seconds": timing[0],
                }
                if answer_report is not None:
                    answer = answer_report.get("result", answer_report.get("answer", answer_report))
                    if not isinstance(answer, dict):
                        answer = answer_report
                    generation_stats = answer.get("generation_stats") or answer_report.get("usage")
                    claims = answer.get("answer_claims") or []
                    step.update(
                        {
                            "status": answer.get("status"),
                            "mode": answer.get("mode")
                            or answer_report.get("routing", {}).get("route"),
                            "tokens": generation_stats,
                            "has_citations": any(
                                isinstance(claim, dict) and bool(claim.get("citations"))
                                for claim in claims
                            ),
                            "answer": str(answer.get("answer", answer.get("explanation", "")))[
                                :500
                            ],
                        }
                    )
                report["steps"].append(step)
                previous_question = question
                if timing[1] is not None:
                    report["peak_memory_bytes"] = max(report["peak_memory_bytes"] or 0, timing[1])
                if timing[2]:
                    report["memory_note"] = (
                        "Windows 子程序記憶體無法讀取；peak_memory_bytes 僅包含成功取得的量測。"
                    )
    except Exception as exc:
        _record_error(report, "setup", exc)
    _collect_peak_memory(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure PageCiteRAG on this device.")
    parser.add_argument("--document", required=True, help="PDF, TXT, or Markdown document.")
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--questions", help="Optional UTF-8 JSON array of question strings.")
    parser.add_argument("--out", default="device-check.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = _redact_api_keys(run_check(args))
    destination = Path(args.out)
    try:
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"Could not write report: {type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr)
        return 1
    print(destination)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
