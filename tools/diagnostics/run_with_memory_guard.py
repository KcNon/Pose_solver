#!/usr/bin/env python3
"""Run a command with fail-closed RAM and GPU-memory monitoring.

The guard watches the complete child process group, not only its root process.
If a configured limit is crossed, it interrupts the group and escalates to
SIGTERM/SIGKILL only when the command does not stop within the grace period.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


GIB = 1024**3
MIB = 1024**2


def read_meminfo(path: Path = Path("/proc/meminfo")) -> dict[str, int]:
    """Return selected Linux memory counters in bytes."""
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        fields = raw.strip().split()
        if not fields:
            continue
        multiplier = 1024 if len(fields) > 1 and fields[1] == "kB" else 1
        values[key] = int(fields[0]) * multiplier
    required = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
    missing = [key for key in required if key not in values]
    if missing:
        raise RuntimeError(f"missing /proc/meminfo fields: {missing}")
    return {key: values[key] for key in required}


def process_group_rss_bytes(process_group: int) -> tuple[int, list[int]]:
    """Sum resident memory for every process in a Linux process group."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    total_pages = 0
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            # comm may contain spaces and parentheses, so split after its
            # final closing parenthesis.  The remaining fields start at state.
            remainder = raw[raw.rfind(")") + 2 :].split()
            if int(remainder[2]) != process_group:
                continue
            rss_pages = max(0, int(remainder[21]))
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            IndexError,
            ValueError,
        ):
            continue
        pids.append(int(entry.name))
        total_pages += rss_pages
    return total_pages * page_size, sorted(pids)


def parse_gpu_rows(output: str) -> dict[int, dict[str, int]]:
    rows: dict[int, dict[str, int]] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        index, used, free, utilization = map(int, fields)
        rows[index] = {
            "memory_used_mib": used,
            "memory_free_mib": free,
            "utilization_percent": utilization,
        }
    return rows


def query_gpus() -> dict[int, dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = parse_gpu_rows(result.stdout)
    if not rows:
        raise RuntimeError("nvidia-smi returned no parseable GPU rows")
    return rows


def stop_process_group(pid: int, grace_seconds: float) -> None:
    """Interrupt a process group, then escalate if it remains alive."""
    for sig, timeout in (
        (signal.SIGINT, grace_seconds),
        (signal.SIGTERM, min(grace_seconds, 5.0)),
        (signal.SIGKILL, 2.0),
    ):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.2)


def _gib(value: int) -> float:
    return value / GIB


def memory_limit_reasons(
    meminfo: dict[str, int],
    process_group_rss_bytes: int,
    *,
    minimum_available_gib: float,
    maximum_process_rss_gib: float,
) -> list[str]:
    """Return deterministic host-memory limit failures for one sample."""

    reasons = []
    if _gib(meminfo["MemAvailable"]) < float(minimum_available_gib):
        reasons.append(
            f"system available RAM {_gib(meminfo['MemAvailable']):.2f}GiB"
            f" < {float(minimum_available_gib):.2f}GiB"
        )
    if _gib(process_group_rss_bytes) > float(maximum_process_rss_gib):
        reasons.append(
            f"process-group RSS {_gib(process_group_rss_bytes):.2f}GiB"
            f" > {float(maximum_process_rss_gib):.2f}GiB"
        )
    return reasons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--minimum-available-gib", type=float, default=100.0)
    parser.add_argument("--maximum-process-rss-gib", type=float, default=32.0)
    parser.add_argument("--minimum-gpu-free-mib", type=int)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--report-seconds", type=float, default=10.0)
    parser.add_argument("--stop-grace-seconds", type=float, default=2.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("a command is required after --")
    if args.poll_seconds <= 0 or args.report_seconds <= 0:
        raise SystemExit("poll/report intervals must be positive")
    if args.minimum_available_gib <= 0 or args.maximum_process_rss_gib <= 0:
        raise SystemExit("memory limits must be positive")
    if args.stop_grace_seconds < 0:
        raise SystemExit("stop grace cannot be negative")

    gpu_indices: list[int] = []
    child_env = os.environ.copy()
    child_env["POSE_SOLVER_MEMORY_GUARD_ACTIVE"] = "1"
    if args.cuda_visible_devices:
        gpu_indices = [
            int(value.strip())
            for value in args.cuda_visible_devices.split(",")
            if value.strip()
        ]
        if not gpu_indices or len(gpu_indices) != len(set(gpu_indices)):
            raise SystemExit("CUDA devices must be a non-empty unique list")
        child_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(value) for value in gpu_indices
        )
    if args.minimum_gpu_free_mib is not None and not gpu_indices:
        raise SystemExit(
            "--minimum-gpu-free-mib requires --cuda-visible-devices"
        )

    log_handle = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log.open("a", encoding="utf-8")

    def emit(event: dict[str, Any]) -> None:
        event = {"wall_time": time.time(), **event}
        if log_handle is not None:
            log_handle.write(json.dumps(event, sort_keys=True) + "\n")
            log_handle.flush()

    initial_mem = read_meminfo()
    initial_reasons = memory_limit_reasons(
        initial_mem,
        0,
        minimum_available_gib=args.minimum_available_gib,
        maximum_process_rss_gib=args.maximum_process_rss_gib,
    )
    if initial_reasons:
        reason = "; ".join(initial_reasons)
        print(
            f"[memory-guard] START REJECTED: {reason}",
            file=sys.stderr,
            flush=True,
        )
        emit({"event": "start_rejected", "reason": reason})
        if log_handle is not None:
            log_handle.close()
        return 125

    started = time.monotonic()
    process = subprocess.Popen(
        command,
        env=child_env,
        start_new_session=True,
    )
    previous_signal_handlers = {}

    def interrupt_guard(_signum, _frame) -> None:
        raise KeyboardInterrupt

    for handled_signal in (signal.SIGTERM, signal.SIGHUP):
        previous_signal_handlers[handled_signal] = signal.signal(
            handled_signal, interrupt_guard
        )
    emit(
        {
            "event": "start",
            "pid": process.pid,
            "command": command,
            "cuda_visible_devices": gpu_indices,
            "limits": {
                "minimum_available_gib": args.minimum_available_gib,
                "maximum_process_rss_gib": args.maximum_process_rss_gib,
                "minimum_gpu_free_mib": args.minimum_gpu_free_mib,
            },
        }
    )
    print(
        "[memory-guard] "
        f"pid={process.pid} CUDA={gpu_indices or 'CPU'} "
        f"min_available={args.minimum_available_gib:.1f}GiB "
        f"max_rss={args.maximum_process_rss_gib:.1f}GiB "
        f"min_gpu_free={args.minimum_gpu_free_mib}MiB",
        flush=True,
    )

    last_report = -float("inf")
    gpu_query_failures = 0
    guard_reason: str | None = None
    return_code: int | None = None
    try:
        while True:
            now = time.monotonic()
            mem = read_meminfo()
            rss, pids = process_group_rss_bytes(process.pid)
            gpu_rows: dict[int, dict[str, int]] = {}
            gpu_error = None
            # Host RAM is the fail-critical signal for this incident.  Do not
            # let a slow/hung nvidia-smi call stretch the one-second RAM poll.
            # GPU polling is enabled only when a GPU free-memory gate was
            # explicitly configured; CUDA visibility is still always fixed.
            if gpu_indices and args.minimum_gpu_free_mib is not None:
                try:
                    all_gpu_rows = query_gpus()
                    gpu_rows = {
                        index: all_gpu_rows[index]
                        for index in gpu_indices
                        if index in all_gpu_rows
                    }
                    missing = sorted(set(gpu_indices) - set(gpu_rows))
                    if missing:
                        raise RuntimeError(f"missing GPU indices: {missing}")
                    gpu_query_failures = 0
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    gpu_query_failures += 1
                    gpu_error = str(exc)

            sample = {
                "event": "sample",
                "elapsed_seconds": now - started,
                "process_group_rss_bytes": rss,
                "process_ids": pids,
                "mem_available_bytes": mem["MemAvailable"],
                "swap_used_bytes": mem["SwapTotal"] - mem["SwapFree"],
                "gpus": gpu_rows,
                "gpu_error": gpu_error,
            }
            emit(sample)

            reasons = memory_limit_reasons(
                mem,
                rss,
                minimum_available_gib=args.minimum_available_gib,
                maximum_process_rss_gib=args.maximum_process_rss_gib,
            )
            if args.minimum_gpu_free_mib is not None:
                for index, row in gpu_rows.items():
                    if row["memory_free_mib"] < args.minimum_gpu_free_mib:
                        reasons.append(
                            f"GPU {index} free {row['memory_free_mib']}MiB"
                            f" < {args.minimum_gpu_free_mib}MiB"
                        )
                if gpu_query_failures >= 3:
                    reasons.append(
                        "GPU monitoring failed three consecutive times: "
                        f"{gpu_error}"
                    )

            if now - last_report >= args.report_seconds or reasons:
                gpu_text = ", ".join(
                    f"{index}:free={row['memory_free_mib']}MiB"
                    for index, row in gpu_rows.items()
                ) or (
                    f"error={gpu_error}"
                    if gpu_error
                    else ("monitoring-disabled" if gpu_indices else "CPU")
                )
                print(
                    "[memory-guard] "
                    f"t={now-started:.1f}s rss={_gib(rss):.2f}GiB "
                    f"available={_gib(mem['MemAvailable']):.2f}GiB "
                    f"swap_used={_gib(mem['SwapTotal']-mem['SwapFree']):.2f}GiB "
                    f"gpu=({gpu_text})",
                    flush=True,
                )
                last_report = now

            return_code = process.poll()
            if reasons and return_code is None:
                guard_reason = "; ".join(reasons)
                print(
                    f"[memory-guard] LIMIT EXCEEDED: {guard_reason}; stopping",
                    file=sys.stderr,
                    flush=True,
                )
                emit({"event": "limit_exceeded", "reason": guard_reason})
                stop_process_group(process.pid, args.stop_grace_seconds)
                return_code = process.wait()
                break
            if return_code is not None:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        guard_reason = "monitor interrupted"
        print("[memory-guard] interrupted; stopping child group", flush=True)
        stop_process_group(process.pid, args.stop_grace_seconds)
        return_code = process.wait()
    finally:
        if process.poll() is None:
            # Fail closed on every unexpected monitor exception.  Without
            # this cleanup, a stage can outlive its monitor as an orphan and
            # continue allocating until the host becomes unresponsive.
            stop_process_group(process.pid, args.stop_grace_seconds)
            return_code = process.wait()
        for handled_signal, previous in previous_signal_handlers.items():
            signal.signal(handled_signal, previous)
        emit(
            {
                "event": "finish",
                "elapsed_seconds": time.monotonic() - started,
                "return_code": return_code,
                "guard_reason": guard_reason,
            }
        )
        if log_handle is not None:
            log_handle.close()

    if guard_reason == "monitor interrupted":
        return 130
    if guard_reason is not None:
        return 125
    return int(return_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
