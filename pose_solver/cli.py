"""Command-line interface for the single-source pose pipeline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from pose_solver.config import PipelineConfig, load_pipeline_config
from pose_solver.pipeline import PipelineRunner, VALID_STAGES, inspect_result


ROOT = Path(__file__).resolve().parents[1]
MEMORY_GUARD_ACTIVE = "POSE_SOLVER_MEMORY_GUARD_ACTIVE"


def memory_guard_command(
    config: PipelineConfig,
    arguments: Sequence[str],
) -> list[str]:
    """Build the fail-closed wrapper command for one unified pipeline run."""

    guard = config.memory_guard
    return [
        sys.executable,
        "-u",
        str(ROOT / "tools" / "diagnostics" / "run_with_memory_guard.py"),
        "--log",
        str(config.output_root / "runtime" / "memory_guard.jsonl"),
        "--cuda-visible-devices",
        ",".join(str(device) for device in config.devices),
        "--minimum-available-gib",
        str(guard.minimum_available_gib),
        "--maximum-process-rss-gib",
        str(guard.maximum_process_rss_gib),
        "--poll-seconds",
        str(guard.poll_seconds),
        "--report-seconds",
        str(guard.report_seconds),
        "--stop-grace-seconds",
        str(guard.stop_grace_seconds),
        "--",
        sys.executable,
        "-m",
        "pose_solver",
        *[str(value) for value in arguments],
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pose-solver",
        description="Reusable multi-view mask, depth, and part-pose pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run or resume the pipeline")
    run.add_argument("--config", required=True, help="single source JSON")
    run.add_argument("--stage", choices=VALID_STAGES, default="all")
    run.add_argument("--force", action="store_true")
    run.add_argument(
        "--skip-review",
        action="store_true",
        help="reuse/produce pose without the multi-view review artifact",
    )
    run.add_argument(
        "--skip-render",
        action="store_true",
        help="reuse/produce pose without regenerating the primary-view video",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and materialize configs without launching stages",
    )
    inspect = subparsers.add_parser(
        "inspect", help="summarize existing artifacts without modifying them"
    )
    inspect.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    original_arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(original_arguments)
    config = load_pipeline_config(Path(args.config))
    if args.command == "inspect":
        print(json.dumps(inspect_result(config), ensure_ascii=False, indent=2))
        return
    guard_required = bool(
        config.memory_guard.enabled
        and args.stage != "preflight"
        and not args.dry_run
        and os.environ.get(MEMORY_GUARD_ACTIVE) != "1"
    )
    if guard_required:
        command = memory_guard_command(config, original_arguments)
        environment = os.environ.copy()
        environment[MEMORY_GUARD_ACTIVE] = "1"
        print(
            "[pipeline] starting mandatory memory guard: "
            f"max_rss={config.memory_guard.maximum_process_rss_gib:.1f}GiB, "
            f"min_available={config.memory_guard.minimum_available_gib:.1f}GiB",
            flush=True,
        )
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=False,
        )
        if completed.returncode:
            raise SystemExit(completed.returncode)
        return
    PipelineRunner(
        config,
        force=args.force,
        dry_run=args.dry_run,
        skip_review=args.skip_review,
        skip_render=args.skip_render,
    ).run(args.stage)
