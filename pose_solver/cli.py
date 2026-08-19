"""Command-line interface for the single-source pose pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pose_solver.config import load_pipeline_config
from pose_solver.pipeline import PipelineRunner, VALID_STAGES, inspect_result


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
    args = build_parser().parse_args(argv)
    config = load_pipeline_config(Path(args.config))
    if args.command == "inspect":
        print(json.dumps(inspect_result(config), ensure_ascii=False, indent=2))
        return
    PipelineRunner(
        config,
        force=args.force,
        dry_run=args.dry_run,
    ).run(args.stage)
