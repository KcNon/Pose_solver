#!/usr/bin/env python3
"""Validate an assembly-oriented pose trajectory without modifying it."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.assembly_task import evaluate_assembly_task
from common.io_utils import load_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--connector-report", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    config = load_json(args.config)
    task = config.get("assembly_task")
    if not isinstance(task, dict) or not task.get("enabled", True):
        raise ValueError("pose config has no enabled assembly_task")
    connector = (
        load_json(args.connector_report)
        if args.connector_report is not None and args.connector_report.is_file()
        else None
    )
    report = evaluate_assembly_task(
        task,
        load_json(args.trajectory),
        trajectory_path=args.trajectory,
        connector_report=connector,
    )
    report["config"] = str(args.config.resolve())
    write_json(args.report, report)
    print(f"assembly task readiness -> {args.report}", flush=True)
    print(
        f"pose_product_ready={report['pose_product_ready']} | "
        f"physics_replay_ready={report['physics_replay_ready']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
