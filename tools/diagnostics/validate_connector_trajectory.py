#!/usr/bin/env python
"""Evaluate connector-level insertion readiness for a solved trajectory."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.connector_geometry import evaluate_connectors
from common.io_utils import load_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    config = load_json(args.config)
    connectors = dict(config.get("connectors", {}))
    if not connectors:
        raise ValueError("pose config has no connectors")
    report = evaluate_connectors(connectors, load_json(args.trajectory))
    report["config"] = str(args.config.resolve())
    report["trajectory"] = str(args.trajectory.resolve())
    write_json(args.report, report)
    print(f"connector readiness -> {args.report}", flush=True)
    print(f"simulation_ready={report['simulation_ready']}", flush=True)


if __name__ == "__main__":
    main()
