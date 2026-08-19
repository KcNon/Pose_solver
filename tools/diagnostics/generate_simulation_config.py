#!/usr/bin/env python
"""Generate an auditable Isaac config from a solved pose trajectory."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.simulation_autoconfig import generate_simulation_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--state-report", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--asset-output-dir", type=Path, required=True)
    parser.add_argument("--asset-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = generate_simulation_config(
        trajectory_path=args.trajectory,
        state_report=load_json(args.state_report),
        mesh_dir=args.mesh_dir,
        output_dir=args.asset_output_dir,
        asset_name=args.asset_name,
    )
    write_json(args.output, config)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
