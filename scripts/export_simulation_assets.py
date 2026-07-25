#!/usr/bin/env python3
"""Export canonical meshes and URDF assets from a solved trajectory."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/simulation_assets.json",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    from common.simulation_export import export_simulation_assets

    export_simulation_assets(args.config, args.project_root, args.output_dir)


if __name__ == "__main__":
    main()
