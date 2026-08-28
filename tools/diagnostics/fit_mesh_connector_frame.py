#!/usr/bin/env python
"""Fit a cylindrical connector axis and origin directly from a mesh slab."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.connector_geometry import fit_cylindrical_axis_from_slab
from common.io_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument(
        "--selector-axis", required=True, choices=("x", "y", "z")
    )
    parser.add_argument("--minimum", required=True, type=float)
    parser.add_argument("--maximum", required=True, type=float)
    parser.add_argument("--direction-sign", required=True, type=float)
    parser.add_argument("--origin-coordinate", required=True, type=float)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--quantile", type=float, default=0.02)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    mesh_path = args.mesh.expanduser().resolve()
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.vertices):
        raise ValueError(f"expected a non-empty triangle mesh: {mesh_path}")
    report = fit_cylindrical_axis_from_slab(
        np.asarray(loaded.vertices, dtype=np.float64),
        selector_axis=args.selector_axis,
        minimum=args.minimum,
        maximum=args.maximum,
        direction_sign=args.direction_sign,
        origin_coordinate=args.origin_coordinate,
        bins=args.bins,
        quantile=args.quantile,
    )
    report["mesh"] = str(mesh_path)
    write_json(args.output, report)
    print(f"connector frame -> {args.output.expanduser().resolve()}")
    print(f"axis_part={report['axis_part']}")
    print(f"origin_raw={report['origin_raw']}")


if __name__ == "__main__":
    main()
