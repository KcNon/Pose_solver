#!/usr/bin/env python
"""Resample a per-frame depth gauge onto another timestamp grid."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-count", required=True, type=int)
    parser.add_argument("--source-fps", required=True, type=float)
    parser.add_argument("--target-fps", required=True, type=float)
    parser.add_argument(
        "--target-start-seconds",
        required=True,
        type=float,
        help="target frame zero expressed on the source timeline",
    )
    args = parser.parse_args()
    if args.target_count <= 0 or args.source_fps <= 0 or args.target_fps <= 0:
        raise ValueError("counts and frame rates must be positive")

    source = load_json(Path(args.input))
    source_items = sorted(
        (int(key), value) for key, value in source["frames"].items()
    )
    source_ids = np.asarray([item[0] for item in source_items], dtype=np.float64)
    shifts = np.asarray(
        [item[1]["shift_m"] for item in source_items], dtype=np.float64
    )
    target_seconds = (
        args.target_start_seconds
        + np.arange(args.target_count, dtype=np.float64) / args.target_fps
    )
    source_positions = target_seconds * args.source_fps
    if source_positions.min() < source_ids.min() or source_positions.max() > source_ids.max():
        raise ValueError(
            "target grid falls outside source gauge: "
            f"{source_positions.min():.3f}..{source_positions.max():.3f} vs "
            f"{source_ids.min():.3f}..{source_ids.max():.3f}"
        )

    target_shifts = np.stack([
        np.interp(source_positions, source_ids, shifts[:, view])
        for view in range(shifts.shape[1])
    ], axis=1)
    result = deepcopy(source)
    result["frames"] = {
        f"{index:06d}": {
            "shift_m": [float(value) for value in target_shifts[index]],
            "n_pixels": [0] * shifts.shape[1],
            "interpolated": [True] * shifts.shape[1],
        }
        for index in range(args.target_count)
    }
    result["resampling"] = {
        "source": str(Path(args.input).resolve()),
        "source_fps": float(args.source_fps),
        "target_fps": float(args.target_fps),
        "target_start_seconds": float(args.target_start_seconds),
        "target_count": int(args.target_count),
        "source_position_range": [
            float(source_positions.min()),
            float(source_positions.max()),
        ],
    }
    write_json(Path(args.output), result)


if __name__ == "__main__":
    main()
