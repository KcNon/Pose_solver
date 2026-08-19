#!/usr/bin/env python
"""Merge disjoint quality-cloud summaries produced by parallel frame shards."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json


def merge_summaries(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("at least one summary is required")
    summaries = [load_json(path) for path in paths]
    merged = dict(summaries[0])
    merged["frames"] = {}
    compatibility_keys = (
        "schema_version", "method", "depth_gauge", "output_root", "views",
        "parameters",
    )
    for path, summary in zip(paths, summaries):
        for key in compatibility_keys:
            if summary.get(key) != summaries[0].get(key):
                raise ValueError(f"{path}: incompatible {key}")
        overlap = set(merged["frames"]).intersection(summary.get("frames", {}))
        if overlap:
            raise ValueError(f"{path}: overlapping frames {sorted(overlap)[:3]}")
        merged["frames"].update(summary.get("frames", {}))
    merged["frames"] = {
        frame: merged["frames"][frame] for frame in sorted(merged["frames"])
    }
    merged["shards"] = [str(path.resolve()) for path in paths]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [Path(value) for value in args.summaries]
    merged = merge_summaries(paths)
    output = Path(args.output)
    write_json(output, merged)
    frames = list(merged["frames"])
    if frames:
        write_json(
            output.with_name("complete.json"),
            {"status": "complete", "frames": [int(frames[0]), int(frames[-1])]},
        )
    print(f"merged {len(paths)} shards / {len(frames)} frames -> {output}")


if __name__ == "__main__":
    main()
