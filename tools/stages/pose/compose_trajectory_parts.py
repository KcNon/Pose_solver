#!/usr/bin/env python3
"""Replace selected parts of a trajectory from compatible source trajectories."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.trajectory_io import refresh_trajectory_derived_fields


def _parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--part-source must be PART=TRAJECTORY, got {value!r}")
    part, path = value.split("=", 1)
    if not part or not path:
        raise ValueError(f"--part-source must be PART=TRAJECTORY, got {value!r}")
    return part, Path(path).expanduser().resolve()


def compose_trajectory_parts(base: dict, sources: dict[str, dict]) -> tuple[dict, dict]:
    """Return a copy of ``base`` with compatible per-frame part records replaced."""

    result = copy.deepcopy(base)
    base_parts = set(map(str, base["parts"]))
    unknown = sorted(set(sources).difference(base_parts))
    if unknown:
        raise ValueError(f"base trajectory lacks parts {unknown}")
    audit = {"method": "compatible_per_part_trajectory_composition", "parts": {}}
    for part, source in sources.items():
        if part not in source.get("parts", []):
            raise ValueError(f"source trajectory lacks part {part}")
        missing = sorted(set(base["frames"]).difference(source.get("frames", {})))
        if missing:
            raise ValueError(f"{part}: source lacks frame {missing[0]}")
        replaced = 0
        for timestamp in base["frames"]:
            source_parts = source["frames"][timestamp].get("parts", {})
            if part not in source_parts:
                raise ValueError(f"{part}: source frame {timestamp} lacks part record")
            record = copy.deepcopy(source_parts[part])
            record["source"] = str(record.get("source", "pose")) + "+composed_part"
            result["frames"][timestamp]["parts"][part] = record
            replaced += 1
        for key in ("scales", "raw_mesh_origins"):
            if part in source.get(key, {}):
                result.setdefault(key, {})[part] = copy.deepcopy(source[key][part])
        audit["parts"][part] = {"replaced_frames": replaced}
    refresh_trajectory_derived_fields(result)
    return result, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--part-source", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    pairs = [_parse_source(value) for value in args.part_source]
    if len(dict(pairs)) != len(pairs):
        raise ValueError("duplicate part in --part-source")
    source_paths = dict(pairs)
    result, audit = compose_trajectory_parts(
        load_json(args.base.expanduser().resolve()),
        {part: load_json(path) for part, path in source_paths.items()},
    )
    audit.update({
        "base": str(args.base.expanduser().resolve()),
        "sources": {part: str(path) for part, path in source_paths.items()},
        "output": str(args.output.expanduser().resolve()),
    })
    write_json(args.output, result)
    if args.report is not None:
        write_json(args.report, audit)
    print(f"composed trajectory -> {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
