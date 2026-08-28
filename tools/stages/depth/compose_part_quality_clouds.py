#!/usr/bin/env python3
"""Compose one quality-cloud artifact from independently selected part roots."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json


def _parse_part_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--part-source must be PART=ROOT, got {value!r}")
    part, root = value.split("=", 1)
    if not part or not root:
        raise ValueError(f"--part-source must be PART=ROOT, got {value!r}")
    return part, Path(root).expanduser().resolve()


def _link_or_copy(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    try:
        os.link(source, output)
    except OSError:
        shutil.copy2(source, output)


def compose_part_roots(
    sources: dict[str, Path],
    output: Path,
    *,
    frame_start: int,
    frame_end: int,
) -> dict:
    """Compose per-part PLYs and quality rows over an explicit frame range."""

    if not sources:
        raise ValueError("at least one part source is required")
    summaries = {
        part: load_json(root / "quality_cloud_summary.json")
        for part, root in sources.items()
    }
    view_sets = {tuple(summary["views"]) for summary in summaries.values()}
    if len(view_sets) != 1:
        raise ValueError("quality-cloud sources must use the same ordered views")
    views = list(next(iter(view_sets)))
    frames: dict[str, dict] = {}
    copied = {part: 0 for part in sources}
    missing = {part: [] for part in sources}
    for frame in range(frame_start, frame_end + 1):
        timestamp = f"{frame:06d}"
        frame_rows = {}
        for part, root in sources.items():
            source_frames = summaries[part].get("frames", {})
            if timestamp not in source_frames or part not in source_frames[timestamp]:
                raise ValueError(f"{part}: summary lacks frame {timestamp}")
            frame_rows[part] = source_frames[timestamp][part]
            source_ply = root / timestamp / f"{part}.ply"
            if source_ply.exists():
                _link_or_copy(
                    source_ply, output / timestamp / f"{part}.ply"
                )
                copied[part] += 1
            else:
                missing[part].append(frame)
            source_views = root / timestamp / "views" / part
            if source_views.exists():
                for source_view in source_views.glob("*.ply"):
                    _link_or_copy(
                        source_view,
                        output / timestamp / "views" / part / source_view.name,
                    )
        frames[timestamp] = frame_rows

    report = {
        "schema_version": 2,
        "method": "per_part_quality_cloud_composition",
        "output_root": str(output),
        "views": views,
        "parts": list(sources),
        "sources": {part: str(root) for part, root in sources.items()},
        "parameters_by_part": {
            part: summary.get("parameters", {})
            for part, summary in summaries.items()
        },
        "copied_frame_clouds": copied,
        "missing_frame_clouds": missing,
        "frames": frames,
    }
    write_json(output / "quality_cloud_summary.json", report)
    write_json(output / "complete.json", {
        "method": report["method"],
        "output_root": str(output),
        "frame_range": [frame_start, frame_end],
        "parts": list(sources),
        "copied_frame_clouds": copied,
        "missing_frame_clouds": missing,
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part-source", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-start", required=True, type=int)
    parser.add_argument("--frame-end", required=True, type=int)
    args = parser.parse_args()
    if args.frame_start < 0 or args.frame_end < args.frame_start:
        raise ValueError("invalid frame range")
    pairs = [_parse_part_source(value) for value in args.part_source]
    sources = dict(pairs)
    if len(sources) != len(pairs):
        raise ValueError("duplicate part in --part-source")
    output = args.output.expanduser().resolve()
    report = compose_part_roots(
        sources,
        output,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
    )
    print(f"composed quality clouds -> {output}")
    print(f"frame clouds -> {report['copied_frame_clouds']}")


if __name__ == "__main__":
    main()
