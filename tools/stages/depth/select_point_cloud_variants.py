#!/usr/bin/env python
"""Build a fail-closed per-part point-cloud root from several variants."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.io_utils import write_json


def parse_part_sources(values: list[str]) -> dict[str, Path]:
    """Parse repeated ``PART=ROOT`` arguments and reject ambiguous entries."""

    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected PART=ROOT, got {value!r}")
        part, root = value.split("=", 1)
        part = part.strip()
        if not part or not root.strip():
            raise ValueError(f"expected non-empty PART=ROOT, got {value!r}")
        if part in result:
            raise ValueError(f"duplicate point-cloud source for {part!r}")
        result[part] = Path(root).expanduser().resolve()
    if not result:
        raise ValueError("at least one --part-source is required")
    return result


def validate_selection(
    part_sources: dict[str, Path],
    frames: list[int],
    *,
    allow_missing: bool = False,
    required_frames_by_part: dict[str, set[int]] | None = None,
) -> tuple[list[dict], dict[str, list[int]]]:
    """Resolve every required artifact before creating any output links."""

    rows = []
    missing = []
    missing_frames = {part: [] for part in part_sources}
    required_frames_by_part = required_frames_by_part or {}
    for frame in frames:
        timestamp = f"{frame:06d}"
        for part, source_root in sorted(part_sources.items()):
            cloud = source_root / timestamp / f"{part}.ply"
            views = source_root / timestamp / "views" / part
            cloud_exists = cloud.is_file()
            views_exist = views.is_dir()
            if not cloud_exists:
                missing_frames[part].append(frame)
                required = frame in required_frames_by_part.get(part, set())
                if allow_missing and not required:
                    continue
                missing.append(str(cloud))
                continue
            if not views_exist:
                missing.append(str(views))
                continue
            rows.append({
                "timestamp": timestamp,
                "part": part,
                "cloud": cloud,
                "views": views,
            })
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... {len(missing) - 20} more"
        raise FileNotFoundError(
            f"point-cloud selection is incomplete ({len(missing)} missing):\n"
            f"{preview}{suffix}"
        )
    return rows, missing_frames


def parse_required_ranges(
    values: list[str],
    parts: set[str],
) -> dict[str, set[int]]:
    """Parse repeated ``PART=START:END`` requirements into exact frame sets."""

    result: dict[str, set[int]] = {}
    for value in values:
        if "=" not in value or ":" not in value:
            raise ValueError(f"expected PART=START:END, got {value!r}")
        part, bounds = value.split("=", 1)
        start_text, end_text = bounds.split(":", 1)
        if part not in parts:
            raise ValueError(f"required range references unknown part {part!r}")
        start, end = int(start_text), int(end_text)
        if end < start:
            raise ValueError(f"invalid required range {value!r}")
        result.setdefault(part, set()).update(range(start, end + 1))
    return result


def _link_exact(source: Path, destination: Path) -> None:
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        raise FileExistsError(
            f"existing link has a different target: {destination}"
        )
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing path: {destination}")
    destination.symlink_to(source, target_is_directory=source.is_dir())


def materialize_selection(
    output_root: Path,
    part_sources: dict[str, Path],
    frames: list[int],
    *,
    allow_missing: bool = False,
    required_frames_by_part: dict[str, set[int]] | None = None,
) -> dict:
    """Create deterministic links and return the reproducibility manifest."""

    rows, missing_frames = validate_selection(
        part_sources,
        frames,
        allow_missing=allow_missing,
        required_frames_by_part=required_frames_by_part,
    )
    output_root = output_root.resolve()
    for row in rows:
        frame_root = output_root / row["timestamp"]
        (frame_root / "views").mkdir(parents=True, exist_ok=True)
        _link_exact(row["cloud"], frame_root / f"{row['part']}.ply")
        _link_exact(row["views"], frame_root / "views" / row["part"])
    manifest = {
        "schema_version": 1,
        "method": "per_part_symbolic_link_selection",
        "output_root": str(output_root),
        "frame_start": min(frames),
        "frame_end": max(frames),
        "frame_count": len(frames),
        "parts": {
            part: str(source.resolve())
            for part, source in sorted(part_sources.items())
        },
        "artifact_count": len(rows),
        "allow_missing": bool(allow_missing),
        "missing_frames_by_part": missing_frames,
        "required_frames_by_part": {
            part: sorted(values)
            for part, values in sorted(
                (required_frames_by_part or {}).items()
            )
        },
    }
    write_json(output_root / "selection_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument(
        "--part-source",
        action="append",
        default=[],
        metavar="PART=ROOT",
        help="select one complete point-cloud variant for a part; repeatable",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "preserve quality-gate holes instead of fabricating clouds; "
            "required ranges still fail closed"
        ),
    )
    parser.add_argument(
        "--required-range",
        action="append",
        default=[],
        metavar="PART=START:END",
        help="frames that must exist even with --allow-missing; repeatable",
    )
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("--end must be greater than or equal to --start")
    try:
        sources = parse_part_sources(args.part_source)
        required = parse_required_ranges(
            args.required_range, set(sources)
        )
        manifest = materialize_selection(
            Path(args.output_root),
            sources,
            list(range(args.start, args.end + 1)),
            allow_missing=args.allow_missing,
            required_frames_by_part=required,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    print(
        f"selected {manifest['frame_count']} frames x "
        f"{len(manifest['parts'])} parts -> {manifest['output_root']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
