"""Bounded normalization helpers for externally produced reusable inputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from common.io_utils import write_json


VALID_MASK_LAYOUTS = {"frame_first", "view_first"}
MAX_NORMALIZED_MASK_LINKS = 100_000


def normalize_view_first_masks(
    source_root: Path,
    output_root: Path,
    *,
    views: Iterable[str],
    frame_start: int,
    frame_end: int,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Expose ``view/frame.png`` masks as ``frame/view.png`` symlinks.

    Only the configured closed frame range and views are touched.  The hard
    link-count ceiling keeps an accidental broad configuration from creating
    an unbounded output tree.
    """

    source_root = Path(source_root)
    output_root = Path(output_root)
    ordered_views = [str(view) for view in views]
    frame_count = int(frame_end) - int(frame_start) + 1
    link_count = frame_count * len(ordered_views)
    if frame_count <= 0 or not ordered_views:
        raise ValueError("mask normalization requires frames and views")
    if link_count > MAX_NORMALIZED_MASK_LINKS:
        raise ValueError(
            f"refusing to normalize {link_count} masks; hard limit is "
            f"{MAX_NORMALIZED_MASK_LINKS}"
        )

    sources: list[tuple[Path, Path]] = []
    for frame in range(int(frame_start), int(frame_end) + 1):
        timestamp = f"{frame:06d}"
        for view in ordered_views:
            source = source_root / view / f"{timestamp}.png"
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = output_root / timestamp / f"{view}.png"
            sources.append((source, destination))

    created = reused = replaced = 0
    if not dry_run:
        for source, destination in sources:
            resolved_source = source.resolve()
            if destination.is_symlink():
                if destination.resolve() == resolved_source:
                    reused += 1
                    continue
                if not force:
                    raise FileExistsError(
                        f"mask link points elsewhere: {destination}"
                    )
                destination.unlink()
                replaced += 1
            elif destination.exists():
                if not force:
                    raise FileExistsError(
                        f"refusing to replace existing mask: {destination}"
                    )
                destination.unlink()
                replaced += 1
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(resolved_source)
            created += 1

        write_json(
            output_root.parent / "normalized_reuse_masks.json",
            {
                "source_layout": "view_first",
                "output_layout": "frame_first",
                "source_root": str(source_root.resolve()),
                "output_root": str(output_root.resolve()),
                "frame_range": [int(frame_start), int(frame_end)],
                "views": ordered_views,
                "links": link_count,
                "created": created,
                "reused": reused,
                "replaced": replaced,
            },
        )

    return {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "links": link_count,
        "created": created,
        "reused": reused,
        "replaced": replaced,
        "dry_run": bool(dry_run),
    }
