"""Compatibility helpers for old palette-mask branch layouts.

New code should write one binary track per part.  These helpers only exist so
older temporal/body/repair outputs can be imported into that representation
and composed by the same canonical implementation.
"""
from __future__ import annotations

from pathlib import Path
import shutil
from typing import Callable, Iterable

import numpy as np

from .io import (
    frame_path,
    load_label_mask,
    save_binary_mask,
    track_path,
)
from .schema import MaskPipelineConfig, PartSpec


LEGACY_PARTS = (
    PartSpec("lid", 1, (255, 59, 48), 0, ("lid",)),
    PartSpec("body", 2, (52, 199, 89), 0, ("body",)),
    PartSpec("inner_pot", 3, (0, 122, 255), 0, ("inner pot",)),
)


def timestamp_directories(root: Path) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(
        (
            path.name for path in root.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=int,
    )


def parse_part_starts(values: Iterable[str]) -> dict[str, int]:
    valid = {part.name for part in LEGACY_PARTS}
    result: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected PART=FRAME, got {value!r}")
        part, frame = value.split("=", 1)
        if part not in valid:
            raise ValueError(f"unknown part in visibility start: {part}")
        result[part] = int(frame)
    return result


def make_legacy_config(
    *,
    frames_dir: Path,
    output_root: Path,
    views: Iterable[str],
    part_starts: dict[str, int] | None = None,
) -> MaskPipelineConfig:
    starts = part_starts or {}
    parts = tuple(
        PartSpec(
            part.name,
            part.id,
            part.color,
            int(starts.get(part.name, 0)),
            part.prompts,
        )
        for part in LEGACY_PARTS
    )
    return MaskPipelineConfig(
        source_path=output_root / "legacy_import.json",
        raw={"compatibility_import": True},
        frames_dir=frames_dir,
        work_root=output_root / "_legacy_tracks",
        output_root=output_root,
        views=tuple(views),
        parts=parts,
        occlusion_order=("lid", "inner_pot", "body"),
    )


def import_palette_parts(
    source_root: Path,
    config: MaskPipelineConfig,
    timestamps: Iterable[str],
    part_names: Iterable[str],
    *,
    views: Iterable[str] | None = None,
    select: Callable[[str, str], bool] | None = None,
) -> None:
    """Extract selected labels from an old palette tree into binary tracks."""
    selected_views = list(views or config.views)
    for timestamp in timestamps:
        for view in selected_views:
            if select is not None and not select(timestamp, view):
                continue
            source = source_root / timestamp / f"{view}.png"
            label = load_label_mask(source)
            for part_name in part_names:
                part = config.part_map[part_name]
                save_binary_mask(
                    track_path(config.tracks_root, part_name, timestamp, view),
                    label == part.id,
                )


def copy_bbox_metadata(source_root: Path, destination_root: Path) -> None:
    source = source_root / "bbox.json"
    if source.exists():
        destination_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination_root / "bbox.json")


def write_review_artifacts(
    config: MaskPipelineConfig,
    timestamps: list[str],
    *,
    fps: float = 10.0,
) -> None:
    """Write per-view overlay videos and compact contact sheets."""
    import cv2

    preview_root = config.output_root / "preview"
    preview_root.mkdir(parents=True, exist_ok=True)
    for view in config.views:
        writer = None
        cells: list[np.ndarray] = []
        for index, timestamp in enumerate(timestamps):
            image = cv2.imread(str(frame_path(config.frames_dir, view, timestamp)))
            if image is None:
                raise RuntimeError(f"failed to read frame {timestamp}/{view}")
            label = load_label_mask(config.masks_root / timestamp / f"{view}.png")
            overlay = image.copy()
            for part in config.parts:
                selected = label == part.id
                if not selected.any():
                    continue
                bgr = np.asarray(part.color[::-1], dtype=np.float32)
                overlay[selected] = (
                    overlay[selected].astype(np.float32) * 0.52 + bgr * 0.48
                ).astype(np.uint8)
            if writer is None:
                height, width = overlay.shape[:2]
                writer = cv2.VideoWriter(
                    str(preview_root / f"{view}_overlay.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"failed to create review video for {view}")
            writer.write(overlay)
            cell = cv2.resize(overlay, (240, 135), interpolation=cv2.INTER_AREA)
            cv2.putText(
                cell,
                f"{timestamp} ({index + 1}/{len(timestamps)})",
                (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cells.append(cell)
        if writer is not None:
            writer.release()
        if not cells:
            continue
        columns = min(10, len(cells))
        blank = np.zeros_like(cells[0])
        cells.extend([blank] * ((columns - len(cells) % columns) % columns))
        sheet = np.vstack([
            np.hstack(cells[offset:offset + columns])
            for offset in range(0, len(cells), columns)
        ])
        cv2.imwrite(
            str(preview_root / f"{view}_contact_sheet.jpg"),
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )
