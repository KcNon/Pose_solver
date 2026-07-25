"""Compose independent part tracks into one indexed multi-part mask."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .io import (
    frame_path,
    load_binary_mask,
    load_label_mask,
    save_label_mask,
    track_path,
    write_json,
)
from .quality import summarize_area_series
from .schema import MaskPipelineConfig


def compose_frame(
    masks: Mapping[str, np.ndarray],
    config: MaskPipelineConfig,
    frame_number: int,
    shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    resolved: dict[str, np.ndarray] = {}
    occupied = np.zeros(shape, dtype=bool)
    part_map = config.part_map
    for name in config.occlusion_order:
        part = part_map[name]
        mask = np.asarray(masks.get(name, np.zeros(shape, bool)), dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"mask shape mismatch for {name}: {mask.shape} != {shape}")
        if frame_number < part.start_frame:
            mask = np.zeros(shape, dtype=bool)
        visible = mask & ~occupied
        resolved[name] = visible
        occupied |= visible
    label = np.zeros(shape, dtype=np.uint8)
    for part in config.parts:
        label[resolved[part.name]] = part.id
    return label, resolved


def compose_track_tree(
    config: MaskPipelineConfig,
    timestamps: list[str],
    *,
    tracks_root: Path | None = None,
    output_root: Path | None = None,
    views: list[str] | None = None,
    shape_source_root: Path | None = None,
) -> dict:
    from PIL import Image

    tracks_root = tracks_root or config.tracks_root
    output_root = output_root or config.masks_root
    selected_views = views or list(config.views)
    areas = {
        view: {part.name: [] for part in config.parts}
        for view in selected_views
    }
    for timestamp in timestamps:
        frame_number = int(timestamp)
        for view in selected_views:
            if shape_source_root is None:
                with Image.open(frame_path(config.frames_dir, view, timestamp)) as image:
                    shape = (image.height, image.width)
            else:
                shape = load_label_mask(
                    shape_source_root / timestamp / f"{view}.png"
                ).shape
            masks = {}
            for part in config.parts:
                path = track_path(tracks_root, part.name, timestamp, view)
                if frame_number >= part.start_frame and not path.exists():
                    raise FileNotFoundError(
                        f"active part track is missing: {path}"
                    )
                masks[part.name] = load_binary_mask(path, shape)
            label, resolved = compose_frame(masks, config, frame_number, shape)
            save_label_mask(output_root / timestamp / f"{view}.png", label, config.parts)
            for part in config.parts:
                areas[view][part.name].append(int(resolved[part.name].sum()))
    quality = {
        view: {
            part.name: summarize_area_series(
                timestamps,
                areas[view][part.name],
                start_frame=part.start_frame,
            )
            for part in config.parts
        }
        for view in selected_views
    }
    summary = {
        "method": "independent_part_tracks_then_front_to_back_composition",
        "views": selected_views,
        "parts": {
            part.name: {
                "id": part.id,
                "color": list(part.color),
                "start_frame": part.start_frame,
                "prompts": list(part.prompts),
            }
            for part in config.parts
        },
        "occlusion_order": list(config.occlusion_order),
        "frames_per_view": len(timestamps),
        "tracks_root": str(tracks_root),
        "masks_root": str(output_root),
        "quality": quality,
    }
    write_json(output_root.parent / "mask_manifest.json", summary)
    write_json(output_root.parent / "quality_report.json", quality)
    return summary
