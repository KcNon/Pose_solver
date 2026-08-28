#!/usr/bin/env python3
"""Derive rigid-only mask and mesh inputs without changing source artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image
import trimesh


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.rigid_observation import halfspace_face_mask, thick_core_region


def _resolved_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _save_label_mask(path: Path, labels: np.ndarray, source: Image.Image) -> None:
    output = Image.fromarray(np.asarray(labels, dtype=np.uint8), mode="P")
    palette = source.getpalette()
    if palette is not None:
        output.putpalette(palette)
    if "transparency" in source.info:
        output.info["transparency"] = source.info["transparency"]
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path)


def _derive_masks(
    source_root: Path,
    output_root: Path,
    *,
    views: list[str],
    frame_start: int,
    frame_end: int,
    part_id: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    for frame in range(frame_start, frame_end + 1):
        timestamp = f"{frame:06d}"
        for view in views:
            source_path = source_root / timestamp / f"{view}.png"
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            with Image.open(source_path) as image:
                labels = np.asarray(image).copy()
                selected, stats = thick_core_region(
                    labels == part_id,
                    erosion_radius=int(settings["erosion_radius_pixels"]),
                    restore_radius=int(settings["restore_radius_pixels"]),
                    minimum_core_pixels=int(
                        settings.get("minimum_core_pixels", 1)
                    ),
                )
                labels[labels == part_id] = 0
                labels[selected] = part_id
                _save_label_mask(
                    output_root / timestamp / f"{view}.png", labels, image
                )
            observations.append({
                "frame": frame,
                "view": view,
                **stats,
            })

    retained = [
        row["retained_fraction"]
        for row in observations
        if row["source_pixels"] > 0
    ]
    return {
        "observations": len(observations),
        "source_visible_observations": sum(
            row["source_pixels"] > 0 for row in observations
        ),
        "rigid_visible_observations": sum(
            row["output_pixels"] > 0 for row in observations
        ),
        "no_rigid_core": [
            {"frame": row["frame"], "view": row["view"]}
            for row in observations
            if row["source_pixels"] > 0 and row["output_pixels"] == 0
        ],
        "retained_fraction": {
            "minimum": float(min(retained)) if retained else None,
            "median": float(np.median(retained)) if retained else None,
            "maximum": float(max(retained)) if retained else None,
        },
        "per_observation": observations,
    }


def _derive_meshes(
    source_root: Path,
    output_root: Path,
    *,
    part: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_root.glob("*.glb")):
        if source.stem != part:
            shutil.copy2(source, output_root / source.name)

    source_path = source_root / f"{part}.glb"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    loaded = trimesh.load(source_path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise ValueError(f"expected a non-empty triangle mesh: {source_path}")
    source_faces = int(len(loaded.faces))
    source_vertices = int(len(loaded.vertices))
    source_bounds = np.asarray(loaded.bounds, dtype=float).tolist()
    keep = halfspace_face_mask(
        loaded.vertices,
        loaded.faces,
        axis=str(settings["axis"]),
        minimum=(
            float(settings["minimum"])
            if settings.get("minimum") is not None
            else None
        ),
        maximum=(
            float(settings["maximum"])
            if settings.get("maximum") is not None
            else None
        ),
    )
    if not np.any(keep):
        raise ValueError("mesh halfspace rejected every face")
    loaded.update_faces(keep)
    loaded.remove_unreferenced_vertices()
    output_path = output_root / f"{part}.glb"
    loaded.export(output_path)
    return {
        "source": str(source_path),
        "output": str(output_path),
        "axis": settings["axis"],
        "minimum": settings.get("minimum"),
        "maximum": settings.get("maximum"),
        "source_vertices": source_vertices,
        "output_vertices": int(len(loaded.vertices)),
        "source_faces": source_faces,
        "output_faces": int(len(loaded.faces)),
        "retained_face_fraction": float(len(loaded.faces) / source_faces),
        "source_bounds": source_bounds,
        "output_bounds": np.asarray(loaded.bounds, dtype=float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_json(config_path)
    base = config_path.parent
    output_root = _resolved_path(base, config["output_root"])
    report_path = output_root / "diagnostics" / "rigid_pose_inputs.json"
    if report_path.exists() and not args.force:
        print(f"[resume] {report_path}")
        return

    part = config["part"]
    frame_start, frame_end = map(int, config["frame_range"])
    if frame_start < 0 or frame_end < frame_start:
        raise ValueError("frame_range must be [start, end]")
    views = [str(view) for view in config["views"]]
    if not views or len(views) != len(set(views)):
        raise ValueError("views must be non-empty and unique")

    mask_report = _derive_masks(
        _resolved_path(base, config["source_masks_dir"]),
        output_root / "masks",
        views=views,
        frame_start=frame_start,
        frame_end=frame_end,
        part_id=int(part["id"]),
        settings=config["mask"],
    )
    mesh_report = _derive_meshes(
        _resolved_path(base, config["source_mesh_dir"]),
        output_root / "meshes",
        part=str(part["name"]),
        settings=config["mesh"],
    )
    report = {
        "method": "thick_mask_core_and_mesh_halfspace",
        "config": str(config_path),
        "output_root": str(output_root),
        "part": part,
        "frame_range": [frame_start, frame_end],
        "views": views,
        "mask_settings": config["mask"],
        "mesh_settings": config["mesh"],
        "mask": mask_report,
        "mesh": mesh_report,
    }
    write_json(report_path, report)
    print(f"rigid masks -> {output_root / 'masks'}")
    print(f"rigid meshes -> {output_root / 'meshes'}")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
