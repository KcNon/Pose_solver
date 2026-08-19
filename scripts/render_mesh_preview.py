#!/usr/bin/env python3
"""Render a tightly cropped reference preview for a mesh asset."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.mesh_render import SceneRenderer


def look_at_extrinsic(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward), axis=0)
    extrinsic = np.zeros((3, 4), dtype=np.float64)
    extrinsic[:, :3] = rotation
    extrinsic[:, 3] = -rotation @ position
    return extrinsic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    mesh_path = args.mesh.expanduser().resolve()
    loaded = trimesh.load(mesh_path, process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Expected a non-empty triangle mesh: {mesh_path}")
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else mesh_path.parent / "preview.jpg"
    )

    render_size = max(512, int(args.size))
    center = np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0)
    extent = max(float(np.max(mesh.extents)), 1e-6)
    camera_position = center + np.array([0.65 * extent, -2.0 * extent, 0.25 * extent])
    focal = 1.6 * render_size
    intrinsic = np.array(
        [
            [focal, 0.0, render_size / 2],
            [0.0, focal, render_size / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    extrinsic = look_at_extrinsic(camera_position, center)
    with SceneRenderer(render_size, render_size) as renderer:
        color, depth = renderer.render([(mesh, np.eye(4))], intrinsic, extrinsic)

    visible = depth > 0
    if not np.any(visible):
        raise RuntimeError(f"Preview render is empty: {mesh_path}")
    rows, cols = np.nonzero(visible)
    padding = max(8, int(0.08 * max(rows.ptp(), cols.ptp())))
    y0 = max(0, int(rows.min()) - padding)
    y1 = min(render_size, int(rows.max()) + padding + 1)
    x0 = max(0, int(cols.min()) - padding)
    x1 = min(render_size, int(cols.max()) + padding + 1)
    color[~visible] = 245
    crop = Image.fromarray(color[y0:y1, x0:x1])
    target_size = int(args.size)
    crop.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), color=(245, 245, 245))
    canvas.paste(crop, ((target_size - crop.width) // 2, (target_size - crop.height) // 2))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    print(output_path)


if __name__ == "__main__":
    main()
