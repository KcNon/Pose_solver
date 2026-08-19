#!/usr/bin/env python3
"""Render a four-view review image from a Particulate eval prediction."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.mesh_render import SceneRenderer


PALETTE = np.asarray(
    [
        [255, 59, 48, 255],
        [52, 199, 89, 255],
        [0, 122, 255, 255],
        [255, 204, 0, 255],
        [175, 82, 222, 255],
        [255, 149, 0, 255],
    ],
    dtype=np.uint8,
)


def look_at_extrinsic(
    camera_position: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray,
) -> np.ndarray:
    forward = target - camera_position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        raise ValueError("Camera direction is parallel to world_up")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward), axis=0)
    extrinsic = np.zeros((3, 4), dtype=np.float64)
    extrinsic[:, :3] = rotation
    extrinsic[:, 3] = -rotation @ camera_position
    return extrinsic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    prediction_dir = args.prediction_dir.expanduser().resolve()
    prediction = np.load(prediction_dir / "eval" / "pred.npz")
    loaded = trimesh.load(prediction_dir / "eval" / "pred.obj", process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    labels = np.asarray(prediction["face_part_ids"], dtype=np.int64)
    if len(labels) != len(mesh.faces):
        raise ValueError("Prediction face labels do not match pred.obj")

    parts: list[tuple[trimesh.Trimesh, np.ndarray]] = []
    for part_id in np.unique(labels):
        part = mesh.submesh([labels == part_id], append=True, repair=False)
        part.visual.face_colors = np.tile(PALETTE[int(part_id) % len(PALETTE)], (len(part.faces), 1))
        parts.append((part, np.eye(4)))

    center = np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0)
    extent = float(np.max(mesh.extents))
    distance = 1.8 * extent
    offsets = (
        np.array([0.0, -distance, 0.15 * extent]),
        np.array([distance, 0.0, 0.15 * extent]),
        np.array([0.85 * distance, -0.85 * distance, 0.35 * extent]),
        np.array([-0.85 * distance, -0.85 * distance, 0.35 * extent]),
    )
    size = int(args.size)
    focal = 1.55 * size
    intrinsic = np.array(
        [[focal, 0.0, size / 2], [0.0, focal, size / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    images = []
    with SceneRenderer(size, size) as renderer:
        for offset in offsets:
            extrinsic = look_at_extrinsic(center + offset, center, np.array([0.0, 0.0, 1.0]))
            color, _ = renderer.render(parts, intrinsic, extrinsic)
            images.append(Image.fromarray(color))

    canvas = Image.new("RGB", (2 * size, 2 * size), color=(245, 245, 245))
    for index, image in enumerate(images):
        canvas.paste(image, ((index % 2) * size, (index // 2) * size))
    draw = ImageDraw.Draw(canvas)
    legend = "  ".join(f"part {int(i)}" for i in np.unique(labels))
    draw.rectangle((0, 0, 2 * size, 26), fill=(255, 255, 255))
    draw.text((8, 6), legend, fill=(0, 0, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
