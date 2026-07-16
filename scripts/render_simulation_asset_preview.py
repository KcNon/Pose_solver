#!/usr/bin/env python3
"""Render the exported display assembly and solved part coordinate axes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def camera_pose(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z_back = eye - target
    z_back /= np.linalg.norm(z_back)
    x_axis = np.cross(up, z_back)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_back, x_axis)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((x_axis, y_axis, z_back))
    pose[:3, 3] = eye
    return pose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=PROJECT_ROOT / "experiments/rice_cooker_simulation_assets",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    asset_root = args.asset_root.resolve()
    manifest = json.loads((asset_root / "manifest.json").read_text(encoding="utf-8"))
    output = (args.output or asset_root / "qa/assembly_preview_axes.png").resolve()

    scene = pyrender.Scene(bg_color=[0.94, 0.95, 0.97, 1.0], ambient_light=[0.45, 0.45, 0.45])
    for part, info in manifest["parts"].items():
        mesh_path = asset_root / info["visual_mesh"]
        loaded = trimesh.load(mesh_path, process=False)
        mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
        transform = np.asarray(manifest["assembled_T_body_from_part"][part], dtype=np.float64)
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=transform)
        axes = trimesh.creation.axis(
            transform=transform,
            origin_size=0.006,
            axis_radius=0.0018,
            axis_length=0.065,
        )
        scene.add(pyrender.Mesh.from_trimesh(axes, smooth=False))

    camera = pyrender.PerspectiveCamera(yfov=np.radians(38.0), aspectRatio=args.width / args.height)
    pose = camera_pose(
        eye=np.array([0.42, 0.28, 0.43]),
        target=np.array([0.0, 0.015, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    scene.add(camera, pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0), pose=pose)
    fill_pose = camera_pose(
        eye=np.array([-0.35, 0.25, 0.2]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    scene.add(pyrender.DirectionalLight(color=[0.8, 0.86, 1.0], intensity=2.0), pose=fill_pose)

    renderer = pyrender.OffscreenRenderer(args.width, args.height)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color[..., :3]).save(output)
    pixels = color[..., :3].astype(np.float32)
    report = {
        "image": str(output),
        "size_bytes": output.stat().st_size,
        "mean_rgb": float(pixels.mean()),
        "pixel_std": float(pixels.std()),
        "parts": list(manifest["parts"]),
        "axes": "Each axis is the solved canonical part frame: X red, Y green, Z blue.",
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
