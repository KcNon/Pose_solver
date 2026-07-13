#!/usr/bin/env python
"""Render a fused trajectory as overlay, mesh-only, and mesh+XYZ-axis videos."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.mesh_render import SceneRenderer
from common.normalized_recon import load_recon, scale_intrinsics


COLORS_RGB = {
    "body": (75, 190, 95),
    "inner_pot": (245, 135, 40),
    "lid": (85, 145, 245),
}
AXIS_COLORS_RGB = {
    "x": (255, 40, 40),
    "y": (40, 235, 70),
    "z": (50, 100, 255),
}


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def solid_mesh(mesh: trimesh.Trimesh, rgb: tuple[int, int, int]) -> trimesh.Trimesh:
    result = mesh.copy()
    rgba = np.asarray([*rgb, 255], dtype=np.uint8)
    result.visual = trimesh.visual.ColorVisuals(
        mesh=result, face_colors=np.tile(rgba, (len(result.faces), 1)))
    return result


def colored_primitive(mesh: trimesh.Trimesh, rgb: tuple[int, int, int]) -> trimesh.Trimesh:
    rgba = np.asarray([*rgb, 255], dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, face_colors=np.tile(rgba, (len(mesh.faces), 1)))
    return mesh


def make_axis_meshes(length: float, radius: float) -> list[trimesh.Trimesh]:
    values = []
    directions = {
        "x": np.asarray([1.0, 0.0, 0.0]),
        "y": np.asarray([0.0, 1.0, 0.0]),
        "z": np.asarray([0.0, 0.0, 1.0]),
    }
    shaft_length = 0.78 * length
    head_length = 0.22 * length
    for name, direction in directions.items():
        align = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)
        shaft_tf = np.asarray(align, float)
        shaft_tf[:3, 3] = direction * (shaft_length / 2.0)
        shaft = trimesh.creation.cylinder(
            radius=radius, height=shaft_length, sections=16, transform=shaft_tf)
        head_tf = np.asarray(align, float)
        head_tf[:3, 3] = direction * (shaft_length + head_length / 2.0)
        head = trimesh.creation.cone(
            radius=2.1 * radius, height=head_length, sections=20, transform=head_tf)
        values.extend([
            colored_primitive(shaft, AXIS_COLORS_RGB[name]),
            colored_primitive(head, AXIS_COLORS_RGB[name]),
        ])
    values.append(colored_primitive(
        trimesh.creation.icosphere(subdivisions=2, radius=1.7 * radius), (245, 245, 245)))
    return values


def camera(frame: int, cfg: dict, view_index: int,
           output_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    recon = load_recon(cfg, f"{frame:06d}", backend=cfg["recon_backend"])
    K = scale_intrinsics(recon["intrinsics"][view_index], recon["depth_hw"], output_hw)
    return K, recon["extrinsics"][view_index]


def project_origin(T: np.ndarray, K: np.ndarray, E: np.ndarray) -> tuple[int, int] | None:
    point = E[:3, :3] @ T[:3, 3] + E[:3, 3]
    if point[2] <= 0.01:
        return None
    uvw = K @ point
    return int(round(uvw[0] / uvw[2])), int(round(uvw[1] / uvw[2]))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def annotate_axes(image: np.ndarray, records: dict, K: np.ndarray, E: np.ndarray) -> None:
    for part, record in records.items():
        T = np.asarray(record["T_world_from_part"], float)
        uv = project_origin(T, K, E)
        if uv is None:
            continue
        x, y = uv
        cv2.putText(image, part, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, part, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (20, 20, 20), 1, cv2.LINE_AA)


def annotate_status(image: np.ndarray, frame: int, records: dict) -> None:
    label = (f"frame {frame:03d} | inner: {records['inner_pot']['state']} | "
             f"lid: {records['lid']['state']}")
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(image, label, (14, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "pose_multiview_111.json"))
    parser.add_argument("--view", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    output = Path(cfg["output_root"])
    trajectory = load_json(output / "pose" / "trajectory.json")
    view = args.view or cfg["render"]["primary_view"]
    view_index = cfg["views"].index(view)
    width, height = args.width, args.height
    fps = float(cfg["render"]["fps"])
    render_dir = output / "render" / view
    render_dir.mkdir(parents=True, exist_ok=True)

    raw_meshes = {
        part: trimesh.load(Path(cfg["mesh_dir"]) / f"{part}.glb", force="mesh")
        for part in cfg["parts"]
    }
    color_meshes = {
        part: solid_mesh(raw_meshes[part], COLORS_RGB[part]) for part in cfg["parts"]
    }
    axes = make_axis_meshes(float(cfg["render"]["axis_length_m"]),
                            float(cfg["render"]["axis_radius_m"]))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {
        "overlay": cv2.VideoWriter(str(render_dir / "overlay.mp4"), fourcc, fps, (width, height)),
        "mesh_only": cv2.VideoWriter(str(render_dir / "mesh_only.mp4"), fourcc, fps, (width, height)),
        "mesh_axes": cv2.VideoWriter(str(render_dir / "mesh_axes.mp4"), fourcc, fps, (width, height)),
    }
    if not all(writer.isOpened() for writer in writers.values()):
        raise RuntimeError("could not open output video writers")

    sample_frames = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110}
    samples = []
    metrics = {"view": view, "frames": {}}
    mask_root = Path(cfg["masks_dir"])
    try:
        with SceneRenderer(width, height) as renderer:
            for key, frame_record in trajectory["frames"].items():
                frame = int(key)
                records = frame_record["parts"]
                image_path = Path(cfg["frames_dir"]) / view / f"{key}.jpg"
                source = cv2.imread(str(image_path))
                if source is None:
                    raise FileNotFoundError(image_path)
                source = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
                K, E = camera(frame, cfg, view_index, (height, width))

                transforms = {
                    part: np.asarray(records[part]["S_world_from_raw_mesh"], float)
                    for part in cfg["parts"]
                }
                rigid = {
                    part: np.asarray(records[part]["T_world_from_part"], float)
                    for part in cfg["parts"]
                }
                mesh_parts = [(color_meshes[part], transforms[part]) for part in cfg["parts"]]
                mesh_rgb, mesh_depth = renderer.render(mesh_parts, K, E)
                mesh_bgr = cv2.cvtColor(mesh_rgb, cv2.COLOR_RGB2BGR)
                mesh_bgr[mesh_depth <= 0] = 0

                axis_parts = list(mesh_parts)
                for part in cfg["parts"]:
                    axis_parts.extend((axis, rigid[part]) for axis in axes)
                axes_rgb, axes_depth = renderer.render(axis_parts, K, E)
                axes_bgr = cv2.cvtColor(axes_rgb, cv2.COLOR_RGB2BGR)
                axes_bgr[axes_depth <= 0] = 0
                annotate_axes(axes_bgr, records, K, E)
                annotate_status(axes_bgr, frame, records)

                foreground = mesh_depth > 0
                overlay = source.copy().astype(np.float32)
                overlay[foreground] = 0.42 * overlay[foreground] + 0.58 * mesh_bgr[foreground]
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                annotate_status(overlay, frame, records)

                seg = renderer.render_seg(
                    [(part, raw_meshes[part], transforms[part]) for part in cfg["parts"]], K, E)
                labels = np.asarray(Image.open(mask_root / key / f"{view}.png"))
                labels = cv2.resize(labels.astype(np.uint8), (width, height),
                                    interpolation=cv2.INTER_NEAREST)
                metrics["frames"][key] = {
                    part: {
                        "silhouette_iou": iou(seg[part], labels == int(cfg["part_ids"][part])),
                        "rendered_pixels": int(seg[part].sum()),
                        "mask_pixels": int((labels == int(cfg["part_ids"][part])).sum()),
                        "state": records[part]["state"],
                    }
                    for part in cfg["parts"]
                }

                writers["overlay"].write(overlay)
                writers["mesh_only"].write(mesh_bgr)
                writers["mesh_axes"].write(axes_bgr)
                if frame in sample_frames:
                    tile = np.hstack([
                        cv2.resize(overlay, (480, 270), interpolation=cv2.INTER_AREA),
                        cv2.resize(axes_bgr, (480, 270), interpolation=cv2.INTER_AREA),
                    ])
                    samples.append(tile)
                print(f"rendered {frame:03d}/110", flush=True)
    finally:
        for writer in writers.values():
            writer.release()

    with open(render_dir / "metrics.json", "w", encoding="utf-8") as f:
        metrics["summary"] = {}
        for part in cfg["parts"]:
            visible = [
                value[part]["silhouette_iou"] for value in metrics["frames"].values()
                if value[part]["mask_pixels"] > 0 and value[part]["rendered_pixels"] > 0
            ]
            moving = [
                value[part]["silhouette_iou"] for value in metrics["frames"].values()
                if value[part]["state"] == "moving"
                and value[part]["mask_pixels"] > 0 and value[part]["rendered_pixels"] > 0
            ]
            metrics["summary"][part] = {
                "visible_frames": len(visible),
                "visible_mean_iou": float(np.mean(visible)) if visible else None,
                "visible_median_iou": float(np.median(visible)) if visible else None,
                "moving_visible_frames": len(moving),
                "moving_visible_mean_iou": float(np.mean(moving)) if moving else None,
            }
        json.dump(metrics, f, indent=2)
    if samples:
        contact = np.vstack(samples)
        cv2.imwrite(str(render_dir / "contact_sheet.jpg"), contact,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote videos to {render_dir}", flush=True)


if __name__ == "__main__":
    main()
