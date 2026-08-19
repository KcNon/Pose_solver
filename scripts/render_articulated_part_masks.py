#!/usr/bin/env python
"""Split an observed whole-object mask by rendering registered link meshes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.masking.io import (
    frame_path,
    load_label_mask,
    save_binary_mask,
    save_label_mask,
    write_json,
)
from common.masking.schema import load_mask_pipeline_config
from common.mesh_render import SceneRenderer
from common.normalized_recon import load_recon
from common.pose_visualization import camera_from_recon


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    union = int(np.logical_or(first, second).sum())
    return float(np.logical_and(first, second).sum()) / union if union else 1.0


def _fill_partition(
    whole: np.ndarray,
    seeds: dict[str, np.ndarray],
    names: list[str],
) -> dict[str, np.ndarray]:
    """Assign every whole-mask pixel to its nearest rendered link seed."""
    missing = [name for name in names if not seeds[name].any()]
    if missing:
        raise RuntimeError(f"rendered link seeds are empty: {missing}")
    distances = np.stack([
        cv2.distanceTransform(
            (~seeds[name]).astype(np.uint8), cv2.DIST_L2, 5
        )
        for name in names
    ])
    owner = np.argmin(distances, axis=0)
    return {
        name: np.logical_and(whole, owner == index)
        for index, name in enumerate(names)
    }


def _preview(
    image_path: Path,
    masks: dict[str, np.ndarray],
    config,
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read {image_path}")
    overlay = image.copy()
    for part in config.parts:
        selected = masks[part.name]
        color = np.asarray(part.color[::-1], dtype=np.float32)
        overlay[selected] = (
            0.45 * image[selected].astype(np.float32) + 0.55 * color
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            overlay, contours, -1, tuple(int(value) for value in color), 2
        )
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--mask-config", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--reference-part", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validated-seed-root", type=Path)
    parser.add_argument("--timestamps", nargs="+")
    parser.add_argument("--minimum-render-iou", type=float, default=0.65)
    args = parser.parse_args()

    pose_config = json.loads(args.pose_config.read_text(encoding="utf-8"))
    mask_config = load_mask_pipeline_config(args.mask_config)
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    frames = trajectory.get("frames", {})
    timestamps = list(args.timestamps or sorted(frames, key=int))
    unknown = set(timestamps).difference(frames)
    if unknown:
        raise ValueError(f"timestamps missing from trajectory: {sorted(unknown)}")
    names = list(mask_config.part_names)
    mesh_dir = Path(mask_config.raw["mesh_dir"])
    def part_mesh_path(name: str) -> Path:
        flat = mesh_dir / f"{name}.glb"
        nested = mesh_dir / name / "model.glb"
        if flat.exists():
            return flat
        if nested.exists():
            return nested
        raise FileNotFoundError(
            f"missing mesh for {name!r}; tried {flat} and {nested}"
        )

    meshes = {
        name: trimesh.load(part_mesh_path(name), force="mesh")
        for name in names
    }
    whole_masks_root = Path(pose_config["masks_dir"])
    views = list(pose_config["views"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "method": "registered_articulated_mesh_id_render_then_whole_mask_voronoi_fill",
        "pose_config": str(args.pose_config.resolve()),
        "mask_config": str(args.mask_config.resolve()),
        "trajectory": str(args.trajectory.resolve()),
        "reference_part": args.reference_part,
        "parts": names,
        "frames": {},
    }

    for timestamp in timestamps:
        record = frames[timestamp]["parts"][args.reference_part]
        transform = np.asarray(record["S_world_from_raw_mesh"], dtype=np.float64)
        recon = load_recon(pose_config, timestamp)
        frame_report = {}
        previews = []
        for view_index, view in enumerate(views):
            image_path = frame_path(Path(pose_config["frames_dir"]), view, timestamp)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to read {image_path}")
            output_hw = image.shape[:2]
            intrinsics, extrinsics = camera_from_recon(
                recon, view_index, output_hw
            )
            with SceneRenderer(output_hw[1], output_hw[0]) as renderer:
                rendered = renderer.render_seg(
                    [
                        (name, meshes[name], transform)
                        for name in names
                    ],
                    intrinsics,
                    extrinsics,
                )
            whole = load_label_mask(
                whole_masks_root / timestamp / f"{view}.png"
            ) > 0
            rendered_union = np.logical_or.reduce(
                [rendered[name] for name in names]
            )
            render_iou = _iou(rendered_union, whole)
            if render_iou < args.minimum_render_iou:
                raise RuntimeError(
                    f"{timestamp}/{view}: render IoU {render_iou:.3f} below "
                    f"{args.minimum_render_iou:.3f}"
                )
            clipped = {
                name: np.logical_and(rendered[name], whole)
                for name in names
            }
            partition = _fill_partition(whole, clipped, names)
            label = np.zeros(output_hw, dtype=np.uint8)
            for part in mask_config.parts:
                label[partition[part.name]] = part.id
                if args.validated_seed_root is not None:
                    save_binary_mask(
                        args.validated_seed_root
                        / part.name / timestamp / f"{view}.png",
                        partition[part.name],
                    )
            output = args.output_root / "masks" / timestamp / f"{view}.png"
            save_label_mask(output, label, mask_config.parts)
            preview = _preview(image_path, partition, mask_config)
            preview_path = args.output_root / "preview" / timestamp / f"{view}.jpg"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_path), preview)
            previews.append(cv2.resize(preview, (480, 270), interpolation=cv2.INTER_AREA))
            frame_report[view] = {
                "render_iou": render_iou,
                "whole_pixels": int(whole.sum()),
                "rendered_union_pixels": int(rendered_union.sum()),
                "seed_pixels": {
                    name: int(clipped[name].sum()) for name in names
                },
                "output_pixels": {
                    name: int(partition[name].sum()) for name in names
                },
            }

        columns = 4
        rows = math.ceil(len(previews) / columns)
        blank = np.zeros_like(previews[0])
        previews.extend([blank.copy() for _ in range(rows * columns - len(previews))])
        sheet = np.vstack([
            np.hstack(previews[offset : offset + columns])
            for offset in range(0, len(previews), columns)
        ])
        cv2.imwrite(
            str(args.output_root / "preview" / f"{timestamp}_all_views.jpg"),
            sheet,
        )
        report["frames"][timestamp] = frame_report
        mean_iou = float(np.mean([
            value["render_iou"] for value in frame_report.values()
        ]))
        print(f"{timestamp}: rendered part masks, mean IoU={mean_iou:.3f}", flush=True)

    write_json(args.output_root / "manifest.json", report)
    print(f"part masks -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
