#!/usr/bin/env python3
"""Remove flexible mask regions using a coarse rigid-pose mesh projection."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import trimesh


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.mesh_render import SceneRenderer
from common.normalized_recon import load_recon
from common.pose_visualization import camera_from_recon
from common.rigid_observation import pose_guided_rigid_region


def _save_labels(path: Path, labels: np.ndarray, source: Image.Image) -> None:
    output = Image.fromarray(np.asarray(labels, dtype=np.uint8), mode="P")
    palette = source.getpalette()
    if palette is not None:
        output.putpalette(palette)
    if "transparency" in source.info:
        output.info["transparency"] = source.info["transparency"]
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--rigid-mesh", required=True, type=Path)
    parser.add_argument("--part", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--dilation-pixels", type=int, default=14)
    parser.add_argument("--minimum-output-pixels", type=int, default=25)
    parser.add_argument("--minimum-render-overlap", type=float, default=0.10)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    trajectory_path = args.trajectory.expanduser().resolve()
    mesh_path = args.rigid_mesh.expanduser().resolve()
    config = load_json(config_path)
    trajectory = load_json(trajectory_path)
    part = str(args.part)
    if part not in trajectory["parts"] or part not in config["part_ids"]:
        raise ValueError(f"unknown part {part!r}")
    part_id = int(config["part_ids"][part])
    output_root = args.output_root.expanduser().resolve()
    report_path = output_root / "pose_guided_rigid_masks.json"
    if report_path.exists() and not args.force:
        print(f"[resume] {report_path}")
        return
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("rigid mesh must contain triangles")

    rows = []
    frame_start = (
        int(args.start_frame)
        if args.start_frame is not None
        else int(config["frames"]["start"])
    )
    frame_end = (
        int(args.end_frame)
        if args.end_frame is not None
        else int(config["frames"]["end"])
    )
    if (
        frame_start < int(config["frames"]["start"])
        or frame_end > int(config["frames"]["end"])
        or frame_start > frame_end
    ):
        raise ValueError("requested frame range lies outside the config")
    source_root = Path(config["masks_dir"])
    with SceneRenderer(
        args.width, args.height, cache_mesh_resources=True
    ) as renderer:
        for frame in range(frame_start, frame_end + 1):
            timestamp = f"{frame:06d}"
            record = trajectory["frames"][timestamp]["parts"][part]
            transform = np.asarray(
                record["S_world_from_raw_mesh"], dtype=np.float64
            )
            recon = load_recon(
                config, timestamp, backend=config["recon_backend"]
            )
            for view_index, view in enumerate(config["views"]):
                source_path = source_root / timestamp / f"{view}.png"
                with Image.open(source_path) as image:
                    labels = np.asarray(image).copy()
                    K, E = camera_from_recon(
                        recon, view_index, (args.height, args.width)
                    )
                    rendered_small = renderer.render_seg(
                        [(part, mesh, transform)], K, E
                    )[part]
                    rendered = cv2.resize(
                        rendered_small.astype(np.uint8),
                        (labels.shape[1], labels.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    scale = max(
                        labels.shape[1] / args.width,
                        labels.shape[0] / args.height,
                    )
                    selected, stats = pose_guided_rigid_region(
                        labels == part_id,
                        rendered,
                        dilation_radius=int(
                            round(args.dilation_pixels * scale)
                        ),
                        minimum_output_pixels=args.minimum_output_pixels,
                        minimum_render_overlap=args.minimum_render_overlap,
                    )
                    labels[labels == part_id] = 0
                    labels[selected] = part_id
                    _save_labels(
                        output_root / "masks" / timestamp / f"{view}.png",
                        labels,
                        image,
                    )
                rows.append({"frame": frame, "view": view, **stats})
            if (frame - frame_start) % 20 == 0:
                print(
                    f"pose-guided masks {frame - frame_start + 1}/"
                    f"{frame_end - frame_start + 1}",
                    flush=True,
                )

    accepted = [row for row in rows if row["status"] == "ok"]
    report = {
        "schema_version": 1,
        "method": "observed_mask_intersect_dilated_rigid_mesh_projection",
        "config": str(config_path),
        "trajectory": str(trajectory_path),
        "rigid_mesh": str(mesh_path),
        "part": part,
        "part_id": part_id,
        "frame_range": [frame_start, frame_end],
        "views": list(config["views"]),
        "resolution": [args.width, args.height],
        "dilation_pixels_at_resolution": args.dilation_pixels,
        "observations": len(rows),
        "accepted_observations": len(accepted),
        "rejected_observations": [
            {"frame": row["frame"], "view": row["view"]}
            for row in rows if row["status"] != "ok"
        ],
        "retained_fraction": {
            "minimum": min(
                (row["retained_fraction"] for row in accepted), default=None
            ),
            "median": (
                float(np.median([
                    row["retained_fraction"] for row in accepted
                ])) if accepted else None
            ),
            "maximum": max(
                (row["retained_fraction"] for row in accepted), default=None
            ),
        },
        "per_observation": rows,
    }
    write_json(report_path, report)
    print(f"pose-guided rigid masks -> {output_root / 'masks'}")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
