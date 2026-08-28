#!/usr/bin/env python
"""Export multi-view pose metrics, review sheets, and a manual-GT template."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.geom import project_points
from common.mesh_render import SceneRenderer
from common.normalized_recon import load_recon
from common.pose_visualization import (
    camera_from_recon,
    part_color,
    silhouette_review_metrics,
    solid_mesh,
)
from common.symmetry import symmetry_spec_from_state


def record_visible_in_view(record: dict, view: str) -> bool:
    """Use the trajectory's camera-level visibility when available."""

    if record.get("pose_valid") is False:
        return False
    visible_views = record.get("visible_views")
    if visible_views is not None:
        return str(view) in {str(value) for value in visible_views}
    return int(record.get("observing_views", 0)) > 0


def draw_pose_axes(
    image: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    length_m: float,
) -> None:
    """Draw the part-frame XYZ axes in red, green, and blue."""

    points = np.vstack([
        pose[:3, 3],
        pose[:3, 3] + pose[:3, 0] * length_m,
        pose[:3, 3] + pose[:3, 1] * length_m,
        pose[:3, 3] + pose[:3, 2] * length_m,
    ])
    uv, depth = project_points(points, intrinsics, extrinsics)
    if not np.all(np.isfinite(uv)) or np.any(depth <= 0.0):
        return
    origin = tuple(np.rint(uv[0]).astype(int))
    for index, color in enumerate(
        ((0, 0, 255), (0, 255, 0), (255, 0, 0)), 1
    ):
        endpoint = tuple(np.rint(uv[index]).astype(int))
        cv2.arrowedLine(
            image, origin, endpoint, color, 3, cv2.LINE_AA, tipLength=0.18
        )


def draw_mask_guides(
    image: np.ndarray,
    labels: np.ndarray,
    part_id: int,
) -> None:
    """Draw the observed mask contour and its image-space bounding box."""

    mask = labels == int(part_id)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
    ys, xs = np.nonzero(mask)
    if len(xs):
        cv2.rectangle(
            image,
            (int(xs.min()), int(ys.min())),
            (int(xs.max()), int(ys.max())),
            (0, 255, 255),
            2,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--review-width", type=int, default=640)
    parser.add_argument("--review-height", type=int, default=360)
    parser.add_argument("--focus-part", default=None)
    parser.add_argument(
        "--focus-only",
        action="store_true",
        help="render only --focus-part in review sheets",
    )
    parser.add_argument(
        "--draw-pose-guides",
        action="store_true",
        help="draw the focus mask contour, bbox, and XYZ pose axes",
    )
    parser.add_argument("--keyframes", type=int, nargs="*", default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="override diagnostics/multiview_metrics.json",
    )
    parser.add_argument(
        "--metrics-keyframes-only",
        action="store_true",
        help="compute metrics only on --keyframes instead of the full trajectory",
    )
    parser.add_argument(
        "--skip-review-sheets",
        action="store_true",
        help="write multiview metrics only, without review sheets or GT template",
    )
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    output = Path(args.output_root or cfg["output_root"]).resolve()
    trajectory_path = Path(args.trajectory or output / "pose" / "trajectory.json").resolve()
    trajectory = load_json(trajectory_path)
    focus_part = str(
        args.focus_part
        or cfg.get("review", {}).get(
            "focus_part", trajectory["parts"][-1]
        )
    )
    if focus_part not in trajectory["parts"]:
        raise ValueError(f"unknown review focus part: {focus_part}")
    configured_keyframes = args.keyframes or cfg.get("review_keyframes")
    if configured_keyframes is None:
        available = sorted(int(key) for key in trajectory["frames"])
        sample_indices = np.linspace(
            0, len(available) - 1, min(8, len(available)), dtype=int
        )
        configured_keyframes = [available[index] for index in sample_indices]
    keyframes = set(map(int, configured_keyframes))
    trajectory_items = list(trajectory["frames"].items())
    trajectory_items = [
        item
        for item in trajectory_items
        if (args.start_frame is None or int(item[0]) >= args.start_frame)
        and (args.end_frame is None or int(item[0]) <= args.end_frame)
    ]
    if args.metrics_keyframes_only:
        trajectory_items = [
            item for item in trajectory_items if int(item[0]) in keyframes
        ]
    raw_meshes = {
        part: trimesh.load(Path(cfg["mesh_dir"]) / f"{part}.glb", force="mesh")
        for part in trajectory["parts"]
    }
    color_meshes = {
        part: solid_mesh(raw_meshes[part], part_color(index))
        for index, part in enumerate(trajectory["parts"])
    }
    report = {
        "config": str(Path(args.config).resolve()), "trajectory": str(trajectory_path),
        "resolution": [args.width, args.height], "frames": {}, "summary": {},
    }
    accumulator = {
        part: {view: {"iou": [], "chamfer": []} for view in cfg["views"]}
        for part in trajectory["parts"]
    }
    review_dir = output / "review" / "keyframes"
    review_dir.mkdir(parents=True, exist_ok=True)

    with SceneRenderer(
        args.width, args.height, cache_mesh_resources=True
    ) as renderer:
        for frame_index, (timestamp, frame_record) in enumerate(trajectory_items):
            frame = int(timestamp)
            recon = load_recon(cfg, timestamp, backend=cfg["recon_backend"])
            transforms = {part: np.asarray(frame_record["parts"][part]["S_world_from_raw_mesh"], float)
                          for part in trajectory["parts"]}
            report["frames"][timestamp] = {}
            for view_index, view in enumerate(cfg["views"]):
                visible_parts = [
                    part
                    for part in trajectory["parts"]
                    if record_visible_in_view(
                        frame_record["parts"][part], view
                    )
                ]
                K, E = camera_from_recon(
                    recon, view_index, (args.height, args.width)
                )
                rendered = renderer.render_seg(
                    [
                        (part, raw_meshes[part], transforms[part])
                        for part in visible_parts
                    ],
                    K,
                    E,
                )
                labels = np.asarray(Image.open(Path(cfg["masks_dir"]) / timestamp / f"{view}.png"))
                labels = cv2.resize(labels.astype(np.uint8), (args.width, args.height),
                                    interpolation=cv2.INTER_NEAREST)
                report["frames"][timestamp][view] = {}
                for part in trajectory["parts"]:
                    target = labels == int(cfg["part_ids"][part])
                    prediction = rendered.get(
                        part, np.zeros((args.height, args.width), dtype=bool)
                    )
                    value_iou, chamfer = silhouette_review_metrics(
                        prediction, target
                    )
                    item = {
                        "silhouette_iou": value_iou, "contour_chamfer_px": chamfer,
                        "mask_pixels": int(target.sum()), "rendered_pixels": int(prediction.sum()),
                    }
                    report["frames"][timestamp][view][part] = item
                    if target.any() and prediction.any():
                        accumulator[part][view]["iou"].append(value_iou)
                        accumulator[part][view]["chamfer"].append(chamfer)
            if frame_index % 10 == 0:
                print(
                    f"metrics {frame_index + 1}/{len(trajectory_items)}",
                    flush=True,
                )

    for part in trajectory["parts"]:
        report["summary"][part] = {"per_view": {}}
        all_iou, all_chamfer = [], []
        for view in cfg["views"]:
            ious = accumulator[part][view]["iou"]
            distances = accumulator[part][view]["chamfer"]
            report["summary"][part]["per_view"][view] = {
                "visible_frames": len(ious),
                "mean_iou": float(np.mean(ious)) if ious else None,
                "mean_contour_chamfer_px": float(np.mean(distances)) if distances else None,
            }
            all_iou.extend(ious)
            all_chamfer.extend(distances)
        report["summary"][part]["all_views"] = {
            "visible_observations": len(all_iou),
            "mean_iou": float(np.mean(all_iou)) if all_iou else None,
            "mean_contour_chamfer_px": float(np.mean(all_chamfer)) if all_chamfer else None,
        }
    metrics_path = Path(
        args.metrics_output
        or output / "diagnostics" / "multiview_metrics.json"
    ).resolve()
    write_json(metrics_path, report)
    if args.skip_review_sheets:
        print(f"wrote {metrics_path}")
        return

    # Higher-resolution, multi-camera review sheets for the frames that will be
    # manually adjusted or approved as ground truth.
    with SceneRenderer(
        args.review_width,
        args.review_height,
        cache_mesh_resources=True,
    ) as renderer:
        for frame in sorted(keyframes):
            timestamp = f"{frame:06d}"
            record = trajectory["frames"][timestamp]
            recon = load_recon(cfg, timestamp, backend=cfg["recon_backend"])
            transforms = {part: np.asarray(record["parts"][part]["S_world_from_raw_mesh"], float)
                          for part in trajectory["parts"]}
            panels = []
            for view_index, view in enumerate(cfg["views"]):
                visible_parts = [
                    part
                    for part in trajectory["parts"]
                    if record_visible_in_view(record["parts"][part], view)
                ]
                K, E = camera_from_recon(
                    recon,
                    view_index,
                    (args.review_height, args.review_width),
                )
                review_parts = (
                    [focus_part]
                    if args.focus_only and focus_part in visible_parts
                    else ([] if args.focus_only else visible_parts)
                )
                rgb, depth = renderer.render(
                    [
                        (color_meshes[part], transforms[part])
                        for part in review_parts
                    ],
                    K,
                    E,
                )
                rendered = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                source = cv2.imread(str(Path(cfg["frames_dir"]) / view / f"{timestamp}.jpg"))
                source = cv2.resize(source, (args.review_width, args.review_height),
                                    interpolation=cv2.INTER_AREA)
                visible = depth > 0
                source[visible] = np.clip(
                    0.42 * source[visible] + 0.58 * rendered[visible], 0, 255).astype(np.uint8)
                if args.draw_pose_guides:
                    labels = np.asarray(
                        Image.open(
                            Path(cfg["masks_dir"]) / timestamp / f"{view}.png"
                        )
                    )
                    labels = cv2.resize(
                        labels.astype(np.uint8),
                        (args.review_width, args.review_height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    draw_mask_guides(
                        source, labels, int(cfg["part_ids"][focus_part])
                    )
                    pose = np.asarray(
                        record["parts"][focus_part]["T_world_from_part"],
                        dtype=np.float64,
                    )
                    axis_length = max(
                        0.025,
                        float(np.max(raw_meshes[focus_part].extents))
                        * float(trajectory["scales"][focus_part])
                        * 0.45,
                    )
                    draw_pose_axes(source, pose, K, E, axis_length)
                focus_metric = report["frames"][timestamp][view][focus_part]
                label = (
                    f"{timestamp} {view} {focus_part} IoU "
                    f"{focus_metric['silhouette_iou']:.3f}"
                )
                cv2.putText(source, label,
                            (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3,
                            cv2.LINE_AA)
                cv2.putText(source, label,
                            (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 1,
                            cv2.LINE_AA)
                panels.append(source)
            columns = 4 if len(panels) > 6 else 3
            blank = np.zeros_like(panels[0])
            panels.extend([blank] * ((columns - len(panels) % columns) % columns))
            sheet = np.vstack([
                np.hstack(panels[index:index + columns])
                for index in range(0, len(panels), columns)
            ])
            cv2.imwrite(str(review_dir / f"{timestamp}.jpg"), sheet,
                        [cv2.IMWRITE_JPEG_QUALITY, 93])
            print(f"review {frame:03d}", flush=True)

    gt_template = {
        "schema": "pose_solver_multiview_keyframe_gt_v1",
        "warning": "Initial poses are algorithm output, not ground truth. A reviewer must adjust and approve them.",
        "coordinate_conventions": trajectory["conventions"],
        "reference_part": trajectory["reference_part"],
        "symmetry": {
            part: symmetry_spec_from_state(
                cfg["states"][part]
            ).as_dict()
            for part in trajectory["parts"]
        },
        "semantic_axes": {
            part: {"defined": False, "origin_definition": None,
                   "x_direction": None, "y_direction": None, "z_direction": None}
            for part in trajectory["parts"]
        },
        "assembly_definitions": [
            {
                "name": rule.get("name"),
                "container": rule.get("container"),
                "moving_part": rule.get("moving_part"),
                "approved_T_container_from_part": None,
                "translation_tolerance_m": None,
                "rotation_tolerance_deg": None,
            }
            for rule in cfg.get("assembly_validation", [])
        ],
        "frames": {},
    }
    for frame in sorted(keyframes):
        timestamp = f"{frame:06d}"
        gt_template["frames"][timestamp] = {
            "review_status": "needs_manual_review",
            "review_sheet": str(review_dir / f"{timestamp}.jpg"),
            "parts": {
                part: {
                    "T_world_from_part_initial": trajectory["frames"][timestamp]["parts"][part]
                    ["T_world_from_part"],
                    "T_world_from_part_gt": None,
                    "confidence": None,
                    "unobservable_dofs": [],
                    "semantic_keypoints_2d": {view: {} for view in cfg["views"]},
                    "view_annotations": {
                        view: {"visible_fraction": None, "truncated": None,
                               "occluder_type": None}
                        for view in cfg["views"]
                    },
                } for part in trajectory["parts"]
            },
        }
    write_json(output / "review" / "gt_keyframes_template.json", gt_template)
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    from common.resource_safety import require_memory_guard

    require_memory_guard("tools/diagnostics/export_multiview_pose_review.py")
    main()
