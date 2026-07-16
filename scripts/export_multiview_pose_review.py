#!/usr/bin/env python
"""Export six-view Pose metrics, review sheets, and a manual-GT template."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.mesh_render import SceneRenderer
from common.normalized_recon import load_recon, scale_intrinsics


COLORS_RGB = {
    "body": (75, 190, 95),
    "inner_pot": (245, 135, 40),
    "lid": (85, 145, 245),
}


def solid_mesh(mesh: trimesh.Trimesh, rgb: tuple[int, int, int]) -> trimesh.Trimesh:
    result = mesh.copy()
    rgba = np.asarray([*rgb, 255], dtype=np.uint8)
    result.visual = trimesh.visual.ColorVisuals(
        mesh=result, face_colors=np.tile(rgba, (len(result.faces), 1)))
    return result


def mask_edge(mask: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8)).astype(bool)


def metrics(rendered: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    union = np.logical_or(rendered, target).sum()
    iou = float(np.logical_and(rendered, target).sum() / union) if union else 1.0
    er, et = mask_edge(rendered), mask_edge(target)
    if not er.any() or not et.any():
        return iou, 20.0
    dt_t = cv2.distanceTransform((~et).astype(np.uint8), cv2.DIST_L2, 3)
    dt_r = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3)
    chamfer = 0.5 * (float(np.minimum(dt_t[er], 20.0).mean())
                     + float(np.minimum(dt_r[et], 20.0).mean()))
    return iou, chamfer


def camera(cfg: dict, recon: dict, view_index: int,
           height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    return (scale_intrinsics(recon["intrinsics"][view_index], recon["depth_hw"],
                             (height, width)), recon["extrinsics"][view_index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "pose_multiview_111_v4.json"))
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--review-width", type=int, default=640)
    parser.add_argument("--review-height", type=int, default=360)
    parser.add_argument("--keyframes", type=int, nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    output = Path(args.output_root or cfg["output_root"]).resolve()
    trajectory_path = Path(args.trajectory or output / "pose" / "trajectory.json").resolve()
    trajectory = load_json(trajectory_path)
    keyframes = set(args.keyframes or cfg.get("review_keyframes") or
                    cfg.get("lid_refinement", {}).get(
                        "keyframes", [0, 12, 20, 40, 50, 70, 80, 85, 100, 108]))
    raw_meshes = {
        part: trimesh.load(Path(cfg["mesh_dir"]) / f"{part}.glb", force="mesh")
        for part in trajectory["parts"]
    }
    color_meshes = {part: solid_mesh(raw_meshes[part], COLORS_RGB[part])
                    for part in trajectory["parts"]}
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

    with SceneRenderer(args.width, args.height) as renderer:
        for frame_index, (timestamp, frame_record) in enumerate(trajectory["frames"].items()):
            frame = int(timestamp)
            recon = load_recon(cfg, timestamp, backend=cfg["recon_backend"])
            transforms = {part: np.asarray(frame_record["parts"][part]["S_world_from_raw_mesh"], float)
                          for part in trajectory["parts"]}
            report["frames"][timestamp] = {}
            for view_index, view in enumerate(cfg["views"]):
                K, E = camera(cfg, recon, view_index, args.height, args.width)
                rendered = renderer.render_seg(
                    [(part, raw_meshes[part], transforms[part]) for part in trajectory["parts"]], K, E)
                labels = np.asarray(Image.open(Path(cfg["masks_dir"]) / timestamp / f"{view}.png"))
                labels = cv2.resize(labels.astype(np.uint8), (args.width, args.height),
                                    interpolation=cv2.INTER_NEAREST)
                report["frames"][timestamp][view] = {}
                for part in trajectory["parts"]:
                    target = labels == int(cfg["part_ids"][part])
                    value_iou, chamfer = metrics(rendered[part], target)
                    item = {
                        "silhouette_iou": value_iou, "contour_chamfer_px": chamfer,
                        "mask_pixels": int(target.sum()), "rendered_pixels": int(rendered[part].sum()),
                    }
                    report["frames"][timestamp][view][part] = item
                    if target.any() and rendered[part].any():
                        accumulator[part][view]["iou"].append(value_iou)
                        accumulator[part][view]["chamfer"].append(chamfer)
            if frame_index % 10 == 0:
                print(f"metrics {frame:03d}/110", flush=True)

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
    write_json(output / "diagnostics" / "multiview_metrics.json", report)

    # Higher-resolution, six-camera review sheets for the frames that will be
    # manually adjusted or approved as ground truth.
    with SceneRenderer(args.review_width, args.review_height) as renderer:
        for frame in sorted(keyframes):
            timestamp = f"{frame:06d}"
            record = trajectory["frames"][timestamp]
            recon = load_recon(cfg, timestamp, backend=cfg["recon_backend"])
            transforms = {part: np.asarray(record["parts"][part]["S_world_from_raw_mesh"], float)
                          for part in trajectory["parts"]}
            panels = []
            for view_index, view in enumerate(cfg["views"]):
                K, E = camera(cfg, recon, view_index, args.review_height, args.review_width)
                rgb, depth = renderer.render(
                    [(color_meshes[part], transforms[part]) for part in trajectory["parts"]], K, E)
                rendered = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                source = cv2.imread(str(Path(cfg["frames_dir"]) / view / f"{timestamp}.jpg"))
                source = cv2.resize(source, (args.review_width, args.review_height),
                                    interpolation=cv2.INTER_AREA)
                visible = depth > 0
                source[visible] = np.clip(
                    0.42 * source[visible] + 0.58 * rendered[visible], 0, 255).astype(np.uint8)
                lid_metric = report["frames"][timestamp][view]["lid"]
                cv2.putText(source, f"{timestamp} {view} lid IoU {lid_metric['silhouette_iou']:.3f}",
                            (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3,
                            cv2.LINE_AA)
                cv2.putText(source, f"{timestamp} {view} lid IoU {lid_metric['silhouette_iou']:.3f}",
                            (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 1,
                            cv2.LINE_AA)
                panels.append(source)
            sheet = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
            cv2.imwrite(str(review_dir / f"{timestamp}.jpg"), sheet,
                        [cv2.IMWRITE_JPEG_QUALITY, 93])
            print(f"review {frame:03d}", flush=True)

    gt_template = {
        "schema": "pose_solver_multiview_keyframe_gt_v1",
        "warning": "Initial poses are algorithm output, not ground truth. A reviewer must adjust and approve them.",
        "coordinate_conventions": trajectory["conventions"],
        "reference_part": trajectory["reference_part"],
        "symmetry": {
            "body": {"equivalent_rotations": []},
            "inner_pot": {"axis": "local_y", "continuous_or_discrete": None},
            "lid": {"axis": "local_y", "continuous_or_discrete": None},
        },
        "semantic_axes": {
            part: {"defined": False, "origin_definition": None,
                   "x_direction": None, "y_direction": None, "z_direction": None}
            for part in trajectory["parts"]
        },
        "assembly_definitions": {
            "T_body_from_inner_pot_assembled": None,
            "T_body_from_lid_assembled": None,
            "translation_tolerance_m": None,
            "rotation_tolerance_deg": None,
            "allowed_penetration_m": None,
        },
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
    print(f"wrote {output / 'diagnostics' / 'multiview_metrics.json'}")


if __name__ == "__main__":
    main()
