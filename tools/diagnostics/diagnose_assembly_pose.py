#!/usr/bin/env python3
"""Compare an observed assembly pose with bounded axial translations.

The diagnostic is deliberately read-only with respect to the pose trajectory.
It scores the exact visual mesh against the synchronized multi-view masks, so
an Isaac contact correction cannot silently become a pose correction.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
import trimesh


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.exact_render_refinement import ExactMultiViewRenderObjective
from common.io_utils import load_json, write_json
from common.mesh_render import SceneRenderer
from common.pose_config import validate_pose_config
from common.pose_transforms import similarity_from_rigid
from common.resource_safety import require_memory_guard
from tools.stages.pose.refine_pose_render_loss import load_observations


MAX_FRAMES = 8
MAX_OFFSETS = 41
MAX_VIEWS = 16
MAX_FACES = 1_000_000
MAX_WIDTH = 640
MAX_HEIGHT = 360


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def axial_candidate_pose(
    moving_pose: np.ndarray,
    reference_pose: np.ndarray,
    axis_reference: np.ndarray,
    offset_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis_reference, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("reference axis must be finite and non-zero")
    axis /= norm
    world_axis = np.asarray(reference_pose, dtype=np.float64)[:3, :3] @ axis
    result = np.asarray(moving_pose, dtype=np.float64).copy()
    result[:3, 3] += float(offset_m) * world_axis
    return result, world_axis


def summarize_candidates(
    frame_rows: list[dict[str, Any]],
    offsets_m: list[float],
    configured_correction_m: float,
) -> dict[str, Any]:
    candidates = []
    for offset in offsets_m:
        per_frame = [
            frame["candidates"][f"{offset:+.6f}"] for frame in frame_rows
        ]
        candidates.append({
            "offset_m": float(offset),
            "mean_loss": float(np.mean([row["loss"] for row in per_frame])),
            "mean_iou": float(
                np.mean([row["mean_iou"] for row in per_frame])
            ),
            "worst_view_iou": float(
                min(row["worst_view_iou"] for row in per_frame)
            ),
            "mean_target_coverage": float(np.mean([
                row["mean_target_coverage"] for row in per_frame
            ])),
            "evaluated_frames": len(per_frame),
        })
    best = min(candidates, key=lambda row: row["mean_loss"])
    raw = min(candidates, key=lambda row: abs(row["offset_m"]))
    corrected = min(
        candidates,
        key=lambda row: abs(row["offset_m"] - configured_correction_m),
    )
    spacing = min(
        (
            abs(right - left)
            for left, right in zip(offsets_m[:-1], offsets_m[1:])
            if not math.isclose(left, right)
        ),
        default=0.0,
    )
    raw_supported = abs(float(best["offset_m"])) <= max(0.005, spacing)
    correction_supported = (
        abs(float(best["offset_m"]) - configured_correction_m)
        <= max(0.005, spacing)
    )
    iou_delta = float(corrected["mean_iou"] - raw["mean_iou"])
    loss_delta = float(corrected["mean_loss"] - raw["mean_loss"])
    if raw_supported and loss_delta > 0.005 and iou_delta < -0.01:
        diagnosis = "visual_pose_supports_raw_pose_and_rejects_contact_correction"
    elif correction_supported and loss_delta < -0.005 and iou_delta > 0.01:
        diagnosis = "visual_pose_supports_configured_axial_correction"
    else:
        diagnosis = "visual_evidence_inconclusive"
    return {
        "candidates": candidates,
        "best": best,
        "raw_pose": raw,
        "configured_correction": corrected,
        "configured_minus_raw": {
            "loss": loss_delta,
            "mean_iou": iou_delta,
            "mean_target_coverage": float(
                corrected["mean_target_coverage"]
                - raw["mean_target_coverage"]
            ),
        },
        "diagnosis": diagnosis,
    }


def _compact_score(score: dict[str, Any]) -> dict[str, Any]:
    return {
        key: score[key]
        for key in (
            "loss",
            "worst_view_loss",
            "mean_iou",
            "worst_view_iou",
            "mean_contour_chamfer_px",
            "mean_target_coverage",
        )
    }


def _image_path(frames_dir: Path, view: str, timestamp: str) -> Path | None:
    for suffix in (".jpg", ".png", ".jpeg"):
        path = frames_dir / view / f"{timestamp}{suffix}"
        if path.is_file():
            return path
    return None


def _outline(mask: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(
        np.asarray(mask, dtype=np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)


def _comparison_tile(
    cfg: dict[str, Any],
    timestamp: str,
    observation: Any,
    renderer: SceneRenderer,
    mesh: trimesh.Trimesh,
    scale: float,
    raw_origin: np.ndarray,
    raw_pose: np.ndarray,
    corrected_pose: np.ndarray,
) -> Image.Image:
    height, width = observation.target_mask.shape
    image_path = _image_path(Path(cfg["frames_dir"]), observation.view, timestamp)
    if image_path is None:
        background = np.full((height, width, 3), 48, dtype=np.uint8)
    else:
        background = cv2.resize(
            np.asarray(Image.open(image_path).convert("RGB")),
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
    predictions = []
    for pose in (raw_pose, corrected_pose):
        transform = similarity_from_rigid(pose, scale, raw_origin)
        _, depth = renderer.render(
            [(mesh, transform)], observation.intrinsics, observation.extrinsics
        )
        predictions.append(depth > 0.0)
    output = background.copy()
    output[_outline(observation.target_mask)] = [70, 255, 70]
    output[_outline(predictions[0])] = [0, 255, 255]
    output[_outline(predictions[1])] = [255, 60, 60]
    tile = Image.fromarray(output)
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, width, 18), fill=(0, 0, 0))
    draw.text(
        (4, 3),
        f"{timestamp} {observation.view} target=green raw=cyan corrected=red",
        fill=(255, 255, 255),
    )
    return tile


def _save_sheet(tiles: list[Image.Image], output: Path) -> None:
    if not tiles:
        return
    columns = min(4, len(tiles))
    rows = int(math.ceil(len(tiles) / columns))
    width, height = tiles[0].size
    sheet = Image.new("RGB", (columns * width, rows * height), (20, 20, 20))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * width, (index // columns) * height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=95)


def main() -> None:
    require_memory_guard("tools/diagnostics/diagnose_assembly_pose.py")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--reference-part", required=True)
    parser.add_argument("--moving-part", required=True)
    parser.add_argument("--frames", required=True, nargs="+", type=int)
    parser.add_argument(
        "--axis-reference", nargs=3, type=float, default=[0.0, 1.0, 0.0]
    )
    parser.add_argument("--offsets-mm", required=True, nargs="+", type=float)
    parser.add_argument("--configured-correction-mm", type=float, default=15.0)
    parser.add_argument("--resolution", nargs=2, type=int, default=[320, 180])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--visualization", type=Path)
    parser.add_argument("--maximum-faces", type=int, default=MAX_FACES)
    args = parser.parse_args()

    frames = sorted(set(args.frames))
    offsets_m = sorted(set(float(value) / 1000.0 for value in args.offsets_mm))
    width, height = map(int, args.resolution)
    if not frames or len(frames) > MAX_FRAMES:
        raise ValueError(f"frames must contain 1..{MAX_FRAMES} unique values")
    if not offsets_m or len(offsets_m) > MAX_OFFSETS:
        raise ValueError(f"offsets must contain 1..{MAX_OFFSETS} unique values")
    if max(abs(value) for value in offsets_m) > 0.05:
        raise ValueError("absolute axial offset must not exceed 50 mm")
    if not (1 <= width <= MAX_WIDTH and 1 <= height <= MAX_HEIGHT):
        raise ValueError(f"resolution must not exceed {MAX_WIDTH}x{MAX_HEIGHT}")

    config_path = args.config.expanduser().resolve()
    trajectory_path = args.trajectory.expanduser().resolve()
    cfg = validate_pose_config(load_json(config_path), check_paths=True)
    trajectory = load_json(trajectory_path)
    if len(cfg["views"]) > MAX_VIEWS:
        raise ValueError(f"refusing {len(cfg['views'])} views; limit={MAX_VIEWS}")
    for part in (args.reference_part, args.moving_part):
        if part not in trajectory["parts"]:
            raise ValueError(f"unknown trajectory part: {part}")

    mesh_path = Path(cfg["mesh_dir"]) / f"{args.moving_part}.glb"
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"expected non-empty mesh: {mesh_path}")
    if len(mesh.faces) > min(MAX_FACES, int(args.maximum_faces)):
        raise ValueError(
            f"refusing {len(mesh.faces)} faces; limit={args.maximum_faces}"
        )

    settings = {
        "resolution": [width, height],
        "min_mask_pixels": 20,
        "minimum_full_mask_pixels": 100,
        "maximum_mask_area_ratio": 4.0,
        "minimum_mask_area_ratio": 0.0,
        "known_part_occlusion_aware": True,
        "occlusion_aware": False,
        "mask_primary": True,
        "use_cloud_supported_view_gate": False,
        "trim_worst_views": 1,
        "weights": {
            "iou": 1.0,
            "contour": 0.05,
            "target_coverage": 0.10,
            "depth": 0.0,
        },
    }
    scale = float(trajectory["scales"][args.moving_part])
    raw_origin = np.asarray(
        trajectory["raw_mesh_origins"][args.moving_part], dtype=np.float64
    )
    axis_reference = np.asarray(args.axis_reference, dtype=np.float64)
    frame_rows = []
    tiles: list[Image.Image] = []
    with SceneRenderer(width, height, cache_mesh_resources=True) as renderer:
        for frame in frames:
            timestamp = f"{frame:06d}"
            if timestamp not in trajectory["frames"]:
                raise ValueError(f"trajectory does not contain frame {timestamp}")
            records = trajectory["frames"][timestamp]["parts"]
            reference_pose = np.asarray(
                records[args.reference_part]["T_world_from_part"],
                dtype=np.float64,
            )
            moving_pose = np.asarray(
                records[args.moving_part]["T_world_from_part"],
                dtype=np.float64,
            )
            observations = load_observations(
                cfg,
                timestamp,
                args.moving_part,
                settings,
                observation_state=str(
                    records[args.moving_part].get("observation_state", "")
                ),
            )
            if not observations:
                raise RuntimeError(f"no usable mask observations at {timestamp}")
            objective = ExactMultiViewRenderObjective(
                mesh, scale, raw_origin, observations, settings, renderer
            )
            candidate_rows = {}
            poses = {}
            world_axis = None
            for offset in offsets_m:
                candidate, world_axis = axial_candidate_pose(
                    moving_pose, reference_pose, axis_reference, offset
                )
                poses[f"{offset:+.6f}"] = candidate
                candidate_rows[f"{offset:+.6f}"] = _compact_score(
                    objective.evaluate(candidate)
                )
            frame_rows.append({
                "frame": frame,
                "timestamp": timestamp,
                "state": records[args.moving_part].get("state"),
                "observation_state": records[args.moving_part].get(
                    "observation_state"
                ),
                "observing_views": records[args.moving_part].get(
                    "observing_views"
                ),
                "usable_mask_views": [item.view for item in observations],
                "axis_world": np.asarray(world_axis, dtype=float).tolist(),
                "candidates": candidate_rows,
            })
            if args.visualization is not None:
                raw_key = min(poses, key=lambda key: abs(float(key)))
                correction_m = float(args.configured_correction_mm) / 1000.0
                corrected_key = min(
                    poses, key=lambda key: abs(float(key) - correction_m)
                )
                for observation in observations:
                    tiles.append(_comparison_tile(
                        cfg,
                        timestamp,
                        observation,
                        renderer,
                        mesh,
                        scale,
                        raw_origin,
                        poses[raw_key],
                        poses[corrected_key],
                    ))

    configured_correction_m = float(args.configured_correction_mm) / 1000.0
    summary = summarize_candidates(
        frame_rows, offsets_m, configured_correction_m
    )
    report = {
        "schema_version": 1,
        "method": "exact_multiview_mask_axial_pose_diagnostic",
        "read_only": True,
        "inputs": {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "trajectory": str(trajectory_path),
            "trajectory_sha256": sha256_file(trajectory_path),
            "mesh": str(mesh_path),
            "mesh_sha256": sha256_file(mesh_path),
        },
        "reference_part": args.reference_part,
        "moving_part": args.moving_part,
        "reference_axis_part": (
            axis_reference / np.linalg.norm(axis_reference)
        ).tolist(),
        "configured_correction_m": configured_correction_m,
        "frames": frame_rows,
        "summary": summary,
        "interpretation": {
            "visual_pose_scope": (
                "Tests whether an axial translation improves exact multi-view "
                "mask agreement; it does not validate threads or collision clearance."
            ),
            "physics_scope": (
                "A PhysX contact correction rejected here must remain simulation "
                "metadata and must not be written into the observed pose trajectory."
            ),
        },
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    if args.visualization is not None:
        _save_sheet(tiles, args.visualization.expanduser().resolve())
    print(f"assembly pose diagnostic -> {output_path}")
    print(
        f"diagnosis={summary['diagnosis']} "
        f"best_offset_mm={summary['best']['offset_m'] * 1000.0:.2f} "
        f"raw_iou={summary['raw_pose']['mean_iou']:.4f} "
        f"corrected_iou={summary['configured_correction']['mean_iou']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
