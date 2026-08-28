#!/usr/bin/env python
"""Locally refine a fixed camera rig from bounded static-part observations.

This diagnostic never edits DA3 artifacts or production configuration.  It
builds visibility-gated cross-view correspondences in the existing world gauge,
estimates one constant depth correction per camera, then estimates small SE(3)
camera-cloud corrections.  Fixed held-out correspondences measure whether the
correction generalizes beyond the calibration frames.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.backproject_utils import load_palette_masks
from common.camera_rig_refinement import (
    RigCorrespondence,
    choose_anchor,
    connected_components,
    corrected_extrinsics,
    correspondence_metrics,
    optimize_depth_corrections,
    optimize_pose_corrections,
)
from common.depth_gauge import apply_depth_gauge, load_depth_gauge
from common.io_utils import load_json, write_json
from common.normalized_recon import load_recon
from common.quality_cloud import eroded_mask, smooth_depth_mask
from common.resource_safety import require_memory_guard


MAX_FRAMES_PER_SPLIT = 24
MAX_VIEWS = 16
MAX_POINTS_PER_VIEW = 10_000
MAX_CORRESPONDENCES_PER_PAIR_FRAME = 2_000
MAX_TOTAL_CORRESPONDENCES = 100_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--part", required=True)
    parser.add_argument("--train-frames", type=int, nargs="+", required=True)
    parser.add_argument(
        "--validation-frames", type=int, nargs="+", required=True
    )
    parser.add_argument("--depth-gauge", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-erode", type=int, default=2)
    parser.add_argument("--depth-edge-mm", type=float, default=20.0)
    parser.add_argument("--confidence-quantile", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-points-per-view", type=int, default=2500)
    parser.add_argument(
        "--max-correspondences-per-pair-frame", type=int, default=400
    )
    parser.add_argument("--maximum-initial-residual-mm", type=float, default=25.0)
    parser.add_argument("--minimum-pair-correspondences", type=int, default=100)
    parser.add_argument("--data-sigma-mm", type=float, default=5.0)
    parser.add_argument("--inlier-threshold-mm", type=float, default=10.0)
    parser.add_argument("--maximum-depth-correction-mm", type=float, default=30.0)
    parser.add_argument("--depth-prior-mm", type=float, default=15.0)
    parser.add_argument("--maximum-rotation-deg", type=float, default=3.0)
    parser.add_argument("--rotation-prior-deg", type=float, default=1.0)
    parser.add_argument("--maximum-translation-mm", type=float, default=30.0)
    parser.add_argument("--translation-prior-mm", type=float, default=10.0)
    parser.add_argument("--max-nfev", type=int, default=60)
    return parser


def _validate_args(args: argparse.Namespace, view_count: int) -> None:
    train = set(args.train_frames)
    validation = set(args.validation_frames)
    if len(args.train_frames) != len(train):
        raise ValueError("training frames must be unique")
    if len(args.validation_frames) != len(validation):
        raise ValueError("validation frames must be unique")
    if train & validation:
        raise ValueError("training and validation frames must be disjoint")
    if not 1 <= len(train) <= MAX_FRAMES_PER_SPLIT:
        raise ValueError(f"training frames must contain 1..{MAX_FRAMES_PER_SPLIT} values")
    if not 1 <= len(validation) <= MAX_FRAMES_PER_SPLIT:
        raise ValueError(
            f"validation frames must contain 1..{MAX_FRAMES_PER_SPLIT} values"
        )
    if not 2 <= view_count <= MAX_VIEWS:
        raise ValueError(f"view count must be in [2, {MAX_VIEWS}]")
    if not 1 <= args.stride <= 16:
        raise ValueError("stride must be in [1, 16]")
    if not 100 <= args.max_points_per_view <= MAX_POINTS_PER_VIEW:
        raise ValueError(
            f"max-points-per-view must be in [100, {MAX_POINTS_PER_VIEW}]"
        )
    if not (
        50
        <= args.max_correspondences_per_pair_frame
        <= MAX_CORRESPONDENCES_PER_PAIR_FRAME
    ):
        raise ValueError(
            "max-correspondences-per-pair-frame must be in "
            f"[50, {MAX_CORRESPONDENCES_PER_PAIR_FRAME}]"
        )
    for name in (
        "depth_edge_mm",
        "maximum_initial_residual_mm",
        "data_sigma_mm",
        "inlier_threshold_mm",
        "maximum_depth_correction_mm",
        "depth_prior_mm",
        "maximum_rotation_deg",
        "rotation_prior_deg",
        "maximum_translation_mm",
        "translation_prior_mm",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.confidence_quantile < 1.0:
        raise ValueError("confidence-quantile must be in [0, 1)")
    if not 1 <= args.max_nfev <= 200:
        raise ValueError("max-nfev must be in [1, 200]")


def _camera_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    z = depth[rows, columns].astype(np.float64)
    x = (columns.astype(np.float64) - intrinsic[0, 2]) / intrinsic[0, 0] * z
    y = (rows.astype(np.float64) - intrinsic[1, 2]) / intrinsic[1, 1] * z
    return np.stack((x, y, z), axis=1)


def _world_points(
    camera_points: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(extrinsic[:3, :3], np.float64)
    translation = np.asarray(extrinsic[:3, 3], np.float64)
    world = (camera_points - translation[None]) @ rotation
    center = -rotation.T @ translation
    rays = world - center[None]
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    return world, rays


def _view_samples(
    depth: np.ndarray,
    confidence: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    mask: np.ndarray,
    *,
    mask_erode: int,
    depth_edge_m: float,
    confidence_quantile: float,
    stride: int,
    max_points: int,
    seed: int,
) -> dict[str, np.ndarray]:
    filtered = eroded_mask(mask, mask_erode)
    valid = filtered & np.isfinite(depth) & (depth > 1e-3)
    valid &= np.isfinite(confidence) & (confidence > 0)
    if depth_edge_m > 0:
        valid &= smooth_depth_mask(depth, depth_edge_m)
    values = confidence[valid]
    if len(values):
        valid &= confidence >= float(np.quantile(values, confidence_quantile))
    sampled = np.zeros_like(valid)
    sampled[::stride, ::stride] = True
    rows, columns = np.nonzero(valid & sampled)
    if len(rows) > max_points:
        selected = np.random.default_rng(seed).choice(
            len(rows), max_points, replace=False
        )
        rows, columns = rows[selected], columns[selected]
    camera = _camera_points(depth, intrinsic, rows, columns)
    world, rays = _world_points(camera, extrinsic)
    return {
        "valid": valid,
        "rows": rows,
        "columns": columns,
        "camera": camera,
        "world": world,
        "rays": rays,
    }


def _directed_matches(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    target_depth: np.ndarray,
    target_intrinsic: np.ndarray,
    target_extrinsic: np.ndarray,
    *,
    maximum_residual_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not len(source["world"]):
        empty = np.empty((0, 3), np.float64)
        return empty, empty.copy(), empty.copy(), empty.copy()
    rotation = np.asarray(target_extrinsic[:3, :3], np.float64)
    translation = np.asarray(target_extrinsic[:3, 3], np.float64)
    camera = source["world"] @ rotation.T + translation[None]
    projected = camera @ np.asarray(target_intrinsic, np.float64).T
    safe_z = np.where(np.abs(projected[:, 2]) > 1e-12, projected[:, 2], 1e-12)
    columns = np.rint(projected[:, 0] / safe_z).astype(np.int64)
    rows = np.rint(projected[:, 1] / safe_z).astype(np.int64)
    height, width = target_depth.shape
    inside = (
        (camera[:, 2] > 1e-4)
        & (columns >= 0)
        & (columns < width)
        & (rows >= 0)
        & (rows < height)
    )
    indices = np.flatnonzero(inside)
    if not len(indices):
        empty = np.empty((0, 3), np.float64)
        return empty, empty.copy(), empty.copy(), empty.copy()
    rows_inside = rows[indices]
    columns_inside = columns[indices]
    valid = target["valid"][rows_inside, columns_inside]
    target_z = target_depth[rows_inside, columns_inside]
    valid &= np.isfinite(target_z) & (target_z > 1e-3)
    valid &= np.abs(camera[indices, 2] - target_z) <= maximum_residual_m
    indices = indices[valid]
    rows_inside = rows[indices]
    columns_inside = columns[indices]
    if not len(indices):
        empty = np.empty((0, 3), np.float64)
        return empty, empty.copy(), empty.copy(), empty.copy()
    target_camera = _camera_points(
        target_depth, target_intrinsic, rows_inside, columns_inside
    )
    target_world, target_rays = _world_points(target_camera, target_extrinsic)
    return (
        source["world"][indices],
        target_world,
        source["rays"][indices],
        target_rays,
    )


def _build_frame_correspondences(
    config: dict,
    part: str,
    frame: int,
    gauge: dict | None,
    args: argparse.Namespace,
    *,
    allowed_pairs: set[tuple[int, int]] | None = None,
) -> tuple[list[RigCorrespondence], dict[str, Any], np.ndarray]:
    timestamp = f"{frame:06d}"
    recon = load_recon(config, timestamp, backend=config["recon_backend"])
    depth = np.asarray(recon["depth"], np.float32)
    if gauge is not None:
        depth = apply_depth_gauge(depth, gauge, timestamp)
    masks = load_palette_masks(
        config["masks_dir"],
        timestamp,
        config["parts"],
        recon["depth_hw"],
        views=config["views"],
        part_ids=config.get("part_ids"),
        resize_mode=str(config.get("quality_cloud", {}).get("mask_resize_mode", "nearest")),
        coverage_threshold=float(
            config.get("quality_cloud", {}).get("mask_coverage_threshold", 0.25)
        ),
    )[part]
    samples = []
    for view in range(len(config["views"])):
        samples.append(_view_samples(
            depth[view],
            recon["conf"][view],
            recon["intrinsics"][view],
            recon["extrinsics"][view],
            masks[view],
            mask_erode=args.mask_erode,
            depth_edge_m=args.depth_edge_mm / 1000.0,
            confidence_quantile=args.confidence_quantile,
            stride=args.stride,
            max_points=args.max_points_per_view,
            seed=frame * 101 + view,
        ))
    batches = []
    pair_counts = {}
    rng = np.random.default_rng(frame)
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            pair = (left, right)
            if allowed_pairs is not None and pair not in allowed_pairs:
                continue
            directions = []
            directions.append(_directed_matches(
                samples[left], samples[right], depth[right],
                recon["intrinsics"][right], recon["extrinsics"][right],
                maximum_residual_m=args.maximum_initial_residual_mm / 1000.0,
            ))
            reverse = _directed_matches(
                samples[right], samples[left], depth[left],
                recon["intrinsics"][left], recon["extrinsics"][left],
                maximum_residual_m=args.maximum_initial_residual_mm / 1000.0,
            )
            directions.append((reverse[1], reverse[0], reverse[3], reverse[2]))
            usable = [item for item in directions if len(item[0])]
            if not usable:
                pair_counts[pair] = 0
                continue
            arrays = [np.concatenate([item[index] for item in usable], axis=0)
                      for index in range(4)]
            count = len(arrays[0])
            if count > args.max_correspondences_per_pair_frame:
                selected = rng.choice(
                    count, args.max_correspondences_per_pair_frame, replace=False
                )
                arrays = [value[selected] for value in arrays]
            pair_counts[pair] = int(len(arrays[0]))
            batches.append(RigCorrespondence(
                view_a=left,
                view_b=right,
                points_a=arrays[0],
                points_b=arrays[1],
                rays_a=arrays[2],
                rays_b=arrays[3],
                frame=frame,
            ))
    report = {
        "sampled_points_by_view": [int(len(item["world"])) for item in samples],
        "pair_correspondences": {
            f"{left}-{right}": int(count)
            for (left, right), count in sorted(pair_counts.items())
        },
    }
    return batches, report, np.asarray(recon["extrinsics"], np.float64)


def _collect(
    config: dict,
    part: str,
    frames: list[int],
    gauge: dict | None,
    args: argparse.Namespace,
    *,
    allowed_pairs: set[tuple[int, int]] | None = None,
) -> tuple[list[RigCorrespondence], dict[str, Any], np.ndarray]:
    batches = []
    reports = {}
    reference_extrinsics = None
    for position, frame in enumerate(frames):
        current, report, extrinsics = _build_frame_correspondences(
            config, part, frame, gauge, args, allowed_pairs=allowed_pairs
        )
        if reference_extrinsics is None:
            reference_extrinsics = extrinsics
        elif not np.allclose(reference_extrinsics, extrinsics, atol=1e-5, rtol=0):
            raise RuntimeError(
                f"frame {frame:06d} does not use the same fixed camera rig"
            )
        batches.extend(current)
        reports[f"{frame:06d}"] = report
        total = sum(len(batch.points_a) for batch in batches)
        if total > MAX_TOTAL_CORRESPONDENCES:
            raise RuntimeError(
                f"correspondence hard limit exceeded: {total} > "
                f"{MAX_TOTAL_CORRESPONDENCES}"
            )
        print(
            f"[{position + 1}/{len(frames)}] {frame:06d}: "
            f"{sum(len(item.points_a) for item in current)} matches",
            flush=True,
        )
    assert reference_extrinsics is not None
    return batches, reports, reference_extrinsics


def _edge_counts(
    correspondences: list[RigCorrespondence],
) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for batch in correspondences:
        counts[tuple(sorted((batch.view_a, batch.view_b)))] += len(batch.points_a)
    return dict(counts)


def _metric_bundle(
    train: list[RigCorrespondence],
    validation: list[RigCorrespondence],
    depth: np.ndarray,
    deltas: np.ndarray,
    threshold_m: float,
) -> dict[str, Any]:
    return {
        "train": correspondence_metrics(
            train, depth, deltas, inlier_threshold_m=threshold_m
        ),
        "validation": correspondence_metrics(
            validation, depth, deltas, inlier_threshold_m=threshold_m
        ),
    }


def main() -> None:
    require_memory_guard("tools/diagnostics/refine_camera_rig_from_static_part.py")
    args = _parser().parse_args()
    config = load_json(args.config.resolve())
    if args.part not in config["parts"]:
        raise ValueError(f"unknown part: {args.part}")
    views = [str(value) for value in config["views"]]
    _validate_args(args, len(views))
    gauge = load_depth_gauge(str(args.depth_gauge.resolve())) if args.depth_gauge else None

    train, train_frames, original_extrinsics = _collect(
        config, args.part, args.train_frames, gauge, args
    )
    raw_edges = _edge_counts(train)
    retained_edges = {
        pair: count for pair, count in raw_edges.items()
        if count >= args.minimum_pair_correspondences
    }
    components = connected_components(len(views), retained_edges)
    active_views = components[0]
    if len(active_views) < 2:
        raise RuntimeError("no connected camera pair has enough correspondences")
    anchor = choose_anchor(active_views, retained_edges)
    retained_pairs = {
        pair for pair in retained_edges
        if pair[0] in active_views and pair[1] in active_views
    }
    train = [
        batch for batch in train
        if tuple(sorted((batch.view_a, batch.view_b))) in retained_pairs
    ]
    validation, validation_frames, validation_extrinsics = _collect(
        config,
        args.part,
        args.validation_frames,
        gauge,
        args,
        allowed_pairs=retained_pairs,
    )
    combined_correspondences = sum(len(item.points_a) for item in train) + sum(
        len(item.points_a) for item in validation
    )
    if combined_correspondences > MAX_TOTAL_CORRESPONDENCES:
        raise RuntimeError(
            "combined training/validation correspondence hard limit exceeded: "
            f"{combined_correspondences} > {MAX_TOTAL_CORRESPONDENCES}"
        )
    if not np.allclose(original_extrinsics, validation_extrinsics, atol=1e-5, rtol=0):
        raise RuntimeError("training and validation do not share one fixed rig")

    view_count = len(views)
    zero_depth = np.zeros(view_count, np.float64)
    identity = np.tile(np.eye(4, dtype=np.float64), (view_count, 1, 1))
    baseline = _metric_bundle(
        train, validation, zero_depth, identity,
        args.inlier_threshold_mm / 1000.0,
    )
    depth_correction, depth_status = optimize_depth_corrections(
        train,
        view_count=view_count,
        active_views=active_views,
        anchor=anchor,
        data_sigma_m=args.data_sigma_mm / 1000.0,
        prior_sigma_m=args.depth_prior_mm / 1000.0,
        maximum_correction_m=args.maximum_depth_correction_mm / 1000.0,
        max_nfev=args.max_nfev,
    )
    depth_only = _metric_bundle(
        train, validation, depth_correction, identity,
        args.inlier_threshold_mm / 1000.0,
    )
    deltas, pose_status = optimize_pose_corrections(
        train,
        depth_correction,
        view_count=view_count,
        active_views=active_views,
        anchor=anchor,
        data_sigma_m=args.data_sigma_mm / 1000.0,
        rotation_prior_deg=args.rotation_prior_deg,
        translation_prior_m=args.translation_prior_mm / 1000.0,
        maximum_rotation_deg=args.maximum_rotation_deg,
        maximum_translation_m=args.maximum_translation_mm / 1000.0,
        max_nfev=args.max_nfev,
    )
    refined = _metric_bundle(
        train, validation, depth_correction, deltas,
        args.inlier_threshold_mm / 1000.0,
    )
    corrected = corrected_extrinsics(original_extrinsics, deltas)

    corrections = {}
    hit_bound = False
    for index, view in enumerate(views):
        rotation_deg = float(
            np.degrees(np.linalg.norm(
                cv2.Rodrigues(deltas[index, :3, :3])[0].reshape(3)
            ))
        )
        translation_mm = float(np.linalg.norm(deltas[index, :3, 3]) * 1000.0)
        depth_mm = float(depth_correction[index] * 1000.0)
        at_bound = bool(
            rotation_deg >= 0.98 * args.maximum_rotation_deg
            or translation_mm >= 0.98 * args.maximum_translation_mm
            or abs(depth_mm) >= 0.98 * args.maximum_depth_correction_mm
        )
        hit_bound |= at_bound
        corrections[view] = {
            "active": bool(index in active_views),
            "anchor": bool(index == anchor),
            "depth_correction_mm": depth_mm,
            "rotation_correction_deg": rotation_deg,
            "translation_correction_mm": translation_mm,
            "hit_bound": at_bound,
            "delta_world": deltas[index].tolist(),
            "original_extrinsic_world_to_camera": original_extrinsics[index].tolist(),
            "corrected_extrinsic_world_to_camera": corrected[index].tolist(),
        }

    base_valid = baseline["validation"]
    depth_valid = depth_only["validation"]
    refined_valid = refined["validation"]
    extrinsic_median_gain = (
        1.0 - refined_valid["median_m"] / depth_valid["median_m"]
        if depth_valid["median_m"] else None
    )
    extrinsic_p90_gain = (
        1.0 - refined_valid["p90_m"] / depth_valid["p90_m"]
        if depth_valid["p90_m"] else None
    )
    full_graph = len(active_views) == view_count
    recommend_replacement = bool(
        full_graph
        and not hit_bound
        and pose_status["success"]
        and extrinsic_median_gain is not None
        and extrinsic_p90_gain is not None
        and extrinsic_median_gain >= 0.10
        and extrinsic_p90_gain >= 0.05
    )
    report = {
        "schema_version": 1,
        "method": "bounded_static_part_depth_then_se3_rig_refinement",
        "config": str(args.config.resolve()),
        "part": args.part,
        "depth_gauge": str(args.depth_gauge.resolve()) if args.depth_gauge else None,
        "views": views,
        "train_frames": args.train_frames,
        "validation_frames": args.validation_frames,
        "hard_limits": {
            "maximum_frames_per_split": MAX_FRAMES_PER_SPLIT,
            "maximum_views": MAX_VIEWS,
            "maximum_points_per_view": MAX_POINTS_PER_VIEW,
            "maximum_correspondences_per_pair_frame": (
                MAX_CORRESPONDENCES_PER_PAIR_FRAME
            ),
            "maximum_total_correspondences": MAX_TOTAL_CORRESPONDENCES,
        },
        "parameters": {
            key: value for key, value in vars(args).items()
            if key not in {"config", "output", "depth_gauge"}
        },
        "camera_graph": {
            "raw_edge_correspondences": {
                f"{left}-{right}": count
                for (left, right), count in sorted(raw_edges.items())
            },
            "retained_edge_correspondences": {
                f"{left}-{right}": count
                for (left, right), count in sorted(retained_edges.items())
            },
            "components": components,
            "active_view_indices": active_views,
            "active_views": [views[index] for index in active_views],
            "anchor_index": anchor,
            "anchor_view": views[anchor],
        },
        "correspondence_build": {
            "train": train_frames,
            "validation": validation_frames,
            "retained_train_samples": int(sum(len(item.points_a) for item in train)),
            "retained_validation_samples": int(
                sum(len(item.points_a) for item in validation)
            ),
        },
        "optimization": {
            "depth": depth_status,
            "pose": pose_status,
        },
        "metrics": {
            "baseline": baseline,
            "depth_only": depth_only,
            "depth_and_extrinsics": refined,
        },
        "validation_improvement": {
            "baseline_to_depth_median_fraction": (
                1.0 - depth_valid["median_m"] / base_valid["median_m"]
                if base_valid["median_m"] else None
            ),
            "baseline_to_refined_median_fraction": (
                1.0 - refined_valid["median_m"] / base_valid["median_m"]
                if base_valid["median_m"] else None
            ),
            "depth_to_extrinsics_median_fraction": extrinsic_median_gain,
            "depth_to_extrinsics_p90_fraction": extrinsic_p90_gain,
        },
        "corrections": corrections,
        "recommend_production_rig_replacement": recommend_replacement,
        "replacement_gate": {
            "full_camera_graph": full_graph,
            "no_parameter_hit_bound": not hit_bound,
            "optimizer_success": bool(pose_status["success"]),
            "minimum_heldout_median_gain_over_depth_only": 0.10,
            "minimum_heldout_p90_gain_over_depth_only": 0.05,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(
        "validation P50/P90 mm: "
        f"baseline {base_valid['median_m'] * 1000.0:.2f}/"
        f"{base_valid['p90_m'] * 1000.0:.2f}; "
        f"depth {depth_valid['median_m'] * 1000.0:.2f}/"
        f"{depth_valid['p90_m'] * 1000.0:.2f}; "
        f"refined {refined_valid['median_m'] * 1000.0:.2f}/"
        f"{refined_valid['p90_m'] * 1000.0:.2f}",
        flush=True,
    )
    print(
        "production rig replacement: "
        + ("recommended" if recommend_replacement else "not recommended"),
        flush=True,
    )
    print(f"report -> {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
