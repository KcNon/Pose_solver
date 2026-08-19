#!/usr/bin/env python
"""Select a cross-frame-consistent core camera set for one static part.

The calibration is deliberately discrete.  It does not move cameras or alter
depth until the existing observations prove that a small, stable subset is
internally consistent across every requested anchor frame.  This prevents a
bad view from thickening the fused surface and gives an auditable alternative
to unconstrained extrinsic optimization.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.backproject_utils import load_palette_masks, load_recon_colors
from common.io_utils import load_json, write_json
from common.normalized_recon import load_recon
from common.quality_cloud import (
    ViewCloud,
    assign_cross_view_support,
    cross_view_consistency,
    fuse_supported_clouds,
    prepare_view_clouds,
    reprojection_depth_consistency,
    supported_view_clouds,
)


def _empty_view(candidate_pixels: int) -> ViewCloud:
    return ViewCloud(
        points=np.empty((0, 3), dtype=np.float64),
        colors=np.empty((0, 3), dtype=np.uint8),
        confidence=np.empty((0,), dtype=np.float32),
        support=np.empty((0,), dtype=np.int16),
        candidate_pixels=int(candidate_pixels),
    )


def _copy_view(cloud: ViewCloud) -> ViewCloud:
    return ViewCloud(
        points=np.asarray(cloud.points).copy(),
        colors=np.asarray(cloud.colors).copy(),
        confidence=np.asarray(cloud.confidence).copy(),
        support=np.zeros(len(cloud.points), dtype=np.int16),
        candidate_pixels=int(cloud.candidate_pixels),
    )


def evaluate_subset(
    cached_frames: dict[int, dict[str, Any]],
    subset: tuple[int, ...],
    *,
    support_radius_m: float,
    min_support: int,
    minimum_supported_points_per_view: int,
    minimum_fused_points: int,
) -> dict[str, Any]:
    active = set(subset)
    frame_rows: dict[str, Any] = {}
    valid = True
    for frame, cached in cached_frames.items():
        base_clouds: list[ViewCloud] = cached["clouds"]
        clouds = [
            _copy_view(cloud) if index in active
            else _empty_view(cloud.candidate_pixels)
            for index, cloud in enumerate(base_clouds)
        ]
        assign_cross_view_support(clouds, support_radius_m)
        points, _colors, stats = fuse_supported_clouds(
            clouds,
            min_support=min_support,
            retain_unsupported_fraction=0.0,
            max_points=100_000,
            seed=frame,
        )
        supported = supported_view_clouds(clouds, min_support=min_support)
        cross = cross_view_consistency(supported, support_radius_m)
        masks = [
            mask if index in active else np.zeros_like(mask)
            for index, mask in enumerate(cached["masks"])
        ]
        reprojection = reprojection_depth_consistency(
            supported,
            cached["depth"],
            cached["intrinsics"],
            cached["extrinsics"],
            masks,
            threshold_m=support_radius_m,
        )
        candidate_points = int(sum(
            len(cloud.points) for index, cloud in enumerate(base_clouds)
            if index in active
        ))
        supported_views = int(sum(
            len(cloud.points) >= minimum_supported_points_per_view
            for cloud in supported
        ))
        row = {
            "fused_points": int(len(points)),
            "candidate_points": candidate_points,
            "support_fraction": float(
                len(points) / max(candidate_points, 1)
            ),
            "supported_views": supported_views,
            "cross_view": cross,
            "reprojection_depth": reprojection,
        }
        frame_rows[f"{frame:06d}"] = row
        valid &= bool(
            len(points) >= minimum_fused_points
            and supported_views >= len(subset)
            and reprojection.get("median_m") is not None
            and reprojection.get("p90_m") is not None
        )

    medians = [
        row["reprojection_depth"]["median_m"]
        for row in frame_rows.values()
        if row["reprojection_depth"].get("median_m") is not None
    ]
    p90_values = [
        row["reprojection_depth"]["p90_m"]
        for row in frame_rows.values()
        if row["reprojection_depth"].get("p90_m") is not None
    ]
    support_values = [row["support_fraction"] for row in frame_rows.values()]
    # P50 is the primary geometric objective. P90 guards against a small good
    # overlap hiding a thick tail, while the weak support term breaks close
    # ties in favour of a denser core.
    selection_score = (
        float(np.mean(medians))
        + 0.25 * float(np.mean(p90_values))
        + 0.005 * (1.0 - float(np.mean(support_values)))
        if valid and medians and p90_values
        else float("inf")
    )
    return {
        "view_indices": list(subset),
        "valid": bool(valid),
        "selection_score": selection_score,
        "mean_reprojection_median_m": (
            float(np.mean(medians)) if medians else None
        ),
        "mean_reprojection_p90_m": (
            float(np.mean(p90_values)) if p90_values else None
        ),
        "mean_support_fraction": (
            float(np.mean(support_values)) if support_values else None
        ),
        "minimum_fused_points": min(
            row["fused_points"] for row in frame_rows.values()
        ),
        "frames": frame_rows,
    }


def _pair_matrix(
    candidate: dict[str, Any],
    view_count: int,
) -> np.ndarray:
    totals = np.zeros((view_count, view_count), dtype=np.float64)
    weights = np.zeros((view_count, view_count), dtype=np.float64)
    for frame in candidate["frames"].values():
        for row in frame["reprojection_depth"].get("pairs", []):
            source, target = [int(value) for value in row["views"]]
            weight = max(1, int(row.get("samples", 1)))
            totals[source, target] += float(row["median_m"]) * weight
            weights[source, target] += weight
    matrix = np.full((view_count, view_count), np.nan, dtype=np.float64)
    valid = weights > 0
    matrix[valid] = totals[valid] / weights[valid]
    return matrix


def _draw_heatmap(
    matrix: np.ndarray,
    views: list[str],
    title: str,
    *,
    maximum_mm: float = 30.0,
) -> np.ndarray:
    cell = 112
    left, top = 170, 100
    width = left + cell * len(views) + 30
    height = top + cell * len(views) + 45
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)
    cv2.putText(
        canvas, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (245, 245, 245), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "source view -> target view | median depth residual (mm)",
        (20, 68), cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (175, 175, 175), 1, cv2.LINE_AA,
    )
    for index, view in enumerate(views):
        short = view[-6:]
        cv2.putText(
            canvas, short, (left + index * cell + 18, top - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas, short, (18, top + index * cell + 62),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1,
            cv2.LINE_AA,
        )
        for target in range(len(views)):
            x0 = left + target * cell
            y0 = top + index * cell
            value = matrix[index, target]
            if index == target:
                color = (55, 55, 55)
                label = "self"
            elif not np.isfinite(value):
                color = (30, 30, 30)
                label = "off"
            else:
                normalized = int(np.clip(value * 1000.0 / maximum_mm, 0, 1) * 255)
                color = tuple(int(v) for v in cv2.applyColorMap(
                    np.asarray([[normalized]], dtype=np.uint8),
                    cv2.COLORMAP_TURBO,
                )[0, 0])
                label = f"{value * 1000.0:.1f}"
            cv2.rectangle(
                canvas, (x0, y0), (x0 + cell - 3, y0 + cell - 3),
                color, -1,
            )
            cv2.putText(
                canvas, label, (x0 + 25, y0 + 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
    return canvas


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    summary = {
        key: candidate[key]
        for key in (
            "view_indices",
            "valid",
            "selection_score",
            "mean_reprojection_median_m",
            "mean_reprojection_p90_m",
            "mean_support_fraction",
            "minimum_fused_points",
        )
    }
    if not np.isfinite(float(summary["selection_score"])):
        summary["selection_score"] = None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--part", required=True)
    parser.add_argument("--minimum-core-views", type=int, default=4)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_json(args.config.resolve())
    if args.part not in cfg["parts"]:
        raise ValueError(f"unknown part: {args.part}")
    views = [str(value) for value in cfg["views"]]
    if not 3 <= args.minimum_core_views <= len(views):
        raise ValueError("minimum-core-views must be in [3, number of views]")
    settings = cfg.get("quality_cloud", {})
    support_radius_m = float(
        settings.get("cross_view_support_mm", 10.0)
    ) / 1000.0
    min_support = int(settings.get("min_support_by_part", {}).get(
        args.part, settings.get("min_support", 1)
    ))
    minimum_fused_points = int(
        settings.get("minimum_fused_points_by_part", {}).get(
            args.part, settings.get("minimum_fused_points", 300)
        )
    )
    minimum_supported_points_per_view = int(
        settings.get("minimum_supported_points_per_view", 30)
    )

    cached_frames: dict[int, dict[str, Any]] = {}
    for frame in args.frames:
        timestamp = f"{frame:06d}"
        recon = load_recon(cfg, timestamp, backend=cfg["recon_backend"])
        masks = load_palette_masks(
            cfg["masks_dir"], timestamp, cfg["parts"], recon["depth_hw"],
            views=views, part_ids=cfg.get("part_ids"),
        )[args.part]
        colors = load_recon_colors(recon, cfg, timestamp)
        clouds = prepare_view_clouds(
            recon["depth"], colors, recon["intrinsics"],
            recon["extrinsics"], recon["conf"], masks,
            conf_quantile=float(settings.get("conf_quantile", 0.5)),
            mask_erode=int(settings.get("mask_erode", 1)),
            depth_edge_m=float(settings.get("depth_edge_mm", 8.0)) / 1000.0,
            stride=int(settings.get("stride", 2)),
        )
        cached_frames[int(frame)] = {
            "clouds": clouds,
            "masks": masks,
            "depth": recon["depth"],
            "intrinsics": recon["intrinsics"],
            "extrinsics": recon["extrinsics"],
        }

    candidates = []
    for count in range(args.minimum_core_views, len(views) + 1):
        for subset in combinations(range(len(views)), count):
            candidates.append(evaluate_subset(
                cached_frames,
                subset,
                support_radius_m=support_radius_m,
                min_support=min_support,
                minimum_supported_points_per_view=(
                    minimum_supported_points_per_view
                ),
                minimum_fused_points=minimum_fused_points,
            ))
    valid_candidates = [row for row in candidates if row["valid"]]
    if not valid_candidates:
        raise RuntimeError("no camera subset satisfies the configured quality gates")
    selected = min(valid_candidates, key=lambda row: row["selection_score"])
    baseline = next(
        row for row in candidates if len(row["view_indices"]) == len(views)
    )
    for row in candidates:
        row["views"] = [views[index] for index in row["view_indices"]]

    args.output_root.mkdir(parents=True, exist_ok=True)
    before = _draw_heatmap(
        _pair_matrix(baseline, len(views)), views, "Before: all configured views"
    )
    after = _draw_heatmap(
        _pair_matrix(selected, len(views)), views,
        "After: automatically selected stable core views",
    )
    heatmap_path = args.output_root / "reprojection_residual_before_after.jpg"
    cv2.imwrite(str(heatmap_path), np.concatenate((before, after), axis=1))

    baseline_median = float(baseline["mean_reprojection_median_m"])
    selected_median = float(selected["mean_reprojection_median_m"])
    baseline_p90 = float(baseline["mean_reprojection_p90_m"])
    selected_p90 = float(selected["mean_reprojection_p90_m"])
    report = {
        "schema_version": 1,
        "method": "exhaustive_static_anchor_core_view_selection",
        "config": str(args.config.resolve()),
        "part": args.part,
        "frames": [int(value) for value in args.frames],
        "views": views,
        "minimum_core_views": int(args.minimum_core_views),
        "minimum_fused_points": minimum_fused_points,
        "selected_views": selected["views"],
        "baseline": _candidate_summary(baseline),
        "selected": _candidate_summary(selected),
        "improvement": {
            "reprojection_median_reduction_fraction": float(
                1.0 - selected_median / baseline_median
            ),
            "reprojection_p90_reduction_fraction": float(
                1.0 - selected_p90 / baseline_p90
            ),
        },
        "ranked_candidates": [
            {
                **_candidate_summary(row),
                "views": row["views"],
            }
            for row in sorted(candidates, key=lambda row: row["selection_score"])
        ],
        "heatmap": str(heatmap_path.resolve()),
    }
    report_path = args.output_root / "core_view_calibration.json"
    write_json(report_path, report)
    print("selected views: " + ", ".join(selected["views"]), flush=True)
    print(
        "reprojection P50 "
        f"{baseline_median * 1000.0:.2f} -> {selected_median * 1000.0:.2f} mm; "
        "P90 "
        f"{baseline_p90 * 1000.0:.2f} -> {selected_p90 * 1000.0:.2f} mm",
        flush=True,
    )
    print(f"report -> {report_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
