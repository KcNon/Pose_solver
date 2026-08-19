#!/usr/bin/env python
"""Build cross-view-supported per-part clouds without losing view identity."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.backproject_utils import load_palette_masks, load_recon_colors
from common.cloud_io import write_ply
from common.depth_gauge import apply_depth_gauge, load_depth_gauge
from common.io_utils import load_json, write_json
from common.multiview_quality import mask_area_quality
from common.normalized_recon import load_recon
from common.quality_cloud import (
    assign_cross_view_support,
    cross_view_consistency,
    fuse_supported_clouds,
    filter_centroid_consistent_views,
    prepare_view_clouds,
    quality_gate,
    reprojection_depth_consistency,
    supported_view_clouds,
)


def quality_cloud_root(config: dict, settings: dict) -> Path:
    configured = settings.get("point_cloud_root") or config.get(
        "quality_point_cloud_root"
    )
    if configured:
        return Path(configured).resolve()
    artifact_root = Path(
        config.get("point_cloud_output_root", config["output_root"])
    ).resolve()
    variant = str(settings.get(
        "variant", f"{config['recon_backend']}_quality"
    ))
    return artifact_root / "parts_ply" / variant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument(
        "--timestamps",
        nargs="+",
        help="optional sparse frame list for quality experiments",
    )
    parser.add_argument(
        "--depth-gauge",
        type=Path,
        help="override config depth_gauge_path for an isolated calibration run",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="override the configured quality-cloud root",
    )
    parser.add_argument(
        "--part-views",
        action="append",
        default=[],
        metavar="PART=VIEW,VIEW,...",
        help=(
            "restrict one part to a calibrated core view set; repeat for "
            "multiple parts"
        ),
    )
    parser.add_argument(
        "--part-min-support",
        action="append",
        default=[],
        metavar="PART=N",
        help="override the cross-view support count for one part",
    )
    parser.add_argument("--no-per-view", action="store_true")
    parser.add_argument(
        "--summary-name",
        default="quality_cloud_summary.json",
        help="summary filename under the quality-cloud root",
    )
    parser.add_argument(
        "--complete-name",
        default="complete.json",
        help="completion-marker filename under the quality-cloud root",
    )
    args = parser.parse_args()

    config_path = Path(args.pipeline).resolve()
    config = load_json(config_path)
    settings = dict(config.get("quality_cloud", {}))
    output = (
        args.output_root.resolve()
        if args.output_root is not None
        else quality_cloud_root(config, settings)
    )
    output.mkdir(parents=True, exist_ok=True)
    configured_frames = config.get("frames", {})
    start = int(
        configured_frames.get("start", 0) if args.start is None else args.start
    )
    end = int(
        configured_frames.get("end", start) if args.end is None else args.end
    )
    if args.timestamps and (args.start is not None or args.end is not None):
        raise ValueError("--timestamps cannot be combined with --start/--end")
    frame_numbers = (
        sorted({int(value) for value in args.timestamps})
        if args.timestamps
        else list(range(start, end + 1))
    )
    views = [str(view) for view in config["views"]]
    active_views_by_part = {
        str(part): [str(view) for view in selected]
        for part, selected in settings.get(
            "active_views_by_part", {}
        ).items()
    }
    for value in args.part_views:
        if "=" not in value:
            raise ValueError(
                f"--part-views must be PART=VIEW,VIEW,..., got {value!r}"
            )
        part, selected = value.split("=", 1)
        selected_views = [item for item in selected.split(",") if item]
        if part not in config["parts"]:
            raise ValueError(f"unknown part in --part-views: {part}")
        unknown = sorted(set(selected_views).difference(views))
        if unknown:
            raise ValueError(f"{part}: unknown active views {unknown}")
        if not selected_views:
            raise ValueError(f"{part}: active view set cannot be empty")
        active_views_by_part[part] = selected_views
    gauge_path = (
        str(args.depth_gauge.resolve())
        if args.depth_gauge is not None
        else config.get("depth_gauge_path")
    )
    gauge = load_depth_gauge(gauge_path) if gauge_path else None
    parameters = {
        "conf_quantile": float(settings.get("conf_quantile", 0.5)),
        "mask_resize_mode": str(settings.get("mask_resize_mode", "nearest")),
        "mask_coverage_threshold": float(
            settings.get("mask_coverage_threshold", 0.25)
        ),
        "mask_coverage_parts": [
            str(part) for part in settings.get("mask_coverage_parts", [])
        ],
        "mask_erode": int(settings.get("mask_erode", 1)),
        "mask_erode_by_part": {
            str(part): int(value)
            for part, value in settings.get("mask_erode_by_part", {}).items()
        },
        "depth_edge_m": float(settings.get("depth_edge_mm", 8.0)) / 1000.0,
        "depth_edge_m_by_part": {
            str(part): float(value) / 1000.0
            for part, value in settings.get("depth_edge_mm_by_part", {}).items()
        },
        "support_radius_m": float(
            settings.get("cross_view_support_mm", 10.0)
        ) / 1000.0,
        "min_support": int(settings.get("min_support", 1)),
        "min_support_by_part": {
            str(part): int(value)
            for part, value in settings.get("min_support_by_part", {}).items()
        },
        "retain_unsupported_fraction": float(
            settings.get("retain_unsupported_fraction", 0.1)
        ),
        "retain_unsupported_fraction_by_part": {
            str(part): float(value)
            for part, value in settings.get(
                "retain_unsupported_fraction_by_part", {}
            ).items()
        },
        "stride": int(settings.get("stride", 2)),
        "stride_by_part": {
            str(part): int(value)
            for part, value in settings.get("stride_by_part", {}).items()
        },
        "fusion_voxel_m": float(
            settings.get("fusion_voxel_mm", 0.0)
        ) / 1000.0,
        "fusion_voxel_m_by_part": {
            str(part): float(value) / 1000.0
            for part, value in settings.get(
                "fusion_voxel_mm_by_part", {}
            ).items()
        },
        "max_points": int(settings.get("max_points", 80000)),
        "minimum_mask_pixels": int(
            settings.get("minimum_mask_pixels", 800)
        ),
        "minimum_mask_pixels_by_part": {
            str(part): int(value)
            for part, value in settings.get(
                "minimum_mask_pixels_by_part", {}
            ).items()
        },
        "maximum_mask_area_ratio": float(
            settings.get("maximum_mask_area_ratio", 4.0)
        ),
        "minimum_fused_points": int(settings.get("minimum_fused_points", 300)),
        "minimum_fused_points_by_part": {
            str(part): int(value)
            for part, value in settings.get(
                "minimum_fused_points_by_part", {}
            ).items()
        },
        "minimum_supported_views": int(settings.get("minimum_supported_views", 3)),
        "minimum_supported_views_by_part": {
            str(part): int(value)
            for part, value in settings.get(
                "minimum_supported_views_by_part", {}
            ).items()
        },
        "allow_single_view_by_part": {
            str(part): bool(value)
            for part, value in settings.get(
                "allow_single_view_by_part", {}
            ).items()
        },
        "allow_reprojection_override_by_part": {
            str(part): bool(value)
            for part, value in settings.get(
                "allow_reprojection_override_by_part", {}
            ).items()
        },
        "maximum_view_centroid_distance_m_by_part": {
            str(part): float(value) / 1000.0
            for part, value in settings.get(
                "maximum_view_centroid_distance_mm_by_part", {}
            ).items()
        },
        "minimum_view_centroid_points_by_part": {
            str(part): int(value)
            for part, value in settings.get(
                "minimum_view_centroid_points_by_part", {}
            ).items()
        },
        "minimum_supported_points_per_view": int(
            settings.get("minimum_supported_points_per_view", 30)
        ),
        "minimum_support_fraction": float(
            settings.get("minimum_support_fraction", 0.02)
        ),
        "maximum_cross_view_median_m": float(
            settings.get("maximum_cross_view_median_mm", 40.0)
        ) / 1000.0,
        "minimum_cross_view_overlap_ratio": float(
            settings.get("minimum_cross_view_overlap_ratio", 0.10)
        ),
        "maximum_reprojection_median_m": float(
            settings.get("maximum_reprojection_median_mm", 40.0)
        ) / 1000.0,
        "minimum_reprojection_inlier_ratio": float(
            settings.get("minimum_reprojection_inlier_ratio", 0.10)
        ),
        "active_views_by_part": active_views_by_part,
    }
    if parameters["stride"] < 1 or any(
        value < 1 for value in parameters["stride_by_part"].values()
    ):
        raise ValueError("quality_cloud stride values must be at least 1")
    for value in args.part_min_support:
        if "=" not in value:
            raise ValueError(
                f"--part-min-support must be PART=N, got {value!r}"
            )
        part, count = value.split("=", 1)
        if part not in config["parts"]:
            raise ValueError(
                f"unknown part in --part-min-support: {part}"
            )
        parsed_count = int(count)
        if parsed_count < 0:
            raise ValueError("part minimum support must be non-negative")
        parameters["min_support_by_part"][part] = parsed_count
    summary = {
        "schema_version": 2,
        "method": "per_view_depth_edge_filter_cross_view_support",
        "config": str(config_path),
        "depth_gauge": gauge_path,
        "output_root": str(output),
        "views": views,
        "parameters": parameters,
        "frames": {},
    }
    part_starts = config.get("part_start_frames", {})
    for frame_position, frame in enumerate(frame_numbers, start=1):
        timestamp = f"{frame:06d}"
        recon = load_recon(
            config, timestamp, backend=config["recon_backend"]
        )
        depth = np.asarray(recon["depth"], dtype=np.float32)
        if gauge is not None:
            depth = apply_depth_gauge(depth, gauge, timestamp)
        colors = load_recon_colors(recon, config, timestamp)
        masks = load_palette_masks(
            config["masks_dir"],
            timestamp,
            config["parts"],
            recon["depth_hw"],
            views=views,
            part_ids=config.get("part_ids"),
            resize_mode=parameters["mask_resize_mode"],
            coverage_threshold=parameters["mask_coverage_threshold"],
            coverage_parts=parameters["mask_coverage_parts"] or None,
        )
        frame_dir = output / timestamp
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_report = {}
        for part_index, part in enumerate(config["parts"]):
            part_path = frame_dir / f"{part}.ply"
            view_dir = frame_dir / "views" / part
            if frame < int(part_starts.get(part, start)):
                part_path.unlink(missing_ok=True)
                if view_dir.exists():
                    for stale in view_dir.glob("*.ply"):
                        stale.unlink()
                frame_report[part] = {"status": "before_part_start"}
                continue
            raw_masks = {
                view: np.asarray(masks[part][index], dtype=np.uint8)
                for index, view in enumerate(views)
            }
            minimum_mask_pixels = int(
                parameters["minimum_mask_pixels_by_part"].get(
                    part, parameters["minimum_mask_pixels"]
                )
            )
            min_support = int(
                parameters["min_support_by_part"].get(
                    part, parameters["min_support"]
                )
            )
            retain_unsupported_fraction = float(
                parameters["retain_unsupported_fraction_by_part"].get(
                    part, parameters["retain_unsupported_fraction"]
                )
            )
            mask_quality = mask_area_quality(
                raw_masks,
                1,
                minimum_pixels=minimum_mask_pixels,
                maximum_area_ratio=parameters["maximum_mask_area_ratio"],
            )
            filtered_masks = [
                (
                    masks[part][index]
                    if mask_quality["views"][view]["valid"]
                    else np.zeros_like(masks[part][index], dtype=bool)
                )
                for index, view in enumerate(views)
            ]
            active_views = set(
                parameters["active_views_by_part"].get(part, views)
            )
            filtered_masks = [
                mask if view in active_views else np.zeros_like(mask)
                for view, mask in zip(views, filtered_masks)
            ]
            clouds = prepare_view_clouds(
                depth,
                colors,
                recon["intrinsics"],
                recon["extrinsics"],
                recon["conf"],
                filtered_masks,
                conf_quantile=parameters["conf_quantile"],
                mask_erode=parameters["mask_erode_by_part"].get(
                    part, parameters["mask_erode"]
                ),
                depth_edge_m=parameters["depth_edge_m_by_part"].get(
                    part, parameters["depth_edge_m"]
                ),
                stride=parameters["stride_by_part"].get(
                    part, parameters["stride"]
                ),
            )
            centroid_report = {"enabled": False}
            centroid_radius = parameters[
                "maximum_view_centroid_distance_m_by_part"
            ].get(part)
            if centroid_radius is not None:
                clouds, centroid_report = filter_centroid_consistent_views(
                    clouds,
                    radius_m=centroid_radius,
                    minimum_points=parameters[
                        "minimum_view_centroid_points_by_part"
                    ].get(part, parameters["minimum_supported_points_per_view"]),
                )
                selected_indices = set(
                    centroid_report["selected_view_indices"]
                )
                filtered_masks = [
                    mask if index in selected_indices else np.zeros_like(mask)
                    for index, mask in enumerate(filtered_masks)
                ]
            assign_cross_view_support(
                clouds, parameters["support_radius_m"]
            )
            points, point_colors, stats = fuse_supported_clouds(
                clouds,
                min_support=min_support,
                retain_unsupported_fraction=retain_unsupported_fraction,
                voxel_size_m=parameters[
                    "fusion_voxel_m_by_part"
                ].get(part, parameters["fusion_voxel_m"]),
                max_points=parameters["max_points"],
                seed=frame * 31 + part_index,
            )
            supported_clouds = supported_view_clouds(
                clouds, min_support=min_support
            )
            cross_report = cross_view_consistency(
                supported_clouds, parameters["support_radius_m"]
            )
            reprojection_report = reprojection_depth_consistency(
                supported_clouds,
                depth,
                recon["intrinsics"],
                recon["extrinsics"],
                filtered_masks,
                threshold_m=parameters["support_radius_m"],
            )
            gate = quality_gate(
                stats,
                cross_report,
                reprojection_report,
                minimum_fused_points=parameters[
                    "minimum_fused_points_by_part"
                ].get(part, parameters["minimum_fused_points"]),
                minimum_supported_views=parameters[
                    "minimum_supported_views_by_part"
                ].get(part, parameters["minimum_supported_views"]),
                minimum_supported_points_per_view=parameters[
                    "minimum_supported_points_per_view"
                ],
                minimum_support_fraction=parameters["minimum_support_fraction"],
                maximum_cross_view_median_m=parameters[
                    "maximum_cross_view_median_m"
                ],
                minimum_cross_view_overlap_ratio=parameters[
                    "minimum_cross_view_overlap_ratio"
                ],
                maximum_reprojection_median_m=parameters[
                    "maximum_reprojection_median_m"
                ],
                minimum_reprojection_inlier_ratio=parameters[
                    "minimum_reprojection_inlier_ratio"
                ],
                allow_single_view=parameters[
                    "allow_single_view_by_part"
                ].get(part, False),
                allow_reprojection_override=parameters[
                    "allow_reprojection_override_by_part"
                ].get(part, False),
            )
            if gate["passed"]:
                write_ply(part_path, points, point_colors)
            else:
                part_path.unlink(missing_ok=True)
            if not args.no_per_view:
                if view_dir.exists():
                    for stale in view_dir.glob("*.ply"):
                        stale.unlink()
                for view, cloud in zip(views, clouds):
                    if not len(cloud.points):
                        continue
                    view_dir.mkdir(parents=True, exist_ok=True)
                    write_ply(
                        view_dir / f"{view}.ply",
                        cloud.points,
                        cloud.colors,
                    )
            frame_report[part] = {
                "status": "ok" if gate["passed"] else "rejected_quality",
                "mask_quality": mask_quality,
                "view_centroid_consistency": centroid_report,
                **stats,
                "cross_view": cross_report,
                "reprojection_depth": reprojection_report,
                "quality_gate": gate,
            }
        summary["frames"][timestamp] = frame_report
        if frame_position % 10 == 0 or frame_position == len(frame_numbers):
            counts = " ".join(
                f"{part}={frame_report[part].get('fused_points', 0)}"
                for part in config["parts"]
            )
            print(
                f"quality cloud {frame_position}/{len(frame_numbers)} "
                f"{timestamp} {counts}",
                flush=True,
            )
    write_json(output / args.summary_name, summary)
    write_json(
        output / args.complete_name,
        {
            "status": "complete",
            "frames": [int(value) for value in frame_numbers],
        },
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
