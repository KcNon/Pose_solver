#!/usr/bin/env python
"""Measure point-cloud density separately from multi-view geometric quality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.cloud_io import read_ply_xyz
from common.io_utils import load_json, write_json
from common.mask_io import frame_path
from common.normalized_recon import load_recon


def _quantiles(values: list[float]) -> dict | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], float)
    if not len(finite):
        return None
    p10, median, p90 = np.quantile(finite, [0.1, 0.5, 0.9])
    return {
        "count": int(len(finite)),
        "p10": float(p10),
        "median": float(median),
        "p90": float(p90),
    }


def point_metrics(
    points: np.ndarray,
    *,
    max_nn_points: int,
    seed: int,
    workers: int = 4,
) -> dict:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    result = {"point_count": int(len(points))}
    if not len(points):
        return result
    extent = points.max(axis=0) - points.min(axis=0)
    result["bbox_extent_mm"] = (extent * 1000.0).tolist()
    result["bbox_diagonal_mm"] = float(np.linalg.norm(extent) * 1000.0)
    if len(points) < 2:
        return result
    rng = np.random.default_rng(seed)
    if len(points) > max_nn_points:
        query = points[rng.choice(len(points), max_nn_points, replace=False)]
    else:
        query = points
    if not 1 <= int(workers) <= 32:
        raise ValueError("nearest-neighbour workers must be in [1, 32]")
    distances, _ = cKDTree(points).query(
        query, k=2, workers=int(workers)
    )
    spacing = distances[:, 1] * 1000.0
    result["nearest_neighbor_mm"] = {
        "samples": int(len(spacing)),
        "p10": float(np.quantile(spacing, 0.1)),
        "median": float(np.median(spacing)),
        "p90": float(np.quantile(spacing, 0.9)),
    }
    return result


def quality_root(config: dict) -> Path:
    settings = config.get("quality_cloud", {})
    configured = settings.get("point_cloud_root") or config.get(
        "quality_point_cloud_root"
    )
    if configured:
        return Path(configured).resolve()
    artifact_root = Path(
        config.get("point_cloud_output_root", config["output_root"])
    ).resolve()
    variant = settings.get(
        "variant", f"{config['recon_backend']}_quality"
    )
    return artifact_root / "parts_ply" / str(variant)


def select_timestamps(config: dict, requested: list[str], count: int) -> list[str]:
    if requested:
        return [f"{int(value):06d}" for value in requested]
    frames = config.get("frames", {})
    start = int(frames.get("start", 0))
    end = int(frames.get("end", start))
    available = np.arange(start, end + 1, dtype=int)
    if len(available) <= count:
        selected = available
    else:
        selected = available[np.unique(np.linspace(
            0, len(available) - 1, count, dtype=int
        ))]
    return [f"{int(value):06d}" for value in selected]


def markdown_report(report: dict) -> str:
    source = report["source"]
    lines = [
        "# Point-cloud quality report",
        "",
        f"- Config: `{report['config']}`",
        f"- Source image: {source['source_hw'][1]}x{source['source_hw'][0]}",
        (
            f"- DA3 depth: {source['depth_hw'][1]}x{source['depth_hw'][0]} "
            f"({100.0 * source['depth_to_source_area_ratio']:.2f}% of source pixels)"
        ),
        f"- Views: {source['view_count']}",
        "",
        "| Part | Frames | Points median | NN median (mm) | Cross-view median (mm) | Reprojection median (mm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for part, row in report["aggregate_by_part"].items():
        def median(name: str) -> str:
            value = row.get(name)
            return "-" if value is None else f"{value['median']:.3f}"
        lines.append(
            f"| {part} | {row['frames']} | {median('point_count')} | "
            f"{median('nearest_neighbor_median_mm')} | "
            f"{median('cross_view_median_mm')} | "
            f"{median('reprojection_median_mm')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--quality-root", type=Path)
    parser.add_argument("--timestamps", nargs="*", default=[])
    parser.add_argument("--parts", nargs="*")
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--max-nn-points", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    timestamps = select_timestamps(
        config, args.timestamps, max(1, int(args.sample_count))
    )
    parts = [str(value) for value in (args.parts or config["parts"])]
    cloud_root = (
        args.quality_root.resolve()
        if args.quality_root is not None
        else quality_root(config)
    )
    summary_path = cloud_root / "quality_cloud_summary.json"
    quality_summary = (
        load_json(summary_path) if summary_path.exists() else {"frames": {}}
    )

    first_timestamp = timestamps[0]
    recon = load_recon(config, first_timestamp, backend=config["recon_backend"])
    source_path = frame_path(
        config["frames_dir"],
        config.get("frames_layout", "normalized"),
        first_timestamp,
        str(config["views"][0]),
    )
    with Image.open(source_path) as image:
        source_hw = (int(image.height), int(image.width))
    depth_hw = tuple(int(value) for value in recon["depth_hw"])
    source = {
        "source_hw": list(source_hw),
        "depth_hw": list(depth_hw),
        "view_count": int(recon["n_views"]),
        "raw_depth_points_per_view": int(np.prod(depth_hw)),
        "raw_depth_points_all_views": int(np.prod(depth_hw) * recon["n_views"]),
        "depth_to_source_area_ratio": float(
            np.prod(depth_hw) / np.prod(source_hw)
        ),
    }
    artifact_path = Path(recon["path"])
    with np.load(artifact_path) as artifact:
        source["artifact_metadata"] = {
            key: np.asarray(artifact[key]).tolist()
            for key in (
                "pose_solver_depth_schema_version",
                "process_res",
                "process_res_method",
                "source_image_hw",
                "processed_image_hw",
                "camera_frames",
                "model_dir",
                "use_ray_pose",
                "ref_view_strategy",
            )
            if key in artifact.files
        }

    frame_rows = {}
    accumulators = {
        part: {
            "point_count": [],
            "nearest_neighbor_median_mm": [],
            "cross_view_median_mm": [],
            "reprojection_median_mm": [],
        }
        for part in parts
    }
    for frame_index, timestamp in enumerate(timestamps):
        frame_report = {}
        summary_frame = quality_summary.get("frames", {}).get(timestamp, {})
        for part_index, part in enumerate(parts):
            path = cloud_root / timestamp / f"{part}.ply"
            if not path.exists():
                frame_report[part] = {
                    "status": summary_frame.get(part, {}).get(
                        "status", "missing_cloud"
                    )
                }
                continue
            metrics = point_metrics(
                read_ply_xyz(path),
                max_nn_points=int(args.max_nn_points),
                seed=frame_index * 101 + part_index,
                workers=int(args.workers),
            )
            quality = summary_frame.get(part, {})
            cross = (quality.get("cross_view") or {}).get("median_m")
            reprojection = (quality.get("reprojection_depth") or {}).get(
                "median_m"
            )
            metrics.update({
                "status": quality.get("status", "cloud_without_summary"),
                "cross_view_median_mm": (
                    None if cross is None else float(cross) * 1000.0
                ),
                "reprojection_median_mm": (
                    None if reprojection is None else float(reprojection) * 1000.0
                ),
                "view_point_counts": {
                    path.stem: int(len(read_ply_xyz(path)))
                    for path in sorted(
                        (cloud_root / timestamp / "views" / part).glob("*.ply")
                    )
                },
            })
            frame_report[part] = metrics
            accumulators[part]["point_count"].append(metrics["point_count"])
            nearest = metrics.get("nearest_neighbor_mm", {}).get("median")
            if nearest is not None:
                accumulators[part]["nearest_neighbor_median_mm"].append(nearest)
            if metrics["cross_view_median_mm"] is not None:
                accumulators[part]["cross_view_median_mm"].append(
                    metrics["cross_view_median_mm"]
                )
            if metrics["reprojection_median_mm"] is not None:
                accumulators[part]["reprojection_median_mm"].append(
                    metrics["reprojection_median_mm"]
                )
        frame_rows[timestamp] = frame_report

    aggregate = {}
    for part, values in accumulators.items():
        aggregate[part] = {
            "frames": len(values["point_count"]),
            **{name: _quantiles(samples) for name, samples in values.items()},
        }
    report = {
        "schema_version": 1,
        "config": str(config_path),
        "quality_root": str(cloud_root),
        "quality_summary": str(summary_path) if summary_path.exists() else None,
        "timestamps": timestamps,
        "parts": parts,
        "source": source,
        "frames": frame_rows,
        "aggregate_by_part": aggregate,
    }
    write_json(args.output, report)
    markdown_path = args.markdown or args.output.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
