#!/usr/bin/env python
"""Detect per-part motion states from multi-view masks, without manual ranges.

Per frame and view the part mask is reduced to a bbox center and an area.  The
bbox center, like the silhouette envelope used by the lid tracker, stays useful
when the middle of the object is occluded, unlike a mask centroid.  The motion
score is the median over views of the center displacement, combined with the
area log-ratio so motion toward a camera is not missed.  A hysteresis state
machine with minimum dwell times turns the noisy score into stable states:

    unobserved      no view sees the part
    occluded        visible, but silhouette support collapsed vs its typical size
    static / moving hysteresis on the motion score
    assembled       an explicitly configured child that approached its
                    assembly parent and stayed settled for a confirmation
                    window; the state remains latched through later occlusion

Detected moving ranges are compared against the manual ``dynamic_ranges`` in
the config so the detector can be judged before it replaces them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.cloud_io import read_ply_xyz
from common.multiview_quality import mask_area_quality


def median_filter(values: np.ndarray, width: int = 5) -> np.ndarray:
    half = width // 2
    padded = np.pad(values, half, mode="edge")
    result = []
    for i in range(len(values)):
        window = padded[i:i + width]
        result.append(np.nanmedian(window) if np.isfinite(window).any() else np.nan)
    return np.asarray(result)


def mask_observations(mask_root: Path, frames: list[str], views: list[str],
                      part_ids: dict[str, int], min_px: int,
                      maximum_area_ratio: float) -> dict:
    """Per part: bbox centers (F,V,2), areas (F,V); NaN/0 when below min_px."""
    parts = list(part_ids)
    centers = {p: np.full((len(frames), len(views), 2), np.nan) for p in parts}
    areas = {p: np.zeros((len(frames), len(views))) for p in parts}
    for fi, timestamp in enumerate(frames):
        labels_by_view = {
            view: np.asarray(Image.open(mask_root / timestamp / f"{view}.png"))
            for view in views
            if (mask_root / timestamp / f"{view}.png").exists()
        }
        qualities = {
            part: mask_area_quality(
                labels_by_view,
                pid,
                minimum_pixels=min_px,
                maximum_area_ratio=maximum_area_ratio,
            )
            for part, pid in part_ids.items()
        }
        for vi, view in enumerate(views):
            if view not in labels_by_view:
                continue
            labels = labels_by_view[view]
            for part, pid in part_ids.items():
                if not qualities[part]["views"][view]["valid"]:
                    continue
                y, x = np.where(labels == pid)
                areas[part][fi, vi] = len(x)
                centers[part][fi, vi] = [(x.min() + x.max()) / 2.0, (y.min() + y.max()) / 2.0]
        if fi % 20 == 0:
            print(f"masks {timestamp} ({fi + 1}/{len(frames)})", flush=True)
    return {"centers": centers, "areas": areas}


def rolling_median(values: np.ndarray, width: int = 11) -> np.ndarray:
    """Per-view rolling median over frames; zeros (invisible) are ignored."""
    half = width // 2
    out = np.zeros_like(values, dtype=float)
    for f in range(values.shape[0]):
        window = values[max(0, f - half):f + half + 1]
        for v in range(values.shape[1]):
            positive = window[:, v][window[:, v] > 0]
            out[f, v] = np.median(positive) if len(positive) else 0.0
    return out


def motion_score(centers: np.ndarray, areas: np.ndarray,
                 occlusion_drop: float = 0.7,
                 motion_lag: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Occlusion-gated median-over-views center displacement and area change.

    A view whose mask area collapses below ``occlusion_drop`` of its rolling
    median is being carved by an occluder (hand, another part); its silhouette
    shift is occlusion, not motion, so it is excluded from the vote.  Sustained
    changes re-enter the rolling median within a few frames and stay counted.
    """
    n_frames = centers.shape[0]
    typical = rolling_median(areas)
    valid = ~np.isnan(centers[:, :, 0]) & (areas >= occlusion_drop * typical) & (areas > 0)
    disp = np.full(n_frames, np.nan)
    dlog = np.full(n_frames, np.nan)
    lag = max(1, int(motion_lag))
    for f in range(1, n_frames):
        displacements = []
        area_changes = []
        for previous in sorted({f - 1, max(0, f - lag)}):
            both = valid[f] & valid[previous]
            if not both.any():
                continue
            displacements.append(
                float(np.median(np.linalg.norm(
                    centers[f, both] - centers[previous, both], axis=1
                )))
            )
            area_changes.append(
                float(np.median(np.abs(np.log(
                    areas[f, both] / areas[previous, both]
                ))))
            )
        if displacements:
            # A short-baseline vote preserves sudden-motion sensitivity while
            # the lagged vote catches slow deliberate assembly motion.
            disp[f] = max(displacements)
            dlog[f] = max(area_changes)
    disp[0], dlog[0] = disp[1], dlog[1]
    return median_filter(disp), median_filter(dlog)


def cloud_motion_mm(cloud_root: Path, frames: list[str], part: str,
                    max_points: int = 4000, seed: int = 0,
                    motion_lag: int = 3) -> np.ndarray:
    """Median NN distance (mm) from each frame's part cloud to the previous one.

    Pure visibility change keeps the visible points on the same physical
    surface, so the robust NN distance stays at the depth-noise floor; true
    rigid motion displaces the whole surface.  This is the signal that
    separates "an occluder settled onto the part" from "the part moved",
    which no 2D silhouette statistic can do.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)

    def load(timestamp: str) -> np.ndarray | None:
        path = cloud_root / timestamp / f"{part}.ply"
        if not path.exists():
            return None
        points = read_ply_xyz(str(path))
        if len(points) < 100:
            return None
        if len(points) > max_points:
            points = points[rng.choice(len(points), max_points, replace=False)]
        return points

    out = np.full(len(frames), np.nan)
    clouds: list[np.ndarray | None] = []
    lag = max(1, int(motion_lag))
    previous_valid: int | None = None
    for timestamp in frames:
        clouds.append(load(timestamp))
    for f in range(1, len(frames)):
        current = clouds[f]
        if current is None:
            continue
        comparisons = []
        indices = {f - 1, max(0, f - lag)}
        if previous_valid is not None:
            indices.add(previous_valid)
        for previous in sorted(indices):
            target = clouds[previous]
            if target is None:
                continue
            forward, _ = cKDTree(target).query(current, k=1)
            reverse, _ = cKDTree(current).query(target, k=1)
            # Visibility changes move a partial cloud's centroid and make one
            # directed NN residual large even when the physical part stayed
            # fixed. Real rigid motion raises both directed residuals. The
            # smaller median is therefore an occlusion-robust motion veto.
            surface_shift = min(
                float(np.median(forward)), float(np.median(reverse))
            ) * 1000.0
            comparisons.append(surface_shift)
        if comparisons:
            out[f] = max(comparisons)
        previous_valid = f
    out[0] = out[1]
    return median_filter(out, 3)


def hysteresis_states(disp: np.ndarray, dlog: np.ndarray, d3d: np.ndarray,
                      visible: np.ndarray, cfg: argparse.Namespace) -> list[str]:
    """static/moving with enter/exit dwell; unobserved frames keep the latch."""
    hot_2d = (disp > cfg.disp_hi) | (dlog > cfg.area_hi)
    cold_2d = (disp < cfg.disp_lo) & (dlog < cfg.area_lo)
    strong_2d = (
        (disp > cfg.disp_force_hi) | (dlog > cfg.area_force_hi)
    )
    # Strong multi-view silhouette motion can recover an in-place rotation for
    # which nearest-neighbour cloud displacement stays below ``surf_hi_mm``.
    # It must not, however, override a cloud that is still at the static noise
    # floor: hand occlusion and mask fragmentation can move a bbox by dozens of
    # pixels while the physical surface remains fixed.  The intermediate
    # ``surf_force_min_mm`` band preserves rotation sensitivity without
    # unlocking a previously static pose on contradictory evidence.
    surf_force_min_mm = float(getattr(
        cfg, "surf_force_min_mm", 0.5 * float(cfg.surf_lo_mm)
    ))
    cloud_available = np.isfinite(d3d)
    strong_2d_with_support = strong_2d & (
        ~cloud_available | (d3d > surf_force_min_mm)
    )
    hot = strong_2d_with_support | np.where(
        cloud_available, hot_2d & (d3d > cfg.surf_hi_mm), hot_2d
    )
    cold = ~hot & np.where(
        cloud_available, cold_2d | (d3d < cfg.surf_lo_mm), cold_2d
    )
    states = []
    moving = False
    run = 0
    for f in range(len(disp)):
        if visible[f] == 0:
            states.append("unobserved")
            run = 0
            continue
        if np.isnan(disp[f]):
            # Visible but every view is occlusion-corrupted: no measurement,
            # keep the kinematic latch instead of resetting it.
            states.append("moving" if moving else "static")
            run = 0
            continue
        if not moving:
            run = run + 1 if hot[f] else 0
            if run >= cfg.dwell_on:
                moving = True
                for back in range(1, cfg.dwell_on):
                    if states[-back] == "static":
                        states[-back] = "moving"
                run = 0
        else:
            run = run + 1 if cold[f] else 0
            if run >= cfg.dwell_off:
                moving = False
                for back in range(1, cfg.dwell_off):
                    if states[-back] == "moving":
                        states[-back] = "static"
                run = 0
        states.append("moving" if moving else "static")
    return states


def occlusion_flags(areas: np.ndarray, visible: np.ndarray, ratio: float) -> np.ndarray:
    """Silhouette support collapse: total area far below its typical value."""
    total = areas.sum(axis=1)
    typical = np.quantile(total[total > 0], 0.75) if (total > 0).any() else 0.0
    return (visible > 0) & (total < ratio * typical)


def cloud_centroid(cloud_root: Path, timestamp: str, part: str) -> np.ndarray | None:
    path = cloud_root / timestamp / f"{part}.ply"
    if not path.exists():
        return None
    points = read_ply_xyz(str(path))
    return points.mean(axis=0) if len(points) >= 30 else None


def assembly_latch_interval(
    states: list[str],
    distances: list[float | None],
    tol_m: float,
    *,
    minimum_stable_frames: int = 10,
    minimum_approach_m: float = 0.03,
) -> tuple[int, int] | None:
    """Return ``(settled_start, confirmation_frame)`` for one-way assembly.

    This is called only for a child with an explicit ``assembly_parent``.
    Requiring motion, a stable distance window, and a meaningful approach
    prevents a later lid/hand occlusion from being mistaken for a new motion.
    """

    if len(states) != len(distances):
        raise ValueError("assembly states and distances must have equal length")
    size = max(2, int(minimum_stable_frames))
    tolerance = float(tol_m)
    approach = float(minimum_approach_m)
    if tolerance <= 0.0 or approach < 0.0:
        raise ValueError("assembly thresholds are invalid")

    moving_seen = False
    pre_motion_distances: list[float] = []
    for index, state in enumerate(states):
        distance = distances[index]
        if state == "moving":
            moving_seen = True
            continue
        if not moving_seen:
            if distance is not None and np.isfinite(float(distance)):
                pre_motion_distances.append(float(distance))
            continue
        if index == 0 or states[index - 1] != "moving":
            continue
        stop = min(len(states), index + size)
        if stop - index < size:
            continue
        if any(value == "moving" for value in states[index:stop]):
            continue
        window = np.asarray([
            float(value)
            for value in distances[index:stop]
            if value is not None and np.isfinite(float(value))
        ], dtype=np.float64)
        if len(window) < max(3, size // 2):
            continue
        center = float(np.median(window))
        if float(np.max(np.abs(window - center))) > tolerance:
            continue
        if not pre_motion_distances:
            continue
        baseline = float(np.median(np.asarray(pre_motion_distances)))
        if baseline - center < approach:
            continue
        return index, stop - 1
    return None


def mark_assembled(
    states: list[str],
    distances: list[float | None],
    tol_m: float,
    *,
    minimum_stable_frames: int = 10,
    minimum_approach_m: float = 0.03,
) -> tuple[list[str], tuple[int, int] | None]:
    """Latch an explicitly configured assembled child after confirmation."""

    latch = assembly_latch_interval(
        states,
        distances,
        tol_m,
        minimum_stable_frames=minimum_stable_frames,
        minimum_approach_m=minimum_approach_m,
    )
    if latch is None:
        return list(states), None
    start, _ = latch
    out = list(states)
    out[start:] = ["assembled"] * (len(out) - start)
    return out, latch


def ranges_of(states: list[str], name: str, start: int) -> list[list[int]]:
    result = []
    for f, state in enumerate(states):
        if state != name:
            continue
        if result and result[-1][1] == start + f - 1:
            result[-1][1] = start + f
        else:
            result.append([start + f, start + f])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--min-px", type=int, default=800, help="min mask pixels for a view to count as visible")
    parser.add_argument("--disp-hi", type=float, default=6.0, help="px/frame to enter moving")
    parser.add_argument("--disp-lo", type=float, default=2.5, help="px/frame to exit moving")
    parser.add_argument(
        "--disp-force-hi", type=float, default=12.0,
        help="robust multi-view displacement that does not require a 3D vote",
    )
    parser.add_argument("--area-hi", type=float, default=0.10, help="|dlog area| to enter moving")
    parser.add_argument("--area-lo", type=float, default=0.04, help="|dlog area| to exit moving")
    parser.add_argument(
        "--area-force-hi", type=float, default=0.20,
        help="robust area change that does not require a 3D vote",
    )
    parser.add_argument("--surf-hi-mm", type=float, default=6.0, help="3D surface shift to enter moving")
    parser.add_argument("--surf-lo-mm", type=float, default=4.0, help="3D surface shift to exit moving")
    parser.add_argument(
        "--surf-force-min-mm", type=float, default=2.0,
        help=(
            "minimum 3D shift required for strong 2D motion to bypass the "
            "normal surface-enter threshold"
        ),
    )
    parser.add_argument("--dwell-on", type=int, default=2)
    parser.add_argument("--dwell-off", type=int, default=4)
    parser.add_argument(
        "--motion-lag", type=int, default=3,
        help="additional frame baseline used to detect slow assembly motion",
    )
    parser.add_argument(
        "--motion-baseline-seconds", type=float, default=0.4,
        help="FPS-aware lag when frames.fps is present in the pose config",
    )
    parser.add_argument("--dwell-on-seconds", type=float, default=0.25)
    parser.add_argument("--dwell-off-seconds", type=float, default=0.6)
    parser.add_argument("--occlusion-ratio", type=float, default=0.35)
    parser.add_argument("--assembled-tol-m", type=float, default=0.03)
    parser.add_argument("--assembled-minimum-stable-frames", type=int, default=10)
    parser.add_argument("--assembled-minimum-approach-m", type=float, default=0.03)
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    timeline_fps = float(cfg.get("frames", {}).get("fps", 0.0) or 0.0)
    if timeline_fps > 0.0:
        args.motion_lag = max(
            1, int(round(timeline_fps * args.motion_baseline_seconds))
        )
        args.dwell_on = max(
            1, int(round(timeline_fps * args.dwell_on_seconds))
        )
        args.dwell_off = max(
            1, int(round(timeline_fps * args.dwell_off_seconds))
        )
    views = cfg["views"]
    start, end = int(cfg["frames"]["start"]), int(cfg["frames"]["end"])
    frames = [f"{f:06d}" for f in range(start, end + 1)]
    part_ids = {p: int(cfg["part_ids"][p]) for p in cfg["parts"]}
    mask_root = Path(cfg["masks_dir"])
    cloud_root = Path(cfg.get(
        "point_cloud_root", Path(cfg["output_root"]) / "parts_ply" / cfg["recon_backend"]))
    configured_parents = {
        part: str(cfg.get("states", {}).get(part, {}).get("assembly_parent"))
        for part in cfg["parts"]
        if cfg.get("states", {}).get(part, {}).get("assembly_parent") is not None
    }
    unknown_parents = sorted(
        set(configured_parents.values()).difference(cfg["parts"])
    )
    if unknown_parents:
        raise ValueError(f"unknown assembly parents: {unknown_parents}")

    view_quality = cfg.get("view_quality", {})
    obs = mask_observations(
        mask_root,
        frames,
        views,
        part_ids,
        int(view_quality.get("minimum_full_mask_pixels", args.min_px)),
        float(view_quality.get("maximum_mask_area_ratio", 4.0)),
    )

    parent_centroids = {
        parent: [
            cloud_centroid(cloud_root, timestamp, parent)
            for timestamp in frames
        ]
        for parent in sorted(set(configured_parents.values()))
    }
    report = {
        "config": args.config,
        "frames": [start, end],
        "point_cloud_root": str(cloud_root),
        "point_cloud_variant": cfg.get("point_cloud_variant", cfg["recon_backend"]),
        "thresholds": vars(args),
        "parts": {},
    }
    for part in cfg["parts"]:
        centers, areas = obs["centers"][part], obs["areas"][part]
        visible = (~np.isnan(centers[:, :, 0])).sum(axis=1)
        disp, dlog = motion_score(
            centers, areas, motion_lag=args.motion_lag
        )
        d3d = cloud_motion_mm(
            cloud_root, frames, part, motion_lag=args.motion_lag
        )
        states = hysteresis_states(disp, dlog, d3d, visible, args)
        occluded = occlusion_flags(areas, visible, args.occlusion_ratio)
        states = [("occluded" if occluded[f] and states[f] == "static" else states[f])
                  for f in range(len(states))]
        parent = configured_parents.get(part)
        distances: list[float | None] = []
        for f, timestamp in enumerate(frames):
            child = cloud_centroid(cloud_root, timestamp, part) if parent else None
            parent_center = parent_centroids[parent][f] if parent else None
            distances.append(
                float(np.linalg.norm(child - parent_center))
                if child is not None and parent_center is not None
                else None
            )
        latch = None
        if parent is not None:
            latch_settings = cfg.get("states", {}).get(part, {}).get(
                "assembly_latch", {}
            )
            states, latch = mark_assembled(
                states,
                distances,
                float(latch_settings.get(
                    "distance_tolerance_m", args.assembled_tol_m
                )),
                minimum_stable_frames=int(latch_settings.get(
                    "minimum_stable_frames",
                    args.assembled_minimum_stable_frames,
                )),
                minimum_approach_m=float(latch_settings.get(
                    "minimum_approach_m", args.assembled_minimum_approach_m
                )),
            )

        detected = ranges_of(states, "moving", start)
        manual = cfg["states"][part].get("dynamic_ranges", [])
        report["parts"][part] = {
            "states": {frames[f]: {
                "state": states[f], "observing_views": int(visible[f]),
                "motion_px": None if np.isnan(disp[f]) else float(disp[f]),
                "surface_shift_mm": None if np.isnan(d3d[f]) else float(d3d[f]),
                "assembly_parent_distance_m": distances[f],
            } for f in range(len(frames))},
            "detected_moving_ranges": detected,
            "manual_dynamic_ranges": manual,
            "detected_assembled_from": next(
                (start + f for f, s in enumerate(states) if s == "assembled"), None),
            "detected_assembled_confirmed_at": (
                None if latch is None else start + int(latch[1])
            ),
            "assembly_parent": parent,
        }
        counts = {s: states.count(s) for s in sorted(set(states))}
        print(f"\n{part}: {counts}")
        print(f"  detected moving ranges: {detected}")
        print(f"  manual dynamic ranges:  {manual}")

    out = Path(cfg["output_root"]) / "diagnostics" / "part_states.json"
    write_json(out, report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
