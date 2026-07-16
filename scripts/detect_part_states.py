#!/usr/bin/env python
"""Detect per-part motion states from six-view masks, without manual ranges.

Per frame and view the part mask is reduced to a bbox center and an area.  The
bbox center, like the silhouette envelope used by the lid tracker, stays useful
when the middle of the object is occluded, unlike a mask centroid.  The motion
score is the median over views of the center displacement, combined with the
area log-ratio so motion toward a camera is not missed.  A hysteresis state
machine with minimum dwell times turns the noisy score into stable states:

    unobserved      no view sees the part
    occluded        visible, but silhouette support collapsed vs its typical size
    static / moving hysteresis on the motion score
    assembled       static after the last motion, near the final body-relative
                    offset (a proximity label, not a contact check)

Detected moving ranges are compared against the manual ``dynamic_ranges`` in
the config so the detector can be judged before it replaces them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.mesh_align import read_ply_xyz


def median_filter(values: np.ndarray, width: int = 5) -> np.ndarray:
    half = width // 2
    padded = np.pad(values, half, mode="edge")
    result = []
    for i in range(len(values)):
        window = padded[i:i + width]
        result.append(np.nanmedian(window) if np.isfinite(window).any() else np.nan)
    return np.asarray(result)


def mask_observations(mask_root: Path, frames: list[str], views: list[str],
                      part_ids: dict[str, int], min_px: int) -> dict:
    """Per part: bbox centers (F,V,2), areas (F,V); NaN/0 when below min_px."""
    parts = list(part_ids)
    centers = {p: np.full((len(frames), len(views), 2), np.nan) for p in parts}
    areas = {p: np.zeros((len(frames), len(views))) for p in parts}
    for fi, timestamp in enumerate(frames):
        for vi, view in enumerate(views):
            labels = np.asarray(Image.open(mask_root / timestamp / f"{view}.png"))
            for part, pid in part_ids.items():
                y, x = np.where(labels == pid)
                if len(x) < min_px:
                    continue
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
                 occlusion_drop: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
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
    for f in range(1, n_frames):
        both = valid[f] & valid[f - 1]
        if not both.any():
            continue
        disp[f] = np.median(np.linalg.norm(centers[f, both] - centers[f - 1, both], axis=1))
        dlog[f] = np.median(np.abs(np.log(areas[f, both] / areas[f - 1, both])))
    disp[0], dlog[0] = disp[1], dlog[1]
    return median_filter(disp), median_filter(dlog)


def cloud_motion_mm(cloud_root: Path, frames: list[str], part: str,
                    max_points: int = 4000, seed: int = 0) -> np.ndarray:
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
    previous = load(frames[0])
    for f in range(1, len(frames)):
        current = load(frames[f])
        if current is not None and previous is not None:
            distances, _ = cKDTree(previous).query(current, k=1)
            out[f] = float(np.median(distances)) * 1000.0
        if current is not None:
            previous = current
    out[0] = out[1]
    return median_filter(out, 3)


def hysteresis_states(disp: np.ndarray, dlog: np.ndarray, d3d: np.ndarray,
                      visible: np.ndarray, cfg: argparse.Namespace) -> list[str]:
    """static/moving with enter/exit dwell; unobserved frames keep the latch."""
    hot_2d = (disp > cfg.disp_hi) | (dlog > cfg.area_hi)
    cold_2d = (disp < cfg.disp_lo) & (dlog < cfg.area_lo)
    # The 3D surface check vetoes occlusion-induced 2D motion; where no cloud
    # pair exists the 2D vote stands alone.
    hot = np.where(np.isnan(d3d), hot_2d, hot_2d & (d3d > cfg.surf_hi_mm))
    cold = np.where(np.isnan(d3d), cold_2d, cold_2d | (d3d < cfg.surf_lo_mm))
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


def mark_assembled(states: list[str], distances: list[float | None],
                   tol_m: float) -> list[str]:
    """Post-motion static frames near the final body-relative offset."""
    last_moving = max((i for i, s in enumerate(states) if s == "moving"), default=None)
    final = next((d for d in reversed(distances) if d is not None), None)
    if last_moving is None or final is None:
        return states
    out = list(states)
    for f in range(last_moving + 1, len(states)):
        d = distances[f]
        if out[f] == "static" and d is not None and abs(d - final) <= tol_m:
            out[f] = "assembled"
    return out


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
    parser.add_argument("--config", default=str(ROOT / "configs" / "pose_multiview_111.json"))
    parser.add_argument("--min-px", type=int, default=800, help="min mask pixels for a view to count as visible")
    parser.add_argument("--disp-hi", type=float, default=6.0, help="px/frame to enter moving")
    parser.add_argument("--disp-lo", type=float, default=2.5, help="px/frame to exit moving")
    parser.add_argument("--area-hi", type=float, default=0.10, help="|dlog area| to enter moving")
    parser.add_argument("--area-lo", type=float, default=0.04, help="|dlog area| to exit moving")
    parser.add_argument("--surf-hi-mm", type=float, default=6.0, help="3D surface shift to enter moving")
    parser.add_argument("--surf-lo-mm", type=float, default=4.0, help="3D surface shift to exit moving")
    parser.add_argument("--dwell-on", type=int, default=2)
    parser.add_argument("--dwell-off", type=int, default=4)
    parser.add_argument("--occlusion-ratio", type=float, default=0.35)
    parser.add_argument("--assembled-tol-m", type=float, default=0.03)
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    views = cfg["views"]
    start, end = int(cfg["frames"]["start"]), int(cfg["frames"]["end"])
    frames = [f"{f:06d}" for f in range(start, end + 1)]
    part_ids = {p: int(cfg["part_ids"][p]) for p in cfg["parts"]}
    mask_root = Path(cfg["masks_dir"])
    cloud_root = Path(cfg.get(
        "point_cloud_root", Path(cfg["output_root"]) / "parts_ply" / cfg["recon_backend"]))
    body = cfg["reference_part"]

    obs = mask_observations(mask_root, frames, views, part_ids, args.min_px)

    body_centroids = [cloud_centroid(cloud_root, t, body) for t in frames]
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
        disp, dlog = motion_score(centers, areas)
        d3d = cloud_motion_mm(cloud_root, frames, part)
        states = hysteresis_states(disp, dlog, d3d, visible, args)
        occluded = occlusion_flags(areas, visible, args.occlusion_ratio)
        states = [("occluded" if occluded[f] and states[f] == "static" else states[f])
                  for f in range(len(states))]
        distances: list[float | None] = []
        for f, timestamp in enumerate(frames):
            c = cloud_centroid(cloud_root, timestamp, part) if part != body else None
            b = body_centroids[f]
            distances.append(float(np.linalg.norm(c - b)) if c is not None and b is not None else None)
        if part != body:
            states = mark_assembled(states, distances, args.assembled_tol_m)

        detected = ranges_of(states, "moving", start)
        manual = cfg["states"][part].get("dynamic_ranges", [])
        report["parts"][part] = {
            "states": {frames[f]: {
                "state": states[f], "observing_views": int(visible[f]),
                "motion_px": None if np.isnan(disp[f]) else float(disp[f]),
                "surface_shift_mm": None if np.isnan(d3d[f]) else float(d3d[f]),
                "body_distance_m": distances[f],
            } for f in range(len(frames))},
            "detected_moving_ranges": detected,
            "manual_dynamic_ranges": manual,
            "detected_assembled_from": next(
                (start + f for f, s in enumerate(states) if s == "assembled"), None),
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
