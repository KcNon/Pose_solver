#!/usr/bin/env python
"""Estimate each part's frame-to-frame motion by registering adjacent point clouds.

No mesh is involved: the only inputs are the per-frame part clouds produced by the depth
pipeline, so this works before any part reconstruction exists.

    .venv/bin/python scripts/estimate_adjacent_frame_pose.py \
        --clouds experiments/objects_0825/Object-9/depth/parts_ply/da3_self_cond_quality \
        --out    experiments/objects_0825/Object-9/adjacent_pose

For every consecutive pair the cloud of frame t+1 is aligned onto frame t with the
project's multi-scale GICP, giving T_t->t+1 in world coordinates. Pairs are also composed
into a cumulative trajectory relative to the first frame each part appears in.

Registration quality is reported per pair rather than assumed: a pair whose fitness or
inlier RMSE falls outside the thresholds is marked ``reliable: false`` and breaks the
cumulative chain, because composing a bad relative transform corrupts every later frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.cloud_io import read_ply_xyz
from common.gicp import multiscale_gicp, transform_angle

GICP_CFG = {
    "voxel_sizes_m": [0.01, 0.005, 0.002],
    "max_correspondence_m": [0.05, 0.02, 0.01],
    "max_iterations": 60,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clouds", type=Path, required=True,
                   help="parts_ply/<variant> dir holding <frame>/<part>.ply")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--parts", nargs="*", default=None,
                   help="Parts to solve (default: every part found)")
    p.add_argument("--min-points", type=int, default=300,
                   help="Skip a frame whose cloud is smaller than this")
    p.add_argument("--min-fitness", type=float, default=0.30,
                   help="Pairs below this 8mm fitness are flagged unreliable")
    p.add_argument("--max-rmse-mm", type=float, default=6.0,
                   help="Pairs above this inlier RMSE are flagged unreliable")
    return p.parse_args()


def discover(clouds: Path) -> tuple[list[str], list[str]]:
    frames = sorted(d.name for d in clouds.iterdir() if d.is_dir())
    parts = sorted({p.stem for d in clouds.iterdir() if d.is_dir() for p in d.glob("*.ply")})
    return frames, parts


def main() -> int:
    args = parse_args()
    frames, found = discover(args.clouds)
    parts = args.parts or found
    if not frames:
        raise SystemExit(f"no frame dirs under {args.clouds}")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(frames)} frames, parts: {parts}", flush=True)

    report: dict[str, dict] = {}
    for part in parts:
        clouds: dict[str, np.ndarray] = {}
        for f in frames:
            path = args.clouds / f / f"{part}.ply"
            if not path.exists():
                continue
            pts = read_ply_xyz(path)
            if len(pts) >= args.min_points:
                clouds[f] = np.ascontiguousarray(pts, dtype=np.float64)
        available = sorted(clouds)
        print(f"\n[{part}] {len(available)}/{len(frames)} frames with >= {args.min_points} points",
              flush=True)
        if len(available) < 2:
            report[part] = {"frames_with_cloud": len(available), "pairs": []}
            continue

        pairs = []
        for a, b in zip(available, available[1:]):
            T, quality = multiscale_gicp(clouds[b], clouds[a], np.eye(4), GICP_CFG)
            rmse = quality["inlier_rmse_m"]
            reliable = (
                quality["fitness_8mm"] >= args.min_fitness
                and rmse is not None
                and rmse * 1000.0 <= args.max_rmse_mm
            )
            # Adjacent frames are 1/30 s apart, so a large jump is a registration failure
            # rather than real motion; recording it keeps the chain honest.
            pairs.append({
                "from": a, "to": b,
                "consecutive": int(b) - int(a) == 1,
                "T_from_to": T.tolist(),
                "translation_mm": float(np.linalg.norm(T[:3, 3]) * 1000.0),
                "rotation_deg": transform_angle(T),
                "fitness_8mm": quality["fitness_8mm"],
                "inlier_rmse_mm": None if rmse is None else rmse * 1000.0,
                "median_nn_mm": quality["median_nn_m"] * 1000.0,
                "n_source": quality["n_source"], "n_target": quality["n_target"],
                "reliable": bool(reliable),
            })

        # Compose into a trajectory; a break restarts the chain so one bad pair does not
        # silently poison everything downstream of it.
        trajectory: dict[str, dict] = {}
        accumulated = np.eye(4)
        anchor = available[0]
        trajectory[anchor] = {"T_from_anchor": accumulated.tolist(), "anchor": anchor}
        for pair in pairs:
            if not (pair["reliable"] and pair["consecutive"]):
                anchor = pair["to"]
                accumulated = np.eye(4)
            else:
                accumulated = np.asarray(pair["T_from_to"]) @ accumulated
            trajectory[pair["to"]] = {"T_from_anchor": accumulated.tolist(), "anchor": anchor}

        good = sum(1 for p in pairs if p["reliable"])
        breaks = sum(1 for p in pairs if not p["reliable"])
        med_t = float(np.median([p["translation_mm"] for p in pairs])) if pairs else 0.0
        med_r = float(np.median([p["rotation_deg"] for p in pairs])) if pairs else 0.0
        print(f"[{part}] {good}/{len(pairs)} pairs reliable, {breaks} chain breaks; "
              f"median step {med_t:.2f} mm / {med_r:.2f} deg", flush=True)
        report[part] = {
            "frames_with_cloud": len(available),
            "pairs_total": len(pairs), "pairs_reliable": good, "chain_breaks": breaks,
            "median_translation_mm": med_t, "median_rotation_deg": med_r,
            "pairs": pairs, "trajectory": trajectory,
        }
        (args.out / f"{part}_adjacent_pose.json").write_text(
            json.dumps(report[part], indent=2) + "\n", encoding="utf-8")

    (args.out / "summary.json").write_text(json.dumps({
        "clouds": str(args.clouds),
        "method": "multi-scale GICP between adjacent frame clouds, mesh-free",
        "gicp": GICP_CFG,
        "thresholds": {"min_fitness_8mm": args.min_fitness,
                       "max_inlier_rmse_mm": args.max_rmse_mm,
                       "min_points": args.min_points},
        "parts": {k: {kk: vv for kk, vv in v.items() if kk not in ("pairs", "trajectory")}
                  for k, v in report.items()},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
