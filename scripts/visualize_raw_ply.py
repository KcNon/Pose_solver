#!/usr/bin/env python
"""Project raw part PLY clouds onto full-res frames (no ICP).

Output (flat, no per-frame folders):
  outputs/normalized/proj_vis/{backend}/raw/montage_{timestamp}.jpg
  outputs/normalized/videos/{backend}_raw_montage.mp4   (--video)

Example:
    .venv/bin/python scripts/visualize_raw_ply.py --backend da3_self_cond --all --video
    .venv/bin/python scripts/visualize_raw_ply.py --timestamp 000000 --video-only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.geom import project_points
from common.mask_io import PART_COLORS, VIEW_NAMES, list_timestamps
from common.normalized_recon import (
    all_timestamps,
    load_pipeline,
    load_view_bundle,
    output_root,
    read_ply,
    resolve_backend,
    sample_timestamps,
)


def bgr(part: str) -> tuple[int, int, int]:
    r, g, b = PART_COLORS[part]
    return (b, g, r)


def ply_root(cfg: dict, backend: str, ply_tag: str | None) -> str:
    name = backend if not ply_tag else f"{backend}_{ply_tag}"
    return os.path.join(output_root(cfg), "parts_ply", name)


def vis_root(cfg: dict, backend: str, subdir: str) -> str:
    return os.path.join(output_root(cfg), "proj_vis", backend, subdir)


def montage_path(cfg: dict, backend: str, subdir: str, timestamp: str) -> str:
    return os.path.join(vis_root(cfg, backend, subdir), f"montage_{timestamp}.jpg")


def video_out_path(cfg: dict, backend: str, subdir: str) -> str:
    return os.path.join(output_root(cfg), "videos", f"{backend}_{subdir}_montage.mp4")


def draw_points(img, uv, z, color, depth_map=None, depth_tol=0.05, radius=2, no_occlusion=False):
    out = img.copy()
    h, w = img.shape[:2]
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    ok = (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, zz = u[ok], v[ok], z[ok]
    if not no_occlusion and depth_map is not None and len(u):
        dm = depth_map[v, u]
        vis = ~np.isfinite(dm) | (zz <= dm + depth_tol)
        u, v = u[vis], v[vis]
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            out[np.clip(v + dv, 0, h - 1), np.clip(u + du, 0, w - 1)] = color
    return out


def render_montage(cfg, backend: str, timestamp: str, parts: list[str], *,
                   depth_tol: float, no_occlusion: bool, ply_tag: str | None) -> np.ndarray:
    ply_dir = os.path.join(ply_root(cfg, backend, ply_tag), timestamp)
    clouds: dict[str, np.ndarray] = {}
    for part in parts:
        path = os.path.join(ply_dir, f"{part}.ply")
        if os.path.exists(path):
            pts = read_ply(path)
            if len(pts):
                clouds[part] = pts

    bundle = load_view_bundle(cfg, timestamp, backend=backend)
    images, depth, K, E = bundle["images"], bundle["depth"], bundle["intrinsics"], bundle["extrinsics"]
    occ_label = "no-occlusion" if no_occlusion else f"occ={depth_tol}"
    panels = []

    for v, vname in enumerate(VIEW_NAMES):
        photo = cv2.cvtColor(images[v], cv2.COLOR_RGB2BGR)
        overlay = photo.copy()
        for part in parts:
            if part not in clouds:
                continue
            uv, z = project_points(clouds[part], K[v], E[v])
            dm = None if no_occlusion else depth[v]
            overlay = draw_points(overlay, uv, z, bgr(part), depth_map=dm,
                                  depth_tol=depth_tol, no_occlusion=no_occlusion)
        cv2.putText(photo, f"photo {timestamp} {vname}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"raw ply {occ_label} {timestamp} {vname}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(np.hstack([photo, np.full((photo.shape[0], 4, 3), 255, np.uint8), overlay]))

    return np.vstack([cv2.resize(p, (p.shape[1] // 2, p.shape[0] // 2)) for p in panels])


def visualize_timestamp(cfg, backend: str, timestamp: str, parts: list[str], *,
                        depth_tol: float, no_occlusion: bool, ply_tag: str | None,
                        subdir: str) -> str:
    montage = render_montage(cfg, backend, timestamp, parts,
                             depth_tol=depth_tol, no_occlusion=no_occlusion, ply_tag=ply_tag)
    out_path = montage_path(cfg, backend, subdir, timestamp)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, montage, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path


def annotate_frame(img: np.ndarray, timestamp: str, idx: int, total: int) -> np.ndarray:
    out = img.copy()
    label = f"[{idx + 1}/{total}] {timestamp}  raw ply overlay"
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_video_ffmpeg(frames: list[np.ndarray], out_path: str, fps: float) -> None:
    tmp_dir = out_path + ".frames"
    os.makedirs(tmp_dir, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(tmp_dir, f"{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(tmp_dir, "%05d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_path,
    ]
    subprocess.run(cmd, check=True)
    for name in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, name))
    os.rmdir(tmp_dir)


def export_video(cfg, backend: str, subdir: str, timestamps: list[str], fps: float) -> str:
    frames, missing = [], []
    for i, ts in enumerate(timestamps):
        path = montage_path(cfg, backend, subdir, ts)
        img = cv2.imread(path)
        if img is None:
            missing.append(ts)
            continue
        frames.append(annotate_frame(img, ts, i, len(timestamps)))
    if missing:
        print(f"warning: {len(missing)} montages missing")
        for ts in missing[:5]:
            print(f"  missing: montage_{ts}.jpg")
    if not frames:
        raise SystemExit("no montage images found for video export")
    out_path = video_out_path(cfg, backend, subdir)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_video_ffmpeg(frames, out_path, fps)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp")
    ap.add_argument("--timestamps", nargs="+")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--video-only", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline_normalized.json"))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--depth-tol", type=float, default=0.05)
    ap.add_argument("--no-occlusion", action="store_true", default=True)
    ap.add_argument("--occlusion", action="store_true", help="enable depth occlusion test")
    ap.add_argument("--ply-tag", default=None)
    ap.add_argument("--subdir", default="raw")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    no_occlusion = args.no_occlusion and not args.occlusion

    if not args.video_only and not args.timestamp and not args.timestamps and not args.all and not args.sample:
        ap.error("one of --timestamp, --timestamps, --all, --sample, --video-only required")

    cfg = load_pipeline(args.pipeline)
    backend = resolve_backend(cfg, args.backend)
    parts = cfg.get("parts", ["lid", "body", "inner_pot"])

    if args.timestamp:
        timestamps = [args.timestamp]
    elif args.timestamps:
        timestamps = args.timestamps
    elif args.sample:
        timestamps = sample_timestamps()
    else:
        timestamps = all_timestamps(cfg)

    if not args.video_only:
        print(f"backend={backend} subdir={args.subdir} no_occlusion={no_occlusion} frames={len(timestamps)}")
        done = skipped = 0
        for i, ts in enumerate(timestamps):
            out_path = montage_path(cfg, backend, args.subdir, ts)
            if args.skip_existing and os.path.exists(out_path):
                skipped += 1
                continue
            print(f"[{i + 1}/{len(timestamps)}] {ts}")
            path = visualize_timestamp(
                cfg, backend, ts, parts,
                depth_tol=args.depth_tol, no_occlusion=no_occlusion,
                ply_tag=args.ply_tag, subdir=args.subdir,
            )
            print(f"  -> {path}")
            done += 1
        print(f"done: {done} rendered, {skipped} skipped")

    if args.video or args.video_only:
        print(f"exporting video ({len(timestamps)} frames @ {args.fps} fps)...")
        vpath = export_video(cfg, backend, args.subdir, timestamps, args.fps)
        print(f"video -> {vpath}  (~{len(timestamps) / args.fps:.1f}s)")


if __name__ == "__main__":
    main()
