"""Align source timestamp parts to reference timestamp parts.

Every common part gets its own rigid registration (ICP or GICP). If a part is truly
static in world frame, registration should converge to T ~= I on its own — never hard-code identity.

Init modes (--init-mode):
  centroid  translation only (legacy)
  kabsch    NN Kabsch on part cloud
  camera    VGGT/DA3 multi-view camera-center alignment
  both      camera global init + per-part Kabsch (default)
  rigid     use lid (or --global-part) init for all parts

Output: pose_<src>_to_<ref>.json  with per-part 4x4 T (src -> ref).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.icp import (
    apply_transform,
    build_icp_init,
    camera_world_align_init,
    compose_transform,
    downsample,
    register_points,
    write_ply,
)
from common.recon_loader import load_recon, resolve_backend


def load_pipeline(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_ply(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    with open(path, encoding="ascii") as f:
        header = []
        while True:
            line = f.readline()
            header.append(line)
            if line.strip() == "end_header":
                break
        has_color = any("property uchar red" in h for h in header)
        pts, cols = [], [] if has_color else None
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if has_color:
                cols.append([int(parts[3]), int(parts[4]), int(parts[5])])
    return np.asarray(pts, dtype=np.float32), cols


def load_parts(ts: str) -> list[str]:
    with open(os.path.join(ROOT, "configs", f"parts_{ts}.json"), encoding="utf-8") as f:
        return json.load(f)["parts"]


def compute_global_init_from_part(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    T_camera: np.ndarray | None,
    init_mode: str,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Build a shared init using one reference part (typically lid)."""
    if init_mode == "rigid":
        # rigid uses global built externally; fallback to both logic here.
        init_mode = "both"
    if init_mode == "both":
        if T_camera is None:
            raise ValueError("both/rigid requires camera extrinsics")
        src_cam = apply_transform(src_pts, T_camera)
        T_local, _ = build_icp_init(src_cam, ref_pts, "kabsch", seed=seed)
        T = compose_transform(T_camera, T_local)
        return T, {"mode": "both", "components": ["camera", "kabsch_nn"], "global_part": True}
    if init_mode == "camera":
        if T_camera is None:
            raise ValueError("camera mode requires extrinsics")
        return T_camera, {"mode": "camera", "components": ["camera"], "global_part": True}
    T, info = build_icp_init(src_pts, ref_pts, init_mode, T_camera=T_camera, seed=seed)
    info["global_part"] = True
    return T, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="000019")
    ap.add_argument("--src", default="000018")
    ap.add_argument("--max-pts", type=int, default=30000)
    ap.add_argument("--max-iters", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", default=None, help="default: outputs/icp/<backend>/")
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline.json"))
    ap.add_argument("--recon-backend", choices=["vggt", "da3"], default=None)
    ap.add_argument(
        "--init-mode",
        choices=["centroid", "kabsch", "camera", "both", "rigid"],
        default="both",
        help="ICP initialization strategy (default: both)",
    )
    ap.add_argument(
        "--global-part",
        default="lid",
        help="part used to build shared init for rigid mode (default: lid)",
    )
    ap.add_argument(
        "--reg-method",
        choices=["icp", "gicp"],
        default="icp",
        help="registration backend: icp (kornia-rs) or gicp (small-gicp)",
    )
    ap.add_argument(
        "--gicp-voxel",
        type=float,
        default=None,
        help="GICP downsampling resolution in meters (default: auto from cloud extent)",
    )
    ap.add_argument(
        "--gicp-max-dist",
        type=float,
        default=None,
        help="GICP max correspondence distance in meters (default: 8x --gicp-voxel)",
    )
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    backend = resolve_backend(cfg, args.recon_backend)
    if args.out is None:
        out_name = backend if args.reg_method == "icp" else f"{backend}_gicp"
        args.out = os.path.join(ROOT, "outputs", "icp", out_name)
    print(f"recon backend: {backend}")
    print(f"reg method: {args.reg_method}")
    print(f"ply root: outputs/parts_ply/{backend}/")
    print(f"init mode: {args.init_mode}")

    ref_parts = set(load_parts(args.ref))
    src_parts = set(load_parts(args.src))
    common = sorted(ref_parts & src_parts, key=lambda p: (p != "lid", p))
    if not common:
        raise SystemExit(f"no common parts between {args.ref} and {args.src}")

    ply_root = os.path.join(ROOT, "outputs", "parts_ply", backend)
    os.makedirs(args.out, exist_ok=True)

    T_camera = None
    camera_info = None
    if args.init_mode in ("camera", "both", "rigid"):
        recon_ref = load_recon(cfg, args.ref, backend=backend)
        recon_src = load_recon(cfg, args.src, backend=backend)
        T_camera = camera_world_align_init(recon_src["extrinsics"], recon_ref["extrinsics"])
        camera_info = {
            "camera_centers_ref": camera_centers_json(recon_ref["extrinsics"]),
            "camera_centers_src": camera_centers_json(recon_src["extrinsics"]),
            "transform_src_to_ref": T_camera.tolist(),
        }
        print(f"camera init: t={T_camera[:3, 3]}")

    # Preload + downsample all part clouds once.
    clouds: dict[str, dict] = {}
    for pname in common:
        ref_pts, ref_cols = read_ply(os.path.join(ply_root, args.ref, f"{pname}.ply"))
        src_pts, src_cols = read_ply(os.path.join(ply_root, args.src, f"{pname}.ply"))
        ref_ds, _ = downsample(ref_pts, None, args.max_pts, seed=1)
        src_ds, _ = downsample(src_pts, None, args.max_pts, seed=2)
        clouds[pname] = {
            "ref": ref_ds, "src": src_ds,
            "ref_full": ref_pts, "src_full": src_pts,
            "ref_cols": ref_cols, "src_cols": src_cols,
        }
        print(f"[{pname}] ref={len(ref_ds)} src={len(src_ds)}")

    T_global = None
    global_init_info = None
    if args.init_mode == "rigid":
        gpart = args.global_part
        if gpart not in clouds:
            raise SystemExit(f"--global-part {gpart} not in common parts {common}")
        T_global, global_init_info = compute_global_init_from_part(
            clouds[gpart]["src"], clouds[gpart]["ref"], T_camera, "both", seed=3,
        )
        print(f"rigid global init from {gpart}: t={T_global[:3, 3]}")

    summary = {
        "ref": args.ref,
        "src": args.src,
        "recon_backend": backend,
        "reg_method": args.reg_method,
        "init_mode": args.init_mode,
        "global_part": args.global_part if args.init_mode == "rigid" else None,
        "camera_init": camera_info,
        "global_init": global_init_info,
        "common_parts": common,
        "ref_only_parts": sorted(ref_parts - src_parts),
        "src_only_parts": sorted(src_parts - ref_parts),
        "parts": {},
    }

    for pname in common:
        ref_pts = clouds[pname]["ref"]
        src_pts = clouds[pname]["src"]
        init, init_info = build_icp_init(
            src_pts, ref_pts,
            mode="rigid" if args.init_mode == "rigid" else args.init_mode,
            T_camera=T_camera,
            T_global=T_global,
            seed=4,
        )
        T, info = register_points(
            src_pts, ref_pts, method=args.reg_method, init=init,
            max_iters=args.max_iters, tol=args.tol,
            downsampling_resolution=args.gicp_voxel,
            max_correspondence_distance=args.gicp_max_dist,
        )
        info["init"] = init_info
        info["init_transform"] = init.tolist()

        aligned = apply_transform(src_pts, T).astype(np.float32)
        np.save(os.path.join(args.out, f"T_{args.src}_to_{args.ref}_{pname}.npy"), T)
        write_ply(
            os.path.join(args.out, f"aligned_{args.src}_{pname}.ply"),
            aligned, clouds[pname]["src_cols"],
        )
        write_ply(
            os.path.join(args.out, f"ref_{args.ref}_{pname}.ply"),
            ref_pts, clouds[pname]["ref_cols"],
        )
        summary["parts"][pname] = {
            "transform_src_to_ref": T.tolist(),
            "icp": info,
            "translation": T[:3, 3].tolist(),
            "rotation": T[:3, :3].tolist(),
        }
        nn = info.get("nn_rmse", info["rmse"])
        print(
            f"  [{pname}] {args.reg_method.upper()} rmse={info['rmse']:.6f} "
            f"nn_rmse={nn:.6f} iters={info['iters']} t={T[:3, 3]}"
        )

    out_json = os.path.join(args.out, f"pose_{args.src}_to_{args.ref}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out_json)


def camera_centers_json(E: np.ndarray) -> list[list[float]]:
    from common.icp import camera_centers_from_extrinsics
    C = camera_centers_from_extrinsics(E)
    return C.tolist()


if __name__ == "__main__":
    main()
