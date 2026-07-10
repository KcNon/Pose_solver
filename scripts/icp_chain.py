#!/usr/bin/env python
"""Chain per-part ICP on normalized sample frames with cumulative verification."""
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
    nn_rmse,
    register_points,
    write_ply,
)
from common.normalized_recon import (
    icp_out_dir,
    load_pipeline,
    load_recon,
    parts_ply_dir,
    read_ply,
    resolve_backend,
    sample_segments,
)


def transform_error(T_a: np.ndarray, T_b: np.ndarray) -> dict[str, float]:
    dT = np.linalg.inv(T_a) @ T_b
    trans = float(np.linalg.norm(dT[:3, 3]))
    R = dT[:3, :3]
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    rot = float(np.degrees(np.arccos(cos)))
    return {"trans_err_m": trans, "rot_err_deg": rot}


def load_part_cloud(ply_root: str, ts: str, part: str, max_pts: int) -> tuple[np.ndarray, np.ndarray | None] | None:
    path = os.path.join(ply_root, ts, f"{part}.ply")
    if not os.path.exists(path):
        return None
    pts, cols = read_ply(path)
    if len(pts) < 10:
        return None
    pts_ds, cols_ds = downsample(pts, cols, max_pts, seed=hash((ts, part)) % 10000)
    return pts_ds, cols_ds


def register_pair(
    ref: str,
    src: str,
    parts: list[str],
    ply_root: str,
    cfg: dict,
    backend: str,
    args,
    prev_T: dict[str, np.ndarray | None],
) -> tuple[dict, dict[str, np.ndarray | None]]:
    recon_ref = load_recon(cfg, ref, backend=backend)
    recon_src = load_recon(cfg, src, backend=backend)
    T_camera = camera_world_align_init(recon_src["extrinsics"], recon_ref["extrinsics"])

    summary = {
        "ref": ref,
        "src": src,
        "recon_backend": backend,
        "reg_method": args.reg_method,
        "init_mode": args.init_mode,
        "parts": {},
    }
    new_prev: dict[str, np.ndarray | None] = {}

    for part in parts:
        ref_cloud = load_part_cloud(ply_root, ref, part, args.max_pts)
        src_cloud = load_part_cloud(ply_root, src, part, args.max_pts)
        if ref_cloud is None or src_cloud is None:
            print(f"  [{part}] skip (missing or too sparse)")
            new_prev[part] = prev_T.get(part)
            continue

        ref_pts, ref_cols = ref_cloud
        src_pts, src_cols = src_cloud

        if prev_T.get(part) is not None:
            init = prev_T[part]
            init_info = {"mode": "chain_prev", "components": ["prev_pair"]}
        elif args.init_mode == "both":
            init, init_info = build_icp_init(
                src_pts, ref_pts, mode="both", T_camera=T_camera, seed=4,
            )
        else:
            init, init_info = build_icp_init(
                src_pts, ref_pts, mode=args.init_mode, T_camera=T_camera, seed=4,
            )

        T, info = register_points(
            src_pts, ref_pts, method=args.reg_method, init=init,
            max_iters=args.max_iters, tol=args.tol,
        )
        info["init"] = init_info
        info["init_transform"] = init.tolist()
        new_prev[part] = T

        summary["parts"][part] = {
            "transform_src_to_ref": T.tolist(),
            "icp": info,
            "translation": T[:3, 3].tolist(),
            "rotation": T[:3, :3].tolist(),
            "n_ref": len(ref_pts),
            "n_src": len(src_pts),
        }
        nn = info.get("nn_rmse", info["rmse"])
        print(
            f"  [{part}] rmse={info['rmse']:.6f} nn_rmse={nn:.6f} "
            f"init={init_info['mode']} t={T[:3, 3]}"
        )

    return summary, new_prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline_normalized.json"))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--max-pts", type=int, default=30000)
    ap.add_argument("--max-iters", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--init-mode", choices=["centroid", "kabsch", "camera", "both"], default="both")
    ap.add_argument("--reg-method", choices=["icp", "gicp"], default="icp")
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    backend = resolve_backend(cfg, args.backend)
    parts = cfg.get("parts", ["lid", "body", "inner_pot"])
    ply_root = parts_ply_dir(cfg, backend)
    out_dir = icp_out_dir(cfg, backend)
    os.makedirs(out_dir, exist_ok=True)

    chain_summary = {
        "backend": backend,
        "parts": parts,
        "segments": [],
        "pair_poses": {},
    }

    for seg in sample_segments():
        print(f"\n===== segment {seg[0]}..{seg[-1]} =====")
        prev_T: dict[str, np.ndarray | None] = {p: None for p in parts}
        cumulative: dict[str, np.ndarray] = {}
        seg_info = {"frames": seg, "pairs": [], "cumulative": {}}

        for i in range(1, len(seg)):
            ref, src = seg[i - 1], seg[i]
            pair_key = f"{src}_to_{ref}"
            print(f"\n-- pair {src} -> {ref} --")
            summary, prev_T = register_pair(
                ref, src, parts, ply_root, cfg, backend, args, prev_T,
            )
            out_json = os.path.join(out_dir, f"pose_{src}_to_{ref}.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            chain_summary["pair_poses"][pair_key] = out_json
            seg_info["pairs"].append(pair_key)

            for part, pinfo in summary["parts"].items():
                T = np.array(pinfo["transform_src_to_ref"], dtype=np.float64)
                if part not in cumulative:
                    cumulative[part] = T.copy()
                else:
                    cumulative[part] = compose_transform(T, cumulative[part])

        anchor, last = seg[0], seg[-1]
        for part, T_chain in cumulative.items():
            src_cloud = load_part_cloud(ply_root, last, part, args.max_pts)
            ref_cloud = load_part_cloud(ply_root, anchor, part, args.max_pts)
            if src_cloud is None or ref_cloud is None:
                continue
            src_pts, _ = src_cloud
            ref_pts, ref_cols = ref_cloud

            recon_ref = load_recon(cfg, anchor, backend=backend)
            recon_src = load_recon(cfg, last, backend=backend)
            T_camera = camera_world_align_init(recon_src["extrinsics"], recon_ref["extrinsics"])
            init, _ = build_icp_init(
                src_pts, ref_pts, mode="both", T_camera=T_camera, seed=5,
            )
            T_direct, _ = register_points(
                src_pts, ref_pts, method=args.reg_method, init=init,
                max_iters=args.max_iters, tol=args.tol,
            )

            err = transform_error(T_chain, T_direct)
            seg_info["cumulative"][part] = {
                "anchor": anchor,
                "last": last,
                "T_last_to_anchor_chain": T_chain.tolist(),
                "T_last_to_anchor_direct": T_direct.tolist(),
                "chain_vs_direct": err,
                "chain_rmse_on_src": nn_rmse(src_pts, ref_pts, T_chain),
                "direct_rmse_on_src": nn_rmse(src_pts, ref_pts, T_direct),
            }
            write_ply(
                os.path.join(out_dir, f"aligned_{last}_{part}_to_{anchor}.ply"),
                apply_transform(src_pts, T_chain).astype(np.float32),
                ref_cols,
            )
            write_ply(
                os.path.join(out_dir, f"aligned_{last}_{part}_to_{anchor}_direct.ply"),
                apply_transform(src_pts, T_direct).astype(np.float32),
                ref_cols,
            )
            print(
                f"  [verify {part}] chain vs direct: "
                f"dT={err['trans_err_m']:.4f}m dR={err['rot_err_deg']:.2f}deg"
            )

        chain_summary["segments"].append(seg_info)

    summary_path = os.path.join(out_dir, "chain_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(chain_summary, f, indent=2)
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
