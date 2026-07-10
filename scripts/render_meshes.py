"""Render posed meshes back into DA3 camera views.

For the frame where the meshes were aligned (``--from-ts``, default 000019) the
mesh->world transform is used directly. For other frames the mesh is carried over
with the inter-frame ICP pose (outputs/icp[/backend]/pose_{ts}_to_{from_ts}.json,
field transform_src_to_ref maps ts->from_ts).

Per view we write: textured rgb, overlay on the photo, per-part binary masks,
16-bit depth, and a normal map; plus an overlay montage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.mesh_render import SceneRenderer, normals_from_depth
from common.recon_loader import load_view_bundle, resolve_backend

VIEW_NAMES = ["2-1", "2-2", "2-3", "2-4", "2-5", "2-6"]
DEFAULT_PARTS = ["body", "lid", "inner_pot"]
OVERLAY_COLORS = {"body": (60, 60, 220), "lid": (220, 60, 60), "inner_pot": (60, 200, 60)}


def load_pipeline(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_interframe_pose(backend: str, ts: str, from_ts: str) -> dict | None:
    for cand in (
        os.path.join(ROOT, "outputs", "icp", backend, f"pose_{ts}_to_{from_ts}.json"),
        os.path.join(ROOT, "outputs", "icp", f"pose_{ts}_to_{from_ts}.json"),
    ):
        if os.path.exists(cand):
            with open(cand, encoding="utf-8") as f:
                return json.load(f)
    return None


def build_part_transforms(backend: str, ts: str, from_ts: str, parts: list[str],
                          mesh_dir: str) -> dict[str, dict]:
    """{part: {mesh, T_world}} for the target frame `ts`."""
    pose_dir = os.path.join(ROOT, "outputs", "mesh_pose", backend, from_ts)
    inter = None
    if ts != from_ts:
        inter = find_interframe_pose(backend, ts, from_ts)
        if inter is None:
            raise FileNotFoundError(
                f"no inter-frame pose {ts}->{from_ts} under outputs/icp")

    out = {}
    for part in parts:
        npy = os.path.join(pose_dir, f"{part}.npy")
        glb = os.path.join(mesh_dir, f"{part}.glb")
        if not (os.path.exists(npy) and os.path.exists(glb)):
            print(f"[skip] {part}: missing mesh pose or glb")
            continue
        T_mesh_world = np.load(npy)  # mesh -> from_ts world

        if ts == from_ts:
            T_world = T_mesh_world
        else:
            pinfo = inter.get("parts", {}).get(part)
            if pinfo is None:
                print(f"[skip] {part}: no inter-frame pose for {ts}->{from_ts}")
                continue
            T_ts_to_from = np.array(pinfo["transform_src_to_ref"], dtype=np.float64)
            T_world = np.linalg.inv(T_ts_to_from) @ T_mesh_world

        out[part] = {"mesh": trimesh.load(glb, force="mesh"), "T_world": T_world}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="000019", help="target frame to render into")
    ap.add_argument("--from-ts", default="000019", help="frame the meshes were aligned in")
    ap.add_argument("--parts", nargs="*", default=None)
    ap.add_argument("--mesh-dir", default=os.path.join(ROOT, "mesh"))
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline.json"))
    ap.add_argument("--recon-backend", choices=["vggt", "da3"], default=None)
    ap.add_argument("--alpha", type=float, default=0.55, help="overlay blend weight")
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    backend = resolve_backend(cfg, args.recon_backend)
    parts = args.parts or DEFAULT_PARTS

    part_tf = build_part_transforms(backend, args.ts, args.from_ts, parts, args.mesh_dir)
    if not part_tf:
        raise SystemExit("no parts to render")

    bundle = load_view_bundle(cfg, args.ts, VIEW_NAMES, backend=backend)
    images, K, E = bundle["images"], bundle["intrinsics"], bundle["extrinsics"]
    n_views, H, W = images.shape[:3]

    out_dir = os.path.join(ROOT, "outputs", "mesh_render", backend, args.ts)
    os.makedirs(out_dir, exist_ok=True)

    color_parts = [(d["mesh"], d["T_world"]) for d in part_tf.values()]
    seg_parts = [(name, d["mesh"], d["T_world"]) for name, d in part_tf.items()]

    overlays = []
    with SceneRenderer(W, H) as rend:
        for v in range(n_views):
            prefix = f"view{v}_{VIEW_NAMES[v]}"
            color, depth = rend.render(color_parts, K[v], E[v])
            masks = rend.render_seg(seg_parts, K[v], E[v])
            fg = depth > 0

            # textured rgb (black bg)
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_rgb.png"),
                        cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

            # overlay: tint each part's visible pixels over the photo
            base = cv2.cvtColor(images[v], cv2.COLOR_RGB2BGR).astype(np.float32)
            ov = base.copy()
            for name, m in masks.items():
                col = np.array(OVERLAY_COLORS.get(name, (200, 200, 200)), np.float32)
                ov[m] = (1 - args.alpha) * base[m] + args.alpha * col
            overlay = ov.astype(np.uint8)
            cv2.putText(overlay, f"[{backend}] mesh->{args.ts} ({VIEW_NAMES[v]})",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_overlay.jpg"), overlay,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            overlays.append(overlay)

            # per-part binary masks
            for name, m in masks.items():
                cv2.imwrite(os.path.join(out_dir, f"{prefix}_{name}_mask.png"),
                            (m.astype(np.uint8) * 255))

            # depth (16-bit millimetres) + normal map
            depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_depth.png"), depth_mm)
            normal = normals_from_depth(depth, K[v])
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_normal.png"),
                        cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))

    montage = np.vstack([cv2.resize(o, (o.shape[1] // 2, o.shape[0] // 2)) for o in overlays])
    cv2.imwrite(os.path.join(out_dir, "montage.jpg"), montage, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[ok] wrote {n_views} views + montage to {out_dir}")
    for name, d in part_tf.items():
        print(f"     part {name}: T_world det={np.linalg.det(d['T_world'][:3, :3]):.4f}")


if __name__ == "__main__":
    main()
