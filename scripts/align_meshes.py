"""Align generated glb meshes to observed part clouds (mesh -> DA3 world).

For each part of a frame, register mesh/{part}.glb to
outputs/parts_ply/{backend}/{ts}/{part}.ply with a 7DoF similarity and save the
mesh->world transform to outputs/mesh_pose/{backend}/{ts}/{part}.{json,npy}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.mesh_align import align_mesh_to_cloud, read_ply_xyz
from common.recon_loader import output_paths, resolve_backend

DEFAULT_PARTS = ["body", "lid", "inner_pot"]


def load_pipeline(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="000019")
    ap.add_argument("--parts", nargs="*", default=None)
    ap.add_argument("--mesh-dir", default=os.path.join(ROOT, "mesh"))
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline.json"))
    ap.add_argument("--recon-backend", choices=["vggt", "da3"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    backend = resolve_backend(cfg, args.recon_backend)
    parts = args.parts or DEFAULT_PARTS

    ply_dir = output_paths(ROOT, backend, args.ts)["parts_ply"]
    out_dir = os.path.join(ROOT, "outputs", "mesh_pose", backend, args.ts)
    os.makedirs(out_dir, exist_ok=True)

    for part in parts:
        glb = os.path.join(args.mesh_dir, f"{part}.glb")
        ply = os.path.join(ply_dir, f"{part}.ply")
        if not os.path.exists(glb):
            print(f"[skip] {part}: no mesh {glb}")
            continue
        if not os.path.exists(ply):
            print(f"[skip] {part}: no cloud {ply}")
            continue

        mesh = trimesh.load(glb, force="mesh")
        obs = read_ply_xyz(ply)
        res = align_mesh_to_cloud(mesh, obs, seed=args.seed)

        T = res["T_mesh_to_world"]
        np.save(os.path.join(out_dir, f"{part}.npy"), T)
        with open(os.path.join(out_dir, f"{part}.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ts": args.ts,
                    "part": part,
                    "backend": backend,
                    "mesh": os.path.relpath(glb, ROOT),
                    "cloud": os.path.relpath(ply, ROOT),
                    "T_mesh_to_world": T.tolist(),
                    "scale": res["scale"],
                    "rotation": res["R"].tolist(),
                    "translation": res["t"].tolist(),
                    "fit_rmse": res["fit_rmse"],
                    "icp_cost": res["icp_cost"],
                    "n_obs": res["n_obs"],
                    "n_mesh_sample": res["n_mesh_sample"],
                },
                f,
                indent=2,
            )
        print(f"[ok] {part}: scale={res['scale']:.4f} fit_rmse={res['fit_rmse']:.5f} "
              f"(n_obs={res['n_obs']}) -> {out_dir}/{part}.json")


if __name__ == "__main__":
    main()
