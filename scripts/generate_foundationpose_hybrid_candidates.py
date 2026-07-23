#!/usr/bin/env python
"""Generate FoundationPose global-rotation candidates with fixed solver translation.

The bundled SPOT checkout has a usable FoundationPose refiner but no score
checkpoint.  This stage therefore uses FoundationPose's global SO(3) grid and
PoseRefinePredictor only.  Candidate ranking is deliberately deferred to
``evaluate_foundationpose_hybrid.py``, where every pose is scored in the common
world frame against all six cameras.

Run this stage with the SPOT conda environment.  It never edits a trajectory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
OBJECT_REPO = Path("/data_ft_9_10/ziang/code/object_centric_diffusion")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OBJECT_REPO))
sys.path.insert(0, str(OBJECT_REPO / "foundation_pose"))

from common.backproject_utils import load_recon_colors
from common.depth_gauge import apply_depth_gauge, load_depth_gauge
from common.io_utils import load_json, write_json
from common.normalized_recon import load_recon


DEFAULT_TARGETS = ["body:20", "lid:50", "lid:84", "lid:88", "lid:100", "lid:108"]


def parse_targets(values: list[str]) -> list[tuple[str, int]]:
    result = []
    for value in values:
        try:
            part, frame = value.split(":", 1)
            frame_number = int(frame)
        except ValueError as error:
            raise ValueError(f"target must be PART:FRAME, got {value!r}") from error
        if part not in {"body", "inner_pot", "lid"}:
            raise ValueError(f"unknown part in target {value!r}")
        result.append((part, frame_number))
    return result


def homogeneous_extrinsic(value: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :4] = np.asarray(value, np.float64)[:3, :4]
    return result


def centered_metric_mesh(path: Path, scale: float,
                         raw_origin: np.ndarray) -> tuple[trimesh.Trimesh, np.ndarray]:
    mesh = trimesh.load(path, force="mesh", process=False).copy()
    mesh.vertices = np.asarray(
        (np.asarray(mesh.vertices, np.float64) - raw_origin[None]) * scale,
        dtype=np.float32,
    )
    center = 0.5 * (mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0))
    mesh.vertices = mesh.vertices - center[None]
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        material = mesh.visual.material
        image = getattr(material, "image", None)
        if image is None:
            image = getattr(material, "baseColorTexture", None)
        if image is not None:
            mesh.visual = trimesh.visual.texture.TextureVisuals(
                uv=np.asarray(mesh.visual.uv).copy(),
                material=trimesh.visual.material.SimpleMaterial(image=image),
            )
    return mesh, center


def global_rotation_grid(sample_views_icosphere, euler_matrix,
                         n_views: int, inplane_step_deg: float) -> np.ndarray:
    rotations = []
    for cam_in_object in sample_views_icosphere(n_views=n_views):
        for angle in np.deg2rad(np.arange(0.0, 360.0, inplane_step_deg)):
            candidate = np.linalg.inv(cam_in_object @ euler_matrix(0.0, 0.0, angle))
            rotations.append(candidate[:3, :3])
    return np.asarray(rotations, np.float64)


def farthest_rotation_subset(rotations: np.ndarray, maximum: int) -> np.ndarray:
    """Deterministic farthest-point subset under SO(3) geodesic distance."""
    if len(rotations) <= maximum:
        return rotations
    quaternions = Rotation.from_matrix(rotations).as_quat()
    selected = [0]
    similarity = np.abs(quaternions @ quaternions[0])
    min_distance = 2.0 * np.arccos(np.clip(similarity, 0.0, 1.0))
    for _ in range(1, maximum):
        index = int(np.argmax(min_distance))
        selected.append(index)
        similarity = np.abs(quaternions @ quaternions[index])
        distance = 2.0 * np.arccos(np.clip(similarity, 0.0, 1.0))
        min_distance = np.minimum(min_distance, distance)
    return rotations[np.asarray(selected)]


def visible_source_views(cfg: dict, frame: int, part: str,
                         depth_hw: tuple[int, int], maximum: int) -> list[dict]:
    timestamp = f"{frame:06d}"
    candidates = []
    for index, view in enumerate(cfg["views"]):
        labels = np.asarray(Image.open(
            Path(cfg["masks_dir"]) / timestamp / f"{view}.png"))
        mask = cv2.resize(
            (labels == int(cfg["part_ids"][part])).astype(np.uint8),
            (depth_hw[1], depth_hw[0]), interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        touches = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        candidates.append({"index": index, "view": view, "pixels": int(mask.sum()),
                           "touches_border": touches})
    candidates = [item for item in candidates if item["pixels"] >= 40]
    candidates.sort(key=lambda item: (item["touches_border"], -item["pixels"]))
    return candidates[:maximum]


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(a.T @ b).magnitude()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pose_multiview_111_quality.json"))
    parser.add_argument("--trajectory", default=str(
        ROOT / "experiments/three_part_multiview_111f/outputs_v6_global_pose/accepted_final/pose/trajectory.json"))
    parser.add_argument("--depth-gauge", default=None)
    parser.add_argument("--output", default=str(
        ROOT / "experiments/three_part_multiview_111f/outputs_v7_foundationpose_hybrid/candidates.json"))
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--source-views", type=int, default=2)
    parser.add_argument("--icosphere-views", type=int, default=20)
    parser.add_argument("--inplane-step-deg", type=float, default=90.0)
    parser.add_argument("--max-hypotheses", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()

    try:
        import torch
        import nvdiffrast.torch as dr
        from foundation_pose.Utils import (
            compute_mesh_diameter,
            depth2xyzmap,
            euler_matrix,
            make_mesh_tensors,
            sample_views_icosphere,
        )
        from foundation_pose.learning.training.predict_pose_refine import PoseRefinePredictor
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "FoundationPose dependencies are unavailable; run with `conda run -n spot python ...`"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("FoundationPose hybrid generation requires CUDA")

    cfg = load_json(Path(args.config))
    trajectory_path = Path(args.trajectory).resolve()
    trajectory = load_json(trajectory_path)
    gauge_path = args.depth_gauge or cfg.get("depth_gauge_path")
    gauge = load_depth_gauge(gauge_path) if gauge_path else None
    targets = parse_targets(args.targets)
    rotations = global_rotation_grid(
        sample_views_icosphere, euler_matrix, args.icosphere_views, args.inplane_step_deg)
    rotations = farthest_rotation_subset(rotations, args.max_hypotheses)

    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    mesh_cache = {}
    output = {
        "schema_version": 1,
        "status": "complete",
        "mode": "FoundationPose global SO(3) grid + PoseRefinePredictor; solver translation and scale fixed",
        "config": str(Path(args.config).resolve()),
        "trajectory": str(trajectory_path),
        "depth_gauge": str(Path(gauge_path).resolve()) if gauge_path else None,
        "parameters": {
            "source_views": args.source_views,
            "icosphere_views": args.icosphere_views,
            "inplane_step_deg": args.inplane_step_deg,
            "hypotheses_after_subsampling": int(len(rotations)),
            "refiner_iterations": args.iterations,
            "translation_policy": "keep input T_world_from_part translation after refinement",
        },
        "targets": {},
    }

    for part, frame in targets:
        key = f"{part}:{frame:06d}"
        timestamp = f"{frame:06d}"
        initial_world = np.asarray(
            trajectory["frames"][timestamp]["parts"][part]["T_world_from_part"], np.float64)
        if part not in mesh_cache:
            scale = float(trajectory["scales"][part])
            raw_origin = np.asarray(trajectory["raw_mesh_origins"][part], np.float64)
            mesh, center = centered_metric_mesh(Path(cfg["mesh_dir"]) / f"{part}.glb", scale, raw_origin)
            mesh_cache[part] = {
                "mesh": mesh,
                "center": center,
                "mesh_tensors": make_mesh_tensors(mesh),
                "diameter": float(compute_mesh_diameter(
                    mesh.vertices, n_sample=min(10000, len(mesh.vertices)))),
            }
        asset = mesh_cache[part]
        recon = load_recon(cfg, timestamp, backend=cfg["recon_backend"])
        depth = apply_depth_gauge(recon["depth"], gauge, timestamp) if gauge else recon["depth"]
        rgb = load_recon_colors(recon, cfg, timestamp)
        source_views = visible_source_views(cfg, frame, part, recon["depth_hw"], args.source_views)
        T_part_from_centered = np.eye(4)
        T_part_from_centered[:3, 3] = asset["center"]
        T_centered_from_part = np.linalg.inv(T_part_from_centered)
        candidates = [{
            "source": "baseline",
            "view": None,
            "rotation_from_initial_deg": 0.0,
            "T_world_from_part": initial_world.tolist(),
        }]

        for source in source_views:
            index = int(source["index"])
            E = homogeneous_extrinsic(recon["extrinsics"][index])
            initial_camera_centered = E @ initial_world @ T_part_from_centered
            hypotheses = np.repeat(np.eye(4, dtype=np.float32)[None], len(rotations), axis=0)
            hypotheses[:, :3, :3] = rotations.astype(np.float32)
            hypotheses[:, :3, 3] = initial_camera_centered[:3, 3]
            K = np.asarray(recon["intrinsics"][index], np.float32)
            view_depth = np.asarray(depth[index], np.float32)
            xyz_map = depth2xyzmap(view_depth, K)
            refined, _ = refiner.predict(
                mesh=asset["mesh"], mesh_tensors=asset["mesh_tensors"],
                rgb=rgb[index], depth=view_depth, K=K,
                ob_in_cams=hypotheses, xyz_map=xyz_map, normal_map=None,
                glctx=glctx, mesh_diameter=asset["diameter"],
                iteration=args.iterations, get_vis=False,
            )
            refined = refined.detach().cpu().numpy().reshape(-1, 4, 4)
            for hypothesis_index, camera_centered in enumerate(refined):
                world = np.linalg.inv(E) @ camera_centered @ T_centered_from_part
                # The hybrid isolates rotation initialization.  Translation is
                # supplied by the multi-view solver, not mask-interior depth.
                world[:3, 3] = initial_world[:3, 3]
                candidates.append({
                    "source": "foundationpose_global_refiner",
                    "view": source["view"],
                    "hypothesis_index": hypothesis_index,
                    "rotation_from_initial_deg": rotation_error_deg(
                        initial_world[:3, :3], world[:3, :3]),
                    "T_world_from_part": world.tolist(),
                })
        output["targets"][key] = {
            "part": part,
            "frame": frame,
            "initial_T_world_from_part": initial_world.tolist(),
            "source_views": source_views,
            "candidate_count": len(candidates),
            "candidates": candidates,
        }
        print(f"{key}: {len(candidates)} candidates from "
              f"{[item['view'] for item in source_views]}", flush=True)

    write_json(Path(args.output).resolve(), output)
    print(f"wrote {Path(args.output).resolve()}", flush=True)


if __name__ == "__main__":
    main()
