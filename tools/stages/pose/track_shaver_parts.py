#!/usr/bin/env python
"""Chain-initialised two-part pose tracking for a hinged shaver.

The generic solver picks each anchor pose with a multi-basin PCA search.  That
is the right default when nothing is known about the object, but it fails on a
shaver: the cutter head is close to a solid of revolution, so several basins
score almost identically and the winner is chosen essentially at random.

This stage replaces the search with a chain:

1. ``anchor``  fit the *whole* mesh to the whole-object cloud once.  A complete
   object has no rotational ambiguity worth worrying about, and this is also
   where the raw-mesh scale is fixed.
2. ``split``   seed both parts from that single whole-object similarity.  The
   part meshes live in the same canonical frame as the whole mesh, so the seed
   is exact up to the hinge angle at the anchor frame.  Each part is then
   refined against its own cloud *locally* -- never re-initialised from PCA, so
   a near-symmetric part cannot jump basins.
3. ``track``   walk forward one frame at a time, initialising every frame from
   the previous accepted pose of the same part.

Scale is solved once at the anchor and held fixed afterwards; per-frame ICP is
rigid only.  A frame whose observation is too thin, or whose registration moves
further than the configured step limits, holds the previous pose and is marked
so downstream stages can tell propagation from observation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.backproject_utils import fuse_part_cloud
from common.gicp import multiscale_gicp, pair_quality, subsample, transform_angle
from common.io_utils import load_json, write_json
from common.mesh_align import align_mesh_to_cloud
from common.pose_transforms import (
    decompose_similarity,
    rigid_from_similarity,
    similarity_from_rigid,
    transform_points,
)
from common.trajectory_io import (
    refresh_trajectory_derived_fields,
    world_pose_record,
    write_trajectory_files,
)


# Correspondence radii must stay well below the part's own size.  A radius
# comparable to the part turns the refinement back into a global search, and a
# near-symmetric part will happily jump basins even from a good initial guess.
DEFAULT_REGISTRATION = {
    "voxel_sizes_m": [0.006, 0.003, 0.0015],
    "max_correspondence_m": [0.020, 0.010, 0.005],
    "max_iterations": 60,
    "max_points": 16000,
}

DEFAULT_GATES = {
    "minimum_cloud_points": 150,
    "minimum_observing_views": 2,
    "minimum_fitness_8mm": 0.10,
    "maximum_median_nn_m": 0.020,
    "maximum_translation_step_m": 0.050,
    "maximum_rotation_step_deg": 20.0,
}

DEFAULT_DEGENERACY = {
    "enabled": True,
    "probe_angles_deg": [30.0, 60.0, 90.0, 180.0],
    # A part that maps onto itself to within this distance under a rotation
    # carries no usable evidence about that rotation.  Keep it at or above the
    # registration's own residual, otherwise real structure is discarded.
    "threshold_m": 0.006,
    "axes": {},
}


DEFAULT_HINGE = {
    "enabled": False,
    "anchor_sweep_deg": [-70.0, 70.0],
    "anchor_step_deg": 2.0,
    # The per-frame window doubles as a rate limit: a joint cannot travel
    # further between consecutive frames than the mechanism allows, and a wide
    # window lets a decaying observation walk the angle away a few degrees at
    # a time while every single step still looks acceptable.
    "track_window_deg": 3.0,
    "track_step_deg": 0.25,
    "surface_points": 20000,
    "maximum_median_nn_m": 0.008,
    # An angle is also rejected when its residual is far worse than the run's
    # own established quality, which catches drift that no fixed threshold
    # tuned for a different sequence would.
    "residual_tolerance_factor": 2.0,
    "minimum_cloud_points": 150,
    # "parent_chain" places the child entirely from the parent and the angle.
    # "child_icp" keeps that orientation but takes the position from the
    # child's own registration, so a drifting parent corrupts only the part
    # of the child pose that the child's own geometry cannot supply anyway.
    "position_source": "parent_chain",
}


def _hinge_transform(
    axis: np.ndarray, point: np.ndarray, degrees: float
) -> np.ndarray:
    """Rotation by ``degrees`` about the joint line, in canonical mesh coords."""
    rotation = Rotation.from_rotvec(axis * np.radians(float(degrees))).as_matrix()
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    value[:3, 3] = point - rotation @ point
    return value


def _solve_hinge_angle(
    child_points: np.ndarray,
    parent_similarity: np.ndarray,
    observed: np.ndarray,
    axis: np.ndarray,
    point: np.ndarray,
    candidates: np.ndarray,
) -> tuple[float, float]:
    """Pick the joint angle whose posed child best explains the observation.

    A revolute child sitting off its own joint line moves mostly *sideways* as
    the angle changes, and a point cloud localises a body's position far more
    reliably than its orientation.  So the single joint angle is recoverable
    even when the child's own three rotations are not.
    """
    tree = cKDTree(observed)
    best_angle, best_cost = float(candidates[0]), float("inf")
    for angle in candidates:
        posed = transform_points(
            child_points, parent_similarity @ _hinge_transform(axis, point, angle)
        )
        distances, _ = cKDTree(posed).query(observed, k=1)
        cost = float(np.median(distances))
        if cost < best_cost:
            best_angle, best_cost = float(angle), cost
    del tree
    return best_angle, best_cost


def _principal_axes(points: np.ndarray) -> np.ndarray:
    """Columns are the part's principal axes, largest variance first."""
    centred = np.asarray(points, dtype=np.float64)
    centred = centred - centred.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(centred.T))
    order = np.argsort(values)[::-1]
    return vectors[:, order]


def _degenerate_axes(
    points: np.ndarray,
    axes: np.ndarray,
    settings: dict[str, Any],
) -> tuple[list[int], list[float]]:
    """Principal axes about which the part is its own rotation.

    Rotating the sampled surface about a candidate axis and measuring how far
    the result sits from the original surface answers directly whether ICP can
    ever recover that angle -- no assumption about the object is needed.
    """
    tree = cKDTree(points)
    residuals: list[float] = []
    for index in range(3):
        worst = 0.0
        for angle in settings["probe_angles_deg"]:
            rotation = Rotation.from_rotvec(
                axes[:, index] * np.radians(float(angle))
            ).as_matrix()
            distances, _ = tree.query(points @ rotation.T, k=1)
            worst = max(worst, float(np.median(distances)))
        residuals.append(worst)
    threshold = float(settings["threshold_m"])
    return [i for i, value in enumerate(residuals) if value < threshold], residuals


def _swing_twist(rotation: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, float]:
    """Split ``rotation`` into the part about ``axis`` and everything else."""
    quaternion = Rotation.from_matrix(rotation).as_quat()  # x, y, z, w
    vector, scalar = quaternion[:3], quaternion[3]
    projected = float(np.dot(vector, axis)) * axis
    twist = np.array([*projected, scalar], dtype=np.float64)
    norm = float(np.linalg.norm(twist))
    if norm < 1e-9:  # a half turn exactly off-axis has no twist component
        return rotation, 0.0
    twist_rotation = Rotation.from_quat(twist / norm)
    swing = Rotation.from_matrix(rotation) * twist_rotation.inv()
    return swing.as_matrix(), float(np.degrees(twist_rotation.magnitude()))


def _project_rotation(
    previous: np.ndarray,
    candidate: np.ndarray,
    axes: np.ndarray,
    degenerate: list[int],
) -> tuple[np.ndarray, float]:
    """Keep the observable part of a registration and discard the rest.

    The translation is left untouched: the canonical points are centred on the
    part origin, so the registered translation *is* the observed centroid and
    a rotation about that centroid cannot move it.
    """
    if not degenerate:
        return candidate, 0.0
    result = candidate.copy()
    if len(degenerate) >= 2:
        # Two free axes leave the orientation essentially unconstrained; there
        # is nothing to keep, so carry the previous orientation forward.
        result[:3, :3] = previous[:3, :3]
        relative = previous[:3, :3].T @ candidate[:3, :3]
        return result, float(
            np.degrees(Rotation.from_matrix(relative).magnitude())
        )
    relative = previous[:3, :3].T @ candidate[:3, :3]
    swing, dropped = _swing_twist(relative, axes[:, degenerate[0]])
    result[:3, :3] = previous[:3, :3] @ swing
    return result, dropped


def _frame_key(frame: int) -> str:
    return f"{frame:06d}"


def _load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    return image.any(axis=2) if image.ndim == 3 else image > 0


class FrameSource:
    """Reads one frame's reconstruction and resamples masks onto it."""

    def __init__(self, config: dict[str, Any]):
        self.depth_root = Path(config["depth_root"])
        self.views = [str(view) for view in config["views"]]
        self.whole_masks = Path(config["whole_masks"])
        self.part_mask_roots = [
            Path(value) for value in config["part_mask_roots"]
        ]
        self.cloud = config.get("cloud", {})
        # Every part of a frame reads the same reconstruction; keep only the
        # current one so memory stays flat over a long sequence.
        self._cached_frame: int | None = None
        self._cached: dict[str, np.ndarray] | None = None

    def _reconstruction(self, frame: int) -> dict[str, np.ndarray]:
        if self._cached_frame == frame and self._cached is not None:
            return self._cached
        value = self._read_reconstruction(frame)
        self._cached_frame, self._cached = frame, value
        return value

    def _read_reconstruction(self, frame: int) -> dict[str, np.ndarray]:
        path = self.depth_root / _frame_key(frame) / "predictions.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        names = [str(value) for value in data["view_names"]]
        if names != self.views:
            raise ValueError(
                f"{path}: reconstruction views {names} do not match the "
                f"configured views {self.views}"
            )
        images = data["images"]
        colors = np.stack([
            (np.transpose(images[index], (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
            if images[index].max() <= 1.5
            else np.transpose(images[index], (1, 2, 0)).astype(np.uint8)
            for index in range(len(self.views))
        ])
        return {
            "depth": data["depth"][..., 0],
            "intrinsic": data["intrinsic"],
            "extrinsic": data["extrinsic"],
            "conf": data["depth_conf"],
            "colors": colors,
        }

    def _part_mask(self, part: str, frame: int, view: str) -> np.ndarray | None:
        """Whole-object masks live in a per-view tree, part tracks per frame."""
        if part == "__whole__":
            return _load_mask(self.whole_masks / view / f"{_frame_key(frame)}.png")
        for root in self.part_mask_roots:
            mask = _load_mask(root / part / _frame_key(frame) / f"{view}.png")
            if mask is not None:
                return mask
        return None

    def cloud_for(self, part: str, frame: int) -> tuple[np.ndarray, dict[str, Any]]:
        recon = self._reconstruction(frame)
        height, width = recon["depth"].shape[1:]
        masks = []
        observing = []
        for view in self.views:
            mask = self._part_mask(part, frame, view)
            if mask is None or not mask.any():
                masks.append(np.zeros((height, width), dtype=bool))
                continue
            observing.append(view)
            masks.append(
                cv2.resize(
                    mask.astype(np.uint8), (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            )
        points, _colors, stats = fuse_part_cloud(
            recon["depth"], recon["colors"], recon["intrinsic"],
            recon["extrinsic"], recon["conf"], masks,
            conf_mode=str(self.cloud.get("conf_mode", "adaptive")),
            conf_quantile=float(self.cloud.get("conf_quantile", 0.25)),
            stride=int(self.cloud.get("stride", 1)),
            max_pts=int(self.cloud.get("max_points", 120000)),
            seed=int(self.cloud.get("seed", 0)),
            views=self.views,
        )
        return points, {
            "observing_views": observing,
            "per_view_points": {key: int(value) for key, value in stats.items()},
        }


def _anchor_similarity(
    mesh: trimesh.Trimesh,
    cloud: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the complete mesh once, free scale, keeping every PCA basin."""
    fit = align_mesh_to_cloud(
        mesh, cloud,
        n_mesh_sample=int(settings.get("mesh_samples", 40000)),
        n_obs_max=int(settings.get("observation_samples", 16000)),
        coarse_iters=int(settings.get("coarse_iterations", 30)),
        fine_iters=int(settings.get("fine_iterations", 100)),
        seed=int(settings.get("seed", 0)),
        return_candidates=True,
    )
    report = {
        "scale": float(fit["scale"]),
        "fit_rmse_m": float(fit["fit_rmse"]),
        "icp_cost": float(fit["icp_cost"]),
        "observation_points": int(fit["n_obs"]),
        "candidates": [
            {
                "label": candidate["label"],
                "fit_rmse_m": float(candidate["fit_rmse"]),
                "icp_cost": float(candidate["icp_cost"]),
            }
            for candidate in fit["candidate_fits"]
        ],
    }
    return np.asarray(fit["T_mesh_to_world"], dtype=np.float64), report


def _register(
    canonical: np.ndarray,
    pose: np.ndarray,
    observed: np.ndarray,
    registration: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine ``pose`` so the posed canonical points land on ``observed``.

    The delta is solved in world space with the posed model as the source, so
    the initial guess is the identity and GICP only has to close the residual
    the previous frame left behind.
    """
    source = subsample(
        transform_points(canonical, pose),
        int(registration.get("max_points", 16000)),
        seed=0,
    )
    target = subsample(
        observed, int(registration.get("max_points", 16000)), seed=1
    )
    delta, quality = multiscale_gicp(source, target, np.eye(4), registration)
    return delta @ pose, quality


def _step(previous: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """Displacement of the part frame, not of the world-space delta.

    The delta's translation column mixes in the lever arm from the world
    origin, so a small rotation of a distant object reads as a huge jump.  The
    part origin's own displacement is the quantity the step limits describe.
    """
    translation = float(
        np.linalg.norm(candidate[:3, 3] - previous[:3, 3])
    )
    rotation = transform_angle(candidate @ np.linalg.inv(previous))
    return translation, rotation


def _gate(
    quality: dict[str, Any],
    translation_step: float,
    rotation_step: float,
    points: int,
    observing: int,
    gates: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if points < int(gates["minimum_cloud_points"]):
        reasons.append(f"cloud_points<{gates['minimum_cloud_points']}")
    if observing < int(gates["minimum_observing_views"]):
        reasons.append(f"observing_views<{gates['minimum_observing_views']}")
    if quality["fitness_8mm"] < float(gates["minimum_fitness_8mm"]):
        reasons.append(f"fitness<{gates['minimum_fitness_8mm']}")
    if quality["median_nn_m"] > float(gates["maximum_median_nn_m"]):
        reasons.append(f"median_nn>{gates['maximum_median_nn_m']}")
    if translation_step > float(gates["maximum_translation_step_m"]):
        reasons.append(f"translation_step>{gates['maximum_translation_step_m']}")
    if rotation_step > float(gates["maximum_rotation_step_deg"]):
        reasons.append(f"rotation_step>{gates['maximum_rotation_step_deg']}")
    return not reasons, reasons


def track(config: dict[str, Any]) -> dict[str, Any]:
    parts = {str(name): Path(value) for name, value in config["parts"].items()}
    reference = str(config["reference_part"])
    if reference not in parts:
        raise ValueError(f"reference_part {reference!r} is not a tracked part")
    anchor_frame = int(config["anchor_frame"])
    start = int(config["frame_range"][0])
    end = int(config["frame_range"][1])
    if not start <= anchor_frame <= end:
        raise ValueError("anchor_frame must lie inside frame_range")
    shared = {**DEFAULT_REGISTRATION, **config.get("registration", {})}
    overrides = config.get("part_registration", {})
    unknown = set(overrides).difference(parts)
    if unknown:
        raise ValueError(f"part_registration for unknown parts: {sorted(unknown)}")
    registration = {
        name: {**shared, **overrides.get(name, {})} for name in parts
    }
    for name, values in registration.items():
        if len(values["voxel_sizes_m"]) != len(values["max_correspondence_m"]):
            raise ValueError(
                f"{name}: registration voxel_sizes_m and max_correspondence_m "
                "must have the same length"
            )
    gates = {**DEFAULT_GATES, **config.get("gates", {})}
    anchor_settings = config.get("anchor", {})
    surface_points = int(config.get("surface_points", 30000))

    source = FrameSource(config)
    meshes = {
        name: trimesh.load(path, force="mesh", process=False)
        for name, path in parts.items()
    }
    origins = {
        name: np.asarray(mesh.centroid, dtype=np.float64)
        for name, mesh in meshes.items()
    }

    # --- 1. anchor: one whole-object fit fixes the scale and the basin -------
    whole_mesh = trimesh.load(config["whole_mesh"], force="mesh", process=False)
    whole_cloud, whole_stats = source.cloud_for("__whole__", anchor_frame)
    if not len(whole_cloud):
        raise RuntimeError(f"empty whole-object cloud at frame {anchor_frame}")
    whole_similarity, anchor_report = _anchor_similarity(
        whole_mesh, whole_cloud, anchor_settings
    )
    scale, _rotation, _translation = decompose_similarity(whole_similarity)
    anchor_report["observing_views"] = len(whole_stats["observing_views"])
    print(
        f"anchor {anchor_frame}: scale={scale:.6f} "
        f"rmse={anchor_report['fit_rmse_m'] * 1000:.2f}mm "
        f"cloud={len(whole_cloud)}pts "
        f"views={anchor_report['observing_views']}",
        flush=True,
    )

    # --- 2. split: seed both parts from the whole fit, refine locally --------
    canonical = {
        name: scale * (
            np.asarray(
                trimesh.sample.sample_surface(
                    meshes[name], surface_points,
                    seed=np.random.default_rng(index),
                )[0],
                dtype=np.float64,
            ) - origins[name]
        )
        for index, name in enumerate(parts)
    }
    # Which rotations can this object's geometry actually pin down?
    degeneracy = {**DEFAULT_DEGENERACY, **config.get("degeneracy", {})}
    part_axes: dict[str, np.ndarray] = {}
    degenerate: dict[str, list[int]] = {}
    degeneracy_report: dict[str, Any] = {}
    for name in parts:
        axes = _principal_axes(canonical[name])
        part_axes[name] = axes
        if not degeneracy["enabled"]:
            free, residuals = [], [None, None, None]
        elif name in degeneracy["axes"]:
            free = [int(value) for value in degeneracy["axes"][name]]
            _, residuals = _degenerate_axes(canonical[name], axes, degeneracy)
        else:
            free, residuals = _degenerate_axes(canonical[name], axes, degeneracy)
        degenerate[name] = free
        degeneracy_report[name] = {
            "degenerate_axes": free,
            "self_similarity_m": residuals,
            "threshold_m": degeneracy["threshold_m"],
            "policy": (
                "orientation_held" if len(free) >= 2
                else "twist_dropped" if free else "full_rotation"
            ),
        }
        print(
            f"  degeneracy {name}: axes={free} "
            f"residuals={[None if r is None else round(r * 1000, 2) for r in residuals]}mm "
            f"-> {degeneracy_report[name]['policy']}",
            flush=True,
        )

    # A revolute child is driven by its parent plus one angle, not by six free
    # parameters of its own.
    hinge = {**DEFAULT_HINGE, **config.get("hinge", {})}
    hinge_child = str(hinge.get("child", "")) if hinge["enabled"] else ""
    if hinge_child:
        if hinge_child not in parts:
            raise ValueError(f"hinge child {hinge_child!r} is not a tracked part")
        hinge_parent = str(hinge["parent"])
        if hinge_parent not in parts or hinge_parent == hinge_child:
            raise ValueError("hinge parent must be a different tracked part")
        hinge_axis = np.asarray(hinge["axis"], dtype=np.float64)
        hinge_axis = hinge_axis / np.linalg.norm(hinge_axis)
        hinge_point = np.asarray(hinge["point"], dtype=np.float64)
        hinge_raw = np.asarray(
            trimesh.sample.sample_surface(
                meshes[hinge_child], int(hinge["surface_points"]),
                seed=np.random.default_rng(7),
            )[0],
            dtype=np.float64,
        )
        order = [hinge_parent] + [n for n in parts if n != hinge_parent]
    else:
        order = list(parts)

    poses: dict[str, np.ndarray] = {}
    split_report: dict[str, Any] = {}
    for name in order:
        seeded = rigid_from_similarity(whole_similarity, origins[name])
        cloud, stats = source.cloud_for(name, anchor_frame)
        entry: dict[str, Any] = {
            "cloud_points": int(len(cloud)),
            "observing_views": len(stats["observing_views"]),
            "seed": "whole_object_similarity",
        }
        if hinge_child and name == hinge_child:
            parent_similarity = similarity_from_rigid(
                poses[hinge_parent], scale, origins[hinge_parent]
            )
            sweep = np.arange(
                float(hinge["anchor_sweep_deg"][0]),
                float(hinge["anchor_sweep_deg"][1]) + 1e-9,
                float(hinge["anchor_step_deg"]),
            )
            angle, cost = _solve_hinge_angle(
                hinge_raw, parent_similarity, cloud,
                hinge_axis, hinge_point, sweep,
            )
            hinge_angle = angle
            poses[name] = rigid_from_similarity(
                parent_similarity @ _hinge_transform(hinge_axis, hinge_point, angle),
                origins[name],
            )
            entry.update(
                accepted=cost <= float(hinge["maximum_median_nn_m"]),
                reasons=[] if cost <= float(hinge["maximum_median_nn_m"]) else ["hinge_residual"],
                seed="hinge_from_parent", hinge_angle_deg=angle,
                hinge_residual_m=cost,
                rotation_from_seed_deg=0.0, translation_from_seed_m=0.0,
            )
        elif len(cloud) < int(gates["minimum_cloud_points"]):
            poses[name] = seeded
            entry.update(accepted=False, reasons=["cloud_points"])
        else:
            registered, quality = _register(
                canonical[name], seeded, cloud, registration[name]
            )
            refined, dropped = _project_rotation(
                seeded, registered, part_axes[name], degenerate[name]
            )
            translation_step, rotation_step = _step(seeded, refined)
            entry["dropped_rotation_deg"] = dropped
            accepted, reasons = _gate(
                quality, translation_step, rotation_step, len(cloud),
                len(stats["observing_views"]), gates,
            )
            poses[name] = refined if accepted else seeded
            entry.update(
                accepted=accepted, reasons=reasons,
                fitness_8mm=quality["fitness_8mm"],
                median_nn_m=quality["median_nn_m"],
                trimmed_rmse_m=quality["trimmed_rmse_m"],
                translation_from_seed_m=translation_step,
                rotation_from_seed_deg=rotation_step,
            )
        split_report[name] = entry
        print(
            f"  split {name}: accepted={entry['accepted']} "
            f"cloud={entry['cloud_points']}pts "
            f"views={entry['observing_views']} "
            f"drot={entry.get('rotation_from_seed_deg', 0.0):.2f}deg "
            f"dtrans={entry.get('translation_from_seed_m', 0.0) * 1000:.1f}mm"
            + (f" reasons={entry['reasons']}" if entry.get("reasons") else ""),
            flush=True,
        )

    # --- 3. track: every frame initialised from the previous accepted pose ---
    trajectory: dict[str, Any] = {
        "schema_version": 1,
        "source": "track_shaver_parts",
        "parts": list(parts),
        "reference_part": reference,
        "anchor_frame": anchor_frame,
        "scales": {name: float(scale) for name in parts},
        "raw_mesh_origins": {
            name: origins[name].tolist() for name in parts
        },
        "frames": {},
    }
    frame_reports: dict[str, Any] = {}
    holds = {name: 0 for name in parts}
    hinge_residuals: list[float] = []
    for frame in range(anchor_frame, end + 1):
        key = _frame_key(frame)
        record: dict[str, Any] = {"parts": {}}
        report: dict[str, Any] = {}
        for name in order:
            if frame == anchor_frame:
                entry = {
                    "state": "anchor",
                    "source": "whole_object_split",
                    "observing_views": split_report[name]["observing_views"],
                    "accepted": True,
                }
            else:
                cloud, stats = source.cloud_for(name, frame)
                observing = len(stats["observing_views"])
                previous = poses[name]
                dropped = 0.0
                if hinge_child and name == hinge_child:
                    parent_similarity = similarity_from_rigid(
                        poses[hinge_parent], scale, origins[hinge_parent]
                    )
                    if len(cloud) < int(hinge["minimum_cloud_points"]):
                        accepted, reasons, cost = False, ["cloud_points"], None
                        angle = hinge_angle
                    else:
                        window = float(hinge["track_window_deg"])
                        step = float(hinge["track_step_deg"])
                        angle, cost = _solve_hinge_angle(
                            hinge_raw, parent_similarity, cloud,
                            hinge_axis, hinge_point,
                            np.arange(hinge_angle - window,
                                      hinge_angle + window + 1e-9, step),
                        )
                        reasons = []
                        if cost > float(hinge["maximum_median_nn_m"]):
                            reasons.append("hinge_residual")
                        tolerance = float(hinge["residual_tolerance_factor"])
                        if hinge_residuals and tolerance > 0.0:
                            budget = tolerance * float(np.median(hinge_residuals))
                            if cost > budget:
                                reasons.append("hinge_residual_degraded")
                        accepted = not reasons
                    if accepted:
                        hinge_angle = angle
                        hinge_residuals.append(cost)
                        holds[name] = 0
                    else:
                        holds[name] += 1
                    # The child rides its parent every frame regardless: even a
                    # stale joint angle beats a pose frozen in world space.
                    chained = rigid_from_similarity(
                        parent_similarity
                        @ _hinge_transform(hinge_axis, hinge_point, hinge_angle),
                        origins[name],
                    )
                    if str(hinge["position_source"]) == "child_icp" and len(cloud):
                        # Canonical points are centred on the part origin, so a
                        # registration's translation is the observed centroid --
                        # the one quantity a featureless child does pin down.
                        registered, _quality = _register(
                            canonical[name], previous, cloud, registration[name]
                        )
                        chained[:3, 3] = registered[:3, 3]
                    poses[name] = chained
                    entry = {
                        "state": "tracked" if accepted else "held_angle",
                        "source": "hinge_from_parent",
                        "observing_views": observing,
                        "accepted": accepted,
                        "reasons": reasons,
                        "cloud_points": int(len(cloud)),
                        "consecutive_holds": holds[name],
                        "hinge_angle_deg": hinge_angle,
                        "hinge_residual_m": cost,
                    }
                    record["parts"][name] = world_pose_record(
                        poses[name], state=entry["state"],
                        source=entry["source"],
                        observing_views=entry["observing_views"],
                    )
                    record["parts"][name]["pose_valid"] = bool(entry["accepted"])
                    record["parts"][name]["hinge_angle_deg"] = hinge_angle
                    record["parts"][name]["S_world_from_raw_mesh"] = (
                        similarity_from_rigid(poses[name], scale, origins[name])
                    ).tolist()
                    report[name] = entry
                    continue
                if len(cloud) < int(gates["minimum_cloud_points"]):
                    accepted, reasons, quality = False, ["cloud_points"], {}
                    candidate = previous
                    translation_step = rotation_step = 0.0
                else:
                    registered, quality = _register(
                        canonical[name], previous, cloud, registration[name]
                    )
                    # Discard the unobservable rotation before gating, so a
                    # part whose orientation geometry cannot resolve still
                    # follows the motion instead of freezing in place.
                    candidate, dropped = _project_rotation(
                        previous, registered, part_axes[name], degenerate[name]
                    )
                    translation_step, rotation_step = _step(previous, candidate)
                    accepted, reasons = _gate(
                        quality, translation_step, rotation_step,
                        len(cloud), observing, gates,
                    )
                if accepted:
                    poses[name] = candidate
                    holds[name] = 0
                else:
                    holds[name] += 1
                entry = {
                    "state": "tracked" if accepted else "held",
                    "source": "chain_icp" if accepted else "previous_frame",
                    "observing_views": observing,
                    "accepted": accepted,
                    "reasons": reasons,
                    "cloud_points": int(len(cloud)),
                    "consecutive_holds": holds[name],
                    "translation_step_m": translation_step,
                    "rotation_step_deg": rotation_step,
                    "dropped_rotation_deg": dropped,
                    "fitness_8mm": quality.get("fitness_8mm"),
                    "median_nn_m": quality.get("median_nn_m"),
                    "trimmed_rmse_m": quality.get("trimmed_rmse_m"),
                }
            record["parts"][name] = world_pose_record(
                poses[name],
                state=entry["state"],
                source=entry["source"],
                observing_views=entry["observing_views"],
            )
            record["parts"][name]["pose_valid"] = bool(entry["accepted"])
            record["parts"][name]["S_world_from_raw_mesh"] = similarity_from_rigid(
                poses[name], scale, origins[name]
            ).tolist()
            report[name] = entry
        trajectory["frames"][key] = record
        frame_reports[key] = report
        flags = "  ".join(
            f"{name}:{report[name]['state'][0]}"
            f"{'' if report[name]['accepted'] else '(' + ','.join(report[name]['reasons']) + ')'}"
            for name in parts
        )
        print(f"frame {key}  {flags}", flush=True)

    refresh_trajectory_derived_fields(trajectory)
    accepted_counts = {
        name: sum(
            1 for value in frame_reports.values() if value[name]["accepted"]
        )
        for name in parts
    }
    summary = {
        "anchor": anchor_report,
        "degeneracy": degeneracy_report,
        "split": split_report,
        "scale": float(scale),
        "frames": len(frame_reports),
        "accepted": accepted_counts,
        "accepted_fraction": {
            name: accepted_counts[name] / max(len(frame_reports), 1)
            for name in parts
        },
        "per_frame": frame_reports,
    }
    return {"trajectory": trajectory, "report": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chain-initialised two-part shaver pose tracking",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-trajectory")
    parser.add_argument("--report")
    parser.add_argument("--anchor-frame", type=int)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_json(args.config)
    if args.anchor_frame is not None:
        config["anchor_frame"] = args.anchor_frame
    if args.frame_start is not None:
        config["frame_range"][0] = args.frame_start
    if args.frame_end is not None:
        config["frame_range"][1] = args.frame_end
    result = track(config)
    trajectory_path = Path(
        args.output_trajectory or config["output_trajectory"]
    )
    write_trajectory_files(result["trajectory"], trajectory_path)
    report_path = Path(args.report or config.get("report", ""))
    if str(report_path):
        write_json(report_path, result["report"])
    accepted = result["report"]["accepted_fraction"]
    print(
        "accepted "
        + "  ".join(f"{name}={value:.3f}" for name, value in accepted.items()),
        flush=True,
    )
    print(f"trajectory -> {trajectory_path}", flush=True)


if __name__ == "__main__":
    main()
