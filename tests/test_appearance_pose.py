from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from common.appearance_pose import (
    _is_static_transition,
    candidate_local_rotations,
    select_candidate_chain,
)
from common.calibration_cache import fingerprint_files
from common.mesh_align import align_mesh_to_cloud
from common.pose_transforms import decompose_similarity
import trimesh


def _pose_x(angle_deg: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("x", angle_deg, degrees=True).as_matrix()
    return pose


def test_candidate_modes_are_proper_rotations_and_include_axis_flip() -> None:
    axis = np.array([0.0, 1.0, 0.0])
    axial = candidate_local_rotations(axis, "axial", 15.0)
    flipped = candidate_local_rotations(axis, "axis_flip", 30.0)

    assert len(axial) == 24
    assert len(flipped) == 2
    for candidate in axial + flipped:
        rotation = candidate["local_transform"][:3, :3]
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        assert np.isclose(np.linalg.det(rotation), 1.0)

    flip_rotation = next(row for row in flipped if row["axis_flipped"])[
        "local_transform"
    ][:3, :3]
    assert np.allclose(flip_rotation @ axis, -axis, atol=1e-7)


def test_chain_rejects_geometric_axis_flip_using_motion_prior() -> None:
    anchor_frames = [10, 20]
    candidate_rows = [
        [{"pose": _pose_x(0), "score": 0.0}],
        [
            {"pose": _pose_x(170), "score": 0.0},
            {"pose": _pose_x(10), "score": -0.05},
        ],
    ]
    selected, report = select_candidate_chain(
        anchor_frames,
        candidate_rows,
        transition_weight=1.0,
        max_rotation_deg_per_frame=5.0,
    )
    assert selected == [0, 1]
    assert report["path_score"] > -0.2


def test_fingerprint_changes_for_content_and_observation_stat(tmp_path: Path) -> None:
    content = tmp_path / "cloud.ply"
    observation = tmp_path / "mask.png"
    content.write_bytes(b"first")
    observation.write_bytes(b"mask")
    kwargs = {
        "config": {"states": {"object": {"appearance": {"enabled": True}}}},
        "content_files": [("cloud", content)],
        "stat_files": [("mask", observation)],
    }
    first = fingerprint_files(**kwargs)["sha256"]

    content.write_bytes(b"second")
    second = fingerprint_files(**kwargs)["sha256"]
    assert second != first

    stat = observation.stat()
    os.utime(observation, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    third = fingerprint_files(**kwargs)["sha256"]
    assert third != second


class CalibrationRegressionTests(unittest.TestCase):
    def test_static_transition_includes_dynamic_boundary_anchors(self):
        self.assertTrue(_is_static_transition(95, 206, [[96, 205]]))
        self.assertFalse(_is_static_transition(95, 206, [[120, 180]]))

    def test_fixed_scale_alignment_keeps_requested_scale(self):
        mesh = trimesh.creation.box(extents=[1.0, 0.7, 0.35])
        np.random.seed(13)
        raw, _ = trimesh.sample.sample_surface(mesh, 5000)
        scale = 0.31
        rotation = Rotation.from_euler(
            "xyz", [18.0, -11.0, 27.0], degrees=True
        ).as_matrix()
        translation = np.asarray([0.25, -0.12, 0.7])
        observed = scale * (raw @ rotation.T) + translation
        fit = align_mesh_to_cloud(
            mesh,
            observed,
            n_mesh_sample=8000,
            n_obs_max=5000,
            coarse_iters=20,
            fine_iters=80,
            seed=17,
            fixed_scale=scale,
        )
        fitted_scale, _rotation, fitted_translation = decompose_similarity(
            fit["T_mesh_to_world"]
        )
        self.assertAlmostEqual(fitted_scale, scale, places=8)
        np.testing.assert_allclose(
            fitted_translation, translation, atol=0.025
        )
        self.assertLess(fit["fit_rmse"], 0.025)
