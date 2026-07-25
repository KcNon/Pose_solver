from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from common.appearance_pose import (
    candidate_local_rotations,
    select_candidate_chain,
)
from common.calibration_cache import fingerprint_files


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
