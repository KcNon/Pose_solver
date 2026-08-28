from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from common.appearance_pose import (
    _table_support_score,
    _is_static_transition,
    _masked_photometric_correlation,
    align_pose_axis,
    candidate_local_rotations,
    carry_pose_orientation,
    normalize_similarity_to_support_plane,
    rigid_core_mask,
    select_candidate_chain,
    table_yaw_angles,
)
from common.silhouette_scale_calibration import (
    _configured_prior_cross_frame_gate,
    _scale_area_gate,
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


def test_table_yaw_angles_cover_full_turn_without_duplicate_endpoint() -> None:
    assert table_yaw_angles(90.0) == [0.0, 90.0, 180.0, 270.0]
    angles = table_yaw_angles(100.0)
    assert angles == [0.0, 90.0, 180.0, 270.0]
    assert len(set(angles)) == len(angles)


def test_rigid_core_mask_removes_thin_appendage_and_small_component() -> None:
    mask = np.zeros((60, 80), dtype=bool)
    mask[15:45, 20:50] = True
    mask[29:31, 50:75] = True
    mask[5:8, 65:68] = True

    core = rigid_core_mask(
        mask, opening_pixels=2, keep_largest=True
    )

    assert int(core[:, 55:].sum()) == 0
    assert int(core[5:8, 65:68].sum()) == 0
    assert int(core[20:40, 25:45].sum()) > 300


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


def test_chain_rejects_large_static_translation_jump() -> None:
    stable = np.eye(4)
    jumped = np.eye(4)
    jumped[0, 3] = 0.30
    candidate_rows = [
        [{"pose": stable, "score": 0.0}],
        [
            {"pose": jumped, "score": 0.1},
            {"pose": stable, "score": 0.0},
        ],
    ]
    selected, _report = select_candidate_chain(
        [100, 200],
        candidate_rows,
        transition_weight=1.0,
        max_rotation_deg_per_frame=10.0,
        max_translation_m_per_frame=0.05,
        static_ranges=[[101, 199]],
    )
    assert selected == [0, 1]


def test_hard_rate_counts_motion_frames_not_static_waiting_time() -> None:
    selected, _report = select_candidate_chain(
        [165, 230],
        [
            [{"pose": _pose_x(0), "score": 0.0}],
            [
                {"pose": _pose_x(174), "score": 1.0},
                {"pose": _pose_x(64), "score": 0.0},
            ],
        ],
        transition_weight=0.1,
        max_rotation_deg_per_frame=8.0,
        dynamic_ranges=[[206, 221]],
        hard_rotation_rate=True,
    )
    assert selected == [0, 1]


def test_photometric_correlation_distinguishes_spatial_texture_flip() -> None:
    observed = np.zeros((48, 64, 3), dtype=np.uint8)
    observed[:, :32] = [210, 40, 35]
    observed[:, 32:] = [35, 70, 210]
    mask = np.ones((48, 64), dtype=bool)
    matching = _masked_photometric_correlation(
        observed,
        observed.copy(),
        mask,
        mask,
        erosion_pixels=1,
        blur_sigma=1.0,
        minimum_pixels=100,
    )
    flipped = _masked_photometric_correlation(
        observed,
        observed[:, ::-1].copy(),
        mask,
        mask,
        erosion_pixels=1,
        blur_sigma=1.0,
        minimum_pixels=100,
    )
    assert matching is not None and flipped is not None
    assert matching > 0.95
    assert flipped < -0.8


def test_support_normalization_removes_candidate_height_bias() -> None:
    vertices = np.asarray([
        [-0.5, -0.5, -0.2],
        [-0.5, 0.5, -0.2],
        [0.5, -0.5, -0.2],
        [0.5, 0.5, -0.2],
        [-0.5, -0.5, 0.2],
        [0.5, 0.5, 0.2],
    ])
    candidate = np.eye(4)
    candidate[2, 3] = 0.035
    normal = np.asarray([0.0, 0.0, 1.0])
    point = np.zeros(3)

    before, _ = _table_support_score(
        vertices,
        candidate,
        normal,
        point,
        contact_quantile=0.01,
        gap_cap_m=0.02,
        penetration_tolerance_m=0.005,
    )
    normalized, report = normalize_similarity_to_support_plane(
        vertices,
        candidate,
        normal,
        point,
        contact_quantile=0.01,
        maximum_shift_m=0.25,
    )
    after, support = _table_support_score(
        vertices,
        normalized,
        normal,
        point,
        contact_quantile=0.01,
        gap_cap_m=0.02,
        penetration_tolerance_m=0.005,
    )

    assert report["accepted"]
    assert abs(report["translation_along_normal_m"] - 0.165) < 1e-9
    assert abs(report["contact_gap_after_m"]) < 1e-12
    assert abs(support["contact_gap_m"]) < 1e-9
    assert after > before


def test_support_normalization_rejects_implausibly_large_shift() -> None:
    vertices = np.asarray([[0.0, 0.0, -0.1], [0.0, 0.0, 0.1]])
    candidate = np.eye(4)
    candidate[2, 3] = 1.0
    normalized, report = normalize_similarity_to_support_plane(
        vertices,
        candidate,
        np.asarray([0.0, 0.0, 1.0]),
        np.zeros(3),
        contact_quantile=0.01,
        maximum_shift_m=0.05,
    )
    assert not report["accepted"]
    np.testing.assert_allclose(normalized, candidate)


def test_configured_scale_prior_rejects_cross_frame_degradation() -> None:
    baseline = {
        "frames": {
            "000010": {"optimize_iou": 0.50, "holdout_iou": 0.50},
            "000020": {"optimize_iou": 0.70, "holdout_iou": 0.70},
        }
    }
    proposed = {
        "frames": {
            "000010": {"optimize_iou": 0.47, "holdout_iou": 0.48},
            "000020": {"optimize_iou": 0.78, "holdout_iou": 0.78},
        }
    }
    passed, report = _configured_prior_cross_frame_gate(
        baseline,
        proposed,
        maximum_iou_degradation=0.01,
        minimum_improved_fraction=0.5,
    )
    assert not passed
    assert report["worst_optimize_iou_delta"] < -0.01


def test_scale_area_gate_rejects_render_loss_driven_oversizing() -> None:
    passed, baseline_error, proposed_error = _scale_area_gate(
        1.02,
        1.28,
        maximum_log_degradation=0.03,
    )
    assert not passed
    assert proposed_error > baseline_error

    improved, _, _ = _scale_area_gate(
        0.62,
        0.98,
        maximum_log_degradation=0.03,
    )
    assert improved


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
    def test_chain_rate_uses_pose_written_after_candidate_fallback(self):
        selected, _report = select_candidate_chain(
            [10, 20],
            [
                [{
                    "pose": _pose_x(170),
                    "transition_pose": _pose_x(0),
                    "score": 0.0,
                }],
                [
                    {"pose": _pose_x(170), "score": 1.0},
                    {"pose": _pose_x(10), "score": 0.0},
                ],
            ],
            transition_weight=0.0,
            max_rotation_deg_per_frame=5.0,
            hard_rotation_rate=True,
        )
        self.assertEqual(selected, [0, 1])

    def test_carry_pose_orientation_preserves_current_centre(self):
        previous = _pose_x(73)
        previous[:3, 3] = [8.0, 9.0, 10.0]
        current = _pose_x(-41)
        current[:3, 3] = [1.0, 2.0, 3.0]

        carried = carry_pose_orientation(current, previous)

        np.testing.assert_allclose(carried[:3, :3], previous[:3, :3])
        np.testing.assert_allclose(carried[:3, 3], current[:3, 3])

    def test_chain_honors_candidate_selection_gate(self):
        selected, report = select_candidate_chain(
            [10],
            [[
                {
                    "pose": _pose_x(0),
                    "score": 1.0,
                    "selection_score": -1.0e12,
                },
                {
                    "pose": _pose_x(0),
                    "score": 0.2,
                    "selection_score": 0.2,
                },
            ]],
            transition_weight=0.0,
            max_rotation_deg_per_frame=10.0,
        )
        self.assertEqual(selected, [1])
        self.assertAlmostEqual(report["path_score"], 0.2)

    def test_chain_can_traverse_soft_silhouette_gate_failure(self):
        selected, report = select_candidate_chain(
            [10, 11],
            [
                [{"pose": _pose_x(0), "score": 0.0}],
                [{
                    "pose": _pose_x(5),
                    "score": 0.5,
                    "selection_score": -0.5,
                    "candidate_gate_passed": False,
                    "semantic_candidate_gate_passed": True,
                }],
            ],
            transition_weight=0.0,
            max_rotation_deg_per_frame=10.0,
            hard_rotation_rate=True,
        )
        self.assertEqual(selected, [0, 0])
        self.assertAlmostEqual(report["path_score"], -0.5)

    def test_chain_fails_closed_when_every_candidate_is_gated(self):
        with self.assertRaisesRegex(
            RuntimeError, "no orientation candidate passes"
        ):
            select_candidate_chain(
                [10],
                [[{
                    "pose": _pose_x(0),
                    "score": 1.0,
                    "selection_score": -1.0e12,
                }]],
                transition_weight=0.0,
                max_rotation_deg_per_frame=10.0,
            )

    def test_support_axis_candidate_preserves_origin_and_faces_target(self):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler(
            "xyz", [37.0, -52.0, 19.0], degrees=True
        ).as_matrix()
        pose[:3, 3] = [0.2, -0.4, 1.3]
        axis_raw = np.array([0.0, 0.0, 1.0])
        target_world = np.array([0.1, 0.7, 0.7])
        aligned, correction_deg = align_pose_axis(
            pose, axis_raw, target_world
        )
        np.testing.assert_allclose(aligned[:3, 3], pose[:3, 3])
        np.testing.assert_allclose(
            aligned[:3, :3] @ axis_raw,
            target_world / np.linalg.norm(target_world),
            atol=1e-10,
        )
        self.assertGreater(correction_deg, 0.0)
        self.assertAlmostEqual(np.linalg.det(aligned[:3, :3]), 1.0)

    def test_opening_direction_gate_keeps_slanted_upward_candidate(self):
        table_up = np.array([0.0, 0.0, 1.0])
        opening_raw = np.array([0.0, 0.0, 1.0])
        slanted_up = _pose_x(-64.0)
        downward = _pose_x(116.0)
        slanted_alignment = float(
            np.dot(slanted_up[:3, :3] @ opening_raw, table_up)
        )
        downward_alignment = float(
            np.dot(downward[:3, :3] @ opening_raw, table_up)
        )
        self.assertGreaterEqual(slanted_alignment, 0.2)
        self.assertLess(downward_alignment, 0.2)
        self.assertLess(slanted_alignment, 1.0)

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
            return_candidates=True,
        )
        fitted_scale, _rotation, fitted_translation = decompose_similarity(
            fit["T_mesh_to_world"]
        )
        self.assertAlmostEqual(fitted_scale, scale, places=8)
        np.testing.assert_allclose(
            fitted_translation, translation, atol=0.025
        )
        self.assertLess(fit["fit_rmse"], 0.025)
        self.assertEqual(len(fit["candidate_fits"]), 4)
