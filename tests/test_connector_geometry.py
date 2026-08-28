from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from common.connector_geometry import (
    evaluate_connector_trajectory,
    fit_cylindrical_axis_from_slab,
    validate_connector_config,
)


def trajectory(moving_poses: list[np.ndarray]) -> dict:
    frames = {}
    for frame, moving in enumerate(moving_poses):
        frames[f"{frame:06d}"] = {
            "parts": {
                "body": {"T_world_from_part": np.eye(4).tolist()},
                "tip": {"T_world_from_part": moving.tolist()},
            }
        }
    return {
        "parts": ["body", "tip"],
        "scales": {"body": 1.0, "tip": 1.0},
        "raw_mesh_origins": {
            "body": [0.0, 0.0, 0.0],
            "tip": [0.0, 0.0, 0.0],
        },
        "frames": frames,
    }


def insert_connector() -> dict:
    return {
        "type": "insert",
        "reference_part": "body",
        "moving_part": "tip",
        "reference_axis_part": [0.0, 0.0, 1.0],
        "moving_axis_part": [0.0, 0.0, 1.0],
        "reference_origin_part_m": [0.0, 0.0, 0.0],
        "moving_origin_part_m": [0.0, 0.0, 0.0],
        "validation_frame_range": [0, 0],
        "maximum_axis_angle_deg": 1.0,
        "maximum_radial_offset_m": 0.001,
        "insertion_length_m": 0.02,
        "radial_clearance_m": 0.002,
        "collision_validation": {"passed": True},
    }


class ConnectorGeometryTest(unittest.TestCase):
    def test_cylindrical_axis_is_fitted_from_mesh_cross_sections(self) -> None:
        axial = np.repeat(np.linspace(-1.0, 1.0, 9), 200)
        angle = np.tile(np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False), 9)
        vertices = np.column_stack((
            axial,
            0.1 * axial + 0.2 + 0.3 * np.cos(angle),
            -0.05 * axial + 0.3 + 0.3 * np.sin(angle),
        ))
        report = fit_cylindrical_axis_from_slab(
            vertices,
            selector_axis="x",
            minimum=-1.0,
            maximum=1.0,
            direction_sign=-1.0,
            origin_coordinate=1.0,
            bins=8,
            minimum_points_per_bin=50,
        )
        expected_axis = np.asarray([-1.0, -0.1, 0.05])
        expected_axis /= np.linalg.norm(expected_axis)
        np.testing.assert_allclose(
            report["axis_part"], expected_axis, atol=2e-3
        )
        np.testing.assert_allclose(
            report["origin_raw"], [1.0, 0.3, 0.25], atol=2e-3
        )

    def test_aligned_insert_with_clearance_is_ready(self) -> None:
        connector = insert_connector()
        report = evaluate_connector_trajectory(
            "insert", connector, trajectory([np.eye(4)])
        )
        self.assertTrue(report["kinematic_alignment_passed"])
        self.assertTrue(report["simulation_ready"])
        self.assertAlmostEqual(
            report["summary"]["minimum_clearance_margin_m"], 0.002
        )

    def test_radial_error_fails_clearance_and_pose_limits(self) -> None:
        moving = np.eye(4)
        moving[0, 3] = 0.003
        report = evaluate_connector_trajectory(
            "insert", insert_connector(), trajectory([moving])
        )
        self.assertFalse(report["simulation_ready"])
        self.assertIn("radial_offset_exceeds_limit", report["failures"])
        self.assertIn("insufficient_radial_clearance", report["failures"])

    def test_screw_fails_closed_without_manufacturing_metadata(self) -> None:
        connector = insert_connector()
        connector["type"] = "screw"
        connector["thread"] = {}
        report = evaluate_connector_trajectory(
            "thread", connector, trajectory([np.eye(4)])
        )
        self.assertTrue(report["kinematic_alignment_passed"])
        self.assertFalse(report["manufacturing_metadata_complete"])
        self.assertFalse(report["simulation_ready"])
        self.assertIn("thread.pitch_m", report["missing_metadata"])

    def test_right_handed_helix_matches_pitch(self) -> None:
        first = np.eye(4)
        second = np.eye(4)
        second[:3, :3] = Rotation.from_euler(
            "z", 90.0, degrees=True
        ).as_matrix()
        second[2, 3] = 0.0025
        connector = insert_connector()
        connector.update({
            "type": "screw",
            "validation_frame_range": [0, 1],
            "thread": {
                "pitch_m": 0.01,
                "handedness": "right",
                "reference_zero_direction_part": [1.0, 0.0, 0.0],
                "moving_zero_direction_part": [1.0, 0.0, 0.0],
                "entry_phase_rad": 0.0,
                "maximum_helical_residual_m": 1e-5,
            },
        })
        report = evaluate_connector_trajectory(
            "thread", connector, trajectory([first, second])
        )
        self.assertTrue(report["simulation_ready"])
        self.assertAlmostEqual(
            report["thread"]["maximum_helical_residual_m"], 0.0
        )

    def test_config_accepts_incomplete_dimensions_for_diagnostics(self) -> None:
        connector = insert_connector()
        connector["radial_clearance_m"] = None
        validate_connector_config(
            {"insert": connector},
            parts=["body", "tip"],
            frame_start=0,
            frame_end=0,
        )

    def test_diagnostic_ranges_can_extend_before_readiness_range(self) -> None:
        poses = []
        for offset in (0.01, 0.005, 0.0):
            pose = np.eye(4)
            pose[0, 3] = offset
            poses.append(pose)
        connector = insert_connector()
        connector["validation_frame_range"] = [2, 2]
        connector["diagnostic_ranges"] = {"approach": [0, 1]}
        report = evaluate_connector_trajectory(
            "insert", connector, trajectory(poses)
        )
        self.assertTrue(report["simulation_ready"])
        self.assertAlmostEqual(
            report["diagnostic_ranges"]["approach"][
                "maximum_radial_offset_m"
            ],
            0.01,
        )

    def test_low_quality_connector_axis_fit_fails_closed(self) -> None:
        connector = insert_connector()
        connector["geometry_evidence"] = {
            "maximum_rms_residual_m": 0.002,
            "reference": {"rms_residual_m": 0.0005},
            "moving": {"rms_residual_m": 0.009},
        }
        report = evaluate_connector_trajectory(
            "insert", connector, trajectory([np.eye(4)])
        )
        self.assertFalse(report["geometry_evidence"]["passed"])
        self.assertIn("connector_geometry_evidence_failed", report["failures"])


if __name__ == "__main__":
    unittest.main()
