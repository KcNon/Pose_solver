from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from common.physics_pose_projection import (
    axis_angle_matrix,
    project_pose_without_axial_yaw,
    write_physics_refined_trajectory,
)


class PhysicsPoseProjectionTests(unittest.TestCase):
    def test_projection_keeps_visual_axial_yaw(self) -> None:
        visual = np.eye(4)
        visual[:3, :3] = axis_angle_matrix(np.asarray([0.0, 0.0, 1.0]), 0.7)
        physical = visual.copy()
        physical[:3, :3] = axis_angle_matrix(np.asarray([1.0, 0.0, 0.0]), 0.1) @ physical[:3, :3]
        physical[:3, 3] = [0.001, -0.002, 0.003]
        projected = project_pose_without_axial_yaw(
            visual, physical, np.asarray([0.0, 0.0, -1.0])
        )
        np.testing.assert_allclose(projected[:3, 3], physical[:3, 3])
        np.testing.assert_allclose(
            projected[:3, :3] @ np.asarray([0.0, 0.0, -1.0]),
            physical[:3, :3] @ np.asarray([0.0, 0.0, -1.0]),
            atol=1e-10,
        )

    def test_refined_trajectory_ramps_and_preserves_transform_convention(self) -> None:
        identity = np.eye(4)
        trajectory = {
            "parts": ["nozzle", "body"],
            "scales": {"nozzle": 2.0, "body": 1.0},
            "raw_mesh_origins": {"nozzle": [1.0, 0.0, 0.0], "body": [0.0, 0.0, 0.0]},
            "frames": {},
        }
        for frame_id in range(4):
            trajectory["frames"][f"{frame_id:06d}"] = {
                "parts": {
                    "body": {"T_world_from_part": identity.tolist()},
                    "nozzle": {
                        "source": "visual",
                        "T_body_from_part": identity.tolist(),
                        "T_world_from_part": identity.tolist(),
                        "S_world_from_raw_mesh": identity.tolist(),
                    },
                }
            }
        refined = identity.copy()
        refined[0, 3] = 0.004
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trajectory.json"
            summary = write_physics_refined_trajectory(
                trajectory,
                moving_part="nozzle",
                reference_part="body",
                visual_body_pose=identity,
                refined_body_pose=refined,
                apply_frame_range=[1, 3],
                report_path=Path(directory) / "report.json",
                output_path=output,
            )
            self.assertEqual(summary["changed_frame_count"], 3)
            record = __import__("json").loads(output.read_text())
            self.assertAlmostEqual(record["frames"]["000002"]["parts"]["nozzle"]["T_body_from_part"][0][3], 0.002)
            world = np.asarray(record["frames"]["000003"]["parts"]["nozzle"]["T_world_from_part"])
            render = np.asarray(record["frames"]["000003"]["parts"]["nozzle"]["S_world_from_raw_mesh"])
            canonical = np.diag([2.0, 2.0, 2.0, 1.0])
            canonical[:3, 3] = [-2.0, 0.0, 0.0]
            np.testing.assert_allclose(render, world @ canonical)


if __name__ == "__main__":
    unittest.main()
