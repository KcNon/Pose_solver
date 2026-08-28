import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tools.diagnostics.retime_tracking_endpoint_rotation import (
    retime_rotation,
    rotation_fraction,
)


def make_pose(yaw: float = 0.0) -> list[list[float]]:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
    pose[:3, 3] = [0.1, -0.2, 1.0]
    return pose.tolist()


class RetimeEndpointRotationTests(unittest.TestCase):
    def test_delayed_fraction(self):
        self.assertEqual(rotation_fraction(5, 5, 9, 1.0), 0.0)
        self.assertEqual(rotation_fraction(7, 5, 9, 1.0), 0.5)
        self.assertEqual(rotation_fraction(9, 5, 9, 1.0), 1.0)

    def test_removes_linear_endpoint_rotation_without_moving_center(self):
        frames = {}
        for frame, yaw in ((0, 0.0), (1, 45.0), (2, 90.0)):
            frames[f"{frame:06d}"] = {
                "parts": {"pump": {"T_world_from_part": make_pose(yaw)}}
            }
        trajectory = {
            "parts": ["pump"],
            "reference_part": "pump",
            "frames": frames,
        }
        solver = {"parts": ["pump"], "frames": {
            "000000": {"parts": {"pump": {"T_world_from_part": make_pose()}}},
            "000001": {"parts": {"pump": {"T_world_from_part": make_pose(45.0)}}},
            "000002": {"parts": {"pump": {"T_world_from_part": make_pose(90.0)}}},
        }}
        identity = np.eye(4).tolist()
        registrations = {"pump": {
            "000001_to_000000": {"T_target_from_source": identity},
            "000002_to_000001": {"T_target_from_source": identity},
        }}

        result, report = retime_rotation(
            trajectory,
            solver,
            registrations,
            part="pump",
            window_start=0,
            window_end=2,
            schedule_start=0,
            schedule_end=2,
            terminal_fraction=0.0,
            follow_end=2,
        )

        for frame in range(3):
            pose = np.asarray(
                result["frames"][f"{frame:06d}"]["parts"]["pump"]["T_world_from_part"]
            )
            self.assertAlmostEqual(
                Rotation.from_matrix(pose[:3, :3]).magnitude(), 0.0, places=9
            )
            np.testing.assert_allclose(pose[:3, 3], [0.1, -0.2, 1.0])
        self.assertAlmostEqual(report["endpoint_rotation_deg"], 90.0)


if __name__ == "__main__":
    unittest.main()
