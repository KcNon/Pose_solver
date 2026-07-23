import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from common.pose_refinement import cap_pose_delta, limit_pose_velocity


class PoseConstraintTests(unittest.TestCase):
    def test_assembly_delta_is_bounded(self):
        delta = np.eye(4)
        delta[:3, 3] = [0.03, 0.0, 0.0]
        delta[:3, :3] = Rotation.from_euler("x", 30, degrees=True).as_matrix()
        bounded, _ = cap_pose_delta(delta, 0.012, 12.0)
        self.assertAlmostEqual(np.linalg.norm(bounded[:3, 3]), 0.012, places=8)
        self.assertAlmostEqual(np.degrees(Rotation.from_matrix(
            bounded[:3, :3]).magnitude()), 12.0, places=8)

    def test_lid_velocity_gate_keeps_tilt_but_limits_speed(self):
        poses = {0: np.eye(4), 1: np.eye(4)}
        poses[1][:3, 3] = [0.08, 0.0, 0.0]
        poses[1][:3, :3] = Rotation.from_euler("x", 10, degrees=True).as_matrix()
        limited, report = limit_pose_velocity(poses, 0.04, 3.0)
        self.assertAlmostEqual(np.linalg.norm(limited[1][:3, 3]), 0.04, places=8)
        self.assertAlmostEqual(np.degrees(Rotation.from_matrix(
            limited[1][:3, :3]).magnitude()), 3.0, places=8)
        self.assertEqual(report["limited_frames"], [1])


if __name__ == "__main__":
    unittest.main()
