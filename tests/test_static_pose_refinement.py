import unittest

import numpy as np

from common.static_pose_refinement import (
    align_pose_to_support_plane,
    containing_static_range,
    merge_static_pose_refinements,
)


def trajectory() -> dict:
    identity = np.eye(4).tolist()
    return {
        "parts": ["part"],
        "reference_part": "part",
        "scales": {"part": 1.0},
        "raw_mesh_origins": {"part": [0.0, 0.0, 0.0]},
        "frames": {
            f"{frame:06d}": {
                "parts": {
                    "part": {
                        "state": "static",
                        "source": "test",
                        "observing_views": 1,
                        "T_world_from_part": identity,
                    }
                }
            }
            for frame in range(10, 15)
        },
    }


class StaticPoseRefinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {"states": {"part": {"static_ranges": [[10, 13]]}}}

    def test_containing_static_range(self) -> None:
        self.assertEqual(containing_static_range(self.config, "part", 12), (10, 13))
        self.assertIsNone(containing_static_range(self.config, "part", 14))

    def test_align_pose_to_support_plane_changes_only_translation(self) -> None:
        pose = np.eye(4)
        pose[:3, 3] = [0.2, -0.1, 0.03]
        vertices = np.asarray(
            [[-0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [0.0, 0.0, 0.2]]
        )
        aligned, report = align_pose_to_support_plane(
            pose,
            vertices,
            {
                "normal_world": [0.0, 0.0, 1.0],
                "point_world": [0.0, 0.0, 0.0],
            },
            contact_quantile=0.0,
            maximum_shift_m=0.05,
        )
        self.assertTrue(report["accepted"])
        np.testing.assert_allclose(aligned[:3, :3], pose[:3, :3])
        np.testing.assert_allclose(aligned[:2, 3], pose[:2, 3])
        self.assertAlmostEqual(aligned[2, 3], 0.0)
        self.assertAlmostEqual(report["bottom_gap_before_m"], 0.03)

    def test_align_pose_to_support_plane_rejects_large_shift(self) -> None:
        pose = np.eye(4)
        pose[2, 3] = 0.10
        aligned, report = align_pose_to_support_plane(
            pose,
            np.asarray([[0.0, 0.0, 0.0]]),
            {
                "normal_world": [0.0, 0.0, 2.0],
                "point_world": [0.0, 0.0, 0.0],
            },
            contact_quantile=0.0,
            maximum_shift_m=0.02,
        )
        self.assertFalse(report["accepted"])
        np.testing.assert_allclose(aligned, pose)

    def test_merge_one_frame_without_propagation(self) -> None:
        baseline = trajectory()
        refined = trajectory()
        refined["frames"]["000012"]["parts"]["part"]["T_world_from_part"][0][3] = 0.02
        result, report = merge_static_pose_refinements(
            self.config, baseline, {"part": refined}, frame=12
        )
        self.assertAlmostEqual(
            result["frames"]["000012"]["parts"]["part"]["T_world_from_part"][0][3],
            0.02,
        )
        self.assertEqual(
            result["frames"]["000011"]["parts"]["part"]["T_world_from_part"][0][3],
            0.0,
        )
        self.assertEqual(report["parts"]["part"]["applied_frames"], 1)

    def test_propagates_only_containing_static_range(self) -> None:
        baseline = trajectory()
        refined = trajectory()
        refined["frames"]["000012"]["parts"]["part"]["T_world_from_part"][2][3] = 0.03
        result, report = merge_static_pose_refinements(
            self.config,
            baseline,
            {"part": refined},
            frame=12,
            propagate=True,
        )
        for frame in range(10, 14):
            self.assertAlmostEqual(
                result["frames"][f"{frame:06d}"]["parts"]["part"]["T_world_from_part"][2][3],
                0.03,
            )
        self.assertEqual(
            result["frames"]["000014"]["parts"]["part"]["T_world_from_part"][2][3],
            0.0,
        )
        self.assertEqual(report["parts"]["part"]["applied_range"], [10, 13])


if __name__ == "__main__":
    unittest.main()
