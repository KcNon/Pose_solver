import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from common.pose_refinement import (
    cap_pose_delta,
    interpolate_untrusted_pose_frames,
    limit_pose_velocity,
    smooth_pose_ranges,
)
from tools.stages.pose.stabilize_static_pose import interpolate_pose
from tools.stages.pose.refine_pose_render_loss import (
    exact_holdout_gate,
    other_part_touch_ratio,
    resolve_frame_view_split,
)
from tools.stages.pose.render_multiview_pose import resolved_render_settings


class PoseConstraintTests(unittest.TestCase):
    def test_other_part_touch_ratio_is_a_boundary_fraction(self):
        target = np.zeros((15, 15), dtype=bool)
        target[4:11, 4:11] = True
        other = np.zeros_like(target)
        other[3:12, 11:13] = True
        ratio = other_part_touch_ratio(
            target, other, dilation_pixels=1
        )
        self.assertGreater(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)

    def test_minimal_pose_config_has_render_defaults(self):
        settings = resolved_render_settings({
            "views": ["camera_0"],
            "frames": {"start": 1, "end": 2},
        })
        self.assertEqual(settings["primary_view"], "camera_0")
        self.assertEqual(settings["fps"], 5.0)
        self.assertGreater(settings["axis_length_m"], 0.0)

    def test_exact_triangle_stage_cannot_bypass_holdout_gate(self):
        accepted, report = exact_holdout_gate(
            {"loss": 0.30, "mean_iou": 0.70},
            {"loss": 0.34, "mean_iou": 0.65},
            {
                "maximum_holdout_degradation": 0.015,
                "minimum_holdout_iou": 0.20,
            },
        )
        self.assertFalse(accepted)
        self.assertIn("holdout_loss_degradation", report["failures"])

    def test_exact_triangle_holdout_gate_accepts_improvement(self):
        accepted, report = exact_holdout_gate(
            {"loss": 0.40, "mean_iou": 0.55},
            {"loss": 0.32, "mean_iou": 0.65},
            {
                "maximum_holdout_degradation": 0.015,
                "minimum_holdout_iou": 0.20,
            },
        )
        self.assertTrue(accepted)
        self.assertLess(report["loss_degradation"], 0.0)

    def test_exact_triangle_strict_gate_requires_holdout(self):
        accepted, report = exact_holdout_gate(
            None,
            None,
            {"require_independent_holdout": True},
        )
        self.assertFalse(accepted)
        self.assertIn("independent_holdout_missing", report["failures"])

    def test_rotating_holdout_is_removed_from_optimization(self):
        available = {"a", "b", "c", "d"}
        first_opt, first_holdout, first_report = resolve_frame_view_split(
            available,
            ["a", "b", "c", "d"],
            [],
            frame=0,
            minimum_optimize_views=3,
            minimum_holdout_views=1,
            require_independent_holdout=True,
            auto_holdout_policy="rotating",
        )
        next_opt, next_holdout, _ = resolve_frame_view_split(
            available,
            ["a", "b", "c", "d"],
            [],
            frame=1,
            minimum_optimize_views=3,
            minimum_holdout_views=1,
            require_independent_holdout=True,
            auto_holdout_policy="rotating",
        )
        self.assertEqual(first_holdout, ["a"])
        self.assertEqual(next_holdout, ["b"])
        self.assertFalse(set(first_opt).intersection(first_holdout))
        self.assertFalse(set(next_opt).intersection(next_holdout))
        self.assertTrue(first_report["independent_holdout_satisfied"])

    def test_strict_view_split_fails_closed_when_too_few_views(self):
        optimize, holdout, report = resolve_frame_view_split(
            {"a", "b", "c"},
            ["a", "b", "c"],
            [],
            frame=0,
            minimum_optimize_views=3,
            minimum_holdout_views=0,
            require_independent_holdout=True,
            auto_holdout_policy="rotating",
        )
        self.assertEqual(optimize, ["a", "b", "c"])
        self.assertEqual(holdout, [])
        self.assertFalse(report["independent_holdout_satisfied"])

    def test_pose_bridge_uses_geodesic_rotation(self):
        start = np.eye(4)
        end = np.eye(4)
        end[:3, :3] = Rotation.from_euler(
            "z", 90.0, degrees=True
        ).as_matrix()
        end[:3, 3] = [0.3, -0.15, 0.06]

        middle = interpolate_pose(start, end, 0.5)

        angle = np.degrees(
            Rotation.from_matrix(middle[:3, :3]).magnitude()
        )
        self.assertAlmostEqual(angle, 45.0, places=7)
        np.testing.assert_allclose(
            middle[:3, 3], [0.15, -0.075, 0.03], atol=1e-10
        )

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

    def test_range_smoothing_reduces_spike_and_keeps_boundaries(self):
        poses = {frame: np.eye(4) for frame in range(5)}
        poses[2][:3, 3] = [0.12, 0.0, 0.0]
        poses[2][:3, :3] = Rotation.from_euler(
            "x", 60, degrees=True
        ).as_matrix()
        smoothed = smooth_pose_ranges(poses, [(1, 3)], passes=2)
        np.testing.assert_allclose(smoothed[0], poses[0])
        np.testing.assert_allclose(smoothed[4], poses[4])
        self.assertLess(smoothed[2][0, 3], poses[2][0, 3])
        self.assertLess(
            np.degrees(Rotation.from_matrix(smoothed[2][:3, :3]).magnitude()),
            60.0,
        )

    def test_range_smoothing_preserves_rejected_frame_anchor(self):
        poses = {frame: np.eye(4) for frame in range(5)}
        poses[2][:3, 3] = [0.12, 0.0, 0.0]
        smoothed = smooth_pose_ranges(
            poses, [(1, 3)], passes=3, fixed_frames={2}
        )
        np.testing.assert_allclose(smoothed[2], poses[2])

    def test_untrusted_pose_is_interpolated_not_fixed(self):
        poses = {frame: np.eye(4) for frame in range(5)}
        poses[1][:3, 3] = [0.02, 0.0, 0.0]
        poses[2][:3, 3] = [0.30, 0.0, 0.0]
        poses[3][:3, 3] = [0.06, 0.0, 0.0]
        poses[1][:3, :3] = Rotation.from_euler(
            "z", 10.0, degrees=True
        ).as_matrix()
        poses[3][:3, :3] = Rotation.from_euler(
            "z", 30.0, degrees=True
        ).as_matrix()
        filled, replaced = interpolate_untrusted_pose_frames(
            poses, [(1, 3)], trusted_frames={1, 3}
        )
        self.assertEqual(replaced, [2])
        self.assertAlmostEqual(filled[2][0, 3], 0.04, places=8)
        self.assertAlmostEqual(
            np.degrees(Rotation.from_matrix(filled[2][:3, :3]).magnitude()),
            20.0,
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
