from __future__ import annotations

import unittest

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from common.occlusion_masks import known_occluder_mask
from common.render_loss_refinement import (
    MultiViewRenderObjective,
    RenderObservation,
    apply_world_pose_delta,
    clamp_pose_step,
    coarse_reacquire_pose,
    foreground_occlusion_mask,
    rasterize_surface_points,
    refine_pose_coordinate_search,
    symmetry_aware_rotation_directions,
)


class RenderLossRefinementTests(unittest.TestCase):
    def test_known_occluder_labels_can_isolate_hand_ablation(self) -> None:
        labels = np.asarray([[0, 1, 2, 3]], dtype=np.uint8)
        rigid_only = known_occluder_mask(labels, 1, [1, 2])
        with_hand = known_occluder_mask(labels, 1, [1, 2, 3])
        legacy = known_occluder_mask(labels, 1)

        np.testing.assert_array_equal(
            rigid_only, np.asarray([[False, False, True, False]])
        )
        np.testing.assert_array_equal(
            with_hand, np.asarray([[False, False, True, True]])
        )
        np.testing.assert_array_equal(with_hand, legacy)

    def setUp(self) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.3)
        self.points = np.asarray(mesh.vertices, dtype=np.float64)
        self.K = np.array(
            [[110.0, 0.0, 80.0], [0.0, 110.0, 45.0], [0.0, 0.0, 1.0]]
        )
        self.E = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
        self.target_pose = np.eye(4)
        self.target_pose[:3, 3] = [0.0, 0.0, 2.0]
        world = (
            self.points @ self.target_pose[:3, :3].T
            + self.target_pose[:3, 3]
        )
        target, _ = rasterize_surface_points(
            world, self.K, self.E, (90, 160), dilation_pixels=2
        )
        self.observations = [
            RenderObservation(
                view="opt",
                intrinsics=self.K,
                extrinsics=self.E,
                target_mask=target,
            ),
            RenderObservation(
                view="holdout",
                intrinsics=self.K,
                extrinsics=self.E,
                target_mask=target,
            ),
        ]

    def test_coordinate_search_improves_shifted_silhouette(self) -> None:
        objective = MultiViewRenderObjective(
            self.points,
            self.observations,
            {
                "dilation_pixels": 2,
                "weights": {
                    "iou": 1.0,
                    "contour": 0.2,
                    "target_coverage": 0.1,
                    "depth": 0.0,
                },
            },
        )
        initial = self.target_pose.copy()
        initial[0, 3] += 0.06
        refined, report = refine_pose_coordinate_search(
            objective,
            initial,
            optimize_views=["opt"],
            holdout_views=["holdout"],
            translation_steps_m=[0.04, 0.02, 0.01],
            rotation_steps_deg=[1.0],
            symmetry_axis_part=None,
            optimize_rotation=False,
            maximum_translation_delta_m=0.10,
            maximum_rotation_delta_deg=5.0,
            minimum_improvement=0.001,
            maximum_holdout_degradation=0.001,
            prior_weight=0.0,
            temporal_weight=0.0,
        )
        self.assertTrue(report["accepted"])
        self.assertLess(abs(refined[0, 3]), abs(initial[0, 3]))
        self.assertGreater(
            report["refined_optimize"]["mean_iou"],
            report["baseline_optimize"]["mean_iou"],
        )

    def test_foreground_depth_marks_amodal_render_as_unknown(self) -> None:
        predicted = np.full((5, 5), np.inf, np.float32)
        predicted[2, 2] = 2.0
        observed = np.full((5, 5), np.inf, np.float32)
        observed[2, 2] = 1.0
        occluded = foreground_occlusion_mask(
            predicted, observed, margin_m=0.02, dilation_pixels=0
        )
        self.assertTrue(occluded[2, 2])
        observed[2, 2] = 1.99
        occluded = foreground_occlusion_mask(
            predicted, observed, margin_m=0.02, dilation_pixels=0
        )
        self.assertFalse(occluded[2, 2])

    def test_known_part_occluder_is_not_silhouette_background(self) -> None:
        target = self.observations[0].target_mask.copy()
        known = np.zeros_like(target)
        known[:, :80] = True
        objective = MultiViewRenderObjective(
            self.points,
            [RenderObservation(
                view="opt",
                intrinsics=self.K,
                extrinsics=self.E,
                target_mask=target & ~known,
                known_occluder_mask=known,
            )],
            {"dilation_pixels": 2},
        )
        result = objective.evaluate(self.target_pose, ["opt"])
        self.assertGreater(result["mean_iou"], 0.9)
        self.assertGreater(
            result["views"][0]["ignored_occluded_pixels"], 0
        )

    def test_coarse_reacquire_escapes_local_translation_basin(self) -> None:
        objective = MultiViewRenderObjective(
            self.points,
            self.observations,
            {"dilation_pixels": 2, "weights": {"depth": 0.0}},
        )
        initial = self.target_pose.copy()
        initial[0, 3] += 0.16
        selected, report = coarse_reacquire_pose(
            objective,
            initial,
            views=["opt", "holdout"],
            translation_radii_m=[0.08],
            rotation_angles_deg=[],
            symmetry_axis_part=None,
            optimize_rotation=False,
            alternating_passes=2,
        )
        self.assertGreater(report["loss_improvement"], 0.0)
        self.assertLess(abs(selected[0, 3]), abs(initial[0, 3]))

    def test_holdout_gate_rejects_view_specific_improvement(self) -> None:
        initial = self.target_pose.copy()
        initial[0, 3] += 0.06
        initial_world = (
            self.points @ initial[:3, :3].T + initial[:3, 3]
        )
        holdout_target, _ = rasterize_surface_points(
            initial_world, self.K, self.E, (90, 160), dilation_pixels=2
        )
        objective = MultiViewRenderObjective(
            self.points,
            [
                self.observations[0],
                RenderObservation(
                    view="holdout",
                    intrinsics=self.K,
                    extrinsics=self.E,
                    target_mask=holdout_target,
                ),
            ],
            {
                "dilation_pixels": 2,
                "weights": {
                    "iou": 1.0,
                    "contour": 0.2,
                    "target_coverage": 0.1,
                    "depth": 0.0,
                },
            },
        )
        selected, report = refine_pose_coordinate_search(
            objective,
            initial,
            optimize_views=["opt"],
            holdout_views=["holdout"],
            translation_steps_m=[0.04, 0.02, 0.01],
            rotation_steps_deg=[1.0],
            symmetry_axis_part=None,
            optimize_rotation=False,
            maximum_translation_delta_m=0.10,
            maximum_rotation_delta_deg=5.0,
            minimum_improvement=0.001,
            maximum_holdout_degradation=0.0,
            prior_weight=0.0,
            temporal_weight=0.0,
        )
        self.assertFalse(report["accepted"])
        self.assertGreater(report["optimize_loss_improvement"], 0.0)
        self.assertGreater(report["holdout_loss_degradation"], 0.0)
        np.testing.assert_allclose(selected, initial)

    def test_absolute_iou_gate_rejects_relative_only_improvement(self) -> None:
        objective = MultiViewRenderObjective(
            self.points,
            self.observations,
            {
                "dilation_pixels": 2,
                "weights": {"iou": 1.0, "contour": 0.2},
            },
        )
        initial = self.target_pose.copy()
        initial[0, 3] += 0.06
        selected, report = refine_pose_coordinate_search(
            objective,
            initial,
            optimize_views=["opt"],
            holdout_views=[],
            translation_steps_m=[0.04, 0.02],
            rotation_steps_deg=[1.0],
            symmetry_axis_part=None,
            optimize_rotation=False,
            maximum_translation_delta_m=0.10,
            maximum_rotation_delta_deg=5.0,
            minimum_improvement=0.001,
            maximum_holdout_degradation=0.0,
            minimum_refined_iou=1.01,
            prior_weight=0.0,
            temporal_weight=0.0,
        )
        self.assertFalse(report["accepted"])
        self.assertIn(
            "optimize_iou_below_minimum",
            report["absolute_gate_failures"],
        )
        np.testing.assert_allclose(selected, initial)

    def test_continuous_symmetry_excludes_axial_rotation(self) -> None:
        pose = np.eye(4)
        directions = symmetry_aware_rotation_directions(
            pose, np.array([0.0, 1.0, 0.0])
        )
        self.assertEqual(len(directions), 2)
        for direction in directions:
            self.assertAlmostEqual(
                float(np.dot(direction, [0.0, 1.0, 0.0])), 0.0
            )
            self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0)

    def test_pose_delta_keeps_rigid_scale(self) -> None:
        pose = apply_world_pose_delta(
            np.eye(4),
            [0.01, -0.02, 0.03],
            [0.1, -0.05, 0.02],
        )
        np.testing.assert_allclose(
            pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12
        )
        self.assertAlmostEqual(float(np.linalg.det(pose[:3, :3])), 1.0)

    def test_clamp_pose_step_respects_translation_and_rotation_limits(self):
        target = apply_world_pose_delta(
            np.eye(4), [0.3, 0.0, 0.0], [0.0, 0.0, np.deg2rad(60.0)]
        )
        clamped = clamp_pose_step(
            np.eye(4),
            target,
            maximum_translation_m=0.1,
            maximum_rotation_deg=15.0,
        )
        self.assertAlmostEqual(float(np.linalg.norm(clamped[:3, 3])), 0.1)
        angle = Rotation.from_matrix(clamped[:3, :3]).magnitude()
        self.assertLessEqual(float(np.degrees(angle)), 15.0)
        self.assertAlmostEqual(float(np.degrees(angle)), 15.0, places=6)


if __name__ == "__main__":
    unittest.main()
