import unittest

import numpy as np
import trimesh

from common.trajectory_constraints import (
    CylindricalContainer,
    SampledSurface,
    evaluate_insertion_trajectory,
    evaluate_screw_trajectory,
    evaluate_surface_contact_trajectory,
    insertion_pose_metrics,
    pairwise_alignment_metrics,
    project_coaxial_pose,
    project_pairwise_alignment,
    refine_insert_trajectory,
    refine_screw_trajectory,
    solve_monotonic_axial_path,
    surface_pair_pose_metrics,
)


def container() -> CylindricalContainer:
    return CylindricalContainer.from_spec(
        {
            "type": "cylindrical_container",
            "axis": "y",
            "inner_radius_m": 0.10,
            "outer_radius_m": 0.13,
            "floor_top_m": 0.0,
            "rim_top_m": 0.12,
            "floor_thickness_m": 0.01,
        }
    )


def pose(x: float, y: float, z: float = 0.0) -> np.ndarray:
    value = np.eye(4)
    value[:3, 3] = [x, y, z]
    return value


class InsertionMetricTests(unittest.TestCase):
    def test_centered_insert_inside_cavity_is_allowed(self):
        points = trimesh.creation.cylinder(
            radius=0.08, height=0.08, sections=32
        ).vertices
        # trimesh cylinders use Z; rotate the sample proxy onto container Y.
        points = points[:, [0, 2, 1]]
        result = insertion_pose_metrics(
            pose(0.0, 0.04),
            points,
            container(),
            containment_active=True,
            contact_tolerance_m=1e-4,
        )
        self.assertEqual(result["max_penetration_m"], 0.0)

    def test_side_wall_and_floor_penetration_are_detected(self):
        side_points = np.asarray([[0.115, 0.05, 0.0]])
        side = insertion_pose_metrics(
            np.eye(4),
            side_points,
            container(),
            containment_active=False,
        )
        self.assertGreater(side["max_penetration_m"], 0.0)
        floor_points = np.asarray([[0.0, -0.005, 0.0]])
        floor = insertion_pose_metrics(
            np.eye(4),
            floor_points,
            container(),
            containment_active=True,
        )
        self.assertGreater(floor["max_penetration_m"], 0.0)

    def test_continuous_samples_detect_tunnelling(self):
        points = np.asarray([[0.0, 0.05, 0.0]])
        poses = {0: pose(0.15, 0.0), 1: pose(0.0, 0.0)}
        result = evaluate_insertion_trajectory(
            poses,
            points,
            container(),
            substeps=12,
            entry_center_radius_m=0.09,
            contact_tolerance_m=0.0,
            entry_frame=1,
        )
        self.assertGreater(result["violating_samples"], 0)
        self.assertGreater(result["max_penetration_m"], 0.0)


class AxialPathTests(unittest.TestCase):
    def test_dynamic_program_rejects_visual_backtracking(self):
        # Independent visual minima would select 0, 2, 1, 3.  The physical
        # insertion path must remain directed and finish at the stable anchor.
        costs = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0],
                [2.0, 1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0, 2.0],
                [3.0, 2.0, 1.0, 0.0],
            ]
        )
        grid = np.asarray([0.0, 0.01, 0.02, 0.03])
        path, total = solve_monotonic_axial_path(
            costs,
            grid,
            direction=1.0,
            maximum_step_m=0.02,
            maximum_backtrack_m=0.0,
            temporal_weight=0.01,
            terminal_index=3,
        )
        self.assertTrue(np.all(np.diff(grid[path]) >= 0.0))
        self.assertEqual(int(path[-1]), 3)
        self.assertTrue(np.isfinite(total))

    def test_descending_path_uses_negative_direction(self):
        costs = np.zeros((3, 3), dtype=np.float64)
        grid = np.asarray([0.0, 0.01, 0.02])
        path, _ = solve_monotonic_axial_path(
            costs,
            grid,
            direction=-1.0,
            maximum_step_m=0.02,
            terminal_index=0,
        )
        self.assertTrue(np.all(np.diff(grid[path]) <= 0.0))
        self.assertEqual(int(path[-1]), 0)


class InsertionOptimizationTests(unittest.TestCase):
    def test_bounded_search_removes_floor_penetration(self):
        moving = trimesh.creation.cylinder(
            radius=0.08, height=0.08, sections=32
        )
        points = np.asarray(moving.vertices)[:, [0, 2, 1]]
        poses = {
            0: pose(0.0, 0.02),
            1: pose(0.0, 0.02),
        }
        config = {
            "entry_center_radius_m": 0.09,
            "entry_frame": 0,
            "contact_tolerance_m": 0.0,
            "continuous_substeps": 4,
            "optimize_frame_range": [0, 1],
            "penetration_scale_m": 0.005,
            "maximum_translation_delta_m": 0.04,
            "maximum_rotation_delta_deg": 5.0,
            "translation_steps_m": [0.02, 0.01, 0.005],
            "rotation_steps_deg": [2.0, 1.0, 0.5],
            "prior_weight": 0.0,
            "smoothness_weight": 0.0,
        }
        refined, report = refine_insert_trajectory(
            poses, points, container(), config
        )
        self.assertGreater(report["before"]["max_penetration_m"], 0.0)
        self.assertEqual(report["proposed_after"]["max_penetration_m"], 0.0)
        self.assertGreater(refined[0][1, 3], poses[0][1, 3])


class ScrewInsertionTests(unittest.TestCase):
    def test_screw_projection_aligns_axes_and_couples_pitch(self):
        values = {frame: np.eye(4) for frame in range(7)}
        for frame, value in values.items():
            value[:3, 3] = [0.02 * (1.0 - frame / 6.0), 0.0, 0.0]
        config = {
            "insertion_start_frame": 0,
            "contact_frame": 2,
            "seat_frame": 6,
            "terminal_anchor_frame": 6,
            "reference_axis_part": "z",
            "moving_axis_part": "z",
            "reference_axis_origin_part_m": [0.0, 0.0, 0.0],
            "moving_axis_origin_part_m": [0.0, 0.0, 0.0],
            "target_axis_offset_m": 0.0,
            "pitch_m_per_turn": 0.01,
            "turns": 1.0,
            "handedness": 1.0,
            "phase_observation_weight": 0.0,
        }
        refined, optimization = refine_screw_trajectory(values, config)
        report = evaluate_screw_trajectory(refined, config)
        self.assertAlmostEqual(report["max_axis_angle_deg"], 0.0, places=6)
        self.assertAlmostEqual(report["max_axis_offset_m"], 0.0, places=6)
        self.assertAlmostEqual(report["max_helix_residual_m"], 0.0, places=6)
        self.assertEqual(report["monotonic_axial_violations"], 0)
        self.assertAlmostEqual(
            refined[6][2, 3] - refined[2][2, 3], 0.01, places=6
        )
        self.assertAlmostEqual(optimization["effective_turns"], 1.0, places=6)

    def test_screw_evaluation_requires_and_reports_real_collision_geometry(self):
        fixed = sampled_box([1.0, 1.0, 1.0], 31)
        moving = sampled_box([1.0, 1.0, 1.0], 32)
        values = {0: pose(0.75, 0.0), 1: pose(0.75, 0.0)}
        report = evaluate_screw_trajectory(
            values,
            {
                "contact_frame": 0,
                "reference_axis_part": "z",
                "moving_axis_part": "z",
                "pitch_m_per_turn": 0.0,
                "collision_continuous_substeps": 1,
                "contact_tolerance_m": 0.0,
                "near_field_m": 0.3,
            },
            moving_surface=moving,
            reference_surface=fixed,
        )
        self.assertTrue(report["collision_evaluated"])
        self.assertGreater(report["max_penetration_m"], 0.0)
        self.assertGreater(report["violating_samples"], 0)


def sampled_box(extents: list[float], seed: int) -> SampledSurface:
    mesh = trimesh.creation.box(extents=extents)
    points, faces = trimesh.sample.sample_surface(mesh, 6000, seed=seed)
    return SampledSurface(points, mesh.face_normals[faces])


class GenericPairwiseContactTests(unittest.TestCase):
    def test_arbitrary_surface_pair_detects_overlap_and_separation(self):
        fixed = sampled_box([1.0, 1.0, 1.0], 11)
        moving = sampled_box([1.0, 1.0, 1.0], 12)
        overlap = pose(0.75, 0.0)
        colliding = surface_pair_pose_metrics(
            overlap,
            moving,
            fixed,
            contact_tolerance_m=0.0,
            near_field_m=0.3,
        )
        separated = surface_pair_pose_metrics(
            pose(1.2, 0.0),
            moving,
            fixed,
            contact_tolerance_m=0.0,
            near_field_m=0.3,
        )
        self.assertGreater(colliding["max_penetration_m"], 0.1)
        self.assertEqual(separated["max_penetration_m"], 0.0)

    def test_pairwise_alignment_is_category_agnostic(self):
        value = np.eye(4)
        angle = np.deg2rad(10.0)
        value[:3, :3] = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        value[:3, 3] = [0.03, 0.2, 0.04]
        projected = project_pairwise_alignment(
            value,
            reference_axis=[0.0, 1.0, 0.0],
            moving_axis=[0.0, 1.0, 0.0],
            allow_axis_flip=False,
            target_axis_offset_m=0.01,
        )
        metrics = pairwise_alignment_metrics(
            projected,
            reference_axis=[0.0, 1.0, 0.0],
            moving_axis=[0.0, 1.0, 0.0],
        )
        self.assertAlmostEqual(metrics["axis_angle_deg"], 0.0, places=6)
        self.assertAlmostEqual(metrics["axis_offset_m"], 0.01, places=6)

    def test_coaxial_projection_uses_axis_origins_not_part_origins(self):
        value = np.eye(4)
        value[:3, :3] = trimesh.transformations.rotation_matrix(
            np.deg2rad(35.0), [1.0, 0.0, 0.0]
        )[:3, :3]
        value[:3, 3] = [0.3, 0.2, -0.1]
        fixed_origin = np.asarray([0.2, 0.0, -0.1])
        moving_origin = np.asarray([0.1, 0.0, 0.0])
        projected = project_coaxial_pose(
            value,
            reference_axis=[0.0, 1.0, 0.0],
            moving_axis=[0.0, 0.0, 1.0],
            reference_axis_origin_m=fixed_origin,
            moving_axis_origin_m=moving_origin,
            target_axis_offset_m=0.0,
        )
        transformed_origin = (
            projected[:3, :3] @ moving_origin + projected[:3, 3]
        )
        offset = transformed_origin - fixed_origin
        radial = offset - np.dot(offset, [0.0, 1.0, 0.0]) * np.asarray(
            [0.0, 1.0, 0.0]
        )
        np.testing.assert_allclose(radial, np.zeros(3), atol=1e-9)
        np.testing.assert_allclose(
            projected[:3, :3] @ np.asarray([0.0, 0.0, 1.0]),
            [0.0, 1.0, 0.0],
            atol=1e-9,
        )

    def test_continuous_pairwise_evaluation_catches_tunnelling(self):
        fixed = sampled_box([1.0, 1.0, 1.0], 21)
        moving = sampled_box([0.2, 0.2, 0.2], 22)
        values = {
            0: pose(-0.8, 0.0),
            1: pose(0.8, 0.0),
        }
        report = evaluate_surface_contact_trajectory(
            values,
            moving,
            fixed,
            {
                "continuous_substeps": 16,
                "contact_start_frame": 2,
                "contact_tolerance_m": 0.0,
                "near_field_m": 0.2,
            },
        )
        self.assertGreater(report["max_penetration_m"], 0.0)
        self.assertGreater(report["violating_samples"], 0)


if __name__ == "__main__":
    unittest.main()
