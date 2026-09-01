from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from common.multiframe_pose import (
    boundary_candidate_mask,
    internal_continuity_gate,
    multiframe_settings,
    parallel_transport_orientations,
    solve_pose_candidate_path,
    static_candidate_scores,
    validate_multiframe_settings,
)
from tools.stages.pose.annotate_pose_observability import annotate_range
from tools.stages.pose.optimize_multiframe_pose import (
    AggregateStaticObjective,
    _coaxial_snap_window,
    _orientation_transport_window,
)


def translated(x: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(x)
    return pose


class MultiframePoseTest(unittest.TestCase):
    def test_static_objective_uses_relative_pose_across_frame_groups(self) -> None:
        class Objective:
            def __init__(self, target_x: float) -> None:
                self.target_x = target_x

            def evaluate(self, pose: np.ndarray) -> dict:
                error = abs(float(pose[0, 3]) - self.target_x)
                return {
                    "loss": error,
                    "worst_view_loss": error,
                    "mean_iou": 1.0 - error,
                    "worst_view_iou": 1.0 - error,
                    "mean_contour_chamfer_px": error,
                    "mean_target_coverage": 1.0 - error,
                    "views": [{"view": "camera", "loss": error, "iou": 1.0 - error}],
                }

        trajectory = {
            "frames": {
                "000001": {"parts": {"body": {"T_world_from_part": translated(1.0).tolist()}}},
                "000002": {"parts": {"body": {"T_world_from_part": translated(2.0).tolist()}}},
            }
        }
        objective = AggregateStaticObjective(
            trajectory,
            part="piece",
            reference_part="body",
            objectives={1: Objective(1.25), 2: Objective(2.25)},
            optimize_frames=[1],
            holdout_frames=[2],
        )
        relative = translated(0.25)
        self.assertAlmostEqual(
            objective.evaluate(relative, ["optimize_frames"])["loss"], 0.0
        )
        self.assertAlmostEqual(
            objective.evaluate(relative, ["holdout_frames"])["loss"], 0.0
        )

    def test_internal_continuity_gate_rejects_rotation_regression(self) -> None:
        passed, report = internal_continuity_gate(
            [(0.010, 6.8), (0.008, 2.0)],
            [(0.006, 16.57), (0.005, 1.0)],
        )
        self.assertFalse(passed)
        self.assertAlmostEqual(report["maximum_rotation_step_deg"], 6.8)
        self.assertAlmostEqual(
            report["selected_max_rotation_step_deg"], 16.57
        )

    def test_internal_continuity_gate_allows_explicit_tolerance(self) -> None:
        passed, _ = internal_continuity_gate(
            [(0.010, 6.8)],
            [(0.011, 7.2)],
            maximum_translation_degradation_m=0.001,
            maximum_rotation_degradation_deg=0.4,
        )
        self.assertTrue(passed)

    def test_observability_annotation_does_not_change_pose(self) -> None:
        original = {
            "frames": {
                f"{frame:06d}": {
                    "parts": {
                        "piece": {
                            "T_world_from_part": translated(frame).tolist(),
                            "source": "visual",
                        }
                    }
                }
                for frame in range(2, 5)
            }
        }
        result, report = annotate_range(
            original, part="piece", start=2, end=4
        )
        self.assertEqual(report["maximum_transform_change"], 0.0)
        self.assertEqual(
            result["frames"]["000003"]["parts"]["piece"]["observability"],
            "occluded_unverified",
        )
        self.assertEqual(
            original["frames"]["000003"]["parts"]["piece"]["source"],
            "visual",
        )

    def test_boundary_gate_rejects_endpoint_jump_that_baseline_does_not_have(
        self,
    ) -> None:
        mask, report = boundary_candidate_mask(
            [translated(0.0), translated(0.08)],
            boundary_pose=translated(0.0),
            baseline_pose=translated(0.0),
            maximum_translation_degradation_m=0.002,
        )
        self.assertEqual(mask.tolist(), [True, False])
        self.assertAlmostEqual(report["baseline_translation_step_m"], 0.0)

    def test_boundary_gate_allows_candidate_that_improves_existing_jump(
        self,
    ) -> None:
        mask, _ = boundary_candidate_mask(
            [translated(0.08), translated(0.02)],
            boundary_pose=translated(0.0),
            baseline_pose=translated(0.08),
            maximum_translation_degradation_m=0.0,
        )
        self.assertEqual(mask.tolist(), [True, True])

    def test_legacy_observed_assembly_maps_to_relation_window(self) -> None:
        settings = multiframe_settings({
            "observed_assembly_regularization": {
                "enabled": True,
                "relations": [{
                    "name": "insert",
                    "type": "coaxial_insert",
                    "reference_part": "body",
                    "moving_part": "tip",
                }],
            }
        })
        self.assertEqual(settings["legacy_source"], "observed_assembly_regularization")
        self.assertEqual(settings["windows"][0]["mode"], "relation_window")
        self.assertEqual(
            settings["windows"][0]["relation_type"], "coaxial_insert"
        )

    def test_window_contract_rejects_unknown_part(self) -> None:
        config = {
            "multiframe_optimization": {
                "enabled": True,
                "windows": [{
                    "name": "bad",
                    "mode": "static_window",
                    "part": "missing",
                    "frame_range": [2, 4],
                }],
            }
        }
        with self.assertRaisesRegex(ValueError, "unknown part"):
            validate_multiframe_settings(
                config, parts=["piece"], frame_start=2, frame_end=4
            )

    def test_bridge_window_is_a_supported_generic_mode(self) -> None:
        settings = validate_multiframe_settings(
            {
                "multiframe_optimization": {
                    "enabled": True,
                    "windows": [{
                        "name": "hidden_interval",
                        "mode": "bridge_window",
                        "part": "piece",
                        "reference_part": "base",
                        "frame_range": [2, 4],
                    }],
                }
            },
            parts=["piece", "base"],
            frame_start=1,
            frame_end=5,
        )
        self.assertEqual(settings["windows"][0]["mode"], "bridge_window")

    def test_parallel_transport_preserves_tilt_and_removes_axial_spin(self) -> None:
        z = np.asarray([0.0, 0.0, 1.0])
        rotations = {
            0: Rotation.from_euler("z", 30.0, degrees=True).as_matrix(),
            1: (
                Rotation.from_euler("y", 30.0, degrees=True)
                * Rotation.from_euler("z", 80.0, degrees=True)
            ).as_matrix(),
            2: (
                Rotation.from_euler("y", 60.0, degrees=True)
                * Rotation.from_euler("z", 120.0, degrees=True)
            ).as_matrix(),
        }
        result = parallel_transport_orientations(
            rotations, z, seed_frame=0
        )
        for frame in rotations:
            np.testing.assert_allclose(
                result[frame] @ z, rotations[frame] @ z, atol=1e-10
            )
        self.assertLess(
            np.degrees(
                Rotation.from_matrix(result[0].T @ result[1]).magnitude()
            ),
            31.0,
        )
        self.assertGreater(
            np.degrees(
                Rotation.from_matrix(rotations[0].T @ rotations[1]).magnitude()
            ),
            50.0,
        )

    def test_mechanism_windows_are_supported(self) -> None:
        settings = validate_multiframe_settings(
            {
                "multiframe_optimization": {
                    "enabled": True,
                    "windows": [
                        {
                            "name": "seat",
                            "mode": "coaxial_snap_window",
                            "reference_part": "body",
                            "moving_part": "piece",
                            "frame_range": [3, 4],
                            "terminal_anchor_frame": 5,
                            "static_follow_range": [5, 6],
                            "reference_axis_part": [0, 1, 0],
                            "moving_axis_part": [0, 0, 1],
                        },
                        {
                            "name": "no_spin",
                            "mode": "orientation_transport_window",
                            "part": "piece",
                            "reference_part": "body",
                            "frame_range": [2, 5],
                            "seed_frame": 5,
                            "moving_axis_part": [0, 0, 1],
                        },
                    ],
                }
            },
            parts=["piece", "body"],
            frame_start=1,
            frame_end=6,
        )
        self.assertEqual(
            [window["mode"] for window in settings["windows"]],
            ["coaxial_snap_window", "orientation_transport_window"],
        )

    def test_coaxial_snap_can_set_absolute_contact_offset(self) -> None:
        body = np.eye(4, dtype=np.float64)
        moving = np.eye(4, dtype=np.float64)
        moving[:3, 3] = [0.1, 0.2, 0.3]
        baseline = {
            "parts": ["body", "piece"],
            "scales": {"body": 1.0, "piece": 1.0},
            "raw_mesh_origins": {"body": [0, 0, 0], "piece": [0, 0, 0]},
            "frames": {
                f"{frame:06d}": {
                    "parts": {
                        "body": {"T_world_from_part": body.tolist()},
                        "piece": {"T_world_from_part": moving.tolist()},
                    }
                }
                for frame in range(1, 5)
            },
        }
        trajectory = {
            **baseline,
            "frames": {
                key: {"parts": {name: dict(value) for name, value in row["parts"].items()}}
                for key, row in baseline["frames"].items()
            },
        }
        report = _coaxial_snap_window(
            {},
            baseline,
            trajectory,
            {
                "reference_part": "body",
                "moving_part": "piece",
                "frame_range": [2, 3],
                "terminal_anchor_frame": 4,
                "static_follow_range": [4, 4],
                "reference_axis_part": [0, 1, 0],
                "moving_axis_part": [0, 0, 1],
                "target_axis_offset_m": 0.0,
                "target_axial_offset_m": 0.0,
            },
        )
        self.assertAlmostEqual(report["terminal_after"]["axis_angle_deg"], 0.0)
        self.assertAlmostEqual(report["terminal_after"]["axis_offset_m"], 0.0)
        self.assertAlmostEqual(report["terminal_after"]["axial_offset_m"], 0.0)

    def test_full_orientation_lock_preserves_centres_after_pickup(self) -> None:
        frames = {}
        for frame, angle in enumerate((0.0, 20.0, 50.0, 90.0), start=1):
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = Rotation.from_euler(
                "z", angle, degrees=True
            ).as_matrix()
            pose[0, 3] = float(frame)
            frames[f"{frame:06d}"] = {
                "parts": {"piece": {"T_world_from_part": pose.tolist()}}
            }
        baseline = {"frames": frames}
        trajectory = {
            "frames": {
                key: {"parts": {"piece": dict(row["parts"]["piece"])}}
                for key, row in frames.items()
            }
        }
        report = _orientation_transport_window(
            baseline,
            trajectory,
            {
                "part": "piece",
                "frame_range": [1, 4],
                "seed_frame": 4,
                "lock_full_orientation": True,
                "lock_start_frame": 2,
                "moving_axis_part": [0, 0, 1],
            },
        )
        target = np.asarray(
            trajectory["frames"]["000004"]["parts"]["piece"]["T_world_from_part"]
        )[:3, :3]
        for frame in (2, 3, 4):
            pose = np.asarray(
                trajectory["frames"][f"{frame:06d}"]["parts"]["piece"]["T_world_from_part"]
            )
            np.testing.assert_allclose(pose[:3, :3], target, atol=1e-10)
            self.assertEqual(float(pose[0, 3]), float(frame))
        self.assertTrue(report["translations_changed"] is False)

    def test_full_orientation_lock_ignores_noisy_internal_rotations(self) -> None:
        frames = {}
        for frame, angle in enumerate((0.0, 140.0, -80.0, 90.0), start=1):
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = Rotation.from_euler(
                "z", angle, degrees=True
            ).as_matrix()
            frames[f"{frame:06d}"] = {
                "parts": {"piece": {"T_world_from_part": pose.tolist()}}
            }
        baseline = {"frames": frames}
        trajectory = {"frames": deepcopy(frames)}
        _orientation_transport_window(
            baseline,
            trajectory,
            {
                "part": "piece",
                "frame_range": [1, 4],
                "seed_frame": 4,
                "lock_full_orientation": True,
                "lock_start_frame": 4,
                "moving_axis_part": [0, 0, 1],
            },
        )
        output = [
            Rotation.from_matrix(np.asarray(
                trajectory["frames"][f"{frame:06d}"]["parts"]["piece"][
                    "T_world_from_part"
                ]
            )[:3, :3]).as_euler("zyx", degrees=True)[0]
            for frame in range(1, 5)
        ]
        np.testing.assert_allclose(
            output, [0.0, 70.0 / 3.0, 200.0 / 3.0, 90.0], atol=1e-8
        )

    def test_static_score_uses_all_evidence_frames(self) -> None:
        candidates = [translated(0.0), translated(1.0)]

        def score(frame: int, pose: np.ndarray) -> float:
            targets = {1: 0.9, 2: 1.0, 3: 1.1}
            return abs(float(pose[0, 3]) - targets[frame])

        scores = static_candidate_scores(candidates, [1, 2, 3], score)
        self.assertGreater(scores[0], scores[1])

    def test_dynamic_path_rejects_single_frame_visual_outlier(self) -> None:
        candidates = [
            [translated(0.00), translated(0.08)],
            [translated(0.01), translated(0.09)],
            [translated(0.02), translated(0.10)],
        ]
        # The isolated low unary at x=0.09 cannot be reached with the physical
        # 3 cm step limit, so the joint path stays on the smooth branch.
        unary = [
            np.asarray([0.0, 1.0]),
            np.asarray([0.4, 0.0]),
            np.asarray([0.0, 1.0]),
        ]
        selected, _ = solve_pose_candidate_path(
            unary,
            candidates,
            maximum_translation_step_m=0.03,
            maximum_rotation_step_deg=10.0,
            temporal_weight=0.2,
        )
        self.assertEqual(selected, [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
