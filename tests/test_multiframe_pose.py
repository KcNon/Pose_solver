from __future__ import annotations

import unittest

import numpy as np

from common.multiframe_pose import (
    boundary_candidate_mask,
    internal_continuity_gate,
    multiframe_settings,
    solve_pose_candidate_path,
    static_candidate_scores,
    validate_multiframe_settings,
)
from tools.stages.pose.annotate_pose_observability import annotate_range


def translated(x: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(x)
    return pose


class MultiframePoseTest(unittest.TestCase):
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
