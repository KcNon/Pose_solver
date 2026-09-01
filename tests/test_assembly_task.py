from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common.assembly_task import (
    evaluate_assembly_task,
    validate_assembly_task_config,
)
from common.io_utils import write_json


def _trajectory() -> dict:
    frames = {}
    for frame, offset in enumerate((0.2, 0.1, 0.02, 0.0, 0.0)):
        moving = np.eye(4)
        moving[2, 3] = offset
        frames[f"{frame:06d}"] = {
            "parts": {
                "body": {"T_world_from_part": np.eye(4).tolist()},
                "nozzle": {"T_world_from_part": moving.tolist()},
            }
        }
    return {"parts": ["body", "nozzle"], "frames": frames}


def _task(report: Path) -> dict:
    return {
        "reference_part": "body",
        "moving_part": "nozzle",
        "reference_axis_part": [0.0, 0.0, 1.0],
        "phases": {
            "pregrasp": {
                "frame_range": [0, 0],
                "physics_semantic": "inactive",
            },
            "transport": {
                "frame_range": [1, 1],
                "physics_semantic": "external_kinematic_constraint",
            },
            "approach": {
                "frame_range": [2, 2],
                "physics_semantic": "connector_constraint",
            },
            "assembled": {
                "frame_range": [3, 4],
                "physics_semantic": "terminal_hold",
            },
        },
        "terminal_frame_range": [3, 4],
        "terminal_target_frame": 4,
        "terminal_visual_evidence": {
            "report": str(report),
            "accepted_diagnoses": ["raw_supported"],
            "maximum_abs_best_offset_m": 0.005,
        },
        "external_constraint_model_available": False,
        "omitted_geometry": ["tube"],
    }


class AssemblyTaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.trajectory_path = self.root / "trajectory.json"
        write_json(self.trajectory_path, _trajectory())
        digest = hashlib.sha256(self.trajectory_path.read_bytes()).hexdigest()
        self.visual_report = self.root / "visual.json"
        write_json(self.visual_report, {
            "inputs": {"trajectory_sha256": digest},
            "summary": {
                "diagnosis": "raw_supported",
                "best": {"offset_m": 0.004},
            },
        })

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_pose_and_physics_readiness_are_separate(self) -> None:
        result = evaluate_assembly_task(
            _task(self.visual_report),
            _trajectory(),
            trajectory_path=self.trajectory_path,
            connector_report={"simulation_ready": False},
        )
        self.assertTrue(result["pose_product_ready"])
        self.assertFalse(result["physics_replay_ready"])
        self.assertEqual(
            result["terminal_target"]["maximum_translation_residual_m"],
            0.0,
        )
        self.assertIn(
            "external_hand_or_gripper_model_missing",
            result["physics_blockers"],
        )
        self.assertTrue(
            result["phases"]["transport"]["requires_external_constraint"]
        )

    def test_visual_evidence_must_match_exact_trajectory(self) -> None:
        self.trajectory_path.write_text("{}", encoding="utf-8")
        result = evaluate_assembly_task(
            _task(self.visual_report),
            _trajectory(),
            trajectory_path=self.trajectory_path,
            connector_report={"simulation_ready": True},
        )
        self.assertFalse(result["pose_product_ready"])
        self.assertIn(
            "terminal_visual_evidence_trajectory_mismatch",
            result["pose_failures"],
        )

    def test_phase_ranges_must_cover_trajectory_without_gaps(self) -> None:
        task = _task(self.visual_report)
        task["phases"]["approach"]["frame_range"] = [3, 3]
        with self.assertRaisesRegex(ValueError, "ordered, contiguous"):
            validate_assembly_task_config(
                task,
                parts=["body", "nozzle"],
                frame_start=0,
                frame_end=4,
            )


if __name__ == "__main__":
    unittest.main()
