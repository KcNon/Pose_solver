from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from common.cloud_io import read_ply_xyz, write_ply
from common.pose_config import validate_pose_config
from common.pose_validation import (
    validate_assembly_entries,
    validate_trajectory,
    validate_world_poses,
)
from common.trajectory_io import refresh_trajectory_derived_fields


def base_config() -> dict:
    return {
        "frames": {"start": 0, "end": 2},
        "views": ["left", "right"],
        "parts": ["body", "part"],
        "part_ids": {"body": 1, "part": 2},
        "reference_part": "body",
        "states": {
            "body": {
                "method": "cloud_registration",
                "static_ranges": [[0, 2]],
                "dynamic_ranges": [],
            },
            "part": {
                "method": "cloud_registration",
                "static_ranges": [[0, 0], [2, 2]],
                "dynamic_ranges": [[1, 1]],
            },
        },
        "registration": {
            "voxel_sizes_m": [0.01, 0.005],
            "max_correspondence_m": [0.08, 0.04],
        },
        "mesh_dir": "/unused",
        "masks_dir": "/unused",
        "output_root": "/unused",
        "recon_backend": "test",
    }


class PoseConfigTests(unittest.TestCase):
    def test_valid_config(self):
        self.assertIsInstance(validate_pose_config(base_config()), dict)

    def test_uncovered_state_range_is_rejected(self):
        config = base_config()
        config["states"]["part"]["dynamic_ranges"] = []
        with self.assertRaisesRegex(ValueError, "do not cover"):
            validate_pose_config(config)

    def test_duplicate_part_ids_are_rejected(self):
        config = base_config()
        config["part_ids"]["part"] = 1
        with self.assertRaisesRegex(ValueError, "must be unique"):
            validate_pose_config(config)


class CloudIoTests(unittest.TestCase):
    def test_ascii_ply_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.ply"
            points = np.asarray([[1.0, 2.0, 3.0], [-0.5, 0.25, 9.0]])
            colors = np.asarray([[255, 0, 1], [2, 3, 4]], dtype=np.uint8)
            write_ply(path, points, colors)
            np.testing.assert_allclose(read_ply_xyz(path), points)


class TrajectoryTests(unittest.TestCase):
    def trajectory(self) -> dict:
        frames = {}
        for frame in range(3):
            body = np.eye(4)
            body[0, 3] = 0.1 * frame
            part = np.eye(4)
            part[:3, :3] = Rotation.from_euler(
                "z", 10.0 * frame, degrees=True
            ).as_matrix()
            part[:3, 3] = [1.0 + 0.2 * frame, 0.0, 0.0]
            frames[f"{frame:06d}"] = {
                "parts": {
                    "body": {
                        "state": "static",
                        "source": "test",
                        "observing_views": 2,
                        "T_world_from_part": body.tolist(),
                    },
                    "part": {
                        "state": "moving",
                        "source": "test",
                        "observing_views": 2,
                        "T_world_from_part": part.tolist(),
                    },
                }
            }
        return {
            "parts": ["body", "part"],
            "reference_part": "body",
            "scales": {"body": 1.0, "part": 2.0},
            "raw_mesh_origins": {
                "body": [0.0, 0.0, 0.0],
                "part": [0.5, 0.0, 0.0],
            },
            "frames": frames,
        }

    def test_refresh_is_idempotent_and_uses_per_frame_body_pose(self):
        trajectory = refresh_trajectory_derived_fields(self.trajectory())
        first = copy.deepcopy(trajectory)
        refresh_trajectory_derived_fields(trajectory)
        self.assertEqual(first, trajectory)
        translation = trajectory["frames"]["000002"]["parts"]["part"][
            "translation_body_m"
        ]
        np.testing.assert_allclose(translation, [1.2, 0.0, 0.0])

    def test_world_pose_validation_reports_step_violation(self):
        config = base_config()
        config["states"]["part"]["validation"] = {
            "max_translation_step_m": 0.05
        }
        body = {frame: np.eye(4) for frame in range(3)}
        part = {}
        for frame in range(3):
            pose = np.eye(4)
            pose[0, 3] = frame * 0.1
            part[frame] = pose
        report, failures = validate_world_poses(
            config, {"body": body, "part": part}
        )
        self.assertEqual(len(report["part"]["violations"]), 2)
        self.assertTrue(failures)

    def test_serialized_trajectory_validation(self):
        report, failures = validate_trajectory(
            base_config(), self.trajectory()
        )
        self.assertFalse(failures)
        self.assertTrue(report["passed"])
        self.assertIn("part", report["motion"])


class AssemblyValidationTests(unittest.TestCase):
    def test_moving_part_must_clear_rim_before_centering(self):
        with tempfile.TemporaryDirectory() as directory:
            mesh_dir = Path(directory)
            trimesh.creation.box(extents=[0.4, 0.2, 0.4]).export(
                mesh_dir / "body.glb"
            )
            trimesh.creation.box(extents=[0.1, 0.1, 0.1]).export(
                mesh_dir / "insert.glb"
            )
            config = {
                "mesh_dir": str(mesh_dir),
                "assembly_validation": [{
                    "name": "insert_into_body",
                    "container": "body",
                    "moving_part": "insert",
                    "frame_range": [0, 2],
                    "container_axis_part": [0.0, 1.0, 0.0],
                    "max_center_radial_m": 0.1,
                }],
            }

            def pose(x, y):
                value = np.eye(4)
                value[:3, 3] = [x, y, 0.0]
                return value.tolist()

            trajectory = {
                "parts": ["body", "insert"],
                "scales": {"body": 1.0, "insert": 1.0},
                "raw_mesh_origins": {
                    "body": [0.0, 0.0, 0.0],
                    "insert": [0.0, 0.0, 0.0],
                },
                "frames": {
                    "000000": {"parts": {
                        "body": {"T_world_from_part": pose(0.0, 0.0)},
                        "insert": {"T_world_from_part": pose(0.2, 0.2)},
                    }},
                    "000001": {"parts": {
                        "body": {"T_world_from_part": pose(0.0, 0.0)},
                        "insert": {
                            "T_world_from_part": pose(0.05, 0.2)
                        },
                    }},
                    "000002": {"parts": {
                        "body": {"T_world_from_part": pose(0.0, 0.0)},
                        "insert": {
                            "T_world_from_part": pose(0.02, 0.05)
                        },
                    }},
                },
            }
            report, failures = validate_assembly_entries(
                config, trajectory
            )
            self.assertFalse(failures)
            self.assertTrue(report[0]["passed"])
            self.assertEqual(report[0]["entry_crossings"][0]["frame"], 1)

            trajectory["frames"]["000001"]["parts"]["insert"][
                "T_world_from_part"
            ] = pose(0.05, 0.0)
            report, failures = validate_assembly_entries(
                config, trajectory
            )
            self.assertTrue(failures)
            self.assertFalse(report[0]["passed"])


if __name__ == "__main__":
    unittest.main()
