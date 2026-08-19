from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from common.cloud_io import read_ply_xyz, write_ply
from common.pose_config import validate_pose_config
from common.pose_tracking import (
    bridge_pose_ranges,
    select_temporal_anchor,
    track_anchor_relative_registration,
)
from common.pose_autoconfig import (
    choose_reference_part,
    infer_anchors,
    resolve_pose_config,
)
from common.pose_validation import (
    validate_assembly_entries,
    validate_trajectory,
    validate_world_poses,
)
from common.trajectory_io import refresh_trajectory_derived_fields
from tools.stages.pose.render_multiview_pose import (
    depth_visible_foreground,
    record_visible_in_view,
)
from tools.diagnostics.export_multiview_pose_review import (
    record_visible_in_view as review_record_visible_in_view,
)


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

    def test_nonpenetration_relation_is_supported(self):
        config = base_config()
        config["trajectory_constraints"] = {
            "enabled": True,
            "geometry_proxy_config": "/unused/proxies.json",
            "relations": [{
                "name": "parts_do_not_interpenetrate",
                "type": "nonpenetration",
                "reference_part": "body",
                "moving_part": "part",
                "frame_range": [0, 2],
                "surface_points": 1000,
                "near_field_m": 0.02,
            }],
        }
        self.assertIsInstance(validate_pose_config(config), dict)

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

    def test_table_support_ranges_must_be_in_timeline(self):
        config = base_config()
        config["automation"] = {
            "table_support_ranges": {"part": [[0, 3]]}
        }
        with self.assertRaisesRegex(ValueError, "table support range"):
            validate_pose_config(config)

    def test_automatic_state_ranges_and_anchors_are_resolved(self):
        config = base_config()
        config["part_start_frames"] = {"body": 0, "part": 1}
        config["automation"] = {
            "enabled": True,
            "use_detected_states": True,
            "infer_calibration_frames": True,
            "infer_anchors": True,
            "minimum_observing_views": 1,
            "minimum_dynamic_frames": 1,
            "calibration_window_frames": 2,
            "anchor_window_frames": 2,
        }
        config["states"]["body"]["calibration_frames"] = [0]
        config["states"]["part"]["anchor_frames"] = [1]
        report = {"parts": {}}
        for part in ("body", "part"):
            states = {}
            for frame in range(3):
                moving = part == "part" and frame == 1
                states[f"{frame:06d}"] = {
                    "state": "moving" if moving else "static",
                    "observing_views": 2,
                    "motion_px": 5.0 if moving else 0.0,
                    "surface_shift_mm": 8.0 if moving else 1.0,
                }
            report["parts"][part] = {
                "states": states,
                "detected_moving_ranges": (
                    [[1, 1]] if part == "part" else []
                ),
            }
        resolved, audit = resolve_pose_config(config, report)
        self.assertEqual(
            resolved["states"]["part"]["dynamic_ranges"], [[1, 1]]
        )
        self.assertEqual(
            resolved["states"]["part"]["static_ranges"],
            [[0, 0], [2, 2]],
        )
        self.assertEqual(
            resolved["static_pose_consensus"]["parts"]["part"],
            [[0, 0], [2, 2]],
        )
        self.assertEqual(
            resolved["static_pose_consensus"]["parts"]["body"],
            [[0, 2]],
        )
        self.assertEqual(
            audit["parts"]["part"]["static_pose_lock_ranges"],
            [[0, 0], [2, 2]],
        )
        self.assertEqual(
            resolved["states"]["part"]["anchor_frames"], [2]
        )
        self.assertEqual(len(
            resolved["states"]["body"]["calibration_frames"]
        ), 2)
        self.assertIn("part", audit["parts"])
        validate_pose_config(resolved)

    def test_reference_selection_penalizes_observed_motion(self):
        config = base_config()
        config["part_start_frames"] = {"body": 1, "part": 0}
        report = {"parts": {}}
        report["parts"]["body"] = {
            "states": {
                f"{frame:06d}": {
                    "state": "static",
                    "observing_views": 2,
                }
                for frame in range(1, 3)
            }
        }
        report["parts"]["part"] = {
            "states": {
                f"{frame:06d}": {
                    "state": "moving" if frame == 1 else "static",
                    "observing_views": 2,
                }
                for frame in range(3)
            }
        }
        reference, scores = choose_reference_part(config, report)
        self.assertEqual(reference, "body")
        self.assertGreater(scores["body"], scores["part"])

    def test_moving_reference_keeps_detected_motion_and_anchors(self):
        config = base_config()
        config["automation"] = {
            "enabled": True,
            "use_detected_states": True,
            "allow_moving_reference": True,
            "infer_anchors": True,
            "minimum_observing_views": 1,
            "minimum_dynamic_frames": 1,
            "anchor_window_frames": 2,
        }
        report = {"parts": {}}
        for part in ("body", "part"):
            report["parts"][part] = {
                "states": {
                    f"{frame:06d}": {
                        "state": (
                            "moving"
                            if part == "body" and frame == 1
                            else "static"
                        ),
                        "observing_views": 2,
                        "motion_px": 8.0 if frame == 1 else 0.0,
                        "surface_shift_mm": 9.0 if frame == 1 else 1.0,
                    }
                    for frame in range(3)
                },
                "detected_moving_ranges": (
                    [[1, 1]] if part == "body" else []
                ),
            }

        resolved, _ = resolve_pose_config(config, report)

        self.assertEqual(
            resolved["states"]["body"]["dynamic_ranges"], [[1, 1]]
        )
        self.assertEqual(
            resolved["states"]["body"]["static_ranges"],
            [[0, 0], [2, 2]],
        )
        self.assertEqual(resolved["states"]["body"]["anchor_frames"], [0, 2])

    def test_automatic_anchors_avoid_motion_boundaries(self):
        config = base_config()
        config["frames"] = {"start": 0, "end": 12}
        config["part_start_frames"] = {"body": 0, "part": 0}
        config["automation"] = {
            "enabled": True,
            "use_detected_states": True,
            "infer_anchors": True,
            "minimum_observing_views": 2,
            "minimum_dynamic_frames": 1,
            "anchor_window_frames": 3,
        }
        report = {"parts": {}}
        for part in ("body", "part"):
            states = {}
            for frame in range(13):
                moving = part == "part" and 3 <= frame <= 5
                states[f"{frame:06d}"] = {
                    "state": "moving" if moving else "static",
                    "observing_views": 1 if moving else 4,
                    "motion_px": 20.0 if moving else 0.0,
                    "surface_shift_mm": 20.0 if moving else 0.0,
                }
            report["parts"][part] = {
                "states": states,
                "detected_moving_ranges": [[3, 5]] if part == "part" else [],
            }

        resolved, _ = resolve_pose_config(config, report)
        anchors = resolved["states"]["part"]["anchor_frames"]
        self.assertTrue(anchors)
        self.assertTrue(all(anchor < 3 or anchor > 5 for anchor in anchors))
        self.assertNotIn(3, anchors)
        self.assertNotIn(5, anchors)

    def test_automatic_anchor_fails_after_unobserved_static_gap(self):
        config = base_config()
        config["frames"] = {"start": 0, "end": 8}
        config["part_start_frames"] = {"body": 0, "part": 0}
        config["automation"] = {
            "enabled": True,
            "use_detected_states": True,
            "infer_anchors": True,
            "minimum_observing_views": 2,
            "minimum_dynamic_frames": 1,
            "anchor_window_frames": 2,
        }
        report = {"parts": {}}
        for part in ("body", "part"):
            states = {}
            for frame in range(9):
                moving = part == "part" and (1 <= frame <= 2 or 6 <= frame <= 7)
                observed = part == "body" or frame <= 2 or frame >= 6
                states[f"{frame:06d}"] = {
                    "state": "moving" if moving else "static",
                    "observing_views": 4 if observed else 0,
                }
            report["parts"][part] = {
                "states": states,
                "detected_moving_ranges": (
                    [[1, 2], [6, 7]] if part == "part" else []
                ),
            }

        with self.assertRaisesRegex(
            RuntimeError, "no trusted stable anchor after"
        ):
            resolve_pose_config(config, report)

    def test_anchor_inference_rejects_failed_quality_clouds(self):
        rows = {
            frame: {
                "state": "static" if frame >= 3 else "moving",
                "observing_views": 4,
                "cloud_quality": {
                    "status": "rejected_quality" if frame in {3, 4} else "ok",
                    "quality_gate": {"support_fraction": 0.5},
                    "cross_view": {"median_m": 0.01, "overlap_ratio": 0.5},
                    "reprojection_depth": {
                        "median_m": 0.01,
                        "inlier_ratio": 0.5,
                    },
                },
            }
            for frame in range(8)
        }
        anchors, windows = infer_anchors(
            rows,
            [[0, 2]],
            sequence_start=0,
            sequence_end=7,
            part_start=0,
            minimum_views=2,
            window_size=2,
            search_frames=5,
        )
        self.assertEqual(anchors, [5])
        self.assertEqual(windows["5"], [5, 6])

    def test_anchor_inference_proposes_time_separated_backups(self):
        rows = {
            frame: {
                "state": "moving" if frame < 3 else "static",
                "observing_views": 4,
                "motion_px": 0.0,
                "surface_shift_mm": float(frame % 3),
            }
            for frame in range(30)
        }
        anchors, _ = infer_anchors(
            rows,
            [[0, 2]],
            sequence_start=0,
            sequence_end=29,
            part_start=0,
            minimum_views=2,
            window_size=3,
            candidates_per_interval=3,
            candidate_minimum_separation=6,
        )
        self.assertEqual(len(anchors), 3)
        self.assertTrue(all(
            second - first >= 6
            for first, second in zip(anchors, anchors[1:])
        ))


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

    def test_renderer_uses_per_view_visibility_and_pose_validity(self):
        record = {
            "pose_valid": True,
            "observing_views": 2,
            "visible_views": ["left", "top"],
        }
        self.assertTrue(record_visible_in_view(record, "left"))
        self.assertFalse(record_visible_in_view(record, "right"))
        self.assertFalse(review_record_visible_in_view(record, "right"))
        record["pose_valid"] = False
        self.assertFalse(record_visible_in_view(record, "left"))
        self.assertFalse(review_record_visible_in_view(record, "left"))

    def test_renderer_hides_mesh_behind_observed_scene_depth(self):
        mesh_depth = np.asarray([[1.0, 1.0], [0.0, 1.0]], np.float32)
        observed = np.asarray([[0.5, 0.99], [0.2, 2.0]], np.float32)
        visible = depth_visible_foreground(
            mesh_depth,
            observed,
            margin_m=0.02,
            dilation_pixels=0,
        )
        np.testing.assert_array_equal(
            visible,
            np.asarray([[False, True], [False, True]]),
        )

    def test_local_pose_bridge_preserves_independent_endpoints(self):
        poses = {frame: np.eye(4) for frame in range(6)}
        poses[4][:3, :3] = Rotation.from_euler(
            "z", 90.0, degrees=True
        ).as_matrix()
        poses[4][0, 3] = 0.4
        reports = bridge_pose_ranges(poses, [[1, 3]])
        np.testing.assert_allclose(poses[0], np.eye(4))
        self.assertAlmostEqual(poses[4][0, 3], 0.4)
        self.assertAlmostEqual(poses[2][0, 3], 0.2)
        angle = Rotation.from_matrix(poses[2][:3, :3]).magnitude()
        self.assertAlmostEqual(float(np.degrees(angle)), 45.0)
        self.assertEqual(reports[0]["endpoint_frames"], [0, 4])

    def test_temporal_anchor_does_not_switch_inside_static_interval(self):
        anchors = [10, 100]
        static_ranges = [[0, 80], [91, 120]]
        dynamic_ranges = [[81, 90]]
        self.assertEqual(
            select_temporal_anchor(
                60,
                anchors,
                static_ranges=static_ranges,
                dynamic_ranges=dynamic_ranges,
            ),
            10,
        )
        self.assertEqual(
            select_temporal_anchor(
                85,
                anchors,
                static_ranges=static_ranges,
                dynamic_ranges=dynamic_ranges,
            ),
            10,
        )
        self.assertEqual(
            select_temporal_anchor(
                86,
                anchors,
                static_ranges=static_ranges,
                dynamic_ranges=dynamic_ranges,
            ),
            100,
        )

    def test_anchor_tracking_resets_orientation_after_visibility_gap(self):
        mesh = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
        anchor = np.eye(4)
        reacquired = np.eye(4)
        reacquired[:3, :3] = Rotation.from_euler(
            "z", 90.0, degrees=True
        ).as_matrix()
        points = np.asarray(mesh.vertices, dtype=np.float64)

        with patch(
            "common.pose_tracking.trimesh.sample.sample_surface",
            return_value=(points, np.zeros(len(points), dtype=np.int64)),
        ), patch(
            "common.pose_tracking.load_part_cloud",
            return_value=points,
        ), patch(
            "common.pose_tracking.multiscale_gicp",
            return_value=(
                np.linalg.inv(reacquired),
                {"fitness_8mm": 0.9, "median_nn_m": 0.002},
            ),
        ):
            poses, report = track_anchor_relative_registration(
                "part",
                mesh,
                1.0,
                np.zeros(3),
                0,
                3,
                {0: anchor},
                Path("/unused"),
                {
                    "max_points": 100,
                    "maximum_absolute_rotation_step_deg": 30.0,
                    "maximum_absolute_translation_step_m": 0.08,
                    "anchor_smoothing_passes": 0,
                    "reentry_minimum_absolute_fitness": 0.5,
                },
                observable_frames={0, 3},
            )

        angle = Rotation.from_matrix(poses[3][:3, :3]).magnitude()
        self.assertAlmostEqual(float(np.degrees(angle)), 90.0)
        self.assertEqual(report["000001"]["status"], "visibility_rejected")
        self.assertTrue(report["000003"]["tracklet_entry"])
        self.assertEqual(
            report["000003"]["continuity_reset"],
            "visibility_tracklet_reacquisition",
        )
        self.assertNotIn("orientation_fallback", report["000003"])

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

            advisory, advisory_failures = validate_trajectory(
                config | {
                    "parts": ["body", "insert"],
                    "states": {
                        "body": {"static_ranges": [[0, 2]], "dynamic_ranges": []},
                        "insert": {"static_ranges": [[0, 2]], "dynamic_ranges": []},
                    },
                    "reference_part": "body",
                },
                trajectory,
                enforce_assembly=False,
            )
            self.assertFalse(advisory_failures)
            self.assertFalse(advisory["assembly"][0]["passed"])
            self.assertTrue(advisory["assembly_advisory_failures"])
            self.assertFalse(report[0]["passed"])


if __name__ == "__main__":
    unittest.main()
