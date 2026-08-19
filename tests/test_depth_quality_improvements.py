from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from common.backproject_utils import load_palette_masks
from common.depth_artifact import prediction_compatibility, prediction_metadata
from common.quality_cloud import ViewCloud, fuse_supported_clouds
from scripts.run_depth_pipeline import build_da3_command
from tools.diagnostics.render_multiview_point_cloud import (
    projection_bases,
    render_projection,
)
from tools.stages.depth.select_point_cloud_variants import (
    materialize_selection,
    parse_part_sources,
    parse_required_ranges,
)


def prediction_payload() -> dict[str, np.ndarray]:
    payload = {
        "images": np.zeros((2, 3, 280, 504), np.float32),
        "depth": np.ones((2, 280, 504, 1), np.float32),
        "depth_conf": np.ones((2, 280, 504), np.float32),
        "extrinsic": np.zeros((2, 3, 4), np.float32),
        "intrinsic": np.zeros((2, 3, 3), np.float32),
        "world_points_from_depth": np.zeros((2, 280, 504, 3), np.float32),
        "view_names": np.asarray(["left", "right"]),
    }
    payload.update(prediction_metadata(
        process_res=504,
        process_res_method="upper_bound_resize",
        source_image_hw=(1080, 1920),
        processed_image_hw=(280, 504),
        camera_frames=["000100"],
        model_dir="/model",
        use_ray_pose=False,
        ref_view_strategy="saddle_balanced",
    ))
    return payload


class DepthArtifactTests(unittest.TestCase):
    def compatibility(self, payload, **overrides):
        request = {
            "views": ["left", "right"],
            "process_res": 504,
            "process_res_method": "upper_bound_resize",
            "source_image_hw": (1080, 1920),
            "camera_frames": ["000100"],
            "model_dir": "/model",
            "use_ray_pose": False,
            "ref_view_strategy": "saddle_balanced",
        }
        request.update(overrides)
        return prediction_compatibility(payload, **request)

    def test_metadata_compatible_prediction_resumes(self) -> None:
        valid, reason = self.compatibility(prediction_payload())
        self.assertTrue(valid, reason)

    def test_resolution_change_invalidates_prediction(self) -> None:
        valid, reason = self.compatibility(
            prediction_payload(), process_res=756
        )
        self.assertFalse(valid)
        self.assertIn("process_res", reason)

    def test_legacy_resume_requires_explicit_shape_policy(self) -> None:
        payload = prediction_payload()
        for key in list(payload):
            if key in {
                "pose_solver_depth_schema_version",
                "process_res",
                "process_res_method",
                "source_image_hw",
                "processed_image_hw",
                "camera_frames",
                "model_dir",
                "use_ray_pose",
                "ref_view_strategy",
            }:
                del payload[key]
        self.assertFalse(self.compatibility(payload)[0])
        self.assertTrue(self.compatibility(
            payload, allow_legacy_shape_resume=True
        )[0])

    def test_pipeline_command_contains_resolution_and_multiframe_rig(self) -> None:
        command = build_da3_command(
            {
                "frames_dir": "/frames",
                "da3_self_cond_dir": "/depth",
                "views": ["left", "right"],
            },
            {
                "da3_python": "/da3/python",
                "camera_frames": [10, 20, 30],
                "process_res": 756,
                "full_w": 1920,
                "full_h": 1080,
                "use_ray_pose": True,
            },
            ["000100"],
        )
        self.assertEqual(command[command.index("--process-res") + 1], "756")
        start = command.index("--camera-frames")
        self.assertEqual(command[start + 1:start + 4], ["000010", "000020", "000030"])
        self.assertIn("--use-ray-pose", command)

    def test_pipeline_command_can_reuse_fixed_rig_npz(self) -> None:
        command = build_da3_command(
            {
                "frames_dir": "/frames",
                "da3_self_cond_dir": "/depth",
                "views": ["left", "right"],
            },
            {
                "da3_python": "/da3/python",
                "camera_npz": "/previous/predictions.npz",
                "process_res": 1008,
            },
            ["000100"],
        )
        self.assertIn("--camera-npz", command)
        self.assertNotIn("--camera-frame", command)
        self.assertNotIn("--camera-frames", command)


class PointCloudResolutionTests(unittest.TestCase):
    def test_coverage_resize_preserves_thin_part(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "000000"
            frame.mkdir()
            labels = np.zeros((8, 8), np.uint8)
            labels[1, 1] = 1
            Image.fromarray(labels, mode="P").save(frame / "camera.png")
            nearest = load_palette_masks(
                directory,
                "000000",
                ["thin"],
                (2, 2),
                views=["camera"],
                part_ids={"thin": 1},
            )["thin"][0]
            coverage = load_palette_masks(
                directory,
                "000000",
                ["thin"],
                (2, 2),
                views=["camera"],
                part_ids={"thin": 1},
                resize_mode="coverage",
                coverage_threshold=0.05,
            )["thin"][0]
            hybrid = load_palette_masks(
                directory,
                "000000",
                ["thin"],
                (2, 2),
                views=["camera"],
                part_ids={"thin": 1},
                resize_mode="hybrid",
                coverage_threshold=0.05,
                coverage_parts=["thin"],
            )["thin"][0]
            self.assertEqual(int(nearest.sum()), 0)
            self.assertEqual(int(coverage.sum()), 1)
            self.assertEqual(int(hybrid.sum()), 1)

    def test_confidence_weighted_voxel_fusion(self) -> None:
        cloud = ViewCloud(
            points=np.asarray([[0.0, 0.0, 1.0], [0.001, 0.0, 1.0]]),
            colors=np.asarray([[0, 0, 0], [100, 0, 0]], np.uint8),
            confidence=np.asarray([1.0, 3.0], np.float32),
            support=np.asarray([1, 1], np.int16),
            candidate_pixels=2,
        )
        points, colors, stats = fuse_supported_clouds(
            [cloud], min_support=1, voxel_size_m=0.005
        )
        self.assertEqual(stats["pre_voxel_points"], 2)
        self.assertEqual(stats["fused_points"], 1)
        self.assertAlmostEqual(float(points[0, 0]), 0.00075, places=6)
        self.assertEqual(int(colors[0, 0]), 75)

    def test_dependency_light_renderer(self) -> None:
        cloud = np.asarray([
            [-0.01, -0.01, 1.0],
            [0.01, -0.01, 1.0],
            [0.01, 0.01, 1.0],
            [-0.01, 0.01, 1.0],
        ])
        image = render_projection(
            [cloud],
            [(255, 100, 50)],
            projection_bases()[0][1],
            size=(320, 240),
            title="test",
        )
        self.assertEqual(image.size, (320, 240))
        self.assertGreater(np.asarray(image).std(), 0.0)

    def test_per_part_cloud_selection_is_complete_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main_root = root / "main_source"
            collector_root = root / "collector_source"
            for source, part in (
                (main_root, "main"),
                (collector_root, "collector"),
            ):
                for frame in (1, 2):
                    frame_root = source / f"{frame:06d}"
                    (frame_root / "views" / part).mkdir(parents=True)
                    (frame_root / f"{part}.ply").write_text("ply\n")
                    (frame_root / "views" / part / "camera.ply").write_text(
                        "ply\n"
                    )
            sources = parse_part_sources([
                f"main={main_root}",
                f"collector={collector_root}",
            ])
            output = root / "selected"
            first = materialize_selection(output, sources, [1, 2])
            second = materialize_selection(output, sources, [1, 2])
            self.assertEqual(first, second)
            self.assertEqual(first["artifact_count"], 4)
            self.assertEqual(
                (output / "000001" / "main.ply").resolve(),
                (main_root / "000001" / "main.ply").resolve(),
            )

    def test_per_part_cloud_selection_fails_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "000001" / "views" / "main").mkdir(parents=True)
            (source / "000001" / "main.ply").write_text("ply\n")
            output = root / "selected"
            with self.assertRaises(FileNotFoundError):
                materialize_selection(output, {"main": source}, [1, 2])
            self.assertFalse(output.exists())

    def test_per_part_cloud_selection_preserves_quality_gate_holes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "000001" / "views" / "main").mkdir(parents=True)
            (source / "000001" / "main.ply").write_text("ply\n")
            # A rejected fused cloud may intentionally retain its per-view
            # diagnostics; that is still a quality-gate hole, not corruption.
            (source / "000002" / "views" / "main").mkdir(parents=True)
            required = parse_required_ranges(["main=1:1"], {"main"})
            manifest = materialize_selection(
                root / "selected",
                {"main": source},
                [1, 2],
                allow_missing=True,
                required_frames_by_part=required,
            )
            self.assertEqual(manifest["missing_frames_by_part"], {"main": [2]})
            self.assertFalse((root / "selected" / "000002").exists())


if __name__ == "__main__":
    unittest.main()
