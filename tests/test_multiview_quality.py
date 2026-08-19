import unittest
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from common.multiview_quality import (
    cloud_supported_view_quality,
    mask_area_quality,
    part_visibility_quality,
    temporal_mask_area_references,
    valid_mask_views,
)


class MaskAreaQualityTests(unittest.TestCase):
    def test_temporal_references_are_computed_per_camera(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for frame in (10, 11):
                target = root / f"{frame:06d}"
                target.mkdir()
                for view, count in {"wide": 80, "tight": 20}.items():
                    labels = np.zeros((10, 10), dtype=np.uint8)
                    labels.flat[:count] = 1
                    Image.fromarray(labels).save(target / f"{view}.png")
            references = temporal_mask_area_references(
                root,
                1,
                ["wide", "tight"],
                (10, 11),
                {
                    "mask_area_reference_mode": "per_view_temporal",
                    "temporal_reference_ranges": [[10, 11]],
                    "presence_minimum_full_mask_pixels": 1,
                },
            )
        self.assertEqual(references, {"wide": 80.0, "tight": 20.0})

    def test_per_view_temporal_reference_does_not_compare_unlike_cameras(self) -> None:
        masks = {}
        for name, count in {"wide": 8000, "tight": 1000}.items():
            labels = np.zeros((100, 100), dtype=np.uint8)
            labels.flat[:count] = 1
            masks[name] = labels
        report = mask_area_quality(
            masks,
            1,
            minimum_pixels=100,
            maximum_area_ratio=2.0,
            minimum_area_ratio=0.5,
            reference_areas={"wide": 8000, "tight": 1000},
        )
        self.assertTrue(report["views"]["wide"]["valid"])
        self.assertTrue(report["views"]["tight"]["valid"])
        self.assertEqual(report["area_reference_mode"], "per_view_temporal")

    def test_rejects_full_object_outlier_without_dropping_thin_views(self) -> None:
        masks = {}
        for name, count in {
            "a": 1300,
            "b": 2600,
            "c": 4000,
            "d": 5800,
            "bad": 36000,
        }.items():
            labels = np.zeros((200, 200), dtype=np.uint8)
            labels.flat[:count] = 1
            masks[name] = labels
        valid, report = valid_mask_views(
            masks,
            1,
            minimum_pixels=800,
            maximum_area_ratio=4.0,
        )
        self.assertEqual(valid, ["a", "b", "c", "d"])
        self.assertEqual(
            report["views"]["bad"]["reasons"],
            ["area_above_cross_view_ratio"],
        )

    def test_reports_empty_and_tiny_masks(self) -> None:
        masks = {
            "empty": np.zeros((10, 10), dtype=np.uint8),
            "tiny": np.pad(np.ones((2, 2), dtype=np.uint8), ((0, 8), (0, 8))),
        }
        report = mask_area_quality(masks, 1, minimum_pixels=5)
        self.assertFalse(report["views"]["empty"]["valid"])
        self.assertFalse(report["views"]["tiny"]["valid"])

    def test_validates_parameters(self) -> None:
        with self.assertRaises(ValueError):
            mask_area_quality({}, 1, minimum_pixels=0)
        with self.assertRaises(ValueError):
            mask_area_quality({}, 1, maximum_area_ratio=1.0)

    def test_cloud_support_rejects_an_occluded_view(self) -> None:
        report = cloud_supported_view_quality(
            {
                "status": "rejected_quality",
                "mask_quality": {
                    "views": {
                        "left": {"valid": True},
                        "right": {"valid": True},
                        "top": {"valid": True},
                    }
                },
                "views": [
                    {"candidate_points": 50, "supported_points": 45},
                    {"candidate_points": 60, "supported_points": 5},
                    {"candidate_points": 100, "supported_points": 30},
                ],
            },
            ["left", "right", "top"],
            minimum_supported_points=30,
            minimum_support_fraction=0.25,
        )
        self.assertTrue(report["available"])
        self.assertEqual(report["valid_view_count"], 2)
        self.assertTrue(report["views"]["left"]["valid"])
        self.assertFalse(report["views"]["right"]["valid"])
        self.assertTrue(report["views"]["top"]["valid"])

    def test_pose_visibility_requires_cross_view_supported_cameras(self) -> None:
        masks = {
            name: np.ones((20, 20), dtype=np.uint8)
            for name in ("left", "right", "top")
        }
        mask_report = mask_area_quality(masks, 1, minimum_pixels=20)
        cloud_report = cloud_supported_view_quality(
            {
                "views": [
                    {"candidate_points": 100, "supported_points": 80},
                    {"candidate_points": 100, "supported_points": 5},
                    {"candidate_points": 100, "supported_points": 0},
                ],
            },
            ["left", "right", "top"],
            minimum_supported_points=30,
            minimum_support_fraction=0.25,
        )
        report = part_visibility_quality(
            mask_report,
            ["left", "right", "top"],
            cloud_report=cloud_report,
            require_cloud_support=True,
            minimum_pose_views=2,
        )
        self.assertEqual(report["visible_views"], ["left"])
        self.assertEqual(report["observing_views"], 1)
        self.assertFalse(report["pose_valid"])

    def test_required_cloud_support_fails_closed_when_summary_is_missing(self) -> None:
        mask_report = mask_area_quality(
            {"left": np.ones((10, 10), dtype=np.uint8)},
            1,
            minimum_pixels=10,
        )
        report = part_visibility_quality(
            mask_report,
            ["left"],
            require_cloud_support=True,
        )
        self.assertFalse(report["pose_valid"])
        self.assertIn(
            "cross_view_support_unavailable",
            report["views"]["left"]["reasons"],
        )

    def test_rejected_cloud_frame_cannot_create_a_pose_tracklet(self) -> None:
        views = ["left", "right", "top"]
        mask_report = mask_area_quality(
            {view: np.ones((20, 20), dtype=np.uint8) for view in views},
            1,
            minimum_pixels=20,
        )
        cloud_report = cloud_supported_view_quality(
            {
                "status": "rejected_quality",
                "quality_gate": {
                    "passed": False,
                    "reasons": ["cross_view_median_above_maximum"],
                },
                "views": [
                    {"candidate_points": 100, "supported_points": 90}
                    for _ in views
                ],
            },
            views,
            minimum_supported_points=30,
            minimum_support_fraction=0.25,
        )
        report = part_visibility_quality(
            mask_report,
            views,
            cloud_report=cloud_report,
            require_cloud_support=True,
            minimum_pose_views=3,
        )
        self.assertEqual(report["observing_views"], 3)
        self.assertFalse(report["pose_valid"])
        self.assertFalse(report["cloud_frame_accepted"])
        self.assertIn("cloud_frame_quality_failed", report["frame_reasons"])

    def test_visible_mask_survives_unreliable_cloud_without_pose_update(self) -> None:
        views = ["left", "right", "top"]
        mask_report = mask_area_quality(
            {view: np.ones((20, 20), dtype=np.uint8) for view in views},
            1,
            minimum_pixels=20,
        )
        cloud_report = cloud_supported_view_quality(
            {
                "status": "rejected_quality",
                "quality_gate": {"passed": False, "reasons": ["bad_depth"]},
                "views": [
                    {"candidate_points": 100, "supported_points": 0}
                    for _ in views
                ],
            },
            views,
        )
        report = part_visibility_quality(
            mask_report,
            views,
            cloud_report=cloud_report,
            require_cloud_support=True,
            minimum_pose_views=2,
        )
        self.assertFalse(report["tracking_valid"])
        self.assertTrue(report["render_valid"])
        self.assertEqual(report["observation_state"], "visible_cloud_unreliable")
        self.assertEqual(report["mask_visible_views"], views)

    def test_zero_masks_are_consensus_out_of_frame(self) -> None:
        views = ["left", "right"]
        mask_report = mask_area_quality(
            {view: np.zeros((20, 20), dtype=np.uint8) for view in views},
            1,
            minimum_pixels=20,
        )
        report = part_visibility_quality(mask_report, views)
        self.assertFalse(report["render_valid"])
        self.assertEqual(report["observation_state"], "out_of_frame")


if __name__ == "__main__":
    unittest.main()
