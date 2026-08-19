import unittest

from common.stable_anchor import (
    centered_stable_window,
    first_settled_window,
    rank_stable_frames,
    select_separated_candidates,
)


def cloud(overlap=0.8, residual=0.005):
    return {
        "status": "ok",
        "quality_gate": {"passed": True, "support_fraction": 0.9},
        "cross_view": {"overlap_ratio": overlap, "median_m": residual},
        "reprojection_depth": {
            "inlier_ratio": 0.8,
            "median_m": residual,
        },
        "mask_quality": {
            "valid_view_count": 8,
            "views": {
                str(index): {"area_to_median": 1.0} for index in range(8)
            },
        },
    }


class StableAnchorTest(unittest.TestCase):
    def test_legacy_quality_summary_uses_status_and_view_support(self):
        legacy = cloud()
        legacy.pop("quality_gate")
        legacy["views"] = [
            {"candidate_points": 100, "supported_points": 90},
            {"candidate_points": 50, "supported_points": 40},
        ]
        states = {
            frame: {
                "state": "static",
                "observing_views": 8,
                "motion_px": 0.1,
                "surface_shift_mm": 1.0,
            }
            for frame in range(7)
        }
        ranked = rank_stable_frames(
            states,
            {frame: legacy for frame in states},
            start=0,
            end=6,
            maximum_views=8,
        )
        self.assertTrue(any(row["usable"] for row in ranked))
        self.assertEqual(ranked[0]["cloud_quality_source"], "legacy_status")
        self.assertAlmostEqual(ranked[0]["support_fraction"], 130 / 150)

    def test_ranking_prefers_low_motion_consistent_cloud(self):
        states = {
            frame: {
                "state": "static",
                "observing_views": 8,
                "motion_px": 0.2 if frame == 5 else 2.0,
                "surface_shift_mm": 1.0 if frame == 5 else 20.0,
            }
            for frame in range(3, 8)
        }
        clouds = {frame: cloud() for frame in states}
        ranked = rank_stable_frames(
            states, clouds, start=3, end=7, maximum_views=8
        )
        self.assertEqual(ranked[0]["frame"], 5)

    def test_candidates_are_time_separated(self):
        ranking = [
            {"frame": 10, "usable": True, "shortlist_score": 1.0},
            {"frame": 12, "usable": True, "shortlist_score": 0.9},
            {"frame": 20, "usable": True, "shortlist_score": 0.8},
        ]
        self.assertEqual(
            select_separated_candidates(
                ranking, count=2, minimum_separation=5
            ),
            [10, 20],
        )

    def test_window_does_not_cross_motion(self):
        states = {
            frame: {
                "state": "moving" if frame == 8 else "static",
                "observing_views": 8,
            }
            for frame in range(5, 12)
        }
        window = centered_stable_window(
            states, 9, start=5, end=11, size=5
        )
        self.assertNotIn(8, window)
        self.assertIn(9, window)

    def test_first_settled_window_uses_first_release_event(self):
        states = {
            frame: {
                "state": (
                    "moving" if frame < 3 or 12 <= frame <= 13 else "static"
                ),
                "observing_views": 8,
                "motion_px": 0.1,
                "surface_shift_mm": 1.0,
            }
            for frame in range(20)
        }
        clouds = {frame: cloud() for frame in states}
        window, report = first_settled_window(
            states,
            clouds,
            start=0,
            end=19,
            maximum_views=8,
            minimum_views=3,
            size=4,
            settling_frames=2,
        )
        self.assertEqual(window, [5, 6, 7, 8])
        self.assertEqual(report["representative_frame"], 5)

    def test_first_settled_window_fails_without_quality_cloud(self):
        states = {
            frame: {
                "state": "static",
                "observing_views": 8,
            }
            for frame in range(10)
        }
        rejected = {
            frame: {"status": "rejected_quality"} for frame in states
        }
        with self.assertRaisesRegex(RuntimeError, "no trusted settled"):
            first_settled_window(
                states,
                rejected,
                start=0,
                end=9,
                maximum_views=8,
                minimum_views=3,
                size=4,
            )


if __name__ == "__main__":
    unittest.main()
