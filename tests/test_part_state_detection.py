import unittest
from argparse import Namespace

import numpy as np

from tools.diagnostics.detect_part_states import (
    assembly_latch_interval,
    hysteresis_states,
    mark_assembled,
    motion_score,
)


class MotionScoreTests(unittest.TestCase):
    def test_assembly_latch_ignores_later_occlusion_motion(self):
        states = (
            ["static"] * 3
            + ["moving"] * 3
            + ["static"] * 5
            + ["moving"] * 2
            + ["static"] * 2
        )
        distances = (
            [0.30, 0.31, 0.30]
            + [0.26, 0.20, 0.14]
            + [0.101, 0.099, 0.100, 0.102, 0.098]
            + [0.101, 0.100]
            + [0.099, 0.101]
        )
        latch = assembly_latch_interval(
            states,
            distances,
            0.01,
            minimum_stable_frames=5,
            minimum_approach_m=0.05,
        )
        self.assertEqual(latch, (6, 10))
        marked, _ = mark_assembled(
            states,
            distances,
            0.01,
            minimum_stable_frames=5,
            minimum_approach_m=0.05,
        )
        self.assertEqual(marked[6:], ["assembled"] * 9)

    def test_assembly_latch_rejects_static_part_without_approach(self):
        self.assertIsNone(assembly_latch_interval(
            ["static", "moving", "static", "static", "static"],
            [0.10, 0.10, 0.10, 0.101, 0.099],
            0.01,
            minimum_stable_frames=3,
            minimum_approach_m=0.03,
        ))

    def test_lagged_vote_detects_slow_cumulative_motion(self):
        frames, views = 12, 4
        centers = np.zeros((frames, views, 2), dtype=np.float64)
        centers[:, :, 0] = np.arange(frames)[:, None]
        areas = np.full((frames, views), 2000.0, dtype=np.float64)

        immediate, _ = motion_score(centers, areas, motion_lag=1)
        lagged, _ = motion_score(centers, areas, motion_lag=3)

        self.assertAlmostEqual(float(np.median(immediate[5:9])), 1.0)
        self.assertAlmostEqual(float(np.median(lagged[5:9])), 3.0)

    @staticmethod
    def _state_config():
        return Namespace(
            disp_hi=5.0,
            disp_lo=2.0,
            disp_force_hi=6.0,
            area_hi=0.10,
            area_lo=0.04,
            area_force_hi=0.20,
            surf_hi_mm=4.0,
            surf_lo_mm=3.0,
            surf_force_min_mm=2.0,
            dwell_on=2,
            dwell_off=3,
        )

    def test_strong_multiview_motion_survives_moderate_cloud_nn_shift(self):
        cfg = self._state_config()
        displacement = np.asarray([0.5, 7.0, 8.0, 7.0, 0.5, 0.5, 0.5])
        area_change = np.zeros_like(displacement)
        # In-place rotation can stay below the ordinary 4 mm entry threshold,
        # but it still rises above the measured static cloud-noise floor.
        cloud_shift = np.full_like(displacement, 2.5)
        states = hysteresis_states(
            displacement,
            area_change,
            cloud_shift,
            np.full(len(displacement), 6),
            cfg,
        )

        self.assertEqual(states[1:4], ["moving", "moving", "moving"])
        self.assertEqual(states[-1], "static")

    def test_strong_2d_mask_shift_cannot_unlock_static_cloud(self):
        cfg = self._state_config()
        displacement = np.asarray(
            [0.5, 22.0, 45.0, 45.0, 45.0, 0.5, 0.5]
        )
        area_change = np.zeros_like(displacement)
        # Object-9 frames 194..199 exhibit this exact contradiction: the mask
        # bbox moves strongly while the fused surface remains within ~1 mm.
        cloud_shift = np.asarray([0.9, 0.9, 0.84, 0.84, 0.99, 0.9, 0.8])
        states = hysteresis_states(
            displacement,
            area_change,
            cloud_shift,
            np.full(len(displacement), 8),
            cfg,
        )

        self.assertEqual(states, ["static"] * len(displacement))


if __name__ == "__main__":
    unittest.main()
