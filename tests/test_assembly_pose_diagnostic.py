import unittest

import numpy as np

from tools.diagnostics.diagnose_assembly_pose import (
    axial_candidate_pose,
    summarize_candidates,
)


class AssemblyPoseDiagnosticTests(unittest.TestCase):
    def test_axial_candidate_uses_reference_rotation(self) -> None:
        reference = np.eye(4)
        reference[:3, :3] = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        candidate, axis = axial_candidate_pose(
            np.eye(4), reference, np.array([1.0, 0.0, 0.0]), 0.015
        )
        np.testing.assert_allclose(axis, [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(candidate[:3, 3], [0.0, 0.015, 0.0])

    def test_summary_rejects_visually_worse_contact_correction(self) -> None:
        offsets = [0.0, 0.015]
        frames = []
        for frame in (427, 500):
            frames.append({
                "frame": frame,
                "candidates": {
                    "+0.000000": {
                        "loss": 0.20,
                        "mean_iou": 0.72,
                        "worst_view_iou": 0.50,
                        "mean_target_coverage": 0.75,
                    },
                    "+0.015000": {
                        "loss": 0.29,
                        "mean_iou": 0.65,
                        "worst_view_iou": 0.40,
                        "mean_target_coverage": 0.66,
                    },
                },
            })
        summary = summarize_candidates(frames, offsets, 0.015)
        self.assertEqual(
            summary["diagnosis"],
            "visual_pose_supports_raw_pose_and_rejects_contact_correction",
        )
        self.assertEqual(summary["best"]["offset_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
