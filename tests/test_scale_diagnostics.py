import unittest

from common.scale_diagnostics import (
    pareto_indices,
    select_visual_gated_candidate,
)


class ScaleDiagnosticsTests(unittest.TestCase):
    def test_pareto_front_excludes_dominated_candidates(self):
        rows = [
            {"visual_loss": 0.10, "max_penetration_m": 0.010},
            {"visual_loss": 0.11, "max_penetration_m": 0.005},
            {"visual_loss": 0.12, "max_penetration_m": 0.012},
        ]
        self.assertEqual(
            pareto_indices(rows, ("visual_loss", "max_penetration_m")),
            [0, 1],
        )

    def test_visual_gate_prefers_lowest_penetration(self):
        rows = [
            {
                "scale_factor": 0.96,
                "visual_loss": 0.125,
                "max_penetration_m": 0.001,
            },
            {
                "scale_factor": 1.0,
                "visual_loss": 0.100,
                "max_penetration_m": 0.010,
            },
            {
                "scale_factor": 1.04,
                "visual_loss": 0.140,
                "max_penetration_m": 0.015,
            },
        ]
        selected, report = select_visual_gated_candidate(
            rows, maximum_visual_loss_degradation=0.03
        )
        self.assertEqual(selected, 0)
        self.assertEqual(report["eligible_indices"], [0, 1])
        self.assertFalse(report["trajectory_mutated"])


if __name__ == "__main__":
    unittest.main()
