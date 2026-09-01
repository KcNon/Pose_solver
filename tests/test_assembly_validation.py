from __future__ import annotations

import unittest

from common.assembly_validation import (
    assembly_validation_settings,
    generate_standard_perturbation_trials,
    summarize_validation_trials,
    validate_assembly_interface,
    validation_readiness_report,
)


class AssemblyValidationTests(unittest.TestCase):
    def test_standard_trials_are_symmetric_and_start_from_visual_pose(self) -> None:
        trials = generate_standard_perturbation_trials([0.001], [2.0])
        self.assertEqual(trials[0]["name"], "visual_pose_release")
        self.assertEqual(trials[0]["kind"], "baseline")
        self.assertEqual(len(trials), 9)
        self.assertEqual(
            sum(row["kind"] == "translation_perturbation" for row in trials), 4
        )
        self.assertEqual(sum(row["kind"] == "tilt_perturbation" for row in trials), 4)

    def test_validation_protocol_forbids_pose_mutation(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot enable pose mutation"):
            assembly_validation_settings(
                {"assembly_validation": {"allow_pose_mutation": True}}
            )
        settings = assembly_validation_settings({})
        self.assertFalse(settings["allow_pose_mutation"])
        self.assertEqual(settings["initial_height_m"], 0.0)
        self.assertEqual(settings["target_pose_source"], "frozen_visual_assembly_pose")
        self.assertEqual(len(settings["trials"]), 25)

    def test_cylindrical_interface_reports_clearance_and_provenance(self) -> None:
        result = validate_assembly_interface(
            {
                "type": "cylindrical_insertion",
                "reference_part": "body",
                "moving_part": "nozzle",
                "reference_axis_part": [0, 1, 0],
                "moving_axis_part": [0, 0, -2],
                "reference_outer_radius_m": 0.016,
                "moving_inner_radius_m": 0.0165,
                "parameter_source": "engineering_estimate",
                "confidence": "low",
            },
            parts={"body", "nozzle"},
        )
        self.assertAlmostEqual(result["radial_clearance_m"], 0.0005)
        self.assertEqual(result["parameter_source"], "engineering_estimate")
        self.assertEqual(result["moving_axis_part"], [0.0, 0.0, -1.0])

    def test_summary_does_not_confuse_recovery_with_baseline_validity(self) -> None:
        settings = assembly_validation_settings(
            {
                "assembly_validation": {
                    "translation_levels_m": [0.001],
                    "tilt_levels_deg": [1.0],
                }
            }
        )
        rows = []
        for index, trial in enumerate(settings["trials"]):
            rows.append({"input": trial, "success": index == 0})
        summary = summarize_validation_trials(rows, settings)
        self.assertTrue(summary["validation_passed"])
        self.assertEqual(summary["perturbation_success_rate"], 0.0)
        self.assertFalse(summary["pose_mutated"])

    def test_readiness_allows_run_but_not_metric_claim_for_estimates(self) -> None:
        interface = validate_assembly_interface(
            {
                "type": "cylindrical_insertion",
                "reference_part": "body",
                "moving_part": "nozzle",
                "reference_axis_part": [0, 1, 0],
                "moving_axis_part": [0, 1, 0],
                "reference_outer_radius_m": 0.016,
                "moving_inner_radius_m": 0.0165,
                "parameter_source": "engineering_estimate",
                "confidence": "low",
            },
            parts={"body", "nozzle"},
        )
        report = validation_readiness_report(
            interface,
            part_info={
                "body": {
                    "collision_proxy": {"type": "compound"},
                    "mass_source": "configured_assumption",
                },
                "nozzle": {
                    "collision_proxy": {"type": "cylindrical_sleeve"},
                    "mass_source": "configured_assumption",
                },
            },
            transform_checks={"body": {"passed": True}, "nozzle": {"passed": True}},
            simulation={"contact_offset_m": 0.0001},
        )
        self.assertTrue(report["runnable"])
        self.assertFalse(report["metric_physical_accuracy_claim_allowed"])
        self.assertIn(
            "connector_dimensions_not_metric_ground_truth", report["warnings"]
        )
        self.assertGreater(
            report["contact_margin"]["remaining_detection_clearance_m"], 0.0
        )

    def test_readiness_rejects_contact_margin_that_closes_clearance(self) -> None:
        interface = {
            "type": "cylindrical_insertion",
            "parameter_source": "measured",
            "confidence": "high",
            "radial_clearance_m": 0.0005,
        }
        report = validation_readiness_report(
            interface,
            part_info={},
            transform_checks={},
            simulation={"contact_offset_m": 0.001},
        )
        self.assertFalse(report["runnable"])
        self.assertIn(
            "combined_contact_offset_closes_radial_clearance", report["failures"]
        )


if __name__ == "__main__":
    unittest.main()
