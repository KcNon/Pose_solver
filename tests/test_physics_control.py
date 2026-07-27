from __future__ import annotations

import unittest

from common.physics_control import (
    select_control_profile,
    settled_contact_settings,
)


class PhysicsControlTests(unittest.TestCase):
    def test_tracking_is_used_before_contact_is_latched(self) -> None:
        settings = settled_contact_settings(
            {
                "settled_contact_control": {
                    "enabled": True,
                    "states": ["static"],
                    "frequency_radps": 8.0,
                    "damping_ratio": 2.0,
                }
            }
        )
        profile = select_control_profile(
            state="static",
            contact_latched=False,
            position_error_m=0.004,
            tracking_frequency_radps=16.0,
            settled_settings=settings,
        )
        self.assertEqual(profile["mode"], "tracking")
        self.assertEqual(profile["frequency_radps"], 16.0)
        self.assertEqual(profile["damping_ratio"], 1.0)

    def test_latched_static_contact_uses_compliant_profile(self) -> None:
        settings = settled_contact_settings(
            {
                "settled_contact_control": {
                    "enabled": True,
                    "states": ["static"],
                    "frequency_radps": 7.0,
                    "damping_ratio": 2.5,
                }
            }
        )
        profile = select_control_profile(
            state="static",
            contact_latched=True,
            position_error_m=0.004,
            tracking_frequency_radps=16.0,
            settled_settings=settings,
        )
        self.assertEqual(profile["mode"], "settled_contact")
        self.assertEqual(profile["frequency_radps"], 7.0)
        self.assertEqual(profile["damping_ratio"], 2.5)

    def test_non_static_state_does_not_use_latched_profile(self) -> None:
        settings = settled_contact_settings(
            {"settled_contact_control": {"enabled": True}}
        )
        profile = select_control_profile(
            state="moving",
            contact_latched=True,
            position_error_m=0.004,
            tracking_frequency_radps=16.0,
            settled_settings=settings,
        )
        self.assertEqual(profile["mode"], "tracking")

    def test_large_pose_error_does_not_use_compliant_profile(self) -> None:
        settings = settled_contact_settings(
            {
                "settled_contact_control": {
                    "enabled": True,
                    "maximum_position_error_m": 0.01,
                }
            }
        )
        profile = select_control_profile(
            state="static",
            contact_latched=True,
            position_error_m=0.06,
            tracking_frequency_radps=16.0,
            settled_settings=settings,
        )
        self.assertEqual(profile["mode"], "tracking")

    def test_invalid_offsets_are_rejected_by_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "frequency_radps"):
            settled_contact_settings(
                {
                    "settled_contact_control": {
                        "enabled": True,
                        "frequency_radps": 0.0,
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()
