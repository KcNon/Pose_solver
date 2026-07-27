"""Pure helpers for selecting force-control profiles during contact."""
from __future__ import annotations

from typing import Any


def settled_contact_settings(simulation: dict[str, Any]) -> dict[str, Any]:
    """Return validated settings for contact-aware settled-state control."""
    raw = simulation.get("settled_contact_control", {})
    enabled = bool(raw.get("enabled", False))
    states = tuple(str(value) for value in raw.get("states", ["static"]))
    frequency = float(raw.get("frequency_radps", 8.0))
    damping_ratio = float(raw.get("damping_ratio", 2.0))
    maximum_position_error = float(
        raw.get("maximum_position_error_m", 0.01)
    )
    if not states:
        raise ValueError("settled_contact_control.states must not be empty")
    if frequency <= 0.0:
        raise ValueError(
            "settled_contact_control.frequency_radps must be positive"
        )
    if damping_ratio <= 0.0:
        raise ValueError(
            "settled_contact_control.damping_ratio must be positive"
        )
    if maximum_position_error <= 0.0:
        raise ValueError(
            "settled_contact_control.maximum_position_error_m "
            "must be positive"
        )
    return {
        "enabled": enabled,
        "states": states,
        "frequency_radps": frequency,
        "damping_ratio": damping_ratio,
        "maximum_position_error_m": maximum_position_error,
    }


def select_control_profile(
    *,
    state: str,
    contact_latched: bool,
    position_error_m: float,
    tracking_frequency_radps: float,
    settled_settings: dict[str, Any],
) -> dict[str, Any]:
    """Select tracking or compliant settled-contact controller parameters."""
    if (
        settled_settings["enabled"]
        and contact_latched
        and state in settled_settings["states"]
        and position_error_m
        <= float(settled_settings["maximum_position_error_m"])
    ):
        return {
            "mode": "settled_contact",
            "frequency_radps": float(settled_settings["frequency_radps"]),
            "damping_ratio": float(settled_settings["damping_ratio"]),
        }
    return {
        "mode": "tracking",
        "frequency_radps": float(tracking_frequency_radps),
        "damping_ratio": 1.0,
    }
