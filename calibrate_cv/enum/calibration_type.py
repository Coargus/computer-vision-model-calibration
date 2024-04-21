"""ENUM for calibration type."""

from __future__ import annotations

import enum


class CalibrationType(enum.Enum):
    """ENUM for calibration type."""

    CONFORMAL_PREDICTION = "conformal_prediction"
    TEMPERATURE_SCALING = "temperature_scaling"
