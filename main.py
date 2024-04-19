"""Example main run script for the project."""

from __future__ import annotations

import numpy as np
from cog_cv_abstraction.image.detection.object import ObjectDetectionModelBase

from calibrate_cv.calibrator import ComputerVisionModelCalibrator


def main(
    image: list[np.ndarray] | np.ndarray,
    label: list | str,
    cv_model: ObjectDetectionModelBase | any,
) -> None:
    ComputerVisionModelCalibrator()


if __name__ == "__main__":
    main()
