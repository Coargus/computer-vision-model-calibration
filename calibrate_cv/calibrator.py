"""Calibrator class for calibrating computer vision models."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class ComputerVisionModelCalibrator:
    """Calibrator class for calibrating computer vision models."""

    def __init__(
        self,
        cv_model_name: str,
        method: str = "conformal_prediction",
    ) -> None:
        """Initialize the calibrator with the model and data loader."""
        self.cv_model_name = cv_model_name
        self.method = method
        self.distribution = {}

    def update_distribution(  # noqa: PLR0913
        self,
        ground_truth_class_id: any,
        ground_truth_label: str,
        predicted_class_id: any,
        predicted_label: str,
        confidence: float,
        is_detected: bool,
    ) -> None:
        """Update data distribution for sampling."""
        if predicted_label not in self.distribution:
            self.distribution[predicted_label] = {
                "predicted_class_id": [],
                "ground_truth_label": [],
                "ground_truth_class_id": [],
                "confidence": [],
                "is_detected": [],
            }
        self.distribution[predicted_label]["predicted_class_id"].append(
            predicted_class_id
        )
        self.distribution[predicted_label]["ground_truth_label"].append(
            ground_truth_label
        )
        self.distribution[predicted_label]["ground_truth_class_id"].append(
            ground_truth_class_id
        )
        self.distribution[predicted_label]["confidence"].append(confidence)
        self.distribution[predicted_label]["is_detected"].append(is_detected)

    def save_distribution(self, saving_dir: str | Path) -> None:
        """Save the distribution to a file."""
        if isinstance(saving_dir, str):
            destination_path = Path(saving_dir)

        data_to_export = []
        for predicted_label, contents in self.distribution.items():
            for i in range(len(contents["predicted_class_id"])):
                row = {
                    "predicted_label": predicted_label,
                    "predicted_class_id": contents["predicted_class_id"][i],
                    "ground_truth_label": contents["ground_truth_label"][i],
                    "ground_truth_class_id": contents["ground_truth_class_id"][
                        i
                    ],
                    "confidence": contents["confidence"][i],
                    "is_detected": contents["is_detected"][i],
                }
                data_to_export.append(row)

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data_to_export)

        timestamp = datetime.now(timezone.utc).isoformat()

        metadata = {
            "timestamp": timestamp,
            "cv_model": self.cv_model_name,
            "calibration_method": self.method,
            "total_number_of_images": int(
                len(data_to_export) / len(self.distribution[predicted_label])
            ),
        }
        destination_path = destination_path / f"calibration_result_{timestamp}"

        destination_path.mkdir(parents=True, exist_ok=True)

        file_path = destination_path / self.cv_model_name
        json_path = destination_path / "metadata.json"
        csv_path = destination_path / f"{file_path}_data_distribution.csv"
        pickle_path = destination_path / f"{file_path}_data_distribution.pkl"

        # Save the combined data and metadata to a JSON file
        with Path(json_path).open("w") as json_file:
            json.dump(metadata, json_file, indent=4)

        with Path(pickle_path).open("wb") as pickle_file:
            pickle.dump(self.distribution, pickle_file)

        df.to_csv(csv_path, index=False)
