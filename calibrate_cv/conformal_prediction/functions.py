"""Conformal Prediction functions for calibrating computer vision models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def process_raw_data_distribution(
    data_distribution_path: str,
    sample_number: int = 2000,
    random_state: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate model with conformal prediction."""
    data = pd.read_csv(data_distribution_path)
    data_sample = data.sample(n=sample_number, random_state=random_state)
    confidence = data_sample["confidence"].to_numpy()
    prediction = data_sample["predicted_class_id"].to_numpy()
    ground_truth = data_sample["ground_truth_class_id"].to_numpy()
    return confidence, prediction, ground_truth


def get_non_conformity_score(
    confidence: np.ndarray, prediction: np.ndarray, ground_truth: np.ndarray
) -> list:
    """Get non-conformity score."""
    nc_scores = []
    for i, conf in enumerate(confidence):
        if prediction[i] == ground_truth[i]:
            nc_scores.append(1 - conf)
        else:
            nc_scores.append(conf)
    return nc_scores


def estimate_guarantee(
    confidence_score: float, non_conformity_score: list
) -> float:
    """Estimate guarantee.

    Args:
        confidence_score (float): Confidence score.
        non_conformity_score (list): Non-conformity score.

    Returns:
    float: Existence guarantee.
    """
    threshold = 0.5
    total = len(non_conformity_score)
    count = 0
    for nc_score in non_conformity_score:
        if nc_score < confidence_score:
            count += 1

    if confidence_score >= threshold:
        # estimate existence_guarantee

        return count / total
    # estimate non-existence_guarantee
    return 1 - count / total


def calibrate_confidence_score(
    confidence: float, non_conformity_score: list
) -> float:
    """Calibrate confidence."""
    return estimate_guarantee(confidence, non_conformity_score)
