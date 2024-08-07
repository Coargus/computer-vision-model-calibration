"""This is an example script of calibrating Yolo model with ImageNet dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from cog_cv_abstraction.image.detection.object import (
        ObjectDetectionModelBase,
    )
from cog_cv_imagenet import CogImageNetDataloader
from cvias.image.detection.object.yolo import Yolo

from calibrate_cv.calibrator import ComputerVisionModelCalibrator


def main(
    calibration_dataset: dict[str, np.ndarray],
    detection_model: ObjectDetectionModelBase | any,
) -> None:
    """Calibrate the model."""
    calibrator = ComputerVisionModelCalibrator(
        cv_model_name="YOLOv9e", method="conformal_prediction"
    )
    all_labels = calibration_dataset.keys()
    for ground_truth_label, images in calibration_dataset.items():
        for img in images:
            for label in all_labels:
                # Get the predictions from the model.
                detected_obj = detection_model.detect(img, classes=[label])
                # Get the predictions for the label.
                # Calibrate the model.
                true_positive = False
                ground_truth_class_id = detection_model.get_class_id_from_name(
                    ground_truth_label
                )
                predicted_class_id = detection_model.get_class_id_from_name(
                    detected_obj.name
                )
                if label == detected_obj.name:
                    true_positive = True

                calibrator.update_distribution(
                    ground_truth_class_id=ground_truth_class_id,
                    ground_truth_label=ground_truth_label,
                    predicted_class_id=predicted_class_id,
                    predicted_label=detected_obj.name,
                    confidence=detected_obj.confidence,
                    is_detected=true_positive,
                )
    calibrator.save_distribution(
        "/home/mc76728/repos/Coargus/computer-vision-model-calibration"
    )


if __name__ == "__main__":
    # 1. Load the image and the model.
    detection_model = Yolo(
        model_name="YOLOv9e", explicit_checkpoint_path=None, gpu_number=0
    )
    # 2. Load the dataset.
    image_dir = "/store/datasets/ILSVRC"
    loader = CogImageNetDataloader(
        imagenet_dir_path=image_dir, mapping_to="coco"
    )
    dataset = loader.dataset
    dataset.get_all_images_by_label("person")
    subset = [
        "person",
        "car",
        "airplane",
        "train",
        "boat",
        "bench",
        "knife",
        "chair",
        "cell phone",
        "traffic light",
        "stop sign",
    ]
    calibration_dataset = {}
    for obj in subset:
        data = dataset.get_all_images_by_label(obj.replace(" ", "_"))
        calibration_dataset[obj] = data

    # 3. Run a runner.
    main(
        calibration_dataset=calibration_dataset, detection_model=detection_model
    )
