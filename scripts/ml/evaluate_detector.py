#!/usr/bin/env python3
"""Evaluate an INT8 OpenVINO bird detector against an Ultralytics validation split."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import torch
import yaml
from ultralytics.utils.metrics import ap_per_class, box_iou

from birdspotter.detection import OpenVinoDetectorBackend, letterbox, restore_box

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def label_boxes(label_path: Path, width: int, height: int) -> np.ndarray:
    """Read normalized YOLO labels as pixel xyxy boxes."""
    boxes = []
    for line in label_path.read_text().splitlines():
        values = [float(value) for value in line.split()]
        if len(values) != 5:
            raise ValueError(f"Invalid detection label in {label_path}: {line}")
        class_id, center_x, center_y, box_width, box_height = values
        if round(class_id) != 0:
            continue
        boxes.append(
            [
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                (center_x + box_width / 2) * width,
                (center_y + box_height / 2) * height,
            ]
        )
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def true_positives(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Match predictions to targets independently at each COCO IoU threshold."""
    correct = np.zeros((len(predictions), len(IOU_THRESHOLDS)), dtype=bool)
    if not len(predictions) or not len(targets):
        return correct
    iou = box_iou(torch.from_numpy(targets), torch.from_numpy(predictions)).numpy()
    for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
        target_indexes, prediction_indexes = np.nonzero(iou >= threshold)
        if not len(prediction_indexes):
            continue
        matches = np.column_stack(
            (target_indexes, prediction_indexes, iou[target_indexes, prediction_indexes])
        )
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), threshold_index] = True
    return correct


def resolve_validation_images(dataset_yaml: Path) -> tuple[list[Path], Path]:
    """Resolve validation images and labels from an Ultralytics dataset YAML."""
    data = yaml.safe_load(dataset_yaml.read_text())
    root = Path(data.get("path", dataset_yaml.parent))
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    image_dir = root / data["val"]
    label_dir = root / str(data["val"]).replace("images", "labels", 1)
    images = sorted(path for path in image_dir.iterdir() if path.is_file())
    if not images:
        raise FileNotFoundError(f"No validation images found in {image_dir}")
    return images, label_dir


def evaluate(model_dir: Path, dataset_yaml: Path, limit: int | None, device: str) -> None:
    """Run the detector and print Ultralytics-compatible detection metrics."""
    images, label_dir = resolve_validation_images(dataset_yaml)
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive")
        images = images[:limit]
    backend = OpenVinoDetectorBackend(
        model_dir,
        device=device,
        performance_hint="THROUGHPUT",
    )
    infer_queue = ov.AsyncInferQueue(backend.compiled_model)
    correct, confidences = [], []
    target_count = 0
    completed = 0
    started = time.perf_counter()
    progress_lock = threading.Lock()

    def collect_result(request: ov.InferRequest, user_data: tuple) -> None:
        """Collect one asynchronous inference result."""
        nonlocal completed
        targets, scale, padding, image_shape = user_data
        rows = np.asarray(request.get_output_tensor(0).data).squeeze(0)
        detections = [
            (restore_box(row[:4], scale, padding, image_shape), float(row[4]))
            for row in rows
            if float(row[4]) >= 0.001 and round(float(row[5])) == 0
        ]
        predictions = np.asarray([box for box, _ in detections], dtype=np.float32).reshape(-1, 4)
        with progress_lock:
            correct.append(true_positives(predictions, targets))
            confidences.extend(confidence for _, confidence in detections)
            completed += 1
            if completed % 100 == 0 or completed == len(images):
                elapsed = time.perf_counter() - started
                print(
                    f"Evaluated {completed}/{len(images)} images "
                    f"({completed / elapsed:.2f} images/s)",
                    flush=True,
                )

    infer_queue.set_callback(collect_result)

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode validation image: {image_path}")
        targets = label_boxes(label_dir / f"{image_path.stem}.txt", image.shape[1], image.shape[0])
        prepared, scale, padding = letterbox(image, backend.input_shape[-2:])
        tensor = prepared[:, :, ::-1].transpose(2, 0, 1)
        tensor = np.ascontiguousarray(tensor[None], dtype=np.float32) / 255.0
        infer_queue.start_async(tensor, (targets, scale, padding, image.shape))
        target_count += len(targets)
    infer_queue.wait_all()

    correct_array = np.concatenate(correct) if correct else np.empty((0, 10), dtype=bool)
    confidence_array = np.asarray(confidences, dtype=np.float32)
    prediction_classes = np.zeros(len(confidences), dtype=np.float32)
    target_classes = np.zeros(target_count, dtype=np.float32)
    _, _, precision, recall, _, average_precision, *_ = ap_per_class(
        correct_array,
        confidence_array,
        prediction_classes,
        target_classes,
        names={0: "bird"},
    )
    print(f"Images: {len(images)}")
    print(f"Targets: {target_count}")
    print(f"Predictions: {len(confidences)}")
    print(f"Precision: {float(precision[0]):.5f}")
    print(f"Recall: {float(recall[0]):.5f}")
    print(f"mAP50: {float(average_precision[0, 0]):.5f}")
    print(f"mAP50-95: {float(average_precision[0].mean()):.5f}")


def main() -> None:
    """Parse command-line arguments and evaluate an OpenVINO detector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="CPU")
    args = parser.parse_args()
    evaluate(args.model_dir.resolve(), args.data.resolve(), args.limit, args.device)


if __name__ == "__main__":
    main()
