"""Shared mask validation helpers for the SAM 2.1 OpenVINO pipeline."""

from __future__ import annotations

import cv2
import numpy as np

from birdspotter.types import Box


def component_for_box(mask: np.ndarray, box: Box) -> np.ndarray:
    """Keep the connected mask component that overlaps the detector box most."""

    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary.astype(bool)

    height, width = binary.shape
    x1 = int(np.clip(np.floor(box[0]), 0, width))
    y1 = int(np.clip(np.floor(box[1]), 0, height))
    x2 = int(np.clip(np.ceil(box[2]), 0, width))
    y2 = int(np.clip(np.ceil(box[3]), 0, height))

    best_label = 0
    best_key = (-1, -1)
    for label in range(1, count):
        overlap = int(np.count_nonzero(labels[y1:y2, x1:x2] == label))
        area = int(stats[label, cv2.CC_STAT_AREA])
        key = (overlap, area)
        if key > best_key:
            best_label = label
            best_key = key
    return labels == best_label


def validate_mask(mask: np.ndarray, box: Box, *, minimum_pixels: int = 64) -> None:
    """Reject empty or implausible masks before writing a final bird image."""

    if mask.ndim != 2:
        raise ValueError("Segmentation mask must be two-dimensional")
    pixels = int(np.count_nonzero(mask))
    if pixels < minimum_pixels:
        raise ValueError(f"Segmentation mask is too small ({pixels} pixels)")

    height, width = mask.shape
    x1 = int(np.clip(np.floor(box[0]), 0, width))
    y1 = int(np.clip(np.floor(box[1]), 0, height))
    x2 = int(np.clip(np.ceil(box[2]), 0, width))
    y2 = int(np.clip(np.ceil(box[3]), 0, height))
    if x2 <= x1 or y2 <= y1 or not np.any(mask[y1:y2, x1:x2]):
        raise ValueError("Segmentation mask does not overlap the detector box")
