"""Image-cropping helpers shared by the deployment and demo scripts."""

from __future__ import annotations

import numpy as np

from birdspotter.types import Box


def expanded_crop(
    image_bgr: np.ndarray,
    box: Box,
    *,
    margin_fraction: float = 0.15,
) -> tuple[np.ndarray, Box, tuple[int, int, int, int]]:
    """Crop around a detection and translate its box into crop coordinates."""

    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    margin_x = (x2 - x1) * margin_fraction
    margin_y = (y2 - y1) * margin_fraction
    crop_x1 = max(0, int(np.floor(x1 - margin_x)))
    crop_y1 = max(0, int(np.floor(y1 - margin_y)))
    crop_x2 = min(width, int(np.ceil(x2 + margin_x)))
    crop_y2 = min(height, int(np.ceil(y2 + margin_y)))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError("Detection produced an empty segmentation crop")
    local_box = (
        x1 - crop_x1,
        y1 - crop_y1,
        x2 - crop_x1,
        y2 - crop_y1,
    )
    return (
        image_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy(),
        local_box,
        (crop_x1, crop_y1, crop_x2, crop_y2),
    )
