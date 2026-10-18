"""Contract shared by the fixed-prompt SAM 3 OpenVINO and TensorRT exports."""

import cv2
import numpy as np

SAM3_IMAGE_SIZE = 504
MODEL_FILENAME = "sam3_bird.xml"
ENGINE_FILENAME = "sam3_bird.engine"


def check_frame(image_bgr: np.ndarray) -> None:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or not image_bgr.size:
        raise ValueError("SAM 3 expects a nonempty HxWx3 BGR image")


def mask_from_outputs(
    score: float, logits: np.ndarray, frame_shape: tuple[int, ...], confidence: float
) -> tuple[np.ndarray, float]:
    """Scale the best object's mask to the frame; reject absent birds and empty masks."""
    if not np.isfinite(score) or score < confidence:
        raise ValueError(f"SAM 3 found no confident bird (score={score:.3f})")
    if logits.shape != (SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE) or not np.isfinite(logits).all():
        raise ValueError("SAM 3 returned invalid mask logits")
    height, width = frame_shape[:2]
    mask = cv2.resize(logits.astype(np.float32), (width, height)) > 0
    if not mask.any():
        raise ValueError("SAM 3 returned an empty bird mask")
    return mask, score
