"""Creation of segmented bird and full-frame gallery images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def mask_bounds(mask: np.ndarray, padding: int = 2) -> tuple[int, int, int, int]:
    """Return an exclusive xyxy crop around foreground pixels."""

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot crop an empty mask")
    height, width = mask.shape
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(width, int(xs.max()) + 1 + padding)
    y2 = min(height, int(ys.max()) + 1 + padding)
    return x1, y1, x2, y2


def make_bgra(
    image_bgr: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Apply a mask as alpha and tightly crop the original source pixels."""

    if image_bgr.shape[:2] != mask.shape:
        raise ValueError("Image and mask dimensions differ")
    x1, y1, x2, y2 = mask_bounds(mask)
    alpha = np.where(mask, 255, 0).astype(np.uint8)
    bgra = np.dstack((image_bgr, alpha))
    return bgra[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def write_image(
    output_path: Path,
    image_bgr: np.ndarray,
    mask: np.ndarray,
) -> Path:
    """Atomically write the final tightly cropped RGBA PNG."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgra, _ = make_bgra(image_bgr, mask)

    temporary_output = output_path.with_name(f".{output_path.name}.part.png")
    if not cv2.imwrite(str(temporary_output), bgra):
        raise OSError(f"Failed to write {temporary_output}")
    try:
        temporary_output.replace(output_path)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    return output_path


def make_gallery_frame(
    image_bgr: np.ndarray,
    crop_mask: np.ndarray,
    crop_origin: tuple[int, int],
    detection_box: tuple[float, float, float, float],
    *,
    background_brightness: float = 0.3,
) -> np.ndarray:
    """Dim pixels outside a crop-local mask and outline the detection."""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Gallery image must be an HxWx3 BGR image")
    if crop_mask.ndim != 2:
        raise ValueError("Gallery mask must be two-dimensional")
    if not 0 <= background_brightness <= 1:
        raise ValueError("Background brightness must be between zero and one")

    frame_height, frame_width = image_bgr.shape[:2]
    origin_x, origin_y = crop_origin
    mask_height, mask_width = crop_mask.shape
    if (
        origin_x < 0
        or origin_y < 0
        or origin_x + mask_width > frame_width
        or origin_y + mask_height > frame_height
    ):
        raise ValueError("Crop mask falls outside the gallery image")

    full_mask = np.zeros((frame_height, frame_width), dtype=bool)
    full_mask[origin_y : origin_y + mask_height, origin_x : origin_x + mask_width] = crop_mask
    output = np.rint(image_bgr.astype(np.float32) * background_brightness).astype(np.uint8)
    output[full_mask] = image_bgr[full_mask]

    x1, y1, x2, y2 = detection_box
    top_left = (round(x1), round(y1))
    bottom_right = (round(x2), round(y2))
    thickness = max(2, round(min(frame_height, frame_width) / 400))
    cv2.rectangle(output, top_left, bottom_right, (80, 220, 80), thickness, cv2.LINE_AA)
    return output


def write_gallery_frame(
    output_path: Path,
    image_bgr: np.ndarray,
    crop_mask: np.ndarray,
    crop_origin: tuple[int, int],
    detection_box: tuple[float, float, float, float],
) -> Path:
    """Atomically write a full-frame gallery visualization."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualization = make_gallery_frame(image_bgr, crop_mask, crop_origin, detection_box)
    temporary_output = output_path.with_name(f".{output_path.name}.part.png")
    if not cv2.imwrite(str(temporary_output), visualization):
        raise OSError(f"Failed to write {temporary_output}")
    try:
        temporary_output.replace(output_path)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    return output_path
