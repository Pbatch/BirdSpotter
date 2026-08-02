"""Creation of the downstream transparent bird image."""

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
