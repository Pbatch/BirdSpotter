"""Creation of segmented bird images and per-window sighting data."""

from __future__ import annotations

import json
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


def write_window(  # noqa: PLR0913
    window_dir: Path,
    source_bgr: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    crop_mask: np.ndarray | None,
    *,
    confidence: float | None,
    sam_confidence: float | None,
) -> Path:
    """Atomically write one selection window's frame, optional mask, and label.

    ``source_bgr`` is the uncropped camera frame and ``crop_mask`` is SAM's mask
    over the ROI crop. The label stores the ROI and bbox in full-frame pixels, and
    the mask is saved cropped to that bbox. A missing mask means no bird was
    segmented, and the bbox is then null. The label is written last so its
    presence marks a complete window.
    """

    height, width = source_bgr.shape[:2]
    left, top, right, bottom = roi if roi is not None else (0, 0, width, height)
    if crop_mask is not None and crop_mask.shape != (bottom - top, right - left):
        raise ValueError("Mask dimensions differ from the ROI")
    window_dir.mkdir(parents=True, exist_ok=True)
    write_frame(window_dir / "full_frame.jpg", source_bgr)
    bbox = None
    if crop_mask is not None:
        x1, y1, x2, y2 = mask_bounds(crop_mask)
        bbox = [left + x1, top + y1, left + x2, top + y2]
        _write_encoded(
            window_dir / "mask.png",
            np.where(crop_mask[y1:y2, x1:x2], 255, 0).astype(np.uint8),
            [],
        )
    label = {
        "width": width,
        "height": height,
        "roi": list(roi) if roi is not None else None,
        "bbox": bbox,
        "confidence": confidence,
        "sam_confidence": sam_confidence,
    }
    label_path = window_dir / "label.json"
    temporary = label_path.with_name(f".{label_path.name}.part")
    temporary.write_text(json.dumps(label, indent=2) + "\n")
    temporary.replace(label_path)
    return label_path


def write_frame(output_path: Path, image_bgr: np.ndarray) -> Path:
    """Atomically write a full camera frame as a JPEG."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_encoded(output_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return output_path


def _write_encoded(output_path: Path, image: np.ndarray, params: list[int]) -> None:
    """Encode an image and atomically move it into place."""

    encoded, payload = cv2.imencode(output_path.suffix, image, params)
    if not encoded:
        raise OSError(f"Failed to encode {output_path}")
    temporary = output_path.with_name(f".{output_path.name}.part")
    temporary.write_bytes(payload.tobytes())
    temporary.replace(output_path)
