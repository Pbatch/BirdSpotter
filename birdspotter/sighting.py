"""On-disk sightings: one folder per five-minute window with its frame, bird mask and label."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import cv2
import numpy as np

from birdspotter.capture import Roi

WINDOW_MINUTES = 5
WINDOW_FORMAT = "%Y-%m-%d_%H-%M"
WINDOW_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$")
FRAME_FILENAME = "full_frame.jpg"
MASK_FILENAME = "mask.png"
LABEL_FILENAME = "label.json"


@dataclass(frozen=True, slots=True)
class Sighting:
    """One saved selection window, as described by its ``label.json``.

    The ROI and bbox are in full-frame pixels. A null bbox means no bird was segmented.
    """

    path: Path
    width: int
    height: int
    roi: Roi | None
    bbox: Roi | None
    confidence: float | None
    sam_confidence: float | None

    @property
    def has_bird(self) -> bool:
        return self.bbox is not None

    @property
    def frame_roi(self) -> Roi:
        """Return the ROI, or the whole frame when the window was classified uncropped."""

        return self.roi if self.roi is not None else (0, 0, self.width, self.height)

    @classmethod
    def load(cls, path: Path) -> Self:
        label = json.loads((path / LABEL_FILENAME).read_text())
        return cls(
            path=path,
            width=label["width"],
            height=label["height"],
            roi=_box(label["roi"]),
            bbox=_box(label["bbox"]),
            confidence=label["confidence"],
            sam_confidence=label["sam_confidence"],
        )

    def save_label(self) -> Path:
        """Atomically write ``label.json``, whose presence marks the window as complete."""

        label = {
            "width": self.width,
            "height": self.height,
            "roi": list(self.roi) if self.roi is not None else None,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "confidence": self.confidence,
            "sam_confidence": self.sam_confidence,
        }
        label_path = self.path / LABEL_FILENAME
        temporary = label_path.with_name(f".{label_path.name}.part")
        temporary.write_text(json.dumps(label, indent=2) + "\n")
        temporary.replace(label_path)
        return label_path


def _box(value: list[int] | None) -> Roi | None:
    return None if value is None else (value[0], value[1], value[2], value[3])


def window_start(timestamp: datetime) -> datetime:
    """Return the start of the clock-aligned five-minute UTC window containing a timestamp."""

    timestamp = timestamp.astimezone(UTC)
    return timestamp.replace(
        minute=timestamp.minute - timestamp.minute % WINDOW_MINUTES, second=0, microsecond=0
    )


def window_dir(output_dir: Path, captured_at: datetime) -> Path:
    """Return the directory named after the start of a frame's selection window."""

    return output_dir / f"{window_start(captured_at):{WINDOW_FORMAT}}"


def recent_birds(output_dir: Path, limit: int) -> list[Sighting]:
    """Return the newest complete sightings containing a segmented bird, newest first."""

    birds = []
    windows = sorted(
        (
            path
            for path in output_dir.iterdir()
            if path.is_dir() and WINDOW_NAME.fullmatch(path.name)
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    for window in windows:
        if not (window / LABEL_FILENAME).is_file() or not (window / MASK_FILENAME).is_file():
            continue
        sighting = Sighting.load(window)
        if not sighting.has_bird:
            continue
        birds.append(sighting)
        if len(birds) == limit:
            break
    return birds


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


def write_window(  # noqa: PLR0913
    window_dir: Path,
    source_bgr: np.ndarray,
    roi: Roi | None,
    crop_mask: np.ndarray | None,
    *,
    confidence: float | None,
    sam_confidence: float | None,
) -> Sighting:
    """Atomically write one selection window's frame, optional mask, and label.

    ``source_bgr`` is the uncropped camera frame and ``crop_mask`` is SAM's mask
    over the ROI crop. The mask is saved cropped to the bird's bbox. The label is
    written last so its presence marks a complete window.
    """

    height, width = source_bgr.shape[:2]
    left, top, right, bottom = roi if roi is not None else (0, 0, width, height)
    if crop_mask is not None and crop_mask.shape != (bottom - top, right - left):
        raise ValueError("Mask dimensions differ from the ROI")
    window_dir.mkdir(parents=True, exist_ok=True)
    _write_encoded(window_dir / FRAME_FILENAME, source_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    bbox = None
    if crop_mask is not None:
        x1, y1, x2, y2 = mask_bounds(crop_mask)
        bbox = (left + x1, top + y1, left + x2, top + y2)
        _write_encoded(
            window_dir / MASK_FILENAME,
            np.where(crop_mask[y1:y2, x1:x2], 255, 0).astype(np.uint8),
            [],
        )
    sighting = Sighting(
        path=window_dir,
        width=width,
        height=height,
        roi=roi,
        bbox=bbox,
        confidence=confidence,
        sam_confidence=sam_confidence,
    )
    sighting.save_label()
    return sighting


def _write_encoded(output_path: Path, image: np.ndarray, params: list[int]) -> None:
    """Encode an image and atomically move it into place."""

    encoded, payload = cv2.imencode(output_path.suffix, image, params)
    if not encoded:
        raise OSError(f"Failed to encode {output_path}")
    temporary = output_path.with_name(f".{output_path.name}.part")
    temporary.write_bytes(payload.tobytes())
    temporary.replace(output_path)
