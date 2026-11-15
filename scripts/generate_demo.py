"""Generate transparent SAM 3 segmentations for the demo images.

The motion gate needs a stream of frames to build its background, so the demo
images, which are single stills, go straight to SAM 3.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from birdspotter.segmentation import sam3_segmenter
from birdspotter.sighting import mask_bounds
from birdspotter.weights import default_weights_dir, sam3_openvino_dir

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "demo" / "images"
SEGMENTATIONS_DIR = ROOT / "demo" / "segmentations"


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


def main() -> None:
    """Write transparent SAM 3 segmentations for the demo images."""

    segmenter = sam3_segmenter(sam3_openvino_dir(default_weights_dir()))
    SEGMENTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(IMAGES_DIR.glob("*.png")):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode demo image: {image_path}")
        segmentation_path = SEGMENTATIONS_DIR / image_path.name
        try:
            mask, _ = segmenter.segment(image)
        except ValueError as error:
            segmentation_path.unlink(missing_ok=True)
            print(f"Skipped {image_path.name}: {error}")
            continue
        write_image(segmentation_path, image, mask)
        print(f"Generated {segmentation_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
