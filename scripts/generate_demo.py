"""Generate classifier annotations and final transparent segmentations for demo images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from birdspotter.classifier import BirdClassifier
from birdspotter.segmentation import sam3_segmenter
from birdspotter.sighting import mask_bounds
from birdspotter.weights import classifier_path, default_weights_dir, sam3_openvino_dir

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "demo" / "images"
ANNOTATIONS_DIR = ROOT / "demo" / "annotations"
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
    """Write annotations and segmentations for classifier-positive demo images."""

    classifier = BirdClassifier(classifier_path(default_weights_dir()))
    segmenter = sam3_segmenter(sam3_openvino_dir(default_weights_dir()))
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(IMAGES_DIR.glob("*.png")):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode demo image: {image_path}")
        confidence = classifier.classify(image)
        if confidence is None:
            (ANNOTATIONS_DIR / image_path.name).unlink(missing_ok=True)
            (SEGMENTATIONS_DIR / image_path.name).unlink(missing_ok=True)
            print(f"Skipped {image_path.name}: classifier found no bird")
            continue

        annotation = image.copy()
        label = f"bird probability {confidence:.2f}"
        label_origin = (12, 24)
        cv2.putText(
            annotation,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotation,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        annotation_path = ANNOTATIONS_DIR / image_path.name
        if not cv2.imwrite(str(annotation_path), annotation):
            raise OSError(f"Failed to write annotation: {annotation_path}")

        segmentation_path = SEGMENTATIONS_DIR / image_path.name
        try:
            mask, _ = segmenter.segment(image)
        except ValueError as error:
            segmentation_path.unlink(missing_ok=True)
            print(f"Skipped {image_path.name}: {error}")
            continue
        write_image(segmentation_path, image, mask)
        print(
            "Generated "
            f"{annotation_path.relative_to(ROOT)} and {segmentation_path.relative_to(ROOT)}"
        )


if __name__ == "__main__":
    main()
