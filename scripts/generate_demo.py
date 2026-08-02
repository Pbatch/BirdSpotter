"""Generate detector annotations and final transparent segmentations for demo images."""

from __future__ import annotations

from pathlib import Path

import cv2

from birdspotter.crop import expanded_crop
from birdspotter.detection import BirdDetector
from birdspotter.models import default_weights_dir, detector_path, sam21_openvino_dir
from birdspotter.output import write_image
from birdspotter.sam21_openvino import Sam21OpenVinoSegmenter

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "demo" / "images"
ANNOTATIONS_DIR = ROOT / "demo" / "annotations"
SEGMENTATIONS_DIR = ROOT / "demo" / "segmentations"


def main() -> None:
    """Write one annotation and one final PNG segmentation for every demo image."""

    detector = BirdDetector(detector_path(default_weights_dir()))
    segmenter = Sam21OpenVinoSegmenter(sam21_openvino_dir(default_weights_dir()))
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(IMAGES_DIR.glob("*.png")):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode demo image: {image_path}")
        detections = detector.detect(image)
        if not detections:
            raise ValueError(f"No bird detected in demo image: {image_path}")

        detection = detections[0]
        x1, y1, x2, y2 = (round(value) for value in detection.box)
        annotation = image.copy()
        cv2.rectangle(annotation, (x1, y1), (x2, y2), (0, 0, 0), 5)
        cv2.rectangle(annotation, (x1, y1), (x2, y2), (255, 255, 255), 2)
        label = f"bird {detection.confidence:.2f}"
        label_origin = (x1, max(18, y1 - 8))
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

        crop, crop_box, _ = expanded_crop(image, detection.box)
        mask, _ = segmenter.segment(crop, crop_box)
        segmentation_path = SEGMENTATIONS_DIR / image_path.name
        write_image(segmentation_path, crop, mask)
        print(
            "Generated "
            f"{annotation_path.relative_to(ROOT)} and {segmentation_path.relative_to(ROOT)}"
        )


if __name__ == "__main__":
    main()
