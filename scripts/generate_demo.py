"""Generate classifier annotations and final transparent segmentations for demo images."""

from __future__ import annotations

from pathlib import Path

import cv2

from birdspotter.classification import BirdClassifier
from birdspotter.models import classifier_path, default_weights_dir, sam3_openvino_dir
from birdspotter.output import write_image
from birdspotter.sam3_openvino import Sam3OpenVinoSegmenter

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "demo" / "images"
ANNOTATIONS_DIR = ROOT / "demo" / "annotations"
SEGMENTATIONS_DIR = ROOT / "demo" / "segmentations"


def main() -> None:
    """Write annotations and segmentations for classifier-positive demo images."""

    classifier = BirdClassifier(classifier_path(default_weights_dir()))
    segmenter = Sam3OpenVinoSegmenter(sam3_openvino_dir(default_weights_dir()))
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(IMAGES_DIR.glob("*.png")):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode demo image: {image_path}")
        classification = classifier.classify(image)
        if classification is None:
            (ANNOTATIONS_DIR / image_path.name).unlink(missing_ok=True)
            (SEGMENTATIONS_DIR / image_path.name).unlink(missing_ok=True)
            print(f"Skipped {image_path.name}: classifier found no bird")
            continue

        annotation = image.copy()
        label = f"bird probability {classification.confidence:.2f}"
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
