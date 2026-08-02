"""Draw bird detections using the runtime INT8 OpenVINO model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from birdspotter.detection import BirdDetector
from birdspotter.models import default_weights_dir, detector_path

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
ANNOTATED_DIR = ROOT / "annotated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw YOLO26s bird detections on an image in images/."
    )
    parser.add_argument(
        "image",
        help="Image filename, such as garden.png (images/garden.png also works).",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.10,
        help="Minimum bird confidence between 0 and 1 (default: 0.10).",
    )
    return parser.parse_args()


def resolve_image_path(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (
            ROOT / candidate if candidate.parts[:1] == ("images",) else IMAGES_DIR / candidate
        )
    candidate = candidate.resolve()
    try:
        candidate.relative_to(IMAGES_DIR.resolve())
    except ValueError as error:
        raise ValueError(f"Input image must be inside {IMAGES_DIR}") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"Image not found: {candidate}")
    return candidate


def annotate(image_path: Path, confidence: float) -> Path:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")
    detector = BirdDetector(
        detector_path(default_weights_dir()),
        confidence=confidence,
    )
    detections = detector.detect(image)
    for detection in detections:
        x1, y1, x2, y2 = (round(value) for value in detection.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 220, 40), 2)
        cv2.putText(
            image,
            f"bird {detection.confidence:.2f}",
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 220, 40),
            1,
            cv2.LINE_AA,
        )

    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ANNOTATED_DIR / image_path.name
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Failed to write {output_path}")
    print(f"Saved {len(detections)} bird detection(s) to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1")
    annotate(resolve_image_path(args.image), args.confidence)


if __name__ == "__main__":
    main()
