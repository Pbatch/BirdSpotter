#!/usr/bin/env python3
"""Find images where the classifier and SAM 3 disagree, and queue them for review_labels.py.

Scores every image in <images-dir>/meta.jsonl with the classifier, then runs the
fixed-prompt SAM 3 TensorRT engine only where the classifier scores at least --sam-above.
SAM's bird mask is saved as a translucent overlay (masks/*.png) when its score reaches
--mask-threshold, so the reviewer sees what SAM found. The queue holds every image,
strongest disagreement |p_classifier - p_sam| first; images SAM skipped have a null
sam_score and come last.

    python scripts/ml/mine_classifier_errors.py data/inaturalist/negatives-uk-1000
    python scripts/ml/review_labels.py data/inaturalist/negatives-uk-1000/review_queue.jsonl \\
        --image-root data/inaturalist/negatives-uk-1000
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from birdspotter.classifier import BirdClassifier
from birdspotter.segmentation.tensorrt_backend import Sam3TensorRtSegmenter

ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER = (
    ROOT
    / "weights/classifier/candidates/mobilenetv4-small-640-a100-b128-lr1e4-dp01-bce-clean-tensorrt"
)
SAM3 = ROOT / "weights/sam3/tensorrt-504"


def mask_overlay(logits: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Draw SAM's mask as a red RGBA overlay: a light fill with an opaque outline."""
    height, width = shape[:2]
    mask = cv2.resize(logits.astype(np.float32), (width, height)) > 0
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[mask] = (60, 60, 255, 90)  # BGRA
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (60, 60, 255, 255), max(2, width // 250))
    return overlay


def describe(p_classifier: float, p_sam: float) -> str:
    def call(p: float) -> str:
        return "bird" if p >= 0.5 else "no bird"

    return f"classifier {call(p_classifier)}, SAM {call(p_sam)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images_dir", type=Path, help="Folder with meta.jsonl and its images")
    parser.add_argument("--classifier", type=Path, default=CLASSIFIER)
    parser.add_argument("--sam", type=Path, default=SAM3)
    parser.add_argument("--mask-threshold", type=float, default=0.1)
    parser.add_argument(
        "--sam-above",
        type=float,
        default=0.5,
        help="Run SAM only when the classifier scores at least this (default: %(default)s)",
    )
    parser.add_argument("--queue", type=Path, help="Default: <images-dir>/review_queue.jsonl")
    args = parser.parse_args()
    if not 0 <= args.sam_above <= 1:
        parser.error("--sam-above must be between 0 and 1")

    classifier = BirdClassifier(args.classifier)
    sam = Sam3TensorRtSegmenter(args.sam)
    rows = [json.loads(line) for line in (args.images_dir / "meta.jsonl").open()]
    (args.images_dir / "masks").mkdir(exist_ok=True)
    queue: list[dict] = []
    for row in tqdm(rows, desc="mining"):
        image = cv2.imread(str(args.images_dir / row["image"]))
        if image is None:
            continue
        p_classifier = classifier.predict(image)
        source = " · ".join(str(row[key]) for key in ("group", "taxon") if row.get(key))
        if p_classifier < args.sam_above:
            queue.append(
                {
                    "image": row["image"],
                    "given": "classifier no bird, SAM skipped",
                    "p_bird": round(p_classifier, 4),
                    "sam_score": None,
                    "source": source,
                    "disagreement": None,
                }
            )
            continue
        outputs = sam.run(sam.prepare(image))
        p_sam = float(outputs["score"].reshape(-1)[0])
        item = {
            "image": row["image"],
            "given": describe(p_classifier, p_sam),
            "p_bird": round(p_classifier, 4),
            "sam_score": round(p_sam, 4),
            "source": source,
            "disagreement": round(abs(p_classifier - p_sam), 4),
        }
        if p_sam >= args.mask_threshold:
            mask = f"masks/{Path(row['image']).stem}.png"
            overlay = mask_overlay(outputs["mask_logits"].squeeze(), image.shape)
            cv2.imwrite(str(args.images_dir / mask), overlay)
            item["mask"] = mask
        queue.append(item)
    queue.sort(key=lambda item: 1 if item["disagreement"] is None else -item["disagreement"])
    output = args.queue or args.images_dir / "review_queue.jsonl"
    with output.open("w") as handle:
        for item in queue:
            handle.write(json.dumps(item) + "\n")
    counts: dict[str, int] = {}
    for item in queue:
        counts[item["given"]] = counts.get(item["given"], 0) + 1
    print(f"Queued {len(queue)} images in {output}; {sum('mask' in i for i in queue)} with masks")
    for given, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {count:5}  {given}")


if __name__ == "__main__":
    main()
