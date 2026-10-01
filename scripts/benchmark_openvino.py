"""Benchmark BirdSpotter's deployed OpenVINO inference components."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import cv2

from birdspotter.classification import BirdClassifier
from birdspotter.models import classifier_path, default_weights_dir, sam3_openvino_dir
from birdspotter.sam3_openvino import Sam3OpenVinoSegmenter, preprocess_image


def benchmark(name: str, call: Callable[[], object], *, runs: int) -> None:
    """Warm and time a zero-argument inference callable."""

    for _ in range(5):
        call()
    durations_ms: list[float] = []
    for _ in range(runs):
        started = perf_counter()
        call()
        durations_ms.append((perf_counter() - started) * 1000)
    average_ms = mean(durations_ms)
    print(
        f"{name}: mean={average_ms:.1f} ms median={median(durations_ms):.1f} ms "
        f"fps={1000 / average_ms:.2f} runs={runs}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("demo/images/1.png"))
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be positive")

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode benchmark image: {args.image}")

    weights_dir = default_weights_dir()
    classifier = BirdClassifier(classifier_path(weights_dir))
    segmenter = Sam3OpenVinoSegmenter(sam3_openvino_dir(weights_dir))
    classifier_tensor = classifier.preprocess(image)
    segmenter_tensor = preprocess_image(image)
    if not classifier.classify(image):
        raise RuntimeError("Benchmark image contains no detected bird")

    benchmark(
        "classifier OpenVINO", lambda: classifier.backend.run(classifier_tensor), runs=args.runs
    )
    benchmark(
        "SAM 3 OpenVINO", lambda: segmenter.backend({"image": segmenter_tensor}), runs=args.runs
    )
    benchmark("classifier application", lambda: classifier.classify(image), runs=args.runs)
    benchmark("SAM 3 application", lambda: segmenter.segment(image), runs=args.runs)

    def classifier_plus_sam() -> None:
        if classifier.classify(image):
            segmenter.segment(image)

    benchmark("classifier plus SAM", classifier_plus_sam, runs=max(1, args.runs // 2))


if __name__ == "__main__":
    main()
