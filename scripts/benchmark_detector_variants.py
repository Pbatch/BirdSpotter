#!/usr/bin/env python3
"""Compare end-to-end latency and predictions for OpenVINO detector variants."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import cv2
import numpy as np

from birdspotter.detection import BirdDetector


def load_images(paths: list[Path]) -> list[tuple[Path, np.ndarray]]:
    """Decode benchmark images once so disk access is excluded from timings."""

    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode benchmark image: {path}")
        images.append((path, image))
    return images


def benchmark_model(
    model_path: Path,
    images: list[tuple[Path, np.ndarray]],
    *,
    warmup: int,
    runs: int,
) -> None:
    """Print prediction summaries and warmed end-to-end detector timings."""

    detector = BirdDetector(model_path)
    for path, image in images:
        detections = detector.detect(image)
        summary = "none"
        if detections:
            top = detections[0]
            summary = (
                f"count={len(detections)} confidence={top.confidence:.3f} "
                f"box={tuple(round(value, 1) for value in top.box)}"
            )
        print(f"prediction model={model_path.name} image={path.name} {summary}")

    for index in range(warmup):
        detector.detect(images[index % len(images)][1])

    durations_ms = []
    for index in range(runs):
        image = images[index % len(images)][1]
        started = perf_counter()
        detector.detect(image)
        durations_ms.append((perf_counter() - started) * 1000)

    ordered = sorted(durations_ms)
    percentile_95 = ordered[math.ceil(len(ordered) * 0.95) - 1]
    average = mean(durations_ms)
    height, width = detector.input_shape
    print(
        f"benchmark model={model_path.name} input={height}x{width} "
        f"mean={average:.1f}ms median={median(durations_ms):.1f}ms "
        f"p95={percentile_95:.1f}ms fps={1000 / average:.2f} runs={runs}"
    )


def main() -> None:
    """Parse model and image paths and benchmark every model in order."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, type=Path)
    parser.add_argument("--image", action="append", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("Warmup must be non-negative and runs must be positive")

    images = load_images([path.resolve() for path in args.image])
    for model_path in args.model:
        benchmark_model(model_path.resolve(), images, warmup=args.warmup, runs=args.runs)


if __name__ == "__main__":
    main()
