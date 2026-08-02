"""Benchmark BirdSpotter's deployed OpenVINO inference components."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import cv2
import numpy as np

from birdspotter.crop import expanded_crop
from birdspotter.detection import BirdDetector, letterbox
from birdspotter.models import default_weights_dir, detector_path, sam21_openvino_dir
from birdspotter.sam21_openvino import Sam21OpenVinoSegmenter, preprocess_image, scale_box


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
    detector = BirdDetector(detector_path(weights_dir))
    segmenter = Sam21OpenVinoSegmenter(sam21_openvino_dir(weights_dir))
    prepared, _, _ = letterbox(image, detector.input_shape)
    detector_tensor = (
        np.ascontiguousarray(prepared[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        / 255.0
    )
    segmenter_tensor, resized_shape = preprocess_image(image)
    image_embeddings, high_res_256, high_res_128 = segmenter.encoder([segmenter_tensor]).values()
    centre_box = (
        image.shape[1] * 0.25,
        image.shape[0] * 0.25,
        image.shape[1] * 0.75,
        image.shape[0] * 0.75,
    )
    mask_inputs = {
        "image_embeddings": image_embeddings,
        "point_coordinates": scale_box(centre_box, image.shape[:2], resized_shape),
        "point_labels": np.array([[2, 3]], dtype=np.int32),
        "high_res_features_256": high_res_256,
        "high_res_features_128": high_res_128,
    }
    detections = detector.detect(image)
    if not detections:
        raise RuntimeError("Benchmark image contains no detected bird")
    crop, crop_box, _ = expanded_crop(image, detections[0].box)

    benchmark("detector OpenVINO", lambda: detector.backend.run(detector_tensor), runs=args.runs)
    benchmark(
        "SAM image encoder OpenVINO", lambda: segmenter.encoder([segmenter_tensor]), runs=args.runs
    )
    benchmark(
        "SAM mask predictor OpenVINO", lambda: segmenter.mask_predictor(mask_inputs), runs=args.runs
    )
    benchmark("detector application", lambda: detector.detect(image), runs=args.runs)
    benchmark("SAM application", lambda: segmenter.segment(crop, crop_box), runs=args.runs)

    def detector_plus_sam() -> None:
        current_detection = detector.detect(image)[0]
        current_crop, current_box, _ = expanded_crop(image, current_detection.box)
        segmenter.segment(current_crop, current_box)

    benchmark("detector plus SAM", detector_plus_sam, runs=max(1, args.runs // 2))


if __name__ == "__main__":
    main()
