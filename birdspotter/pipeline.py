"""Orchestration for image and webcam bird selection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from birdspotter.capture import LatestFrameCamera
from birdspotter.detection import BirdDetector
from birdspotter.output import write_image
from birdspotter.types import BirdCandidate, Box, better_candidate


class Segmenter(Protocol):
    def segment(self, image_bgr: np.ndarray, box: Box) -> tuple[np.ndarray, float]: ...

    def describe(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class SavedBird:
    image_path: Path
    confidence: float


def expanded_crop(
    image_bgr: np.ndarray,
    box: Box,
    *,
    margin_fraction: float = 0.15,
) -> tuple[np.ndarray, Box, tuple[int, int, int, int]]:
    """Crop around a detection and translate its box into crop coordinates."""

    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    margin_x = (x2 - x1) * margin_fraction
    margin_y = (y2 - y1) * margin_fraction
    crop_x1 = max(0, int(np.floor(x1 - margin_x)))
    crop_y1 = max(0, int(np.floor(y1 - margin_y)))
    crop_x2 = min(width, int(np.ceil(x2 + margin_x)))
    crop_y2 = min(height, int(np.ceil(y2 + margin_y)))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError("Detection produced an empty segmentation crop")
    local_box = (
        x1 - crop_x1,
        y1 - crop_y1,
        x2 - crop_x1,
        y2 - crop_y1,
    )
    return (
        image_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy(),
        local_box,
        (crop_x1, crop_y1, crop_x2, crop_y2),
    )


class BirdPipeline:
    def __init__(
        self,
        detector: BirdDetector,
        segmenter: Segmenter,
        output_dir: Path,
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.output_dir = output_dir

    def save_candidate(self, candidate: BirdCandidate) -> SavedBird:
        crop_bgr, crop_box, _ = expanded_crop(
            candidate.frame_bgr,
            candidate.detection.box,
        )
        mask, _ = self.segmenter.segment(crop_bgr, crop_box)
        timestamp = candidate.captured_at.astimezone(UTC)
        basename = timestamp.strftime("bird_%Y%m%dT%H%M%S_%fZ")
        output_path = self.output_dir / f"{basename}.png"
        image_path = write_image(output_path, crop_bgr, mask)
        return SavedBird(
            image_path=image_path,
            confidence=candidate.detection.confidence,
        )

    def process_image(self, image_path: Path) -> SavedBird | None:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image: {image_path}")
        detector_started = time.monotonic()
        detections = self.detector.detect(image)
        detector_seconds = time.monotonic() - detector_started
        if not detections:
            return None
        candidate = BirdCandidate(
            detection=detections[0],
            frame_bgr=image,
            frame_sequence=0,
            captured_at=datetime.now(UTC),
            detector_seconds=detector_seconds,
        )
        return self.save_candidate(candidate)

    def process_webcam(
        self,
        *,
        device: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        sample_fps: float = 2.0,
        window_seconds: float = 2.0,
        duration_seconds: float | None = None,
        max_outputs: int = 1,
    ) -> list[SavedBird]:
        if sample_fps <= 0 or window_seconds <= 0:
            raise ValueError("sample_fps and window_seconds must be positive")
        if max_outputs < 0:
            raise ValueError("max_outputs cannot be negative")

        results: list[SavedBird] = []
        best: BirdCandidate | None = None
        start = time.monotonic()
        window_start = start
        next_sample = start
        sequence = 0
        detector_samples = 0
        detector_seconds_total = 0.0

        with LatestFrameCamera(
            device,
            width=width,
            height=height,
            fps=fps,
        ) as camera:
            print(f"Camera settings: {camera.actual_settings()}")
            while True:
                now = time.monotonic()
                if duration_seconds is not None and now - start >= duration_seconds:
                    break
                if now < next_sample:
                    time.sleep(min(0.02, next_sample - now))
                    continue

                frame = camera.newest(after_sequence=sequence)
                sequence = frame.sequence
                detector_started = time.monotonic()
                detections = self.detector.detect(frame.image_bgr)
                detector_seconds = time.monotonic() - detector_started
                detector_samples += 1
                detector_seconds_total += detector_seconds
                if detections:
                    candidate = BirdCandidate(
                        detection=detections[0],
                        frame_bgr=frame.image_bgr.copy(),
                        frame_sequence=frame.sequence,
                        captured_at=frame.captured_at,
                        detector_seconds=detector_seconds,
                    )
                    best = better_candidate(best, candidate)
                    print(
                        f"Bird frame={frame.sequence} "
                        f"confidence={candidate.detection.confidence:.3f}"
                    )
                next_sample = time.monotonic() + (1.0 / sample_fps)

                if time.monotonic() - window_start >= window_seconds:
                    if best is not None:
                        try:
                            saved = self.save_candidate(best)
                            results.append(saved)
                            print(f"Saved {saved.image_path} confidence={saved.confidence:.3f}")
                        except ValueError as error:
                            print(f"Rejected candidate: {error}")
                    best = None
                    window_start = time.monotonic()
                    next_sample = window_start
                    if max_outputs and len(results) >= max_outputs:
                        break

        elapsed = time.monotonic() - start
        if detector_samples:
            print(
                f"Detector summary: samples={detector_samples} "
                f"average_seconds={detector_seconds_total / detector_samples:.3f} "
                f"effective_fps={detector_samples / elapsed:.3f}"
            )
        if best is not None and (not max_outputs or len(results) < max_outputs):
            try:
                results.append(self.save_candidate(best))
            except ValueError as error:
                print(f"Rejected final candidate: {error}")
        return results
