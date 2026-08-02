"""Continuously save the highest-confidence bird from each five-minute window."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from birdspotter.capture import CapturedFrame, LatestFrameCamera
from birdspotter.crop import expanded_crop
from birdspotter.detection import BirdDetector
from birdspotter.models import default_weights_dir, detector_path, sam21_openvino_dir
from birdspotter.output import write_image
from birdspotter.sam21_openvino import Sam21OpenVinoSegmenter
from birdspotter.types import BirdCandidate

DETECTOR_FPS = 1.0
WINDOW_MINUTES = 5


def rounded_to_five_minutes(timestamp: datetime) -> datetime:
    """Round a UTC timestamp to its nearest five-minute boundary."""

    timestamp = timestamp.astimezone(UTC)
    elapsed = timedelta(
        minutes=timestamp.minute,
        seconds=timestamp.second,
        microseconds=timestamp.microsecond,
    )
    interval = timedelta(minutes=WINDOW_MINUTES)
    rounded_intervals = (elapsed + interval / 2) // interval
    return timestamp.replace(minute=0, second=0, microsecond=0) + rounded_intervals * interval


def output_path(output_dir: Path, candidate: BirdCandidate) -> Path:
    """Return the required confidence-and-time based final PNG path."""

    timestamp = rounded_to_five_minutes(candidate.captured_at)
    confidence_percent = round(candidate.detection.confidence * 100)
    return output_dir / f"bird_conf_{confidence_percent}_ts_{timestamp:%Y-%m-%d_%H-%M}.png"


def save_candidate(
    candidate: BirdCandidate,
    segmenter: Sam21OpenVinoSegmenter,
    output_dir: Path,
) -> Path:
    """Segment and save one selected bird as a transparent PNG."""

    crop, crop_box, _ = expanded_crop(candidate.frame_bgr, candidate.detection.box)
    mask, _ = segmenter.segment(crop, crop_box)
    return write_image(output_path(output_dir, candidate), crop, mask)


def parse_arguments() -> argparse.Namespace:
    """Parse deployment configuration from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=896)
    parser.add_argument("--camera-fps", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("segmented"))
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=None,
        help="Stop after this duration (default: run continuously)",
    )
    args = parser.parse_args()
    if args.duration_minutes is not None and args.duration_minutes <= 0:
        raise ValueError("--duration-minutes must be positive")
    return args


def update_window_winner(
    detector: BirdDetector,
    frame: CapturedFrame,
    current: BirdCandidate | None,
) -> BirdCandidate | None:
    """Return the current window winner after inspecting one camera frame."""

    detections = detector.detect(frame.image_bgr)
    if not detections:
        return current
    candidate = BirdCandidate(
        detection=detections[0],
        frame_bgr=frame.image_bgr.copy(),
        frame_sequence=frame.sequence,
        captured_at=frame.captured_at,
        detector_seconds=0.0,
    )
    if current is not None and candidate.detection.confidence <= current.detection.confidence:
        return current
    print(
        f"Window best: confidence={candidate.detection.confidence:.3f} "
        f"frame={candidate.frame_sequence}"
    )
    return candidate


def save_best_candidate(
    candidate: BirdCandidate | None,
    segmenter: Sam21OpenVinoSegmenter,
    output_dir: Path,
    *,
    partial_window: bool = False,
) -> None:
    """Segment and save a window winner when one exists."""

    if candidate is None:
        return
    try:
        saved = save_candidate(candidate, segmenter, output_dir)
    except ValueError as error:
        print(f"Rejected selected bird: {error}")
    else:
        label = "final partial-window bird" if partial_window else "bird"
        print(f"Saved {label}: {saved}")


def run_detection_loop(
    camera: LatestFrameCamera,
    detector: BirdDetector,
    segmenter: Sam21OpenVinoSegmenter,
    output_dir: Path,
    duration_minutes: float | None,
) -> BirdCandidate | None:
    """Run detector windows and return any unsaved partial-window winner."""

    started = time.monotonic()
    window_started = started
    next_detection = started
    sequence = 0
    best: BirdCandidate | None = None
    while duration_minutes is None or time.monotonic() - started < duration_minutes * 60:
        now = time.monotonic()
        if now < next_detection:
            time.sleep(min(0.02, next_detection - now))
            continue

        frame = camera.newest(after_sequence=sequence)
        sequence = frame.sequence
        best = update_window_winner(detector, frame, best)
        next_detection = time.monotonic() + 1 / DETECTOR_FPS
        if time.monotonic() - window_started >= WINDOW_MINUTES * 60:
            save_best_candidate(best, segmenter, output_dir)
            best = None
            window_started = time.monotonic()
            next_detection = window_started
    return best


def main() -> None:
    """Run the deployment loop until interrupted or an optional duration elapses."""

    args = parse_arguments()
    weights_dir = default_weights_dir()
    detector = BirdDetector(detector_path(weights_dir))
    segmenter = Sam21OpenVinoSegmenter(sam21_openvino_dir(weights_dir))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with LatestFrameCamera(
        args.device,
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
    ) as camera:
        print(f"Camera settings: {camera.actual_settings()}")
        best = run_detection_loop(camera, detector, segmenter, output_dir, args.duration_minutes)
    save_best_candidate(best, segmenter, output_dir, partial_window=True)


if __name__ == "__main__":
    main()
