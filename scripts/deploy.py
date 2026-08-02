"""Continuously save the highest-confidence bird from each five-minute window."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger

from birdspotter.capture import CapturedFrame, LatestFrameCamera
from birdspotter.crop import expanded_crop
from birdspotter.detection import BirdDetector
from birdspotter.models import default_weights_dir, detector_path, sam21_openvino_dir
from birdspotter.output import write_image
from birdspotter.sam21_openvino import Sam21OpenVinoSegmenter
from birdspotter.types import BirdCandidate

DETECTOR_FPS = 1.0
WINDOW_MINUTES = 5
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def configure_logging(log_level: str) -> None:
    """Send structured, colour-free logs to systemd's journal."""

    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        colorize=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}",
    )


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

    started = time.perf_counter()
    crop, crop_box, _ = expanded_crop(candidate.frame_bgr, candidate.detection.box)
    mask, sam_score = segmenter.segment(crop, crop_box)
    saved = write_image(output_path(output_dir, candidate), crop, mask)
    logger.info(
        "Saved bird | frame={} confidence={:.3f} detector_seconds={:.3f} "
        "sam_score={:.3f} segmentation_seconds={:.3f} path={}",
        candidate.frame_sequence,
        candidate.detection.confidence,
        candidate.detector_seconds,
        sam_score,
        time.perf_counter() - started,
        saved,
    )
    return saved


def parse_arguments() -> argparse.Namespace:
    """Parse deployment configuration from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=896)
    parser.add_argument("--camera-fps", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("segmented"))
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        type=str.upper,
        help="Journal verbosity (default: %(default)s)",
    )
    return parser.parse_args()


def update_window_winner(
    detector: BirdDetector,
    frame: CapturedFrame,
    current: BirdCandidate | None,
) -> BirdCandidate | None:
    """Return the current window winner after inspecting one camera frame."""

    started = time.perf_counter()
    detections = detector.detect(frame.image_bgr)
    detector_seconds = time.perf_counter() - started
    if not detections:
        logger.debug(
            "Detector | frame={} seconds={:.3f} detections=0",
            frame.sequence,
            detector_seconds,
        )
        return current
    logger.debug(
        "Detector | frame={} seconds={:.3f} detections={} top_confidence={:.3f}",
        frame.sequence,
        detector_seconds,
        len(detections),
        detections[0].confidence,
    )
    candidate = BirdCandidate(
        detection=detections[0],
        frame_bgr=frame.image_bgr.copy(),
        frame_sequence=frame.sequence,
        captured_at=frame.captured_at,
        detector_seconds=detector_seconds,
    )
    if current is not None and candidate.detection.confidence <= current.detection.confidence:
        logger.debug(
            "Retained window winner | frame={} confidence={:.3f} candidate_confidence={:.3f}",
            current.frame_sequence,
            current.detection.confidence,
            candidate.detection.confidence,
        )
        return current
    logger.info(
        "Window best | frame={} confidence={:.3f} box={}",
        candidate.frame_sequence,
        candidate.detection.confidence,
        tuple(round(value, 1) for value in candidate.detection.box),
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
        logger.warning("Rejected selected bird | error={}", error)
    else:
        label = "final partial-window bird" if partial_window else "bird"
        logger.info("Completed {} | path={}", label, saved)


def run_detection_loop(
    camera: LatestFrameCamera,
    detector: BirdDetector,
    segmenter: Sam21OpenVinoSegmenter,
    output_dir: Path,
) -> BirdCandidate | None:
    """Run detector windows and return any unsaved partial-window winner."""

    started = time.monotonic()
    window_started = started
    next_detection = started
    sequence = 0
    best: BirdCandidate | None = None
    while True:
        now = time.monotonic()
        if now < next_detection:
            time.sleep(min(0.02, next_detection - now))
            continue

        frame = camera.newest(after_sequence=sequence)
        sequence = frame.sequence
        best = update_window_winner(detector, frame, best)
        # Advance from the prior target time rather than from inference completion.
        # This preserves the requested cadence when inference is fast and naturally
        # skips the wait when an inference call overruns its one-second budget.
        next_detection += 1 / DETECTOR_FPS
        if time.monotonic() - window_started >= WINDOW_MINUTES * 60:
            if best is None:
                logger.info("Selection window complete | no bird passed the confidence threshold")
            else:
                logger.info(
                    "Selection window complete | winner_frame={} winner_confidence={:.3f}",
                    best.frame_sequence,
                    best.detection.confidence,
                )
            save_best_candidate(best, segmenter, output_dir)
            best = None
            window_started = time.monotonic()
            next_detection = window_started
    return best


def main() -> None:
    """Run the deployment loop until interrupted."""

    args = parse_arguments()
    configure_logging(args.log_level)
    weights_dir = default_weights_dir()
    detector = BirdDetector(detector_path(weights_dir))
    segmenter = Sam21OpenVinoSegmenter(sam21_openvino_dir(weights_dir))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting BirdSpotter | device={} camera_request={}x{}@{}fps detector_fps={} "
        "window_minutes={} output_dir={}",
        args.device,
        args.width,
        args.height,
        args.camera_fps,
        DETECTOR_FPS,
        WINDOW_MINUTES,
        output_dir,
    )
    logger.debug("Detector configuration | {}", detector.describe())
    logger.debug("Segmenter configuration | {}", segmenter.describe())

    with LatestFrameCamera(
        args.device,
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
    ) as camera:
        logger.info("Camera settings | {}", camera.actual_settings())
        best = run_detection_loop(camera, detector, segmenter, output_dir)
    save_best_candidate(best, segmenter, output_dir, partial_window=True)


if __name__ == "__main__":
    main()
