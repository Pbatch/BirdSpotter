"""Continuously save the highest-confidence bird from each five-minute window."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from pathlib import Path

from loguru import logger

from birdspotter.capture import Capture, CapturedFrame
from birdspotter.classification import BirdClassifier
from birdspotter.gallery import (
    DEFAULT_GALLERY_HOST,
    load_roi_config,
    start_gallery_server,
    write_roi_config,
)
from birdspotter.models import classifier_path, default_weights_dir, sam3_openvino_dir
from birdspotter.output import mask_bounds, write_gallery_frame, write_image
from birdspotter.sam3_openvino import Sam3OpenVinoSegmenter
from birdspotter.types import BirdCandidate

CLASSIFIER_FPS = 1.0
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
    confidence_percent = round(candidate.classification.confidence * 100)
    return output_dir / f"bird_conf_{confidence_percent}_ts_{timestamp:%Y-%m-%d_%H-%M}.png"


def save_candidate(
    candidate: BirdCandidate,
    segmenter: Sam3OpenVinoSegmenter,
    output_dir: Path,
) -> Path:
    """Segment and save one selected bird as a transparent PNG."""

    started = time.perf_counter()
    mask, sam_score = segmenter.segment(candidate.frame_bgr)
    saved = write_image(output_path(output_dir, candidate), candidate.frame_bgr, mask)
    write_gallery_frame(
        output_dir / "gallery" / saved.name,
        candidate.frame_bgr,
        mask,
        (0, 0),
        mask_bounds(mask),
    )
    logger.info(
        "Saved bird | frame={} confidence={:.3f} classifier_seconds={:.3f} "
        "sam_score={:.3f} segmentation_seconds={:.3f} path={}",
        candidate.frame_sequence,
        candidate.classification.confidence,
        candidate.classifier_seconds,
        sam_score,
        time.perf_counter() - started,
        saved,
    )
    return saved


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse deployment configuration from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--device", type=int, default=0, help="Local V4L2 camera number")
    source.add_argument(
        "--rtsp-url",
        default=os.environ.get("BIRDSPOTTER_RTSP_URL"),
        metavar="URL",
        help="RTSP source URL (defaults to the BIRDSPOTTER_RTSP_URL environment variable)",
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=896)
    parser.add_argument("--camera-fps", type=int, default=5)
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Crop every decoded camera frame to this pixel ROI before classification",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("segmented"))
    parser.add_argument(
        "--web-host", default=DEFAULT_GALLERY_HOST, help="Gallery server bind address"
    )
    parser.add_argument("--web-port", type=int, default=8080, help="Gallery server port")
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        type=str.upper,
        help="Journal verbosity (default: %(default)s)",
    )
    return parser.parse_args(argv)


def update_window_winner(
    classifier: BirdClassifier,
    frame: CapturedFrame,
    current: BirdCandidate | None,
) -> BirdCandidate | None:
    """Return the current window winner after inspecting one camera frame."""

    started = time.perf_counter()
    classification = classifier.classify(frame.image_bgr)
    classifier_seconds = time.perf_counter() - started
    if classification is None:
        logger.debug(
            "Classifier | frame={} seconds={:.3f} bird_present=False",
            frame.sequence,
            classifier_seconds,
        )
        return current
    logger.debug(
        "Classifier | frame={} seconds={:.3f} bird_present=True confidence={:.3f}",
        frame.sequence,
        classifier_seconds,
        classification.confidence,
    )
    candidate = BirdCandidate(
        classification=classification,
        frame_bgr=frame.image_bgr.copy(),
        frame_sequence=frame.sequence,
        captured_at=frame.captured_at,
        classifier_seconds=classifier_seconds,
    )
    if (
        current is not None
        and candidate.classification.confidence <= current.classification.confidence
    ):
        logger.debug(
            "Retained window winner | frame={} confidence={:.3f} candidate_confidence={:.3f}",
            current.frame_sequence,
            current.classification.confidence,
            candidate.classification.confidence,
        )
        return current
    logger.info(
        "Window best | frame={} confidence={:.3f}",
        candidate.frame_sequence,
        candidate.classification.confidence,
    )
    return candidate


def save_best_candidate(
    candidate: BirdCandidate | None,
    segmenter: Sam3OpenVinoSegmenter,
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


def run_classification_loop(
    camera: Capture,
    classifier: BirdClassifier,
    segmenter: Sam3OpenVinoSegmenter,
    output_dir: Path,
) -> BirdCandidate | None:
    """Run classifier windows and return any unsaved partial-window winner."""

    started = time.monotonic()
    window_started = started
    next_classification = started
    sequence = 0
    best: BirdCandidate | None = None
    roi_revision = 0
    while True:
        now = time.monotonic()
        if now < next_classification:
            time.sleep(min(0.02, next_classification - now))
            continue

        frame = camera.newest(after_sequence=sequence)
        sequence = frame.sequence
        if frame.roi_revision != roi_revision:
            logger.info("Classification ROI changed | discarding current window candidate")
            best = None
            roi_revision = frame.roi_revision
        best = update_window_winner(classifier, frame, best)
        # Advance from the prior target time rather than from inference completion.
        # This preserves the requested cadence when inference is fast and naturally
        # skips the wait when an inference call overruns its one-second budget.
        next_classification += 1 / CLASSIFIER_FPS
        if time.monotonic() - window_started >= WINDOW_MINUTES * 60:
            if best is None:
                logger.info("Selection window complete | no bird passed the confidence threshold")
            else:
                logger.info(
                    "Selection window complete | winner_frame={} winner_confidence={:.3f}",
                    best.frame_sequence,
                    best.classification.confidence,
                )
            save_best_candidate(best, segmenter, output_dir)
            best = None
            window_started = time.monotonic()
            next_classification = window_started
    return best


def main() -> None:
    """Run the deployment loop until interrupted."""

    args = parse_arguments()
    configure_logging(args.log_level)
    weights_dir = default_weights_dir()
    classifier = BirdClassifier(classifier_path(weights_dir))
    segmenter = Sam3OpenVinoSegmenter(sam3_openvino_dir(weights_dir))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_config_path = output_dir / "roi.json"
    roi = tuple(args.roi) if args.roi is not None else load_roi_config(roi_config_path)
    if args.roi is not None:
        write_roi_config(roi_config_path, roi)
    source = args.rtsp_url if args.rtsp_url is not None else args.device
    camera_source = Capture(source).source_name()
    logger.info(
        "Starting BirdSpotter | source={} camera_request={}x{}@{}fps classifier_fps={} "
        "window_minutes={} output_dir={}",
        camera_source,
        args.width,
        args.height,
        args.camera_fps,
        CLASSIFIER_FPS,
        WINDOW_MINUTES,
        output_dir,
    )
    logger.debug("Classifier configuration | {}", classifier.describe())
    logger.debug("Segmenter configuration | {}", segmenter.describe())

    with Capture(
        source,
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
        roi=roi,
    ) as camera:
        gallery_server = start_gallery_server(
            output_dir,
            args.web_host,
            args.web_port,
            icon_path=Path(str(files("birdspotter").joinpath("static", "icon.png"))),
            camera=camera,
            roi_config_path=roi_config_path,
        )
        logger.info(
            "Gallery server | address=http://{}:{} latest=10",
            args.web_host,
            gallery_server.server_port,
        )
        try:
            logger.info("Camera settings | {}", camera.actual_settings())
            best = run_classification_loop(camera, classifier, segmenter, output_dir)
            save_best_candidate(best, segmenter, output_dir, partial_window=True)
        finally:
            gallery_server.shutdown()
            gallery_server.server_close()


if __name__ == "__main__":
    main()
