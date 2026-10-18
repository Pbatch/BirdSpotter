"""Continuously save the highest-confidence bird from each five-minute window."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import NoReturn

from loguru import logger

from birdspotter.capture import Capture, CapturedFrame, load_roi_config, write_roi_config
from birdspotter.classifier import BirdClassifier
from birdspotter.segmentation import Segmenter, sam3_segmenter
from birdspotter.sighting import (
    WINDOW_MINUTES,
    Sighting,
    window_dir,
    window_start,
    write_window,
)
from birdspotter.web.gallery import DEFAULT_GALLERY_HOST, start_gallery_server
from birdspotter.weights import classifier_path, default_weights_dir, sam3_openvino_dir

CLASSIFIER_FPS = 1.0
CAMERA_RETRY_SECONDS = 5.0
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

# The highest-confidence classifier-positive frame in the current window.
Winner = tuple[CapturedFrame, float]


def configure_logging(log_level: str) -> None:
    """Send structured, colour-free logs to systemd's journal."""

    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        colorize=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}",
    )


def save_bird_window(
    frame: CapturedFrame,
    confidence: float,
    segmenter: Segmenter,
    output_dir: Path,
) -> Sighting:
    """Segment one selected bird and save its window frame, mask, and label."""

    started = time.perf_counter()
    try:
        mask, sam_score = segmenter.segment(frame.image_bgr)
    except ValueError as error:
        logger.warning("Rejected selected bird | error={}", error)
        mask, sam_score = None, None
    if mask is not None and not mask.any():
        logger.warning("Rejected selected bird | error=SAM 3 returned an empty mask")
        mask = None
    saved = write_window(
        window_dir(output_dir, frame.captured_at),
        frame.uncropped_bgr,
        frame.roi,
        mask,
        confidence=confidence,
        sam_confidence=sam_score,
    )
    logger.info(
        "Saved window | frame={} confidence={:.3f} sam_score={} "
        "segmentation_seconds={:.3f} bird={} path={}",
        frame.sequence,
        confidence,
        sam_score,
        time.perf_counter() - started,
        mask is not None,
        saved.path,
    )
    return saved


def save_empty_window(frame: CapturedFrame, output_dir: Path) -> Sighting:
    """Save the full frame and label for a window that produced no bird."""

    saved = write_window(
        window_dir(output_dir, frame.captured_at),
        frame.uncropped_bgr,
        frame.roi,
        None,
        confidence=None,
        sam_confidence=None,
    )
    logger.info("Saved window without bird | path={}", saved.path)
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
        "--sam3-model",
        type=Path,
        default=sam3_openvino_dir(default_weights_dir()),
        help="SAM 3 export directory; one holding a TensorRT engine runs on TensorRT",
    )
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
    current: Winner | None,
) -> Winner | None:
    """Return the current window winner after inspecting one camera frame."""

    started = time.perf_counter()
    confidence = classifier.classify(frame.image_bgr)
    classifier_seconds = time.perf_counter() - started
    if confidence is None:
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
        confidence,
    )
    if current is not None and confidence <= current[1]:
        logger.debug(
            "Retained window winner | frame={} confidence={:.3f} candidate_confidence={:.3f}",
            current[0].sequence,
            current[1],
            confidence,
        )
        return current
    logger.info("Window best | frame={} confidence={:.3f}", frame.sequence, confidence)
    return frame, confidence


def run_classification_loop(
    camera: Capture,
    classifier: BirdClassifier,
    segmenter: Segmenter,
    output_dir: Path,
) -> NoReturn:
    """Run classifier windows forever, saving each completed window."""

    next_classification = time.monotonic()
    sequence = 0
    best: Winner | None = None
    latest: CapturedFrame | None = None
    roi_revision = 0
    while True:
        now = time.monotonic()
        if now < next_classification:
            time.sleep(min(0.02, next_classification - now))
            continue

        frame = next_frame(camera, sequence)
        sequence = frame.sequence
        # Windows are aligned to the UTC clock; the first frame of a new window
        # closes the previous one, whose frames all share its directory name.
        if latest is not None and window_start(frame.captured_at) != window_start(
            latest.captured_at
        ):
            if best is None:
                logger.info("Selection window complete | no bird passed the confidence threshold")
                save_empty_window(latest, output_dir)
            else:
                best_frame, best_confidence = best
                logger.info(
                    "Selection window complete | winner_frame={} winner_confidence={:.3f}",
                    best_frame.sequence,
                    best_confidence,
                )
                save_bird_window(best_frame, best_confidence, segmenter, output_dir)
            best = None
        latest = frame
        if frame.roi_revision != roi_revision:
            logger.info("Classification ROI changed | discarding current window candidate")
            best = None
            roi_revision = frame.roi_revision
        best = update_window_winner(classifier, frame, best)
        # Advance from the prior target time rather than from inference completion.
        # This preserves the requested cadence when inference is fast and naturally
        # skips the wait when an inference call overruns its one-second budget.
        next_classification += 1 / CLASSIFIER_FPS


def next_frame(camera: Capture, after_sequence: int) -> CapturedFrame:
    """Return the next camera frame, reopening the stream whenever it stalls or fails."""

    while True:
        try:
            return camera.newest(after_sequence=after_sequence)
        except (TimeoutError, RuntimeError) as error:
            logger.warning("Camera stalled | {} | reconnecting", error.__cause__ or error)
        try:
            camera.reconnect()
        except RuntimeError as error:
            logger.warning(
                "Camera reconnect failed | {} | retrying in {}s", error, CAMERA_RETRY_SECONDS
            )
            time.sleep(CAMERA_RETRY_SECONDS)


def main() -> None:
    """Run the deployment loop until interrupted."""

    args = parse_arguments()
    configure_logging(args.log_level)
    weights_dir = default_weights_dir()
    classifier = BirdClassifier(classifier_path(weights_dir))
    segmenter = sam3_segmenter(args.sam3_model)
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

    with (
        Capture(
            source,
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
            roi=roi,
        ) as camera,
        start_gallery_server(
            output_dir,
            args.web_host,
            args.web_port,
            icon_path=Path(str(files("birdspotter.web").joinpath("static", "icon.png"))),
            camera=camera,
            roi_config_path=roi_config_path,
        ) as gallery_server,
    ):
        logger.info(
            "Gallery server | address=http://{}:{} latest=10",
            args.web_host,
            gallery_server.server_port,
        )
        logger.info("Camera settings | {}", camera.actual_settings())
        run_classification_loop(camera, classifier, segmenter, output_dir)


if __name__ == "__main__":
    main()
