"""Continuously save the bird from each five-minute window's most-changed frame."""

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
from birdspotter.motion import MotionGate
from birdspotter.segmentation import Segmenter, sam3_segmenter
from birdspotter.sighting import (
    WINDOW_MINUTES,
    Sighting,
    window_dir,
    window_start,
    write_window,
)
from birdspotter.web.gallery import DEFAULT_GALLERY_HOST, start_gallery_server
from birdspotter.weights import default_weights_dir, sam3_openvino_dir

GATE_FPS = 1.0
CAMERA_RETRY_SECONDS = 5.0
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

# The highest-scoring frame over the threshold in the current window, and its motion score.
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
    motion: float,
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
        motion=motion,
        sam_confidence=sam_score,
    )
    logger.info(
        "Saved window | frame={} motion={:.1f} sam_score={} "
        "segmentation_seconds={:.3f} bird={} path={}",
        frame.sequence,
        motion,
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
        motion=None,
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
        help="Crop every decoded camera frame to this pixel ROI before the motion gate",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("segmented"))
    parser.add_argument(
        "--sam3-model",
        type=Path,
        default=sam3_openvino_dir(default_weights_dir()),
        help="SAM 3 export directory; one holding a TensorRT engine runs on TensorRT",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=70.0,
        help="Grey-level difference from the background median that sends a frame to SAM 3 "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--background",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Seconds of past frames in the background median; gating starts once they are "
        "collected after start-up or an ROI change (default: %(default)s)",
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
    gate: MotionGate,
    frame: CapturedFrame,
    current: Winner | None,
) -> Winner | None:
    """Return the current window winner after inspecting one camera frame."""

    started = time.perf_counter()
    motion = gate.score(frame.image_bgr)
    gate_seconds = time.perf_counter() - started
    moved = motion >= gate.threshold
    logger.debug(
        "Motion | frame={} seconds={:.3f} motion={:.1f} moved={} ready={}",
        frame.sequence,
        gate_seconds,
        motion,
        moved,
        gate.ready,
    )
    if not moved:
        return current
    if current is not None and motion <= current[1]:
        return current
    logger.info("Window best | frame={} motion={:.1f}", frame.sequence, motion)
    return frame, motion


def run_detection_loop(
    camera: Capture,
    gate: MotionGate,
    segmenter: Segmenter,
    output_dir: Path,
) -> NoReturn:
    """Run motion-gated windows forever, saving each completed window."""

    next_check = time.monotonic()
    sequence = 0
    best: Winner | None = None
    latest: CapturedFrame | None = None
    roi_revision = 0
    while True:
        now = time.monotonic()
        if now < next_check:
            time.sleep(min(0.02, next_check - now))
            continue

        frame = next_frame(camera, sequence)
        sequence = frame.sequence
        # Windows are aligned to the UTC clock; the first frame of a new window
        # closes the previous one, whose frames all share its directory name.
        if latest is not None and window_start(frame.captured_at) != window_start(
            latest.captured_at
        ):
            if best is None:
                logger.info("Selection window complete | no frame passed the motion threshold")
                save_empty_window(latest, output_dir)
            else:
                best_frame, best_motion = best
                logger.info(
                    "Selection window complete | winner_frame={} winner_motion={:.1f}",
                    best_frame.sequence,
                    best_motion,
                )
                save_bird_window(best_frame, best_motion, segmenter, output_dir)
            best = None
        latest = frame
        if frame.roi_revision != roi_revision:
            logger.info("ROI changed | discarding window candidate and background")
            best = None
            gate.reset()
            roi_revision = frame.roi_revision
        best = update_window_winner(gate, frame, best)
        # Advance from the prior target time rather than from inference completion.
        # This preserves the requested cadence when inference is fast and naturally
        # skips the wait when an inference call overruns its one-second budget.
        next_check += 1 / GATE_FPS


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
    gate = MotionGate(
        threshold=args.motion_threshold,
        history_frames=max(2, round(args.background * GATE_FPS)),
    )
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
        "Starting BirdSpotter | source={} camera_request={}x{}@{}fps gate_fps={} "
        "window_minutes={} output_dir={}",
        camera_source,
        args.width,
        args.height,
        args.camera_fps,
        GATE_FPS,
        WINDOW_MINUTES,
        output_dir,
    )
    logger.debug("Motion gate configuration | {}", gate.describe())
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
        run_detection_loop(camera, gate, segmenter, output_dir)


if __name__ == "__main__":
    main()
