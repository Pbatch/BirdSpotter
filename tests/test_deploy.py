from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from birdspotter.capture import Capture
from birdspotter.gallery import DEFAULT_GALLERY_HOST
from birdspotter.types import BirdCandidate, Classification
from scripts import deploy
from scripts.deploy import output_path, parse_arguments, rounded_to_five_minutes


def test_rounded_to_five_minutes_rounds_half_up_across_an_hour() -> None:
    timestamp = datetime(2026, 8, 2, 12, 58, 30, tzinfo=UTC)

    assert rounded_to_five_minutes(timestamp) == datetime(2026, 8, 2, 13, 0, tzinfo=UTC)


def test_output_path_contains_percentage_and_rounded_time(tmp_path: Path) -> None:
    candidate = BirdCandidate(
        classification=Classification(confidence=0.8234),
        frame_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
        frame_sequence=1,
        captured_at=datetime(2026, 8, 2, 12, 2, 30, tzinfo=UTC),
    )

    assert output_path(tmp_path, candidate) == tmp_path / "bird_conf_82_ts_2026-08-02_12-05.png"


def test_rtsp_url_selects_a_network_camera() -> None:
    args = parse_arguments(["--rtsp-url", "rtsp://camera.example/stream1"])

    assert args.rtsp_url == "rtsp://camera.example/stream1"
    assert args.device == 0
    assert args.web_host == DEFAULT_GALLERY_HOST
    assert args.web_port == 8080


def test_roi_accepts_four_pixel_coordinates() -> None:
    args = parse_arguments(["--roi", "320", "180", "1600", "900"])

    assert args.roi == [320, 180, 1600, 900]


def test_rtsp_source_name_redacts_camera_credentials() -> None:
    camera = Capture("rtsp://birdspotter:password@192.168.1.42:554/stream1")

    assert camera.source_name() == "rtsp://192.168.1.42:554/stream1"


def test_save_candidate_segments_whole_frame_and_outlines_sam_bird(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    mask = np.zeros((20, 40), dtype=bool)
    mask[5:15, 25:35] = True
    candidate = BirdCandidate(
        classification=Classification(confidence=0.9),
        frame_bgr=frame,
        frame_sequence=1,
        captured_at=datetime(2026, 10, 1, tzinfo=UTC),
    )

    class Segmenter:
        @staticmethod
        def segment(image: np.ndarray) -> tuple[np.ndarray, float]:
            assert image is frame
            return mask, 0.95

    def record_image(path: Path, image: np.ndarray, bird_mask: np.ndarray) -> Path:
        assert image is frame
        assert bird_mask is mask
        return path

    def record_gallery(
        path: Path,
        image: np.ndarray,
        bird_mask: np.ndarray,
        origin: tuple[int, int],
        box: tuple[float, float, float, float],
    ) -> Path:
        assert image is frame
        assert bird_mask is mask
        assert origin == (0, 0)
        assert box == (23, 3, 37, 17)
        return path

    monkeypatch.setattr(deploy, "write_image", record_image)
    monkeypatch.setattr(deploy, "write_gallery_frame", record_gallery)
    deploy.save_candidate(candidate, Segmenter(), tmp_path)  # ty: ignore[invalid-argument-type]


def test_no_candidate_does_not_run_sam3(tmp_path: Path) -> None:
    class Segmenter:
        @staticmethod
        def segment(_image: np.ndarray) -> tuple[np.ndarray, float]:
            pytest.fail("SAM 3 must not run without a classifier-positive candidate")

    deploy.save_best_candidate(None, Segmenter(), tmp_path)  # ty: ignore[invalid-argument-type]
