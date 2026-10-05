import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from birdspotter.capture import Capture, CapturedFrame
from birdspotter.gallery import DEFAULT_GALLERY_HOST
from birdspotter.types import BirdCandidate, Classification
from scripts import deploy
from scripts.deploy import parse_arguments, window_dir, window_start


def test_window_start_floors_to_the_clock_aligned_five_minutes() -> None:
    timestamp = datetime(2026, 8, 2, 12, 59, 59, tzinfo=UTC)

    assert window_start(timestamp) == datetime(2026, 8, 2, 12, 55, tzinfo=UTC)


def test_window_dir_is_named_after_the_window_start(tmp_path: Path) -> None:
    assert window_dir(tmp_path, datetime(2026, 8, 2, 12, 4, 59, tzinfo=UTC)) == (
        tmp_path / "2026-08-02_12-00"
    )
    assert window_dir(tmp_path, datetime(2026, 8, 2, 12, 5, tzinfo=UTC)) == (
        tmp_path / "2026-08-02_12-05"
    )


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


def bird_candidate(frame: np.ndarray) -> BirdCandidate:
    return BirdCandidate(
        classification=Classification(confidence=0.9),
        frame_bgr=frame,
        frame_sequence=1,
        captured_at=datetime(2026, 10, 1, tzinfo=UTC),
    )


def test_save_candidate_writes_frame_mask_and_label(tmp_path: Path) -> None:
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    mask = np.zeros((20, 40), dtype=bool)
    mask[5:15, 25:35] = True

    class Segmenter:
        @staticmethod
        def segment(image: np.ndarray) -> tuple[np.ndarray, float]:
            assert image is frame
            return mask, 0.95

    label_path = deploy.save_candidate(bird_candidate(frame), Segmenter(), tmp_path)  # ty: ignore[invalid-argument-type]

    window = tmp_path / "2026-10-01_00-00"
    assert label_path == window / "label.json"
    assert sorted(path.name for path in window.iterdir()) == [
        "full_frame.jpg",
        "label.json",
        "mask.png",
    ]
    assert json.loads(label_path.read_text()) == {
        "width": 40,
        "height": 20,
        "roi": None,
        "bbox": [23, 3, 37, 17],
        "confidence": 0.9,
        "sam_confidence": 0.95,
    }


def test_no_candidate_does_not_run_sam3(tmp_path: Path) -> None:
    class Segmenter:
        @staticmethod
        def segment(_image: np.ndarray) -> tuple[np.ndarray, float]:
            pytest.fail("SAM 3 must not run without a classifier-positive candidate")

    deploy.save_best_candidate(None, Segmenter(), tmp_path)  # ty: ignore[invalid-argument-type]


def test_empty_sam_mask_saves_frame_and_label_without_mask(tmp_path: Path) -> None:
    frame = np.full((20, 40, 3), 90, dtype=np.uint8)

    class Segmenter:
        @staticmethod
        def segment(_image: np.ndarray) -> tuple[np.ndarray, float]:
            return np.zeros((20, 40), dtype=bool), 0.1

    deploy.save_best_candidate(bird_candidate(frame), Segmenter(), tmp_path)  # ty: ignore[invalid-argument-type]

    window = tmp_path / "2026-10-01_00-00"
    assert (window / "full_frame.jpg").is_file()
    assert not (window / "mask.png").exists()
    label = json.loads((window / "label.json").read_text())
    assert label["bbox"] is None
    assert label["confidence"] == 0.9
    assert label["sam_confidence"] == 0.1


def test_empty_window_saves_frame_and_null_label(tmp_path: Path) -> None:
    frame = CapturedFrame(1, datetime(2026, 10, 1, 0, 1, tzinfo=UTC), np.zeros((4, 6, 3), np.uint8))

    label_path = deploy.save_empty_window(frame, tmp_path)

    assert (tmp_path / "2026-10-01_00-00" / "full_frame.jpg").is_file()
    assert not (tmp_path / "2026-10-01_00-00" / "mask.png").exists()
    assert json.loads(label_path.read_text()) == {
        "width": 6,
        "height": 4,
        "roi": None,
        "bbox": None,
        "confidence": None,
        "sam_confidence": None,
    }


def test_loop_saves_one_directory_per_clock_aligned_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    times = [
        datetime(2026, 10, 1, 12, 1, tzinfo=UTC),
        datetime(2026, 10, 1, 12, 4, 59, tzinfo=UTC),
        datetime(2026, 10, 1, 12, 6, tzinfo=UTC),
        datetime(2026, 10, 1, 12, 11, tzinfo=UTC),
    ]

    class Camera:
        @staticmethod
        def newest(*, after_sequence: int) -> CapturedFrame:
            if after_sequence == len(times):
                raise KeyboardInterrupt
            return CapturedFrame(
                after_sequence + 1, times[after_sequence], np.zeros((4, 6, 3), np.uint8)
            )

    monkeypatch.setattr(deploy, "CLASSIFIER_FPS", 1000)
    monkeypatch.setattr(deploy, "update_window_winner", lambda _classifier, _frame, best: best)
    with pytest.raises(KeyboardInterrupt):
        deploy.run_classification_loop(Camera(), None, None, tmp_path)  # ty: ignore[invalid-argument-type]

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "2026-10-01_12-00",
        "2026-10-01_12-05",
    ]


def test_candidate_saves_uncropped_source_frame_and_roi(tmp_path: Path) -> None:
    source = np.zeros((40, 60, 3), dtype=np.uint8)
    crop = source[12:32, 10:40].copy()
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 8:18] = True
    candidate = bird_candidate(crop)
    candidate.source_bgr = source
    candidate.roi = (10, 12, 40, 32)

    class Segmenter:
        @staticmethod
        def segment(image: np.ndarray) -> tuple[np.ndarray, float]:
            assert image is crop
            return mask, 0.95

    label_path = deploy.save_candidate(candidate, Segmenter(), tmp_path)  # ty: ignore[invalid-argument-type]

    label = json.loads(label_path.read_text())
    assert (label["width"], label["height"]) == (60, 40)
    assert label["roi"] == [10, 12, 40, 32]
    assert label["bbox"] == [16, 15, 30, 29]
