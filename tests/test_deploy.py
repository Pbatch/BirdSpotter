import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from birdspotter.capture import Capture, CapturedFrame
from birdspotter.web.gallery import DEFAULT_GALLERY_HOST
from scripts import deploy
from scripts.deploy import parse_arguments


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


def bird_frame(image: np.ndarray) -> CapturedFrame:
    return CapturedFrame(1, datetime(2026, 10, 1, tzinfo=UTC), image)


def test_save_bird_window_writes_frame_mask_and_label(tmp_path: Path) -> None:
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    mask = np.zeros((20, 40), dtype=bool)
    mask[5:15, 25:35] = True

    class Segmenter:
        @staticmethod
        def segment(image: np.ndarray) -> tuple[np.ndarray, float]:
            assert image is frame
            return mask, 0.95

    sighting = deploy.save_bird_window(bird_frame(frame), 0.9, Segmenter(), tmp_path)  # ty: ignore[invalid-argument-type]

    window = tmp_path / "2026-10-01_00-00"
    assert sighting.path == window
    label_path = window / "label.json"
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


def test_empty_sam_mask_saves_frame_and_label_without_mask(tmp_path: Path) -> None:
    frame = np.full((20, 40, 3), 90, dtype=np.uint8)

    class Segmenter:
        @staticmethod
        def segment(_image: np.ndarray) -> tuple[np.ndarray, float]:
            return np.zeros((20, 40), dtype=bool), 0.1

    deploy.save_bird_window(bird_frame(frame), 0.9, Segmenter(), tmp_path)  # ty: ignore[invalid-argument-type]

    window = tmp_path / "2026-10-01_00-00"
    assert (window / "full_frame.jpg").is_file()
    assert not (window / "mask.png").exists()
    label = json.loads((window / "label.json").read_text())
    assert label["bbox"] is None
    assert label["confidence"] == 0.9
    assert label["sam_confidence"] == 0.1


def test_empty_window_saves_frame_and_null_label(tmp_path: Path) -> None:
    frame = CapturedFrame(1, datetime(2026, 10, 1, 0, 1, tzinfo=UTC), np.zeros((4, 6, 3), np.uint8))

    label_path = deploy.save_empty_window(frame, tmp_path).path / "label.json"

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


def test_next_frame_reconnects_after_a_camera_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = CapturedFrame(4, datetime(2026, 10, 1, tzinfo=UTC), np.zeros((4, 6, 3), np.uint8))
    failures = [
        TimeoutError("Timed out waiting for a camera frame"),
        RuntimeError("Camera reader failed"),
    ]
    events: list[str] = []

    class Camera:
        @staticmethod
        def newest(*, after_sequence: int) -> CapturedFrame:
            events.append(f"newest:{after_sequence}")
            if failures:
                raise failures.pop(0)
            return frame

        @staticmethod
        def reconnect() -> None:
            events.append("reconnect")
            if len(failures) == 1:
                raise RuntimeError("Could not open camera")

    monkeypatch.setattr(deploy.time, "sleep", lambda _seconds: events.append("sleep"))

    assert deploy.next_frame(Camera(), 3) is frame  # ty: ignore[invalid-argument-type]
    assert events == [
        "newest:3",
        "reconnect",
        "sleep",
        "newest:3",
        "reconnect",
        "newest:3",
    ]


def test_bird_window_saves_uncropped_source_frame_and_roi(tmp_path: Path) -> None:
    source = np.zeros((40, 60, 3), dtype=np.uint8)
    crop = source[12:32, 10:40].copy()
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 8:18] = True
    frame = CapturedFrame(
        1, datetime(2026, 10, 1, tzinfo=UTC), crop, source_bgr=source, roi=(10, 12, 40, 32)
    )

    class Segmenter:
        @staticmethod
        def segment(image: np.ndarray) -> tuple[np.ndarray, float]:
            assert image is crop
            return mask, 0.95

    sighting = deploy.save_bird_window(frame, 0.9, Segmenter(), tmp_path)  # ty: ignore[invalid-argument-type]

    label = json.loads((sighting.path / "label.json").read_text())
    assert (label["width"], label["height"]) == (60, 40)
    assert label["roi"] == [10, 12, 40, 32]
    assert label["bbox"] == [16, 15, 30, 29]
