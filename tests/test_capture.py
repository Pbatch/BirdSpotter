import threading
from datetime import UTC, datetime

import numpy as np
import pytest

from birdspotter import capture as capture_module
from birdspotter.capture import Capture, CapturedFrame, crop_to_roi


class RecordingCapture:
    def __init__(self, camera: Capture) -> None:
        self.camera = camera
        self.events: list[str] = []

    def grab(self) -> bool:
        self.events.append("grab")
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        self.events.append("read")
        self.camera._stopping.set()  # noqa: SLF001 -- stop the reader after its first decode
        return True, np.zeros((2, 2, 3), dtype=np.uint8)


def test_reader_grabs_until_interval_then_reads_next_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = Capture()
    recording_capture = RecordingCapture(camera)
    times = iter((0.0, 0.25, 0.75, 1.0, 1.05))
    monkeypatch.setattr(capture_module.time, "monotonic", lambda: next(times))

    camera._reader(recording_capture, camera._stopping)  # ty: ignore[invalid-argument-type]  # noqa: SLF001

    assert recording_capture.events == ["grab", "grab", "read"]
    assert camera._latest is not None  # noqa: SLF001
    assert camera._latest.sequence == 1  # noqa: SLF001


def test_crop_to_roi_returns_selected_pixels() -> None:
    image = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)

    cropped = crop_to_roi(image, (2, 3, 9, 8))

    assert cropped.shape == (5, 7, 3)
    assert np.array_equal(cropped, image[3:8, 2:9])
    assert not np.shares_memory(cropped, image)


def test_crop_to_roi_rejects_invalid_coordinates() -> None:
    image = np.zeros((10, 12, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="outside"):
        crop_to_roi(image, (0, 0, 13, 10))
    with pytest.raises(ValueError, match="positive"):
        crop_to_roi(image, (5, 2, 5, 8))


def test_set_roi_updates_the_classifier_frame() -> None:
    camera = Capture()
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    camera._latest_source = CapturedFrame(3, datetime.now(UTC), image)  # noqa: SLF001

    camera.set_roi((2, 3, 9, 8))

    assert camera.roi == (2, 3, 9, 8)
    assert camera._latest is not None  # noqa: SLF001
    assert camera._latest.image_bgr.shape == (5, 7, 3)  # noqa: SLF001
    assert camera._latest.roi_revision == 1  # noqa: SLF001


class BlockedCapture:
    """A capture whose reader thread is stuck inside FFmpeg until unblocked."""

    def __init__(self) -> None:
        self.unblock = threading.Event()
        self.released = False

    def grab(self) -> bool:
        self.unblock.wait()
        return False

    def release(self) -> None:
        self.released = True


def test_close_does_not_release_a_capture_the_reader_is_still_using(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capture_module, "STREAM_TIMEOUT_SECONDS", 0.0)
    camera = Capture()
    blocked = BlockedCapture()
    camera._capture = blocked  # ty: ignore[invalid-assignment]  # noqa: SLF001
    camera._thread = threading.Thread(  # noqa: SLF001
        target=camera._reader,  # noqa: SLF001
        args=(blocked, camera._stopping),  # noqa: SLF001
        daemon=True,
    )
    camera._thread.start()  # noqa: SLF001

    camera.close()

    assert not blocked.released
    blocked.unblock.set()


def test_close_releases_the_capture_after_the_reader_stops() -> None:
    camera = Capture()
    blocked = BlockedCapture()
    blocked.unblock.set()
    camera._capture = blocked  # ty: ignore[invalid-assignment]  # noqa: SLF001
    camera._thread = threading.Thread(  # noqa: SLF001
        target=camera._reader,  # noqa: SLF001
        args=(blocked, camera._stopping),  # noqa: SLF001
        daemon=True,
    )
    camera._thread.start()  # noqa: SLF001

    camera.close()

    assert blocked.released


def test_reconnect_clears_the_reader_error_and_keeps_the_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = Capture()
    camera._sequence = 7  # noqa: SLF001
    camera._error = RuntimeError("Camera stopped returning frames")  # noqa: SLF001
    old_stopping = camera._stopping  # noqa: SLF001
    starts: list[threading.Event] = []
    monkeypatch.setattr(camera, "start", lambda: starts.append(camera._stopping))  # noqa: SLF001

    camera.reconnect()

    assert old_stopping.is_set()
    assert starts == [camera._stopping]  # noqa: SLF001
    assert not camera._stopping.is_set()  # noqa: SLF001
    assert camera._error is None  # noqa: SLF001
    assert camera._sequence == 7  # noqa: SLF001
