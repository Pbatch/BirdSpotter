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
    camera._capture = recording_capture  # ty: ignore[invalid-assignment]  # noqa: SLF001
    times = iter((0.0, 0.25, 0.75, 1.0, 1.05))
    monkeypatch.setattr(capture_module.time, "monotonic", lambda: next(times))

    camera._reader()  # noqa: SLF001

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


def test_set_roi_updates_the_detector_frame() -> None:
    camera = Capture()
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    camera._latest_source = CapturedFrame(3, datetime.now(UTC), image)  # noqa: SLF001

    camera.set_roi((2, 3, 9, 8))

    assert camera.roi == (2, 3, 9, 8)
    assert camera._latest is not None  # noqa: SLF001
    assert camera._latest.image_bgr.shape == (5, 7, 3)  # noqa: SLF001
    assert camera._latest.roi_revision == 1  # noqa: SLF001
