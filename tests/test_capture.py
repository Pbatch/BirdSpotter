import numpy as np
import pytest

from birdspotter import capture as capture_module
from birdspotter.capture import Capture


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
