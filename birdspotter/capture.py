"""Low-latency webcam capture that always exposes the newest decoded frame."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    sequence: int
    captured_at: datetime
    image_bgr: np.ndarray


class LatestFrameCamera:
    """Read full camera frames so inference never accumulates stale frames."""

    def __init__(
        self,
        device: int = 0,
        *,
        width: int = 1600,
        height: int = 896,
        fps: int = 5,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self._capture: cv2.VideoCapture | None = None
        self._condition = threading.Condition()
        self._latest: CapturedFrame | None = None
        self._error: Exception | None = None
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> Self:
        capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open camera /dev/video{self.device}")
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),  # ty: ignore[unresolved-attribute]
        )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
        self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
        self._thread.start()
        return self

    def _reader(self) -> None:
        capture = self._capture
        if capture is None:
            raise RuntimeError("Camera has not been started")
        sequence = 0
        try:
            while not self._stopping.is_set():
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames")
                sequence += 1
                frame = CapturedFrame(sequence, datetime.now(UTC), image)
                with self._condition:
                    self._latest = frame
                    self._condition.notify_all()
        except Exception as error:  # noqa: BLE001 -- thread boundary reports failures to caller
            with self._condition:
                self._error = error
                self._condition.notify_all()

    def newest(self, *, after_sequence: int = 0, timeout: float = 2.0) -> CapturedFrame:
        """Wait for and return a frame newer than the supplied sequence number."""

        deadline = time.monotonic() + timeout
        with self._condition:
            while self._latest is None or self._latest.sequence <= after_sequence:
                if self._error is not None:
                    raise RuntimeError("Camera reader failed") from self._error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a camera frame")
                self._condition.wait(remaining)
            return self._latest

    def close(self) -> None:
        self._stopping.set()
        if self._capture is not None:
            self._capture.release()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def actual_settings(self) -> dict[str, float | str]:
        if self._capture is None:
            raise RuntimeError("Camera has not been started")
        fourcc_number = int(self._capture.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_number >> (8 * index)) & 0xFF) for index in range(4))
        return {
            "width": self._capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": self._capture.get(cv2.CAP_PROP_FPS),
            "fourcc": fourcc,
            "output_crop": "none",
        }

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
