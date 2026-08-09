"""Low-latency webcam capture that exposes a recent frame once per second."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

DECODE_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    sequence: int
    captured_at: datetime
    image_bgr: np.ndarray


class Capture:
    """Drain camera frames and expose one newly decoded frame per second."""

    def __init__(
        self,
        source: int | str = 0,
        *,
        width: int = 1600,
        height: int = 896,
        fps: int = 5,
    ) -> None:
        self.source = source
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
        if isinstance(self.source, str):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        backend = cv2.CAP_V4L2 if isinstance(self.source, int) else cv2.CAP_FFMPEG
        capture = cv2.VideoCapture(self.source, backend)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open camera {self.source_name()}")
        if isinstance(self.source, int):
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
        next_decode = time.monotonic() + DECODE_INTERVAL_SECONDS
        try:
            while not self._stopping.is_set():
                if time.monotonic() < next_decode:
                    if not capture.grab():
                        raise RuntimeError("Camera stopped returning frames")
                    continue

                # read() grabs the frame after those drained above, then decodes it.
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames")
                next_decode = time.monotonic() + DECODE_INTERVAL_SECONDS
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
            "source": self.source_name(),
            "width": self._capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": self._capture.get(cv2.CAP_PROP_FPS),
            "fourcc": fourcc,
            "output_crop": "none",
        }

    def source_name(self) -> str:
        """Describe the source without exposing RTSP credentials in logs."""

        if isinstance(self.source, int):
            return f"/dev/video{self.source}"
        parsed = urlsplit(self.source)
        if not parsed.scheme or not parsed.hostname:
            return "remote stream"
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
