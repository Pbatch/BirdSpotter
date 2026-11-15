"""Greyscale motion gate for SAM 3: flag frames that differ from the recent median.

Each ROI crop is shrunk to a small greyscale image and compared against the per-pixel
median of the previous minute's frames. A bird that has just arrived is not in that
median, so it shows up as a compact, strong difference; slow lighting drift moves the
median along with it. The score is the largest 3x3-averaged absolute difference in grey
levels, so a single small bird is not diluted by the rest of the scene.

The median also ignores the camera's compression texture, which is redrawn at every
H.264 keyframe (every 40 frames on the Tapo C120) and shifts all pixels slightly.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np

WORKING_SIZE = 120
DIFFERENCE_BLUR = 3


def grey_thumbnail(image_bgr: np.ndarray, size: int = WORKING_SIZE) -> np.ndarray:
    """Return a greyscale copy whose longer side is ``size`` pixels."""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or not image_bgr.size:
        raise TypeError("Motion gate input must be a nonempty HxWx3 BGR image")
    height, width = image_bgr.shape[:2]
    scale = size / max(height, width)
    shape = (max(1, round(width * scale)), max(1, round(height * scale)))
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, shape, interpolation=cv2.INTER_AREA).astype(np.float32)


class MotionGate:
    """Score frames by their difference from the median of the recent past."""

    def __init__(
        self,
        *,
        threshold: float = 70.0,
        history_frames: int = 60,
        sample_every: int = 2,
    ) -> None:
        if not 0 < threshold <= 255:
            raise ValueError("Motion threshold must be a grey-level difference in (0, 255]")
        if history_frames < 2:
            raise ValueError("Motion history must hold at least two frames")
        if not 1 <= sample_every <= history_frames:
            raise ValueError("Median sampling step must be between 1 and the history length")
        self.threshold = threshold
        self.sample_every = sample_every
        self.history: deque[np.ndarray] = deque(maxlen=history_frames)

    @property
    def ready(self) -> bool:
        """Whether the history covers the full background period."""

        return len(self.history) == self.history.maxlen

    def score(self, image_bgr: np.ndarray) -> float:
        """Return the frame's motion score, then add it to the history; zero until ready."""

        frame = grey_thumbnail(image_bgr)
        if self.history and self.history[-1].shape != frame.shape:
            self.history.clear()
        score = 0.0
        if self.ready:
            background = np.median(np.stack(list(self.history)[:: self.sample_every]), axis=0)
            difference = cv2.blur(np.abs(frame - background), (DIFFERENCE_BLUR, DIFFERENCE_BLUR))
            score = float(difference.max())
        self.history.append(frame)
        return score

    def check(self, image_bgr: np.ndarray) -> float | None:
        """Return the motion score when it meets the threshold, otherwise None."""

        score = self.score(image_bgr)
        return score if score >= self.threshold else None

    def reset(self) -> None:
        """Forget the background, e.g. after the ROI changes."""

        self.history.clear()

    def describe(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "history_frames": self.history.maxlen,
            "sample_every": self.sample_every,
            "working_size": WORKING_SIZE,
        }
