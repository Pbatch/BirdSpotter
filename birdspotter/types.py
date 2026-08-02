"""Small shared value types used by the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Detection:
    """A bird detection in source-image coordinates."""

    box: Box
    confidence: float
    class_id: int


@dataclass(slots=True)
class BirdCandidate:
    """The best bird seen so far in one selection window."""

    detection: Detection
    frame_bgr: np.ndarray
    frame_sequence: int
    captured_at: datetime
    detector_seconds: float = 0.0


def better_candidate(
    current: BirdCandidate | None,
    candidate: BirdCandidate,
) -> BirdCandidate:
    """Return the candidate with the higher detector confidence."""

    if current is None or candidate.detection.confidence > current.detection.confidence:
        return candidate
    return current
