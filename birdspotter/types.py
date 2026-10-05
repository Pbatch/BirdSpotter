"""Small shared value types used by the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Classification:
    """A positive whole-frame bird classification."""

    confidence: float


@dataclass(slots=True)
class BirdCandidate:
    """The best bird seen so far in one selection window."""

    classification: Classification
    frame_bgr: np.ndarray  # ROI-cropped frame used for classification and segmentation
    frame_sequence: int
    captured_at: datetime
    classifier_seconds: float = 0.0
    source_bgr: np.ndarray | None = None
    roi: tuple[int, int, int, int] | None = None
