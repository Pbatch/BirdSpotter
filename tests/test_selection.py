from datetime import UTC, datetime

import numpy as np

from birdspotter.types import BirdCandidate, Detection, better_candidate


def candidate(confidence: float) -> BirdCandidate:
    return BirdCandidate(
        detection=Detection((1, 2, 10, 20), confidence, 14),
        frame_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
        frame_sequence=1,
        captured_at=datetime.now(UTC),
    )


def test_highest_detector_confidence_wins() -> None:
    lower = candidate(0.51)
    higher = candidate(0.89)

    assert better_candidate(None, lower) is lower
    assert better_candidate(lower, higher) is higher
    assert better_candidate(higher, lower) is higher


def test_equal_confidence_keeps_first_candidate() -> None:
    first = candidate(0.75)
    second = candidate(0.75)

    assert better_candidate(first, second) is first
