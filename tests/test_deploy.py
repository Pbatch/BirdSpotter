from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from birdspotter.types import BirdCandidate, Detection
from scripts.deploy import output_path, rounded_to_five_minutes


def test_rounded_to_five_minutes_rounds_half_up_across_an_hour() -> None:
    timestamp = datetime(2026, 8, 2, 12, 58, 30, tzinfo=UTC)

    assert rounded_to_five_minutes(timestamp) == datetime(2026, 8, 2, 13, 0, tzinfo=UTC)


def test_output_path_contains_percentage_and_rounded_time(tmp_path: Path) -> None:
    candidate = BirdCandidate(
        detection=Detection((1, 2, 3, 4), confidence=0.8234, class_id=14),
        frame_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
        frame_sequence=1,
        captured_at=datetime(2026, 8, 2, 12, 2, 30, tzinfo=UTC),
    )

    assert output_path(tmp_path, candidate) == tmp_path / "bird_conf_82_ts_2026-08-02_12-05.png"
