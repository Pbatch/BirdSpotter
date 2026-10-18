import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from birdspotter.sighting import Sighting, window_dir, window_start, write_window


def test_window_start_floors_to_the_clock_aligned_five_minutes() -> None:
    timestamp = datetime(2026, 8, 2, 12, 59, 59, tzinfo=UTC)

    assert window_start(timestamp) == datetime(2026, 8, 2, 12, 55, tzinfo=UTC)


def test_window_dir_is_named_after_the_window_start(tmp_path: Path) -> None:
    assert window_dir(tmp_path, datetime(2026, 8, 2, 12, 4, 59, tzinfo=UTC)) == (
        tmp_path / "2026-08-02_12-00"
    )
    assert window_dir(tmp_path, datetime(2026, 8, 2, 12, 5, tzinfo=UTC)) == (
        tmp_path / "2026-08-02_12-05"
    )


def test_write_window_saves_frame_cropped_mask_and_label(tmp_path: Path) -> None:
    image = np.full((20, 30, 3), 100, dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 8:18] = True

    sighting = write_window(tmp_path / "w", image, None, mask, confidence=0.8, sam_confidence=0.7)

    assert sighting.bbox == (6, 3, 20, 17)
    label_path = tmp_path / "w" / "label.json"

    assert json.loads(label_path.read_text()) == {
        "width": 30,
        "height": 20,
        "roi": None,
        "bbox": [6, 3, 20, 17],
        "confidence": 0.8,
        "sam_confidence": 0.7,
    }
    frame = cv2.imread(str(tmp_path / "w" / "full_frame.jpg"))
    assert frame is not None
    assert frame.shape == (20, 30, 3)
    saved_mask = cv2.imread(str(tmp_path / "w" / "mask.png"), cv2.IMREAD_UNCHANGED)
    assert saved_mask is not None
    assert saved_mask.shape == (14, 14)
    assert saved_mask[0, 0] == 0
    assert saved_mask[2, 2] == 255
    assert not list((tmp_path / "w").glob(".*.part"))
    cutout = cv2.imread(str(tmp_path / "birds" / "w.png"), cv2.IMREAD_UNCHANGED)
    assert cutout is not None
    assert cutout.shape == (14, 14, 4)
    assert (cutout[..., :3] == 100).all()
    assert (cutout[..., 3] == saved_mask).all()
    assert not list((tmp_path / "birds").glob(".*.part"))


def test_write_window_saves_uncropped_frame_with_full_frame_roi_and_bbox(
    tmp_path: Path,
) -> None:
    source = np.full((40, 60, 3), 100, dtype=np.uint8)
    source[15:29, 16:30] = (10, 200, 30)  # the bird's full-frame bbox, in BGR
    crop_mask = np.zeros((20, 30), dtype=bool)
    crop_mask[5:15, 8:18] = True

    sighting = write_window(
        tmp_path / "w", source, (10, 12, 40, 32), crop_mask, confidence=0.8, sam_confidence=0.7
    )

    label = json.loads((sighting.path / "label.json").read_text())
    assert (label["width"], label["height"]) == (60, 40)
    assert label["roi"] == [10, 12, 40, 32]
    assert label["bbox"] == [16, 15, 30, 29]
    frame = cv2.imread(str(tmp_path / "w" / "full_frame.jpg"))
    assert frame is not None
    assert frame.shape == (40, 60, 3)
    saved_mask = cv2.imread(str(tmp_path / "w" / "mask.png"), cv2.IMREAD_UNCHANGED)
    assert saved_mask is not None
    assert saved_mask.shape == (14, 14)
    # The cutout comes from the full frame at the bbox, not from the ROI's own origin.
    cutout = cv2.imread(str(tmp_path / "birds" / "w.png"), cv2.IMREAD_UNCHANGED)
    assert cutout is not None
    assert (cutout[..., :3] == (10, 200, 30)).all()
    assert (cutout[..., 3] == saved_mask).all()


def test_sighting_load_round_trips_the_saved_label(tmp_path: Path) -> None:
    source = np.full((40, 60, 3), 100, dtype=np.uint8)
    crop_mask = np.zeros((20, 30), dtype=bool)
    crop_mask[5:15, 8:18] = True

    saved = write_window(
        tmp_path / "w", source, (10, 12, 40, 32), crop_mask, confidence=0.8, sam_confidence=0.7
    )

    assert Sighting.load(tmp_path / "w") == saved
    assert saved.has_bird
    assert saved.frame_roi == (10, 12, 40, 32)


def test_empty_sighting_uses_the_whole_frame(tmp_path: Path) -> None:
    empty = write_window(
        tmp_path / "w",
        np.zeros((4, 6, 3), np.uint8),
        None,
        None,
        confidence=None,
        sam_confidence=None,
    )

    assert Sighting.load(tmp_path / "w") == empty
    assert not empty.has_bird
    assert empty.frame_roi == (0, 0, 6, 4)
    assert not (tmp_path / "birds").exists()
