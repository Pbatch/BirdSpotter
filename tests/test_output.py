import json
from pathlib import Path

import cv2
import numpy as np

from birdspotter.output import make_bgra, write_image, write_window


def test_make_bgra_tightly_crops_and_sets_alpha() -> None:
    image = np.full((20, 30, 3), (10, 20, 30), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 8:18] = True

    output, bounds = make_bgra(image, mask)

    assert bounds == (6, 3, 20, 17)
    assert output.shape == (14, 14, 4)
    assert output[0, 0, 3] == 0
    assert output[2, 2, 3] == 255
    assert tuple(output[2, 2, :3]) == (10, 20, 30)


def test_write_image_creates_only_rgba_png(tmp_path: Path) -> None:
    image = np.full((12, 16, 3), 80, dtype=np.uint8)
    mask = np.zeros((12, 16), dtype=bool)
    mask[3:9, 4:12] = True
    output_path = tmp_path / "bird.png"

    image_path = write_image(
        output_path,
        image,
        mask,
    )

    decoded = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    assert decoded.shape[2] == 4
    assert not list(tmp_path.glob("*.mask.png"))
    assert not list(tmp_path.glob("*.json"))


def test_write_window_saves_frame_cropped_mask_and_label(tmp_path: Path) -> None:
    image = np.full((20, 30, 3), 100, dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 8:18] = True

    label_path = write_window(tmp_path / "w", image, None, mask, confidence=0.8, sam_confidence=0.7)

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


def test_write_window_saves_uncropped_frame_with_full_frame_roi_and_bbox(
    tmp_path: Path,
) -> None:
    source = np.full((40, 60, 3), 100, dtype=np.uint8)
    crop_mask = np.zeros((20, 30), dtype=bool)
    crop_mask[5:15, 8:18] = True

    label_path = write_window(
        tmp_path / "w", source, (10, 12, 40, 32), crop_mask, confidence=0.8, sam_confidence=0.7
    )

    label = json.loads(label_path.read_text())
    assert (label["width"], label["height"]) == (60, 40)
    assert label["roi"] == [10, 12, 40, 32]
    assert label["bbox"] == [16, 15, 30, 29]
    frame = cv2.imread(str(tmp_path / "w" / "full_frame.jpg"))
    assert frame is not None
    assert frame.shape == (40, 60, 3)
    saved_mask = cv2.imread(str(tmp_path / "w" / "mask.png"), cv2.IMREAD_UNCHANGED)
    assert saved_mask is not None
    assert saved_mask.shape == (14, 14)
