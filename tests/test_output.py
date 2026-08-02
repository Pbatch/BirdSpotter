from pathlib import Path

import cv2
import numpy as np

from birdspotter.output import make_bgra, write_image


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
