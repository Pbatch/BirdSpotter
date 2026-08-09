import numpy as np

from birdspotter.crop import expanded_crop


def test_expanded_crop_translates_detector_box() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    crop, local_box = expanded_crop(
        image,
        (50, 20, 150, 80),
        margin_fraction=0.10,
    )

    assert crop.shape == (72, 120, 3)
    assert local_box == (10, 6, 110, 66)
