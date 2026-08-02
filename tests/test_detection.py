import numpy as np

from birdspotter.detection import letterbox, restore_box


def test_letterbox_and_restore_round_trip() -> None:
    image = np.zeros((896, 1600, 3), dtype=np.uint8)
    prepared, scale, padding = letterbox(image, (1088, 1920))
    source_box = np.array([240.0, 120.0, 1400.0, 800.0], dtype=np.float32)
    model_box = source_box.copy()
    model_box[[0, 2]] = model_box[[0, 2]] * scale + padding[0]
    model_box[[1, 3]] = model_box[[1, 3]] * scale + padding[1]

    restored = restore_box(model_box, scale, padding, image.shape)

    assert prepared.shape == (1088, 1920, 3)
    assert padding == (0, 6)
    assert np.allclose(restored, source_box, atol=1e-3)
