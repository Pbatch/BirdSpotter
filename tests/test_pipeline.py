from pathlib import Path

import numpy as np

from birdspotter.models import sam21_openvino_dir, sam21_path
from birdspotter.pipeline import expanded_crop


def test_expanded_crop_translates_detector_box() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    crop, local_box, source_crop = expanded_crop(
        image,
        (50, 20, 150, 80),
        margin_fraction=0.10,
    )

    assert source_crop == (40, 14, 160, 86)
    assert crop.shape == (72, 120, 3)
    assert local_box == (10, 6, 110, 66)


def test_sam21_checkpoint_uses_dedicated_weights_directory(tmp_path: Path) -> None:
    assert sam21_path(tmp_path) == tmp_path / "sam2.1" / "sam2.1_l.pt"


def test_sam21_openvino_models_use_dedicated_weights_directory(tmp_path: Path) -> None:
    assert sam21_openvino_dir(tmp_path) == tmp_path / "sam2.1" / "openvino-256"
