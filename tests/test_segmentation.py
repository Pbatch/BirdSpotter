from pathlib import Path

import numpy as np
import pytest

from birdspotter import segmentation
from birdspotter.segmentation import (
    Sam21Segmenter,
    component_for_box,
    scale_box,
    validate_mask,
)


def test_scale_box_uses_resized_image_coordinates() -> None:
    scaled = scale_box((100, 50, 500, 250), (500, 1000))

    assert scaled.shape == (1, 2, 2)
    assert np.allclose(scaled, [[[102.4, 51.2], [512.0, 256.0]]])


def test_component_overlapping_detector_box_is_retained() -> None:
    mask = np.zeros((100, 120), dtype=bool)
    mask[5:45, 5:45] = True
    mask[60:80, 70:95] = True

    selected = component_for_box(mask, (65, 55, 100, 85))

    assert selected[70, 80]
    assert not selected[20, 20]


def test_validate_mask_rejects_mask_outside_box() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[5:20, 5:20] = True

    with pytest.raises(ValueError, match="does not overlap"):
        validate_mask(mask, (50, 50, 90, 90))


def test_sam21_describe_reports_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "sam2.1_l.pt"
    checkpoint.touch()
    model = object()
    monkeypatch.setattr(segmentation, "SAM", lambda _path: model)

    segmenter = Sam21Segmenter(checkpoint)

    assert segmenter.model is model
    assert segmenter.describe()["model"] == "SAM 2.1 Large"
    assert segmenter.describe()["checkpoint"] == "sam2.1_l.pt"
    assert segmenter.describe()["input_size"] == 1024
