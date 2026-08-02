import numpy as np
import pytest

from birdspotter.segmentation import component_for_box, validate_mask


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
