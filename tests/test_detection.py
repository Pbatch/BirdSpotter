from types import SimpleNamespace
from typing import cast

import numpy as np
import torch
from ultralytics import YOLO

from birdspotter.capture import center_square_crop
from birdspotter.detection import letterbox, restore_box
from birdspotter.models import (
    COCO_BIRD_CLASS_ID,
    DETECTOR_BIRD_CLASS_ID,
    bird_only_detector,
)


class FakeDetectionHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nc = 80
        self.reg_max = 16
        self.end2end = True
        self.cv3 = self.classifier_heads()
        self.one2one_cv3 = self.classifier_heads()

    @staticmethod
    def classifier_heads() -> torch.nn.ModuleList:
        return torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Conv2d(2, 80, kernel_size=1, bias=True))]
        )


def final_classifier(classifier_head: torch.nn.ModuleList) -> torch.nn.Conv2d:
    stage = classifier_head[0]
    assert isinstance(stage, torch.nn.Sequential)
    classifier = stage[-1]
    assert isinstance(classifier, torch.nn.Conv2d)
    return classifier


def bird_only_test_model() -> SimpleNamespace:
    head = FakeDetectionHead()
    for classifier_head in (head.cv3, head.one2one_cv3):
        classifier = final_classifier(classifier_head)
        with torch.no_grad():
            classifier.weight.copy_(
                torch.arange(classifier.weight.numel(), dtype=torch.float32).reshape_as(
                    classifier.weight
                )
            )
            classifier_bias = classifier.bias
            assert classifier_bias is not None
            classifier_bias.copy_(torch.arange(80, dtype=torch.float32))
    return SimpleNamespace(model=SimpleNamespace(model=[head], names={}))


def test_letterbox_and_restore_round_trip() -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    prepared, scale, padding = letterbox(image, 640)
    source_box = np.array([240.0, 120.0, 1600.0, 900.0], dtype=np.float32)
    model_box = source_box.copy()
    model_box[[0, 2]] = model_box[[0, 2]] * scale + padding[0]
    model_box[[1, 3]] = model_box[[1, 3]] * scale + padding[1]

    restored = restore_box(model_box, scale, padding, image.shape)

    assert prepared.shape == (640, 640, 3)
    assert np.allclose(restored, source_box, atol=1e-3)


def test_center_square_crop_uses_the_middle_of_a_wide_camera_frame() -> None:
    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)

    cropped = center_square_crop(image)

    assert cropped.shape == (3, 3, 3)
    assert np.array_equal(cropped, image[:, 1:4])


def test_bird_only_detector_preserves_the_pretrained_bird_channel() -> None:
    model = bird_only_test_model()
    head = model.model.model[-1]
    before = []
    for classifier_head in (head.cv3, head.one2one_cv3):
        classifier = final_classifier(classifier_head)
        classifier_bias = classifier.bias
        assert classifier_bias is not None
        before.append(
            (
                classifier.weight[COCO_BIRD_CLASS_ID].clone(),
                classifier_bias[COCO_BIRD_CLASS_ID].clone(),
            )
        )

    bird_only_detector(cast(YOLO, model))

    assert head.nc == 1
    assert head.no == 65
    assert model.model.names == {DETECTOR_BIRD_CLASS_ID: "bird"}
    for classifier_head, (weight, bias) in zip((head.cv3, head.one2one_cv3), before, strict=True):
        classifier = final_classifier(classifier_head)
        classifier_bias = classifier.bias
        assert classifier_bias is not None
        assert classifier.out_channels == 1
        assert torch.equal(classifier.weight[0], weight)
        assert torch.equal(classifier_bias[0], bias)
