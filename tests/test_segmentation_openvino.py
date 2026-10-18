from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import pytest

from birdspotter.segmentation import openvino_backend


def embedded_preprocessing(frame: np.ndarray) -> np.ndarray:
    """Run only the preprocessing that ``embed_preprocessing`` adds to a SAM 3 export."""
    image = ov.op.Parameter(ov.Type.f16, ov.Shape([1, 3, 504, 504]))
    model = ov.Model(image, [image])  # ty: ignore[no-matching-overload]
    model.input().get_tensor().set_names({"image"})
    compiled = ov.Core().compile_model(openvino_backend.embed_preprocessing(model), "CPU")
    return compiled({"image": openvino_backend.prepare_frame(frame)})[0]


def test_embedded_preprocessing_preserves_both_ends_of_the_whole_frame() -> None:
    image = np.zeros((10, 30, 3), dtype=np.uint8)
    image[:, :10, 2] = 255
    image[:, 20:, 0] = 255
    tensor = embedded_preprocessing(image)
    assert tensor.shape == (1, 3, 504, 504)
    np.testing.assert_allclose(tensor[0, :, 0, 0], [1, -1, -1], atol=0.02)
    np.testing.assert_allclose(tensor[0, :, -1, -1], [-1, -1, 1], atol=0.02)


def test_embedded_preprocessing_resizes_and_normalizes_rgb() -> None:
    frame = np.random.default_rng(0).integers(0, 256, (48, 80, 3), dtype=np.uint8)
    resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (504, 504))
    expected = (resized.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
    np.testing.assert_allclose(embedded_preprocessing(frame), expected, atol=0.02)


def test_pin_encoder_splits_at_the_neck_and_keeps_dynamic_ops_off_the_encoder() -> None:
    image = ov.op.Parameter(ov.Type.f32, ov.PartialShape([1, -1]))
    resized = ov.opset13.pad(image, [0, 0], [0, 4], "constant")
    resized = ov.opset13.slice(resized, [0, 0], [1, 4], [1, 1])
    backbone = ov.opset13.relu(ov.opset13.reshape(resized, [1, 4], special_zero=False))
    neck = ov.opset13.multiply(backbone, ov.opset13.constant(np.float32(2)))
    decoder = ov.opset13.add(neck, ov.opset13.constant(np.float32(1)))
    backbone.set_friendly_name("model.vision_encoder.backbone")
    neck.set_friendly_name("model.vision_encoder.neck.proj")
    decoder.set_friendly_name("model.detr_decoder.add")
    model = openvino_backend.pin_encoder(ov.Model(decoder, [image]), "GPU", "CPU")  # ty: ignore[no-matching-overload]
    affinity = {
        op.get_friendly_name(): op.get_rt_info()["affinity"].astype(str)
        for op in model.get_ordered_ops()
    }
    assert affinity["model.vision_encoder.backbone"] == "GPU"
    assert affinity["model.vision_encoder.neck.proj"] == "GPU"
    assert affinity["model.detr_decoder.add"] == "CPU"
    assert affinity[image.get_friendly_name()] == "CPU"
    assert affinity[resized.input(0).get_source_output().get_node().get_friendly_name()] == "CPU"


class FakeBackend:
    def __init__(self, score: float) -> None:
        self.score = score

    @staticmethod
    def output(name: str) -> str:
        return name

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        assert inputs["image"].shape == (1, 20, 40, 3)
        logits = np.full((1, 1, 504, 504), -1, dtype=np.float16)
        logits[:, :, :, :252] = 1
        return {"mask_logits": logits, "score": np.array([self.score])}


@pytest.mark.parametrize("score", [0.49, float("nan")])
def test_segment_rejects_no_confident_bird(score: float) -> None:
    segmenter = openvino_backend.Sam3OpenVinoSegmenter.__new__(
        openvino_backend.Sam3OpenVinoSegmenter
    )
    segmenter.confidence = 0.5
    segmenter.backend = FakeBackend(score)  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match="no confident bird"):
        segmenter.segment(np.zeros((20, 40, 3), dtype=np.uint8))


def test_segment_returns_a_mask_in_full_frame_coordinates() -> None:
    segmenter = openvino_backend.Sam3OpenVinoSegmenter.__new__(
        openvino_backend.Sam3OpenVinoSegmenter
    )
    segmenter.confidence = 0.5
    backend = FakeBackend(0.9)
    segmenter.backend = backend  # ty: ignore[invalid-assignment]
    mask, score = segmenter.segment(np.zeros((20, 40, 3), dtype=np.uint8))
    assert mask.shape == (20, 40)
    assert mask[:, :20].all()
    assert not mask[:, 20:].any()
    assert score == pytest.approx(0.9)


def test_missing_model_explains_how_to_export(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sam3_export"):
        openvino_backend.Sam3OpenVinoSegmenter(tmp_path)
