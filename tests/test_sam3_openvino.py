from pathlib import Path

import numpy as np
import openvino as ov
import pytest

from birdspotter import sam3_openvino


def test_preprocessing_preserves_both_ends_of_the_whole_frame() -> None:
    image = np.zeros((10, 30, 3), dtype=np.uint8)
    image[:, :10, 2] = 255
    image[:, 20:, 0] = 255
    tensor = sam3_openvino.preprocess_image(image)
    assert tensor.shape == (1, 3, 504, 504)
    assert tensor.dtype == np.float16
    np.testing.assert_array_equal(tensor[0, :, 0, 0], [1, -1, -1])
    np.testing.assert_array_equal(tensor[0, :, -1, -1], [-1, -1, 1])


def test_embedded_preprocessing_matches_python_preprocessing() -> None:
    image = ov.op.Parameter(ov.Type.f16, ov.Shape([1, 3, 504, 504]))
    model = ov.Model(image, [image])  # ty: ignore[no-matching-overload]
    model.input().get_tensor().set_names({"image"})
    compiled = ov.Core().compile_model(sam3_openvino.embed_preprocessing(model), "CPU")
    frame = np.random.default_rng(0).integers(0, 256, (48, 80, 3), dtype=np.uint8)
    embedded = compiled({"image": sam3_openvino.prepare_frame(frame)})[0]
    assert embedded.shape == (1, 3, 504, 504)
    np.testing.assert_allclose(embedded, sam3_openvino.preprocess_image(frame), atol=0.02)


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
    model = sam3_openvino.pin_encoder(ov.Model(decoder, [image]), "GPU", "CPU")  # ty: ignore[no-matching-overload]
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
    def __init__(self, score: float, *, embedded_preprocessing: bool = False) -> None:
        self.score = score
        self.input_shape = (1, 20, 40, 3) if embedded_preprocessing else (1, 3, 504, 504)

    @staticmethod
    def output(name: str) -> str:
        return name

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        assert inputs["image"].shape == self.input_shape
        logits = np.full((1, 1, 504, 504), -1, dtype=np.float16)
        logits[:, :, :, :252] = 1
        return {"mask_logits": logits, "score": np.array([self.score])}


@pytest.mark.parametrize("score", [0.49, float("nan")])
def test_segment_rejects_no_confident_bird(score: float) -> None:
    segmenter = sam3_openvino.Sam3OpenVinoSegmenter.__new__(sam3_openvino.Sam3OpenVinoSegmenter)
    segmenter.confidence = 0.5
    segmenter.embedded_preprocessing = False
    segmenter.backend = FakeBackend(score)  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match="no confident bird"):
        segmenter.segment(np.zeros((20, 40, 3), dtype=np.uint8))


@pytest.mark.parametrize("embedded_preprocessing", [False, True])
def test_segment_returns_a_mask_in_full_frame_coordinates(*, embedded_preprocessing: bool) -> None:
    segmenter = sam3_openvino.Sam3OpenVinoSegmenter.__new__(sam3_openvino.Sam3OpenVinoSegmenter)
    segmenter.confidence = 0.5
    segmenter.embedded_preprocessing = embedded_preprocessing
    backend = FakeBackend(0.9, embedded_preprocessing=embedded_preprocessing)
    segmenter.backend = backend  # ty: ignore[invalid-assignment]
    mask, score = segmenter.segment(np.zeros((20, 40, 3), dtype=np.uint8))
    assert mask.shape == (20, 40)
    assert mask[:, :20].all()
    assert not mask[:, 20:].any()
    assert score == pytest.approx(0.9)


def test_missing_model_explains_how_to_export(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sam3_export"):
        sam3_openvino.Sam3OpenVinoSegmenter(tmp_path)
