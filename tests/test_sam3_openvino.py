from pathlib import Path

import numpy as np
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


class FakeBackend:
    def __init__(self, score: float) -> None:
        self.score = score

    @staticmethod
    def output(name: str) -> str:
        return name

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        assert inputs["image"].shape == (1, 3, 504, 504)
        logits = np.full((1, 1, 504, 504), -1, dtype=np.float16)
        logits[:, :, :, :252] = 1
        return {"mask_logits": logits, "score": np.array([self.score])}


@pytest.mark.parametrize("score", [0.49, float("nan")])
def test_segment_rejects_no_confident_bird(score: float) -> None:
    segmenter = sam3_openvino.Sam3OpenVinoSegmenter.__new__(sam3_openvino.Sam3OpenVinoSegmenter)
    segmenter.confidence = 0.5
    segmenter.backend = FakeBackend(score)  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match="no confident bird"):
        segmenter.segment(np.zeros((20, 40, 3), dtype=np.uint8))


def test_segment_returns_a_mask_in_full_frame_coordinates() -> None:
    segmenter = sam3_openvino.Sam3OpenVinoSegmenter.__new__(sam3_openvino.Sam3OpenVinoSegmenter)
    segmenter.confidence = 0.5
    segmenter.backend = FakeBackend(0.9)  # ty: ignore[invalid-assignment]
    mask, score = segmenter.segment(np.zeros((20, 40, 3), dtype=np.uint8))
    assert mask.shape == (20, 40)
    assert mask[:, :20].all()
    assert not mask[:, 20:].any()
    assert score == pytest.approx(0.9)


def test_missing_model_explains_how_to_export(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sam3_export"):
        sam3_openvino.Sam3OpenVinoSegmenter(tmp_path)
