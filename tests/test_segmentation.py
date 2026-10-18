import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from birdspotter import segmentation
from birdspotter.weights import sam3_openvino_dir, sam3_tensorrt_dir


@pytest.mark.parametrize(
    ("files", "expected"),
    [(["sam3_bird.xml", "sam3_bird.bin"], "openvino"), (["sam3_bird.engine"], "tensorrt")],
)
def test_backend_is_chosen_from_the_model_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, files: list[str], expected: str
) -> None:
    for name in files:
        (tmp_path / name).touch()
    monkeypatch.setattr(segmentation, "Sam3OpenVinoSegmenter", lambda _path, **_kwargs: "openvino")
    # A stand-in module, so this runs without the optional TensorRT packages installed.
    monkeypatch.setitem(
        sys.modules,
        "birdspotter.segmentation.tensorrt_backend",
        SimpleNamespace(Sam3TensorRtSegmenter=lambda _path, **_kwargs: "tensorrt"),
    )

    assert segmentation.sam3_segmenter(tmp_path) == expected


TENSORRT_MODEL = sam3_tensorrt_dir(Path("weights"))
OPENVINO_MODEL = sam3_openvino_dir(Path("weights"))


@pytest.mark.skipif(
    importlib.util.find_spec("tensorrt") is None
    or not (TENSORRT_MODEL / "sam3_bird.engine").is_file(),
    reason="needs the tensorrt dependency group, an NVIDIA GPU and a locally built engine",
)
def test_tensorrt_engine_segments_a_bird() -> None:
    segmenter = segmentation.sam3_segmenter(TENSORRT_MODEL)
    image = cv2.imread("demo/images/1.png", cv2.IMREAD_COLOR)
    assert image is not None

    mask, score = segmenter.segment(image)

    assert segmenter.describe()["backend"] == "TensorRT"
    assert mask.shape == image.shape[:2]
    assert 0.5 <= score <= 1.0
    if (OPENVINO_MODEL / "sam3_bird.xml").is_file():
        reference, _ = segmentation.sam3_segmenter(OPENVINO_MODEL).segment(image)
        iou = np.count_nonzero(mask & reference) / np.count_nonzero(mask | reference)
        assert iou > 0.9
