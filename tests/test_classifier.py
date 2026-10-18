import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from birdspotter import classifier as classifier_module
from birdspotter.classifier import BirdClassifier, letterbox


def test_letterbox_preserves_frame() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    prepared, scale, padding = letterbox(image, (224, 224))
    assert prepared.shape == (224, 224, 3)
    assert scale == 1.12
    assert padding == (0, 56)


def test_single_logit_gates_frame() -> None:
    class Backend:
        logit = 0.0

        def run(self, tensor: np.ndarray) -> np.ndarray:
            assert tensor.shape == (1, 224, 224, 3)
            return np.array([[self.logit]], dtype=np.float32)

    classifier = BirdClassifier.__new__(BirdClassifier)
    classifier.input_shape = (224, 224)
    classifier.confidence = 0.5
    backend = Backend()
    classifier.backend = backend  # ty: ignore[invalid-assignment]
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert classifier.predict(frame) == 0.5
    assert classifier.classify(frame) == 0.5
    backend.logit = -1000
    assert classifier.classify(frame) is None
    backend.logit = 1000
    assert classifier.predict(frame) == 1.0


def test_classifier_receives_letterboxed_uint8_pixels() -> None:
    classifier = BirdClassifier.__new__(BirdClassifier)
    classifier.input_shape = (224, 224)
    frame = np.full((100, 200, 3), 7, dtype=np.uint8)

    tensor = classifier.preprocess(frame)

    assert tensor.dtype == np.uint8
    assert tensor.shape == (1, 224, 224, 3)
    assert np.array_equal(tensor[0], letterbox(frame, (224, 224))[0])


@pytest.mark.parametrize(
    ("files", "expected"),
    [(["classifier.xml", "classifier.bin"], "openvino"), (["classifier.engine"], "tensorrt")],
)
def test_backend_is_chosen_from_the_model_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, files: list[str], expected: str
) -> None:
    for name in files:
        (tmp_path / name).touch()
    monkeypatch.setattr(classifier_module, "OpenVinoClassifierBackend", lambda _path: "openvino")
    # A stand-in module, so this runs without the optional TensorRT packages installed.
    monkeypatch.setitem(
        sys.modules,
        "birdspotter.classifier.tensorrt_backend",
        SimpleNamespace(TensorRtClassifierBackend=lambda _path: "tensorrt"),
    )

    assert classifier_module.classifier_backend(tmp_path) == expected


TENSORRT_MODEL = Path(
    "weights/classifier/candidates/mobilenetv4-small-640-a100-b128-lr1e4-dp01-bce-clean-tensorrt"
)


@pytest.mark.skipif(
    importlib.util.find_spec("tensorrt") is None or not TENSORRT_MODEL.is_dir(),
    reason="needs the tensorrt dependency group, an NVIDIA GPU and a locally built engine",
)
def test_tensorrt_engine_runs_a_frame() -> None:
    classifier = BirdClassifier(TENSORRT_MODEL)

    score = classifier.predict(np.zeros((200, 200, 3), dtype=np.uint8))

    assert classifier.describe()["backend"] == "TensorRT"
    assert 0.0 <= score <= 1.0
