import numpy as np

from birdspotter.classification import BirdClassifier, letterbox


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
            assert tensor.shape == (1, 3, 224, 224)
            return np.array([[self.logit]], dtype=np.float32)

    classifier = BirdClassifier.__new__(BirdClassifier)
    classifier.input_shape = (224, 224)
    classifier.confidence = 0.5
    backend = Backend()
    classifier.backend = backend  # ty: ignore[invalid-assignment]
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert classifier.predict(frame) == 0.5
    result = classifier.classify(frame)
    assert result is not None
    assert result.confidence == 0.5
    backend.logit = -1000
    assert classifier.classify(frame) is None
    backend.logit = 1000
    assert classifier.predict(frame) == 1.0
