"""OpenVINO MobileNetV4 frame classification for the SAM 3 gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openvino as ov

from birdspotter.types import Classification


class OpenVinoClassifierBackend:
    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "CPU",
        performance_hint: str = "LATENCY",
    ) -> None:
        model_files = sorted(model_path.glob("*.xml"))
        if len(model_files) != 1:
            raise FileNotFoundError(f"Expected one OpenVINO XML model in {model_path}")
        core = ov.Core()
        cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "openvino" / model_path.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache_dir)})
        model = core.read_model(model_files[0])
        model_input = model.input(0)
        if not model_input.partial_shape.is_static:
            raise TypeError("Classifier must have a fixed input shape")
        self.input_shape = tuple(model_input.shape)
        self.compiled_model = core.compile_model(
            model,
            device,
            {"PERFORMANCE_HINT": performance_hint},
        )
        self.output = self.compiled_model.output(0)
        self.device = device
        self.performance_hint = performance_hint

    def run(self, tensor: np.ndarray) -> np.ndarray:
        return np.asarray(self.compiled_model([tensor])[self.output])

    def describe(self) -> dict[str, object]:
        return {
            "backend": "OpenVINO",
            "precision": "FP16 weights / FP32 inference",
            "device": self.device,
            "performance_hint": self.performance_hint,
        }


def letterbox(
    image_bgr: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize and pad an image to an HxW shape while preserving its aspect ratio."""

    height, width = image_bgr.shape[:2]
    target_height, target_width = shape
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    canvas = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, scale, (pad_x, pad_y)


class BirdClassifier:
    """Classify whole frames with a binary MobileNetV4 OpenVINO model."""

    def __init__(
        self,
        model_path: Path,
        *,
        confidence: float = 0.50,
    ) -> None:
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Classifier model not found: {model_path}. "
                "Install the OpenVINO export produced by Modal training."
            )
        if not 0 <= confidence <= 1:
            raise ValueError("Classifier confidence must be between 0 and 1")

        self.model_path = model_path
        self.confidence = confidence
        self.backend = OpenVinoClassifierBackend(model_path)
        model_height, model_width = self.backend.input_shape[-2:]
        if not isinstance(model_height, int) or not isinstance(model_width, int):
            raise TypeError("Classifier must have a fixed input shape")
        self.input_shape = (model_height, model_width)

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """Preserve the full frame and apply the training normalization."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or not image_bgr.size:
            raise TypeError("Classifier input must be a nonempty HxWx3 BGR image")
        prepared, _, _ = letterbox(image_bgr, self.input_shape)
        rgb = prepared[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])

    def predict(self, image_bgr: np.ndarray) -> float:
        """Return the probability that the whole frame contains a bird."""
        logits = np.asarray(self.backend.run(self.preprocess(image_bgr)))
        if logits.shape != (1, 1) or not np.isfinite(logits).all():
            raise RuntimeError(f"Expected one finite bird logit of shape (1, 1), got {logits}")
        logit = float(logits[0, 0])
        if logit >= 0:
            return float(1 / (1 + np.exp(-logit)))
        exponential = np.exp(logit)
        return float(exponential / (1 + exponential))

    def classify(self, image_bgr: np.ndarray) -> Classification | None:
        """Return a positive frame classification when bird probability meets the threshold."""
        confidence = self.predict(image_bgr)
        if confidence < self.confidence:
            return None
        return Classification(confidence=confidence)

    def describe(self) -> dict[str, Any]:
        """Return runtime information suitable for metadata and diagnostics."""

        return {
            "model": self.model_path.name,
            "input_shape": self.input_shape,
            "confidence_threshold": self.confidence,
            "task": "image-classification",
            "classes": ["bird"],
            "output_activation": "sigmoid",
            "model_output_class_id": 0,
            **self.backend.describe(),
        }
