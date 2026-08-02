"""INT8 OpenVINO bird detection using an end-to-end YOLO export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openvino as ov

from birdspotter.models import COCO_BIRD_CLASS_ID, DETECTOR_BIRD_CLASS_ID
from birdspotter.types import Detection


class OpenVinoDetectorBackend:
    def __init__(self, model_path: Path) -> None:
        model_files = sorted(model_path.glob("*.xml"))
        if len(model_files) != 1:
            raise FileNotFoundError(f"Expected one OpenVINO XML model in {model_path}")
        core = ov.Core()
        cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "openvino" / model_path.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache_dir)})
        model = core.read_model(model_files[0])
        if not any(operation.get_type_name() == "FakeQuantize" for operation in model.get_ops()):
            raise ValueError("Detector must be an INT8-quantized OpenVINO model")
        model_input = model.input(0)
        if not model_input.partial_shape.is_static:
            raise TypeError("Detector must have a fixed input shape")
        self.input_shape = tuple(model_input.shape)
        self.compiled_model = core.compile_model(
            model,
            "CPU",
            {"PERFORMANCE_HINT": "LATENCY"},
        )
        self.output = self.compiled_model.output(0)

    def run(self, tensor: np.ndarray) -> np.ndarray:
        return np.asarray(self.compiled_model([tensor])[self.output])

    def describe(self) -> dict[str, object]:
        return {
            "backend": "OpenVINO",
            "precision": "INT8",
            "device": "CPU",
            "performance_hint": "LATENCY",
        }


def letterbox(
    image_bgr: np.ndarray,
    size: int,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize and pad an image to a square while preserving its aspect ratio."""

    height, width = image_bgr.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, scale, (pad_x, pad_y)


def restore_box(
    box: np.ndarray,
    scale: float,
    padding: tuple[int, int],
    image_shape: tuple[int, ...],
) -> tuple[float, float, float, float]:
    """Map an xyxy box from letterboxed input back to source coordinates."""

    pad_x, pad_y = padding
    height, width = image_shape[:2]
    x1 = float(np.clip((box[0] - pad_x) / scale, 0, width - 1))
    y1 = float(np.clip((box[1] - pad_y) / scale, 0, height - 1))
    x2 = float(np.clip((box[2] - pad_x) / scale, 0, width))
    y2 = float(np.clip((box[3] - pad_y) / scale, 0, height))
    return x1, y1, x2, y2


class BirdDetector:
    """Detect birds using YOLO26's bird-only INT8 OpenVINO output."""

    def __init__(
        self,
        model_path: Path,
        *,
        confidence: float = 0.10,
        input_size: int | None = None,
    ) -> None:
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Detector model not found: {model_path}. Run `birdspotter models`."
            )
        if not 0 <= confidence <= 1:
            raise ValueError("Detector confidence must be between 0 and 1")

        self.model_path = model_path
        self.confidence = confidence
        self.backend = OpenVinoDetectorBackend(model_path)
        model_height, model_width = self.backend.input_shape[-2:]
        if not isinstance(model_height, int) or not isinstance(model_width, int):
            raise TypeError("Detector must have a fixed square input shape")
        if model_height != model_width:
            raise RuntimeError(f"Detector input is not square: {self.backend.input_shape}")
        if input_size is not None and input_size != model_height:
            raise ValueError(
                f"Configured detector input {input_size} does not match model {model_height}"
            )
        self.input_size = model_height

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise TypeError("Detector input must be an HxWx3 BGR image")

        prepared, scale, padding = letterbox(image_bgr, self.input_size)
        tensor = prepared[:, :, ::-1].transpose(2, 0, 1)
        tensor = np.ascontiguousarray(tensor[None], dtype=np.float32) / 255.0
        output = self.backend.run(tensor)
        rows = np.asarray(output).squeeze(0)

        if rows.ndim != 2 or rows.shape[1] < 6:
            raise RuntimeError(
                f"Unexpected detector output shape {np.asarray(output).shape}; "
                "prepare the detector with `birdspotter models`"
            )

        detections: list[Detection] = []
        for row in rows:
            confidence = float(row[4])
            class_id = round(float(row[5]))
            if confidence < self.confidence or class_id != DETECTOR_BIRD_CLASS_ID:
                continue
            box = restore_box(row[:4], scale, padding, image_bgr.shape)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            detections.append(
                Detection(box=box, confidence=confidence, class_id=COCO_BIRD_CLASS_ID)
            )

        return sorted(detections, key=lambda detection: detection.confidence, reverse=True)

    def describe(self) -> dict[str, Any]:
        """Return runtime information suitable for metadata and diagnostics."""

        return {
            "model": self.model_path.name,
            "input_size": self.input_size,
            "confidence_threshold": self.confidence,
            "bird_class_id": COCO_BIRD_CLASS_ID,
            "model_output_class_id": DETECTOR_BIRD_CLASS_ID,
            **self.backend.describe(),
        }
