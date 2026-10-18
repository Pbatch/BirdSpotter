"""Run the fixed-prompt SAM 3 TensorRT engine on the local NVIDIA GPU.

Needs the optional `tensorrt` dependency group; only import this module for an engine.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from birdspotter.segmentation.sam3 import (
    ENGINE_FILENAME,
    SAM3_IMAGE_SIZE,
    check_frame,
    mask_from_outputs,
)
from birdspotter.tensorrt_engine import TensorRtEngine

INPUT_SHAPE = (1, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE, 3)
OUTPUTS = ("mask_logits", "box", "score")


def prepare_frame(image_bgr: np.ndarray) -> np.ndarray:
    """Resize the whole frame on the host; the engine normalises the uint8 BGR pixels."""
    check_frame(image_bgr)
    resized = cv2.resize(image_bgr, (SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE))
    return np.ascontiguousarray(resized[None], dtype=np.uint8)


class Sam3TensorRtSegmenter:
    """Return the highest-confidence bird without a classifier box prompt."""

    def __init__(self, model_dir: Path, *, confidence: float = 0.5) -> None:
        if not 0 <= confidence <= 1:
            raise ValueError("SAM 3 confidence must be between zero and one")
        self.model_path = model_dir / ENGINE_FILENAME
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"SAM 3 TensorRT engine not found: {self.model_path}. "
                "Run `python -m birdspotter.ml.sam3_export --tensorrt`."
            )
        self.confidence = confidence
        self._engine = TensorRtEngine(self.model_path)
        if set(self._engine.inputs) != {"image"} or set(self._engine.outputs) != set(OUTPUTS):
            names = sorted(self._engine.shapes)
            raise TypeError(f"SAM 3 engine has unexpected tensors: {names}")
        if self._engine.shapes["image"] != INPUT_SHAPE or self._engine.dtypes["image"] != np.uint8:
            raise TypeError(f"SAM 3 engine must take uint8 pixels shaped {INPUT_SHAPE}")
        self.device = self._engine.device

    def prepare(self, image_bgr: np.ndarray) -> np.ndarray:
        return prepare_frame(image_bgr)

    def run(self, pixels: np.ndarray) -> dict[str, np.ndarray]:
        """Run one prepared 1x504x504x3 uint8 batch and return every output."""
        return self._engine.run({"image": pixels})

    def segment(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Segment the whole image; reject absent birds and empty masks."""
        result = self.run(self.prepare(image_bgr))
        score = float(result["score"].reshape(-1)[0])
        logits = result["mask_logits"].squeeze()
        return mask_from_outputs(score, logits, image_bgr.shape, self.confidence)

    def describe(self) -> dict[str, object]:
        return {
            "model": "SAM 3",
            "model_path": self.model_path.name,
            "prompt": "bird",
            "device": self.device,
            "input_size": SAM3_IMAGE_SIZE,
            "precision": "FP16",
            "confidence": self.confidence,
            "backend": "TensorRT",
        }
