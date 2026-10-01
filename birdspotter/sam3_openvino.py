"""Whole-image bird segmentation with the fixed-prompt SAM 3 OpenVINO model."""

from pathlib import Path

import cv2
import numpy as np
import openvino as ov

SAM3_IMAGE_SIZE = 504
MODEL_FILENAME = "sam3_bird.xml"


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    """Resize the complete image and normalize RGB pixels for SAM 3."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or not image_bgr.size:
        raise ValueError("SAM 3 expects a nonempty HxWx3 BGR image")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE))
    normalized = resized.astype(np.float32) / 127.5 - 1.0
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float16)


class Sam3OpenVinoSegmenter:
    """Return the highest-confidence bird without a classifier box prompt."""

    def __init__(self, model_dir: Path, *, device: str = "CPU", confidence: float = 0.5) -> None:
        if not 0 <= confidence <= 1:
            raise ValueError("SAM 3 confidence must be between zero and one")
        self.model_path = model_dir / MODEL_FILENAME
        if not self.model_path.is_file() or not self.model_path.with_suffix(".bin").is_file():
            raise FileNotFoundError(
                f"SAM 3 OpenVINO model not found: {self.model_path}. "
                "Run `python -m birdspotter.ml.sam3_export`."
            )
        self.device = device
        self.confidence = confidence
        self.backend = ov.Core().compile_model(
            self.model_path, device, {"PERFORMANCE_HINT": "LATENCY"}
        )

    def segment(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Segment the whole image; reject absent birds and empty masks."""
        result = self.backend({"image": preprocess_image(image_bgr)})
        score = float(np.asarray(result[self.backend.output("score")]).reshape(-1)[0])
        logits = np.asarray(result[self.backend.output("mask_logits")]).squeeze()
        if not np.isfinite(score) or score < self.confidence:
            raise ValueError(f"SAM 3 found no confident bird (score={score:.3f})")
        if logits.shape != (SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE) or not np.isfinite(logits).all():
            raise ValueError("SAM 3 returned invalid mask logits")
        height, width = image_bgr.shape[:2]
        mask = cv2.resize(logits.astype(np.float32), (width, height)) > 0
        if not mask.any():
            raise ValueError("SAM 3 returned an empty bird mask")
        return mask, score

    def describe(self) -> dict[str, object]:
        return {
            "model": "SAM 3",
            "model_path": self.model_path.name,
            "prompt": "bird",
            "device": self.device,
            "input_size": SAM3_IMAGE_SIZE,
            "weight_compression": "W8A16",
            "confidence": self.confidence,
            "backend": "OpenVINO",
        }
