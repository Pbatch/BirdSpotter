"""Inference backends for fixed-prompt SAM 3, chosen from the exported model files."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from birdspotter.segmentation.openvino_backend import Sam3OpenVinoSegmenter
from birdspotter.segmentation.sam3 import ENGINE_FILENAME


class Segmenter(Protocol):
    def segment(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]: ...

    def describe(self) -> dict[str, object]: ...


def sam3_segmenter(model_dir: Path, *, confidence: float = 0.5) -> Segmenter:
    """Pick TensorRT for a directory holding an engine, otherwise OpenVINO."""

    if (model_dir / ENGINE_FILENAME).is_file():
        # TensorRT comes from the optional `tensorrt` dependency group, so it is only
        # imported for an engine and the OpenVINO-only Beelink deploy never loads it.
        from birdspotter.segmentation.tensorrt_backend import (  # noqa: PLC0415
            Sam3TensorRtSegmenter,
        )

        return Sam3TensorRtSegmenter(model_dir, confidence=confidence)
    return Sam3OpenVinoSegmenter(model_dir, confidence=confidence)
