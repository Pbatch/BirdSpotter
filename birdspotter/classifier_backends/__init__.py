"""Inference backends for the frame classifier, chosen from the exported model files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from birdspotter.classifier_backends.openvino_backend import OpenVinoClassifierBackend


class ClassifierBackend(Protocol):
    input_shape: tuple[Any, ...]
    input_dtype: np.dtype

    def run(self, tensor: np.ndarray) -> np.ndarray: ...

    def describe(self) -> dict[str, object]: ...


def classifier_backend(model_path: Path) -> ClassifierBackend:
    """Pick TensorRT for a directory holding an engine, otherwise OpenVINO."""

    if any(model_path.glob("*.engine")):
        # TensorRT comes from the optional `tensorrt` dependency group, so it is only
        # imported for an engine and the OpenVINO-only Beelink deploy never loads it.
        from birdspotter.classifier_backends.tensorrt_backend import (  # noqa: PLC0415
            TensorRtClassifierBackend,
        )

        return TensorRtClassifierBackend(model_path)
    return OpenVinoClassifierBackend(model_path)
