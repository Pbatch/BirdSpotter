"""Run a serialized TensorRT classifier engine on the local NVIDIA GPU.

Needs the optional `tensorrt` dependency group; only import this module for an engine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from birdspotter.tensorrt_engine import TensorRtEngine


class TensorRtClassifierBackend:
    def __init__(self, model_path: Path) -> None:
        engine_files = sorted(model_path.glob("*.engine"))
        if len(engine_files) != 1:
            raise FileNotFoundError(f"Expected one TensorRT engine in {model_path}")
        self._engine = TensorRtEngine(engine_files[0])
        if len(self._engine.inputs) != 1 or len(self._engine.outputs) != 1:
            raise TypeError("Classifier engine must have one input and one output")
        self._input = self._engine.inputs[0]
        self._output = self._engine.outputs[0]
        self.input_shape = self._engine.shapes[self._input]
        self.input_dtype = self._engine.dtypes[self._input]

    def run(self, tensor: np.ndarray) -> np.ndarray:
        return self._engine.run({self._input: tensor})[self._output].astype(np.float32)

    def describe(self) -> dict[str, object]:
        return {
            "backend": "TensorRT",
            "precision": f"{self.input_dtype.name.upper()} engine",
            "device": self._engine.device,
        }
