"""Run an OpenVINO classifier export on the CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openvino as ov


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
        cache_dir = Path(__file__).resolve().parents[2] / ".cache" / "openvino" / model_path.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache_dir)})
        model = core.read_model(model_files[0])
        model_input = model.input(0)
        if not model_input.partial_shape.is_static:
            raise TypeError("Classifier must have a fixed input shape")
        self.input_shape = tuple(model_input.shape)
        self.input_dtype = np.dtype(model_input.element_type.to_dtype())
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
