"""Run a serialized TensorRT classifier engine on the local NVIDIA GPU.

Needs the optional `tensorrt` dependency group; only import this module for an engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cuda.bindings.runtime as cudart
import numpy as np
import tensorrt as trt


def _check(result: tuple[Any, ...]) -> Any:  # noqa: ANN401 -- cuda-bindings returns untyped tuples
    error, *values = result
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA call failed: {error}")
    return values[0] if values else None


class TensorRtClassifierBackend:
    def __init__(self, model_path: Path) -> None:
        engine_files = sorted(model_path.glob("*.engine"))
        if len(engine_files) != 1:
            raise FileNotFoundError(f"Expected one TensorRT engine in {model_path}")
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = self._runtime.deserialize_cuda_engine(engine_files[0].read_bytes())
        if engine is None:
            raise RuntimeError(f"Could not load TensorRT engine {engine_files[0]}")
        self._engine = engine
        names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
        inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(inputs) != 1 or len(outputs) != 1:
            raise TypeError("Classifier engine must have one input and one output")
        self.input_shape = tuple(engine.get_tensor_shape(inputs[0]))
        if any(dimension < 0 for dimension in self.input_shape):
            raise TypeError("Classifier must have a fixed input shape")
        self.input_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(inputs[0])))
        self._output_host = np.empty(
            tuple(engine.get_tensor_shape(outputs[0])),
            dtype=trt.nptype(engine.get_tensor_dtype(outputs[0])),
        )
        self._context = engine.create_execution_context()
        self._stream = _check(cudart.cudaStreamCreate())
        input_bytes = int(np.prod(self.input_shape)) * self.input_dtype.itemsize
        self._input_device = _check(cudart.cudaMalloc(input_bytes))
        self._output_device = _check(cudart.cudaMalloc(self._output_host.nbytes))
        self._context.set_tensor_address(inputs[0], self._input_device)
        self._context.set_tensor_address(outputs[0], self._output_device)
        properties = _check(cudart.cudaGetDeviceProperties(0))
        self.device = properties.name.decode().rstrip("\x00")

    def run(self, tensor: np.ndarray) -> np.ndarray:
        host = np.ascontiguousarray(tensor, dtype=self.input_dtype)
        if host.shape != self.input_shape:
            raise ValueError(f"Expected input shape {self.input_shape}, got {host.shape}")
        _check(
            cudart.cudaMemcpyAsync(
                self._input_device,
                host.ctypes.data,
                host.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self._stream,
            )
        )
        if not self._context.execute_async_v3(self._stream):
            raise RuntimeError("TensorRT inference failed")
        _check(
            cudart.cudaMemcpyAsync(
                self._output_host.ctypes.data,
                self._output_device,
                self._output_host.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self._stream,
            )
        )
        _check(cudart.cudaStreamSynchronize(self._stream))
        return self._output_host.astype(np.float32)

    def describe(self) -> dict[str, object]:
        return {
            "backend": "TensorRT",
            "precision": f"{self.input_dtype.name.upper()} engine",
            "device": self.device,
        }
