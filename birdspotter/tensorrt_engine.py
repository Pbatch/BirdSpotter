"""Run a serialized TensorRT engine on the local NVIDIA GPU.

Needs the optional `tensorrt` dependency group; only import this module for an engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cuda.bindings.runtime as cudart
import numpy as np
import tensorrt as trt

# TensorRT keeps one process-wide logger, so every runtime must share it.
LOGGER = trt.Logger(trt.Logger.WARNING)


def check_cuda(result: tuple[Any, ...]) -> Any:  # noqa: ANN401 -- cuda-bindings returns untyped tuples
    error, *values = result
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA call failed: {error}")
    return values[0] if values else None


class TensorRtEngine:
    """Hold one fixed-shape engine's device buffers and run it on named host arrays."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._runtime = trt.Runtime(LOGGER)
        engine = self._runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Could not load TensorRT engine {path}")
        self._engine = engine
        names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
        self.inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.shapes = {name: tuple(engine.get_tensor_shape(name)) for name in names}
        if any(dimension < 0 for shape in self.shapes.values() for dimension in shape):
            raise TypeError(f"TensorRT engine {path} must have fixed tensor shapes")
        self.dtypes = {name: np.dtype(trt.nptype(engine.get_tensor_dtype(name))) for name in names}
        self._context = engine.create_execution_context()
        self._stream = check_cuda(cudart.cudaStreamCreate())
        self._hosts = {
            name: np.empty(self.shapes[name], dtype=self.dtypes[name]) for name in self.outputs
        }
        self._devices = {}
        for name in names:
            nbytes = int(np.prod(self.shapes[name])) * self.dtypes[name].itemsize
            self._devices[name] = check_cuda(cudart.cudaMalloc(nbytes))
            self._context.set_tensor_address(name, self._devices[name])
        properties = check_cuda(cudart.cudaGetDeviceProperties(0))
        self.device = properties.name.decode().rstrip("\x00")

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Copy every input to the GPU, run the engine, and return copies of its outputs."""
        if set(inputs) != set(self.inputs):
            raise ValueError(f"Expected inputs {self.inputs}, got {sorted(inputs)}")
        hosts = {}
        for name, array in inputs.items():
            if array.shape != self.shapes[name] or array.dtype != self.dtypes[name]:
                raise ValueError(
                    f"Expected {name} as {self.dtypes[name]} shaped {self.shapes[name]}, "
                    f"got {array.dtype} shaped {array.shape}"
                )
            # Keep each contiguous copy alive until the stream has synchronized.
            hosts[name] = np.ascontiguousarray(array)
            check_cuda(
                cudart.cudaMemcpyAsync(
                    self._devices[name],
                    hosts[name].ctypes.data,
                    hosts[name].nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self._stream,
                )
            )
        if not self._context.execute_async_v3(self._stream):
            raise RuntimeError("TensorRT inference failed")
        for name, host in self._hosts.items():
            check_cuda(
                cudart.cudaMemcpyAsync(
                    host.ctypes.data,
                    self._devices[name],
                    host.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self._stream,
                )
            )
        check_cuda(cudart.cudaStreamSynchronize(self._stream))
        return {name: host.copy() for name, host in self._hosts.items()}
