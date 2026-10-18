"""Export fixed-prompt SAM 3 at 504x504 as a W8A16 OpenVINO model or FP16 TensorRT engine.

Run with an ML environment containing torch, transformers, openvino and nncf::

    python -m birdspotter.ml.sam3_export --output-dir weights/sam3/openvino-504

Add ``--tensorrt`` (and the ``tensorrt`` dependency group) to build a TensorRT
engine for this machine's NVIDIA GPU instead; see ``export_sam3_tensorrt``.

Validated dependency versions: transformers 5.18.0, OpenVINO 2026.2.1,
NNCF 3.2.0, NumPy 2.4.2, and PyTorch 2.13.0. Transformers is export-only.
Use ``--device cuda`` to speed up tracing on an NVIDIA GPU; the exported
OpenVINO model has no CUDA dependency.

The default prompt is ``bird``. Its text features are baked into the graph, so
there is no tokenizer or text encoder at runtime. Input is a raw uint8 BGR
frame in NHWC layout with batch size one and any height and width, as OpenCV
decodes it. The graph converts it to RGB, resizes the whole frame to 504x504
and normalizes it with mean=std=0.5. SAM 3 uses 14-pixel patches. Global
rotary positions are rebuilt for 504x504.
Outputs are the highest-confidence object's mask logits (1x1x504x504), its
normalized xyxy box (1x4), and its presence-adjusted score (1). Threshold mask
logits at zero and reject scores below your bird-presence threshold.

W8A16 means symmetric INT8 weight compression with FP16 floating-point tensors,
not calibrated INT8 activations. OpenVINO may promote operations according to
the target device. Reduced resolution and weight compression need validation
on representative bird images before deployment.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import nncf
import openvino as ov
import torch
from openvino.passes import ConstantFolding, Manager
from transformers import Sam3Model, Sam3Processor
from transformers.modeling_outputs import BaseModelOutputWithPooling

from birdspotter.segmentation.openvino_backend import embed_preprocessing
from birdspotter.segmentation.sam3 import ENGINE_FILENAME, MODEL_FILENAME
from birdspotter.weights import default_weights_dir, sam3_openvino_dir, sam3_tensorrt_dir

INPUT_SIZE = 504


class Sam3BirdModel(torch.nn.Module):
    """Bake a concept prompt into SAM 3 and return only its best object."""

    text_features: torch.Tensor
    text_mask: torch.Tensor

    def __init__(
        self, model: Sam3Model, text_features: torch.Tensor, text_mask: torch.Tensor
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("text_features", text_features)
        self.register_buffer("text_mask", text_mask)
        # The frozen text features replace this large module entirely.
        self.model.text_encoder = torch.nn.Identity()  # ty: ignore[invalid-assignment]
        self.model.text_projection = torch.nn.Identity()  # ty: ignore[invalid-assignment]

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model(
            pixel_values=image,
            text_embeds=BaseModelOutputWithPooling(
                pooler_output=cast("torch.FloatTensor", self.text_features)
            ),
            attention_mask=self.text_mask,
            return_dict=True,
        )
        scores = output.pred_logits.sigmoid() * output.presence_logits.sigmoid()
        best = scores.argmax(dim=1)
        masks = output.pred_masks[0, best].unsqueeze(0)
        masks = torch.nn.functional.interpolate(
            masks, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False
        )
        return masks, output.pred_boxes[0, best], scores[0, best]


class PixelInputSam3(torch.nn.Module):
    """Normalise inside the graph so TensorRT inference only sends resized BGR pixels.

    Takes a uint8 BGR batch shaped 1x504x504x3, already resized on the host, and applies
    the OpenVINO export's embedded RGB conversion and [-1, 1] normalisation.
    """

    def __init__(self, model: Sam3BirdModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TensorRT only accepts uint8 into a Cast, so convert before reordering channels.
        rgb = pixels.to(torch.float32)[..., [2, 1, 0]].permute(0, 3, 1, 2)
        return self.model((rgb / 127.5 - 1).to(torch.float16))


def configure_resolution(model: Sam3Model) -> None:
    """Rebuild global RoPE positions without changing pretrained weights."""
    backbone = model.vision_encoder.backbone
    backbone.config.image_size = INPUT_SIZE
    for layer in backbone.layers:
        layer.config.image_size = INPUT_SIZE
        layer.position_ids = layer.precompute_positions(layer.config, layer.window_size).to(
            next(layer.parameters()).device
        )


def load_sam3_bird(
    checkpoint: str, prompt: str, cache_dir: Path | None, device: str
) -> Sam3BirdModel:
    """Load SAM 3 at 504x504 in FP16 with the prompt's text features baked in."""
    if not prompt.strip():
        raise ValueError("The concept prompt must not be empty")
    model = Sam3Model.from_pretrained(
        checkpoint,
        cache_dir=cache_dir,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    processor = Sam3Processor.from_pretrained(checkpoint, cache_dir=cache_dir)
    configure_resolution(model)
    with torch.inference_mode():
        text = processor(text=prompt, return_tensors="pt")
        features = model.get_text_features(**text).pooler_output
    return Sam3BirdModel(model, features, text["attention_mask"]).eval().half().to(device)


def export_sam3_openvino(
    model_dir: Path,
    *,
    checkpoint: str = "facebook/sam3",
    prompt: str = "bird",
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> Path:
    """Export a fixed concept prompt with embedded preprocessing and INT8 weight compression."""
    wrapper = load_sam3_bird(checkpoint, prompt, cache_dir, device)
    model_dir.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float16, device=device)
    print("Converting static 504x504 SAM 3 graph", flush=True)
    # Shape and prompt are fixed; avoid repeating expensive full-model tracing.
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example, check_trace=False, strict=False).eval()
    converted = ov.convert_model(traced.cpu(), input=[example.shape])
    converted.input().get_tensor().set_names({"image"})
    for port, name in zip(converted.outputs, ("mask_logits", "box", "score"), strict=True):
        port.get_tensor().set_names({name})
    # Fold static positional/shape calculations before compression, so the
    # pass cannot expand the compressed INT8 weights back to floating point.
    passes = Manager()
    passes.register_pass(ConstantFolding())
    passes.run_passes(converted)
    for index, node in enumerate(converted.get_ordered_ops()):
        node.set_friendly_name(f"{index}_{node.get_friendly_name()}")
    print("Compressing weights to symmetric INT8 (W8A16)", flush=True)
    compressed = nncf.compress_weights(converted, mode=nncf.CompressWeightsMode.INT8_SYM)
    compressed = embed_preprocessing(compressed)
    model_path = model_dir / MODEL_FILENAME
    ov.save_model(compressed, model_path, compress_to_fp16=True)
    metadata = {
        "checkpoint": checkpoint,
        "prompt": prompt,
        "input_shape": [1, -1, -1, 3],
        "input_dtype": "uint8",
        "input_layout": "NHWC",
        "input_color": "BGR",
        "model_input_size": INPUT_SIZE,
        "embedded_preprocessing": True,
        "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
        "weight_compression": "INT8_SYM",
        "activation_quantization": False,
        "outputs": ["mask_logits", "box", "score"],
        "versions": {
            "torch": torch.__version__,
            "openvino": ov.__version__,
            "nncf": nncf.__version__,
        },
    }
    model_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {model_path}", flush=True)
    return model_path


@contextmanager
def float32_ranges() -> Iterator[None]:
    """Trace half-precision ``torch.arange`` calls as a float32 Range and a Cast.

    SAM 3's sine position encodings build FP16 ranges, and TensorRT only supports
    int32, int64 and float32 Range inputs. The positions are small integers, so the
    cast back to FP16 is exact.
    """
    original = torch.arange

    def arange(*args: Any, dtype: torch.dtype | None = None, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        if dtype in {torch.float16, torch.bfloat16}:
            return original(*args, dtype=torch.float32, **kwargs).to(dtype)
        return original(*args, dtype=dtype, **kwargs)

    with patch.object(torch, "arange", arange):
        yield


def export_sam3_tensorrt(
    model_dir: Path,
    *,
    checkpoint: str = "facebook/sam3",
    prompt: str = "bird",
    cache_dir: Path | None = None,
) -> Path:
    """Build an FP16 TensorRT engine for this machine's NVIDIA GPU.

    The engine takes one uint8 BGR frame already resized to 1x504x504x3 (NHWC) and
    returns the same outputs as the OpenVINO export. TensorRT 11 networks are strongly
    typed, so the FP16 ONNX graph sets the precision; weights are not INT8-compressed.
    Engines only run on the GPU and TensorRT version they were built with.
    """
    import tensorrt as trt  # noqa: PLC0415 -- optional `tensorrt` dependency group

    wrapper = PixelInputSam3(load_sam3_bird(checkpoint, prompt, cache_dir, "cuda")).eval()
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = model_dir / "sam3_bird.onnx"
    example = torch.zeros(1, INPUT_SIZE, INPUT_SIZE, 3, dtype=torch.uint8, device="cuda")
    print("Exporting static 504x504 SAM 3 graph to ONNX", flush=True)
    with torch.inference_mode(), float32_ranges():
        torch.onnx.export(
            wrapper,
            (example,),
            onnx_path,
            input_names=["image"],
            output_names=["mask_logits", "box", "score"],
            dynamo=False,
        )
    # Free the PyTorch weights before TensorRT needs the GPU memory for its build.
    del wrapper
    torch.cuda.empty_cache()
    print("Building FP16 TensorRT engine", flush=True)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "; ".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}: {errors}")
    engine = builder.build_serialized_network(network, builder.create_builder_config())
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path = model_dir / ENGINE_FILENAME
    engine_path.write_bytes(engine)
    metadata = {
        "checkpoint": checkpoint,
        "prompt": prompt,
        "input_shape": [1, INPUT_SIZE, INPUT_SIZE, 3],
        "input_dtype": "uint8",
        "input_layout": "NHWC",
        "input_color": "BGR",
        "model_input_size": INPUT_SIZE,
        "embedded_preprocessing": False,
        "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
        "precision": "FP16",
        "outputs": ["mask_logits", "box", "score"],
        "versions": {"torch": torch.__version__, "tensorrt": trt.__version__},
    }
    engine_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {engine_path}", flush=True)
    return engine_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", default="facebook/sam3")
    parser.add_argument("--prompt", default="bird")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--tensorrt", action="store_true", help="Build a TensorRT engine on this NVIDIA GPU"
    )
    args = parser.parse_args()
    if args.tensorrt:
        export_sam3_tensorrt(
            args.output_dir or sam3_tensorrt_dir(default_weights_dir()),
            checkpoint=args.checkpoint,
            prompt=args.prompt,
            cache_dir=args.cache_dir,
        )
        return
    export_sam3_openvino(
        args.output_dir or sam3_openvino_dir(default_weights_dir()),
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        cache_dir=args.cache_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
