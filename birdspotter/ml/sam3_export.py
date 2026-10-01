"""Export fixed-prompt SAM 3 as a static 504x504 W8A16 OpenVINO model.

Run with an ML environment containing torch, transformers, openvino and nncf::

    python -m birdspotter.ml.sam3_export --output-dir weights/sam3/openvino-504

Validated dependency versions: transformers 5.18.0, OpenVINO 2026.2.1,
NNCF 3.2.0, NumPy 2.4.2, and PyTorch 2.13.0. Transformers is export-only.
Use ``--device cuda`` to speed up tracing on an NVIDIA GPU; the exported
OpenVINO model has no CUDA dependency.

The default prompt is ``bird``. Its text features are baked into the graph, so
there is no tokenizer or text encoder at runtime. Input is normalized RGB FP16
NCHW, with batch size one. Normalize values in [0, 1] with mean=std=0.5.
SAM 3 uses 14-pixel patches. Global rotary positions are rebuilt for 504x504.
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
from pathlib import Path
from typing import cast

import nncf
import openvino as ov
import torch
from openvino.passes import ConstantFolding, Manager
from transformers import Sam3Model, Sam3Processor
from transformers.modeling_outputs import BaseModelOutputWithPooling

INPUT_SIZE = 504
MODEL_FILENAME = "sam3_bird.xml"


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


def configure_resolution(model: Sam3Model) -> None:
    """Rebuild global RoPE positions without changing pretrained weights."""
    backbone = model.vision_encoder.backbone
    backbone.config.image_size = INPUT_SIZE
    for layer in backbone.layers:
        layer.config.image_size = INPUT_SIZE
        layer.position_ids = layer.precompute_positions(layer.config, layer.window_size).to(
            next(layer.parameters()).device
        )


def export_sam3_openvino(
    model_dir: Path,
    *,
    checkpoint: str = "facebook/sam3",
    prompt: str = "bird",
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> Path:
    """Export a fixed concept prompt with static input and INT8 weight compression."""
    if not prompt.strip():
        raise ValueError("The concept prompt must not be empty")
    model_dir.mkdir(parents=True, exist_ok=True)
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
    wrapper = Sam3BirdModel(model, features, text["attention_mask"]).eval().half().to(device)
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
    model_path = model_dir / MODEL_FILENAME
    ov.save_model(compressed, model_path, compress_to_fp16=True)
    metadata = {
        "checkpoint": checkpoint,
        "prompt": prompt,
        "input_shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
        "input_dtype": "float16",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("weights/sam3/openvino-504"))
    parser.add_argument("--checkpoint", default="facebook/sam3")
    parser.add_argument("--prompt", default="bird")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    export_sam3_openvino(
        args.output_dir,
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        cache_dir=args.cache_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
