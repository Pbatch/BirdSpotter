"""Package the MobileNetV4 classifier and optionally upload to Hugging Face."""

import argparse
import shutil
from pathlib import Path

import openvino as ov
import torch

from birdspotter.ml.classifier import CLASSES, MODEL_NAME, MODEL_VARIANTS
from birdspotter.ml.hf_packaging import (
    add_upload_arguments,
    prepare_output,
    upload_package,
    write_manifest,
)
from birdspotter.weights import classifier_path, default_weights_dir

REPOSITORY = "PBatch23888/birdspotter-mobilenetv4"


def package_classifier(openvino_dir: Path, output: Path, checkpoint: Path | None = None) -> None:
    for name in ("classifier.xml", "classifier.bin"):
        if not (openvino_dir / name).is_file():
            raise FileNotFoundError(openvino_dir / name)
    model = ov.Core().read_model(openvino_dir / "classifier.xml")
    if not model.input(0).partial_shape.is_static or not model.output(0).partial_shape.is_static:
        raise ValueError("Classifier must have static input/output shapes")
    shape = tuple(model.input(0).shape)
    if (
        len(shape) != 4
        or shape[0] != 1
        or shape[3] != 3
        or model.input(0).element_type != ov.Type.u8
        or tuple(model.output(0).shape) != (1, 1)
    ):
        raise ValueError("Expected a batch-one uint8 NHWC pixel input and a single bird logit")
    height, width = shape[1:3]
    model_name = MODEL_NAME
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            state["model_name"] not in MODEL_VARIANTS.values()
            or state["classes"] != CLASSES
            or (height, width) != (state["image_size"], state["image_size"])
        ):
            raise ValueError("Checkpoint metadata does not match the OpenVINO classifier")
        model_name = state["model_name"]
    prepare_output(output, openvino_dir)
    artifacts = output / "openvino"
    artifacts.mkdir()
    for name in ("classifier.xml", "classifier.bin"):
        shutil.copy2(openvino_dir / name, artifacts / name)
    if checkpoint is not None:
        shutil.copy2(checkpoint, output / "best.pt")
    (output / "README.md").write_text(f"""---
library_name: openvino
pipeline_tag: image-classification
license: apache-2.0
tags:
  - mobilenetv4
  - birds
  - openvino
---

# BirdSpotter MobileNetV4 classifier

MobileNetV4 convolutional classifier fine-tuned for whole-frame bird classification.
Input: batch-one BGR pixels, 1 x {height} x {width} x 3 (NHWC) uint8, as OpenCV
decodes them. Resize with aspect-ratio preserving letterboxing, padding 114.
The graph converts to RGB and applies the ImageNet normalisation itself
(mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]).
Output: one raw bird logit; apply sigmoid. Bird-positive frames meet the default
0.5 probability threshold and can be passed whole to SAM 3.

`openvino/classifier.xml` and `.bin` are the runtime artifacts (FP16 weights).
`best.pt`, when provided, is the lightweight PyTorch export checkpoint.
`manifest.json` contains artifact sizes, checksums and the model contract.
Install with BirdSpotter's `scripts/download_models.py --classifier-repository
OWNER/REPOSITORY` (on one line) for public repositories.

Derived from [timm/{model_name}](https://huggingface.co/timm/{model_name}),
whose model card specifies Apache-2.0 licensing. Preserve upstream attribution
and licence terms when redistributing.
""")
    write_manifest(
        output,
        {
            "model": model_name,
            "task": "image-classification",
            "classes": CLASSES,
            "image_size": [height, width],
            "input": "uint8 BGR NHWC, letterboxed with padding 114, normalised in-graph",
            "output": "bird_logit",
            "activation": "sigmoid",
            "confidence_threshold": 0.5,
            "precision": "FP16 weights",
        },
    )
    print(f"Packaged classifier: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openvino-dir", type=Path, default=classifier_path(default_weights_dir()))
    parser.add_argument(
        "--checkpoint", type=Path, help="Optional matching best.pt export checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dist/huggingface/birdspotter-mobilenetv4")
    )
    add_upload_arguments(parser, REPOSITORY)
    args = parser.parse_args()
    if not args.upload_only:
        package_classifier(
            args.openvino_dir.resolve(),
            args.output_dir.resolve(),
            args.checkpoint.resolve() if args.checkpoint else None,
        )
    if args.upload or args.upload_only:
        upload_package(
            args.output_dir.resolve(), args.repo_id, repo_type="model", private=args.private
        )


if __name__ == "__main__":
    main()
