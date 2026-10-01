#!/usr/bin/env python3
"""Assemble the SAM3 OpenVINO artifacts into a Hugging Face upload folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from birdspotter.ml.hf_packaging import (
    add_upload_arguments,
    prepare_output,
    upload_package,
    write_manifest,
)

MODEL_CARD = """---
library_name: openvino
pipeline_tag: image-segmentation
license: other
license_name: sam-license
license_link: https://github.com/facebookresearch/sam3/blob/main/LICENSE
tags:
  - sam3
  - segmentation
  - openvino
---

# BirdSpotter SAM3 OpenVINO

OpenVINO conversion of Meta's SAM 3, exported with the fixed text prompt `bird`,
W8A16 weight compression, and a fixed 504 x 504 whole-image input size for
BirdSpotter.

## Files

- `openvino-504/sam3_bird.xml` and `.bin`: highest-confidence bird model.
- `openvino-504/sam3_bird.json`: preprocessing and export metadata.
- `manifest.json`: SHA-256 checksums and sizes for every packaged artifact.

Weights use symmetric INT8 compression with FP16 tensors; CPU execution may
promote operations. See BirdSpotter for whole-image preprocessing and inference.

## Attribution

This conversion is derived from Meta's
[`facebook/sam3`](https://huggingface.co/facebook/sam3)
checkpoint. Use is subject to the upstream model's license and acceptable-use
terms.
"""


def package_sam3(openvino_dir: Path, output_dir: Path) -> None:
    """Copy and describe the SAM3 OpenVINO artifacts."""

    expected = (
        "sam3_bird.xml",
        "sam3_bird.bin",
        "sam3_bird.json",
    )
    missing = [name for name in expected if not (openvino_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing SAM3 OpenVINO artifact: {missing[0]}")
    prepare_output(output_dir, openvino_dir)

    packaged_models = output_dir / "openvino-504"
    packaged_models.mkdir(parents=True, exist_ok=True)
    for name in expected:
        shutil.copy2(openvino_dir / name, packaged_models / name)
    (output_dir / "README.md").write_text(MODEL_CARD)

    write_manifest(output_dir, {"model": "SAM 3", "image_size": [504, 504], "precision": "W8A16"})
    print(f"Created Hugging Face model folder: {output_dir}")


def main() -> None:
    """Parse arguments and create the SAM3 model package."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--openvino-dir",
        type=Path,
        default=Path("weights/sam3/openvino-504"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/huggingface/birdspotter-sam3-openvino"),
    )
    add_upload_arguments(parser, "PBatch23888/birdspotter-sam3-openvino")
    args = parser.parse_args()
    if not args.upload_only:
        package_sam3(args.openvino_dir.resolve(), args.output_dir.resolve())
    if args.upload or args.upload_only:
        upload_package(
            args.output_dir.resolve(), args.repo_id, repo_type="model", private=args.private
        )


if __name__ == "__main__":
    main()
