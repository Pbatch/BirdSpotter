#!/usr/bin/env python3
"""Assemble the SAM21 OpenVINO artifacts into a Hugging Face upload folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

MODEL_CARD = """---
library_name: openvino
pipeline_tag: image-segmentation
license: apache-2.0
tags:
  - sam2
  - segmentation
  - openvino
---

# BirdSpotter SAM21 OpenVINO

OpenVINO conversion of Meta's SAM 2.1 Hiera Large image encoder and
box-prompted mask predictor, exported at a fixed 512 x 512 input size for
BirdSpotter.

## Files

- `openvino-512/sam21_image_encoder.xml` and `.bin`: image encoder IR.
- `openvino-512/sam21_mask_predictor.xml` and `.bin`: prompt and mask decoder IR.
- `manifest.json`: SHA-256 checksums and sizes for every packaged artifact.

These models use their normal exported precision; they are not INT8 quantized.
See the BirdSpotter repository for preprocessing and box-prompted inference code.

## Attribution

This conversion is derived from Meta's
[`facebook/sam2.1-hiera-large`](https://huggingface.co/facebook/sam2.1-hiera-large)
checkpoint. Use is subject to the upstream model's license and acceptable-use
terms.
"""


def sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_sam21(openvino_dir: Path, output_dir: Path) -> None:
    """Copy and describe the SAM21 OpenVINO artifacts."""

    expected = (
        "sam21_image_encoder.xml",
        "sam21_image_encoder.bin",
        "sam21_mask_predictor.xml",
        "sam21_mask_predictor.bin",
    )
    missing = [name for name in expected if not (openvino_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing SAM21 OpenVINO artifact: {missing[0]}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    packaged_models = output_dir / "openvino-512"
    packaged_models.mkdir(parents=True, exist_ok=True)
    for name in expected:
        shutil.copy2(openvino_dir / name, packaged_models / name)
    (output_dir / "README.md").write_text(MODEL_CARD)

    files = [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {
        "format_version": 1,
        "model": "SAM 2.1 Hiera Large",
        "image_size": [512, 512],
        "precision": "floating-point",
        "files": files,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created Hugging Face model folder: {output_dir}")
    print(f"Upload with: hf upload PBatch23888/birdspotter-sam21-openvino {output_dir} .")


def main() -> None:
    """Parse arguments and create the SAM21 model package."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--openvino-dir",
        type=Path,
        default=Path("weights/sam21/openvino-512"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/huggingface/birdspotter-sam21-openvino"),
    )
    args = parser.parse_args()
    package_sam21(args.openvino_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
