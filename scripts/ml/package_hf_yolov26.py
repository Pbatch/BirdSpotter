#!/usr/bin/env python3
"""Package and optionally publish BirdSpotter's YOLO26-L 640x640 detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

DEFAULT_REPOSITORY = "PBatch23888/birdspotter-yolo26"

MODEL_CARD = """---
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - birds
  - object-detection
  - yolo26
  - openvino
---

# BirdSpotter YOLO26-L bird detector

YOLO26-L fine-tuned for single-class bird detection at 640 x 640 input
resolution. BirdSpotter resizes and pads camera images to this input size.

## Files

- `best.pt`: best Ultralytics/PyTorch training checkpoint.
- `openvino-int8/`: static batch-one INT8 OpenVINO IR for 640 x 640 input.
- `training/`: available training configuration, metrics, and plots.
- `manifest.json`: SHA-256 checksums and sizes for the packaged artifacts.

## Usage

### Ultralytics

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("image.jpg", imgsz=640)
```

### OpenVINO

```python
import openvino as ov

core = ov.Core()
model = core.compile_model("openvino-int8/model.xml", "AUTO")
```

The OpenVINO XML filename is recorded in `manifest.json`; replace `model.xml`
above with that filename if it differs.

## Model details

- Task: single-class object detection
- Class: `bird`
- Architecture: YOLO26-L
- Training image size: 640 x 640
- OpenVINO input shape: 1 x 3 x 640 x 640
- OpenVINO precision: INT8

Review the source datasets and their respective licenses before redistributing
training data. Model usage is also subject to the licenses of its dependencies
and base checkpoint.
"""

TRAINING_FILES = (
    "args.yaml",
    "results.csv",
    "results.png",
    "training-config.json",
)


def sha256(path: Path) -> str:
    """Calculate a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(checkpoint: Path, openvino_dir: Path, output_dir: Path) -> None:
    """Validate packaging inputs before creating the destination."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if checkpoint.suffix != ".pt":
        raise ValueError(f"Expected a .pt checkpoint: {checkpoint}")
    if not openvino_dir.is_dir():
        raise FileNotFoundError(f"OpenVINO directory not found: {openvino_dir}")
    if not any(openvino_dir.glob("*.xml")) or not any(openvino_dir.glob("*.bin")):
        raise ValueError(f"Expected .xml and .bin OpenVINO files in: {openvino_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")


def build_manifest(output_dir: Path) -> dict[str, object]:
    """Build a deterministic artifact manifest for the package."""
    files = [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    return {
        "format_version": 1,
        "model": "YOLO26l",
        "class_names": ["bird"],
        "image_size": [640, 640],
        "files": files,
    }


def package_model(
    checkpoint: Path,
    openvino_dir: Path,
    output_dir: Path,
    training_dir: Path | None,
    *,
    repository: str = DEFAULT_REPOSITORY,
) -> None:
    """Copy and describe the model artifacts in a Hub-friendly layout."""
    validate_inputs(checkpoint, openvino_dir, output_dir)
    if training_dir is not None and not training_dir.is_dir():
        raise FileNotFoundError(f"Training output directory not found: {training_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, output_dir / "best.pt")
    shutil.copytree(openvino_dir, output_dir / "openvino-int8")

    if training_dir is not None:
        packaged_training_dir = output_dir / "training"
        for filename in TRAINING_FILES:
            source = training_dir / filename
            if source.is_file():
                packaged_training_dir.mkdir(exist_ok=True)
                shutil.copy2(source, packaged_training_dir / filename)

    (output_dir / "README.md").write_text(MODEL_CARD)
    manifest = build_manifest(output_dir)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created Hugging Face model folder: {output_dir}")
    print(
        f"Upload with: hf upload {repository} {output_dir} . --repo-type model "
        "--delete 'openvino-int8/*' --delete 'training/*'"
    )


def publish_model(output_dir: Path, repository: str) -> None:
    """Publish the package and replace obsolete export and training artifacts."""
    from huggingface_hub import HfApi  # noqa: PLC0415

    commit = HfApi().upload_folder(
        repo_id=repository,
        repo_type="model",
        folder_path=output_dir,
        delete_patterns=["openvino-int8/*", "training/*"],
        commit_message="Update YOLO26-L 640x640 bird detector",
    )
    print(f"Published model: {commit.commit_url}")


def main() -> None:
    """Parse command-line arguments and create the model package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--openvino-dir", required=True, type=Path)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument(
        "--upload", action="store_true", help="Upload the completed package to Hugging Face"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/huggingface/birdspotter-yolo26"),
    )
    args = parser.parse_args()
    package_model(
        args.checkpoint.resolve(),
        args.openvino_dir.resolve(),
        args.output_dir.resolve(),
        args.training_dir.resolve() if args.training_dir is not None else None,
        repository=args.repository,
    )
    if args.upload:
        publish_model(args.output_dir.resolve(), args.repository)


if __name__ == "__main__":
    main()
