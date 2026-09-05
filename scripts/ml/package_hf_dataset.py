#!/usr/bin/env python3
"""Assemble the BirdSpotter YOLO dataset into a Hugging Face upload folder."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
from pathlib import Path

REPOSITORY_ID = "PBatch23888/birds-object-detection-1600x896"
ARCHIVE_NAME = "yolo_birds_1600x896.tar.gz"

DATASET_CARD = """---
pretty_name: Birds Object Detection 1600x896
task_categories:
  - object-detection
size_categories:
  - 10K<n<100K
tags:
  - birds
  - yolo
  - ultralytics
  - object-detection
---

# Birds Object Detection 1600x896

A single-class bird object-detection dataset prepared for Ultralytics at a
1600 x 896 camera resolution. The dataset uses a deterministic 90% training
and 10% validation split.

## Contents

- `data/yolo_birds_1600x896.tar.gz`: complete Ultralytics dataset archive.
- `manifest.json`: archive size, SHA-256 checksum, and dataset metadata.

After extraction, the archive contains `data.yaml` and matching image and YOLO
label trees:

```text
data.yaml
images/train/
images/val/
labels/train/
labels/val/
```

## Download

```python
from huggingface_hub import hf_hub_download

archive = hf_hub_download(
    repo_id="PBatch23888/birds-object-detection-1600x896",
    repo_type="dataset",
    filename="data/yolo_birds_1600x896.tar.gz",
)
```

Private repositories require an authenticated Hugging Face account with
access to the dataset.

## Dataset composition

The data builder combines bird annotations from Open Images, COCO 2017,
Pascal VOC 2012, Birdsnap, and NABirds. Open Images group-of and depiction
annotations are excluded. All retained annotations are mapped to the single
class `bird`.

## Licensing

This is a compilation of multiple upstream datasets. Their respective terms
and licenses continue to apply. Review each upstream dataset's license before
redistributing or using this compilation.
"""


def sha256(path: Path) -> str:
    """Calculate a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def place_archive(source: Path, destination: Path) -> str:
    """Hard-link the archive when possible, falling back to a file copy."""
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, destination)
        return "copy"
    return "hard-link"


def package_dataset(archive: Path, output_dir: Path, image_count: int) -> None:
    """Create a Hub-friendly folder containing the compressed YOLO dataset."""
    if not archive.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive}")
    if not archive.name.endswith((".tar.gz", ".tgz")):
        raise ValueError(f"Expected a .tar.gz or .tgz archive: {archive}")
    if image_count < 1:
        raise ValueError("Image count must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    packaged_archive = data_dir / ARCHIVE_NAME
    placement = place_archive(archive, packaged_archive)
    checksum = sha256(packaged_archive)

    (output_dir / "README.md").write_text(DATASET_CARD)
    manifest = {
        "format_version": 1,
        "format": "ultralytics-yolo",
        "class_names": ["bird"],
        "image_size": [896, 1600],
        "split": {"train": 0.9, "validation": 0.1},
        "image_count": image_count,
        "archive": {
            "path": f"data/{ARCHIVE_NAME}",
            "size": packaged_archive.stat().st_size,
            "sha256": checksum,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created Hugging Face dataset folder: {output_dir}")
    print(f"Archive placement: {placement}")
    print(f"Upload with: hf upload {REPOSITORY_ID} {output_dir} . --repo-type dataset")


def main() -> None:
    """Parse command-line arguments and create the dataset package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/archives") / ARCHIVE_NAME,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/huggingface/birds-object-detection-1600x896"),
    )
    parser.add_argument("--image-count", type=int, default=80_769)
    args = parser.parse_args()
    package_dataset(args.archive.resolve(), args.output_dir.resolve(), args.image_count)


if __name__ == "__main__":
    main()
