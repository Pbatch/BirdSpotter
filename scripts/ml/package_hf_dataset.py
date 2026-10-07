"""Package a classification ImageFolder dataset and optionally upload to Hugging Face."""

import argparse
import shutil
import tarfile
from collections import Counter
from pathlib import Path

from PIL import Image

from birdspotter.ml.hf_packaging import (
    add_upload_arguments,
    prepare_output,
    upload_package,
    write_manifest,
)

REPOSITORY = "PBatch23888/birds-classification"
ARCHIVE_NAME = "classifier_birds.tar.gz"
# Synthetic generations stay private; see package_synthetic_dataset.py.
PRIVATE_PREFIX = "synthetic_"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".ppm", ".pgm"}


def package_dataset(dataset: Path, output: Path) -> None:  # noqa: C901
    counts: dict[str, dict[str, int]] = {}
    sizes: Counter[tuple[int, int]] = Counter()
    images = []
    for split in ("train", "val", "test"):
        directory = dataset / split
        if split == "test" and not directory.exists():
            continue
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        classes = {path.name for path in directory.iterdir() if path.is_dir()}
        if classes != {"bird", "no_bird"}:
            raise ValueError(f"Expected bird/ and no_bird/ directories in {directory}")
        counts[split] = {}
        for label in sorted(classes):
            paths = sorted(
                path
                for path in (directory / label).rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not paths:
                raise ValueError(f"No images in {directory / label}")
            if private := [path for path in paths if path.name.startswith(PRIVATE_PREFIX)]:
                raise ValueError(f"Refusing to package private synthetic images, e.g. {private[0]}")
            counts[split][label] = len(paths)
            for path in paths:
                with Image.open(path) as image:
                    sizes[image.size] += 1
            images.extend(paths)
    prepare_output(output, dataset)
    archive = output / "data" / ARCHIVE_NAME
    archive.parent.mkdir()
    with tarfile.open(archive, "w:gz") as bundle:
        for image in images:
            bundle.add(image, arcname=image.relative_to(dataset).as_posix(), recursive=False)
        for name in ("manifest.jsonl", "build-config.json"):
            source = dataset / name
            if source.is_file():
                bundle.add(source, arcname=name)
                shutil.copy2(source, output / name)
    (output / "README.md").write_text("""---
pretty_name: BirdSpotter Bird Classification
task_categories:
  - image-classification
tags:
  - birds
  - imagefolder
---

# BirdSpotter classification dataset

Whole-frame bird/no-bird classification. Training targets are bird=1 and
no_bird=0 (ImageFolder's alphabetical indices must be remapped).

`data/classifier_birds.tar.gz` contains train/ and val/ ImageFolder trees,
with bird/ and no_bird/ subdirectories, plus test/ when present. The archive
can be uploaded directly to the BirdSpotter Modal training volume.
`manifest.json` contains actual split/class counts, image dimensions and checksums.
Source build configuration and provenance manifest are included when available.
Caches and source downloads are excluded.

Download with `hf download OWNER/REPOSITORY data/classifier_birds.tar.gz
--repo-type dataset --local-dir ./downloaded` (on one line), then extract with
`tar -xzf downloaded/data/classifier_birds.tar.gz -C DESTINATION`.

This is a compilation of upstream images. Their respective licences and terms
continue to apply; consult the source provenance before redistribution.
""")
    metadata = {
        "format": "imagefolder",
        "targets": {"no_bird": 0, "bird": 1},
        "image_count": len(images),
        "counts": counts,
        "image_sizes": [
            {"width": width, "height": height, "count": count}
            for (width, height), count in sorted(sizes.items())
        ],
    }
    write_manifest(output, metadata)
    print(f"Packaged classification dataset: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/classifier_birds"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dist/huggingface/birds-classification")
    )
    add_upload_arguments(parser, REPOSITORY)
    args = parser.parse_args()
    if not args.upload_only:
        package_dataset(args.dataset_dir.resolve(), args.output_dir.resolve())
    if args.upload or args.upload_only:
        upload_package(
            args.output_dir.resolve(), args.repo_id, repo_type="dataset", private=args.private
        )


if __name__ == "__main__":
    main()
