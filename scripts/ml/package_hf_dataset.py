"""Package the classifier dataset as one Parquet file and optionally upload to Hugging Face."""

import argparse
import io
import json
import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from birdspotter.ml.hf_packaging import (
    add_upload_arguments,
    prepare_output,
    upload_package,
    write_manifest,
)

REPOSITORY = "PBatch23888/birds-classification"
ARCHIVE_NAME = "classifier_birds.parquet"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".ppm", ".pgm"}
TARGETS = {"no_bird": 0, "bird": 1}
ROW_GROUP_SIZE = 1024
SCHEMA = pa.schema(
    [
        ("path", pa.string()),
        ("split", pa.string()),
        ("label", pa.string()),
        ("target", pa.int8()),
        ("source", pa.string()),
        ("source_id", pa.string()),
        ("original", pa.string()),
        ("objects", pa.int32()),
        ("relabelled_from", pa.string()),
        ("jpeg", pa.binary()),
    ]
)


def read_manifest(dataset: Path) -> list[dict[str, object]]:
    """Load manifest.jsonl and check it describes exactly the images on disk."""
    rows = [
        json.loads(line)
        for line in (dataset / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    known = set(SCHEMA.names) - {"path", "jpeg"} | {"image"}
    for row in rows:
        if unknown := set(row) - known:
            raise ValueError(f"Manifest fields {sorted(unknown)} have no Parquet column: {row}")
        split, label, *_ = Path(str(row["image"])).parts
        if (split, label) != (row["split"], row["label"]) or TARGETS.get(label) != row["target"]:
            raise ValueError(f"Manifest split, label and target disagree: {row}")
    listed = {str(row["image"]) for row in rows}
    if len(listed) != len(rows):
        raise ValueError("Manifest lists an image more than once")
    on_disk = {
        path.relative_to(dataset).as_posix()
        for split in {str(row["split"]) for row in rows}
        for path in (dataset / split).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if missing := sorted(listed - on_disk):
        raise FileNotFoundError(f"{len(missing)} manifest images are missing, e.g. {missing[0]}")
    if unlisted := sorted(on_disk - listed):
        raise ValueError(f"{len(unlisted)} images are not in the manifest, e.g. {unlisted[0]}")
    return sorted(rows, key=lambda row: (row["split"], row["image"]))


def package_dataset(dataset: Path, output: Path) -> None:
    rows = read_manifest(dataset)
    if not rows:
        raise ValueError(f"No images in {dataset / 'manifest.jsonl'}")
    counts: Counter[tuple[str, str]] = Counter()
    sources: Counter[tuple[str, str]] = Counter()
    sizes: Counter[tuple[int, int]] = Counter()
    prepare_output(output, dataset)
    archive = output / "data" / ARCHIVE_NAME
    archive.parent.mkdir()
    # JPEG bytes are already compressed; zstd only pays off on the metadata columns.
    compression = dict.fromkeys(SCHEMA.names, "zstd") | {"jpeg": "none"}
    with pq.ParquetWriter(archive, SCHEMA, compression=compression) as writer:
        for start in tqdm(range(0, len(rows), ROW_GROUP_SIZE), desc="Packaging", unit="group"):
            group = []
            for row in rows[start : start + ROW_GROUP_SIZE]:
                encoded = (dataset / str(row["image"])).read_bytes()
                with Image.open(io.BytesIO(encoded)) as image:
                    if image.format != "JPEG":
                        raise ValueError(f"{row['image']} is {image.format}, not JPEG")
                    sizes[image.size] += 1
                counts[str(row["split"]), str(row["label"])] += 1
                sources[str(row["split"]), str(row["source"])] += 1
                group.append(
                    {"path": row["image"], "relabelled_from": None, **row, "jpeg": encoded}
                )
            writer.write_table(pa.Table.from_pylist(group, SCHEMA))
    if (dataset / "build-config.json").is_file():
        shutil.copy2(dataset / "build-config.json", output / "build-config.json")
    (output / "README.md").write_text("""---
pretty_name: BirdSpotter Bird Classification
task_categories:
  - image-classification
tags:
  - birds
---

# BirdSpotter classification dataset

Whole-frame bird/no-bird classification. `target` is bird=1 and no_bird=0.

`data/classifier_birds.parquet` has one row per image: the 640x640 letterboxed
JPEG's raw bytes in `jpeg`, its `split` (train or val), `label` and `target`,
and its provenance (`source`, `source_id`, `original` source split, annotated
bird `objects`, and `relabelled_from` for manually corrected labels). The file
can be uploaded directly to the BirdSpotter Modal training volume.
`manifest.json` contains split, class and source counts, image dimensions and
checksums.

Download with `hf download OWNER/REPOSITORY data/classifier_birds.parquet
--repo-type dataset --local-dir ./downloaded` (on one line).

This is a compilation of upstream images. Their respective licences and terms
continue to apply; consult the source provenance before redistribution.
""")
    metadata = {
        "format": "parquet",
        "targets": TARGETS,
        "image_count": len(rows),
        "counts": {
            split: {label: n for (s, label), n in sorted(counts.items()) if s == split}
            for split in sorted({split for split, _ in counts})
        },
        "sources": {
            split: {source: n for (s, source), n in sorted(sources.items()) if s == split}
            for split in sorted({split for split, _ in sources})
        },
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
