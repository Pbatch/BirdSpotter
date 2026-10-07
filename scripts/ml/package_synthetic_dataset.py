"""Package synthetic bird/no_bird pairs as a private, train-only ImageFolder tar.

The tar is meant for the birdspotter-training Modal volume and is passed to training with
--extra-dataset-tars. It is deliberately not built from data/processed/classifier_birds, so
package_hf_dataset.py can never publish these generations.

Every pair goes to train/: frames from one camera a few minutes apart are near duplicates,
so a held-out synthetic split would leak, and val stays comparable with earlier runs.
"""

import argparse
import io
import json
import tarfile
from pathlib import Path

from PIL import Image

SOURCE = "synthetic"
JPEG_QUALITY = 92  # matches build_source_classifier_dataset.py


def completed_runs(root: Path) -> list[Path]:
    """Return run directories whose build-config.json marks a finished generation."""
    runs = sorted(path.parent for path in root.glob("*/build-config.json"))
    if not runs:
        raise FileNotFoundError(f"No completed synthetic runs under {root}")
    return runs


def package(root: Path, archive: Path) -> dict[str, int]:
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".part")
    counts = {"bird": 0, "no_bird": 0}
    manifest = []
    with tarfile.open(temporary, "w:gz") as bundle:
        for run in completed_runs(root):
            for line in (run / "manifest.jsonl").read_text().splitlines():
                record = json.loads(line)
                label = record["label"]
                name = f"{SOURCE}_{run.name}_{record['index']:06d}.jpg"
                arcname = f"train/{label}/{name}"
                data = io.BytesIO()
                with Image.open(run / record["path"]) as image:
                    image.convert("RGB").save(data, "JPEG", quality=JPEG_QUALITY)
                info = tarfile.TarInfo(arcname)
                info.size = data.tell()
                data.seek(0)
                bundle.addfile(info, data)
                counts[label] += 1
                manifest.append(
                    {
                        "image": arcname,
                        "split": "train",
                        "source": SOURCE,
                        "source_id": f"{run.name}/{record['index']}",
                        "original": str(run / record["path"]),
                        "label": label,
                        "target": int(label == "bird"),
                        "bird_prompt": record["bird_prompt"],
                    }
                )
        data = "".join(json.dumps(record) + "\n" for record in manifest).encode()
        info = tarfile.TarInfo("synthetic-manifest.jsonl")
        info.size = len(data)
        bundle.addfile(info, io.BytesIO(data))
    temporary.replace(archive)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-dir", type=Path, default=Path("data/synthetic/segmented"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/synthetic_segmented.tar.gz")
    )
    args = parser.parse_args()
    counts = package(args.synthetic_dir, args.output)
    print(f"Packaged {counts} into {args.output}")


if __name__ == "__main__":
    main()
