import json
import tarfile
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")

from scripts.ml.package_synthetic_dataset import package  # noqa: E402


def write_run(root: Path, name: str, *, complete: bool) -> None:
    run = root / name
    records = []
    for label in ("bird", "no_bird"):
        (run / label).mkdir(parents=True)
        Image.new("RGB", (640, 640), "green").save(run / label / "000000.png")
        records.append(
            {"index": 0, "label": label, "path": f"{label}/000000.png", "bird_prompt": "p"}
        )
    (run / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    if complete:
        (run / "build-config.json").write_text("{}")


def test_package_is_train_only_jpeg_and_skips_incomplete_runs(tmp_path: Path) -> None:
    root = tmp_path / "synthetic"
    write_run(root, "frame-a-n1-s0", complete=True)
    write_run(root, "frame-b-n1-s1", complete=False)
    archive = tmp_path / "out.tar.gz"
    assert package(root, archive) == {"bird": 1, "no_bird": 1}
    with tarfile.open(archive) as bundle:
        names = sorted(bundle.getnames())
        assert names == [
            "synthetic-manifest.jsonl",
            "train/bird/synthetic_frame-a-n1-s0_000000.jpg",
            "train/no_bird/synthetic_frame-a-n1-s0_000000.jpg",
        ]
        image = bundle.extractfile("train/bird/synthetic_frame-a-n1-s0_000000.jpg")
        assert image is not None
        assert Image.open(image).format == "JPEG"
