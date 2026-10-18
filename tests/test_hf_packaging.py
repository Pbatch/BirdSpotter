# Optional ML libraries are checked before importing the packagers.
# ruff: noqa: E402

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

Image = pytest.importorskip("PIL.Image")
ov = pytest.importorskip("openvino")
pytest.importorskip("torch")
pq = pytest.importorskip("pyarrow.parquet")

from birdspotter.ml import hf_packaging
from scripts.download_models import model_files
from scripts.ml.package_hf_classifier import package_classifier
from scripts.ml.package_hf_dataset import package_dataset
from scripts.ml.package_hf_sam3 import package_sam3


def test_dataset_package_preserves_labels_and_provenance(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    rows = []
    for split in ("train", "val"):
        for label, target in (("bird", 1), ("no_bird", 0)):
            folder = source / split / label
            folder.mkdir(parents=True)
            Image.new("RGB", (640, 640), "red").save(folder / "example.jpg")
            rows.append(
                {
                    "image": f"{split}/{label}/example.jpg",
                    "split": split,
                    "source": "coco2017",
                    "source_id": f"{split}-{label}",
                    "original": "train2017",
                    "objects": target,
                    "label": label,
                    "target": target,
                }
            )
    rows[0] |= {"label": "no_bird", "target": 0, "relabelled_from": "bird"}
    rows[0]["image"] = "train/no_bird/relabelled.jpg"
    Image.new("RGB", (640, 640), "blue").save(source / rows[0]["image"])
    (source / "train/bird/example.jpg").unlink()
    (source / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (source / ".metadata").mkdir()
    (source / ".metadata" / "cache.txt").write_text("do not package")
    output = tmp_path / "package"
    package_dataset(source, output)
    table = pq.read_table(output / "data/classifier_birds.parquet")
    assert table["path"].to_pylist() == [
        "train/no_bird/example.jpg",
        "train/no_bird/relabelled.jpg",
        "val/bird/example.jpg",
        "val/no_bird/example.jpg",
    ]
    assert table["target"].to_pylist() == [0, 0, 1, 0]
    assert table["relabelled_from"].to_pylist() == [None, "bird", None, None]
    assert table["jpeg"][2].as_py() == (source / "val/bird/example.jpg").read_bytes()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["image_count"] == 4
    assert manifest["targets"] == {"no_bird": 0, "bird": 1}
    assert manifest["counts"] == {"train": {"no_bird": 2}, "val": {"bird": 1, "no_bird": 1}}
    assert manifest["sources"]["val"] == {"coco2017": 2}
    assert manifest["image_sizes"] == [{"width": 640, "height": 640, "count": 4}]
    assert {entry["path"] for entry in manifest["files"]} == {
        "README.md",
        "data/classifier_birds.parquet",
    }
    with pytest.raises(FileExistsError):
        package_dataset(source, output)
    Image.new("RGB", (640, 640), "red").save(source / "val/bird/unlisted.jpg")
    with pytest.raises(ValueError, match="not in the manifest"):
        package_dataset(source, tmp_path / "unlisted")


def test_model_packages_match_downloader(tmp_path: Path) -> None:
    source = tmp_path / "classifier"
    source.mkdir()
    pixels = ov.opset13.parameter([1, 640, 640, 3], ov.Type.u8)
    values = ov.opset13.convert(pixels, ov.Type.f32)
    logits = ov.opset13.reduce_mean(values, ov.opset13.constant([1, 2, 3]), keep_dims=False)
    logits = ov.opset13.unsqueeze(logits, ov.opset13.constant([1]))
    ov.save_model(ov.Model([logits], [pixels]), source / "classifier.xml")
    output = tmp_path / "classifier-package"
    package_classifier(source, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["image_size"] == [640, 640]
    assert {entry["relative_path"] for entry in model_files(manifest, "openvino/")} == {
        "classifier.xml",
        "classifier.bin",
    }
    sam = tmp_path / "sam"
    sam.mkdir()
    for name in ("sam3_bird.xml", "sam3_bird.bin", "sam3_bird.json"):
        (sam / name).write_text("fixture")
    sam_output = tmp_path / "sam-package"
    package_sam3(sam, sam_output)
    sam_manifest = json.loads((sam_output / "manifest.json").read_text())
    assert len(model_files(sam_manifest, "openvino-504/")) == 3


def test_upload_uses_repo_type_and_checksums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "README.md").write_text("package")
    hf_packaging.write_manifest(tmp_path, {"format": "imagefolder"})
    api = Mock()
    monkeypatch.setattr(hf_packaging, "HfApi", lambda: api)
    hf_packaging.upload_package(tmp_path, "owner/birds", repo_type="dataset", private=True)
    api.create_repo.assert_called_once_with(
        repo_id="owner/birds",
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    assert api.upload_folder.call_args.kwargs["repo_type"] == "dataset"
    (tmp_path / "README.md").write_text("changed")
    with pytest.raises(ValueError, match="changed or missing"):
        hf_packaging.upload_package(tmp_path, "owner/birds", repo_type="dataset")
    assert api.upload_folder.call_count == 1
