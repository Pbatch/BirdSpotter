# Optional ML libraries are checked before importing the packagers.
# ruff: noqa: E402

import json
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest

Image = pytest.importorskip("PIL.Image")
ov = pytest.importorskip("openvino")
pytest.importorskip("torch")

from birdspotter.ml import hf_packaging
from scripts.download_models import model_files
from scripts.ml.package_hf_classifier import package_classifier
from scripts.ml.package_hf_dataset import package_dataset
from scripts.ml.package_hf_sam3 import package_sam3


def test_dataset_package_preserves_labels_and_excludes_cache(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    for split in ("train", "val"):
        for label in ("bird", "no_bird"):
            folder = source / split / label
            folder.mkdir(parents=True)
            Image.new("RGB", (640, 640), "red").save(folder / "example.jpg")
    (source / ".metadata").mkdir()
    (source / ".metadata" / "cache.txt").write_text("do not package")
    output = tmp_path / "package"
    package_dataset(source, output)
    with tarfile.open(output / "data/classifier_birds.tar.gz") as archive:
        assert sorted(archive.getnames()) == [
            "train/bird/example.jpg",
            "train/no_bird/example.jpg",
            "val/bird/example.jpg",
            "val/no_bird/example.jpg",
        ]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["image_count"] == 4
    assert manifest["targets"] == {"bird": 1, "no_bird": 0}
    assert manifest["image_sizes"] == [{"width": 640, "height": 640, "count": 4}]
    with pytest.raises(FileExistsError):
        package_dataset(source, output)


def test_model_packages_match_downloader(tmp_path: Path) -> None:
    source = tmp_path / "classifier"
    source.mkdir()
    pixels = ov.opset13.parameter([1, 3, 640, 640], ov.Type.f32)
    logits = ov.opset13.reduce_mean(pixels, ov.opset13.constant([1, 2, 3]), keep_dims=False)
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


def test_dataset_package_refuses_private_synthetic_images(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    for split in ("train", "val"):
        for label in ("bird", "no_bird"):
            folder = source / split / label
            folder.mkdir(parents=True)
            Image.new("RGB", (640, 640), "red").save(folder / "example.jpg")
    Image.new("RGB", (640, 640), "red").save(source / "train/bird/synthetic_run_000000.jpg")
    with pytest.raises(ValueError, match="private synthetic"):
        package_dataset(source, tmp_path / "package")
    assert not (tmp_path / "package").exists()
