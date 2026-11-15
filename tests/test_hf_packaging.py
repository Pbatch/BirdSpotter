import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from birdspotter.ml import hf_packaging
from scripts.download_models import model_files
from scripts.ml.package_hf_sam3 import package_sam3


def test_sam3_package_matches_downloader(tmp_path: Path) -> None:
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
