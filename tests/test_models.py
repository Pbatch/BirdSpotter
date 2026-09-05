from pathlib import Path

import pytest

from birdspotter.ml import checkpoints


def test_sam21_checkpoint_uses_ultralytics_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / checkpoints.SAM21_SOURCE_FILENAME
    source.write_bytes(b"checkpoint")
    calls: list[tuple[str, str]] = []

    def record_download(*, repo_id: str, filename: str) -> str:
        calls.append((repo_id, filename))
        return str(source)

    monkeypatch.setattr(checkpoints, "hf_hub_download", record_download)
    monkeypatch.setattr(checkpoints, "SAM21_SOURCE_SHA256", checkpoints.sha256(source))

    with checkpoints.sam21_checkpoint() as checkpoint:
        assert checkpoint.name == "sam2.1_l.pt"
        assert checkpoint.is_symlink()
        assert checkpoint.resolve() == source
        temporary_parent = checkpoint.parent

    assert not temporary_parent.exists()
    assert calls == [(checkpoints.SAM21_SOURCE_REPOSITORY, checkpoints.SAM21_SOURCE_FILENAME)]
