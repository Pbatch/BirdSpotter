from pathlib import Path

import pytest

from birdspotter import models


def test_prepare_sam21_checkpoint_downloads_to_development_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, Path, str]] = []

    def record_download(url: str, destination: Path, expected_sha256: str) -> None:
        calls.append((url, destination, expected_sha256))

    monkeypatch.setattr(models, "download", record_download)

    checkpoint = models.prepare_sam21_checkpoint(tmp_path / "weights")

    assert checkpoint == tmp_path / "weights_dev" / "sam2.1" / "sam2.1_l.pt"
    assert calls == [(models.SAM21_SOURCE_URL, checkpoint, models.SAM21_SOURCE_SHA256)]
