"""Download and verify source checkpoints used by model exporters."""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from huggingface_hub import hf_hub_download

SAM21_ULTRALYTICS_FILENAME = "sam2.1_l.pt"
SAM21_SOURCE_REPOSITORY = "facebook/sam2.1-hiera-large"
SAM21_SOURCE_FILENAME = "sam2.1_hiera_large.pt"
SAM21_SOURCE_SHA256 = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"


def sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def sam21_checkpoint() -> Iterator[Path]:
    """Provide the cached SAM21 checkpoint under Ultralytics' required filename."""

    cached = Path(
        hf_hub_download(
            repo_id=SAM21_SOURCE_REPOSITORY,
            filename=SAM21_SOURCE_FILENAME,
        )
    )
    if sha256(cached) != SAM21_SOURCE_SHA256:
        raise RuntimeError(f"Checksum mismatch for downloaded model: {SAM21_SOURCE_FILENAME}")

    with tempfile.TemporaryDirectory(prefix="birdspotter-sam21-") as temporary:
        checkpoint = Path(temporary) / SAM21_ULTRALYTICS_FILENAME
        checkpoint.symlink_to(cached)
        yield checkpoint
