"""Runtime model paths and verified checkpoint downloads."""

from __future__ import annotations

import hashlib
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

DETECTOR_DIRNAME = "yolo26s-bird-1088x1920-openvino-int8"
COCO_BIRD_CLASS_ID = 14
DETECTOR_BIRD_CLASS_ID = 0
DETECTOR_BIRD_CLASS_NAME = "bird"
SAM21_FILENAME = "sam2.1_l.pt"
SAM21_SOURCE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM21_SOURCE_SHA256 = "ab7e1ac9cb9f6eb3bcf197ece044f06a707ec49129361a2b47e93e1db6989efd"
SAM21_OPENVINO_DIRNAME = "openvino-512"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_weights_dir() -> Path:
    return project_root() / "weights"


def development_weights_dir(weights_dir: Path) -> Path:
    """Return the sibling directory used for source checkpoints and export inputs."""

    return weights_dir.with_name("weights_dev")


def detector_path(weights_dir: Path) -> Path:
    return weights_dir / "detector" / DETECTOR_DIRNAME


def sam21_path(weights_dir: Path) -> Path:
    return development_weights_dir(weights_dir) / "sam2.1" / SAM21_FILENAME


def sam21_openvino_dir(weights_dir: Path) -> Path:
    return weights_dir / "sam2.1" / SAM21_OPENVINO_DIRNAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str) -> None:
    """Download one artifact with a temporary file and atomic rename."""

    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError("Model downloads require HTTPS")
    if destination.is_file():
        if sha256(destination) != expected_sha256:
            raise RuntimeError(f"Checksum mismatch for existing model: {destination}")
        print(f"Already present: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    request = urllib.request.Request(  # noqa: S310 -- URL scheme validated above
        url, headers={"User-Agent": "BirdSpotter/0.1"}
    )
    print(f"Downloading {destination.name} ...")
    try:
        with (
            urllib.request.urlopen(  # noqa: S310 -- URL scheme validated above
                request, timeout=120
            ) as response,
            partial.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if sha256(partial) != expected_sha256:
            raise RuntimeError(f"Checksum mismatch for downloaded model: {destination.name}")
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def prepare_sam21_checkpoint(weights_dir: Path) -> Path:
    """Download the official SAM 2.1 Large checkpoint when it is absent."""

    checkpoint = sam21_path(weights_dir)
    download(SAM21_SOURCE_URL, checkpoint, SAM21_SOURCE_SHA256)
    return checkpoint
