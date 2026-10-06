#!/usr/bin/env python3
"""Download and verify BirdSpotter's deployment models from Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

from birdspotter.models import classifier_path, default_weights_dir, sam3_openvino_dir

DEFAULT_REVISION = "main"
CLASSIFIER_MODEL_PREFIX = "openvino/"
SAM3_MODEL_PREFIX = "openvino-504/"
USER_AGENT = "BirdSpotter/0.1"


def sha256(path: Path) -> str:
    """Calculate a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_url(repository: str, revision: str, path: str) -> str:
    """Build an HTTPS URL for a Hugging Face repository file."""
    encoded_repository = urllib.parse.quote(repository, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    encoded_path = urllib.parse.quote(path, safe="/")
    return (
        f"https://huggingface.co/{encoded_repository}/resolve/"
        f"{encoded_revision}/{encoded_path}?download=true"
    )


def request(url: str) -> urllib.request.Request:
    """Build a model download request after validating its scheme."""
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError("Model downloads require HTTPS")
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310


def fetch_manifest(repository: str, revision: str) -> dict[str, Any]:
    """Fetch the model manifest from Hugging Face."""
    url = repository_url(repository, revision, "manifest.json")
    with urllib.request.urlopen(request(url), timeout=120) as response:  # noqa: S310
        manifest = json.load(response)
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError("Unsupported or invalid model manifest")
    return manifest


def model_files(manifest: dict[str, Any], model_prefix: str) -> list[dict[str, Any]]:
    """Return validated OpenVINO entries below a manifest prefix."""

    entries = []
    for entry in manifest.get("files", []):
        if not isinstance(entry, dict) or not str(entry.get("path", "")).startswith(model_prefix):
            continue
        remote_path = str(entry["path"])
        relative_path = PurePosixPath(remote_path).relative_to(model_prefix)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe path in model manifest: {remote_path}")
        digest = entry.get("sha256")
        size = entry.get("size")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Invalid checksum in model manifest: {remote_path}")
        if not isinstance(size, int) or size < 1:
            raise ValueError(f"Invalid size in model manifest: {remote_path}")
        entries.append(
            {
                "path": remote_path,
                "relative_path": str(relative_path),
                "sha256": digest,
                "size": size,
            }
        )
    suffixes = {Path(str(entry["relative_path"])).suffix for entry in entries}
    if ".xml" not in suffixes or ".bin" not in suffixes:
        raise ValueError("Model manifest does not contain an OpenVINO .xml/.bin pair")
    return entries


def file_matches(path: Path, entry: dict[str, Any]) -> bool:
    """Check an installed file against one manifest entry."""
    return (
        path.is_file() and path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"]
    )


def download_file(url: str, destination: Path, entry: dict[str, Any]) -> None:
    """Download and verify one repository file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {entry['path']} ...")
    with (
        urllib.request.urlopen(request(url), timeout=120) as response,  # noqa: S310
        destination.open("wb") as output,
    ):
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if not file_matches(destination, entry):
        raise RuntimeError(f"Size or checksum mismatch for {entry['path']}")


def install_model(
    repository: str,
    revision: str,
    destination: Path,
    model_prefix: str,
    *,
    force: bool,
) -> None:
    """Download, verify, and atomically install an OpenVINO model."""

    manifest = fetch_manifest(repository, revision)
    entries = model_files(manifest, model_prefix)
    if not force and all(
        file_matches(destination / str(entry["relative_path"]), entry) for entry in entries
    ):
        print(f"Already up to date: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    backup = destination.with_name(f".{destination.name}.previous")
    try:
        for entry in entries:
            relative_path = str(entry["relative_path"])
            download_file(
                repository_url(repository, revision, str(entry["path"])),
                temporary / relative_path,
                entry,
            )
        installed_manifest = {
            "repository": repository,
            "revision": revision,
            "source_manifest": manifest,
        }
        (temporary / "manifest.json").write_text(json.dumps(installed_manifest, indent=2) + "\n")

        if backup.exists():
            raise FileExistsError(f"Refusing to overwrite recovery directory: {backup}")
        if destination.exists():
            destination.replace(backup)
        try:
            temporary.replace(destination)
        except Exception:
            if backup.exists():
                backup.replace(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(f"Installed model: {destination}")


def main() -> None:
    """Parse command-line arguments and install the deployment models."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-repository", help="Published MobileNetV4 repository")
    parser.add_argument("--sam3-repository", help="Repository containing the custom SAM 3 export")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--classifier-destination",
        type=Path,
        default=classifier_path(default_weights_dir()),
    )
    parser.add_argument(
        "--sam3-destination",
        type=Path,
        default=sam3_openvino_dir(default_weights_dir()),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.classifier_repository:
        install_model(
            args.classifier_repository,
            args.revision,
            args.classifier_destination.resolve(),
            CLASSIFIER_MODEL_PREFIX,
            force=args.force,
        )
    elif not list(args.classifier_destination.glob("*.xml")):
        raise FileNotFoundError("Train/export MobileNetV4 first or pass --classifier-repository")
    if args.sam3_repository:
        install_model(
            args.sam3_repository,
            args.revision,
            args.sam3_destination.resolve(),
            SAM3_MODEL_PREFIX,
            force=args.force,
        )
    else:
        required = ("sam3_bird.xml", "sam3_bird.bin")
        missing = [name for name in required if not (args.sam3_destination / name).is_file()]
        if missing:
            parser.error(
                "SAM 3 requires a local export or --sam3-repository. "
                "Run `python -m birdspotter.ml.sam3_export` first."
            )
        print(f"Using local SAM 3 export: {args.sam3_destination}")


if __name__ == "__main__":
    main()
