"""Common manifest and upload helpers for Hugging Face packages."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import HfApi


def prepare_output(output: Path, source: Path) -> None:
    if output == source or source in output.parents:
        raise ValueError("Package output must be outside the source directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output: Path, metadata: dict[str, Any]) -> None:
    files = [
        {
            "path": path.relative_to(output).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {"format_version": 1, **metadata, "files": files}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def add_upload_arguments(parser: argparse.ArgumentParser, repository: str) -> None:
    parser.add_argument("--repo-id", default=repository)
    uploads = parser.add_mutually_exclusive_group()
    uploads.add_argument("--upload", action="store_true", help="Package and upload")
    uploads.add_argument("--upload-only", action="store_true", help="Upload an existing package")
    parser.add_argument("--private", action="store_true", help="Create a new private repository")


def upload_package(
    output: Path, repository: str, *, repo_type: Literal["model", "dataset"], private: bool = False
) -> None:
    manifest = json.loads((output / "manifest.json").read_text())
    for entry in manifest["files"]:
        path = output / entry["path"]
        if (
            not path.is_file()
            or path.stat().st_size != entry["size"]
            or sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"Package file changed or missing: {entry['path']}")
    api = HfApi()
    api.create_repo(repo_id=repository, repo_type=repo_type, private=private, exist_ok=True)
    result = api.upload_folder(
        repo_id=repository,
        repo_type=repo_type,
        folder_path=str(output),
        commit_message="Upload BirdSpotter package",
        allow_patterns=[entry["path"] for entry in manifest["files"]] + ["manifest.json"],
    )
    print(f"Uploaded: {result.commit_url}")
