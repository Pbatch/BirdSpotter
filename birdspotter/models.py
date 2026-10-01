"""Paths for deployment-ready model artifacts."""

from __future__ import annotations

from pathlib import Path

CLASSIFIER_DIRNAME = "mobilenetv4-bird-640-openvino"
SAM3_DIRNAME = "sam3"
SAM3_OPENVINO_DIRNAME = "openvino-504"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_weights_dir() -> Path:
    return project_root() / "weights"


def classifier_path(weights_dir: Path) -> Path:
    return weights_dir / "classifier" / CLASSIFIER_DIRNAME


def sam3_openvino_dir(weights_dir: Path) -> Path:
    return weights_dir / SAM3_DIRNAME / SAM3_OPENVINO_DIRNAME
