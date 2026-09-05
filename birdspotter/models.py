"""Paths for deployment-ready model artifacts."""

from __future__ import annotations

from pathlib import Path

DETECTOR_DIRNAME = "yolo26-bird-896x1600-openvino-int8"
COCO_BIRD_CLASS_ID = 14
DETECTOR_BIRD_CLASS_ID = 0
DETECTOR_BIRD_CLASS_NAME = "bird"
SAM21_DIRNAME = "sam21"
SAM21_OPENVINO_DIRNAME = "openvino-512"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_weights_dir() -> Path:
    return project_root() / "weights"


def detector_path(weights_dir: Path) -> Path:
    return weights_dir / "detector" / DETECTOR_DIRNAME


def sam21_openvino_dir(weights_dir: Path) -> Path:
    return weights_dir / SAM21_DIRNAME / SAM21_OPENVINO_DIRNAME
