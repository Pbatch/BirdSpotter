#!/usr/bin/env python3
"""Export a trained YOLO26 checkpoint to INT8 OpenVINO on Modal."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import modal
import yaml
from ultralytics import YOLO

VOLUME_NAME = "birdspotter-training"
VOLUME_ROOT = "/mnt/birdspotter"

app = modal.App("birdspotter-yolo26-openvino-export")
volume = modal.Volume.from_name(VOLUME_NAME)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("libgl1", "libglib2.0-0", "pigz")
    .uv_pip_install(
        "nncf==3.2.0",
        "opencv-python==4.14.0.94",
        "openvino==2026.2.1",
        "pyyaml==6.0.3",
        "ultralytics==8.4.140",
    )
)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={VOLUME_ROOT: volume},
)
def export(
    checkpoint: str,
    dataset_tar: str,
    output_dir: str,
    image_size: int = 640,
    calibration_images: int = 500,
) -> str:
    """Create and persist a static batch-one INT8 OpenVINO export."""
    checkpoint_path = Path(VOLUME_ROOT) / checkpoint.lstrip("/")
    archive = Path(VOLUME_ROOT) / dataset_tar.lstrip("/")
    destination = Path(VOLUME_ROOT) / output_dir.lstrip("/")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if image_size < 32 or image_size % 32:
        raise ValueError("Image size must be at least 32 and divisible by 32")

    dataset_dir = Path("/tmp/birdspotter-calibration")  # noqa: S108
    dataset_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603
        [
            "/bin/tar",
            "--extract",
            "--file",
            str(archive),
            "--directory",
            str(dataset_dir),
            "--use-compress-program=pigz -d",
            "--no-same-owner",
            "--no-same-permissions",
        ],
        check=True,
    )
    yaml_files = sorted(dataset_dir.rglob("data.yaml"))
    if len(yaml_files) != 1:
        raise RuntimeError(f"Expected exactly one data.yaml, found {len(yaml_files)}")
    data = yaml.safe_load(yaml_files[0].read_text())
    data.update(path=str(yaml_files[0].parent.resolve()))
    local_yaml = Path("/tmp/birdspotter-calibration.yaml")  # noqa: S108
    local_yaml.write_text(yaml.safe_dump(data, sort_keys=False))

    local_checkpoint = Path("/tmp") / checkpoint_path.name  # noqa: S108
    shutil.copy2(checkpoint_path, local_checkpoint)
    exported = Path(
        YOLO(str(local_checkpoint)).export(
            format="openvino",
            imgsz=image_size,
            quantize=8,
            dynamic=False,
            batch=1,
            nms=False,
            device="cpu",
            data=str(local_yaml),
            fraction=calibration_images,
        )
    )
    if not exported.is_dir() or not any(exported.glob("*.xml")):
        raise RuntimeError(f"Ultralytics reported an invalid export: {exported}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), destination)
    volume.commit()
    print(f"Saved INT8 OpenVINO export to {destination}")
    return output_dir


@app.local_entrypoint()
def main(
    checkpoint: str,
    dataset_tar: str,
    output_dir: str,
    image_size: int = 640,
    calibration_images: int = 500,
) -> None:
    print(
        export.remote(
            checkpoint,
            dataset_tar,
            output_dir,
            image_size,
            calibration_images,
        )
    )
