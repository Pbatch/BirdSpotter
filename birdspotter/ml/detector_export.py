"""Export a pretrained or fine-tuned YOLO26 bird detector to OpenVINO."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol, cast

import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from birdspotter.models import (
    COCO_BIRD_CLASS_ID,
    DETECTOR_BIRD_CLASS_ID,
    DETECTOR_BIRD_CLASS_NAME,
)

DETECTOR_SOURCE_REPOSITORY = "PBatch23888/birdspotter-yolo26s"
DETECTOR_SOURCE_FILENAME = "best.pt"
DETECTOR_INPUT_SHAPE = (896, 1600)


class BirdOnlyDetectionHead(Protocol):
    nc: int
    no: int
    reg_max: int
    end2end: bool
    cv3: list[torch.nn.Sequential]
    one2one_cv3: list[torch.nn.Sequential]


class BirdOnlyDetectionModel(Protocol):
    model: list[BirdOnlyDetectionHead]
    names: dict[int, str]


class BirdOnlyYoloModel(Protocol):
    model: BirdOnlyDetectionModel


def convolution_pair(value: tuple[int, ...], *, name: str) -> tuple[int, int]:
    """Return a validated two-dimensional Conv2d parameter."""

    if len(value) != 2:
        raise ValueError(f"Expected a two-dimensional {name}, got {value}")
    return value[0], value[1]


def bird_only_detector(model: BirdOnlyYoloModel) -> None:
    """Retain only COCO's pretrained bird classifier in a YOLO detection model."""

    detector_model = model.model
    detector = detector_model.model[-1]
    if detector.nc <= COCO_BIRD_CLASS_ID:
        raise ValueError(
            "Detector has "
            f"{detector.nc} classes; COCO bird class {COCO_BIRD_CLASS_ID} is unavailable"
        )

    classifier_heads = [detector.cv3]
    if detector.end2end:
        classifier_heads.append(detector.one2one_cv3)
    for classifier_head in classifier_heads:
        for stage in classifier_head:
            classifier = stage[-1]
            if not isinstance(classifier, torch.nn.Conv2d):
                raise TypeError(
                    "Expected a Conv2d classifier at the end of each YOLO detection head"
                )
            bird_classifier = torch.nn.Conv2d(
                classifier.in_channels,
                1,
                convolution_pair(classifier.kernel_size, name="kernel size"),
                stride=convolution_pair(classifier.stride, name="stride"),
                padding=(
                    classifier.padding
                    if isinstance(classifier.padding, str)
                    else convolution_pair(classifier.padding, name="padding")
                ),
                dilation=convolution_pair(classifier.dilation, name="dilation"),
                groups=classifier.groups,
                bias=classifier.bias is not None,
                padding_mode=classifier.padding_mode,
            ).to(device=classifier.weight.device, dtype=classifier.weight.dtype)
            with torch.no_grad():
                bird_classifier.weight.copy_(
                    classifier.weight[COCO_BIRD_CLASS_ID : COCO_BIRD_CLASS_ID + 1]
                )
                if classifier.bias is not None:
                    bird_bias = bird_classifier.bias
                    if bird_bias is None:
                        raise RuntimeError("Bird classifier unexpectedly has no bias")
                    bird_bias.copy_(classifier.bias[COCO_BIRD_CLASS_ID : COCO_BIRD_CLASS_ID + 1])
            stage[-1] = bird_classifier

    detector.nc = 1
    detector.no = detector.reg_max * 4 + detector.nc
    detector_model.names = {DETECTOR_BIRD_CLASS_ID: DETECTOR_BIRD_CLASS_NAME}


def export_detector(
    destination: Path,
    *,
    input_shape: tuple[int, int] = DETECTOR_INPUT_SHAPE,
    calibration_data: Path | None = None,
    calibration_images: int = 500,
    source_checkpoint: Path | None = None,
) -> None:
    """Export a bird-only YOLO26 model as a static batch-one INT8 OpenVINO IR."""

    if source_checkpoint is None and destination.is_dir() and any(destination.glob("*.xml")):
        print(f"Already present: {destination}")
        return
    if calibration_data is None:
        raise ValueError("INT8 OpenVINO export requires calibration data")
    if calibration_images < 1:
        raise ValueError("INT8 OpenVINO export requires at least one calibration image")
    height, width = input_shape
    if height % 32 or width % 32:
        raise ValueError("Detector input dimensions must be divisible by 32")

    destination.parent.mkdir(parents=True, exist_ok=True)
    source = source_checkpoint
    if source is None:
        source = Path(
            hf_hub_download(
                repo_id=DETECTOR_SOURCE_REPOSITORY,
                filename=DETECTOR_SOURCE_FILENAME,
            )
        )
    elif not source.is_file():
        raise FileNotFoundError(f"Detector checkpoint not found: {source}")
    print(f"Exporting {source.name} OpenVINO INT8 at {height}x{width} ...")
    model = YOLO(str(source))
    detector_model = cast(BirdOnlyYoloModel, model).model
    if len(detector_model.names) == 1:
        detector_model.names = {DETECTOR_BIRD_CLASS_ID: DETECTOR_BIRD_CLASS_NAME}
    else:
        bird_only_detector(cast(BirdOnlyYoloModel, model))
    exported = Path(
        model.export(
            format="openvino",
            imgsz=input_shape,
            quantize=8,
            dynamic=False,
            batch=1,
            nms=False,
            device="cpu",
            data=str(calibration_data) if calibration_data is not None else None,
            fraction=calibration_images,
        )
    )
    if not exported.is_dir() or not any(exported.glob("*.xml")):
        raise RuntimeError(f"Ultralytics reported an invalid OpenVINO export: {exported}")
    replace_export(exported, destination)
    print(f"Saved OpenVINO detector: {destination}")


def replace_export(exported: Path, destination: Path) -> None:
    """Replace an existing detector export while preserving it if the move fails."""

    if not destination.exists():
        shutil.move(str(exported), destination)
        return

    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    destination.replace(backup)
    try:
        shutil.move(str(exported), destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)
