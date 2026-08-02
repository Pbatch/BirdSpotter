"""Download and export the inference model artifacts."""

from __future__ import annotations

import hashlib
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import torch

DETECTOR_SOURCE = "yolo26s.pt"
DETECTOR_SOURCE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt"
DETECTOR_SOURCE_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
DETECTOR_INPUT_SHAPE = (1088, 1920)
DETECTOR_DIRNAME = "yolo26s-bird-1088x1920-openvino-int8"
COCO_BIRD_CLASS_ID = 14
DETECTOR_BIRD_CLASS_ID = 0
DETECTOR_BIRD_CLASS_NAME = "bird"
SAM21_FILENAME = "sam2.1_l.pt"
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

    import torch  # noqa: PLC0415 -- model-export dependency only

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


def export_detector(
    destination: Path,
    *,
    input_shape: tuple[int, int] = DETECTOR_INPUT_SHAPE,
    calibration_data: Path | None = None,
) -> None:
    """Export a bird-only YOLO26s model as a static batch-one INT8 OpenVINO IR."""

    if destination.is_dir() and any(destination.glob("*.xml")):
        print(f"Already present: {destination}")
        return
    if calibration_data is None:
        raise ValueError("INT8 OpenVINO export requires calibration data")
    height, width = input_shape
    if height % 32 or width % 32:
        raise ValueError("Detector input dimensions must be divisible by 32")

    destination.parent.mkdir(parents=True, exist_ok=True)
    source = development_weights_dir(destination.parent.parent) / "detector" / DETECTOR_SOURCE
    download(DETECTOR_SOURCE_URL, source, DETECTOR_SOURCE_SHA256)
    print(f"Exporting YOLO26s OpenVINO INT8 at {height}x{width} ...")
    from ultralytics import YOLO  # noqa: PLC0415 -- model-export dependency only

    model = YOLO(str(source))
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
        )
    )
    if not exported.is_dir() or not any(exported.glob("*.xml")):
        raise RuntimeError(f"Ultralytics reported an invalid OpenVINO export: {exported}")
    shutil.move(str(exported), destination)
    print(f"Saved OpenVINO detector: {destination}")


def prepare_models(
    weights_dir: Path,
    *,
    calibration_data: Path | None = None,
) -> None:
    """Prepare the active detector artifact."""

    export_detector(
        detector_path(weights_dir),
        calibration_data=calibration_data,
    )
