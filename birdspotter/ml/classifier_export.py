"""Export the trained MobileNetV4 frame classifier to OpenVINO or TensorRT."""

from pathlib import Path

from birdspotter.ml.classifier import export_classifier as export_checkpoint
from birdspotter.ml.classifier import export_classifier_tensorrt


def export_classifier(
    destination: Path, *, source_checkpoint: Path | None = None, tensorrt: bool = False
) -> None:
    if source_checkpoint is None:
        raise ValueError("Provide the MobileNetV4 checkpoint produced by Modal training")
    if tensorrt:
        export_classifier_tensorrt(source_checkpoint, destination)
    else:
        export_checkpoint(source_checkpoint, destination)
