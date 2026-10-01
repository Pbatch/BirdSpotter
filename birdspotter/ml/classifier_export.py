"""Export the trained MobileNetV4 frame classifier to OpenVINO."""

from pathlib import Path

from birdspotter.ml.classifier import export_classifier as export_checkpoint


def export_classifier(destination: Path, *, source_checkpoint: Path | None = None) -> None:
    if source_checkpoint is None:
        raise ValueError("Provide the MobileNetV4 checkpoint produced by Modal training")
    export_checkpoint(source_checkpoint, destination)
