"""Export the active detector and SAM 2.1 OpenVINO model artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from birdspotter.ml.checkpoints import sam21_checkpoint
from birdspotter.ml.detector_export import export_detector
from birdspotter.ml.sam21_export import export_sam21_openvino
from birdspotter.models import (
    default_weights_dir,
    detector_path,
    sam21_openvino_dir,
)


def main() -> None:
    """Export the runtime models from their development checkpoints."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=default_weights_dir(),
        help="OpenVINO artifact directory (default: %(default)s)",
    )
    parser.add_argument(
        "--calibration-data",
        type=Path,
        default=Path("demo/calibration.yaml"),
        help="Representative dataset YAML for detector INT8 calibration (default: %(default)s)",
    )
    parser.add_argument(
        "--calibration-images",
        type=int,
        default=500,
        help="Number of representative images for detector INT8 calibration (default: %(default)s)",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=None,
        help="Fine-tuned one-class YOLO checkpoint; defaults to the public Hugging Face model",
    )
    args = parser.parse_args()
    weights_dir = args.weights_dir.resolve()
    export_detector(
        detector_path(weights_dir),
        calibration_data=args.calibration_data.resolve(),
        calibration_images=args.calibration_images,
        source_checkpoint=(
            args.detector_checkpoint.resolve() if args.detector_checkpoint is not None else None
        ),
    )
    with sam21_checkpoint() as checkpoint:
        encoder_path, mask_predictor_path = export_sam21_openvino(
            checkpoint, sam21_openvino_dir(weights_dir)
        )
    print(f"Saved SAM 2.1 OpenVINO image encoder: {encoder_path}")
    print(f"Saved SAM 2.1 OpenVINO mask predictor: {mask_predictor_path}")


if __name__ == "__main__":
    main()
