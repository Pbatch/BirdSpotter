"""Export the active detector and SAM 2.1 OpenVINO model artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from birdspotter.detector_export import export_detector
from birdspotter.models import (
    default_weights_dir,
    detector_path,
    prepare_sam21_checkpoint,
    sam21_openvino_dir,
)
from birdspotter.sam21_export import export_sam21_openvino


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
    args = parser.parse_args()
    weights_dir = args.weights_dir.resolve()
    export_detector(detector_path(weights_dir), calibration_data=args.calibration_data.resolve())
    encoder_path, mask_predictor_path = export_sam21_openvino(
        prepare_sam21_checkpoint(weights_dir), sam21_openvino_dir(weights_dir)
    )
    print(f"Saved SAM 2.1 OpenVINO image encoder: {encoder_path}")
    print(f"Saved SAM 2.1 OpenVINO mask predictor: {mask_predictor_path}")


if __name__ == "__main__":
    main()
