"""Command-line interface for the bird pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from birdspotter.detection import BirdDetector
from birdspotter.models import (
    DETECTOR_INPUT_SIZE,
    default_weights_dir,
    detector_path,
    edge_decoder_path,
    edge_encoder_path,
    prepare_models,
    sam21_openvino_dir,
    sam21_path,
)
from birdspotter.pipeline import BirdPipeline
from birdspotter.sam21_openvino import Sam21OpenVinoSegmenter, export_sam21_openvino
from birdspotter.segmentation import EdgeSamSegmenter, Sam21Segmenter


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=default_weights_dir(),
        help="Model artifact directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("segmented"),
        help="Final RGBA output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.10,
        help="Minimum bird detector confidence (default: %(default)s)",
    )
    parser.add_argument(
        "--segmenter",
        choices=("edgesam", "sam2.1", "sam2.1-openvino"),
        default="edgesam",
        help="Segmentation model (default: %(default)s)",
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="birdspotter",
        description="Select and segment the highest-confidence bird.",
    )
    commands = root.add_subparsers(dest="command", required=True)

    models = commands.add_parser("models", help="Download/export inference models")
    models.add_argument(
        "--weights-dir",
        type=Path,
        default=default_weights_dir(),
        help="Model artifact directory (default: %(default)s)",
    )
    models.add_argument("--input-size", type=int, default=DETECTOR_INPUT_SIZE)
    models.add_argument(
        "--calibration-data",
        type=Path,
        default=None,
        help="Dataset YAML required when creating the INT8 detector",
    )
    models.add_argument(
        "--export-sam21-openvino",
        action="store_true",
        help="Convert the local SAM 2.1 Large checkpoint to OpenVINO IR",
    )

    image = commands.add_parser("image", help="Process one image")
    add_runtime_options(image)
    image.add_argument("path", type=Path)

    webcam = commands.add_parser("webcam", help="Capture from a V4L2 webcam")
    add_runtime_options(webcam)
    webcam.add_argument("--device", type=int, default=0)
    webcam.add_argument("--width", type=int, default=1920)
    webcam.add_argument("--height", type=int, default=1080)
    webcam.add_argument("--fps", type=int, default=30)
    webcam.add_argument("--sample-fps", type=float, default=1.0)
    webcam.add_argument("--window-seconds", type=float, default=4.0)
    webcam.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds (default: run until output limit)",
    )
    webcam.add_argument(
        "--max-outputs",
        type=int,
        default=1,
        help="Stop after N segmented birds; 0 means unlimited (default: %(default)s)",
    )
    return root


def build_pipeline(args: argparse.Namespace) -> BirdPipeline:
    weights_dir = args.weights_dir.resolve()
    detector = BirdDetector(
        detector_path(weights_dir),
        confidence=args.confidence,
    )
    if args.segmenter == "sam2.1":
        segmenter = Sam21Segmenter(sam21_path(weights_dir))
    elif args.segmenter == "sam2.1-openvino":
        segmenter = Sam21OpenVinoSegmenter(sam21_openvino_dir(weights_dir))
    else:
        segmenter = EdgeSamSegmenter(
            edge_encoder_path(weights_dir),
            edge_decoder_path(weights_dir),
        )
    return BirdPipeline(
        detector,
        segmenter,
        args.output_dir.resolve(),
    )


def main() -> None:
    args = parser().parse_args()
    if args.command == "models":
        weights_dir = args.weights_dir.resolve()
        calibration_data = (
            args.calibration_data.resolve() if args.calibration_data is not None else None
        )
        prepare_models(
            weights_dir,
            input_size=args.input_size,
            calibration_data=calibration_data,
        )
        if args.export_sam21_openvino:
            encoder_path, mask_predictor_path = export_sam21_openvino(
                sam21_path(weights_dir), sam21_openvino_dir(weights_dir)
            )
            print(f"Saved SAM 2.1 OpenVINO image encoder: {encoder_path}")
            print(f"Saved SAM 2.1 OpenVINO mask predictor: {mask_predictor_path}")
        return

    pipeline = build_pipeline(args)
    if args.command == "image":
        result = pipeline.process_image(args.path)
        if result is None:
            print("No bird detected")
            raise SystemExit(2)
        print(f"Saved segmented bird: {result.image_path}")
        return

    results = pipeline.process_webcam(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        sample_fps=args.sample_fps,
        window_seconds=args.window_seconds,
        duration_seconds=args.duration,
        max_outputs=args.max_outputs,
    )
    if not results:
        print("No segmented bird produced")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
