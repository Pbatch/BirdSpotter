"""OpenVINO conversion and box-prompted inference for SAM 2.1."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from birdspotter.segmentation import component_for_box, validate_mask
from birdspotter.types import Box

SAM21_IMAGE_SIZE = 512
SAM21_PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SAM21_PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
ENCODER_FILENAME = "sam21_image_encoder.xml"
MASK_PREDICTOR_FILENAME = "sam21_mask_predictor.xml"


def openvino_paths(model_dir: Path) -> tuple[Path, Path]:
    """Return the image-encoder and mask-predictor IR paths."""

    return model_dir / ENCODER_FILENAME, model_dir / MASK_PREDICTOR_FILENAME


def preprocess_image(image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Apply SAM 2.1's RGB normalization, resize, and bottom-right padding."""

    height, width = image_bgr.shape[:2]
    scale = SAM21_IMAGE_SIZE / max(height, width)
    resized_height, resized_width = round(height * scale), round(width * scale)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    tensor = (resized.astype(np.float32) - SAM21_PIXEL_MEAN) / SAM21_PIXEL_STD
    tensor = tensor.transpose(2, 0, 1)[None]
    tensor = np.pad(
        tensor,
        (
            (0, 0),
            (0, 0),
            (0, SAM21_IMAGE_SIZE - resized_height),
            (0, SAM21_IMAGE_SIZE - resized_width),
        ),
    )
    return tensor, (resized_height, resized_width)


def scale_box(
    box: Box, original_shape: tuple[int, int], resized_shape: tuple[int, int]
) -> np.ndarray:
    """Scale a detector xyxy box to SAM 2.1's padded input coordinates."""

    height, width = original_shape
    resized_height, resized_width = resized_shape
    x1, y1, x2, y2 = box
    return np.array(
        [
            [
                [
                    x1 * resized_width / width,
                    y1 * resized_height / height,
                ],
                [
                    x2 * resized_width / width,
                    y2 * resized_height / height,
                ],
            ]
        ],
        dtype=np.float32,
    )


class Sam21OpenVinoSegmenter:
    """Run OpenVINO-converted SAM 2.1 Large with the detector box as its prompt."""

    def __init__(self, model_dir: Path, *, device: str = "CPU") -> None:
        encoder_path, mask_predictor_path = openvino_paths(model_dir)
        missing = [path for path in (encoder_path, mask_predictor_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "SAM 2.1 OpenVINO model not found: "
                f"{missing[0]}. Run `python scripts/ml/export_models.py`."
            )
        self.encoder_path = encoder_path
        self.mask_predictor_path = mask_predictor_path
        self.device = device
        core = ov.Core()
        self.encoder = core.compile_model(encoder_path, device)
        self.mask_predictor = core.compile_model(mask_predictor_path, device)

    def segment(self, image_bgr: np.ndarray, box: Box) -> tuple[np.ndarray, float]:
        tensor, resized_shape = preprocess_image(image_bgr)
        image_embeddings, high_res_256, high_res_128 = self.encoder([tensor]).values()
        point_coordinates = scale_box(box, image_bgr.shape[:2], resized_shape)
        point_labels = np.array([[2, 3]], dtype=np.int32)
        masks, scores = self.mask_predictor(
            {
                "image_embeddings": image_embeddings,
                "point_coordinates": point_coordinates,
                "point_labels": point_labels,
                "high_res_features_256": high_res_256,
                "high_res_features_128": high_res_128,
            }
        ).values()
        mask = np.asarray(masks).squeeze()
        resized_height, resized_width = resized_shape
        mask = mask[:resized_height, :resized_width]
        mask = cv2.resize(
            mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        mask = component_for_box(mask > 0, box)
        validate_mask(mask, box)
        return mask, float(np.asarray(scores).reshape(-1)[0])

    def describe(self) -> dict[str, object]:
        return {
            "model": "SAM 2.1 Large",
            "encoder": self.encoder_path.name,
            "mask_predictor": self.mask_predictor_path.name,
            "device": self.device,
            "input_size": SAM21_IMAGE_SIZE,
            "backend": "OpenVINO",
        }
