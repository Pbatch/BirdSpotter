"""OpenVINO conversion and box-prompted inference for SAM 2.1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openvino as ov
import torch
from ultralytics import SAM

from birdspotter.segmentation import component_for_box, validate_mask
from birdspotter.types import Box

SAM21_IMAGE_SIZE = 256
SAM21_PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SAM21_PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
ENCODER_FILENAME = "sam2.1_image_encoder.xml"
MASK_PREDICTOR_FILENAME = "sam2.1_mask_predictor.xml"


class Sam21ImageEncoder(torch.nn.Module):
    """Expose SAM 2.1 image features as three OpenVINO-friendly tensors."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model: Any = model

    @torch.no_grad()
    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone_out = self.model.forward_image(image)
        _, vision_features, _, _ = self.model._prepare_backbone_features(  # noqa: SLF001
            backbone_out
        )
        if self.model.directly_add_no_mem_embed:
            vision_features[-1] = vision_features[-1] + self.model.no_mem_embed
        feature_shapes = ((64, 64), (32, 32), (16, 16))
        features = [
            feature.permute(1, 2, 0).view(1, -1, *shape)
            for feature, shape in zip(vision_features, feature_shapes, strict=True)
        ]
        return features[2], features[0], features[1]


class Sam21MaskPredictor(torch.nn.Module):
    """Combine SAM 2.1's prompt encoder and mask decoder for OpenVINO export."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model: Any = model
        self.prompt_encoder: Any = model.sam_prompt_encoder
        self.mask_decoder: Any = model.sam_mask_decoder

    def embed_points(
        self,
        point_coordinates: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> torch.Tensor:
        point_coordinates = point_coordinates.add(0.5).div(self.model.image_size)
        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(  # noqa: SLF001
            point_coordinates
        )
        expanded_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)
        point_embedding = point_embedding * (expanded_labels != -1).to(torch.float32)
        point_embedding = point_embedding + self.prompt_encoder.not_a_point_embed.weight * (
            expanded_labels == -1
        ).to(torch.float32)
        for index in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.prompt_encoder.point_embeddings[
                index
            ].weight * (expanded_labels == index).to(torch.float32)
        return point_embedding

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coordinates: torch.Tensor,
        point_labels: torch.Tensor,
        high_res_features_256: torch.Tensor,
        high_res_features_128: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sparse_embedding = self.embed_points(point_coordinates, point_labels)
        dense_embedding = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            point_coordinates.shape[0],
            -1,
            image_embeddings.shape[-2],
            image_embeddings.shape[-1],
        )
        low_resolution_masks, scores, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[high_res_features_256, high_res_features_128],
        )
        masks = torch.nn.functional.interpolate(
            low_resolution_masks,
            size=(self.model.image_size, self.model.image_size),
            mode="bilinear",
            align_corners=False,
        ).clamp(-32.0, 32.0)
        return masks, scores


def openvino_paths(model_dir: Path) -> tuple[Path, Path]:
    """Return the image-encoder and mask-predictor IR paths."""

    return model_dir / ENCODER_FILENAME, model_dir / MASK_PREDICTOR_FILENAME


def export_sam21_openvino(
    checkpoint_path: Path,
    model_dir: Path,
) -> tuple[Path, Path]:
    """Convert SAM 2.1 Large's image inference components to OpenVINO IR."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM 2.1 model not found: {checkpoint_path}")
    encoder_path, mask_predictor_path = openvino_paths(model_dir)
    if encoder_path.is_file() and mask_predictor_path.is_file():
        return encoder_path, mask_predictor_path

    model = SAM(str(checkpoint_path)).model
    model.set_imgsz((SAM21_IMAGE_SIZE, SAM21_IMAGE_SIZE))
    model_dir.mkdir(parents=True, exist_ok=True)

    if not encoder_path.is_file():
        encoder = Sam21ImageEncoder(model)
        encoder_ir = ov.convert_model(
            encoder,
            example_input=torch.zeros(1, 3, SAM21_IMAGE_SIZE, SAM21_IMAGE_SIZE),
            input=([1, 3, SAM21_IMAGE_SIZE, SAM21_IMAGE_SIZE],),
        )
        ov.save_model(encoder_ir, encoder_path)

    if not mask_predictor_path.is_file():
        mask_predictor = Sam21MaskPredictor(model)
        example_inputs = {
            "image_embeddings": torch.zeros(1, 256, 16, 16),
            "point_coordinates": torch.zeros(1, 2, 2),
            "point_labels": torch.tensor([[2, 3]], dtype=torch.int32),
            "high_res_features_256": torch.zeros(1, 32, 64, 64),
            "high_res_features_128": torch.zeros(1, 64, 32, 32),
        }
        mask_predictor_ir = ov.convert_model(mask_predictor, example_input=example_inputs)
        ov.save_model(mask_predictor_ir, mask_predictor_path)

    return encoder_path, mask_predictor_path


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
                f"SAM 2.1 OpenVINO model not found: {missing[0]}. Run `birdspotter models`."
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
