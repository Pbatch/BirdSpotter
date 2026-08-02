"""Box-prompted EdgeSAM segmentation through its official ONNX exports."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import SAM

from birdspotter.types import Box

EDGE_SIZE = 1024
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)[None, :, None, None]
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)[None, :, None, None]


def resized_shape(height: int, width: int, long_side: int = EDGE_SIZE) -> tuple[int, int]:
    scale = long_side / max(height, width)
    return round(height * scale), round(width * scale)


def scale_box(box: Box, original_shape: tuple[int, int]) -> np.ndarray:
    """Scale an xyxy box to EdgeSAM's resized, pre-padding coordinates."""

    height, width = original_shape
    new_height, new_width = resized_shape(height, width)
    x_scale = new_width / width
    y_scale = new_height / height
    x1, y1, x2, y2 = box
    return np.array(
        [[[x1 * x_scale, y1 * y_scale], [x2 * x_scale, y2 * y_scale]]],
        dtype=np.float32,
    )


def preprocess_image(image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Convert BGR source pixels to EdgeSAM's normalized RGB tensor."""

    height, width = image_bgr.shape[:2]
    new_height, new_width = resized_shape(height, width)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    tensor = resized.transpose(2, 0, 1)[None].astype(np.float32)
    tensor = (tensor - PIXEL_MEAN) / PIXEL_STD
    tensor = np.pad(
        tensor,
        ((0, 0), (0, 0), (0, EDGE_SIZE - new_height), (0, EDGE_SIZE - new_width)),
        constant_values=0,
    )
    return tensor, (new_height, new_width)


def postprocess_mask(
    low_resolution_mask: np.ndarray,
    input_shape: tuple[int, int],
    original_shape: tuple[int, int],
) -> np.ndarray:
    """Remove EdgeSAM padding and resize mask logits to source resolution."""

    mask = np.asarray(low_resolution_mask).squeeze()
    mask = cv2.resize(mask, (EDGE_SIZE, EDGE_SIZE), interpolation=cv2.INTER_LINEAR)
    input_height, input_width = input_shape
    mask = mask[:input_height, :input_width]
    original_height, original_width = original_shape
    return cv2.resize(
        mask,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )


def component_for_box(mask: np.ndarray, box: Box) -> np.ndarray:
    """Keep the connected mask component that overlaps the detector box most."""

    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary.astype(bool)

    height, width = binary.shape
    x1 = int(np.clip(np.floor(box[0]), 0, width))
    y1 = int(np.clip(np.floor(box[1]), 0, height))
    x2 = int(np.clip(np.ceil(box[2]), 0, width))
    y2 = int(np.clip(np.ceil(box[3]), 0, height))

    best_label = 0
    best_key = (-1, -1)
    for label in range(1, count):
        overlap = int(np.count_nonzero(labels[y1:y2, x1:x2] == label))
        area = int(stats[label, cv2.CC_STAT_AREA])
        key = (overlap, area)
        if key > best_key:
            best_label = label
            best_key = key
    return labels == best_label


def validate_mask(mask: np.ndarray, box: Box, *, minimum_pixels: int = 64) -> None:
    """Reject empty or implausible masks before writing a downstream artifact."""

    if mask.ndim != 2:
        raise ValueError("Segmentation mask must be two-dimensional")
    pixels = int(np.count_nonzero(mask))
    if pixels < minimum_pixels:
        raise ValueError(f"Segmentation mask is too small ({pixels} pixels)")

    height, width = mask.shape
    x1 = int(np.clip(np.floor(box[0]), 0, width))
    y1 = int(np.clip(np.floor(box[1]), 0, height))
    x2 = int(np.clip(np.ceil(box[2]), 0, width))
    y2 = int(np.clip(np.ceil(box[3]), 0, height))
    if x2 <= x1 or y2 <= y1 or not np.any(mask[y1:y2, x1:x2]):
        raise ValueError("Segmentation mask does not overlap the detector box")


class EdgeSamSegmenter:
    """Run EdgeSAM encoder and decoder ONNX models with a box prompt."""

    def __init__(
        self,
        encoder_path: Path,
        decoder_path: Path,
        *,
        providers: list[str] | None = None,
    ) -> None:
        missing = [path for path in (encoder_path, decoder_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"EdgeSAM model not found: {missing[0]}. Run `birdspotter models`."
            )

        selected = providers or ["CPUExecutionProvider"]
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.encoder = ort.InferenceSession(str(encoder_path), providers=selected)
        self.decoder = ort.InferenceSession(str(decoder_path), providers=selected)

    def segment(self, image_bgr: np.ndarray, box: Box) -> tuple[np.ndarray, float]:
        tensor, input_shape = preprocess_image(image_bgr)
        embeddings = self.encoder.run(None, {"image": tensor})[0]
        box_coordinates = scale_box(box, image_bgr.shape[:2])
        center_coordinate = box_coordinates.mean(axis=1, keepdims=True)
        point_coordinates = np.concatenate((box_coordinates, center_coordinate), axis=1)
        point_labels = np.array([[2, 3, 1]], dtype=np.float32)
        outputs = self.decoder.run(
            None,
            {
                "image_embeddings": embeddings,
                "point_coords": point_coordinates,
                "point_labels": point_labels,
            },
        )
        scores = np.asarray(outputs[0])
        low_resolution_masks = np.asarray(outputs[1])
        best_index = int(scores.reshape(-1).argmax())
        masks = low_resolution_masks.reshape(-1, *low_resolution_masks.shape[-2:])
        logits = postprocess_mask(
            masks[best_index],
            input_shape,
            image_bgr.shape[:2],
        )
        mask = component_for_box(logits > 0, box)
        validate_mask(mask, box)
        return mask, float(scores.reshape(-1)[best_index])

    def describe(self) -> dict[str, object]:
        return {
            "model": "EdgeSAM-3x",
            "encoder": self.encoder_path.name,
            "decoder": self.decoder_path.name,
            "providers": self.encoder.get_providers(),
        }


class Sam21Segmenter:
    """Run SAM 2.1 with the detector box as its spatial prompt."""

    def __init__(self, model_path: Path, *, device: str = "cpu", image_size: int = 1024) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"SAM 2.1 model not found: {model_path}")
        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        self.model = SAM(str(model_path))

    def segment(self, image_bgr: np.ndarray, box: Box) -> tuple[np.ndarray, float]:
        result = self.model.predict(
            source=image_bgr,
            bboxes=[list(box)],
            device=self.device,
            imgsz=self.image_size,
            verbose=False,
        )[0]
        if result.masks is None or result.boxes is None:
            raise ValueError("SAM 2.1 did not return a mask for the detector box")

        masks = result.masks.data.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        if not len(masks) or not len(scores):
            raise ValueError("SAM 2.1 returned an empty mask result")
        best_index = int(np.argmax(scores))
        mask = masks[best_index]
        if mask.shape != image_bgr.shape[:2]:
            mask = cv2.resize(
                mask,
                (image_bgr.shape[1], image_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = component_for_box(mask > 0.5, box)
        validate_mask(mask, box)
        return mask, float(scores[best_index])

    def describe(self) -> dict[str, object]:
        return {
            "model": "SAM 2.1 Large",
            "checkpoint": self.model_path.name,
            "device": self.device,
            "input_size": self.image_size,
            "backend": "PyTorch",
        }
