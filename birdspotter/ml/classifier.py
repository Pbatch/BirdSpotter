"""Shared MobileNetV4 training, preprocessing and export contract."""

from pathlib import Path

import numpy as np
import openvino as ov
import timm
import torch
from PIL import Image
from torchvision import transforms

from birdspotter.classification import letterbox

MODEL_NAME = "mobilenetv4_conv_large.e600_r384_in1k"
MODEL_VARIANTS = {
    "small": "mobilenetv4_conv_small.e2400_r224_in1k",
    "large": MODEL_NAME,
}
CLASSES = ["bird"]
IMAGE_SIZE = 640
TRAIN_AUGMENTATIONS = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1)]
)


def validate_image_size(image_size: int) -> None:
    if image_size < 32 or image_size % 32:
        raise ValueError("Image size must be a positive multiple of 32")


def transform(
    image: Image.Image,
    *,
    image_size: int = IMAGE_SIZE,
    augment: bool = False,
) -> torch.Tensor:
    validate_image_size(image_size)
    image = image.convert("RGB")
    if augment:
        image = TRAIN_AUGMENTATIONS(image)
    rgb = np.asarray(image)
    prepared, _, _ = letterbox(rgb, (image_size, image_size))
    values = prepared.astype(np.float32) / 255.0
    values = (values - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)))


def export_classifier(checkpoint: Path, destination: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_name = state["model_name"]
    if state["classes"] != CLASSES or model_name not in MODEL_VARIANTS.values():
        raise ValueError("Checkpoint has an incompatible classifier contract")
    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    model.load_state_dict(state["state_dict"])
    model.eval()
    image_size = state.get("image_size", 224)
    validate_image_size(image_size)
    example = torch.zeros(1, 3, image_size, image_size)
    converted = ov.convert_model(model, example_input=example, input=[example.shape])
    destination.mkdir(parents=True, exist_ok=True)
    ov.save_model(converted, destination / "classifier.xml", compress_to_fp16=True)
