"""Image decoding with Albumentations training augmentation, and batched normalisation."""

import os
from dataclasses import dataclass

# The pinned release otherwise queries PyPI for updates on import, in every loader worker.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A  # noqa: N812 -- the library's documented alias
import numpy as np
import torch
from torchvision.io import ImageReadMode, decode_image

from birdspotter.classifier import letterbox

# Strong photometric jitter: greyscale mimics the camera's infrared night frames, blur a
# fogged lens and noise low light.
TRAIN_AUGMENTATION = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=(0.5, 1.5),
            contrast=(0.5, 1.5),
            saturation=(0.4, 1.6),
            hue=(-0.1, 0.1),
            p=1.0,
        ),
        A.ToGray(p=0.2),
        A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(1.5, 1.5), p=0.2),
        A.GaussNoise(std_range=(0.04, 0.04), per_channel=True, p=0.3),
    ]
)


def validate_image_size(image_size: int) -> None:
    if image_size < 32 or image_size % 32:
        raise ValueError("Image size must be a positive multiple of 32")


@dataclass(frozen=True, slots=True)
class TorchvisionFrameLoader:
    image_size: int
    augment: bool = False

    def __post_init__(self) -> None:
        validate_image_size(self.image_size)

    def __call__(self, path: str) -> torch.Tensor:
        frame = decode_image(path, mode=ImageReadMode.RGB)
        letterboxed = frame.shape[-2:] == (self.image_size, self.image_size)
        if letterboxed and not self.augment:
            return frame
        rgb = frame.permute(1, 2, 0).numpy()
        if not letterboxed:
            rgb, _, _ = letterbox(rgb, (self.image_size, self.image_size))
        if self.augment:
            rgb = TRAIN_AUGMENTATION(image=rgb)["image"]
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


class BatchPreprocessor(torch.nn.Module):
    """Convert uint8 batches to normalized float32 on their current device."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("Preprocessing requires uint8 RGB batches with shape Bx3xHxW")
        frames = frames.to(dtype=torch.float32, memory_format=torch.channels_last).div_(255)
        return frames.sub_(self.mean).div_(self.std).contiguous(memory_format=torch.channels_last)
