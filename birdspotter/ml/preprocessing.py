"""Torchvision image decoding and independent per-image batched augmentation."""

from dataclasses import dataclass

import numpy as np
import torch
from torchvision.io import ImageReadMode, decode_image

from birdspotter.classification import letterbox
from birdspotter.ml.classifier import validate_image_size


@dataclass(frozen=True, slots=True)
class TorchvisionFrameLoader:
    image_size: int

    def __post_init__(self) -> None:
        validate_image_size(self.image_size)

    def __call__(self, path: str) -> torch.Tensor:
        frame = decode_image(path, mode=ImageReadMode.RGB)
        if frame.shape[-2:] == (self.image_size, self.image_size):
            return frame
        rgb = frame.permute(1, 2, 0).numpy()
        prepared, _, _ = letterbox(rgb, (self.image_size, self.image_size))
        return torch.from_numpy(np.ascontiguousarray(prepared.transpose(2, 0, 1)))


class BatchPreprocessor(torch.nn.Module):
    """Convert uint8 batches to normalized float32 on their current device."""

    mean: torch.Tensor
    std: torch.Tensor
    grayscale_weights: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "grayscale_weights",
            torch.tensor([0.2989, 0.587, 0.114]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, frames: torch.Tensor, *, augment: bool = False) -> torch.Tensor:
        if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("Preprocessing requires uint8 RGB batches with shape Bx3xHxW")
        frames = frames.to(dtype=torch.float32, memory_format=torch.channels_last).div_(255)
        if augment:
            frames = self.augment(frames)
        return frames.sub_(self.mean).div_(self.std).contiguous(memory_format=torch.channels_last)

    def augment(self, frames: torch.Tensor) -> torch.Tensor:
        shape = (len(frames), 1, 1, 1)
        flips = torch.rand(shape, device=frames.device) < 0.5
        frames = torch.where(flips, frames.flip(-1), frames)
        brightness = torch.empty(shape, device=frames.device).uniform_(0.9, 1.1)
        contrast = torch.empty(shape, device=frames.device).uniform_(0.9, 1.1)
        brightness_first = torch.rand(shape, device=frames.device) < 0.5
        frames = frames.mul(torch.where(brightness_first, brightness, 1.0)).clamp_(0, 1)
        grayscale = (frames * self.grayscale_weights).sum(dim=1, keepdim=True)
        mean = grayscale.mean(dim=(2, 3), keepdim=True)
        frames = ((frames - mean) * contrast + mean).clamp_(0, 1)
        return frames.mul(torch.where(brightness_first, 1.0, brightness)).clamp_(0, 1)
