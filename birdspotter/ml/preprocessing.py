"""Staged-frame loading with CPU JPEG decoding, batched GPU augmentation and normalisation."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F  # noqa: N812 -- PyTorch's documented alias
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg
from tqdm import tqdm

# ITU-R BT.601 luma, as greyscale conversion and saturation use.
LUMA = (0.299, 0.587, 0.114)


def validate_image_size(image_size: int) -> None:
    if image_size < 32 or image_size % 32:
        raise ValueError("Image size must be a positive multiple of 32")


def stage_frames(parquet: Path, directory: Path) -> dict[str, Path]:
    """Split the dataset Parquet into one uncompressed Arrow file per split.

    Parquet needs a whole row group decoded to reach one row, so shuffled reads go through
    Arrow IPC instead, which loader workers can memory-map for zero-copy random access.
    """
    source = pq.ParquetFile(parquet)
    writers: dict[str, pa.ipc.RecordBatchFileWriter] = {}
    try:
        with tqdm(total=source.metadata.num_rows, desc="Staging frames", unit="image") as bar:
            for batch in source.iter_batches(batch_size=1024, columns=["split", "target", "jpeg"]):
                splits = batch["split"].to_numpy(zero_copy_only=False)
                for split in batch["split"].unique().to_pylist():
                    rows = batch.filter(pa.array(splits == split)).select(["target", "jpeg"])
                    if split not in writers:
                        writers[split] = pa.ipc.new_file(directory / f"{split}.arrow", rows.schema)
                    writers[split].write_batch(rows)
                bar.update(batch.num_rows)
    finally:
        for writer in writers.values():
            writer.close()
    return {split: directory / f"{split}.arrow" for split in writers}


class StagedFrames(Dataset[tuple[torch.Tensor, int]]):
    """Decoded uint8 3xHxW frames and bird=1/no_bird=0 targets from a staged Arrow file.

    Loader workers decode the JPEGs on the CPU. Batched nvjpeg decoding on the GPU hung
    training runs on Modal A100s three times (18 Oct 2026), always mid-epoch at a random step.
    Each process maps the file on first access, so pickling for spawned workers stays cheap.
    """

    def __init__(self, path: Path, image_size: int) -> None:
        validate_image_size(image_size)
        self.path = path
        self.image_size = image_size
        self.frames: pa.ChunkedArray | None = None
        with pa.memory_map(str(path)) as source:
            targets = pa.ipc.open_file(source).read_all()["target"]
            self.targets = torch.from_numpy(targets.to_numpy().astype("int64"))

    def __getstate__(self) -> dict[str, object]:
        return {**self.__dict__, "frames": None}

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self.frames is None:
            self.frames = pa.ipc.open_file(pa.memory_map(str(self.path))).read_all()["jpeg"]
        encoded = torch.frombuffer(bytearray(self.frames[index].as_buffer()), dtype=torch.uint8)
        frame = decode_jpeg(encoded, mode=ImageReadMode.RGB)
        if not isinstance(frame, torch.Tensor):  # a single input decodes to a single tensor
            raise TypeError("decode_jpeg returned a list for one image")
        if frame.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Training frames must be letterboxed to {self.image_size}x{self.image_size}, "
                f"got {tuple(frame.shape[-2:])}"
            )
        return frame, int(self.targets[index])


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Strong photometric jitter: greyscale mimics the camera's infrared night frames, blur a
    fogged lens and noise low light."""

    flip_p: float = 0.5
    brightness: tuple[float, float] = (0.5, 1.5)
    contrast: tuple[float, float] = (0.5, 1.5)
    saturation: tuple[float, float] = (0.4, 1.6)
    hue: tuple[float, float] = (-0.1, 0.1)
    gray_p: float = 0.2
    blur_kernel: int = 5
    blur_sigma: float = 1.5
    blur_p: float = 0.2
    noise_std: float = 0.04
    noise_p: float = 0.3

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


TRAIN_AUGMENTATION = AugmentationConfig()


class BatchAugmentation(torch.nn.Module):
    """Apply TRAIN_AUGMENTATION to uint8 Bx3xHxW batches on their current device.

    Each sample draws its own parameters, so one batch sees the same variety the per-image
    CPU pipeline gave.
    """

    luma: torch.Tensor
    blur: torch.Tensor

    def __init__(self, config: AugmentationConfig = TRAIN_AUGMENTATION) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("luma", torch.tensor(LUMA).view(1, 3, 1, 1), persistent=False)
        offsets = torch.arange(config.blur_kernel) - config.blur_kernel // 2
        kernel = torch.exp(-(offsets**2) / (2 * config.blur_sigma**2))
        # Depthwise weights, one row per RGB channel; transposed for the vertical pass.
        self.register_buffer(
            "blur", (kernel / kernel.sum()).view(1, 1, 1, -1).repeat(3, 1, 1, 1), persistent=False
        )

    def uniform(self, frames: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
        low, high = bounds
        return torch.empty(len(frames), 1, 1, 1, device=frames.device).uniform_(low, high)

    def chance(self, frames: torch.Tensor, p: float) -> torch.Tensor:
        return torch.rand(len(frames), 1, 1, 1, device=frames.device) < p

    def gray(self, frames: torch.Tensor) -> torch.Tensor:
        return (frames * self.luma).sum(1, keepdim=True)

    def rotate_hue(self, frames: torch.Tensor, turns: torch.Tensor) -> torch.Tensor:
        """Rotate chroma about the grey axis, a cheap batched stand-in for an HSV hue shift."""
        angle = 2 * math.pi * turns
        cos, sin = angle.cos(), angle.sin()
        mean = frames.mean(1, keepdim=True)
        chroma = frames - mean
        # Rodrigues' rotation about (1, 1, 1) / sqrt(3).
        red, green, blue = chroma.unbind(1)
        cross = torch.stack((blue - green, red - blue, green - red), 1) / math.sqrt(3)
        return mean + chroma * cos + cross * sin

    def gaussian_blur(self, frames: torch.Tensor) -> torch.Tensor:
        pad = self.config.blur_kernel // 2
        frames = F.conv2d(F.pad(frames, (pad, pad, 0, 0), mode="reflect"), self.blur, groups=3)
        vertical = self.blur.transpose(2, 3)
        return F.conv2d(F.pad(frames, (0, 0, pad, pad), mode="reflect"), vertical, groups=3)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("Augmentation requires uint8 RGB batches with shape Bx3xHxW")
        c = self.config
        frames = frames.float().div_(255)
        frames = torch.where(self.chance(frames, c.flip_p), frames.flip(-1), frames)
        frames = (frames * self.uniform(frames, c.brightness)).clamp_(0, 1)
        mean = self.gray(frames).mean((2, 3), keepdim=True)
        frames = torch.lerp(mean, frames, self.uniform(frames, c.contrast)).clamp_(0, 1)
        frames = torch.lerp(self.gray(frames), frames, self.uniform(frames, c.saturation))
        frames = self.rotate_hue(frames.clamp_(0, 1), self.uniform(frames, c.hue)).clamp_(0, 1)
        frames = torch.where(self.chance(frames, c.gray_p), self.gray(frames), frames)
        frames = torch.where(self.chance(frames, c.blur_p), self.gaussian_blur(frames), frames)
        noisy = frames + torch.randn_like(frames) * c.noise_std
        frames = torch.where(self.chance(frames, c.noise_p), noisy, frames)
        return frames.clamp_(0, 1).mul_(255).round_().to(torch.uint8)


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
