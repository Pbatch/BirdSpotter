"""Lightning training for the single-logit MobileNetV4 classifier."""

import math
from dataclasses import dataclass
from pathlib import Path

import lightning
import timm
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
from torchvision.ops import sigmoid_focal_loss

from birdspotter.ml.classifier import (
    CLASSES,
    IMAGE_SIZE,
    MODEL_NAME,
    MODEL_VARIANTS,
    validate_image_size,
)
from birdspotter.ml.preprocessing import BatchPreprocessor


def bird_target(folder_target: int) -> int:
    """ImageFolder sorts bird first; training uses bird=1 and no_bird=0."""
    return 1 - folder_target


@dataclass
class WarmupCosineSchedule:
    """Stateful callable so LambdaLR checkpoints preserve the original schedule length."""

    total_steps: int
    warmup_steps: int

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return 0.1 + 0.9 * step / self.warmup_steps
        progress = min(1.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2


class BirdClassifier(lightning.LightningModule):
    def __init__(  # noqa: PLR0913
        self,
        *,
        pretrained: bool = True,
        learning_rate: float = 0.0003,
        image_size: int = IMAGE_SIZE,
        model_name: str = MODEL_NAME,
        lr_schedule: str = "warmup_cosine",
        drop_path_rate: float = 0.0,
        loss_function: str = "bce",
    ) -> None:
        super().__init__()
        validate_image_size(image_size)
        if model_name not in MODEL_VARIANTS.values():
            raise ValueError("Unsupported MobileNetV4 classifier variant")
        if lr_schedule not in {"constant", "warmup_cosine"}:
            raise ValueError("Learning rate schedule must be constant or warmup_cosine")
        if loss_function not in {"bce", "focal"}:
            raise ValueError("Loss function must be bce or focal")
        self.loss_function = loss_function
        self.lr_schedule = lr_schedule
        self.model_name = model_name
        self.image_size = image_size
        self.save_hyperparameters()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=1, drop_path_rate=drop_path_rate
        )
        # PyTorch supports memory_format, but its Module.to overloads omit it.
        self.model.to(memory_format=torch.channels_last)  # ty: ignore[no-matching-overload]
        self.learning_rate = learning_rate
        self.preprocessor = BatchPreprocessor()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        # Targets and predictions are binary by construction; skip checks that synchronize CUDA.
        metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(validate_args=False),
                "bird_precision": BinaryPrecision(validate_args=False),
                "bird_recall": BinaryRecall(validate_args=False),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.contiguous(memory_format=torch.channels_last)
        return self.model(frames).squeeze(1)

    def on_after_batch_transfer(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        dataloader_idx: int,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frames, targets = batch
        if frames.dtype == torch.uint8:
            frames = self.preprocessor(frames, augment=self.trainer.training)
        return frames, targets

    def step(self, batch: tuple[torch.Tensor, torch.Tensor], split: str) -> torch.Tensor:
        frames, targets = batch
        logits = self(frames)
        loss = (
            sigmoid_focal_loss(logits, targets.float(), alpha=-1, gamma=2, reduction="mean")
            if self.loss_function == "focal"
            else self.criterion(logits, targets.float())
        )
        self.log(
            f"{split}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(targets),
            prog_bar=True,
        )
        # Pass explicit decisions to avoid TorchMetrics guessing whether logits are probabilities.
        predictions = (logits >= 0).long()
        metrics = self.train_metrics if split == "train" else self.val_metrics
        metrics.update(predictions, targets)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        return self.step(batch, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        return self.step(batch, "val")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        if self.lr_schedule == "constant":
            return optimizer
        total_steps = int(self.trainer.estimated_stepping_batches)
        if total_steps < 2 or not self.trainer.max_epochs:
            raise ValueError("Warmup/cosine training requires at least two steps and finite epochs")
        warmup_steps = min(math.ceil(total_steps / self.trainer.max_epochs), total_steps - 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, WarmupCosineSchedule(total_steps, warmup_steps)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def save_export_checkpoint(self, destination: Path) -> None:
        """Save the lightweight checkpoint consumed by the existing OpenVINO exporter."""
        torch.save(
            {
                "model_name": self.model_name,
                "classes": CLASSES,
                "image_size": self.image_size,
                "state_dict": self.model.state_dict(),
            },
            destination,
        )
