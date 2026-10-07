"""Class-based orchestration for classifier training, logging and export."""

import os
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import lightning
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from birdspotter.ml.classifier import (
    MODEL_NAME,
    MODEL_VARIANTS,
    export_classifier,
    validate_image_size,
)
from birdspotter.ml.classifier_training import BirdClassifier, bird_target
from birdspotter.ml.preprocessing import TorchvisionFrameLoader
from birdspotter.ml.training_logging import create_wandb_logger


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    dataset_tar: str
    run_name: str
    epochs: int
    batch_size: int
    resume_checkpoint: str = ""
    precision: str = "bf16-mixed"
    image_size: int = 640
    wandb_project: str = "birdspotter-mobilenetv4"
    wandb_entity: str = ""
    model_variant: str = "small"
    learning_rate: float = 0.0001
    drop_path_rate: float = 0.1
    loss_function: str = "bce"
    # Comma-separated train-only tars (e.g. private synthetic data) extracted over dataset_tar.
    extra_dataset_tars: str = ""

    def __post_init__(self) -> None:
        if not 0 < self.learning_rate < float("inf"):
            raise ValueError("Learning rate must be positive and finite")
        if not 0 <= self.drop_path_rate < 1:
            raise ValueError("Drop path rate must be between zero (inclusive) and one (exclusive)")
        if self.loss_function not in {"bce", "focal"}:
            raise ValueError("Loss function must be bce or focal")

    @property
    def model_name(self) -> str:
        try:
            return MODEL_VARIANTS[self.model_variant]
        except KeyError as error:
            raise ValueError("Model variant must be small or large") from error


class VolumeCheckpoint(ModelCheckpoint):
    def __init__(self, output: Path, commit: Callable[[], None]) -> None:
        super().__init__(
            dirpath=output / "checkpoints",
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            enable_version_counter=False,
        )
        self.commit = commit

    def _save_checkpoint(self, trainer: lightning.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        self.commit()


@dataclass(slots=True)
class ClassifierTrainingJob:
    config: TrainingConfig
    root: Path
    commit: Callable[[], None]

    def extract_datasets(self, data: Path) -> None:
        """Extract the dataset tar, then any extra train-only tars, into one ImageFolder tree."""
        tars = [self.config.dataset_tar, *self.config.extra_dataset_tars.split(",")]
        for tar in filter(None, (tar.strip() for tar in tars)):
            with tarfile.open(self.root / tar.lstrip("/")) as source:
                source.extractall(data, filter="data")

    def run(self) -> None:
        dataset_tar = self.config.dataset_tar
        run_name = self.config.run_name
        epochs = self.config.epochs
        batch_size = self.config.batch_size
        resume_checkpoint = self.config.resume_checkpoint
        precision = self.config.precision
        image_size = self.config.image_size
        wandb_project = self.config.wandb_project
        wandb_entity = self.config.wandb_entity
        if epochs < 1 or batch_size < 1 or Path(run_name).name != run_name:
            raise ValueError("Use positive epochs/batch size and a simple run name")
        if precision not in {"32-true", "16-mixed", "bf16-mixed"}:
            raise ValueError("Precision must be 32-true, 16-mixed, or bf16-mixed")
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("wandb-secret must contain WANDB_API_KEY")
        validate_image_size(image_size)
        model_name = self.config.model_name
        root = self.root
        checkpoint = root / resume_checkpoint.lstrip("/") if resume_checkpoint else None
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if checkpoint is not None:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["hyper_parameters"].get("model_name", MODEL_NAME) != model_name:
                raise ValueError("Resume model variant must match the checkpoint")
            if saved["hyper_parameters"].get("image_size", 224) != image_size:
                raise ValueError("Resume image size must match the checkpoint training resolution")
        output = root / "runs" / run_name
        output.mkdir(parents=True, exist_ok=checkpoint is not None)
        lightning.seed_everything(42, workers=True)

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            self.extract_datasets(data)
            datasets = {
                split: ImageFolder(
                    data / split,
                    loader=TorchvisionFrameLoader(image_size),
                    target_transform=bird_target,
                )
                for split in ("train", "val")
            }
            for dataset in datasets.values():
                if dataset.class_to_idx != {"bird": 0, "no_bird": 1}:
                    raise ValueError("Each split must contain bird/ and no_bird/ directories")
            loaders = {
                split: DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=split == "train",
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True,
                )
                for split, dataset in datasets.items()
            }
            callback = VolumeCheckpoint(output, self.commit)
            wandb_logger = create_wandb_logger(
                output, run_name=run_name, project=wandb_project, entity=wandb_entity or None
            )
            self.commit()
            trainer = lightning.Trainer(
                accelerator="gpu",
                devices=1,
                max_epochs=epochs,
                precision=precision,
                benchmark=True,
                callbacks=[
                    callback,
                    LearningRateMonitor(logging_interval="step"),
                    EarlyStopping(
                        monitor="val_loss",
                        mode="min",
                        patience=3,
                        check_on_train_epoch_end=False,
                    ),
                ],
                logger=[CSVLogger(str(output), name="logs"), wandb_logger],
                default_root_dir=output,
            )
            with wandb_logger.experiment:
                wandb_logger.log_hyperparams(
                    {
                        "dataset_tar": dataset_tar,
                        "extra_dataset_tars": self.config.extra_dataset_tars,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "precision": precision,
                        "image_size": image_size,
                        "model_name": model_name,
                        "learning_rate": self.config.learning_rate,
                        "drop_path_rate": self.config.drop_path_rate,
                        "early_stopping_patience": 3,
                        "loss_function": self.config.loss_function,
                        "focal_gamma": 2 if self.config.loss_function == "focal" else None,
                        "focal_alpha": -1 if self.config.loss_function == "focal" else None,
                        "channels_last": True,
                        "cudnn_benchmark": True,
                        "preprocessing": "torchvision_decode_uint8_gpu_augmentation",
                        "horizontal_flip_probability": 0.5,
                        "brightness_jitter": 0.1,
                        "contrast_jitter": 0.1,
                    }
                )
                trainer.fit(
                    BirdClassifier(
                        pretrained=checkpoint is None,
                        image_size=image_size,
                        model_name=model_name,
                        learning_rate=self.config.learning_rate,
                        drop_path_rate=self.config.drop_path_rate,
                        loss_function=self.config.loss_function,
                        lr_schedule=(
                            saved["hyper_parameters"].get("lr_schedule", "constant")
                            if checkpoint is not None
                            else "warmup_cosine"
                        ),
                    ),
                    train_dataloaders=loaders["train"],
                    val_dataloaders=loaders["val"],
                    ckpt_path=str(checkpoint) if checkpoint is not None else None,
                )
                best = BirdClassifier.load_from_checkpoint(
                    callback.best_model_path,
                    pretrained=False,
                    map_location="cpu",
                    image_size=image_size,
                    model_name=model_name,
                )
                best.save_export_checkpoint(output / "best.pt")
                self.commit()
                export_classifier(output / "best.pt", output / "openvino")
                self.commit()
