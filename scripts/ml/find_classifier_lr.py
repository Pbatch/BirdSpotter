"""Run Lightning's learning-rate range test on the local classifier dataset."""

import argparse
import csv
import json
from pathlib import Path

import lightning
import matplotlib as mpl
import torch
from lightning.pytorch.tuner import Tuner
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from birdspotter.ml.classifier import MODEL_VARIANTS
from birdspotter.ml.classifier_training import BirdClassifier, bird_target
from birdspotter.ml.preprocessing import TorchvisionFrameLoader

mpl.use("Agg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/classifier_birds"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/benchmarks/mobilenetv4-small-lr-finder")
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--max-lr", type=float, default=1e-2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lightning.seed_everything(42, workers=True)
    torch.set_num_threads(4)
    dataset = ImageFolder(
        args.dataset / "train",
        loader=TorchvisionFrameLoader(640),
        target_transform=bird_target,
    )
    if dataset.class_to_idx != {"bird": 0, "no_bird": 1}:
        raise ValueError("Expected bird/ and no_bird/ training folders")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    model = BirdClassifier(model_name=MODEL_VARIANTS["small"], lr_schedule="constant")
    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        benchmark=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        default_root_dir=args.output,
        max_epochs=1,
        max_steps=args.steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    finder = Tuner(trainer).lr_find(
        model,
        train_dataloaders=loader,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_training=args.steps,
        update_attr=False,
    )
    if finder is None:
        raise RuntimeError("Lightning did not return LR finder results")
    result = {
        "model_name": MODEL_VARIANTS["small"],
        "pretrained": True,
        "gpu": torch.cuda.get_device_name(),
        "image_size": 640,
        "batch_size": args.batch_size,
        "precision": "bf16-mixed",
        "min_lr": args.min_lr,
        "max_lr": args.max_lr,
        "requested_steps": args.steps,
        "completed_steps": len(finder.results["lr"]),
        "suggested_lr": finder.suggestion(),
        "results": finder.results,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "results.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["learning_rate", "smoothed_loss"])
        writer.writerows(zip(finder.results["lr"], finder.results["loss"], strict=True))
    figure = finder.plot(suggest=True)
    if not isinstance(figure, Figure):
        raise TypeError("Lightning did not return an LR curve figure")
    figure.savefig(args.output / "lr-curve.png", dpi=160, bbox_inches="tight")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
