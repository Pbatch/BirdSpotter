#!/usr/bin/env python3
"""Fine-tune YOLO26 models for BirdSpotter on Modal.

Setup and run:

    modal setup
    modal secret create wandb-secret WANDB_API_KEY=...
    modal volume create birdspotter-training
    modal volume put birdspotter-training \
        data/archives/yolo_birds_1600x896.tar.gz /datasets/
    modal run --detach scripts/ml/train_yolo26s_modal.py \
        --dataset-tar /datasets/yolo_birds_1600x896.tar.gz \
        --image-size 1600

For a square 640x640 fine-tune initialized from an existing checkpoint:

    modal run --detach scripts/ml/train_yolo26s_modal.py \
        --dataset-tar /datasets/yolo_birds_640x640.tar.gz \
        --run-name yolo26s-bird-640x640-from-1600-10e \
        --image-size 640 --square \
        --initial-checkpoint /runs/yolo26s-bird-1600x896-10e/yolo26s-bird-1600x896-best.pt

For a pretrained YOLO26-L comparison run, omit the S checkpoint and select
the larger base model:

    modal run --detach scripts/ml/train_yolo26s_modal.py \
        --dataset-tar /datasets/yolo_birds_640x640.tar.gz \
        --run-name yolo26l-bird-640x640-native-10e \
        --base-model yolo26l.pt --image-size 640 --square

Download the resulting checkpoints:

    modal volume get birdspotter-training /runs/<run-name> ./trained-model
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import modal
import wandb
import yaml
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

CAMERA_WIDTH = 1600
CAMERA_HEIGHT = 896
VOLUME_ROOT = "/mnt/birdspotter"
VOLUME_NAME = "birdspotter-training"

app = modal.App("birdspotter-yolo26-training")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("libgl1", "libglib2.0-0", "pigz")
    .uv_pip_install(
        "opencv-python==4.14.0.94",
        "pillow==12.3.0",
        "pyyaml==6.0.3",
        "ultralytics==8.4.140",
        "wandb==0.22.3",
    )
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=120,
)
def check_wandb_secret() -> bool:
    """Verify the injected W&B API key without displaying it."""
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("wandb-secret does not contain WANDB_API_KEY")
    if not wandb.login(key=api_key, verify=True, relogin=True):
        raise RuntimeError("Weights & Biases rejected WANDB_API_KEY")
    print("W&B authentication verified")
    return True


@app.function(
    image=image,
    gpu="H200",
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={VOLUME_ROOT: volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(  # noqa: C901, PLR0913, PLR0915
    dataset_tar: str,
    run_name: str,
    *,
    base_model: str = "yolo26s.pt",
    epochs: int = 10,
    batch: float = -1,
    workers: int = 8,
    patience: int = 20,
    seed: int = 42,
    sync_every: int = 5,
    image_size: int,
    rect: bool = True,
    initial_checkpoint: str | None = None,
    resume_checkpoint: str | None = None,
    wandb_project: str = "birdspotter-yolo26s",
    wandb_entity: str | None = None,
) -> str:
    if image_size < 32 or image_size % 32:
        raise ValueError("Image size must be at least 32 and divisible by 32")
    if not base_model.endswith(".pt") or Path(base_model).name != base_model:
        raise ValueError("Base model must be a checkpoint name such as yolo26s.pt")
    if initial_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError("Choose either an initial checkpoint or a resume checkpoint")
    archive = Path(VOLUME_ROOT) / dataset_tar.lstrip("/")
    if not archive.is_file():
        raise FileNotFoundError(f"Upload the dataset tar to the Modal Volume first: {archive}")

    dataset_dir = Path("/tmp/birdspotter-dataset")  # noqa: S108
    print(f"Extracting {archive} to local container storage")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603
        [
            "/bin/tar",
            "--extract",
            "--file",
            str(archive),
            "--directory",
            str(dataset_dir),
            "--use-compress-program=pigz -d",
            "--no-same-owner",
            "--no-same-permissions",
        ],
        check=True,
    )
    yaml_files = sorted(dataset_dir.rglob("data.yaml"))
    if len(yaml_files) != 1:
        raise RuntimeError(f"Expected exactly one data.yaml, found {len(yaml_files)}")

    data = yaml.safe_load(yaml_files[0].read_text())
    data.update(
        path=str(yaml_files[0].parent.resolve()),
        train=data.get("train", "images/train"),
        val=data.get("val", "images/val"),
        names={0: "bird"},
    )
    local_yaml = Path("/tmp/birdspotter-data.yaml")  # noqa: S108
    local_yaml.write_text(yaml.safe_dump(data, sort_keys=False))

    drive_run = Path(VOLUME_ROOT) / "runs" / run_name
    drive_run.mkdir(parents=True, exist_ok=True)
    local_runs = Path("/tmp/birdspotter-runs")  # noqa: S108
    checkpoint_argument = initial_checkpoint or resume_checkpoint
    config = {
        "model": checkpoint_argument or base_model,
        "base_model": base_model,
        "camera_resolution": [CAMERA_HEIGHT, CAMERA_WIDTH],
        "imgsz": image_size,
        "rect": rect,
        "epochs": epochs,
        "batch": batch,
        "workers": workers,
        "patience": patience,
        "seed": seed,
        "dataset_tar": dataset_tar,
        "initial_checkpoint": initial_checkpoint,
        "resume_checkpoint": resume_checkpoint,
    }
    (drive_run / "training-config.json").write_text(json.dumps(config, indent=2) + "\n")
    volume.commit()

    SETTINGS.update({"wandb": True})
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config=config,
        resume="allow",
    )
    initial = Path(VOLUME_ROOT) / initial_checkpoint.lstrip("/") if initial_checkpoint else None
    resume = Path(VOLUME_ROOT) / resume_checkpoint.lstrip("/") if resume_checkpoint else None
    for label, checkpoint in (("Initial", initial), ("Resume", resume)):
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found in Volume: {checkpoint}")
    source_checkpoint = initial or resume
    model = YOLO(str(source_checkpoint) if source_checkpoint else base_model)

    def copy_checkpoint(source: Path | str | None, name: str) -> None:
        if source is None:
            return
        source = Path(source)
        if source.is_file():
            temporary = (drive_run / name).with_suffix(".pt.part")
            shutil.copy2(source, temporary)
            temporary.replace(drive_run / name)

    def sync_checkpoints(trainer: Any) -> None:  # noqa: ANN401
        if sync_every > 0 and (trainer.epoch + 1) % sync_every == 0:
            copy_checkpoint(trainer.last, "last.pt")
            copy_checkpoint(trainer.best, "best.pt")
            volume.commit()

    model.add_callback("on_fit_epoch_end", sync_checkpoints)
    try:
        model.train(
            data=str(local_yaml),
            epochs=epochs,
            imgsz=image_size,
            rect=rect,
            batch=batch,
            workers=workers,
            device=0,
            patience=patience,
            seed=seed,
            deterministic=True,
            single_cls=True,
            pretrained=source_checkpoint is None,
            optimizer="auto",
            cos_lr=True,
            close_mosaic=10,
            amp=True,
            cache=False,
            project=str(local_runs),
            name=run_name,
            exist_ok=True,
            plots=True,
            resume=str(resume) if resume is not None else False,
        )
        trainer: Any = model.trainer
        if trainer is None:
            raise RuntimeError("Ultralytics completed training without creating a trainer")
        resolution = (
            f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
            if rect and image_size == CAMERA_WIDTH
            else f"{image_size}x{image_size}"
        )
        model_stem = Path(base_model).stem
        copy_checkpoint(trainer.best, f"{model_stem}-bird-{resolution}-best.pt")
        copy_checkpoint(trainer.last, f"{model_stem}-bird-{resolution}-last.pt")
        for filename in ("results.csv", "args.yaml", "results.png"):
            source = Path(trainer.save_dir) / filename
            if source.is_file():
                shutil.copy2(source, drive_run / filename)
        volume.commit()

        print(f"Saved checkpoints to Modal Volume: {drive_run}")
        return f"/runs/{run_name}"
    finally:
        if wandb.run is not None:
            wandb.finish()


@app.local_entrypoint()
def main(  # noqa: PLR0913
    dataset_tar: str = "/datasets/yolo_birds_1600x896.tar.gz",
    run_name: str = "yolo26s-bird-1600x896",
    *,
    base_model: str = "yolo26s.pt",
    epochs: int = 10,
    batch: float = -1,
    workers: int = 8,
    patience: int = 20,
    seed: int = 42,
    sync_every: int = 5,
    image_size: int,
    square: bool = False,
    initial_checkpoint: str | None = None,
    resume_checkpoint: str | None = None,
    wandb_project: str = "birdspotter-yolo26s",
    wandb_entity: str | None = None,
) -> None:
    call = train.spawn(
        dataset_tar=dataset_tar,
        run_name=run_name,
        base_model=base_model,
        epochs=epochs,
        batch=batch,
        workers=workers,
        patience=patience,
        seed=seed,
        sync_every=sync_every,
        image_size=image_size,
        rect=not square,
        initial_checkpoint=initial_checkpoint,
        resume_checkpoint=resume_checkpoint,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )
    output = f"/runs/{run_name}"
    download_command = f"modal volume get {VOLUME_NAME} {output} ./trained-model"
    print(f"Training submitted as Function Call {call.object_id}.")
    print("When launched with `modal run --detach`, it will continue after this command exits.")
    print("Follow it with: modal app list; modal app logs APP_ID --follow --timestamps")
    print(f"After training completes, download with:\n{download_command}")
