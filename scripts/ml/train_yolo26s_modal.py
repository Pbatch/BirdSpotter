#!/usr/bin/env python3
"""Fine-tune YOLO26s at BirdSpotter's 1600x896 camera resolution on Modal.

Setup and run:

    modal setup
    modal secret create wandb-secret WANDB_API_KEY=...
    modal volume create birdspotter-training
    modal volume put birdspotter-training \
        data/archives/yolo_birds_1600x896.tar.gz /datasets/
    modal run scripts/ml/train_yolo26s_modal.py \
        --dataset-tar /datasets/yolo_birds_1600x896.tar.gz

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

app = modal.App("birdspotter-yolo26s")
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
    epochs: int = 10,
    batch: float = -1,
    workers: int = 8,
    patience: int = 20,
    seed: int = 42,
    sync_every: int = 5,
    resume_checkpoint: str | None = None,
    wandb_project: str = "birdspotter-yolo26s",
    wandb_entity: str | None = None,
) -> str:
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
    config = {
        "model": "yolo26s.pt",
        "camera_resolution": [CAMERA_HEIGHT, CAMERA_WIDTH],
        "imgsz": CAMERA_WIDTH,
        "rect": True,
        "epochs": epochs,
        "batch": batch,
        "workers": workers,
        "patience": patience,
        "seed": seed,
        "dataset_tar": dataset_tar,
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
    checkpoint = Path(VOLUME_ROOT) / resume_checkpoint.lstrip("/") if resume_checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found in Volume: {checkpoint}")
    model = YOLO(str(checkpoint) if checkpoint else "yolo26s.pt")

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
            imgsz=CAMERA_WIDTH,
            rect=True,
            batch=batch,
            workers=workers,
            device=0,
            patience=patience,
            seed=seed,
            deterministic=True,
            single_cls=True,
            pretrained=checkpoint is None,
            optimizer="auto",
            cos_lr=True,
            close_mosaic=10,
            amp=True,
            cache=False,
            project=str(local_runs),
            name=run_name,
            exist_ok=True,
            plots=True,
            resume=str(checkpoint) if checkpoint is not None else False,
        )
        trainer: Any = model.trainer
        if trainer is None:
            raise RuntimeError("Ultralytics completed training without creating a trainer")
        copy_checkpoint(trainer.best, "yolo26s-bird-1600x896-best.pt")
        copy_checkpoint(trainer.last, "yolo26s-bird-1600x896-last.pt")
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
    epochs: int = 10,
    batch: float = -1,
    workers: int = 8,
    patience: int = 20,
    seed: int = 42,
    sync_every: int = 5,
    resume_checkpoint: str | None = None,
    wandb_project: str = "birdspotter-yolo26s",
    wandb_entity: str | None = None,
) -> None:
    output = train.remote(
        dataset_tar=dataset_tar,
        run_name=run_name,
        epochs=epochs,
        batch=batch,
        workers=workers,
        patience=patience,
        seed=seed,
        sync_every=sync_every,
        resume_checkpoint=resume_checkpoint,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )
    download_command = f"modal volume get {VOLUME_NAME} {output} ./trained-model"
    print(f"Training complete. Download with:\n{download_command}")
