"""Train MobileNetV4 with Lightning on Modal and export the best model to OpenVINO."""

import json
from datetime import UTC, datetime
from pathlib import Path

import modal

app = modal.App("birdspotter-mobilenetv4-training")
volume = modal.Volume.from_name("birdspotter-training", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch",
        "torchvision",
        "timm==1.0.28",
        "lightning==2.6.6",
        "wandb==0.22.3",
        "torchmetrics==1.9.0",
        "pillow==12.3.0",
        "opencv-python-headless==4.14.0.94",
        "openvino==2026.2.1",
        "numpy==2.4.2",
    )
    .add_local_python_source("birdspotter")
)


with image.imports():
    from birdspotter.ml.training_job import ClassifierTrainingJob, TrainingConfig


@app.cls(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=16384,
    timeout=86400,
    volumes={"/mnt/birdspotter": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
class ModalClassifierTrainer:
    @modal.method()
    def train(  # noqa: PLR0913, PLR0917
        self,
        dataset_tar: str,
        run_name: str,
        epochs: int,
        batch_size: int,
        resume_checkpoint: str = "",
        precision: str = "bf16-mixed",
        image_size: int = 640,
        wandb_project: str = "birdspotter-mobilenetv4",
        wandb_entity: str = "",
        model_variant: str = "small",
        learning_rate: float = 0.0001,
        drop_path_rate: float = 0.1,
        loss_function: str = "bce",
        extra_dataset_tars: str = "",
    ) -> None:
        config = TrainingConfig(
            dataset_tar,
            run_name,
            epochs,
            batch_size,
            resume_checkpoint,
            precision,
            image_size,
            wandb_project,
            wandb_entity,
            model_variant,
            learning_rate,
            drop_path_rate,
            loss_function,
            extra_dataset_tars,
        )
        ClassifierTrainingJob(config, Path("/mnt/birdspotter"), volume.commit).run()


@app.local_entrypoint()
def main(  # noqa: PLR0913, PLR0917
    dataset_tar: str = "/datasets/classifier_birds-clean-20261006.tar.gz",
    run_name: str = "mobilenetv4-bird-640",
    epochs: int = 10,
    batch_size: int = 128,
    resume_checkpoint: str = "",
    precision: str = "bf16-mixed",
    image_size: int = 640,
    wandb_project: str = "birdspotter-mobilenetv4",
    wandb_entity: str = "",
    model_variant: str = "small",
    learning_rate: float = 0.0001,
    drop_path_rate: float = 0.1,
    loss_function: str = "bce",
    extra_dataset_tars: str = "",
) -> None:
    submission_dir = Path("data/modal-runs")
    submission_dir.mkdir(parents=True, exist_ok=True)
    call = ModalClassifierTrainer().train.spawn(
        dataset_tar,
        run_name,
        epochs,
        batch_size,
        resume_checkpoint,
        precision,
        image_size,
        wandb_project,
        wandb_entity,
        model_variant,
        learning_rate,
        drop_path_rate,
        loss_function,
        extra_dataset_tars,
    )
    dashboard_url = app.get_dashboard_url()
    print(f"Submitted training run: {run_name}")
    print(f"Modal call ID: {call.object_id}")
    print(f"View run: {dashboard_url}")
    submission = {
        "submitted_at": datetime.now(UTC).isoformat(),
        "app_id": app.app_id,
        "call_id": call.object_id,
        "dashboard_url": dashboard_url,
        "run_name": run_name,
        "model_variant": model_variant,
        "learning_rate": learning_rate,
        "drop_path_rate": drop_path_rate,
        "loss_function": loss_function,
        "dataset_tar": dataset_tar,
        "extra_dataset_tars": extra_dataset_tars,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "precision": precision,
        "resume_checkpoint": resume_checkpoint,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
    }
    submission_path = submission_dir / f"{call.object_id}.json"
    submission_path.write_text(json.dumps(submission, indent=2) + "\n")
    print(f"Submission saved: {submission_path}")
