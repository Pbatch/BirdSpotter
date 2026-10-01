"""Persistent W&B run identity for resumable Modal training."""

import json
from pathlib import Path

from lightning.pytorch.loggers import WandbLogger
from wandb.util import generate_id


def create_wandb_logger(
    output: Path,
    *,
    run_name: str,
    project: str,
    entity: str | None = None,
) -> WandbLogger:
    identity_path = output / "wandb-run.json"
    if identity_path.is_file():
        identity = json.loads(identity_path.read_text())
        if identity["project"] != project or identity["entity"] != entity:
            raise ValueError("Resume requires the original W&B project and entity")
    else:
        identity = {"id": generate_id(), "project": project, "entity": entity}
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    return WandbLogger(
        project=project,
        entity=entity,
        name=run_name,
        id=identity["id"],
        resume="allow",
        save_dir=str(output),
        log_model=False,
    )
