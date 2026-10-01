# Optional ML libraries are checked before importing the logging helper.

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("lightning")
pytest.importorskip("wandb")

from birdspotter.ml import training_logging


def test_wandb_run_identity_survives_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = Mock()
    monkeypatch.setattr(training_logging, "WandbLogger", logger)
    training_logging.create_wandb_logger(tmp_path, run_name="bird", project="birds")
    initial_id = logger.call_args.kwargs["id"]
    assert initial_id
    assert logger.call_args.kwargs["resume"] == "allow"
    assert json.loads((tmp_path / "wandb-run.json").read_text())["id"] == initial_id
    training_logging.create_wandb_logger(tmp_path, run_name="bird", project="birds")
    assert logger.call_args.kwargs["id"] == initial_id
    with pytest.raises(ValueError, match="original W&B project"):
        training_logging.create_wandb_logger(tmp_path, run_name="bird", project="different")
