# Optional ML dependencies are checked before importing the Modal launcher.

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("modal")
pytest.importorskip("lightning")
pytest.importorskip("timm")

from scripts.ml import train_mobilenetv4_modal as launcher


def test_training_submission_saves_call_id_without_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    call = Mock(object_id="fc-test")
    trainer = Mock()
    trainer.train.spawn.return_value = call
    monkeypatch.setattr(launcher, "ModalClassifierTrainer", Mock(return_value=trainer))
    monkeypatch.setattr(
        launcher,
        "app",
        Mock(app_id="ap-test", get_dashboard_url=Mock(return_value="https://modal.com/id/ap-test")),
    )
    launcher.main(
        run_name="small-test",
        model_variant="small",
        epochs=3,
        batch_size=128,
        learning_rate=0.0001,
        drop_path_rate=0.1,
        loss_function="focal",
    )
    trainer.train.spawn.assert_called_once()
    trainer.train.remote.assert_not_called()
    call.get.assert_not_called()
    submission = json.loads((tmp_path / "data/modal-runs/fc-test.json").read_text())
    assert submission["call_id"] == "fc-test"
    assert submission["app_id"] == "ap-test"
    assert submission["run_name"] == "small-test"
    assert submission["model_variant"] == "small"
    assert submission["epochs"] == 3
    assert submission["batch_size"] == 128
    assert submission["learning_rate"] == 0.0001
    assert trainer.train.spawn.call_args.args[3] == 128
    assert submission["drop_path_rate"] == 0.1
    assert submission["loss_function"] == "focal"
    assert trainer.train.spawn.call_args.args[-4:] == (0.0001, 0.1, "focal", "")
    assert submission["extra_dataset_tars"] == ""
