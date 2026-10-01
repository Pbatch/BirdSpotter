# Optional ML dependencies are checked before importing the training module.
# ruff: noqa: E402

from itertools import pairwise
from pathlib import Path
from unittest.mock import Mock

import pytest

lightning = pytest.importorskip("lightning")
torch = pytest.importorskip("torch")

from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from birdspotter.ml import classifier, classifier_training
from birdspotter.ml.classifier import transform, validate_image_size
from birdspotter.ml.preprocessing import BatchPreprocessor, TorchvisionFrameLoader
from birdspotter.ml.training_job import VolumeCheckpoint


def test_focal_loss_downweights_easy_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        classifier_training.timm,
        "create_model",
        lambda *_args, **_kwargs: torch.nn.Linear(1, 1),
    )
    model = classifier_training.BirdClassifier(pretrained=False, loss_function="focal")
    logits = torch.tensor([-4.0, 0.0, 4.0], requires_grad=True)
    targets = torch.tensor([0, 1, 1])
    monkeypatch.setattr(model, "forward", lambda _frames: logits)
    monkeypatch.setattr(model, "log", Mock())
    monkeypatch.setattr(model, "log_dict", Mock())
    loss = model.step((torch.zeros(3, 1), targets), "train")
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction="none"
    )
    probabilities = logits.sigmoid()
    correct_probability = torch.where(targets.bool(), probabilities, 1 - probabilities)
    expected = ((1 - correct_probability).square() * bce).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0].abs() < logits.grad[1].abs()
    assert model.hparams["loss_function"] == "focal"


@pytest.mark.parametrize("model_name", list(classifier.MODEL_VARIANTS.values()))
def test_lightning_checkpoint_resume_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_name: str
) -> None:
    def tiny_model(*_args: object, **_kwargs: object) -> torch.nn.Module:
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 1))

    monkeypatch.setattr(classifier_training.timm, "create_model", tiny_model)
    frames = torch.randint(0, 256, (4, 3, 2, 2), dtype=torch.uint8)
    targets = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(frames, targets), batch_size=2)
    commit = Mock()
    callback = VolumeCheckpoint(tmp_path, commit)
    trainer = lightning.Trainer(
        accelerator="cpu",
        max_epochs=1,
        callbacks=[callback],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        classifier_training.BirdClassifier(pretrained=False, model_name=model_name), loader, loader
    )
    assert trainer.global_step == 2
    assert commit.call_count >= 2
    assert Path(callback.last_model_path).is_file()
    assert "val_bird_precision" in trainer.callback_metrics
    assert "val_bird_recall" in trainer.callback_metrics
    checkpoint = torch.load(callback.last_model_path, weights_only=False)
    assert checkpoint["optimizer_states"][0]["state"]
    assert checkpoint["lr_schedulers"][0]["last_epoch"] == 2
    resumed = classifier_training.BirdClassifier(pretrained=False, model_name=model_name)
    resumed_trainer = lightning.Trainer(
        accelerator="cpu",
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    resumed_trainer.fit(resumed, loader, loader, ckpt_path=callback.last_model_path)
    assert resumed_trainer.global_step == 4
    assert resumed_trainer.lr_scheduler_configs[0].scheduler.last_epoch == 4
    assert resumed_trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(3e-6)
    best = classifier_training.BirdClassifier.load_from_checkpoint(
        callback.best_model_path,
        pretrained=False,
    )
    best.save_export_checkpoint(tmp_path / "best.pt")
    exported = torch.load(tmp_path / "best.pt", weights_only=True)
    assert exported["classes"] == ["bird"]
    assert exported["model_name"] == model_name
    assert exported["image_size"] == 640
    assert set(exported["state_dict"]) == set(best.model.state_dict())
    assert best(best.preprocessor(frames)).shape == (4,)
    assert classifier_training.bird_target(0) == 1
    assert classifier_training.bird_target(1) == 0


@pytest.mark.parametrize("image_size", [224, 640])
def test_training_transform_resolution(image_size: int) -> None:
    image = Image.new("RGB", (800, 400), "red")
    tensor = transform(image, image_size=image_size)
    assert tensor.shape == (3, image_size, image_size)


def test_reject_invalid_training_resolution() -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        validate_image_size(641)


@pytest.mark.parametrize("dimensions", [(64, 64), (96, 32)])
def test_torchvision_decode_and_validation_normalization(
    tmp_path: Path, dimensions: tuple[int, int]
) -> None:
    image = Image.new("RGB", dimensions, (70, 110, 150))
    path = tmp_path / "frame.png"
    image.save(path)
    decoded = TorchvisionFrameLoader(64)(str(path))
    assert decoded.dtype == torch.uint8
    assert decoded.shape == (3, 64, 64)
    preprocessor = BatchPreprocessor()
    normalized = preprocessor(decoded.unsqueeze(0))
    torch.testing.assert_close(normalized[0], transform(image, image_size=64))
    assert torch.equal(normalized, preprocessor(decoded.unsqueeze(0)))
    assert normalized.is_contiguous(memory_format=torch.channels_last)
    assert preprocessor.state_dict() == {}


def test_batch_augmentation_has_independent_per_image_randomness() -> None:
    torch.manual_seed(42)
    frames = torch.full((32, 3, 32, 64), 80, dtype=torch.uint8)
    frames[:, :, :, 32:] = 160
    preprocessor = BatchPreprocessor()
    augmented = preprocessor(frames, augment=True)
    values = (augmented * preprocessor.std + preprocessor.mean) * 255
    left = values[:, :, :, :32].mean(dim=(1, 2, 3))
    right = values[:, :, :, 32:].mean(dim=(1, 2, 3))
    assert (left > right).any()
    assert (left < right).any()
    assert values.mean(dim=(1, 2, 3)).unique().numel() > 1
    assert torch.isfinite(augmented).all()
    assert values.min() >= 0
    assert values.max() <= 255


def test_warmup_cosine_learning_rates_and_resume() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, classifier_training.WarmupCosineSchedule(total_steps=100, warmup_steps=10)
    )
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
        if scheduler.last_epoch == 50:
            optimizer_state = optimizer.state_dict()
            scheduler_state = scheduler.state_dict()
    assert rates[0] == pytest.approx(3e-5)
    assert rates[10] == pytest.approx(3e-4)
    assert rates[100] == pytest.approx(3e-6)
    assert all(left < right for left, right in pairwise(rates[:11]))
    assert all(left > right for left, right in pairwise(rates[10:]))
    resumed_optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
        resumed_optimizer,
        classifier_training.WarmupCosineSchedule(total_steps=200, warmup_steps=20),
    )
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    for expected in rates[51:]:
        resumed_optimizer.step()
        resumed_scheduler.step()
        assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(expected)


def test_augmentations_apply_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("RGB", (64, 32), "red")
    image.paste("blue", (32, 0, 64, 32))
    calls = []

    def flip_and_adjust(frame: Image.Image) -> Image.Image:
        calls.append(frame)
        return frame.transpose(Image.Transpose.FLIP_LEFT_RIGHT).point(lambda value: value * 0.9)

    monkeypatch.setattr(classifier, "TRAIN_AUGMENTATIONS", flip_and_adjust)
    baseline = transform(image, image_size=64)
    assert calls == []
    assert torch.equal(baseline, transform(image, image_size=64))
    augmented = transform(image, image_size=64, augment=True)
    expected = transform(flip_and_adjust(image), image_size=64)
    assert torch.equal(augmented, expected)
    assert not torch.equal(augmented, baseline)
    assert augmented.shape == baseline.shape
