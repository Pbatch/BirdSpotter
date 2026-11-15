import numpy as np
import pytest

from birdspotter.motion import MotionGate, grey_thumbnail


def scene(value: int = 100, shape: tuple[int, int] = (240, 240)) -> np.ndarray:
    return np.full((*shape, 3), value, np.uint8)


def with_bird(image: np.ndarray, value: int = 0) -> np.ndarray:
    image = image.copy()
    image[100:120, 100:120] = value
    return image


def test_thumbnail_keeps_aspect_ratio() -> None:
    assert grey_thumbnail(scene(shape=(240, 480))).shape == (60, 120)


def test_gate_stays_closed_until_the_background_is_full_and_after_reset() -> None:
    gate = MotionGate(threshold=10, history_frames=3, sample_every=1)

    assert [gate.check(with_bird(scene())) for _ in range(3)] == [None, None, None]
    assert gate.ready
    gate.reset()
    assert not gate.ready
    assert gate.check(with_bird(scene())) is None


def test_bird_scores_its_contrast_against_the_background() -> None:
    gate = MotionGate(threshold=50, history_frames=4)
    for _ in range(4):
        gate.score(scene())

    assert gate.check(with_bird(scene())) == pytest.approx(100)


def test_median_ignores_a_bird_in_a_minority_of_past_frames() -> None:
    gate = MotionGate(threshold=50, history_frames=6, sample_every=1)
    for image in [scene(), with_bird(scene()), scene(), scene(), with_bird(scene()), scene()]:
        gate.score(image)

    assert gate.score(with_bird(scene())) == pytest.approx(100)


def test_uniform_lighting_drift_scores_only_the_drift() -> None:
    gate = MotionGate(threshold=50, history_frames=4, sample_every=1)
    for value in (96, 98, 100, 102):
        gate.score(scene(value))

    assert gate.check(scene(104)) is None


def test_resolution_change_restarts_the_background() -> None:
    gate = MotionGate(threshold=10, history_frames=2, sample_every=1)
    gate.score(scene())
    gate.score(scene())
    assert gate.ready

    assert gate.score(scene(shape=(240, 480))) == 0.0
    assert not gate.ready


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": 0}, "grey-level"),
        ({"history_frames": 1}, "two frames"),
        ({"history_frames": 4, "sample_every": 5}, "sampling step"),
    ],
)
def test_invalid_settings_are_rejected(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MotionGate(**kwargs)  # ty: ignore[invalid-argument-type]
