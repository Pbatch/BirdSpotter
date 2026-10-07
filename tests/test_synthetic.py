from birdspotter.ml.synthetic import BIRD_SPECIES, plan_crops


def test_crops_are_square_and_inside_the_image() -> None:
    plans = plan_crops(1920, 1080, count=200, min_fraction=0.3, max_fraction=1.0, seed=1)
    for plan in plans:
        assert 324 <= plan.size <= 1080
        assert 0 <= plan.x <= 1920 - plan.size
        assert 0 <= plan.y <= 1080 - plan.size
        assert any(species in plan.bird_prompt for species in BIRD_SPECIES)


def test_plan_is_deterministic() -> None:
    kwargs = {"count": 50, "min_fraction": 0.5, "max_fraction": 1.0}
    assert plan_crops(100, 100, seed=7, **kwargs) == plan_crops(100, 100, seed=7, **kwargs)
    assert plan_crops(100, 100, seed=7, **kwargs) != plan_crops(100, 100, seed=8, **kwargs)
