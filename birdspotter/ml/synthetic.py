"""Plan random square crops of a seed image, each paired with a bird-added edit."""

import random
from dataclasses import asdict, dataclass

BIRD_SPECIES = [
    "robin",
    "blue tit",
    "great tit",
    "house sparrow",
    "blackbird",
    "goldfinch",
    "wood pigeon",
    "starling",
    "magpie",
    "chaffinch",
    "wren",
    "dunnock",
    "collared dove",
    "long-tailed tit",
]
BIRD_SIZES = ["small and distant", "medium-sized", "close to the camera"]
BIRD_PROMPT = (
    "Keep the scene, framing and camera exactly unchanged. Add one realistic {species}, "
    "{size}, somewhere it would naturally be in this scene. Match the photo's lighting, "
    "focus, perspective and shadows."
)


@dataclass(frozen=True, slots=True)
class CropPlan:
    index: int
    x: int
    y: int
    size: int
    bird_prompt: str
    seed: int

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def plan_crops(  # noqa: PLR0913
    width: int,
    height: int,
    *,
    count: int,
    min_fraction: float,
    max_fraction: float,
    seed: int,
) -> list[CropPlan]:
    """Sample square crops whose side is a fraction of the seed image's shorter side."""
    if count < 1:
        raise ValueError("Count must be positive")
    if not 0 < min_fraction <= max_fraction <= 1:
        raise ValueError("Crop fractions must satisfy 0 < min <= max <= 1")
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security
    short_side = min(width, height)
    plans = []
    for index in range(count):
        size = round(short_side * rng.uniform(min_fraction, max_fraction))
        x = rng.randint(0, width - size)
        y = rng.randint(0, height - size)
        bird_prompt = BIRD_PROMPT.format(
            species=rng.choice(BIRD_SPECIES),
            size=rng.choice(BIRD_SIZES),
        )
        plans.append(CropPlan(index, x, y, size, bird_prompt, rng.randrange(2**31)))
    return plans
