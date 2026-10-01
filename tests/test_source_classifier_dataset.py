import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

Image = pytest.importorskip("PIL.Image")
pytest.importorskip("requests")


@pytest.fixture
def builder(tmp_path: Path) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/ml/build_source_classifier_dataset.py"
    spec = importlib.util.spec_from_file_location("source_builder", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclass resolution needs a registered module during loading.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.CONFIG.output_dir = tmp_path
    module.CONFIG.size = (224, 224)
    return module


def test_source_builder_writes_classifier_labels(builder: ModuleType, tmp_path: Path) -> None:
    manifest = io.StringIO()
    image = Image.new("RGB", (200, 100), "red")
    builder.save("test", "positive", image, [(1, 1, 10, 10)], manifest)
    builder.save("test", "negative", image, [], manifest)
    records = [json.loads(line) for line in manifest.getvalue().splitlines()]
    assert [(row["label"], row["target"]) for row in records] == [("bird", 1), ("no_bird", 0)]
    for record in records:
        path = tmp_path / record["image"]
        assert path.parent.name == record["label"]
        assert path.parent.parent.name == builder.split("test", record["source_id"])
        with Image.open(path) as saved:
            assert saved.size == (224, 224)
            # Padding preserves the full rectangular image rather than cropping it.
            pixel = saved.getpixel((0, 0))
            assert isinstance(pixel, tuple)
            assert all(abs(channel - 114) < 3 for channel in pixel)
    assert not list(tmp_path.rglob("*.txt"))
    builder.save("test", "positive", image, [(1, 1, 10, 10)], manifest)
    assert len(manifest.getvalue().splitlines()) == 2


def test_coco_negative_cap_is_stable_on_resume(builder: ModuleType) -> None:
    images = {key: {} for key in range(20)}
    positive, negative = builder.select_coco_images(images, {1, 5}, 3)
    assert positive == [1, 5]
    assert negative == [0, 2, 3, 4, 6, 7]
    builder.SEEN.update(("coco2017", str(key)) for key in negative)
    assert builder.select_coco_images(images, {1, 5}, 3) == (positive, negative)
    assert builder.select_coco_images(images, set(), 3) == ([], [])


def test_coco_excess_negatives_are_preserved_outside_training(
    builder: ModuleType,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "coco.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for split in ("train2017", "val2017"):
            bundle.writestr(
                f"annotations/instances_{split}.json",
                json.dumps(
                    {
                        "categories": [{"id": 1, "name": "bird"}],
                        "images": [{"id": key} for key in (1, 2, 3, 4)],
                        "annotations": [{"category_id": 1, "image_id": 1}],
                    }
                ),
            )
    builder.CONFIG.coco_negative_ratio = 1
    manifest = tmp_path / "manifest.jsonl"
    with manifest.open("w") as stream:
        image = Image.new("RGB", (200, 100))
        builder.save("coco2017", "1", image, [(1, 1, 10, 10)], stream)
        builder.save("coco2017", "2", image, [], stream)
        builder.save("coco2017", "4", image, [], stream)
    original = [json.loads(line) for line in manifest.read_text().splitlines()]
    excess_path = Path(original[-1]["image"])
    assert builder.prune_coco_negatives(archive, manifest) == 1
    remaining = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert [row["source_id"] for row in remaining] == ["1", "2"]
    assert not (tmp_path / excess_path).exists()
    assert (tmp_path / ".excluded/coco2017" / excess_path).is_file()
    assert (tmp_path / ".excluded/coco2017/manifest-before-cap.jsonl").is_file()
    assert builder.prune_coco_negatives(archive, manifest) == 0
