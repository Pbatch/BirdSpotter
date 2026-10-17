"""Rank COCO bird images by how likely their bird is a depiction rather than a live bird.

COCO, unlike Open Images, has no IsDepiction flag, so paintings, prints, patterns, statues,
ornaments, toys and logos reach the bird class. This crops the largest annotated bird in each
COCO bird image still in the manifest and scores it zero-shot with SigLIP 2 against real-bird
and depiction prompts. Images that also contain an annotated indoor object are marked, since
depictions cluster indoors.

The output is a review queue for review_labels.py, most likely depictions first: B keeps a
live bird, N marks a depiction for exclusion.
"""

import argparse
import json
import zipfile
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "processed" / "classifier_birds"
MODEL = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
IMAGE_SIZE = 640  # dataset images are full COCO frames letterboxed to this size
CROP_MARGIN = 0.25  # of the bird box's longer side, so the model sees some context
INDOOR_SUPERCATEGORIES = {"furniture", "appliance", "kitchen", "electronic", "indoor"}
REAL_PROMPTS = ["a photo of a real bird", "a photo of a live bird outdoors", "a wild bird"]
DEPICTION_PROMPTS = [
    "a painting of a bird",
    "a drawing of a bird",
    "a printed picture of a bird",
    "a toy bird",
    "a stuffed toy bird",
    "a bird statue",
    "a bird ornament",
    "a bird pattern on fabric",
    "a bird logo",
]


def coco_birds(dataset: Path) -> list[dict]:
    """Return COCO bird images in the manifest with their largest bird box and indoor flag."""
    manifest = [json.loads(line) for line in (dataset / "manifest.jsonl").open()]
    wanted = {
        (row["original"], int(row["source_id"])): row
        for row in manifest
        if row["source"] == "coco2017" and row["target"] == 1
    }
    largest: dict[tuple[str, int], tuple[float, list[float]]] = {}
    indoor: set[tuple[str, int]] = set()
    with zipfile.ZipFile(dataset / ".metadata" / "coco.zip") as bundle:
        for split in ("train2017", "val2017"):
            data = json.loads(bundle.read(f"annotations/instances_{split}.json"))
            categories = {row["id"]: row for row in data["categories"]}
            images = {row["id"]: row for row in data["images"]}
            for annotation in data["annotations"]:
                key = (split, annotation["image_id"])
                if key not in wanted:
                    continue
                category = categories[annotation["category_id"]]
                if category["supercategory"] in INDOOR_SUPERCATEGORIES:
                    indoor.add(key)
                if category["name"] != "bird":
                    continue
                image = images[annotation["image_id"]]
                scale = IMAGE_SIZE / max(image["width"], image["height"])
                x, y, width, height = (value * scale for value in annotation["bbox"])
                # Letterboxing centres the resized frame, so shift the box by the padding.
                x += (IMAGE_SIZE - image["width"] * scale) / 2
                y += (IMAGE_SIZE - image["height"] * scale) / 2
                size = max(width, height)
                if size > largest.get(key, (-1.0, []))[0]:
                    largest[key] = (size, [x, y, width, height])
    return [
        {**wanted[key], "largest_bird_px": size, "bird_box": box, "indoor": key in indoor}
        for key, (size, box) in largest.items()
    ]


def crop(dataset: Path, item: dict) -> Image.Image:
    x, y, width, height = item["bird_box"]
    margin = CROP_MARGIN * max(width, height)
    image = Image.open(dataset / item["image"]).convert("RGB")
    return image.crop(
        (
            max(0, x - margin),
            max(0, y - margin),
            min(IMAGE_SIZE, x + width + margin),
            min(IMAGE_SIZE, y + height + margin),
        )
    )


@torch.no_grad()
def score(dataset: Path, items: list[dict], batch_size: int) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(MODEL, revision=REVISION).to(device, dtype).eval()
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    texts = processor(
        text=REAL_PROMPTS + DEPICTION_PROMPTS,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    ).to(device)
    text_features = model.get_text_features(**texts).pooler_output
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    scored = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        pixels = processor(images=[crop(dataset, item) for item in batch], return_tensors="pt")
        image_features = model.get_image_features(
            pixel_values=pixels["pixel_values"].to(device, dtype)
        ).pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.T * model.logit_scale.exp() + model.logit_bias
        depiction = logits.float().softmax(dim=-1)[:, len(REAL_PROMPTS) :]
        for item, probabilities in zip(batch, depiction, strict=True):
            scored.append(
                {
                    **item,
                    "depiction": float(probabilities.sum()),
                    "top_depiction": DEPICTION_PROMPTS[int(probabilities.argmax())],
                }
            )
    return sorted(scored, key=lambda item: -item["depiction"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET / ".metadata" / "depictions_siglip2.jsonl",
        help="Review queue for review_labels.py (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Queue images whose depiction score is at least this (default: %(default)s)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    scored = score(args.dataset, coco_birds(args.dataset), args.batch_size)
    queue = [item for item in scored if item["depiction"] >= args.threshold]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for item in queue:
            kind = item["top_depiction"].removeprefix("a ")
            indoor = " · indoor" if item["indoor"] else ""
            tag = f"depiction {item['depiction']:.2f} ({kind}){indoor}"
            f.write(
                json.dumps(
                    {
                        "image": item["image"],
                        "split": item["split"],
                        "given": "bird",
                        "p_bird": "n/a",
                        "source": tag,
                    }
                )
                + "\n"
            )
    print(f"Scored {len(scored)} COCO bird images; queued {len(queue)} in {args.output}")


if __name__ == "__main__":
    main()
