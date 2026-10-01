#!/usr/bin/env python3
"""Resumable source-corpus builder for full-frame bird classification."""

# Source libraries load lazily to keep local builds and tests lightweight.
# ruff: noqa: PLC0415

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tarfile
import threading
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import requests
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True  # ty: ignore[invalid-assignment]
ROOT = Path(__file__).resolve().parents[2]
UA = "BirdSpotter-dataset-builder/2.0"
S = requests.Session()
S.headers["User-Agent"] = UA
SEEN = set()
LOCK = threading.Lock()
WORKERS = 12
Box = tuple[float, float, float, float]


@dataclass(slots=True)
class BuildConfig:
    output_dir: Path
    source_cache: Path
    size: tuple[int, int]
    coco_negative_ratio: int = 3


CONFIG = BuildConfig(
    output_dir=ROOT / "data" / "processed" / "classifier_birds",
    source_cache=ROOT / "data" / "processed" / "classifier_birds" / ".metadata",
    size=(640, 640),
)


def fetch(url: str, path: Path) -> Path:
    if path.exists() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    headers = {"Range": f"bytes={part.stat().st_size}-"} if part.exists() else {}
    with S.get(url, headers=headers, stream=True, timeout=(30, 300)) as r:
        if r.status_code == 200 and part.exists():
            part.unlink()
        r.raise_for_status()
        initial = part.stat().st_size if part.exists() and r.status_code == 206 else 0
        content_length = r.headers.get("Content-Length")
        total = initial + int(content_length) if content_length else None
        with (
            part.open("ab" if r.status_code == 206 else "wb") as f,
            tqdm(
                total=total,
                initial=initial,
                desc=f"Download {path.name}",
                unit="B",
                unit_scale=True,
                mininterval=1,
            ) as progress,
        ):
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                progress.update(len(chunk))
    part.replace(path)
    return path


def split(source: str, key: str) -> str:
    n = int.from_bytes(hashlib.sha256(f"{source}:{key}".encode()).digest()[:8], "big")
    return "val" if n < (1 << 64) // 10 else "train"


def letterbox(im: Image.Image, boxes: list[Box]) -> tuple[Image.Image, list[Box]]:
    im = im.convert("RGB")
    tw, th = CONFIG.size
    scale = min(tw / im.width, th / im.height)
    rw, rh = round(im.width * scale), round(im.height * scale)
    px, py = (tw - rw) // 2, (th - rh) // 2
    canvas = Image.new("RGB", CONFIG.size, (114, 114, 114))
    canvas.paste(im.resize((rw, rh), Image.Resampling.LANCZOS), (px, py))
    out = []
    for x1, y1, x2, y2 in boxes:
        b = (
            max(0, x1 * scale + px),
            max(0, y1 * scale + py),
            min(tw, x2 * scale + px),
            min(th, y2 * scale + py),
        )
        if b[2] > b[0] and b[3] > b[1]:
            out.append(b)
    return canvas, out


def save(  # noqa: PLR0913
    source: str,
    key: str,
    im: Image.Image,
    boxes: list[Box],
    m: TextIO,
    *,
    original: str = "",
) -> int:
    sp = split(source, key)
    digest = hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest()[:16]
    stem = f"{source}_{digest}"
    label = "bird" if boxes else "no_bird"
    ip = CONFIG.output_dir / sp / label / f"{stem}.jpg"
    ip.parent.mkdir(parents=True, exist_ok=True)
    if not ip.exists():
        prepared, _ = letterbox(im, [])
        temporary = ip.with_suffix(".part")
        prepared.save(temporary, "JPEG", quality=92)
        temporary.replace(ip)
    count = len(boxes)
    identity = (source, key)
    with LOCK:
        if identity not in SEEN:
            m.write(
                json.dumps(
                    {
                        "image": str(ip.relative_to(CONFIG.output_dir)),
                        "split": sp,
                        "source": source,
                        "source_id": key,
                        "original": original,
                        "objects": count,
                        "label": label,
                        "target": int(bool(boxes)),
                    }
                )
                + "\n"
            )
            SEEN.add(identity)
    return 1


def raw(row: dict[str, Any]) -> Image.Image:
    v = row["image"]
    return (
        Image.open(io.BytesIO(v["bytes"])) if v.get("bytes") is not None else Image.open(v["path"])
    )


def openimages(m: TextIO, limit: int | None) -> None:
    import fiftyone as fo
    import fiftyone.zoo as foz

    # FiftyOne provides the optimized, resized (max dimension 1024) Open Images
    # zoo download path. We still pre-read annotations so images with any
    # group/depiction Bird box are excluded before their pixels are downloaded.
    os.environ.setdefault("FIFTYONE_DATASET_ZOO_DIR", str(CONFIG.source_cache / "fiftyone-zoo"))
    os.environ.setdefault("FIFTYONE_DATABASE_DIR", str(CONFIG.source_cache / "fiftyone-db"))
    fo.config.dataset_zoo_dir = str(CONFIG.source_cache / "fiftyone-zoo")
    urls = {
        "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
        "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
    }
    n = 0
    for rs, url in urls.items():
        good = defaultdict(list)
        bad = set()
        with fetch(url, CONFIG.source_cache / f"openimages-{rs}.csv").open(newline="") as f:
            for r in tqdm(
                csv.DictReader(f), desc=f"Open Images/{rs} annotations", unit="rows", mininterval=1
            ):
                if r["LabelName"] != "/m/015p6":
                    continue
                k = r["ImageID"]
                good[k].append(tuple(float(r[x]) for x in ("XMin", "YMin", "XMax", "YMax")))
                if r.get("IsGroupOf") != "0" or r.get("IsDepiction") != "0":
                    bad.add(k)
        keys = [k for k in sorted(set(good) - bad) if ("openimages", k) not in SEEN]
        if limit:
            keys = keys[: max(0, limit - n)]
        if not keys:
            continue
        _, zoo_root = foz.download_zoo_dataset(
            "open-images-v7",
            split=rs,
            label_types=["detections"],
            classes=["Bird"],
            image_ids=keys,
            num_workers=WORKERS,
        )
        data_dir = Path(zoo_root) / rs / "data"
        paths = [data_dir / f"{k}.jpg" for k in keys]

        def process(path: Path, good: dict[str, list[Box]] = good, rs: str = rs) -> int:
            k = path.stem
            with Image.open(path) as im:
                boxes = [
                    (a * im.width, b * im.height, c * im.width, d * im.height)
                    for a, b, c, d in good[k]
                ]
                return save("openimages", k, im, boxes, m, original=rs)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(process, path) for path in paths]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Open Images/{rs}",
                unit="images",
                mininterval=1,
            ):
                n += future.result()
        if limit and n >= limit:
            return


def select_coco_images(
    images: dict[int, dict[str, Any]],
    bird_ids: set[int],
    ratio: int,
) -> tuple[list[int], list[int]]:
    """Choose a stable negative sample before applying resume filtering."""
    positive = sorted(bird_ids & images.keys())
    negative = [key for key in sorted(images) if key not in bird_ids]
    return positive, negative[: len(positive) * ratio]


def prune_coco_negatives(archive: Path, manifest: Path) -> int:
    """Preserve excess negatives outside ImageFolder and remove their active records."""
    if not archive.is_file() or not manifest.is_file():
        return 0
    allowed = set()
    with zipfile.ZipFile(archive) as bundle:
        for source_split in ("train2017", "val2017"):
            data = json.loads(bundle.read(f"annotations/instances_{source_split}.json"))
            bird = next(row["id"] for row in data["categories"] if row["name"] == "bird")
            bird_ids = {
                row["image_id"] for row in data["annotations"] if row["category_id"] == bird
            }
            images = {row["id"]: row for row in data["images"]}
            _, negatives = select_coco_images(images, bird_ids, CONFIG.coco_negative_ratio)
            allowed.update(str(key) for key in negatives)
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    removed = [
        row
        for row in records
        if row["source"] == "coco2017"
        and row.get("label") == "no_bird"
        and str(row["source_id"]) not in allowed
    ]
    if not removed:
        return 0
    excluded = CONFIG.output_dir / ".excluded" / "coco2017"
    excluded.mkdir(parents=True, exist_ok=True)
    backup = excluded / "manifest-before-cap.jsonl"
    if not backup.exists():
        shutil.copy2(manifest, backup)
    for row in removed:
        relative = Path(row["image"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe image path in manifest: {relative}")
        image = CONFIG.output_dir / relative
        if image.is_file():
            destination = excluded / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.replace(destination)
    kept = [
        row
        for row in records
        if not (
            row["source"] == "coco2017"
            and row.get("label") == "no_bird"
            and str(row["source_id"]) not in allowed
        )
    ]
    temporary = manifest.with_suffix(".part")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in kept))
    temporary.replace(manifest)
    print(f"[coco2017] Preserved {len(removed)} excess negatives in {excluded}", flush=True)
    return len(removed)


def coco(m: TextIO, limit: int | None) -> None:  # noqa: C901
    z = fetch(
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        CONFIG.source_cache / "coco.zip",
    )
    n = 0
    with zipfile.ZipFile(z) as f:
        for rs in ("train2017", "val2017"):
            d = json.loads(f.read(f"annotations/instances_{rs}.json"))
            bird = next(x["id"] for x in d["categories"] if x["name"] == "bird")
            boxes = defaultdict(list)
            for a in d["annotations"]:
                if a["category_id"] == bird:
                    x, y, w, h = a["bbox"]
                    boxes[a["image_id"]].append((x, y, x + w, y + h))
            ims = {x["id"]: x for x in d["images"]}
            # Alternate positives and annotated negatives so a limited run includes both.
            positive, negative = select_coco_images(ims, set(boxes), CONFIG.coco_negative_ratio)
            print(
                f"[coco2017/{rs}] selected {len(positive)} birds and {len(negative)} negatives "
                f"(maximum {CONFIG.coco_negative_ratio}:1)",
                flush=True,
            )
            keys = []
            for index in range(max(len(positive), len(negative))):
                for group in (positive, negative):
                    if index < len(group) and ("coco2017", str(group[index])) not in SEEN:
                        keys.append(group[index])  # noqa: PERF401
            if limit:
                keys = keys[: max(0, limit - n)]

            def process(
                k: int,
                ims: dict[int, dict[str, Any]] = ims,
                rs: str = rs,
                boxes: dict[int, list[Box]] = boxes,
            ) -> int:
                rec = ims[k]
                for attempt in range(5):
                    try:
                        r = requests.get(
                            f"http://images.cocodataset.org/{rs}/{rec['file_name']}",
                            headers={"User-Agent": UA},
                            timeout=120,
                        )
                        r.raise_for_status()
                        break
                    except requests.RequestException:
                        if attempt == 4:
                            raise
                        time.sleep(2**attempt)
                return save(
                    "coco2017",
                    str(k),
                    Image.open(io.BytesIO(r.content)),
                    boxes.get(k, []),
                    m,
                    original=rs,
                )

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = [pool.submit(process, key) for key in keys]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"COCO/{rs}",
                    unit="images",
                    mininterval=1,
                ):
                    n += future.result()
            if limit and n >= limit:
                return


def voc(m: TextIO, limit: int | None) -> None:
    from datasets import Image as HFImage
    from datasets import load_dataset

    n = 0
    for rs in ("train", "validation"):
        ds = load_dataset(
            "TNILab/pascal_voc2012_det_train_val", split=rs, streaming=True
        ).cast_column("image", HFImage(decode=False))
        for index, row in enumerate(
            tqdm(ds, desc=f"VOC/{rs} scanned", unit="images", mininterval=1)
        ):
            boxes = [
                (x, y, x + w, y + h)
                for (x, y, w, h), c in zip(
                    row["objects"]["bbox"], row["objects"]["category"], strict=True
                )
                if c == 2
            ]
            k = str(row.get("image_id", f"{rs}-{index}"))
            if ("voc2012", k) in SEEN:
                continue
            n += save("voc2012", k, raw(row), boxes, m, original=rs)
            if limit and n >= limit:
                return


def birdsnap(m: TextIO, limit: int | None) -> None:  # noqa: C901
    from huggingface_hub import snapshot_download

    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    os.environ["HF_HOME"] = str(CONFIG.source_cache / "huggingface-cache")
    os.environ["HF_XET_CACHE"] = str(CONFIG.source_cache / "huggingface-cache" / "xet")
    snapshot = Path(
        snapshot_download(
            repo_id="HuggingFaceM4/Birdsnap",
            repo_type="dataset",
            allow_patterns=["images/*.tar", "annotations.tar.gz"],
            local_dir=CONFIG.source_cache / "birdsnap-xet",
            max_workers=16,
        )
    )
    p = snapshot / "annotations.tar.gz"
    with tarfile.open(p, "r:gz") as tf:
        member = next(x for x in tf.getmembers() if x.name.endswith("images.txt"))
        extracted = tf.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Unable to extract Birdsnap annotations from {p}")
        rows = list(csv.DictReader(io.TextIOWrapper(extracted), delimiter="\t"))
    # Keep a deterministic, balanced sample: 25 images from every species.
    per_species = defaultdict(list)
    for r in rows:
        per_species[r["species_id"]].append(r)
    selected = []
    for species_rows in per_species.values():
        selected.extend(
            sorted(species_rows, key=lambda r: hashlib.sha256(r["path"].encode()).digest())[:25]
        )
    selected = [r for r in selected if ("birdsnap", r["path"]) not in SEEN]
    if limit:
        selected = selected[:limit]
    grouped = defaultdict(list)
    for r in selected:
        grouped[Path(r["path"]).parts[0]].append(r)

    def process_species(item: tuple[str, list[dict[str, str]]]) -> int:
        folder, species_rows = item
        archive = snapshot / "images" / f"{folder}.tar"
        added = 0
        if not archive.is_file():
            raise FileNotFoundError(f"Missing Birdsnap species archive: {archive}")
        with tarfile.open(archive) as tf:
            members = {
                Path(member.name).name: member for member in tf.getmembers() if member.isfile()
            }
            for r in species_rows:
                member = members.get(Path(r["path"]).name)
                if member is None:
                    continue
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                payload = extracted.read()
                b = [
                    (
                        float(r["bb_x1"]),
                        float(r["bb_y1"]),
                        float(r["bb_x2"]),
                        float(r["bb_y2"]),
                    )
                ]
                added += save(
                    "birdsnap",
                    r["path"],
                    Image.open(io.BytesIO(payload)),
                    b,
                    m,
                    original=r["url"],
                )
        return added

    n = 0
    with (
        ThreadPoolExecutor(max_workers=4) as pool,
        tqdm(
            total=len(selected),
            desc="Birdsnap",
            unit="images",
            mininterval=1,
        ) as progress,
    ):
        futures = {pool.submit(process_species, item): len(item[1]) for item in grouped.items()}
        for future in as_completed(futures):
            n += future.result()
            progress.update(futures[future])


def nabirds(m: TextIO, limit: int | None) -> None:
    from datasets import Image as HFImage
    from datasets import load_dataset

    p = fetch(
        "https://raw.githubusercontent.com/Rice-Field/NABirds/master/bounding_boxes.txt",
        CONFIG.source_cache / "nabirds-boxes.txt",
    )
    boxes = {}
    for line in p.read_text().splitlines():
        k, x, y, w, h = line.split()
        boxes[k.replace("-", "")] = (float(x), float(y), float(x) + float(w), float(y) + float(h))
    ds = load_dataset(
        "anjunhu/naively_captioned_nabirds", split="train", streaming=True
    ).cast_column("image", HFImage(decode=False))
    n = 0
    for r in tqdm(ds, desc="NABirds scanned", unit="images", mininterval=1):
        k = Path(r["path"]).stem
        b = boxes.get(k.replace("-", ""))
        if b:
            n += save("nabirds", k, raw(r), [b], m, original=r["path"])
        if limit and n >= limit:
            return


def check_existing_resolution(output_dir: Path, size: tuple[int, int]) -> None:
    previous_config = output_dir / "build-config.json"
    if previous_config.is_file():
        previous = json.loads(previous_config.read_text())
        if (previous["width"], previous["height"]) != size:
            raise ValueError("Existing dataset resolution differs; use a new --output-dir")


def main() -> None:  # noqa: C901
    names = ["openimages", "coco2017", "voc2012", "birdsnap", "nabirds"]
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=CONFIG.output_dir, type=Path)
    p.add_argument("--width", default=640, type=int)
    p.add_argument("--height", default=640, type=int)
    p.add_argument(
        "--source-cache",
        type=Path,
        default=CONFIG.source_cache,
        help="Cache of original source images and annotations (default: %(default)s)",
    )
    p.add_argument("--sources", nargs="+", choices=names, default=names)
    p.add_argument("--limit-per-source", type=int)
    p.add_argument(
        "--coco-negative-ratio",
        type=int,
        default=3,
        help="Maximum COCO no-bird images per bird image (default: %(default)s)",
    )
    a = p.parse_args()
    if a.width < 1 or a.height < 1:
        raise ValueError("Output width and height must be positive")
    if a.limit_per_source is not None and a.limit_per_source < 1:
        raise ValueError("Limit per source must be positive")
    if a.coco_negative_ratio < 0:
        raise ValueError("COCO negative ratio must be nonnegative")
    CONFIG.coco_negative_ratio = a.coco_negative_ratio
    CONFIG.output_dir = a.output_dir.resolve()
    CONFIG.source_cache = a.source_cache.resolve()
    CONFIG.size = (a.width, a.height)
    check_existing_resolution(CONFIG.output_dir, CONFIG.size)
    for sp in ("train", "val"):
        for label in ("bird", "no_bird"):
            (CONFIG.output_dir / sp / label).mkdir(parents=True, exist_ok=True)
    CONFIG.source_cache.mkdir(parents=True, exist_ok=True)
    (CONFIG.output_dir / "build-config.json").write_text(
        json.dumps(
            {
                "format": "imagefolder",
                "targets": {"no_bird": 0, "bird": 1},
                "width": CONFIG.size[0],
                "height": CONFIG.size[1],
                "source_cache": str(CONFIG.source_cache),
                "sources": a.sources,
                "coco_negative_ratio": CONFIG.coco_negative_ratio,
            },
            indent=2,
        )
        + "\n"
    )
    manifest_path = CONFIG.output_dir / "manifest.jsonl"
    if "coco2017" in a.sources:
        prune_coco_negatives(CONFIG.source_cache / "coco.zip", manifest_path)
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            try:
                r = json.loads(line)
                if (CONFIG.output_dir / r["image"]).is_file():
                    SEEN.add((r["source"], str(r["source_id"])))
            except (json.JSONDecodeError, KeyError):
                pass
    funcs = {
        "openimages": openimages,
        "coco2017": coco,
        "voc2012": voc,
        "birdsnap": birdsnap,
        "nabirds": nabirds,
    }
    with manifest_path.open("a", buffering=1) as m:
        for name in tqdm(a.sources, desc="Sources", unit="source", position=0):
            tqdm.write(f"[{time.strftime('%F %T')}] starting {name}")
            funcs[name](m, a.limit_per_source)
            tqdm.write(f"[{time.strftime('%F %T')}] finished {name}")


if __name__ == "__main__":
    main()
    # Avoid a known pyarrow/HF streaming finalizer crash after all files are closed.
    os._exit(0)
