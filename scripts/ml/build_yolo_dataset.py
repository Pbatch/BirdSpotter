#!/usr/bin/env python3
"""Resumable full-corpus builder for a one-class Ultralytics bird dataset."""

import argparse
import csv
import hashlib
import io
import json
import os
import tarfile
import threading
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TextIO

import fiftyone as fo
import fiftyone.zoo as foz
import requests
from datasets import Image as HFImage
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # ty: ignore[invalid-assignment]
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "yolo_birds_1600x896"
META = OUT / ".metadata"
SIZE = (1600, 896)
UA = "BirdSpotter-dataset-builder/2.0"
S = requests.Session()
S.headers["User-Agent"] = UA
SEEN = set()
LOCK = threading.Lock()
WORKERS = 12
Box = tuple[float, float, float, float]


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
        with part.open("ab" if r.status_code == 206 else "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    part.replace(path)
    return path


def split(source: str, key: str) -> str:
    n = int.from_bytes(hashlib.sha256(f"{source}:{key}".encode()).digest()[:8], "big")
    return "val" if n < (1 << 64) // 10 else "train"


def letterbox(im: Image.Image, boxes: list[Box]) -> tuple[Image.Image, list[Box]]:
    im = im.convert("RGB")
    tw, th = SIZE
    scale = min(tw / im.width, th / im.height)
    rw, rh = round(im.width * scale), round(im.height * scale)
    px, py = (tw - rw) // 2, (th - rh) // 2
    canvas = Image.new("RGB", SIZE, (114, 114, 114))
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
    ip = OUT / "images" / sp / f"{stem}.jpg"
    lp = OUT / "labels" / sp / f"{stem}.txt"
    if not (ip.exists() and lp.exists()):
        im, boxes = letterbox(im, boxes)
        if not boxes:
            return 0
        ti = ip.with_suffix(".part")
        tl = lp.with_suffix(".part")
        im.save(ti, "JPEG", quality=92)
        tl.write_text(
            "\n".join(
                " ".join(
                    (
                        "0",
                        f"{(a + c) / 3200:.8f}",
                        f"{(b + d) / 1792:.8f}",
                        f"{(c - a) / 1600:.8f}",
                        f"{(d - b) / 896:.8f}",
                    )
                )
                for a, b, c, d in boxes
            )
            + "\n"
        )
        ti.replace(ip)
        tl.replace(lp)
    count = len(lp.read_text().splitlines())
    identity = (source, key)
    with LOCK:
        if identity not in SEEN:
            m.write(
                json.dumps(
                    {
                        "image": str(ip.relative_to(OUT)),
                        "split": sp,
                        "source": source,
                        "source_id": key,
                        "original": original,
                        "objects": count,
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
    # FiftyOne provides the optimized, resized (max dimension 1024) Open Images
    # zoo download path. We still pre-read annotations so images with any
    # group/depiction Bird box are excluded before their pixels are downloaded.
    os.environ.setdefault("FIFTYONE_DATASET_ZOO_DIR", str(META / "fiftyone-zoo"))
    os.environ.setdefault("FIFTYONE_DATABASE_DIR", str(META / "fiftyone-db"))
    fo.config.dataset_zoo_dir = str(META / "fiftyone-zoo")
    urls = {
        "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
        "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
    }
    n = 0
    for rs, url in urls.items():
        good = defaultdict(list)
        bad = set()
        with fetch(url, META / f"openimages-{rs}.csv").open(newline="") as f:
            for r in csv.DictReader(f):
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
            for added in pool.map(process, paths):
                n += added
        if limit and n >= limit:
            return


def coco(m: TextIO, limit: int | None) -> None:  # noqa: C901
    z = fetch(
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip", META / "coco.zip"
    )
    n = 0
    with zipfile.ZipFile(z) as f:
        for rs in ("train2017", "val2017"):
            d = json.loads(f.read(f"annotations/instances_{rs}.json"))
            bird = next(x["id"] for x in d["categories"] if x["name"] == "bird")
            boxes = defaultdict(list)
            for a in d["annotations"]:
                if a["category_id"] == bird and not a.get("iscrowd", 0):
                    x, y, w, h = a["bbox"]
                    boxes[a["image_id"]].append((x, y, x + w, y + h))
            ims = {x["id"]: x for x in d["images"]}
            keys = [k for k in sorted(boxes) if ("coco2017", str(k)) not in SEEN]
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
                    boxes[k],
                    m,
                    original=rs,
                )

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                for added in pool.map(process, keys):
                    n += added
            if limit and n >= limit:
                return


def voc(m: TextIO, limit: int | None) -> None:
    n = 0
    for rs in ("train", "validation"):
        ds = load_dataset(
            "TNILab/pascal_voc2012_det_train_val", split=rs, streaming=True
        ).cast_column("image", HFImage(decode=False))
        for row in ds:
            boxes = [
                (x, y, x + w, y + h)
                for (x, y, w, h), c in zip(
                    row["objects"]["bbox"], row["objects"]["category"], strict=True
                )
                if c == 2
            ]
            if boxes:
                k = str(row.get("image_id", f"{rs}-{n}"))
                n += save("voc2012", k, raw(row), boxes, m, original=rs)
                if limit and n >= limit:
                    return


def birdsnap(m: TextIO, limit: int | None) -> None:  # noqa: C901
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    os.environ["HF_HOME"] = str(META / "huggingface-cache")
    os.environ["HF_XET_CACHE"] = str(META / "huggingface-cache" / "xet")
    snapshot = Path(
        snapshot_download(
            repo_id="HuggingFaceM4/Birdsnap",
            repo_type="dataset",
            allow_patterns=["images/*.tar", "annotations.tar.gz"],
            local_dir=META / "birdsnap-xet",
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
    with ThreadPoolExecutor(max_workers=4) as pool:
        for added in pool.map(process_species, grouped.items()):
            n += added


def nabirds(m: TextIO, limit: int | None) -> None:
    p = fetch(
        "https://raw.githubusercontent.com/Rice-Field/NABirds/master/bounding_boxes.txt",
        META / "nabirds-boxes.txt",
    )
    boxes = {}
    for line in p.read_text().splitlines():
        k, x, y, w, h = line.split()
        boxes[k.replace("-", "")] = (float(x), float(y), float(x) + float(w), float(y) + float(h))
    ds = load_dataset(
        "anjunhu/naively_captioned_nabirds", split="train", streaming=True
    ).cast_column("image", HFImage(decode=False))
    n = 0
    for r in ds:
        k = Path(r["path"]).stem
        b = boxes.get(k.replace("-", ""))
        if b:
            n += save("nabirds", k, raw(r), [b], m, original=r["path"])
        if limit and n >= limit:
            return


def main() -> None:
    names = ["openimages", "coco2017", "voc2012", "birdsnap", "nabirds"]
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", choices=names, default=names)
    p.add_argument("--limit-per-source", type=int)
    a = p.parse_args()
    for sp in ("train", "val"):
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: bird\n"
    )
    manifest_path = OUT / "manifest.jsonl"
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            try:
                r = json.loads(line)
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
        for name in a.sources:
            print(f"[{time.strftime('%F %T')}] starting {name}", flush=True)
            funcs[name](m, a.limit_per_source)
            print(f"[{time.strftime('%F %T')}] finished {name}", flush=True)


if __name__ == "__main__":
    main()
    # Avoid a known pyarrow/HF streaming finalizer crash after all files are closed.
    os._exit(0)
