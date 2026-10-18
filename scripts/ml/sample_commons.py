#!/usr/bin/env python3
"""Collect freely licensed Wikimedia Commons photos from category trees.

Walks each root category down to --depth levels of subcategories, keeping JPEG photos at
least --min-size pixels on each side and skipping subcategories of drawings, maps and
other depictions. Each root is capped at --per-root files so one large tree cannot
dominate. Writes <out>/meta.jsonl (with licence and attribution) and 960 px thumbnails
in <out>/images, for scripts/ml/mine_classifier_errors.py.

    python scripts/ml/sample_commons.py --out data/commons/garden-structures --dry-run
"""

import argparse
import html
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "BirdSpotter-dataset/0.1 (https://github.com/Pbatch/BirdSpotter)"
# Wikimedia only serves thumbnails at standard widths; 960 is the closest to 1008.
THUMB_WIDTH = 960
ROOTS = (
    "Bird feeders,Bird tables,Bird baths,Nest boxes,Garden fences,Wooden fences,"
    "Fences in the United Kingdom,Fence posts,Garden sheds,Garden furniture,Garden gnomes,"
    "Hedges,Trellises,Patios,Pergolas,Garden ponds,Compost bins,Gardens in England,Garden walls"
)
DEPICTIONS = re.compile(
    r"\b(drawings?|paintings?|illustrations?|maps?|diagrams?|logos?|icons?|postcards?|prints?|"
    r"engravings?|sketches|plans|videos?|in art|clip ?art|stamps?)\b",
    re.IGNORECASE,
)


class Commons:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def query(self, **params: str | int) -> dict:
        params = {"action": "query", "format": "json", "maxlag": 5, **params}
        for attempt in range(5):
            response = self.session.get(API, params=params, timeout=60)
            if response.status_code == 200 and "error" not in (data := response.json()):
                return data
            time.sleep(5 * (attempt + 1))  # maxlag or transient error: back off
        response.raise_for_status()
        raise RuntimeError(f"Commons API error: {response.text[:200]}")

    def members(self, category: str, kind: str, **extra: str | int) -> list[dict]:
        """Every member of one category, following continuation."""
        params = {"list": "categorymembers", "cmtitle": category, "cmtype": kind, "cmlimit": 500}
        if kind == "file":
            params = {
                "generator": "categorymembers",
                "gcmtitle": category,
                "gcmtype": "file",
                "gcmlimit": 50,
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "iiurlwidth": THUMB_WIDTH,
                **extra,
            }
        found, cont = [], {}
        while True:
            data = self.query(**params, **cont)
            if kind == "file":
                found += list(data.get("query", {}).get("pages", {}).values())
            else:
                found += data.get("query", {}).get("categorymembers", [])
            if "continue" not in data:
                return found
            cont = data["continue"]


def plain(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def walk(commons: Commons, root: str, depth: int) -> list[str]:
    """Breadth-first category names under root, skipping depiction subcategories."""
    seen, frontier = {f"Category:{root}"}, [f"Category:{root}"]
    for level in range(depth):
        nxt = []
        for category in tqdm(
            frontier, desc=f"{root[:24]}: subcategories, level {level + 1}", leave=False
        ):
            for sub in commons.members(category, "subcat"):
                title = sub["title"]
                if title not in seen and not DEPICTIONS.search(title):
                    seen.add(title)
                    nxt.append(title)
        frontier = nxt
    return sorted(seen)


def collect(args: argparse.Namespace, existing: set[str]) -> list[dict]:
    commons, rows, seen = Commons(), [], set(existing)
    roots = [r.strip() for r in args.roots.split(",") if r.strip()]
    for number, root in enumerate(roots, 1):
        categories = walk(commons, root, args.depth)
        kept = 0
        bar = tqdm(categories, desc=f"[{number}/{len(roots)}] {root[:24]}: files", leave=False)
        for category in bar:
            bar.set_postfix(kept=kept, total=len(rows))
            if kept >= args.per_root:
                break
            for page in commons.members(category, "file"):
                info = (page.get("imageinfo") or [{}])[0]
                title = page.get("title", "")
                if (
                    title in seen
                    or info.get("mime") != "image/jpeg"
                    or min(info.get("width", 0), info.get("height", 0)) < args.min_size
                    or "thumburl" not in info
                ):
                    continue
                meta = info.get("extmetadata", {})
                licence = plain(meta.get("LicenseShortName", {}).get("value", ""))
                artist = plain(meta.get("Artist", {}).get("value", "")) or "Unknown author"
                seen.add(title)
                rows.append(
                    {
                        "group": root,
                        "category": category[9:],
                        "title": title,
                        "pageid": page["pageid"],
                        "page": info.get("descriptionurl"),
                        "thumb": info["thumburl"],
                        "license": licence,
                        "attribution": f"{artist}, {licence}, via Wikimedia Commons",
                    }
                )
                kept += 1
                if kept >= args.per_root:
                    break
        bar.close()
        print(
            f"[{number}/{len(roots)}] {root}: {len(categories)} categories, kept {kept}", flush=True
        )
    return rows


def download(row: dict, out: Path, session: requests.Session) -> dict | None:
    image = f"images/{row['pageid']}.jpg"
    path = out / image
    if not path.exists():
        for attempt in range(4):
            try:
                response = session.get(row["thumb"], timeout=60)
                if response.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                response.raise_for_status()
            except requests.RequestException:
                time.sleep(3 * (attempt + 1))
                continue
            path.write_bytes(response.content)
            break
        else:
            return None
    return {"image": image, **{k: v for k, v in row.items() if k != "thumb"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roots", default=ROOTS, help="Comma-separated root category names")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--per-root", type=int, default=1500)
    parser.add_argument("--min-size", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/processed/classifier_birds/manifest.jsonl",
        help="Skip Commons files already in this dataset manifest",
    )
    parser.add_argument("--dry-run", action="store_true", help="Count files without downloading")
    args = parser.parse_args()

    existing = set()
    if args.manifest.exists():
        for line in args.manifest.open():
            row = json.loads(line)
            if row["source"].startswith("commons"):
                existing.add(row["source_id"])
    rows = collect(args, existing)
    print(f"{len(rows)} files; licences: {Counter(r['license'] for r in rows).most_common(8)}")
    if args.dry_run:
        return
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(
            tqdm(
                pool.map(lambda row: download(row, args.out, session), rows),
                total=len(rows),
                desc="thumbnails",
            )
        )
    saved = [result for result in results if result is not None]
    with (args.out / "meta.jsonl").open("w") as handle:
        for result in saved:
            handle.write(json.dumps(result) + "\n")
    print(f"Saved {len(saved)} of {len(rows)} to {args.out}")


if __name__ == "__main__":
    main()
