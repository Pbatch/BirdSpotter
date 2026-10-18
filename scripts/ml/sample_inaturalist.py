#!/usr/bin/env python3
"""Sample openly licensed iNaturalist photos from the AWS open-data dumps.

Download the metadata once (no AWS account needed):

    aws s3 cp --no-sign-request s3://inaturalist-open-data/<file>.csv.gz <dump-dir>/

for taxa, observers, observations and photos. The first run builds a Parquet index of
research-grade observations inside --bbox whose photo is CC0 or CC-BY, and later runs
reuse it. Each run then samples --group taxa at random, excluding --exclude taxa,
and downloads the photos straight from the bucket.
"""

# Queries are assembled from this script's own command-line arguments, not untrusted input.
# ruff: noqa: S608

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
PHOTOS = "https://inaturalist-open-data.s3.amazonaws.com/photos"
LICENSES = ("CC0", "CC-BY")
UK = "49.8,-8.7,60.9,1.8"
GROUPS = (
    "plant:47126:450,insect:47158:100,fungus:47170:40,grey_squirrel:46017:130,"
    "hedgehog:43042:90,red_fox:42054:90,brown_rat:44576:50,wood_mouse:45562:50"
)


def csv(path: Path, columns: dict[str, str]) -> str:
    """Read an iNaturalist dump with fixed types; the files are unquoted TSV."""
    return f"read_csv('{path}', delim='\\t', header=true, quote='', escape='', columns={columns})"


def build_index(dumps: Path, index: Path, bbox: str) -> None:
    min_lat, min_lon, max_lat, max_lon = (float(value) for value in bbox.split(","))
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{dumps / '.duckdb-tmp'}'")
    observations = csv(
        dumps / "observations.csv.gz",
        {
            "observation_uuid": "VARCHAR",
            "observer_id": "BIGINT",
            "latitude": "DOUBLE",
            "longitude": "DOUBLE",
            "positional_accuracy": "DOUBLE",
            "taxon_id": "BIGINT",
            "quality_grade": "VARCHAR",
            "observed_on": "VARCHAR",
            "anomaly_score": "DOUBLE",
        },
    )
    photos = csv(
        dumps / "photos.csv.gz",
        {
            "photo_uuid": "VARCHAR",
            "photo_id": "BIGINT",
            "observation_uuid": "VARCHAR",
            "observer_id": "BIGINT",
            "extension": "VARCHAR",
            "license": "VARCHAR",
            "width": "INTEGER",
            "height": "INTEGER",
            "position": "INTEGER",
        },
    )
    taxa = csv(
        dumps / "taxa.csv.gz",
        {
            "taxon_id": "BIGINT",
            "ancestry": "VARCHAR",
            "rank_level": "DOUBLE",
            "rank": "VARCHAR",
            "name": "VARCHAR",
            "active": "BOOLEAN",
        },
    )
    observers = csv(
        dumps / "observers.csv.gz",
        {"observer_id": "BIGINT", "login": "VARCHAR", "name": "VARCHAR"},
    )
    con.execute(f"""
        COPY (
            WITH local AS (
                SELECT * FROM {observations}
                WHERE quality_grade = 'research'
                  AND latitude BETWEEN {min_lat} AND {max_lat}
                  AND longitude BETWEEN {min_lon} AND {max_lon}
            ),
            -- The lowest-positioned openly licensed photo of each local observation.
            photo AS (
                SELECT observation_uuid,
                       arg_min(photo_id, position) AS photo_id,
                       arg_min(extension, position) AS extension,
                       arg_min(license, position) AS license,
                       arg_min(width, position) AS width,
                       arg_min(height, position) AS height
                FROM {photos}
                WHERE license IN {LICENSES}
                  AND observation_uuid IN (SELECT observation_uuid FROM local)
                GROUP BY observation_uuid
            )
            SELECT local.observation_uuid, local.observer_id, observers.login,
                   observers.name AS observer_name, local.latitude, local.longitude,
                   local.observed_on, local.taxon_id, taxa.name AS taxon, taxa.rank,
                   taxa.ancestry, photo.photo_id, photo.extension, photo.license,
                   photo.width, photo.height
            FROM local
            JOIN photo USING (observation_uuid)
            JOIN {taxa} AS taxa USING (taxon_id)
            LEFT JOIN {observers} AS observers USING (observer_id)
        ) TO '{index}' (FORMAT parquet)
    """)


def within(taxon_id: int) -> str:
    """SQL test for a taxon or any of its descendants; ancestry lists ancestors only."""
    return f"(taxon_id = {taxon_id} OR list_contains(string_split(ancestry, '/'), '{taxon_id}'))"


def sample(  # noqa: PLR0913
    index: Path,
    *,
    groups: str,
    exclude: list[int],
    per_observer: int,
    seed: int,
    existing: set[int],
) -> list[dict]:
    specs = [spec.split(":") for spec in groups.split(",")]
    membership = " ".join(f"WHEN {within(int(taxon))} THEN '{name}'" for name, taxon, _ in specs)
    counts = " ".join(f"WHEN '{name}' THEN {int(count)}" for name, _, count in specs)
    excluded = " OR ".join(within(taxon) for taxon in exclude) or "false"
    skip = ",".join(map(str, existing)) or "-1"
    rows = duckdb.sql(f"""
        WITH tagged AS (
            SELECT *, CASE {membership} END AS "group", hash(photo_id + {seed}) AS draw
            FROM '{index}'
            WHERE NOT ({excluded}) AND photo_id NOT IN ({skip})
        ),
        ranked AS (
            SELECT *,
                   row_number() OVER (PARTITION BY "group", observer_id ORDER BY draw) AS mine
            FROM tagged WHERE "group" IS NOT NULL
        ),
        capped AS (
            SELECT *, row_number() OVER (PARTITION BY "group" ORDER BY draw) AS nth
            FROM ranked WHERE mine <= {per_observer}
        )
        SELECT * EXCLUDE (draw, mine, nth) FROM capped
        WHERE nth <= CASE "group" {counts} END
        ORDER BY "group", nth
    """)
    return [dict(zip(rows.columns, row, strict=True)) for row in rows.fetchall()]


def attribution(row: dict) -> str:
    who = row["observer_name"] or row["login"] or f"iNaturalist user {row['observer_id']}"
    if row["license"] == "CC0":
        return f"{who}, no rights reserved (CC0)"
    return f"(c) {who}, some rights reserved (CC BY)"


def download(row: dict, out: Path, size: str, session: requests.Session) -> dict | None:
    image = f"images/{row['group']}_{row['photo_id']}.{row['extension']}"
    path = out / image
    if not path.exists():
        url = f"{PHOTOS}/{row['photo_id']}/{size}.{row['extension']}"
        for _ in range(3):
            try:
                response = session.get(url, timeout=60)
                response.raise_for_status()
            except requests.RequestException:
                continue
            path.write_bytes(response.content)
            break
        else:
            return None
    return {
        "image": image,
        "group": row["group"],
        "taxon_id": row["taxon_id"],
        "taxon": row["taxon"],
        "rank": row["rank"],
        "observation": f"https://www.inaturalist.org/observations/{row['observation_uuid']}",
        "observed_on": row["observed_on"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "photo_id": row["photo_id"],
        "license": row["license"],
        "attribution": attribution(row),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dumps", type=Path, default=ROOT / "data" / "inaturalist" / "open-data-20260927"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", default=GROUPS, help="name:taxon_id:count, comma-separated")
    parser.add_argument(
        "--exclude", default="3", help="Comma-separated taxon IDs to drop (default: birds)"
    )
    parser.add_argument("--bbox", default=UK, help="min_lat,min_lon,max_lat,max_lon")
    parser.add_argument("--per-observer", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    index = args.dumps / f"index-{args.bbox.replace(',', '_')}.parquet"
    if not index.exists():
        print(f"Building {index} from the dumps (one-off)...", flush=True)
        build_index(args.dumps, index, args.bbox)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    meta = args.out / "meta.jsonl"
    previous = [json.loads(line) for line in meta.open()] if meta.exists() else []
    rows = sample(
        index,
        groups=args.groups,
        exclude=[int(taxon) for taxon in args.exclude.split(",") if taxon],
        per_observer=args.per_observer,
        seed=args.seed,
        existing={row["photo_id"] for row in previous},
    )
    session = requests.Session()
    session.headers["User-Agent"] = "BirdSpotter-inaturalist-sampler/1.0"
    session.mount("https://", HTTPAdapter(pool_maxsize=args.workers))
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(
            tqdm(
                pool.map(lambda row: download(row, args.out, args.size, session), rows),
                total=len(rows),
                desc="photos",
            )
        )
    saved = [result for result in results if result is not None]
    with meta.open("a") as handle:
        for result in saved:
            handle.write(json.dumps(result) + "\n")
    print(
        f"Saved {len(saved)} of {len(rows)} photos to {args.out} ({len(rows) - len(saved)} failed)"
    )


if __name__ == "__main__":
    main()
