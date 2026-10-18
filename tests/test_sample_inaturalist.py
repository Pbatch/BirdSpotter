import gzip
from collections.abc import Sequence
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

from scripts.ml.sample_inaturalist import attribution, build_index, sample


def write_tsv(path: Path, rows: Sequence[Sequence[object]]) -> None:
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write("\t".join("" if value is None else str(value) for value in row) + "\n")


def make_dumps(dumps: Path) -> None:
    write_tsv(
        dumps / "taxa.csv.gz",
        [
            ["taxon_id", "ancestry", "rank_level", "rank", "name", "active"],
            [47126, "48460", 70, "kingdom", "Plantae", "true"],
            [100, "48460/47126", 10, "species", "Hedera helix", "true"],
            [3, "48460/1/2/355675", 50, "class", "Aves", "true"],
            [200, "48460/1/2/355675/3", 10, "species", "Parus major", "true"],
            [46017, "48460/1/2/355675/40151", 10, "species", "Sciurus carolinensis", "true"],
        ],
    )
    write_tsv(
        dumps / "observers.csv.gz",
        [["observer_id", "login", "name"], [1, "alice", "Alice A"], [2, "bob", None]],
    )
    header = [
        "observation_uuid",
        "observer_id",
        "latitude",
        "longitude",
        "positional_accuracy",
        "taxon_id",
        "quality_grade",
        "observed_on",
        "anomaly_score",
    ]
    observations = [header]
    photos = [
        [
            "photo_uuid",
            "photo_id",
            "observation_uuid",
            "observer_id",
            "extension",
            "license",
            "width",
            "height",
            "position",
        ]
    ]
    # Five ivy observations by alice and one by bob, all in the UK.
    for index in range(6):
        observer = 1 if index < 5 else 2
        observations.append(
            [f"ivy{index}", observer, 52.0, -1.0, None, 100, "research", "2025-05-01", 1]
        )
        photos.append(
            [f"p{index}", 1000 + index, f"ivy{index}", observer, "jpg", "CC-BY", 500, 400, 0]
        )
    observations += [
        ["tit", 2, 52.0, -1.0, None, 200, "research", "2025-05-01", 1],  # bird: excluded
        ["squirrel", 2, 52.0, -1.0, None, 46017, "research", "2025-05-01", 1],
        ["paris", 2, 48.8, 2.3, None, 100, "research", "2025-05-01", 1],  # outside the UK
        ["casual", 2, 52.0, -1.0, None, 100, "needs_id", "2025-05-01", 1],
        ["nc", 2, 52.0, -1.0, None, 100, "research", "2025-05-01", 1],
    ]
    photos += [
        ["pt", 2000, "tit", 2, "jpg", "CC0", 500, 400, 0],
        # The first photo is non-commercial, so the CC0 second photo is used.
        ["ps0", 3000, "squirrel", 2, "jpg", "CC-BY-NC", 500, 400, 0],
        ["ps1", 3001, "squirrel", 2, "png", "CC0", 500, 400, 1],
        ["pp", 4000, "paris", 2, "jpg", "CC-BY", 500, 400, 0],
        ["pc", 5000, "casual", 2, "jpg", "CC-BY", 500, 400, 0],
        ["pn", 6000, "nc", 2, "jpg", "CC-BY-NC", 500, 400, 0],
    ]
    write_tsv(dumps / "observations.csv.gz", observations)
    write_tsv(dumps / "photos.csv.gz", photos)


def test_index_and_sample_filter_licence_place_taxa_and_observers(tmp_path: Path) -> None:
    make_dumps(tmp_path)
    index = tmp_path / "index.parquet"
    build_index(tmp_path, index, "49.8,-8.7,60.9,1.8")
    rows = sample(
        index,
        groups="plant:47126:10,grey_squirrel:46017:10,bird:3:10",
        exclude=[3],
        per_observer=3,
        seed=0,
        existing={1005},
    )
    plants = [row for row in rows if row["group"] == "plant"]
    squirrels = [row for row in rows if row["group"] == "grey_squirrel"]
    assert {row["observer_id"] for row in plants} == {1}  # bob's ivy photo 1005 is excluded
    assert len(plants) == 3  # alice is capped at three
    assert all(row["license"] == "CC-BY" for row in plants)
    assert [(row["photo_id"], row["extension"], row["license"]) for row in squirrels] == [
        (3001, "png", "CC0")
    ]
    assert not [row for row in rows if row["group"] == "bird"]
    assert attribution(squirrels[0]) == "bob, no rights reserved (CC0)"
    assert attribution(plants[0]) == "(c) Alice A, some rights reserved (CC BY)"
