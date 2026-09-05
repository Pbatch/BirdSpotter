import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from birdspotter.gallery import london_timestamp, recent_birds, render_gallery, start_gallery_server


def create_bird(path: Path, age: int) -> Path:
    path.write_bytes(b"png-data")
    os.utime(path, ns=(age, age))
    return path


def test_recent_birds_returns_only_the_ten_newest_outputs(tmp_path: Path) -> None:
    birds = [
        create_bird(tmp_path / f"bird_conf_{index}_ts_2026-09-05_12-{index:02}.png", index)
        for index in range(12)
    ]
    create_bird(tmp_path / "unrelated.png", 100)

    assert recent_birds(tmp_path) == list(reversed(birds[2:]))


def test_render_gallery_shows_metadata_and_empty_state(tmp_path: Path) -> None:
    empty_page = render_gallery(tmp_path)

    assert b"Waiting for the first bird detection" in empty_page
    assert b"The 10 most recently spotted birds" not in empty_page
    assert b'<link rel="icon" type="image/png" href="/icon.png">' in empty_page
    assert b'<link rel="manifest" href="/manifest.webmanifest">' in empty_page

    bird = create_bird(tmp_path / "bird_conf_82_ts_2026-09-05_12-05.png", 1)
    page = render_gallery(tmp_path)

    assert bird.name.encode() in page
    assert b"2026-09-05 13:05 - 82% conf" in page
    assert b"BST" not in page
    assert b"<small>" not in page


def test_london_timestamp_omits_the_winter_timezone_label() -> None:
    assert london_timestamp("2026-12-05_12-05") == "2026-12-05 12:05"


def test_gallery_server_serves_page_and_bird_but_not_other_files(tmp_path: Path) -> None:
    bird = create_bird(tmp_path / "bird_conf_91_ts_2026-09-05_12-10.png", 1)
    icon = tmp_path / "icon.png"
    icon.write_bytes(b"icon-data")
    create_bird(tmp_path / "private.txt", 2)
    server = start_gallery_server(
        tmp_path,
        "127.0.0.1",
        0,
        icon_path=icon,
    )
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        with urlopen(f"{base_url}/", timeout=2) as response:  # noqa: S310
            assert response.status == 200
            assert bird.name.encode() in response.read()
        with urlopen(f"{base_url}/birds/{bird.name}", timeout=2) as response:  # noqa: S310
            assert response.headers["Content-Type"] == "image/png"
            assert response.read() == b"png-data"
        with urlopen(f"{base_url}/icon.png", timeout=2) as response:  # noqa: S310
            assert response.headers["Content-Type"] == "image/png"
            assert response.read() == b"icon-data"
        with urlopen(f"{base_url}/manifest.webmanifest", timeout=2) as response:  # noqa: S310
            manifest = json.load(response)
            assert manifest["icons"][0]["src"] == "/icon.png"
            assert manifest["display"] == "standalone"
        with pytest.raises(HTTPError) as error:
            urlopen(f"{base_url}/birds/private.txt", timeout=2)  # noqa: S310
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
