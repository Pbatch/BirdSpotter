import json
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from urllib.response import addinfourl

import numpy as np
import pytest

from birdspotter.capture import CapturedFrame
from birdspotter.gallery import (
    load_roi_config,
    london_timestamp,
    recent_birds,
    render_gallery,
    render_sightings,
    start_gallery_server,
    write_roi_config,
)


def create_bird(path: Path, age: int) -> Path:
    path.write_bytes(b"png-data")
    os.utime(path, ns=(age, age))
    return path


def open_http(url: str | Request) -> addinfourl:
    """Open a test-only URL known to use the local HTTP server."""

    return urlopen(url, timeout=2)  # noqa: S310


def test_recent_birds_returns_only_the_ten_newest_outputs(tmp_path: Path) -> None:
    birds = [
        create_bird(tmp_path / f"bird_conf_{index}_ts_2026-09-05_12-{index:02}.png", index)
        for index in range(12)
    ]
    create_bird(tmp_path / "unrelated.png", 100)

    assert recent_birds(tmp_path) == list(reversed(birds[2:]))


def test_templates_show_metadata_default_tab_and_empty_state(tmp_path: Path) -> None:
    empty_page = render_gallery(tmp_path)

    assert b"Waiting for the first bird detection" in empty_page
    assert b'<link rel="icon" type="image/png" href="/icon.png">' in empty_page
    assert b'<link rel="stylesheet" href="/static/gallery.css">' in empty_page
    assert b'<script src="/static/gallery.js?v=live-roi-1" defer></script>' in empty_page
    assert b'id="sightings-tab" class="active"' in empty_page
    assert b'id="sightings-panel" class="tab-panel"' in empty_page
    assert b'id="roi-panel" class="tab-panel" hidden' in empty_page
    assert b'http-equiv="refresh"' not in empty_page

    bird = create_bird(tmp_path / "bird_conf_82_ts_2026-09-05_12-05.png", 1)
    assert bird.name.encode() in render_gallery(tmp_path)
    assert b"2026-09-05 13:05 - 82% conf" in render_sightings(tmp_path)

    gallery_dir = tmp_path / "gallery"
    gallery_dir.mkdir()
    create_bird(gallery_dir / bird.name, 2)
    assert f"/frames/{bird.name}".encode() in render_sightings(tmp_path)


def test_london_timestamp_omits_the_winter_timezone_label() -> None:
    assert london_timestamp("2026-12-05_12-05") == "2026-12-05 12:05"


def test_roi_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "roi.json"

    assert load_roi_config(path) is None
    write_roi_config(path, (100, 200, 900, 648))
    assert load_roi_config(path) == (100, 200, 900, 648)
    write_roi_config(path, None)
    assert load_roi_config(path) is None


class FakeCamera:
    def __init__(self) -> None:
        image = np.full((48, 64, 3), 100, dtype=np.uint8)
        self.frame = CapturedFrame(1, datetime.now(UTC), image)
        self.roi = None

    def newest_source(self, *, timeout: float = 2.0) -> CapturedFrame:
        del timeout
        return self.frame

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        self.roi = roi


def test_starlette_app_updates_production_roi(tmp_path: Path) -> None:
    camera = FakeCamera()
    config_path = tmp_path / "roi.json"
    server = start_gallery_server(
        tmp_path,
        "127.0.0.1",
        0,
        camera=camera,  # ty: ignore[invalid-argument-type]
        roi_config_path=config_path,
    )
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with open_http(f"{base_url}/camera.jpg") as response:
            assert response.headers["Content-Type"] == "image/jpeg"
            assert response.read().startswith(b"\xff\xd8")
        request = Request(  # noqa: S310
            f"{base_url}/roi",
            data=json.dumps({"left": 4, "top": 5, "right": 34, "bottom": 35}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with open_http(request) as response:
            assert response.status == 200
        with open_http(f"{base_url}/roi.json") as response:
            assert json.load(response) == {
                "roi": [4, 5, 34, 35],
                "width": 64,
                "height": 48,
            }
    finally:
        server.shutdown()
        server.server_close()
    assert camera.roi == (4, 5, 34, 35)
    assert load_roi_config(config_path) == camera.roi


def test_starlette_app_serves_gallery_assets_and_protects_other_files(tmp_path: Path) -> None:
    bird = create_bird(tmp_path / "bird_conf_91_ts_2026-09-05_12-10.png", 1)
    icon = tmp_path / "icon.png"
    icon.write_bytes(b"icon-data")
    create_bird(tmp_path / "private.txt", 2)
    gallery_dir = tmp_path / "gallery"
    gallery_dir.mkdir()
    frame = create_bird(gallery_dir / bird.name, 2)
    server = start_gallery_server(tmp_path, "127.0.0.1", 0, icon_path=icon)
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with open_http(f"{base_url}/") as response:
            assert response.headers["Cache-Control"] == "no-cache"
            assert bird.name.encode() in response.read()
        with open_http(f"{base_url}/sightings.html") as response:
            assert bird.name.encode() in response.read()
        with open_http(f"{base_url}/birds/{bird.name}") as response:
            assert response.read() == b"png-data"
        with open_http(f"{base_url}/frames/{frame.name}") as response:
            assert response.read() == b"png-data"
        with open_http(f"{base_url}/icon.png") as response:
            assert response.read() == b"icon-data"
        with open_http(f"{base_url}/manifest.webmanifest") as response:
            assert json.load(response)["display"] == "standalone"
        with open_http(f"{base_url}/static/gallery.js") as response:
            javascript = response.read()
            assert b"setInterval(refreshSightings, 30000)" in javascript
            assert javascript.count(b"selection = squareSelection(start, point(event));") == 2
            assert b"if (showRoi) loadRoi();" in javascript
            assert b"selection = null;" in javascript
        for url in (f"{base_url}/birds/private.txt", f"{base_url}/private.txt"):
            with pytest.raises(HTTPError) as captured:
                open_http(url)
            assert captured.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_uvicorn_wrapper_serves_starlette_app(tmp_path: Path) -> None:
    server = start_gallery_server(tmp_path, "127.0.0.1", 0)
    try:
        with open_http(f"http://127.0.0.1:{server.server_port}/") as response:
            assert response.status == 200
            assert b"BirdSpotter" in response.read()
    finally:
        server.shutdown()
        server.server_close()
