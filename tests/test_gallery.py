import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from urllib.response import addinfourl

import numpy as np
import pytest
from starlette.testclient import TestClient

from birdspotter.capture import CapturedFrame, load_roi_config
from birdspotter.sighting import recent_birds
from birdspotter.web.gallery import (
    create_gallery_app,
    london_timestamp,
    start_gallery_server,
)


def create_window(output_dir: Path, name: str, *, bird: bool = True) -> Path:
    window = output_dir / name
    window.mkdir()
    (window / "full_frame.jpg").write_bytes(b"jpg-data")
    if bird:
        (window / "mask.png").write_bytes(b"png-data")
    label = {
        "width": 30,
        "height": 20,
        "roi": [2, 1, 26, 19],
        "bbox": [6, 3, 20, 17] if bird else None,
        "confidence": 0.82 if bird else None,
        "sam_confidence": 0.9 if bird else None,
    }
    (window / "label.json").write_text(json.dumps(label))
    return window


def open_http(url: str | Request) -> addinfourl:
    """Open a test-only URL known to use the local HTTP server."""

    return urlopen(url, timeout=2)  # noqa: S310


def test_recent_birds_returns_only_the_ten_newest_bird_windows(tmp_path: Path) -> None:
    birds = [create_window(tmp_path, f"2026-09-05_12-{index:02}") for index in range(12)]
    create_window(tmp_path, "2026-09-05_13-00", bird=False)
    (tmp_path / "unrelated").mkdir()
    (tmp_path / "roi.json").write_text("{}")

    assert [sighting.path for sighting in recent_birds(tmp_path, 10)] == list(reversed(birds[2:]))


def test_templates_show_metadata_default_tab_and_empty_state(tmp_path: Path) -> None:
    client = TestClient(create_gallery_app(tmp_path))
    empty_page = client.get("/").content

    assert b"Waiting for the first bird sighting" in empty_page
    assert b'<link rel="icon" type="image/png" href="/icon.png">' in empty_page
    assert b'<link rel="stylesheet" href="/static/gallery.css">' in empty_page
    assert b'<script src="/static/gallery.js?v=sighting-roi-1" defer></script>' in empty_page
    assert b'id="sightings-tab" class="active"' in empty_page
    assert b'id="sightings-panel" class="tab-panel"' in empty_page
    assert b'id="roi-panel" class="tab-panel" hidden' in empty_page
    assert b'http-equiv="refresh"' not in empty_page

    create_window(tmp_path, "2026-09-05_12-05")
    sightings = client.get("/sightings.html").content
    assert b"2026-09-05 13:05 - 82% conf" in sightings
    assert b'data-frame="/windows/2026-09-05_12-05/full_frame.jpg"' in sightings
    assert b'data-mask="/windows/2026-09-05_12-05/mask.png"' in sightings
    assert b'data-bbox="6,3,20,17"' in sightings
    assert b'data-roi="2,1,26,19"' in sightings
    assert b'width="24" height="18"' in sightings


def test_london_timestamp_omits_the_winter_timezone_label() -> None:
    assert london_timestamp("2026-12-05_12-05") == "2026-12-05 12:05"


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
    with start_gallery_server(
        tmp_path,
        "127.0.0.1",
        0,
        camera=camera,  # ty: ignore[invalid-argument-type]
        roi_config_path=config_path,
    ) as server:
        base_url = f"http://127.0.0.1:{server.server_port}"
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
    assert camera.roi == (4, 5, 34, 35)
    assert load_roi_config(config_path) == camera.roi


def test_starlette_app_serves_gallery_assets_and_protects_other_files(tmp_path: Path) -> None:
    window = create_window(tmp_path, "2026-09-05_12-10")
    icon = tmp_path / "icon.png"
    icon.write_bytes(b"icon-data")
    (tmp_path / "private.txt").write_bytes(b"secret")
    with start_gallery_server(tmp_path, "127.0.0.1", 0, icon_path=icon) as server:
        base_url = f"http://127.0.0.1:{server.server_port}"
        with open_http(f"{base_url}/") as response:
            assert response.headers["Cache-Control"] == "no-cache"
            assert b"/windows/2026-09-05_12-10/full_frame.jpg" in response.read()
        with open_http(f"{base_url}/sightings.html") as response:
            assert b"/windows/2026-09-05_12-10/mask.png" in response.read()
        with open_http(f"{base_url}/windows/{window.name}/full_frame.jpg") as response:
            assert response.headers["Content-Type"] == "image/jpeg"
            assert response.read() == b"jpg-data"
        with open_http(f"{base_url}/windows/{window.name}/mask.png") as response:
            assert response.headers["Content-Type"] == "image/png"
            assert response.read() == b"png-data"
        with open_http(f"{base_url}/icon.png") as response:
            assert response.read() == b"icon-data"
        with open_http(f"{base_url}/manifest.webmanifest") as response:
            assert json.load(response)["display"] == "standalone"
        with open_http(f"{base_url}/static/gallery.js") as response:
            javascript = response.read()
            assert b"setInterval(refreshSightings, 30000)" in javascript
            assert b"drawSightings();" in javascript
            assert javascript.count(b"selection = squareSelection(start, point(event));") == 2
            assert b"if (showRoi) loadRoi();" in javascript
            assert b"selection = null;" in javascript
        for url in (
            f"{base_url}/private.txt",
            f"{base_url}/windows/{window.name}/label.json",
            f"{base_url}/windows/{window.name}/private.txt",
            f"{base_url}/windows/unrelated/full_frame.jpg",
        ):
            with pytest.raises(HTTPError) as captured:
                open_http(url)
            assert captured.value.code == 404


def test_uvicorn_wrapper_serves_starlette_app(tmp_path: Path) -> None:
    with (
        start_gallery_server(tmp_path, "127.0.0.1", 0) as server,
        open_http(f"http://127.0.0.1:{server.server_port}/") as response,
    ):
        assert response.status == 200
        assert b"BirdSpotter" in response.read()
