"""Small local web gallery for recent BirdSpotter output."""

from __future__ import annotations

import html
import json
import re
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import quote, unquote, urlsplit
from zoneinfo import ZoneInfo

from loguru import logger

DEFAULT_GALLERY_LIMIT = 10
DEFAULT_GALLERY_HOST = "0.0.0.0"  # noqa: S104 - intended for trusted-LAN access
LONDON_TIMEZONE = ZoneInfo("Europe/London")
BIRD_FILENAME = re.compile(
    r"^bird_conf_(?P<confidence>\d+)_ts_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2})\.png$"
)


def london_timestamp(filename_timestamp: str) -> str:
    """Convert a BirdSpotter UTC filename timestamp to London local time."""

    captured_at = datetime.strptime(filename_timestamp, "%Y-%m-%d_%H-%M").replace(tzinfo=UTC)
    return captured_at.astimezone(LONDON_TIMEZONE).strftime("%Y-%m-%d %H:%M")


def recent_birds(output_dir: Path, limit: int = DEFAULT_GALLERY_LIMIT) -> list[Path]:
    """Return the newest generated bird images, newest first."""

    candidates = (
        path
        for path in output_dir.iterdir()
        if path.is_file() and BIRD_FILENAME.fullmatch(path.name)
    )
    return sorted(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )[:limit]


def render_manifest() -> bytes:
    """Render metadata used when the gallery is installed as a local app."""

    return json.dumps(
        {
            "name": "BirdSpotter",
            "short_name": "BirdSpotter",
            "id": "/",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#101712",
            "theme_color": "#101712",
            "icons": [
                {
                    "src": "/icon.png",
                    "sizes": "512x512",
                    "type": "image/png",
                    "purpose": "any",
                }
            ],
        },
        separators=(",", ":"),
    ).encode()


def render_gallery(output_dir: Path, limit: int = DEFAULT_GALLERY_LIMIT) -> bytes:
    """Render a self-contained HTML page for the latest bird images."""

    cards: list[str] = []
    for path in recent_birds(output_dir, limit):
        match = BIRD_FILENAME.fullmatch(path.name)
        if match is None:  # pragma: no cover - filtered by recent_birds
            continue
        confidence = html.escape(match.group("confidence"))
        timestamp = html.escape(london_timestamp(match.group("timestamp")))
        image_url = f"/birds/{quote(path.name)}"
        cards.append(
            f"""
            <figure class="bird-card">
              <a href="{image_url}">
                <img src="{image_url}" alt="Bird detected at {timestamp}" loading="lazy">
              </a>
              <figcaption>
                {timestamp} - {confidence}% conf
              </figcaption>
            </figure>
            """
        )

    content = "".join(cards)
    if not content:
        content = '<p class="empty">Waiting for the first bird detection&hellip;</p>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="30">
  <meta name="theme-color" content="#101712">
  <link rel="icon" type="image/png" href="/icon.png">
  <link rel="apple-touch-icon" href="/icon.png">
  <link rel="manifest" href="/manifest.webmanifest">
  <title>BirdSpotter</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; background: #101712; color: #eff8f1; }}
    header {{ max-width: 1100px; margin: auto; padding: 2rem 1rem 1rem; }}
    h1 {{ margin: 0; font-size: clamp(2rem, 6vw, 4rem); letter-spacing: -.05em; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1rem; max-width: 1100px; margin: auto; padding: 1rem; }}
    .bird-card {{ margin: 0; overflow: hidden; border: 1px solid #33483a; border-radius: 1rem;
                  background: #18221b; }}
    .bird-card a {{ display: grid; min-height: 240px; place-items: center;
                    background: #dfe7e1; }}
    img {{ display: block; max-width: 100%; max-height: 420px; object-fit: contain; }}
    figcaption {{ padding: .9rem 1rem 1rem; color: #a9baad; white-space: nowrap; }}
    .empty {{ grid-column: 1 / -1; padding: 4rem 1rem; border: 1px dashed #526b59;
              border-radius: 1rem; color: #a9baad; text-align: center; }}
  </style>
</head>
<body>
  <header>
    <h1>BirdSpotter</h1>
  </header>
  <main>{content}</main>
</body>
</html>
""".encode()


class BirdGalleryServer(ThreadingHTTPServer):
    """HTTP server carrying the gallery's output directory."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        output_dir: Path,
        icon_path: Path | None,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.icon_path = icon_path.resolve() if icon_path is not None else None
        super().__init__(address, BirdGalleryRequestHandler)


class BirdGalleryRequestHandler(BaseHTTPRequestHandler):
    """Serve the gallery page and its generated bird images."""

    server: BirdGalleryServer

    def do_GET(self) -> None:
        request_path = urlsplit(self.path).path
        if request_path == "/":
            self._send(render_gallery(self.server.output_dir), "text/html; charset=utf-8")
            return
        if request_path == "/manifest.webmanifest":
            self._send(render_manifest(), "application/manifest+json")
            return
        if request_path == "/icon.png":
            self._send_icon(self.server.icon_path)
            return
        if request_path.startswith("/birds/"):
            self._send_bird(unquote(request_path.removeprefix("/birds/")))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _send_bird(self, filename: str) -> None:
        if Path(filename).name != filename or BIRD_FILENAME.fullmatch(filename) is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        path = self.server.output_dir / filename
        try:
            if path.resolve().parent != self.server.output_dir or not path.is_file():
                raise FileNotFoundError(filename)
            body = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send(body, "image/png", cache_control="public, max-age=31536000, immutable")

    def _send_icon(self, path: Path | None) -> None:
        try:
            if path is None or not path.is_file():
                raise FileNotFoundError
            body = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send(body, "image/png", cache_control="public, max-age=86400")

    def _send(self, body: bytes, content_type: str, *, cache_control: str = "no-store") -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        logger.debug("Gallery request | client={} " + format, self.client_address[0], *args)


def start_gallery_server(
    output_dir: Path,
    host: str,
    port: int,
    *,
    icon_path: Path | None = None,
) -> BirdGalleryServer:
    """Start the gallery server in a background daemon thread."""

    server = BirdGalleryServer((host, port), output_dir, icon_path)
    thread = Thread(target=server.serve_forever, name="birdspotter-gallery", daemon=True)
    thread.start()
    return server
