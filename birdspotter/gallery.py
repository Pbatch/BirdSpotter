"""Starlette web application for sightings and camera ROI selection."""

from __future__ import annotations

import json
import re
import socket
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread
from zoneinfo import ZoneInfo

import cv2
import uvicorn
from loguru import logger
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from birdspotter.capture import Capture, Roi, crop_to_roi

DEFAULT_GALLERY_LIMIT = 10
DEFAULT_GALLERY_HOST = "0.0.0.0"  # noqa: S104 - intended for trusted-LAN access
LONDON_TIMEZONE = ZoneInfo("Europe/London")
PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
TEMPLATE_DIR = PACKAGE_DIR / "templates"
WINDOW_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$")
WINDOW_FILES = {"full_frame.jpg": "image/jpeg", "mask.png": "image/png"}
templates = Jinja2Templates(directory=TEMPLATE_DIR)


def load_roi_config(path: Path) -> Roi | None:
    """Load a persisted ROI, returning full-frame mode when absent."""

    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    value = data.get("roi")
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4 or any(not isinstance(x, int) for x in value):
        raise ValueError(f"Invalid ROI configuration: {path}")
    return tuple(value)


def write_roi_config(path: Path, roi: Roi | None) -> None:
    """Atomically persist the production ROI."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps({"roi": roi}, indent=2) + "\n")
    temporary.replace(path)


def london_timestamp(filename_timestamp: str) -> str:
    """Convert a BirdSpotter UTC filename timestamp to London local time."""

    captured_at = datetime.strptime(filename_timestamp, "%Y-%m-%d_%H-%M").replace(tzinfo=UTC)
    return captured_at.astimezone(LONDON_TIMEZONE).strftime("%Y-%m-%d %H:%M")


def recent_birds(output_dir: Path, limit: int = DEFAULT_GALLERY_LIMIT) -> list[Path]:
    """Return the newest window directories containing a segmented bird, newest first."""

    birds = []
    windows = sorted(
        (
            path
            for path in output_dir.iterdir()
            if path.is_dir() and WINDOW_NAME.fullmatch(path.name)
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    for window in windows:
        label_path = window / "label.json"
        if not window.is_dir() or not label_path.is_file() or not (window / "mask.png").is_file():
            continue
        if json.loads(label_path.read_text()).get("bbox") is None:
            continue
        birds.append(window)
        if len(birds) == limit:
            break
    return birds


def sighting_context(output_dir: Path, limit: int = DEFAULT_GALLERY_LIMIT) -> list[dict]:
    """Build template values for the latest sightings, composed in the browser."""

    sightings = []
    for window in recent_birds(output_dir, limit):
        label = json.loads((window / "label.json").read_text())
        roi = label.get("roi") or [0, 0, label["width"], label["height"]]
        sightings.append(
            {
                "confidence": round(label["confidence"] * 100),
                "timestamp": london_timestamp(window.name),
                "frame_url": f"/windows/{window.name}/full_frame.jpg",
                "mask_url": f"/windows/{window.name}/mask.png",
                "width": roi[2] - roi[0],
                "height": roi[3] - roi[1],
                "roi": ",".join(str(int(value)) for value in roi),
                "bbox": ",".join(str(int(value)) for value in label["bbox"]),
            }
        )
    return sightings


def render_sightings(output_dir: Path, limit: int = DEFAULT_GALLERY_LIMIT) -> bytes:
    """Render only the replaceable recent-sightings card grid contents."""

    template = templates.get_template("sightings.html")
    return template.render(sightings=sighting_context(output_dir, limit)).encode()


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
    """Render the complete gallery page outside a request context."""

    template = templates.get_template("gallery.html")
    return template.render(sightings=sighting_context(output_dir, limit)).encode()


def require_camera(request: Request) -> Capture:
    """Return the configured camera or report an unavailable endpoint."""

    camera: Capture | None = request.app.state.camera
    if camera is None:
        raise HTTPException(503, "Camera ROI selection is unavailable")
    return camera


def homepage(request: Request) -> Response:
    """Render the complete sightings and ROI page."""

    return templates.TemplateResponse(
        request,
        "gallery.html",
        {"sightings": sighting_context(request.app.state.output_dir)},
        headers={"Cache-Control": "no-cache"},
    )


def sightings_fragment(request: Request) -> Response:
    """Render the recent cards for periodic in-page replacement."""

    return templates.TemplateResponse(
        request,
        "sightings.html",
        {"sightings": sighting_context(request.app.state.output_dir)},
    )


def manifest(_: Request) -> Response:
    """Return the progressive-web-app manifest."""

    return Response(render_manifest(), media_type="application/manifest+json")


def icon(request: Request) -> Response:
    """Return the configured gallery icon."""

    path: Path | None = request.app.state.icon_path
    if path is None or not path.is_file():
        raise HTTPException(404)
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def camera_frame(request: Request) -> Response:
    """Encode and return the newest uncropped camera frame."""

    frame = require_camera(request).newest_source(timeout=2).image_bgr
    encoded, payload = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not encoded:
        raise HTTPException(500, "Could not encode camera frame")
    return Response(
        payload.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


def roi_state(request: Request) -> Response:
    """Return the active ROI and uncropped camera dimensions."""

    camera = require_camera(request)
    frame = camera.newest_source(timeout=2).image_bgr
    height, width = frame.shape[:2]
    return JSONResponse({"roi": camera.roi, "width": width, "height": height})


async def update_roi(request: Request) -> Response:
    """Validate, activate, and persist a selected ROI."""

    camera = require_camera(request)
    config_path: Path | None = request.app.state.roi_config_path
    if config_path is None:
        raise HTTPException(503, "Camera ROI selection is unavailable")
    try:
        content_length = int(request.headers.get("content-length", "0"))
    except ValueError as error:
        raise HTTPException(400, "Invalid request size") from error
    if content_length < 1 or content_length > 1024:
        raise HTTPException(400, "Invalid request size")
    try:
        data = await request.json()
        roi: Roi | None = None
        if data is not None:
            roi = (
                int(data["left"]),
                int(data["top"]),
                int(data["right"]),
                int(data["bottom"]),
            )
            frame = camera.newest_source(timeout=2).image_bgr
            crop_to_roi(frame, roi)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(400, str(error)) from error
    camera.set_roi(roi)
    write_roi_config(config_path, roi)
    logger.info("Classification ROI updated | roi={}", roi if roi is not None else "full frame")
    return JSONResponse({"roi": roi})


def window_file(request: Request) -> Response:
    """Safely return a saved full frame or bird mask for browser-side composition."""

    window = request.path_params["window"]
    filename = request.path_params["filename"]
    if WINDOW_NAME.fullmatch(window) is None or filename not in WINDOW_FILES:
        raise HTTPException(404)
    directory = request.app.state.output_dir.resolve()
    path = directory / window / filename
    if path.resolve().parent.parent != directory or not path.is_file():
        raise HTTPException(404)
    return FileResponse(
        path,
        media_type=WINDOW_FILES[filename],
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


def create_gallery_app(
    output_dir: Path,
    *,
    icon_path: Path | None = None,
    camera: Capture | None = None,
    roi_config_path: Path | None = None,
) -> Starlette:
    """Create a configured Starlette gallery application."""

    routes = [
        Route("/", homepage, name="home"),
        Route("/sightings.html", sightings_fragment, name="sightings"),
        Route("/manifest.webmanifest", manifest, name="manifest"),
        Route("/icon.png", icon, name="icon"),
        Route("/camera.jpg", camera_frame, name="camera"),
        Route("/roi.json", roi_state, name="roi-state"),
        Route("/roi", update_roi, methods=["POST"], name="roi-update"),
        Route("/windows/{window}/{filename}", window_file, name="window-file"),
        Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),
    ]
    app = Starlette(routes=routes)
    app.state.output_dir = output_dir.resolve()
    app.state.icon_path = icon_path.resolve() if icon_path is not None else None
    app.state.camera = camera
    app.state.roi_config_path = roi_config_path.resolve() if roi_config_path is not None else None
    return app


class BirdGalleryServer:
    """Run the Starlette gallery through Uvicorn in a background thread."""

    def __init__(self, app: Starlette, host: str, port: int) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((host, port))
        self._socket.listen(128)
        self.server_port = self._socket.getsockname()[1]
        config = uvicorn.Config(app, log_config=None, access_log=False, lifespan="off")
        self._server = uvicorn.Server(config)
        self._thread = Thread(
            target=self._server.run,
            kwargs={"sockets": [self._socket]},
            name="birdspotter-gallery",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + 5
        while not self._server.started and self._thread.is_alive():
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out starting the gallery server")
            time.sleep(0.01)
        if not self._thread.is_alive():
            raise RuntimeError("Gallery server stopped during startup")

    def shutdown(self) -> None:
        """Request a graceful server shutdown and wait for its thread."""

        self._server.should_exit = True
        self._thread.join(timeout=5)

    def server_close(self) -> None:
        """Close the listening socket when it remains open."""

        with suppress(OSError):
            self._socket.close()


def start_gallery_server(  # noqa: PLR0913
    output_dir: Path,
    host: str,
    port: int,
    *,
    icon_path: Path | None = None,
    camera: Capture | None = None,
    roi_config_path: Path | None = None,
) -> BirdGalleryServer:
    """Create the Starlette app and run it in a background Uvicorn server."""

    app = create_gallery_app(
        output_dir,
        icon_path=icon_path,
        camera=camera,
        roi_config_path=roi_config_path,
    )
    return BirdGalleryServer(app, host, port)
