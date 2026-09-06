# BirdSpotter

## Motivation

BirdSpotter watches a V4L2 camera for birds and saves the best detection from
each five-minute window as a transparent PNG. It uses a fine-tuned bird-only YOLO26s
OpenVINO detector followed by SAM 2.1 Large OpenVINO segmentation.

| Bird detection | Final transparent bird |
| --- | --- |
| <img src="demo/annotations/5.png" alt="Bird detection annotation" width="300"> | <img src="demo/segmentations/5.png" alt="Segmented bird" width="300"> |

## Setup

Python 3.12.13 and `uv` are required. Create the lean runtime environment:

```bash
uv sync --locked
```

The default `.venv` is for runtime, tests, and deployment. Runtime model files
belong in `weights/` and are not committed to Git.

Download and verify the public OpenVINO detector and SAM21 models from Hugging Face:

```bash
uv run python scripts/download_models.py
```

## Export

Model export uses a separate environment so Torch, Ultralytics, and regular
OpenCV do not affect the runtime environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-ml uv sync --locked --group ml
UV_PROJECT_ENVIRONMENT=.venv-ml \
  uv run --locked --group ml python scripts/ml/export_models.py
```

The exporter reads the fine-tuned YOLO26s checkpoint from its public Hugging Face
repository and downloads the SAM 2.1 Large checkpoint, then writes OpenVINO artifacts
to `weights/`. Source checkpoints remain in the Hugging Face cache.
Pass `--calibration-data path/to/dataset.yaml` to use different detector calibration
data.

## Demo

Regenerate the five demo annotations and segmentations:

```bash
uv run python scripts/generate_demo.py
```

## Deployment

```bash
uv run python scripts/deploy.py
```

By default, the camera is requested at 1600×896 MJPEG and 5 fps; the detector
uses the same 1600×896 input resolution and runs once per second. The capture loop drains intervening
frames without retrieving them and decodes the next frame once per second. SAM runs
once for the strictly highest-confidence bird in each five-minute window. Outputs are
written to `segmented/bird_conf_XX_ts_YYYY-MM-DD_HH-MM.png`.

The gallery shows the complete captured frame with the non-segmented region dimmed
and the detected bird outlined. The original transparent bird cutouts remain in
`segmented/`; gallery frames are stored in `segmented/gallery/`.

While deployment is running, the web page at `http://HOSTNAME:8080` opens on the 10 most
recently saved birds. Its **Area of interest** tab shows the live uncropped camera view;
drag a rectangle and apply it to update the production detection crop immediately. The ROI
is saved in `segmented/roi.json` and restored after a restart. Choose **Use full frame** to
clear it. Use `--web-host` and `--web-port` to change the default `0.0.0.0:8080` listener;
the page has no authentication, so expose it only on a trusted network.

### Tapo RTSP camera

Create a dedicated Camera Account in the Tapo app, then provide the C120 high-quality
stream as an environment variable.

```bash
export BIRDSPOTTER_RTSP_URL='rtsp://CAMERA_USERNAME:CAMERA_PASSWORD@CAMERA_IP:554/stream1'
uv run python scripts/deploy.py
```

`stream1` is the high-quality stream. BirdSpotter preserves its 1600×896 frames without
an additional detector resize. Use `/stream2` only if
the Beelink cannot keep up with `/stream1`.

Use `--log-level DEBUG` for per-frame detector timing and selection logs.

## Tests and checks

```bash
uv run pytest
uv run pre-commit run --all-files
```
