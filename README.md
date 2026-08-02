# BirdSpotter

BirdSpotter detects birds from a V4L2 camera and saves the highest-confidence
bird from each five-minute window as a transparent PNG. It uses a bird-only
YOLO26s OpenVINO detector and SAM 2.1 Large OpenVINO segmentation.

Defaults: 1600×896 MJPEG camera input at 5 fps, 1088×1920 detector input, and
a 512×512 SAM image encoder. The capture thread keeps only the newest frame.

Outputs are written to `segmented/` as:

```text
bird_conf_XX_ts_YYYY-MM-DD_HH-MM.png
```

## Setup and model export

Python 3.12.13 and `uv` are required. Export the runtime models on a development
machine:

```bash
uv sync --group model-export
uv run --group model-export python scripts/export_models.py
uv sync
uv sync --reinstall-package opencv-python-headless
```

The exporter downloads and verifies the YOLO26s and SAM 2.1 Large checkpoints,
then writes the OpenVINO runtime artifacts to `weights/`. Source checkpoints are
kept in ignored `weights_dev/`. Use `--calibration-data path/to/dataset.yaml` to
replace the default detector calibration data at `demo/calibration.yaml`.

## Deploy

```bash
uv run python scripts/deploy.py
```

The detector is scheduled once per second, retaining only a strictly better
bird confidence within each window. SAM runs once for that window's winner.

## Demo and checks

Regenerate the five demo annotations and segmentations:

```bash
uv run python scripts/generate_demo.py
```

Run tests and development checks:

```bash
uv run pytest
uv run pre-commit run --all-files
```

## Licensing

`LICENSE.md` covers BirdSpotter's original code. SAM 2.1 and YOLO26 checkpoints
and derived artifacts remain subject to their upstream licences; see
[third-party model notices](THIRD_PARTY_NOTICES.md) before distributing them.
