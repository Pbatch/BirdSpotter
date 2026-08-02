# BirdSpotter

This pipeline watches a V4L2 webcam, detects only COCO class `bird`, and keeps
the single detection with the highest object-detector confidence in each fixed
selection window. SAM 2.1 OpenVINO receives that detection as a box prompt. The primary
output is one tightly cropped PNG containing the original bird pixels and a
transparent background.

The initial target is a Beelink Mini S12 with an Intel N100 and a Logitech C920.
The C920 captures 1600×896 MJPEG at 5 fps. YOLO26s uses a static 1088×1920 detector
input with slight vertical letterbox padding and samples the
newest frame at no more than 1 fps by default. The capture thread continually discards stale frames, so a
slow inference call does not build an increasingly delayed video queue.

## Outputs

For each accepted selection window, `segmented/` receives:

- `bird_conf_XX_ts_YYYY-MM-DD_HH-MM.png`: the required tightly cropped BGRA/RGBA bird image

Exactly one bird is segmented. Multiple birds are not merged. Blur, pose, and
exposure scoring are deliberately absent for now; detector confidence is the
only quality proxy. If the winning detector box itself encloses overlapping
birds, SAM can include that overlap; the pipeline does not replace the winning
detection with a lower-confidence box to avoid it.

## Demo assets

`demo/images/` contains five source images. `demo/annotations/` shows the active
detector's selected bird box, and `demo/segmentations/` contains the corresponding
final transparent PNG outputs. Regenerate both with:

```bash
uv run python scripts/generate_demo.py
```

## Environment and model preparation

All Python packages are declared in `pyproject.toml` and installed through
`uv sync`. The project pins Python 3.12 because inference wheels do not yet have
uniform Python 3.14 support.

Prepare the model artifacts on the development machine:

```bash
uv sync --group model-export
uv run --group model-export python scripts/export_models.py
uv sync
uv sync --reinstall-package opencv-python-headless
```

The second `uv sync` removes export-only quantization packages and leaves the runtime
environment at the locked dependency set. OpenCV is explicitly reinstalled because
its GUI and headless wheels share `cv2` files.

The export script creates the bird-only detector from `weights_dev/detector/yolo26s.pt`
using `demo/calibration.yaml` by default, and converts the SAM checkpoint at
`weights_dev/sam2.1/sam2.1_l.pt`. It writes only the runtime OpenVINO artifacts under
`weights/`. Pass `--calibration-data path/to/dataset.yaml` to use a representative
detector calibration dataset.

BirdSpotter always uses the fixed 512×512 SAM 2.1 OpenVINO segmenter. Its conversion
creates an image encoder and box-prompt mask predictor under
`weights/sam2.1/openvino-512/`. SAM 2.1 video tracking needs additional memory-model
conversions and is outside this pipeline.

Detector inference always uses a static batch-one INT8 OpenVINO model whose classification
heads have been reduced from 80 COCO classes to the pretrained `bird` channel (COCO class 14).
The exported graph therefore emits bird detections as class 0; the application maps that back
to COCO class 14 in its metadata. There are no runtime backend or precision switches.

## Deployment loop

Run the five-minute selection loop with:

```bash
uv run python scripts/deploy.py
```

It evaluates the newest camera frame at 1 FPS, retains the strict
highest-confidence bird per five-minute window, then writes one final transparent PNG named
`bird_conf_XX_ts_YYYY-MM-DD_HH-MM.png`. Use `--duration-minutes` for a bounded run.

## Current Beelink deployment

The BirdSpotter working copy is installed at `/home/pbatch/sprite-pipeline`. The standalone
`uv` executable is at `/home/pbatch/.local/bin/uv`. From an SSH session:

```bash
cd /home/pbatch/sprite-pipeline
~/.local/bin/uv sync
~/.local/bin/uv run pytest
~/.local/bin/uv run python scripts/deploy.py
```

The C920 is configured for 1600×896, 5 fps, MJPEG; BirdSpotter retains the full frame
and letterboxes it to 1088×1920. Earlier measurements on the Intel
N100 used the retired YOLO26s/1280 export:

- repeated image inference: 0.76–0.80 seconds per detector call
- live 1080p capture: 1.414 seconds average detector time and 0.398 effective FPS
- SAM 2.1 OpenVINO timing should be remeasured for the active detector/camera configuration
- cold detector-plus-segmenter command: 4.75–4.89 seconds
- peak resident memory: about 703 MB for image commands and 973 MB while capturing
- package temperature after the short live benchmark: 63°C

The larger detector clears the unchanged `0.10` threshold on all five current
validation images. In particular, image 4 improved from `0.018` with
YOLO26n/640 to `0.137`, and image 5 improved from `0.034` to `0.549`.

## Tests

```bash
uv run pytest
```

## Development checks

Ruff linting and formatting plus ty type checking run as pre-commit checks:

```bash
uv sync
uv run pre-commit run --all-files
```

After initializing or cloning the Git repository, install the hook once:

```bash
uv run pre-commit install
```

## Model licensing

YOLO26 and SAM 2.1 are provided through Ultralytics and are subject to Ultralytics'
licensing terms. The pipeline code in this repository does not alter those model licenses.
