# BirdSpotter

This pipeline watches a V4L2 webcam, detects only COCO class `bird`, and keeps
the single detection with the highest object-detector confidence in each fixed
selection window. EdgeSAM or SAM 2.1 receives that detection as a box prompt. The primary
output is one tightly cropped PNG containing the original bird pixels and a
transparent background.

The initial target is a Beelink Mini S12 with an Intel N100 and a Logitech C920.
The C920 delivers 1920×1080 MJPEG at 30 fps; the pipeline centre-crops each frame to
1080×1080 before detection and segmentation. YOLO26s runs at 1280×1280 and samples the
newest frame at no more than 1 fps by default. The capture thread continually discards stale frames, so a
slow inference call does not build an increasingly delayed video queue.

## Outputs

For each accepted selection window, `segmented/` receives:

- `bird_*.png`: the required tightly cropped BGRA/RGBA bird image

Exactly one bird is segmented. Multiple birds are not merged. Blur, pose, and
exposure scoring are deliberately absent for now; detector confidence is the
only quality proxy. If the winning detector box itself encloses overlapping
birds, SAM can include that overlap; the pipeline does not replace the winning
detection with a lower-confidence box to avoid it.

## Environment and model preparation

All Python packages are declared in `pyproject.toml` and installed through
`uv sync`. The project pins Python 3.12 because inference wheels do not yet have
uniform Python 3.14 support.

Prepare the model artifacts on the development machine:

```bash
uv sync --group model-export
uv run birdspotter models --calibration-data path/to/dataset.yaml
uv sync
uv sync --reinstall-package opencv-python-headless
```

The second `uv sync` removes export-only quantization packages and leaves the runtime
environment at the locked dependency set. OpenCV is explicitly reinstalled because
its GUI and headless wheels share `cv2` files.

## Run on an image

```bash
uv run birdspotter image images/feeder.png --confidence 0.10
```

To use SAM 2.1 instead of the default EdgeSAM, place Ultralytics'
`sam2.1_l.pt` checkpoint at `weights/sam2.1/sam2.1_l.pt` and pass
`--segmenter sam2.1`:

```bash
uv run birdspotter image images/feeder.png --segmenter sam2.1
```

For the OpenVINO-optimized SAM 2.1 image path, first convert the same checkpoint:

```bash
uv run birdspotter models --calibration-data path/to/dataset.yaml --export-sam21-openvino
uv run birdspotter image images/feeder.png --segmenter sam2.1-openvino
```

The conversion creates an OpenVINO image encoder and box-prompt mask predictor under
`weights/sam2.1/openvino-256/`, with a fixed 256×256 image input. This path supports
still-image segmentation; SAM 2.1 video tracking needs additional memory-model
conversions.

Detector inference always uses a static batch-one INT8 OpenVINO model whose classification
heads have been reduced from 80 COCO classes to the pretrained `bird` channel (COCO class 14).
The exported graph therefore emits bird detections as class 0; the application maps that back
to COCO class 14 in its metadata. There are no runtime backend or precision switches. Run an
image with:

```bash
uv run birdspotter image images/feeder.png
```

Create the INT8 detector with a representative calibration dataset:

```bash
uv run --group model-export birdspotter models --calibration-data path/to/dataset.yaml
```

Exit status `2` means no bird passed the detector threshold.

## Run on the C920

Stop after producing one segmented bird:

```bash
uv run birdspotter webcam \
  --device 0 \
  --window-seconds 4 \
  --sample-fps 1 \
  --max-outputs 1
```

Run continuously with `--max-outputs 0`. Use `--duration 30` for a bounded
smoke test. The `pbatch` account must belong to the `video` group.

## Current Beelink deployment

The BirdSpotter working copy is installed at `/home/pbatch/sprite-pipeline`. The standalone
`uv` executable is at `/home/pbatch/.local/bin/uv`. From an SSH session:

```bash
cd /home/pbatch/sprite-pipeline
~/.local/bin/uv sync
~/.local/bin/uv run pytest
~/.local/bin/uv run birdspotter webcam --max-outputs 1
```

The C920 has been verified at 1920×1080, 30 fps, MJPEG; BirdSpotter uses its central
1080×1080 square. Measured on the Intel
N100 with YOLO26s/1280:

- repeated image inference: 0.76–0.80 seconds per detector call
- live 1080p capture: 1.414 seconds average detector time and 0.398 effective FPS
- EdgeSAM: 1.72–1.78 seconds for the selected bird
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

YOLO26 is provided through Ultralytics and is subject to Ultralytics' licensing
terms. EdgeSAM's official models and source use the S-Lab License 1.0, which
permits non-commercial use and requires separate permission for commercial use.
The pipeline code in this repository does not alter those model licenses.
