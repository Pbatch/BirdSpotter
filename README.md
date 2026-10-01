# BirdSpotter

BirdSpotter watches a camera, and uses computer vision to make bird cutouts.

It uses MobileNetV4 followed by SAM 3.

| Classification | Segmentation |
| --- | --- |
| <img src="demo/annotations/5.png" alt="Bird detection annotation" width="300"> | <img src="demo/segmentations/5.png" alt="Segmented bird" width="300"> |

## Setup

```bash
uv sync --locked
```

```bash
uv run python scripts/download_models.py
```

## Demo

```bash
uv run python scripts/generate_demo.py
```

## Deployment

```bash
uv run python scripts/deploy.py
```

## Tests and checks

```bash
uv run pytest
uv run pre-commit run --all-files
```
