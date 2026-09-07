# BirdSpotter

BirdSpotter watches a camera, and uses computer vision to make bird cutouts. 

It uses YOLO26s followed by SAM 2.1.

| Bird detection | Final transparent bird |
| --- | --- |
| <img src="demo/annotations/5.png" alt="Bird detection annotation" width="300"> | <img src="demo/segmentations/5.png" alt="Segmented bird" width="300"> |

## Setup

0) Install `uv` (https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)
1) Sync the dependencies
```bash
uv sync --locked
```
2) Download the models
```bash
uv run python scripts/download_models.py
```

## Demo

Regenerate the five demo annotations and segmentations:

```bash
uv run python scripts/generate_demo.py
```

## Deployment

```bash
uv run python scripts/deploy.py
```

Deployment creates a web page at `http://HOSTNAME:8080`.

In `Recent sightings`, view the 10 most recently seen birds.

In `Area of interest`, draw the region of interest that detection uses. 

## Tests and checks

```bash
uv run pytest
uv run pre-commit run --all-files
```
