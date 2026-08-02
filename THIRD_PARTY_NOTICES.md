# Third-party model notices

`LICENSE.md` applies to BirdSpotter's original source code. It does not change
the licences governing third-party packages, model checkpoints, or artifacts
derived from those checkpoints.

## SAM 2.1 Hiera Large

The export script downloads Meta's `sam2.1_hiera_large.pt` checkpoint and
converts it to OpenVINO IR. Meta licenses the SAM 2 model checkpoints under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). If you
redistribute the checkpoint or the generated SAM artifacts, preserve the
applicable Apache-2.0 licence and notices from the
[SAM 2 repository](https://github.com/facebookresearch/sam2).

## Ultralytics YOLO26

The export script downloads Ultralytics' pretrained YOLO26s checkpoint and
produces a bird-only OpenVINO detector from it. Ultralytics makes YOLO26 code,
models, and documentation available under AGPL-3.0 by default, with a separate
Enterprise licence available for uses that do not meet AGPL-3.0 obligations.
See the [Ultralytics licence terms](https://www.ultralytics.com/license) before
using or distributing the YOLO checkpoint or its generated OpenVINO artifacts.

This repository does not contain either checkpoint or generated runtime model
artifacts. They are downloaded or produced locally under `weights_dev/` and
`weights/`.
