# Third-party model notices

`LICENSE.md` applies to BirdSpotter's original source code. It does not change
the licences governing third-party packages, model checkpoints, or artifacts
derived from those checkpoints.

## SAM 3

The export script downloads Meta's `facebook/sam3` checkpoint and converts it
to a fixed-prompt OpenVINO model. The checkpoint and derived artifacts are
subject to the upstream [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE).
Preserve the applicable license and notices when redistributing model artifacts.

## MobileNetV4

Training uses timm's `mobilenetv4_conv_small.e2400_r224_in1k` ImageNet checkpoint.
See the upstream [model card](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k)
for its Apache-2.0 licence and model details.

This repository does not contain either checkpoint or generated runtime model
artifacts. Source checkpoints remain in the Hugging Face cache, and generated
artifacts are stored locally under `weights/`.
