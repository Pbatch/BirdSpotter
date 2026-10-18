"""Shared MobileNetV4 training, preprocessing and export contract."""

from pathlib import Path

import openvino as ov
import timm
import torch

from birdspotter.ml.preprocessing import BatchPreprocessor, validate_image_size

MODEL_NAME = "mobilenetv4_conv_large.e600_r384_in1k"
MODEL_VARIANTS = {
    "small": "mobilenetv4_conv_small.e2400_r224_in1k",
    "large": MODEL_NAME,
}
CLASSES = ["bird"]
IMAGE_SIZE = 640


def load_export_checkpoint(checkpoint: Path) -> tuple[torch.nn.Module, int]:
    """Load the lightweight training export as an eval-mode model and its image size."""

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_name = state["model_name"]
    if state["classes"] != CLASSES or model_name not in MODEL_VARIANTS.values():
        raise ValueError("Checkpoint has an incompatible classifier contract")
    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    model.load_state_dict(state["state_dict"])
    model.eval()
    image_size = state.get("image_size", 224)
    validate_image_size(image_size)
    return model, image_size


class PixelInputClassifier(torch.nn.Module):
    """Normalise inside the exported graph so inference only sends letterboxed pixels.

    Takes uint8 BGR batches shaped BxHxWx3, as BirdClassifier letterboxes them, and applies
    the training normalisation in float32 before running the network at its own precision.
    """

    def __init__(self, model: torch.nn.Module, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.model = model
        self.preprocessor = BatchPreprocessor()
        self.dtype = dtype

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # TensorRT only accepts uint8 into a Cast, so convert before reordering channels.
        rgb = pixels.to(torch.float32)[..., [2, 1, 0]].permute(0, 3, 1, 2)
        normalised = (rgb / 255 - self.preprocessor.mean) / self.preprocessor.std
        return self.model(normalised.to(self.dtype))


def pixel_example(image_size: int) -> torch.Tensor:
    return torch.zeros(1, image_size, image_size, 3, dtype=torch.uint8)


def export_classifier(checkpoint: Path, destination: Path) -> None:
    model, image_size = load_export_checkpoint(checkpoint)
    example = pixel_example(image_size)
    converted = ov.convert_model(
        PixelInputClassifier(model).eval(),
        example_input=example,
        input=[(list(example.shape), ov.Type.u8)],
    )
    destination.mkdir(parents=True, exist_ok=True)
    ov.save_model(converted, destination / "classifier.xml", compress_to_fp16=True)


def export_classifier_tensorrt(checkpoint: Path, destination: Path) -> None:
    """Build an FP16 TensorRT engine for this machine's NVIDIA GPU.

    TensorRT 11 networks are strongly typed, so precision comes from the ONNX graph: the
    network runs in half precision after float32 normalisation. Engines only run on the GPU
    and TensorRT version they were built with.
    """

    import tensorrt as trt  # noqa: PLC0415 -- optional `tensorrt` dependency group

    model, image_size = load_export_checkpoint(checkpoint)
    wrapped = PixelInputClassifier(model.half(), torch.float16).eval()
    destination.mkdir(parents=True, exist_ok=True)
    onnx_path = destination / "classifier.onnx"
    torch.onnx.export(
        wrapped,
        (pixel_example(image_size),),
        onnx_path,
        input_names=["pixels"],
        output_names=["logit"],
        dynamo=False,
    )
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "; ".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}: {errors}")
    engine = builder.build_serialized_network(network, builder.create_builder_config())
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")
    (destination / "classifier.engine").write_bytes(engine)
