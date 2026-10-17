"""Whole-image bird segmentation with the fixed-prompt SAM 3 OpenVINO model."""

from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from openvino.preprocess import ColorFormat, PrePostProcessor, ResizeAlgorithm

SAM3_IMAGE_SIZE = 504
MODEL_FILENAME = "sam3_bird.xml"


def embed_preprocessing(model: ov.Model) -> ov.Model:
    """Accept raw uint8 BGR frames of any size, as OpenCV decodes them.

    The embedded steps match ``preprocess_image``: BGR to RGB, a whole-frame
    resize to 504x504 that ignores aspect ratio, and normalization to [-1, 1].
    """
    ppp = PrePostProcessor(model)
    image = ppp.input("image")
    image.tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC")).set_color_format(
        ColorFormat.BGR
    ).set_spatial_dynamic_shape()
    image.model().set_layout(ov.Layout("NCHW"))
    image.preprocess().convert_element_type(ov.Type.f32).convert_color(ColorFormat.RGB).resize(
        ResizeAlgorithm.RESIZE_LINEAR
    ).mean(127.5).scale(127.5).convert_element_type(ov.Type.f16)
    return ppp.build()


def pin_encoder(model: ov.Model, encoder_device: str, decoder_device: str) -> ov.Model:
    """Assign HETERO affinities: the vision encoder on one device, the rest on another.

    The encoder ends at the neck's feature maps, which the DETR decoder consumes.
    Everything they depend on runs on ``encoder_device``, except dynamically
    shaped ops such as embedded preprocessing, which stay on ``decoder_device``.
    """
    boundary = [
        op
        for op in model.get_ordered_ops()
        if "vision_encoder.neck" in op.get_friendly_name()
        and any(
            "vision_encoder" not in target.get_node().get_friendly_name()
            for output in op.outputs()
            for target in output.get_target_inputs()
        )
    ]
    if not boundary:
        raise ValueError("SAM 3 model has no recognizable vision encoder boundary")
    encoder: set[str] = set()
    stack = list(boundary)
    while stack:
        node = stack.pop()
        if node.get_friendly_name() not in encoder:
            encoder.add(node.get_friendly_name())
            stack.extend(port.get_source_output().get_node() for port in node.inputs())
    for op in model.get_ordered_ops():
        on_encoder = (
            op.get_friendly_name() in encoder
            and op.get_type_name() != "Parameter"
            and all(output.get_partial_shape().is_static for output in op.outputs())
        )
        op.get_rt_info()["affinity"] = encoder_device if on_encoder else decoder_device
    return model


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    """Resize the complete image and normalize RGB pixels for legacy SAM 3 exports."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or not image_bgr.size:
        raise ValueError("SAM 3 expects a nonempty HxWx3 BGR image")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE))
    normalized = resized.astype(np.float32) / 127.5 - 1.0
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float16)


def prepare_frame(image_bgr: np.ndarray) -> np.ndarray:
    """Batch a raw BGR frame for an export with embedded preprocessing."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or not image_bgr.size:
        raise ValueError("SAM 3 expects a nonempty HxWx3 BGR image")
    return np.ascontiguousarray(image_bgr[None], dtype=np.uint8)


class Sam3OpenVinoSegmenter:
    """Return the highest-confidence bird without a classifier box prompt."""

    def __init__(self, model_dir: Path, *, confidence: float = 0.5) -> None:
        if not 0 <= confidence <= 1:
            raise ValueError("SAM 3 confidence must be between zero and one")
        self.model_path = model_dir / MODEL_FILENAME
        if not self.model_path.is_file() or not self.model_path.with_suffix(".bin").is_file():
            raise FileNotFoundError(
                f"SAM 3 OpenVINO model not found: {self.model_path}. "
                "Run `python -m birdspotter.ml.sam3_export`."
            )
        self.confidence = confidence
        core = ov.Core()
        config = {"PERFORMANCE_HINT": "LATENCY"}
        self.encoder_device = "CPU"
        if "GPU" in core.available_devices:
            # The full graph does not compile on the N100 iGPU (its DETR decoder trips
            # a GPU plugin bug), but the encoder, where nearly all time goes, does.
            model = pin_encoder(core.read_model(self.model_path), "GPU", "CPU")
            try:
                self.backend = core.compile_model(model, "HETERO:GPU,CPU", config)
                self.encoder_device = "GPU"
            except RuntimeError:
                pass  # e.g. a non-Intel GPU that OpenVINO lists but cannot compile for
        if self.encoder_device == "CPU":
            self.backend = core.compile_model(self.model_path, "CPU", config)
        # Exports before preprocessing was embedded take normalized FP16 NCHW.
        self.embedded_preprocessing = self.backend.input("image").get_element_type() == ov.Type.u8

    def prepare(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.embedded_preprocessing:
            return prepare_frame(image_bgr)
        return preprocess_image(image_bgr)

    def segment(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Segment the whole image; reject absent birds and empty masks."""
        result = self.backend({"image": self.prepare(image_bgr)})
        score = float(np.asarray(result[self.backend.output("score")]).reshape(-1)[0])
        logits = np.asarray(result[self.backend.output("mask_logits")]).squeeze()
        if not np.isfinite(score) or score < self.confidence:
            raise ValueError(f"SAM 3 found no confident bird (score={score:.3f})")
        if logits.shape != (SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE) or not np.isfinite(logits).all():
            raise ValueError("SAM 3 returned invalid mask logits")
        height, width = image_bgr.shape[:2]
        mask = cv2.resize(logits.astype(np.float32), (width, height)) > 0
        if not mask.any():
            raise ValueError("SAM 3 returned an empty bird mask")
        return mask, score

    def describe(self) -> dict[str, object]:
        return {
            "model": "SAM 3",
            "model_path": self.model_path.name,
            "prompt": "bird",
            "encoder_device": self.encoder_device,
            "input_size": SAM3_IMAGE_SIZE,
            "embedded_preprocessing": self.embedded_preprocessing,
            "weight_compression": "W8A16",
            "confidence": self.confidence,
            "backend": "OpenVINO",
        }
