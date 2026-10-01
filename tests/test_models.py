from pathlib import Path

from birdspotter.models import sam3_openvino_dir


def test_sam3_artifact_path() -> None:
    assert sam3_openvino_dir(Path("weights")) == Path("weights/sam3/openvino-504")
