from birdspotter.cli import parser


def test_image_command_accepts_sam21_segmenter() -> None:
    args = parser().parse_args(["image", "bird.jpg", "--segmenter", "sam2.1"])

    assert args.segmenter == "sam2.1"


def test_image_command_accepts_sam21_openvino_segmenter() -> None:
    args = parser().parse_args(["image", "bird.jpg", "--segmenter", "sam2.1-openvino"])

    assert args.segmenter == "sam2.1-openvino"
