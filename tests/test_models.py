import pytest
import torch

from src.models.fsrcnn import FSRCNN


def test_fsrcnn_forward_pass() -> None:
    model = FSRCNN(scale_factor=2, num_channels=1)
    model.eval()

    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 1, 128, 128)


@pytest.mark.skipif(
    not torch.hub.get_dir(),
    reason="torch hub unavailable",
)
def test_anyup_loads_if_online() -> None:
    try:
        from src.models.anyup import load_anyup

        model = load_anyup(device="cpu")
        assert model is not None
    except Exception as exc:
        pytest.skip(f"AnyUp hub load unavailable: {exc}")


def test_yolo_encoder_decoder_split() -> None:
    try:
        from src.models.yolo import load_yolov8
    except ImportError as exc:
        pytest.skip(f"ultralytics not installed: {exc}")

    device = "cpu"
    yolo = load_yolov8(device=device)
    x = torch.randn(1, 3, 320, 320)

    with torch.no_grad():
        enc_out = yolo.encoder(x)
        split = yolo.decoder(enc_out)
        full = yolo(x)

    assert full.shape == split.shape
    assert torch.allclose(full, split, atol=1e-5)
    assert yolo.upsample_layers == [10, 13]
