from __future__ import annotations

import math
from pathlib import Path
from urllib.request import urlretrieve

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import functional as TF

from src.models.base_model import BaseSRModel

FSRCNN_WEIGHT_URLS = {
    2: "https://www.dropbox.com/s/1k3dker6g7hz76s/fsrcnn_x2.pth?dl=1",
    3: "https://www.dropbox.com/s/pm1ed2nyboulz5z/fsrcnn_x3.pth?dl=1",
    4: "https://www.dropbox.com/s/vsvumpopupdpmmu/fsrcnn_x4.pth?dl=1",
}


class FSRCNN(BaseSRModel):
    """FSRCNN from Dong et al. (Accelerating the Super-Resolution CNN)."""

    def __init__(
        self,
        scale_factor: int = 2,
        num_channels: int = 1,
        d: int = 56,
        s: int = 12,
        m: int = 4,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor

        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d),
        )
        mid_layers: list[nn.Module] = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            mid_layers.extend([nn.Conv2d(s, s, kernel_size=3, padding=1), nn.PReLU(s)])
        mid_layers.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*mid_layers)
        self.last_part = nn.ConvTranspose2d(
            d,
            num_channels,
            kernel_size=9,
            stride=scale_factor,
            padding=4,
            output_padding=scale_factor - 1,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in list(self.first_part) + list(self.mid_part):
            if isinstance(module, nn.Conv2d):
                fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                nn.init.normal_(module.weight.data, mean=0.0, std=math.sqrt(2 / fan_in))
                nn.init.zeros_(module.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_part(x)
        x = self.mid_part(x)
        return self.last_part(x)


def _checkpoint_path(scale: int, checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"fsrcnn_x{scale}.pth"


def download_fsrcnn_weights(scale: int, checkpoint_dir: Path) -> Path:
    if scale not in FSRCNN_WEIGHT_URLS:
        raise ValueError(f"No pretrained weights for scale {scale}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(scale, checkpoint_dir)
    if path.is_file():
        return path

    print(f"Downloading FSRCNN x{scale} weights to {path}")
    urlretrieve(FSRCNN_WEIGHT_URLS[scale], path)
    return path


def load_fsrcnn(
    scale: int = 2,
    d: int = 56,
    s: int = 12,
    m: int = 4,
    checkpoint_dir: str | Path = "artifacts/checkpoints",
    device: str | torch.device | None = None,
) -> FSRCNN:
    weights_path = download_fsrcnn_weights(scale, Path(checkpoint_dir))
    model = FSRCNN(scale_factor=scale, num_channels=1, d=d, s=s, m=m)
    state_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)


def _rgb_to_ycbcr_tensor(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = TF.to_tensor(image).unsqueeze(0)
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.564 * (b - y) + 0.5
    cr = 0.713 * (r - y) + 0.5
    return y, cb, cr


def _ycbcr_to_rgb_tensor(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
    r = y + 1.403 * (cr - 0.5)
    g = y - 0.344 * (cb - 0.5) - 0.714 * (cr - 0.5)
    b = y + 1.773 * (cb - 0.5)
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)


def upscale_image(model: FSRCNN, image: Image.Image) -> Image.Image:
    """Run FSRCNN on the luminance channel and bicubic-upscale chroma."""
    y, cb, cr = _rgb_to_ycbcr_tensor(image)
    device = next(model.parameters()).device
    y = y.to(device)
    cb = cb.to(device)
    cr = cr.to(device)

    with torch.no_grad():
        y_hr = model(y)
        scale = model.scale_factor
        cb_hr = nn.functional.interpolate(cb, scale_factor=scale, mode="bicubic", align_corners=False)
        cr_hr = nn.functional.interpolate(cr, scale_factor=scale, mode="bicubic", align_corners=False)
        rgb = _ycbcr_to_rgb_tensor(y_hr, cb_hr, cr_hr)

    return TF.to_pil_image(rgb.squeeze(0).cpu())
