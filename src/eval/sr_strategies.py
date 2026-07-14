from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch.nn as nn
from PIL import Image

from src.models.anyup import load_anyup
from src.models.espcn import load_espcn, upscale_image as espcn_upscale
from src.models.fsrcnn import load_fsrcnn, upscale_image as fsrcnn_upscale

StrategyName = Literal["baseline", "espcn2x", "fsrcnn2x", "anyup"]
STRATEGIES: tuple[StrategyName, ...] = ("baseline", "espcn2x", "fsrcnn2x", "anyup")


@dataclass
class PreparedImage:
    image: Image.Image
    bbox_scale: float
    strategy: StrategyName


@dataclass
class SRModels:
    espcn: object | None = None
    fsrcnn: nn.Module | None = None
    anyup: nn.Module | None = None


def load_sr_models(
    strategies: list[StrategyName],
    device: str,
    scale: int = 2,
) -> SRModels:
    models = SRModels()
    if "espcn2x" in strategies:
        models.espcn = load_espcn(scale=scale, device=device)
    if "fsrcnn2x" in strategies:
        models.fsrcnn = load_fsrcnn(scale=scale, device=device)
    if "anyup" in strategies:
        models.anyup = load_anyup(device=device)
    return models


def prepare_image(
    strategy: StrategyName,
    lr_image: Image.Image,
    hr_size: tuple[int, int],
    models: SRModels,
    scale: int = 2,
) -> PreparedImage:
    hr_w, hr_h = hr_size
    lr_w, lr_h = lr_image.size
    hr_from_lr_scale = hr_w / lr_w if lr_w else float(scale)

    if strategy == "baseline":
        return PreparedImage(image=lr_image, bbox_scale=hr_from_lr_scale, strategy=strategy)

    if strategy == "espcn2x":
        if models.espcn is None:
            raise RuntimeError("ESPCN model not loaded")
        sr = espcn_upscale(models.espcn, lr_image)
        return PreparedImage(image=sr, bbox_scale=hr_w / sr.size[0], strategy=strategy)

    if strategy == "fsrcnn2x":
        if models.fsrcnn is None:
            raise RuntimeError("FSRCNN model not loaded")
        sr = fsrcnn_upscale(models.fsrcnn, lr_image)
        return PreparedImage(image=sr, bbox_scale=hr_w / sr.size[0], strategy=strategy)

    if strategy == "anyup":
        # Feature-level SR: detect on LR; bbox coords mapped back to HR GT space.
        return PreparedImage(image=lr_image, bbox_scale=hr_from_lr_scale, strategy=strategy)

    raise ValueError(f"Unknown strategy: {strategy}")
