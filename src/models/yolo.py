"""YOLOv8 detector split into encoder (backbone) and decoder (neck + head).

AnyUp will later replace nn.Upsample at layers 10 and 13 (PAN neck).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO

from src.utils.config import load_yaml

DEFAULT_YOLO_CONFIG = Path("configs/models/yolov8.yaml") # model config file



@dataclass
class EncoderOutput:
    """Backbone output plus skip-connection cache for the decoder."""

    features: torch.Tensor
    cache: list[Any]

# why is the encoder needed? if we are loading the enccoder from the weights?
#
class YOLOEncoder(nn.Module):
    """YOLOv8 backbone (layers 0–9): CSPDarknet + SPPF."""
    # CSPDarknet is a CNN that extracts features from the input image
    # SPPF is a spatial pyramid pooling module that pools the features from the input image
    # this helps in capturing the features from the input image at different scales

    def __init__(self, detection_model: nn.Module, end_layer: int = 9) -> None:
        super().__init__()
        self.detection_model = detection_model
        self.layers = detection_model.model
        self.save = detection_model.save
        self.end_layer = end_layer

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        features, cache = _run_layers(
            self.layers,
            x,
            start=0,
            end=self.end_layer + 1,
            save=self.save,
        )
        return EncoderOutput(features=features, cache=cache)


class YOLODecoder(nn.Module):
    """YOLOv8 neck + detection head (layers 10–22)."""

    def __init__(self, detection_model: nn.Module, start_layer: int = 10) -> None:
        super().__init__()
        self.detection_model = detection_model
        self.layers = detection_model.model
        self.save = detection_model.save
        self.start_layer = start_layer
        self.end_layer = len(detection_model.model)

    @property
    def upsample_layer_indices(self) -> list[int]:
        indices = []
        for i in range(self.start_layer, self.end_layer):
            if isinstance(self.layers[i], nn.Upsample):
                indices.append(i)
        return indices

    def forward(self, encoder_output: EncoderOutput) -> torch.Tensor:
        output, _ = _run_layers(
            self.layers,
            encoder_output.features,
            start=self.start_layer,
            end=self.end_layer,
            save=self.save,
            cache=encoder_output.cache,
        )
        return _normalize_output(output)


class YOLOv8Split(nn.Module):
    """Full YOLOv8 with separately accessible encoder and decoder."""

    def __init__(
        self,
        detection_model: nn.Module,
        backbone_end: int = 9,
        decoder_start: int = 10,
    ) -> None:
        super().__init__()
        self.detection_model = detection_model
        self.encoder = YOLOEncoder(detection_model, end_layer=backbone_end)
        self.decoder = YOLODecoder(detection_model, start_layer=decoder_start)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @property
    def upsample_layers(self) -> list[int]:
        return self.decoder.upsample_layer_indices

# this function resolves the input to the layer, 
# if the layer is -1, it returns the current input else 
# it returns the cached input
# cached input is the input to the layer before the current layer
def _resolve_input(
    x: torch.Tensor,
    layer_from: int | list[int],
    cache: list[Any],
    current: torch.Tensor,
) -> torch.Tensor | list[Any]:
    if layer_from == -1:
        return current
    if isinstance(layer_from, int):
        return cache[layer_from]
    return [current if j == -1 else cache[j] for j in layer_from]

# this function runs layers of yolo model
# resolves the input to the layer -> why is it needed?
# because the input to the layer is not the same as the input to the layer before
def _run_layers(
    layers: nn.ModuleList,
    x: torch.Tensor,
    start: int,
    end: int,
    save: list[int],
    cache: list[Any] | None = None,
) -> tuple[torch.Tensor, list[Any]]:
    y: list[Any] = list(cache) if cache is not None else []

    for i in range(start, end):
        module = layers[i]
        x = _resolve_input(x, module.f, y, x)
        x = module(x)

        if i < len(y):
            y[i] = x if i in save else None
        else:
            while len(y) < i:
                y.append(None)
            y.append(x if i in save else None)

    return x, y

# normalize the output to a single tensor for the decoder
def _normalize_output(output: torch.Tensor | tuple) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output

# download the weights from checkpoints on which the model is trained
def _download_weights(weights_name: str, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / weights_name
    if target.is_file():
        return target

    yolo = YOLO(weights_name)
    source = Path(str(getattr(yolo, "ckpt_path", "") or weights_name))
    if source.is_file():
        import shutil

        shutil.copy2(source, target)
        return target

    raise FileNotFoundError(f"Could not download or locate weights: {weights_name}")

# load the yolo model from the weights
def load_yolov8(
    config_path: str | Path = DEFAULT_YOLO_CONFIG,
    device: str | torch.device | None = None,
    checkpoint_dir: str | Path = "artifacts/checkpoints",
) -> YOLOv8Split:
    config = load_yaml(config_path)
    weights_name = config.get("weights", "yolov8n.pt")
    weights_path = _download_weights(weights_name, Path(checkpoint_dir))

    yolo = YOLO(str(weights_path))
    detection_model = yolo.model
    detection_model.eval()
    # split the model into encoder and decoder
    split = YOLOv8Split(
        detection_model=detection_model,
        backbone_end=int(config.get("backbone_end", 9)),
        decoder_start=int(config.get("decoder_start", 10)),
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return split.to(device)

# load the encoder from the yolo model
def load_yolo_encoder(
    config_path: str | Path = DEFAULT_YOLO_CONFIG,
    device: str | torch.device | None = None,
) -> YOLOEncoder:
    """Return the encoder from a loaded YOLOv8 split (loads weights once)."""
    return load_yolov8(config_path=config_path, device=device).encoder

# load the decoder from the yolo model
def load_yolo_decoder(
    config_path: str | Path = DEFAULT_YOLO_CONFIG,
    device: str | torch.device | None = None,
) -> YOLODecoder:
    """Return the decoder from a loaded YOLOv8 split (loads weights once)."""
    return load_yolov8(config_path=config_path, device=device).decoder
