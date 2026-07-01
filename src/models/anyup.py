from __future__ import annotations

import torch


def load_anyup(device: str | torch.device | None = None):
    """Load the default AnyUp upsampler from torch.hub."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.hub.load("wimmerth/anyup", "anyup", trust_repo=True)
    model.eval()
    return model.to(device)
