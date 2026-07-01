from __future__ import annotations

import torch
import torch.nn as nn


class BaseSRModel(nn.Module):
    """Minimal shared interface for super-resolution models."""

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
