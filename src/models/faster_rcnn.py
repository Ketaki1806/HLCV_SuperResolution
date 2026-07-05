from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class FasterRCNNEncoder(nn.Module):
    """
    Backbone only (feature extractor).
    Acts like YOLOEncoder but MUCH simpler.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.backbone = model.backbone

    def forward(self, x: torch.Tensor):
        """
        Returns feature maps only (no detection head).
        """
        return self.backbone(x)


class FasterRCNNDecoder(nn.Module):
    """
    RPN + ROI heads (detection part).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.rpn = model.rpn
        self.roi_heads = model.roi_heads

    def forward(self, images, features):
        """
        Run detection head using precomputed features.
        """
        proposals, _ = self.rpn(images, features, targets=None)
        detections = self.roi_heads(features, proposals, images.image_sizes)
        return detections


class FasterRCNNSplit(nn.Module):
    """
    Full Faster R-CNN with encoder/decoder separation.
    """

    def __init__(self):
        super().__init__()

        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        model.eval()

        self.model = model
        self.encoder = FasterRCNNEncoder(model)
        self.decoder = FasterRCNNDecoder(model)

    def forward(self, images):
        features = self.encoder(images)
        return self.decoder(images, features)