from __future__ import annotations

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as TF
from PIL import Image


# -----------------------------
# Load Faster R-CNN
# -----------------------------
def load_fasterrcnn(device: str = "cpu"):
    """
    Load pretrained Faster R-CNN (COCO weights).
    Used as fixed detector for SR evaluation.
    """

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    return model.to(device)


# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess(image: Image.Image) -> torch.Tensor:
    """
    PIL → Tensor
    """

    return TF.to_tensor(image)


# -----------------------------
# Inference
# -----------------------------
def detect(
    model,
    image: Image.Image,
    device: str = "cpu",
    score_threshold: float = 0.5,
):
    """
    Run Faster R-CNN detection on an image.

    Returns filtered detections:
    - boxes
    - labels
    - scores
    """

    img_tensor = preprocess(image).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    scores = outputs["scores"]
    keep = scores >= score_threshold

    return {
        "boxes": outputs["boxes"][keep],
        "labels": outputs["labels"][keep],
        "scores": outputs["scores"][keep],
    }


# -----------------------------
# Simple wrapper (optional)
# -----------------------------
class FasterRCNNDetector:
    """
    Optional wrapper so all detectors (YOLO / DETR / Faster R-CNN)
    can share same interface style.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = load_fasterrcnn(device)

    def __call__(self, image: Image.Image):
        return detect(self.model, image, self.device) 