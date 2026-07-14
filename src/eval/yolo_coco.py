from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from src.eval.coco_ids import YOLO_CLS_TO_COCO_CATEGORY
from src.eval.sr_strategies import PreparedImage, StrategyName


class AnyUpUpsample(nn.Module):
    """Replace YOLO nn.Upsample with AnyUp feature upsampling (feature-level SR)."""

    def __init__(self, anyup: nn.Module, orig: nn.Upsample, image_holder: dict) -> None:
        super().__init__()
        self.anyup = anyup
        self.image_holder = image_holder
        if isinstance(orig.scale_factor, (int, float)):
            self.scale = float(orig.scale_factor)
        else:
            self.scale = 2.0
        self.f = orig.f
        self.i = getattr(orig, "i", -1)
        self.type = getattr(orig, "type", "AnyUpUpsample")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        guidance = self.image_holder.get("tensor")
        if guidance is None:
            return nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")

        h, w = x.shape[-2:]
        out_h, out_w = int(round(h * self.scale)), int(round(w * self.scale))
        hr_guidance = nn.functional.interpolate(
            guidance, size=(out_h, out_w), mode="bilinear", align_corners=False
        )
        return self.anyup(hr_guidance, x, output_size=(out_h, out_w))


def patch_yolo_with_anyup(detection_model: DetectionModel, anyup: nn.Module) -> dict:
    """Swap nn.Upsample layers in the YOLO neck with AnyUp."""
    image_holder: dict = {"tensor": None}
    layers = detection_model.model
    replaced = 0
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Upsample):
            wrapper = AnyUpUpsample(anyup=anyup, orig=layer, image_holder=image_holder)
            layers[i] = wrapper
            replaced += 1
    if replaced == 0:
        raise RuntimeError("No nn.Upsample layers found in YOLO model for AnyUp patching")
    return image_holder


def load_yolo_detector(
    weights: str | Path = "yolov8n.pt",
    strategy: StrategyName = "baseline",
    anyup: nn.Module | None = None,
    device: str | None = None,
) -> tuple[YOLO, dict | None]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(str(weights))
    image_holder = None
    if strategy == "anyup":
        if anyup is None:
            raise RuntimeError("AnyUp model required for anyup strategy")
        image_holder = patch_yolo_with_anyup(model.model, anyup)
    model.to(device)
    return model, image_holder


def yolo_results_to_coco(
    result,
    image_id: int,
    bbox_scale: float = 1.0,
    conf_threshold: float = 0.001,
) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xywh.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    predictions: list[dict] = []
    for (x, y, w, h), score, cls_idx in zip(boxes, scores, classes):
        if score < conf_threshold:
            continue
        if cls_idx < 0 or cls_idx >= len(YOLO_CLS_TO_COCO_CATEGORY):
            continue
        predictions.append(
            {
                "image_id": int(image_id),
                "category_id": int(YOLO_CLS_TO_COCO_CATEGORY[cls_idx]),
                "bbox": [
                    float(x * bbox_scale),
                    float(y * bbox_scale),
                    float(w * bbox_scale),
                    float(h * bbox_scale),
                ],
                "score": float(score),
            }
        )
    return predictions


def run_yolo_on_prepared(
    model: YOLO,
    prepared: PreparedImage,
    image_holder: dict | None,
    conf: float = 0.001,
    imgsz: int = 640,
) -> list[dict]:
    arr = np.array(prepared.image.convert("RGB"))

    if image_holder is not None:
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(next(model.model.parameters()).device)
        image_holder["tensor"] = tensor

    results = model.predict(source=arr, conf=conf, imgsz=imgsz, verbose=False)
    return yolo_results_to_coco(
        results[0],
        image_id=-1,
        bbox_scale=prepared.bbox_scale,
        conf_threshold=conf,
    )


def iter_coco_val_images(
    lr_dir: Path,
    ann_file: Path,
    max_images: int = 0,
) -> list[dict]:
    coco = COCO(str(ann_file))
    img_ids = sorted(coco.getImgIds())
    if max_images > 0:
        img_ids = img_ids[:max_images]

    rows: list[dict] = []
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        lr_path = lr_dir / info["file_name"]
        if not lr_path.is_file():
            continue
        rows.append(
            {
                "image_id": int(img_id),
                "file_name": info["file_name"],
                "hr_size": (int(info["width"]), int(info["height"])),
                "lr_path": lr_path,
            }
        )
    return rows
